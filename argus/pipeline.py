"""Argus end-to-end orchestration (used by the CLI and the Streamlit dashboard)."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import deps, live
from .config import SimulationConfig, WatchZone
from .coverage import CoverageResult, analyse_coverage, scaled_constellations
from .edge import EdgeCompressor, EdgeResult, downlink_energy_comparison
from .fire_index import cloud_fraction
from .metrics import detect
from .orbit import Capture, OrbitAndDownlinkSimulator, OrbitResult
from .payload import PayloadModel
from .scheduler import ALERT, SCENE, DownlinkProduct, SchedulingResult, run_scheduling_comparison
from .segmentation import SegmentationResult, Segmenter, train_segmentation
from .tiles import MultispectralTileGenerator, firms_to_watch_zones, load_firms_csv

Logger = Callable[[str], None]


@dataclass
class OnboardResult:
    products: list[DownlinkProduct]
    detections: list[dict[str, Any]]
    stats: dict[str, float]
    processing_latency_s: float
    samples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class LiveContext:
    """Snapshot of today's live inputs."""

    epoch: datetime
    hotspots: int
    feeds: dict[str, str]
    clusters: list[live.FireCluster]
    zones: list[WatchZone]
    cities: str
    tles: dict[str, tuple[str, str]]
    tle_provenance: str
    hotspot_points: list[tuple[float, float, float]] = field(default_factory=list)  # (lat, lon, frp) sample


@dataclass
class PipelineResult:
    config: SimulationConfig
    output_dir: Path
    orbit: OrbitResult
    segmentation: SegmentationResult
    edge: EdgeResult
    onboard: OnboardResult
    scheduling: SchedulingResult
    energy: dict[str, float]
    timings: dict[str, float]
    zone_source: str = "built-in watch zones"
    live: LiveContext | None = None
    coverage: CoverageResult | None = None

    def summary_dict(self) -> dict[str, Any]:
        return build_summary(self)


def prepare_live(cfg: SimulationConfig, log: Logger = print) -> LiveContext | None:
    """Load today's FIRMS fires, population and TLEs; task the strongest/most exposed fires."""
    lc = cfg.live
    hotspots, feeds = live.load_firms(offline=lc.offline, log=log)
    if not hotspots:
        log("  no live FIRMS data available (offline and nothing cached) - using built-in scenario")
        return None
    cities = live.load_cities(offline=lc.offline, log=log)
    clusters = live.cluster_hotspots(hotspots)
    live.attach_exposure(clusters, cities, top_n=1500)
    # task: strongest fires (2/3) + most exposed WUI fires (1/3), one per 2-degree cell
    by_frp = clusters[:1500]
    by_risk = sorted(by_frp, key=lambda c: -(c.wui_risk * math.log1p(c.frp_total_mw)))
    chosen: list[live.FireCluster] = []
    cells: set[tuple[int, int]] = set()
    for pool, quota in ((by_frp, lc.n_zones * 2 // 3), (by_risk, lc.n_zones)):
        for c in pool:
            key = (int(c.lat // 2), int(c.lon // 2))
            if key in cells:
                continue
            cells.add(key)
            chosen.append(c)
            if len(chosen) >= quota:
                break
    zones = [WatchZone(
        name=(f"{c.nearest_city} fire" if c.city_dist_km < 80 else f"fire #{c.cluster_id} ({c.lat:.1f},{c.lon:.1f})"),
        lat_deg=c.lat, lon_deg=c.lon, radius_km=max(20.0, c.extent_km + 10.0), wui_risk=c.wui_risk,
        population_density=c.exposed_pop / (math.pi * 50.0**2), fire_events_per_day=0.05, cloud_prob=0.3,
        biome="live FIRMS", live_area_ha=c.est_active_area_ha) for c in chosen]
    tles, tle_prov = live.load_fire_sensor_tles(offline=lc.offline, log=log)
    epoch = live.utc_now_hour()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(hotspots), size=min(6000, len(hotspots)), replace=False)
    points = [(hotspots[i].lat, hotspots[i].lon, hotspots[i].frp_mw) for i in pick]
    ctx = LiveContext(epoch, len(hotspots), feeds, clusters[:200], zones, cities.provenance, tles, tle_prov, points)
    cfg.orbit.zones = tuple(zones)
    cfg.orbit.epoch_utc = epoch
    log(f"  live: {len(hotspots):,} FIRMS hotspots -> {len(clusters):,} fire clusters; tasking {len(zones)} "
        f"({sum(z.wui_risk >= cfg.scheduler.hp_wui_threshold for z in zones)} near population); epoch {epoch:%Y-%m-%d %H:%MZ}")
    return ctx

def run_onboard_inference(captures: list[Capture], segmenter: Segmenter, cfg: SimulationConfig,
                          processing_latency_s: float, log: Logger = print) -> OnboardResult:
    """Run the deployed model on every orbital capture and package downlink products."""
    mc, sc = cfg.model, cfg.scheduler
    gen = MultispectralTileGenerator(mc.tile_size, mc.gsd_m)
    payload = PayloadModel(cfg.payload, mc.tile_size, mc.gsd_m)
    if not captures:
        return OnboardResult([], [], {"captures": 0}, processing_latency_s)
    tiles, masks = gen.generate_many([gen.spec_from_capture(c) for c in captures])
    t0 = time.perf_counter()
    prob = segmenter.predict_proba(tiles)
    infer_s = time.perf_counter() - t0
    detected, conf, npx = detect(prob, mc.detection_threshold, mc.min_fire_pixels, mc.confidence_topk)
    clouds = np.array([cloud_fraction(t) for t in tiles])
    gt_px = masks.reshape(len(masks), -1).sum(axis=1)
    history: dict[tuple[str, str], tuple[float, float]] = {}
    products: list[DownlinkProduct] = []
    rows: list[dict[str, Any]] = []
    for i, c in enumerate(captures):
        area = float(npx[i]) * gen.pixel_area_ha
        growth, is_new = 0.0, False
        if detected[i]:
            key = (c.sat_name, c.zone_name) if c.zone_name else None
            prev = history.get(key) if key else None
            is_new = prev is None
            if prev is not None:
                dt_h = (c.time_s - prev[0]) / 3600.0
                growth = (area - prev[1]) / dt_h if dt_h > 0.1 else 0.0
            if key:
                history[key] = (c.time_s, area)
            kind, size = ALERT, payload.alert_bytes(prob[i] >= mc.detection_threshold)
            quicklook = False
        else:
            kind = SCENE
            quicklook = clouds[i] >= sc.cloud_quicklook_threshold  # Argus cloud-aware packaging
            size = (payload.quicklook_bytes() + cfg.payload.metadata_bytes) if quicklook \
                else payload.compressed_scene_bytes
        gt_fire = bool(gt_px[i] > 0)
        products.append(DownlinkProduct(
            product_id=c.capture_id, capture_id=c.capture_id, sat_name=c.sat_name, capture_time_s=c.time_s,
            available_time_s=c.time_s + processing_latency_s, kind=kind, size_bytes=int(size),
            confidence=float(conf[i]), est_area_ha=area, growth_ha_per_h=float(growth), is_new_detection=is_new,
            wui_risk=c.wui_risk, population_density=c.population_density, zone_name=c.zone_name,
            cloud_fraction=float(clouds[i]), quicklook_only=bool(quicklook),
            gt_fire=gt_fire, gt_high_priority=gt_fire and c.wui_risk >= sc.hp_wui_threshold,
            fire_event_id=c.fire_event_id, gt_cloud=float(c.cloud_cover),
        ))
        rows.append({
            "capture_id": c.capture_id, "sat": c.sat_name, "time_h": c.time_s / 3600.0, "kind": c.kind,
            "zone": c.zone_name or "-", "lat": c.lat_deg, "lon": c.lon_deg, "day": c.is_daytime,
            "gt_fire_px": int(gt_px[i]), "pred_fire_px": int(npx[i]), "confidence": float(conf[i]),
            "detected": bool(detected[i]), "est_area_ha": area,
            "product": "QUICKLOOK" if quicklook else kind, "size_kb": size / 1e3, "cloud_est": float(clouds[i]),
            "wui_risk": c.wui_risk, "conditions": ", ".join(gen.spec_from_capture(c).tags()),
        })
    gt = gt_px > 0
    tp, fp = int(np.sum(detected & gt)), int(np.sum(detected & ~gt))
    fn, tn = int(np.sum(~detected & gt)), int(np.sum(~detected & ~gt))
    n = len(captures)
    stats = {
        "captures": float(n), "inference_s": infer_s, "ms_per_tile_host": infer_s / n * 1e3,
        "detections": float(detected.sum()), "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "alerts": float(sum(p.kind == ALERT for p in products)),
        "scenes": float(sum(p.kind == SCENE for p in products)),
        "offered_mb": sum(p.size_bytes for p in products) / 1e6,
        "raw_all_mb": n * payload.raw_scene_bytes / 1e6,
        "compressed_all_mb": n * payload.compressed_scene_bytes / 1e6,
        "alert_mb": sum(p.size_bytes for p in products if p.kind == ALERT) / 1e6,
        "quicklooks": float(sum(p.quicklook_only for p in products)),
        "cloud_saved_mb": sum(payload.compressed_scene_bytes - p.size_bytes for p in products if p.quicklook_only) / 1e6,
        "hp_true_fires": float(sum(p.gt_high_priority for p in products)),
        "hp_alerts": float(sum(p.gt_high_priority and p.kind == ALERT for p in products)),
    }
    samples = []
    for i in [j for j in range(n) if detected[j]][:3] + [j for j in range(n) if fp and detected[j] and not gt[j]][:1]:
        samples.append({"capture_id": captures[i].capture_id, "tile": tiles[i], "mask": masks[i], "prob": prob[i],
                        "title": f"capture {captures[i].capture_id} ({captures[i].zone_name or 'background'})"})
    log(f"  onboard inference on {n} captures: {int(detected.sum())} detections "
        f"(TP {tp}, FP {fp}, FN {fn}) in {infer_s:.2f}s on host")
    return OnboardResult(products, rows, stats, processing_latency_s, samples)


def run_pipeline(cfg: SimulationConfig, log: Logger = print, firms_csv: str | None = None) -> PipelineResult:
    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    zone_source = "built-in watch zones (November fire-season scenario)"
    live_ctx: LiveContext | None = None
    if cfg.live.enabled:
        t = time.perf_counter()
        log("[0/4] Live data: NASA FIRMS fires, GeoNames population, CelesTrak TLEs ...")
        live_ctx = prepare_live(cfg, log)
        timings["stage0_live_s"] = time.perf_counter() - t
        if live_ctx:
            zone_source = f"{len(live_ctx.zones)} live NASA FIRMS fire clusters (last 24 h)"
    elif firms_csv:
        try:
            zones = firms_to_watch_zones(load_firms_csv(firms_csv))
            if zones:
                cfg.orbit.zones = zones
                zone_source = f"{len(zones)} zones clustered from FIRMS file {Path(firms_csv).name}"
        except Exception as exc:
            log(f"  FIRMS CSV ignored: {type(exc).__name__}: {exc}")

    t = time.perf_counter()
    log("[1/4] Orbit & downlink pass simulation ...")
    orbit = OrbitAndDownlinkSimulator(cfg.orbit, seed=cfg.seed).run()
    coverage: CoverageResult | None = None
    if live_ctx and live_ctx.tles:
        fires = sorted(live_ctx.clusters[:300], key=lambda c: -(c.frp_total_mw * (0.3 + c.wui_risk)))
        fires = [c for c in fires if any(abs(c.lat - z.lat_deg) < 1e-6 for z in live_ctx.zones)][:cfg.live.coverage_fires]
        try:
            from dataclasses import replace as dc_replace

            scaling = {}
            for name, sats in scaled_constellations(cfg.orbit.satellites).items():
                sc_cfg = dc_replace(cfg.orbit, satellites=sats, backend="analytic", time_step_s=60.0,
                                    duration_hours=cfg.orbit.duration_hours, downlink_tail_hours=0.0)
                scaling[name] = OrbitAndDownlinkSimulator(sc_cfg, seed=cfg.seed).propagate()
            coverage = analyse_coverage(fires, live_ctx.tles, orbit.tracks, orbit.epoch_utc, cfg.orbit.duration_hours,
                                        cfg.orbit.field_of_regard_km, cfg.live.coverage_step_s,
                                        cfg.scheduler.reference_nrt_latency_h,
                                        (cfg.scheduler.processing_overhead_s + 90.0) / 3600.0, log, scaling)
        except Exception as exc:
            log(f"  coverage analysis skipped: {type(exc).__name__}: {exc}")
    timings["stage1_orbit_s"] = time.perf_counter() - t
    log(f"  {orbit.backend}: {len(orbit.passes)} usable passes, {len(orbit.captures)} captures "
        f"({timings['stage1_orbit_s']:.1f}s)")

    t = time.perf_counter()
    log("[2/4] Multispectral fire segmentation ...")
    seg = train_segmentation(cfg, log)
    timings["stage2_segmentation_s"] = time.perf_counter() - t

    t = time.perf_counter()
    log("[3/4] Edge compression & flight-hardware benchmark ...")
    edge = EdgeCompressor(cfg, out_dir, log).run(seg)
    timings["stage3_edge_s"] = time.perf_counter() - t
    log(f"  deployed model: {edge.deployed_name}")

    t = time.perf_counter()
    log("[4/4] Onboard inference + downlink prioritization ...")
    processing = cfg.scheduler.processing_overhead_s + edge.deployed_scene_latency_s()
    onboard = run_onboard_inference(orbit.captures, edge.deployed, cfg, processing, log)
    sched = run_scheduling_comparison(onboard.products, orbit.passes, cfg.scheduler, orbit.horizon_s,
                                      cfg.seed, log)
    timings["stage4_scheduling_s"] = time.perf_counter() - t

    payload = PayloadModel(cfg.payload, cfg.model.tile_size, cfg.model.gsd_m)
    rates = [p.mean_rate_mbps for p in orbit.passes if p.duration_s > 0]
    median_rate = float(np.median(rates)) if rates else 1.0
    energy = downlink_energy_comparison(payload, median_rate, cfg.payload.tx_power_w,
                                        int(seg.payload["alert_kb"] * 1e3))
    return PipelineResult(cfg, out_dir, orbit, seg, edge, onboard, sched, energy, timings, zone_source,
                          live_ctx, coverage)


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------
def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, np.ndarray):
        return None  # arrays are not exported
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_summary(r: PipelineResult) -> dict[str, Any]:
    o, s, e, sch = r.orbit, r.segmentation, r.edge, r.scheduling
    return _jsonable({
        "run": {"seed": r.config.seed, "fast": r.config.fast, "epoch_utc": o.epoch_utc,
                "duration_h": o.duration_s / 3600.0, "horizon_h": o.horizon_s / 3600.0,
                "dependencies": {name: ver for name, ver, _ in deps.dependency_rows()},
                "timings_s": r.timings, "zone_source": r.zone_source},
        "stage1_orbit": {"backend": o.backend, "note": o.backend_note, "satellites": o.satellite_summary(),
                         "stations": o.station_summary(), "contention": o.contention,
                         "captures": o.capture_summary(r.config.scheduler.hp_wui_threshold),
                         "total_capacity_mb": o.total_capacity_bytes / 1e6,
                         "tle": {k: list(v) for k, v in o.tle_lines.items()}},
        "live": None if r.live is None else {
            "epoch_utc": r.live.epoch, "hotspots": r.live.hotspots, "feeds": r.live.feeds, "cities": r.live.cities,
            "tles": r.live.tle_provenance, "zones": [asdict(z) for z in r.live.zones],
            "top_clusters": [live.cluster_to_dict(c) for c in r.live.clusters[:25]]},
        "coverage": None if r.coverage is None else {
            "backend": r.coverage.backend, "sensors": r.coverage.sensors_used, "missing": r.coverage.sensors_missing,
            "summary": r.coverage.summary(), "fires": [asdict(f) for f in r.coverage.fires],
            "scaling": r.coverage.scaling},
        "stage2_segmentation": {"backend": s.backend, "model": s.model_name, "params": s.n_params,
                                "train_time_s": s.train_time_s, "history": s.history,
                                "test_metrics": s.test_metrics, "distractors": s.distractor_breakdown,
                                "baseline_metrics": s.baseline_metrics, "baseline_distractors": s.baseline_breakdown,
                                "real_sentinel2": s.real, "calibration": s.calibration,
                                "payload": s.payload, "real_tiles": s.real_tiles},
        "stage3_edge": {"variants": [{k: v for k, v in asdict(x).items() if k != "path"} for x in e.variants],
                        "deployed": e.deployed_name, "deploy_reason": e.deploy_reason,
                        "onnx": {"path": e.onnx_path, "status": e.onnx_status, "latency_ms": e.onnx_latency_ms},
                        "quant_engine": e.quant_engine, "notes": e.notes, "downlink_energy": r.energy},
        "stage4_scheduling": {"onboard": r.onboard.stats, "processing_latency_s": r.onboard.processing_latency_s,
                              "offered_mb": sch.offered_mb, "capacity_mb": sch.capacity_mb,
                              "load_ratio": sch.load_ratio, "rl_class_ranking": sch.rl_class_ranking,
                              "rl_states_visited": sch.rl_states_visited,
                              "rl_validation_returns": sch.validation_returns,
                              "relay": None if sch.relay is None else {
                                  k: v for k, v in asdict(sch.relay).items() if k != "deliveries"},
                              "policies": [m.as_row() for m in sch.metrics]},
    })


def save_summary(r: PipelineResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_summary(r), indent=2), encoding="utf-8")
    return path
