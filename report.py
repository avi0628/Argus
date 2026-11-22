"""Plain-text (ASCII-safe) summary report for the CLI."""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

from . import deps
from .pipeline import PipelineResult

WIDTH = 100


def fmt(v: Any, nd: int = 2) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if math.isnan(float(v)):
            return "n/a"
        return f"{float(v):,.{nd}f}"
    return str(v)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], indent: int = 2) -> str:
    cells = [[str(h) for h in headers]] + [[c if isinstance(c, str) else fmt(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    pad = " " * indent

    def line(row: Sequence[str]) -> str:
        return pad + "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(row, widths)))

    sep = pad + "  ".join("-" * w for w in widths)
    return "\n".join([line(cells[0]), sep] + [line(r) for r in cells[1:]])


def banner(title: str) -> str:
    return "\n" + "=" * WIDTH + f"\n {title}\n" + "=" * WIDTH


def section(title: str) -> str:
    return "\n" + title + "\n" + "-" * len(title)


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def render_report(r: PipelineResult) -> str:
    out: list[str] = []
    add: Callable[[str], None] = out.append
    cfg, o, s, e, ob, sch = r.config, r.orbit, r.segmentation, r.edge, r.onboard, r.scheduling
    hp_thr = cfg.scheduler.hp_wui_threshold

    add(banner("ARGUS - the hundred-eyed watcher: onboard-AI wildfire early warning from CubeSats\n"
               " Bandwidth-aware downlink + relay flash alerts | SDG 9 Industry, Innovation & Infrastructure "
               "(also SDG 13, 11, 15)"))
    repro = "deterministic" if cfg.model.train_threads == 1 else f"{cfg.model.train_threads}-thread training"
    add(f"  Run: seed={cfg.seed}  mode={'fast' if cfg.fast else 'full'} ({repro})  "
        f"{'LIVE' if r.live else 'scenario'} epoch={o.epoch_utc:%Y-%m-%d %H:%M}Z  "
        f"imaging window={o.duration_s / 3600:.0f} h (+{(o.horizon_s - o.duration_s) / 3600:.0f} h downlink tail)")
    add(f"  Dependencies: {deps.short_status()}")
    add(f"  Watch zones: {r.zone_source} ({len(o.zones)})")

    if r.live:
        lv = r.live
        add(banner("STAGE 0 - LIVE DATA (no API keys)"))
        for feed, info in lv.feeds.items():
            add(f"  NASA FIRMS {feed}: {info}")
        add(f"  GeoNames population: {lv.cities} | CelesTrak: {lv.tle_provenance}")
        add(f"  {lv.hotspots:,} hotspots -> fire clusters; top live fires by radiative power:")
        add(table(["fire", "lat", "lon", "hotspots", "FRP MW", "extent km", "nearest city", "dist km",
                   "exposed pop", "WUI risk"],
                  [[f"#{c.cluster_id}", f"{c.lat:.2f}", f"{c.lon:.2f}", c.n_hotspots, c.frp_total_mw, c.extent_km,
                    c.nearest_city[:28], c.city_dist_km if np.isfinite(c.city_dist_km) else "-",
                    int(c.exposed_pop), c.wui_risk] for c in lv.clusters[:10]]))
        add(f"  Tasked {len(lv.zones)} fires as Argus watch zones "
            f"({sum(z.wui_risk >= hp_thr for z in lv.zones)} with WUI risk >= {hp_thr}).")

    # ----------------------------------------------------------------- stage 1
    add(banner("STAGE 1 - ORBIT & DOWNLINK PASS SIMULATOR"))
    add(f"  Propagator: {o.backend}  ({o.backend_note})")
    add(section("Constellation (sun-synchronous LEO)"))
    add(table(["satellite", "alt km", "incl deg", "period min", "passes", "contact min", "capacity MB",
               "captures", "zone caps"],
              [[x["satellite"], x["mean_alt_km"], x["inclination_deg"], x["period_min"], x["passes"],
                x["contact_min"], x["capacity_mb"], x["captures"], x["zone_captures"]] for x in o.satellite_summary()]))
    add(section("Ground-station network (elevation masks, ACM link budget)"))
    add(table(["station", "region", "band", "mask deg", "max Mb/s", "passes", "contact min", "mean Mb/s",
               "mean max-el", "capacity MB"],
              [[x["station"], x["region"], x["band"], x["mask_deg"], x["max_rate_mbps"], x["passes"],
                x["contact_min"], x["mean_rate_mbps"], x["mean_max_elev_deg"], x["capacity_mb"]]
               for x in o.station_summary()]))
    c = o.contention
    add(f"  Contention (1 antenna/station, 1 transmitter/sat): {c['raw_passes']} geometric passes -> "
        f"{c['passes']} usable ({c['station_truncated'] + c['sat_truncated']} truncated, "
        f"{c['station_lost'] + c['sat_lost']} lost)")
    add(f"  Total downlink capacity over the horizon: {o.total_capacity_bytes / 1e6:,.0f} MB")
    if o.passes:
        p = max(o.passes, key=lambda q: q.capacity_bytes)
        add(f"  Best pass: {p.sat_name} -> {p.station_name} at T+{p.start_s / 3600:.2f} h, "
            f"{p.duration_s / 60:.1f} min, max el {p.max_elevation_deg:.1f} deg, "
            f"min range {p.min_range_km:.0f} km, {p.capacity_bytes / 1e6:.1f} MB")
    cs = o.capture_summary(hp_thr)
    add(section("Imaging captures"))
    add(f"  {cs['total']} captures: {cs['watch_zone']} tasked WUI/watch-zone + {cs['background']} background survey "
        f"swaths | day {cs['day']} / night {cs['night']}")
    add(f"  Ground truth: {cs['fire_present']} with active fire ({cs['fire_high_priority']} high-priority WUI, "
        f"{cs['fire_events_observed']} distinct fire events) | distractors: cloudy {cs['cloudy']}, "
        f"sun glint {cs['glint']}, smoke/haze {cs['smoke_or_haze']}")
    if o.tle_lines:
        name, (l1, l2) = next(iter(o.tle_lines.items()))
        add(f"  Generated TLE ({name}):\n    {l1}\n    {l2}")
    if r.coverage and r.coverage.fires:
        cv = r.coverage
        add(section(f"Revisit gap over live fires: operational fire sensors vs. + Argus ({cv.backend})"))
        add(f"  Operational sensors from live TLEs: {', '.join(cv.sensors_used)}"
            + (f" | missing: {', '.join(cv.sensors_missing)}" if cv.sensors_missing else ""))
        add(table(["fire", "WUI", "looks today", "+Argus looks", "max gap h", "max gap +Argus h",
                   "first alert today h", "first alert Argus h"],
                  [[f.name[:30], f.wui_risk, f.existing_looks, f.argus_looks, f.existing_max_gap_h,
                    f.combined_max_gap_h, f.existing_first_alert_h, f.argus_first_alert_h] for f in cv.fires]))
        sm = cv.summary()
        add(f"  Mean max revisit gap {sm['existing_max_gap_h']:.1f} h -> {sm['combined_max_gap_h']:.1f} h with Argus; "
            f"mean time to first alert {sm['existing_first_alert_h']:.1f} h (FIRMS NRT, <= "
            f"{cfg.scheduler.reference_nrt_latency_h:.0f} h after overpass) -> {sm['argus_first_alert_h']:.1f} h "
            f"(Argus onboard AI + relay)")
        if cv.scaling:
            add(section("Constellation scaling (same three planes, satellites evenly re-phased, analytic propagation)"))
            add(table(["constellation", "satellites", "Argus looks/fire/day", "mean max gap h", "worst gap h",
                       "mean first alert h"],
                      [["operational sensors only", 0, 0, sm["existing_max_gap_h"],
                        max(f.existing_max_gap_h for f in cv.fires), sm["existing_first_alert_h"]]]
                      + [[x["constellation"], x["satellites"], x["looks_per_fire"], x["mean_max_gap_h"], x["worst_gap_h"],
                          x["mean_first_alert_h"]] for x in cv.scaling]))

    # ----------------------------------------------------------------- stage 2
    add(banner("STAGE 2 - ARGUSNET MULTISPECTRAL FIRE SEGMENTATION"))
    mc = cfg.model
    add(f"  Model: {s.model_name}  [{s.backend}]  params={s.n_params:,}  inputs: Red/B8A/B11/B12 + NBR, SWIR "
        f"difference, log(B12/B11), log(B12/B8A)")
    if s.note:
        add(f"  Note: {s.note}")
    n_real = len(s.real_test) if s.real_test is not None else 0
    add(f"  Data: {mc.n_train}/{mc.n_val}/{mc.n_test} synthetic {mc.tile_size}x{mc.tile_size} tiles at {mc.gsd_m:.0f} m"
        + (f" + real Sentinel-2 L2A chips (held-out test: {n_real} chips, unseen fire events)" if n_real else "")
        + f" | training time {s.train_time_s:.1f}s")
    h = s.history
    if h:
        picks = sorted({0, len(h) // 2, len(h) - 1})
        add("  Training curve: " + "  ".join(f"ep{int(h[i]['epoch'])}: loss {h[i]['loss']:.3f} / val IoU "
                                              f"{h[i]['val_iou']:.3f}" for i in picks))
    m, bm = s.test_metrics, s.baseline_metrics
    add(section("ArgusNet vs. classical physics threshold (synthetic test set, exact labels)"))
    rows_cmp = [("pixel IoU", "pixel_iou"), ("pixel F1", "pixel_f1"), ("pixel precision", "pixel_precision"),
                ("pixel recall", "pixel_recall"), ("tile detection rate", "tile_detection_rate"),
                ("tile false-alarm rate", "tile_false_alarm_rate"), ("burn-area rel. error", "area_rel_error")]
    add(table(["metric", "ArgusNet", "physics baseline"],
              [[name, pct(m[k]) if "rate" in k or "error" in k else f"{m[k]:.3f}",
                pct(bm.get(k, float("nan"))) if "rate" in k or "error" in k else f"{bm.get(k, float('nan')):.3f}"]
               for name, k in rows_cmp]))
    add(section("Robustness to distractors: detection / false-alarm rate (ArgusNet | baseline)"))
    add(table(["condition", "fire tiles", "detect", "detect (base)", "no-fire tiles", "false alarm",
               "false alarm (base)"],
              [[d["condition"], d["fire_tiles"], pct(d["detection_rate"]), pct(b["detection_rate"]), d["nofire_tiles"],
                pct(d["false_alarm_rate"]), pct(b["false_alarm_rate"])]
               for d, b in zip(s.distractor_breakdown, s.baseline_breakdown or s.distractor_breakdown)]))
    if s.real:
        rv = s.real
        add(section(f"Real Sentinel-2 L2A imagery: {rv['chips']} held-out chips from {rv['scenes']} scenes "
                    f"({rv['fire_chips']} with active fire), independent NASA FIRMS check"))
        def cell(d: dict, rate: str, count: str) -> str:
            return f"{pct(d[rate])} of {int(d[count])}"

        add(table(["metric", "ArgusNet", "physics baseline"], [
            ["agreement with physics pseudo-labels (IoU)", f"{rv['model']['pseudo_iou']:.3f}", "1.000 (label source)"],
            ["FIRMS hotspots burning at the S2 pass found", cell(rv["model"], "hotspot_recall", "hotspots"),
             cell(rv["baseline"], "hotspot_recall", "hotspots")],
            ["detections confirmed by FIRMS (+-12 h, 400 m)", cell(rv["model"], "firms_confirmed_rate", "flagged_chips"),
             cell(rv["baseline"], "firms_confirmed_rate", "flagged_chips")],
            ["false alarms on FIRMS-quiet chips (+-24 h)", cell(rv["model"], "clean_false_alarm_rate", "clean_chips"),
             cell(rv["baseline"], "clean_false_alarm_rate", "clean_chips")],
        ]))
        li = rv.get("late_ignitions", {})
        if li.get("afternoon_hotspots"):
            add(f"  Finding: {int(li['not_burning_at_morning_pass'])} of {int(li['afternoon_hotspots'])} afternoon VIIRS "
                f"hotspots ({pct(li['late_ignition_share'])}) were NOT burning in the same-day morning Sentinel-2 image "
                f"-> fires ignite between today's overpasses; Argus flies 15:30 / 18:30 / 21:30 orbits to see them.")
    if s.calibration:
        cb = s.calibration
        add(f"  Confidence calibration: temperature T={cb.get('temperature', 1.0):.2f}; tile-level ECE "
            + (f"{cb['tile_ece_before']:.3f} -> " if "tile_ece_before" in cb else "") + f"{cb['tile_ece_after']:.3f}")
    pl = s.payload
    add(section("Per-capture payload (bandwidth-aware products)"))
    add(f"  Raw scene {pl['scene_px']}x{pl['scene_px']} px x 4 bands x {cfg.payload.bits_per_sample} bit "
        f"({pl['scene_width_km']:.0f} km): {pl['raw_scene_mb']:.1f} MB -> lossless {pl['compressed_scene_mb']:.1f} MB "
        f"-> metadata-assisted alert {pl['alert_kb']:.0f} kB")
    bd = pl["alert_breakdown_kb"]
    add(f"  Alert = metadata {bd['metadata']:.1f} kB + RLE mask {bd['rle_mask']:.2f} kB + context chip "
        f"{bd['context_chip']:.0f} kB + quicklook {bd['quicklook']:.0f} kB  =>  {pl['reduction_vs_raw']:.0f}x smaller "
        f"than raw, {pl['reduction_vs_compressed']:.0f}x smaller than compressed")
    if s.samples:
        add(section("Sample per-tile outputs"))
        add(table(["test tile", "conditions", "GT px", "pred px", "confidence", "est. area ha", "product",
                   "payload kB"],
                  [[f"#{x['index']}", x["tags"], x["gt_px"], x["pred_px"], x["confidence"], x["area_ha"],
                    x["product"], x["payload_kb"]] for x in s.samples]))
    if s.real_tiles:
        add(section("Real tiles"))
        add(table(list(s.real_tiles[0].keys()), [list(x.values()) for x in s.real_tiles]))

    # ----------------------------------------------------------------- stage 3
    add(banner("STAGE 3 - EDGE COMPRESSION & FLIGHT-HARDWARE BENCHMARK"))
    add(table(["variant", "prec", "params (nz)", "disk kB", "zlib kB", "host ms/tile", "p90 ms", "memory kB",
               "MMAC", "pixel IoU", "dIoU", "tile det", "tile FA"],
              [[v.name, v.precision, f"{v.params_total:,} ({v.params_nonzero:,})", v.size_bytes / 1e3,
                v.size_bytes_compressed / 1e3, v.host_latency_ms, v.host_latency_p90_ms, v.memory_bytes / 1e3,
                v.macs / 1e6, f"{v.iou:.3f}", f"{v.iou_delta:+.3f}", pct(v.metrics["tile_detection_rate"]),
                pct(v.metrics["tile_false_alarm_rate"])] for v in e.variants]))
    add(f"  Host latency: real 64x64 tile, batch 1, {cfg.edge.bench_threads} CPU thread(s), flush-to-zero, median of "
        f"{cfg.edge.bench_runs} runs. Memory = weights + activations (no buffer reuse). Unstructured sparsity "
        f"shrinks storage, not dense MACs.")
    for v in e.variants:
        if v.notes:
            add(f"    - {v.name}: {v.notes}")
    add(section(f"Flight-computer estimates (modelled from MACs; {e.tiles_per_scene} tiles per scene)"))
    rows = []
    for v in e.variants:
        for hw in v.hardware:
            rows.append([v.name, hw.profile, hw.latency_ms_per_tile, hw.energy_mj_per_tile, hw.scene_latency_s,
                         hw.scene_energy_j])
    add(table(["variant", "hardware", "ms/tile", "mJ/tile", "s/scene", "J/scene"], rows))
    for hw in cfg.edge.hardware:
        add(f"    - {hw.short_name}: {hw.name}; {hw.active_power_w:.1f} W active; {hw.notes}")
    add(f"  ONNX: {e.onnx_status}" + (f" | ORT latency {e.onnx_latency_ms:.2f} ms" if e.onnx_latency_ms else ""))
    for n in e.notes:
        add(f"  Note: {n}")
    add(f"  >> Deployed onboard: {e.deployed_name} - {e.deploy_reason}")
    en = r.energy
    dv = e.deployed_variant
    add(f"  SWaP check @ median pass rate {en['rate_mbps']:.2f} Mb/s, {cfg.payload.tx_power_w:.0f} W TX: downlinking one "
        f"raw scene costs {en['raw_j']:.0f} J, lossless {en['compressed_j']:.0f} J, alert {en['alert_j']:.1f} J; "
        f"onboard inference costs {dv.hardware[0].scene_energy_j:.1f} J/scene on {dv.hardware[0].profile}"
        if dv.hardware else "")

    # ----------------------------------------------------------------- stage 4
    add(banner("STAGE 4 - DOWNLINK PRIORITIZATION: FIFO vs URGENCY HEURISTIC vs Q-LEARNING"))
    st = ob.stats
    add(f"  Onboard inference ({e.deployed_name}) on {int(st['captures'])} captures: {int(st['detections'])} alerts "
        f"(TP {int(st['tp'])}, FP {int(st['fp'])}), missed {int(st['fn'])} | precision {pct(st['precision'])}, "
        f"recall {pct(st['recall'])} | product ready {ob.processing_latency_s:.0f} s after capture")
    add(f"  Offered load {sch.offered_mb:,.0f} MB ({int(st['alerts'])} alerts = {st['alert_mb']:.1f} MB, "
        f"{int(st['scenes'])} scenes) vs pass capacity {sch.capacity_mb:,.0f} MB  ->  load ratio "
        f"{sch.load_ratio:.2f}x | raw downlink of everything would need {st['raw_all_mb']:,.0f} MB")
    add(f"  Cloud-aware packaging: {int(st.get('quicklooks', 0))} cloud-covered scenes sent as quicklooks, saving "
        f"{st.get('cloud_saved_mb', 0.0):,.0f} MB of downlink")
    if sch.relay:
        rl_ = sch.relay
        add(f"  Argus relay: {rl_.sent} flash alerts ({cfg.scheduler.relay_packet_bytes} B each, "
            f"{rl_.bytes_sent / 1e3:.1f} kB total) via LEO messaging relay, mean capture->ground "
            f"{rl_.mean_latency_s / 60:.1f} min; {rl_.capped} over daily cap")
    add(f"  Onboard buffer {cfg.scheduler.buffer_capacity_mb:.0f} MB per satellite | alert TTL "
        f"{cfg.scheduler.alert_ttl_hours:.1f} h | high-priority = true fire with WUI risk >= {hp_thr}")
    if sch.rl_returns:
        k = max(1, len(sch.rl_returns) // 10)
        add(f"  Q-learning: {len(sch.rl_returns)} training episodes in {sch.rl_train_time_s:.1f}s "
            f"(episode return, first 10% {np.mean(sch.rl_returns[:k]):.1f} / last 10% {np.mean(sch.rl_returns[-k:]):.1f}; "
            f"episodes are randomised so returns are noisy); {sch.rl_states_visited} states visited")
        if sch.validation_returns:
            add(f"  Held-out validation ({sch.validation_episodes} randomised episodes, mean mission value): "
                + " | ".join(f"{n} {v:.1f}" for n, v in sch.validation_returns.items()))
        add("  Learned class value ranking: " + " > ".join(f"{n} ({v:.2f})" for n, v in sch.rl_class_ranking))
    add(section("Policy comparison"))
    ms = sch.metrics
    rows_def: list[tuple[str, Callable[[Any], str]]] = [
        ("WUI/high-priority fire alerts delivered <= TTL", lambda x: f"{x.hp_timely}/{x.hp_alerts}"),
        ("  mean HP alert latency (min)", lambda x: fmt(x.hp_mean_latency_min, 1)),
        ("  HP alerts late / dropped / expired onboard", lambda x: f"{x.hp_late} / {x.hp_dropped} / {x.hp_expired_onboard}"),
        ("WUI fire events alerted (first alert, min)",
         lambda x: f"{x.hp_events_alerted}/{x.hp_events} ({fmt(x.hp_event_first_alert_min, 1)})"),
        ("  + Argus relay: first WUI notification (min)",
         lambda x: f"{x.hp_events_notified}/{x.hp_events} ({fmt(x.hp_event_first_notice_min, 1)})"),
        ("All true-fire alerts delivered <= TTL", lambda x: f"{x.fire_alerts_timely}/{x.fire_alerts}"),
        ("  mean alert latency capture->ground (min)", lambda x: fmt(x.mean_alert_latency_min, 1)),
        ("  median / p90 alert latency (min)",
         lambda x: f"{fmt(x.median_alert_latency_min, 1)} / {fmt(x.p90_alert_latency_min, 1)}"),
        ("  dropped or expired fire alerts", lambda x: str(x.fire_alerts_dropped_or_expired)),
        ("False-alarm bytes sent (MB)", lambda x: fmt(x.false_alarm_mb_sent, 2)),
        ("Products delivered / dropped / pending", lambda x: f"{x.delivered} / {x.dropped} / {x.pending}"),
        ("Scenes delivered", lambda x: str(x.scenes_delivered)),
        ("Bandwidth utilization", lambda x: pct(x.utilization)),
        ("Mission value score", lambda x: fmt(x.mission_value, 1)),
    ]
    add(table(["metric"] + [x.policy for x in ms], [[name] + [f(x) for x in ms] for name, f in rows_def]))

    # ----------------------------------------------------------------- summary
    add(banner("WHY ARGUS - HEADLINE RESULTS"))
    fifo, heur, rl = ms[0], ms[1], ms[2]
    best = max(ms, key=lambda x: (x.hp_timely, x.mission_value))
    model_short = "ArgusNet" if s.backend == "torch" else "SpectralMLP (no-torch fallback)"
    add(f"  1. Better detector: {model_short} pixel IoU {m['pixel_iou']:.2f} vs {bm.get('pixel_iou', float('nan')):.2f} for the "
        f"classical threshold test; false alarms {pct(m['tile_false_alarm_rate'])} vs "
        f"{pct(bm.get('tile_false_alarm_rate', float('nan')))} (glint, industrial heat, cloud edges).")
    if s.real:
        rv = s.real
        add(f"  2. Works on real imagery: on {rv['chips']} held-out Sentinel-2 chips of this week's fires (unseen fire "
            f"events) it matches the physics labels at IoU {rv['model']['pseudo_iou']:.2f}; "
            f"{pct(rv['model']['firms_confirmed_rate'])} of its detections are confirmed by NASA FIRMS "
            f"(baseline {pct(rv['baseline']['firms_confirmed_rate'])}).")
    add(f"  3. Flies on a CubeSat: {e.deployed_name}, {dv.size_bytes_compressed / 1e3:.0f} kB stored, "
        f"~{dv.hardware[0].scene_energy_j:.0f} J per {pl['scene_width_km']:.0f} km scene on {dv.hardware[0].profile} "
        f"vs {en['compressed_j']:.0f} J to downlink that scene." if dv.hardware else "  3. -")
    add(f"  4. Bandwidth-aware: a {pl['raw_scene_mb']:.0f} MB scene becomes a {pl['alert_kb']:.0f} kB alert "
        f"({pl['reduction_vs_raw']:.0f}x); with {sch.load_ratio:.1f}x more data than downlink, prioritisation delivers "
        f"{heur.hp_timely}/{heur.hp_alerts} WUI alerts on time vs {fifo.hp_timely}/{fifo.hp_alerts} for FIFO "
        f"(best policy: {best.policy}).")
    if sch.relay and np.isfinite(heur.hp_event_first_notice_min):
        add(f"  5. Minutes, not hours: relay flash alerts give a first WUI notification "
            f"{heur.hp_event_first_notice_min:.0f} min after the first capture (ground-station path "
            f"{fmt(heur.hp_event_first_alert_min, 0)} min).")
    if r.coverage and r.coverage.fires:
        sm = r.coverage.summary()
        big = r.coverage.scaling[-1] if r.coverage.scaling else None
        add(f"  6. Fills the gaps: over today's live fires Argus-{len(o.tracks)} cuts the mean worst revisit gap "
            f"{sm['existing_max_gap_h']:.1f} h -> {sm['combined_max_gap_h']:.1f} h and time-to-first-alert "
            f"{sm['existing_first_alert_h']:.1f} h -> {sm['argus_first_alert_h']:.1f} h"
            + (f" ({big['constellation']}: {big['mean_max_gap_h']:.1f} h gap, {big['mean_first_alert_h']:.1f} h "
               f"to first alert)." if big else "."))
    add("  SDG 9: low-cost, open space infrastructure and edge-AI innovation for disaster monitoring "
        "(co-benefits: SDG 13 climate, SDG 11 safer WUI communities, SDG 15 forests).")
    add("  Stage timings (s): " + ", ".join(f"{k.split('_')[0]} {v:.1f}" for k, v in r.timings.items()))
    return "\n".join(x for x in out if x is not None)
