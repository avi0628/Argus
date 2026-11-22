"""ARGUS dashboard (Streamlit).

Run with either:
    streamlit run argus_dashboard.py
    python argus_cli.py dashboard
Running ``python argus_dashboard.py`` directly re-launches itself under Streamlit.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st

    STREAMLIT_OK = True
except Exception:  # streamlit is optional
    st = None  # type: ignore[assignment]
    STREAMLIT_OK = False

try:
    import pandas as pd

    PANDAS_OK = True
except Exception:
    pd = None  # type: ignore[assignment]
    PANDAS_OK = False

import numpy as np

SAT_COLORS = ("#1f77b4", "#2ca02c", "#9467bd", "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f")


def _in_streamlit_runtime() -> bool:
    try:
        from streamlit import runtime

        return bool(runtime.exists())
    except Exception:
        return False


def compute(live_mode: bool, offline: bool, hours: float, seed: int, fast: bool, buffer_mb: float,
            rl_episodes: int, use_real: bool) -> tuple[Any, list[str]]:
    from argus.config import SimulationConfig
    from argus.pipeline import run_pipeline

    cfg = SimulationConfig(seed=seed)
    if fast:
        cfg.apply_fast_mode()
    cfg.orbit.duration_hours = hours
    cfg.live.enabled = live_mode
    cfg.live.offline = offline
    cfg.model.use_real_data = use_real
    cfg.scheduler.buffer_capacity_mb = buffer_mb
    cfg.scheduler.rl_episodes = rl_episodes
    cfg.output_dir = str(ROOT / "outputs" / "dashboard")
    logs: list[str] = []
    return run_pipeline(cfg, log=logs.append), logs


# ---------------------------------------------------------------------------
# image helpers
# ---------------------------------------------------------------------------
def _rgb(tile: np.ndarray) -> np.ndarray:
    from argus.tiles import false_color

    return false_color(tile).astype(np.float32)


def _overlay(tile: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int],
             hotspots: list[dict] | None = None, scale: int = 4) -> np.ndarray:
    img = _rgb(tile)
    if mask is not None:
        m = np.asarray(mask) > 0
        img[m] = 0.3 * img[m] + 0.7 * np.array(color, dtype=np.float32)
    img = np.kron(img.astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))
    for h in hotspots or []:  # FIRMS hotspots as yellow crosses
        r, c = int(round(h["row"] * scale)), int(round(h["col"] * scale))
        for d in range(-5, 6):
            for rr, cc in ((r + d, c), (r, c + d)):
                if 0 <= rr < img.shape[0] and 0 <= cc < img.shape[1]:
                    img[rr, cc] = (255, 255, 0)
    return img


def _heat(prob: np.ndarray, scale: int = 4) -> np.ndarray:
    p = np.clip(np.asarray(prob), 0.0, 1.0)
    rgb = np.stack([p, p**2, 0.15 * (1 - p)], axis=-1)
    return np.kron((rgb * 255).astype(np.uint8), np.ones((scale, scale, 1), dtype=np.uint8))


def _pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
def render() -> None:
    assert st is not None and pd is not None
    st.set_page_config(page_title="Argus", layout="wide")
    st.title("Argus")
    st.caption("Onboard-AI wildfire early warning from CubeSats: live NASA FIRMS fires, real Sentinel-2 imagery, "
               "bandwidth-aware downlink and relay flash alerts | SDG 9 (with SDG 13, 11, 15)")

    with st.sidebar:
        st.header("Mission")
        mode = st.radio("Fire picture", ("Live: today's NASA FIRMS fires", "Scenario: November fire season"))
        live_mode = mode.startswith("Live")
        offline = st.checkbox("Offline (cached data only)", value=False, disabled=not live_mode)
        hours = float(st.slider("Imaging window (h)", 6, 48, 24, step=6))
        seed = int(st.number_input("Seed", value=7, step=1))
        fast = st.checkbox("Fast mode (smaller training run)", value=True)
        use_real = st.checkbox("Train with real Sentinel-2 chips", value=True)
        buffer_mb = float(st.slider("Onboard buffer per satellite (MB)", 128, 4096, 768, step=128))
        rl_episodes = int(st.slider("Q-learning episodes", 50, 1000, 300, step=50))
        st.caption("Each parameter set runs once, then is cached.")

    runner = st.cache_resource(show_spinner=False)(compute)
    with st.spinner("Running Argus (live data, orbits, ArgusNet, edge benchmark, downlink) ..."):
        result, logs = runner(live_mode, offline, hours, seed, fast, buffer_mb, rl_episodes, use_real)
    o, s, e, ob, sch = result.orbit, result.segmentation, result.edge, result.onboard, result.scheduling
    ms = {m.policy: m for m in sch.metrics}
    heur, fifo = ms["Urgency heuristic"], ms["FIFO baseline"]

    k = st.columns(4)
    k[0].metric("ArgusNet pixel IoU", f"{s.test_metrics['pixel_iou']:.2f}",
                f"{s.test_metrics['pixel_iou'] - s.baseline_metrics.get('pixel_iou', 0):+.2f} vs physics baseline")
    k[1].metric("False alarms", _pct(s.test_metrics["tile_false_alarm_rate"]),
                f"baseline {_pct(s.baseline_metrics.get('tile_false_alarm_rate', float('nan')))}", delta_color="off")
    k[2].metric("WUI alerts on time", f"{heur.hp_timely}/{heur.hp_alerts}", f"FIFO {fifo.hp_timely}/{fifo.hp_alerts}",
                delta_color="off")
    k[3].metric("First WUI warning (relay)", f"{heur.hp_event_first_notice_min:.0f} min"
                if not math.isnan(heur.hp_event_first_notice_min) else "n/a",
                f"ground path {heur.hp_event_first_alert_min:.0f} min" if not math.isnan(heur.hp_event_first_alert_min)
                else None, delta_color="off")
    k = st.columns(4)
    k[0].metric("Onboard model", f"{e.deployed_variant.size_bytes_compressed / 1e3:.0f} kB", e.deployed_name,
                delta_color="off")
    k[1].metric("Scene -> alert", f"{s.payload['reduction_vs_raw']:.0f}x smaller", f"{s.payload['alert_kb']:.0f} kB alert",
                delta_color="off")
    k[2].metric("Data vs downlink", f"{sch.load_ratio:.1f}x")
    if result.coverage and result.coverage.fires:
        sm = result.coverage.summary()
        k[3].metric("Time to first alert (live fires)", f"{sm['argus_first_alert_h']:.1f} h",
                    f"today {sm['existing_first_alert_h']:.1f} h", delta_color="off")
    else:
        k[3].metric("Satellites", f"{len(o.tracks)}")

    tabs = st.tabs(["Live fire map", "Constellation & coverage", "ArgusNet model", "Edge hardware",
                    "Downlink & relay", "Captures", "Log"])

    # ------------------------------------------------------------- live map
    with tabs[0]:
        rows: list[dict[str, Any]] = []
        if result.live:
            lv = result.live
            st.subheader(f"{lv.hotspots:,} NASA FIRMS hotspots in the last 24 h (sample shown) | epoch {lv.epoch:%Y-%m-%d %H:%MZ}")
            for la, lo, frp in lv.hotspot_points:
                rows.append({"lat": la, "lon": lo, "color": "#ff4b1f", "size": 3000.0 + 300.0 * min(frp, 300.0)})
        else:
            st.subheader("Scenario watch zones (switch to Live mode for today's fires)")
        for z in o.zones:
            rows.append({"lat": z.lat_deg, "lon": z.lon_deg, "color": "#ff9f1c" if z.wui_risk >= 0.6 else "#ffd166",
                         "size": max(z.radius_km, 25.0) * 1500.0})
        for gs in o.stations:
            rows.append({"lat": gs.lat_deg, "lon": gs.lon_deg, "color": "#000000", "size": 90000.0})
        st.map(pd.DataFrame(rows), latitude="lat", longitude="lon", color="color", size="size")
        st.caption("Red: FIRMS hotspots | orange: tasked fires near people (WUI risk >= 0.6) | yellow: other tasked "
                   "fires | black: ground stations")
        if result.live:
            st.dataframe(pd.DataFrame([{
                "fire": f"#{c.cluster_id}", "lat": round(c.lat, 3), "lon": round(c.lon, 3), "hotspots": c.n_hotspots,
                "FRP MW": round(c.frp_total_mw), "extent km": round(c.extent_km, 1), "nearest city": c.nearest_city,
                "dist km": round(c.city_dist_km) if math.isfinite(c.city_dist_km) else None,
                "exposed pop": int(c.exposed_pop), "WUI risk": round(c.wui_risk, 2)} for c in result.live.clusters[:40]]),
                hide_index=True, height=360)
            st.caption("; ".join(f"{k_}: {v}" for k_, v in result.live.feeds.items()) + f" | {result.live.cities}")

    # ------------------------------------------------------------- constellation
    with tabs[1]:
        st.subheader(f"Argus-{len(o.tracks)} ground tracks ({o.backend})")
        rows = []
        for i, (name, tr) in enumerate(o.tracks.items()):
            step = max(1, len(tr.times_s) // 500)
            for la, lo in zip(tr.lat_deg[::step], tr.lon_deg[::step]):
                rows.append({"lat": float(la), "lon": float(lo), "color": SAT_COLORS[i % len(SAT_COLORS)], "size": 15000.0})
        st.map(pd.DataFrame(rows), latitude="lat", longitude="lon", color="color", size="size")
        c1, c2 = st.columns(2)
        c1.dataframe(pd.DataFrame(o.satellite_summary()).round(2), hide_index=True)
        c2.dataframe(pd.DataFrame(o.station_summary()).round(2), hide_index=True)
        if result.coverage and result.coverage.fires:
            cv = result.coverage
            st.subheader("Revisit gaps over live fires: today's fire satellites vs. + Argus")
            st.caption("Operational sensors from live CelesTrak TLEs: " + ", ".join(cv.sensors_used))
            st.dataframe(pd.DataFrame([{
                "fire": f.name, "WUI risk": round(f.wui_risk, 2), "looks today": f.existing_looks,
                "+Argus looks": f.argus_looks, "max gap h": round(f.existing_max_gap_h, 1),
                "max gap +Argus h": round(f.combined_max_gap_h, 1),
                "first alert today h": round(f.existing_first_alert_h, 1),
                "first alert Argus h": round(f.argus_first_alert_h, 1)} for f in cv.fires]), hide_index=True)
            if cv.scaling:
                sm = cv.summary()
                sc = pd.DataFrame([{"constellation": "today only", "mean max gap h": sm["existing_max_gap_h"],
                                    "mean first alert h": sm["existing_first_alert_h"]}]
                                  + [{"constellation": x["constellation"], "mean max gap h": x["mean_max_gap_h"],
                                      "mean first alert h": x["mean_first_alert_h"]} for x in cv.scaling])
                st.bar_chart(sc.set_index("constellation"), stack=False)
                st.caption("Constellation scaling: same three orbital planes (15:30 / 18:30 / 21:30 local), "
                           "more satellites per plane")

    # ------------------------------------------------------------- model
    with tabs[2]:
        st.subheader(f"{s.model_name} - {s.n_params:,} parameters vs. physics threshold baseline")
        keys = [("pixel IoU", "pixel_iou"), ("pixel F1", "pixel_f1"), ("precision", "pixel_precision"),
                ("recall", "pixel_recall"), ("tile detection rate", "tile_detection_rate"),
                ("tile false-alarm rate", "tile_false_alarm_rate"), ("burn-area error", "area_rel_error")]
        st.dataframe(pd.DataFrame([{"metric": n, "ArgusNet": round(s.test_metrics[k_], 3),
                                    "physics baseline": round(s.baseline_metrics.get(k_, float("nan")), 3)}
                                   for n, k_ in keys]), hide_index=True)
        fa = pd.DataFrame({"ArgusNet": {d["condition"]: d["false_alarm_rate"] for d in s.distractor_breakdown},
                           "physics baseline": {d["condition"]: d["false_alarm_rate"] for d in s.baseline_breakdown}})
        st.markdown("**False-alarm rate by distractor** (synthetic test, exact labels)")
        st.bar_chart(fa.fillna(0.0), stack=False)
        st.line_chart(pd.DataFrame(s.history).set_index("epoch")[["loss", "val_iou"]])
        if s.real:
            rv = s.real
            st.subheader(f"Real Sentinel-2 L2A: {rv['chips']} held-out chips, {rv['scenes']} scenes, unseen fire events")
            st.dataframe(pd.DataFrame([
                {"metric": "IoU vs physics pseudo-labels", "ArgusNet": round(rv["model"]["pseudo_iou"], 3),
                 "physics baseline": 1.0},
                {"metric": "FIRMS hotspots burning at S2 pass found", "ArgusNet": rv["model"]["hotspot_recall"],
                 "physics baseline": rv["baseline"]["hotspot_recall"]},
                {"metric": "detections confirmed by FIRMS", "ArgusNet": rv["model"]["firms_confirmed_rate"],
                 "physics baseline": rv["baseline"]["firms_confirmed_rate"]},
                {"metric": "false alarms on FIRMS-quiet chips", "ArgusNet": rv["model"]["clean_false_alarm_rate"],
                 "physics baseline": rv["baseline"]["clean_false_alarm_rate"]},
            ]), hide_index=True)
            li = rv.get("late_ignitions", {})
            if li.get("afternoon_hotspots"):
                st.info(f"{int(li['not_burning_at_morning_pass'])} of {int(li['afternoon_hotspots'])} afternoon VIIRS "
                        f"hotspots were not burning in the same-day morning Sentinel-2 image: fires ignite between "
                        f"today's overpasses, which is why Argus flies 15:30 / 18:30 / 21:30 orbits.")
            st.markdown("**Real chips**: false colour (B12/B8A/B04), physics label (red), ArgusNet (cyan), "
                        "probability; yellow crosses = NASA FIRMS hotspots within 24 h")
            for smp in s.real_samples:
                cols = st.columns(4)
                cols[0].image(_overlay(smp["tile"], None, (0, 0, 0), smp["hotspots"]), caption=smp["title"][:60])
                cols[1].image(_overlay(smp["tile"], smp["label"], (255, 40, 40), smp["hotspots"]), caption="physics label")
                cols[2].image(_overlay(smp["tile"], smp["prob"] >= 0.5, (0, 255, 255), smp["hotspots"]), caption="ArgusNet")
                cols[3].image(_heat(smp["prob"]), caption=f"max p {float(smp['prob'].max()):.2f}")
        st.markdown("**Synthetic test tiles**: ground truth (red) and ArgusNet (cyan)")
        for smp in s.samples:
            cols = st.columns(4)
            cols[0].image(_overlay(smp["tile"], None, (0, 0, 0)), caption=smp["tags"])
            cols[1].image(_overlay(smp["tile"], smp["mask"], (255, 40, 40)), caption=f"truth {smp['gt_px']} px")
            cols[2].image(_overlay(smp["tile"], smp["pred_mask"], (0, 255, 255)), caption=f"ArgusNet {smp['pred_px']} px")
            cols[3].image(_heat(smp["prob"]), caption=f"conf {smp['confidence']:.2f} -> {smp['product']}")

    # ------------------------------------------------------------- edge
    with tabs[3]:
        st.dataframe(pd.DataFrame([{
            "variant": v.name, "precision": v.precision, "params": v.params_total, "nonzero": v.params_nonzero,
            "stored kB": round(v.size_bytes_compressed / 1e3, 1), "host ms/tile": round(v.host_latency_ms, 2),
            "memory kB": round(v.memory_bytes / 1e3), "MMAC": round(v.macs / 1e6, 1), "IoU": round(v.iou, 3),
            "dIoU": round(v.iou_delta, 3)} for v in e.variants]), hide_index=True)
        st.dataframe(pd.DataFrame([{
            "variant": v.name, "hardware": h.profile, "ms/tile": round(h.latency_ms_per_tile, 2),
            "mJ/tile": round(h.energy_mj_per_tile, 1), "s/scene": round(h.scene_latency_s, 2),
            "J/scene": round(h.scene_energy_j, 1)} for v in e.variants for h in v.hardware]), hide_index=True)
        st.write(f"**Deployed:** {e.deployed_name} - {e.deploy_reason}")
        st.write(f"**ONNX:** {e.onnx_status}")
        en = result.energy
        st.bar_chart(pd.DataFrame({"energy per scene (J)": {
            "downlink raw": en["raw_j"], "downlink lossless": en["compressed_j"], "downlink alert": en["alert_j"],
            "onboard ArgusNet": e.deployed_variant.hardware[0].scene_energy_j if e.deployed_variant.hardware else 0.0}}))

    # ------------------------------------------------------------- downlink
    with tabs[4]:
        st.write(f"Offered {sch.offered_mb:,.0f} MB vs pass capacity {sch.capacity_mb:,.0f} MB (load {sch.load_ratio:.2f}x); "
                 f"{int(ob.stats['alerts'])} alerts, {int(ob.stats['scenes'])} scenes, "
                 f"{int(ob.stats.get('quicklooks', 0))} cloudy scenes sent as quicklooks "
                 f"({ob.stats.get('cloud_saved_mb', 0):,.0f} MB saved).")
        if sch.relay:
            st.write(f"Relay: {sch.relay.sent} flash alerts of {result.config.scheduler.relay_packet_bytes} B, "
                     f"mean capture-to-ground {sch.relay.mean_latency_s / 60:.1f} min.")
        table = pd.DataFrame([m.as_row() for m in sch.metrics]).set_index("policy").T
        st.dataframe(table, height=420)
        c1, c2 = st.columns(2)
        c1.bar_chart(pd.DataFrame({"WUI alerts on time": {m.policy: m.hp_timely for m in sch.metrics},
                                   "WUI alerts total": {m.policy: m.hp_alerts for m in sch.metrics}}), stack=False)
        c2.bar_chart(pd.DataFrame({"first WUI notice (min)": {
            m.policy + " + relay": (0.0 if math.isnan(m.hp_event_first_notice_min) else m.hp_event_first_notice_min)
            for m in sch.metrics} | {m.policy + " ground only": (0.0 if math.isnan(m.hp_event_first_alert_min)
                                                                 else m.hp_event_first_alert_min) for m in sch.metrics}}))
        if sch.validation_returns:
            st.bar_chart(pd.DataFrame({"held-out mean mission value": sch.validation_returns}))
        st.write("Q-learning class ranking: " + " > ".join(n for n, _ in sch.rl_class_ranking))

    with tabs[5]:
        df = pd.DataFrame(ob.detections)
        only = st.checkbox("Only detections", value=True)
        st.dataframe((df[df["detected"]] if only else df).round(3), hide_index=True, height=500)

    with tabs[6]:
        st.code("\n".join(logs) or "(no log)")


def main() -> int:
    if not STREAMLIT_OK or not PANDAS_OK:
        print("The Argus dashboard needs streamlit and pandas:  pip install streamlit pandas")
        print("The CLI works without them:                      python argus_cli.py run --live")
        return 1
    if not _in_streamlit_runtime():
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())])
    render()
    return 0


if __name__ == "__main__":
    if STREAMLIT_OK and _in_streamlit_runtime():
        render()
    else:
        sys.exit(main())
