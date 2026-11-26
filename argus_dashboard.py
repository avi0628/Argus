"""ARGUS mission console (Streamlit).

Run with either:
    streamlit run argus_dashboard.py
    python argus_cli.py dashboard
Running ``python argus_dashboard.py`` directly re-launches itself under Streamlit.

By default the console opens a saved run (outputs/snapshot.pkl.gz, made by make_snapshot.py), so it loads in
seconds. A fresh run with today's NASA data can be started from the sidebar.
"""
from __future__ import annotations

import gzip
import math
import pickle
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

SNAPSHOT = ROOT / "outputs" / "snapshot.pkl.gz"
REPO_URL = "https://github.com/avi0628/Argus"

# palette (matches the slide deck)
EMBER, AMBER, BLUE, GREEN, GREY = "#FF6B2C", "#FFB547", "#5AA9E6", "#3DDC97", "#7D8AA3"
SAT_COLORS = ("#FF6B2C", "#FFB547", "#5AA9E6", "#3DDC97", "#C792EA", "#F78C6C", "#89DDFF", "#AEB8C8")

CSS = """
<style>
.block-container {padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1400px;}
[data-testid="stMetric"] {background: #121A2B; border: 1px solid #22304A; border-radius: 10px; padding: 12px 14px;}
[data-testid="stMetricLabel"] p {font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; color: #9AA6B8;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important;}
[data-testid="stMetricValue"] {font-family: ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-variant-numeric: tabular-nums; font-size: clamp(1.1rem, 1.75vw, 1.65rem);}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {white-space: normal !important; overflow: visible !important;
  text-overflow: clip !important;}
[data-testid="stMetricDelta"] {font-size: .8rem;}
.argus-top {display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; flex-wrap: wrap;}
.argus-top > div:first-child {flex: 1 1 560px;} .argus-top .status {flex: 0 0 auto;}
.argus-title {font-size: 2.1rem; font-weight: 800; letter-spacing: .14em; color: #fff; line-height: 1.1;}
.argus-title span {color: #FF6B2C;}
.argus-sub {color: #AEB8C8; font-size: .95rem; margin: 4px 0 10px;}
.chip {display: inline-block; padding: 3px 10px; border: 1px solid #22304A; border-radius: 999px; font-size: .78rem;
  margin: 0 6px 6px 0; color: #C9D2E0; background: #0F1626;}
.chip b {color: #FFB547; font-weight: 600;}
.mono {font-family: ui-monospace, "SFMono-Regular", Consolas, monospace; font-variant-numeric: tabular-nums;}
.status {border: 1px solid #22304A; background: #0F1626; border-radius: 10px; padding: 8px 12px; font-size: .8rem;
  color: #C9D2E0; min-width: 250px;}
.status .row {display: flex; align-items: center; gap: 8px; margin: 2px 0;}
.beacon {width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex: none;}
.beacon.ok {background: #3DDC97; box-shadow: 0 0 0 0 rgba(61,220,151,.6); animation: pulse 2.2s infinite;}
.beacon.warn {background: #FFB547;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(61,220,151,.55);} 70% {box-shadow: 0 0 0 7px rgba(61,220,151,0);}
  100% {box-shadow: 0 0 0 0 rgba(61,220,151,0);}}
@media (prefers-reduced-motion: reduce) {.beacon.ok {animation: none;}}
.section-label {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: #7D8AA3; margin: 14px 0 6px;}
.card {background: #121A2B; border: 1px solid #22304A; border-radius: 12px; padding: 14px 16px; height: 100%;}
.card h4 {margin: 0 0 6px; font-size: 1rem; color: #fff;}
.card p {margin: 0; color: #AEB8C8; font-size: .9rem; line-height: 1.45;}
.card .big {font-size: 1.6rem; font-weight: 700;}
.pill {display: inline-block; font-size: .7rem; font-weight: 700; padding: 2px 8px; border-radius: 6px; margin-right: 6px;}
.p1 {background: rgba(255,107,44,.18); color: #FF8A4C;} .p2 {background: rgba(255,181,71,.16); color: #FFB547;}
.p3 {background: rgba(125,138,163,.2); color: #AEB8C8;}
.real {color: #3DDC97; font-weight: 600;} .sim {color: #FFB547; font-weight: 600;}
</style>
"""


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


def load_snapshot(mtime: float = 0.0) -> dict[str, Any] | None:
    """mtime is only a cache key, so a new snapshot file is picked up without restarting the app."""
    if not SNAPSHOT.exists():
        return None
    try:
        with gzip.open(SNAPSHOT, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


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


def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _hhmm(seconds: float) -> str:
    m = int(round(seconds / 60.0))
    return f"T+{m // 60:02d}:{m % 60:02d}"


def _gap_svg() -> str:
    """24-hour timeline: today's fire satellites vs Argus passes vs the daily fire peak."""
    x0, x1 = 30, 770
    hx = lambda h: x0 + (x1 - x0) * h / 24
    parts = [f'<svg viewBox="0 0 800 150" width="100%" role="img" aria-label="Satellite passes versus daily fire peak">',
             f'<rect x="{hx(13):.0f}" y="10" width="{hx(20) - hx(13):.0f}" height="112" rx="8" fill="{EMBER}" fill-opacity=".13"/>',
             f'<text x="{hx(16.5):.0f}" y="116" fill="{EMBER}" font-size="12" font-weight="700" text-anchor="middle">daily fire peak</text>',
             f'<line x1="{x0}" y1="70" x2="{x1}" y2="70" stroke="{GREY}" stroke-width="2"/>']
    for h in range(0, 25, 3):
        parts.append(f'<line x1="{hx(h):.0f}" y1="65" x2="{hx(h):.0f}" y2="75" stroke="{GREY}"/>')
        parts.append(f'<text x="{hx(h):.0f}" y="142" fill="#9AA6B8" font-size="11" text-anchor="middle" '
                     f'font-family="ui-monospace,Consolas,monospace">{h:02d}:00</text>')
    for h in (10.5, 13.5):
        parts.append(f'<circle cx="{hx(h):.0f}" cy="88" r="7" fill="{BLUE}"/>')
        parts.append(f'<text x="{hx(h):.0f}" y="110" fill="{BLUE}" font-size="11" text-anchor="middle" '
                     f'font-family="ui-monospace,Consolas,monospace">{int(h)}:30</text>')
    parts.append(f'<text x="{hx(10.5) - 14:.0f}" y="92" fill="{BLUE}" font-size="12" font-weight="700" '
                 f'text-anchor="end">today\'s fire satellites</text>')
    for h in (15.5, 18.5, 21.5):
        parts.append(f'<circle cx="{hx(h):.0f}" cy="52" r="7" fill="{EMBER}"/>')
        parts.append(f'<text x="{hx(h):.0f}" y="36" fill="{EMBER}" font-size="11" text-anchor="middle" '
                     f'font-family="ui-monospace,Consolas,monospace">{int(h)}:30</text>')
    parts.append(f'<text x="{hx(18.5):.0f}" y="20" fill="{EMBER}" font-size="12" font-weight="700" '
                 f'text-anchor="middle">Argus passes</text>')
    parts.append("</svg>")
    return "".join(parts)


def _card(title: str, body: str, big: str | None = None, color: str = "#fff") -> str:
    b = f'<div class="big" style="color:{color}">{big}</div>' if big else ""
    return f'<div class="card">{b}<h4>{title}</h4><p>{body}</p></div>'


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------
def render() -> None:
    assert st is not None and pd is not None
    st.set_page_config(page_title="Argus · Mission console", page_icon="🛰️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    mtime = SNAPSHOT.stat().st_mtime if SNAPSHOT.exists() else 0.0
    snap = st.cache_resource(show_spinner="Opening the saved run ...")(load_snapshot)(mtime)
    with st.sidebar:
        st.markdown("### Data source")
        options = []
        if snap is not None:
            options.append("Saved run (opens instantly)")
        options += ["Fresh run with today's NASA data", "Scenario: November fire season"]
        source = st.radio("Data source", options, label_visibility="collapsed")
        if source.startswith("Fresh"):
            st.caption("Downloads today's fires and orbits, trains the model and simulates the mission. "
                       "Takes 3 to 5 minutes the first time; then it is cached.")
        elif source.startswith("Saved"):
            st.caption(f"Full run with live NASA data from {snap['meta']['created_utc']}. "
                       "Same numbers as our slides and paper.")
        with st.expander("Advanced settings (fresh runs)", expanded=False):
            offline = st.checkbox("Offline (cached data only)", value=False)
            hours = float(st.slider("Imaging window (h)", 6, 48, 24, step=6))
            seed = int(st.number_input("Seed", value=7, step=1))
            fast = st.checkbox("Fast mode (smaller training run)", value=True)
            use_real = st.checkbox("Train with real Sentinel-2 chips", value=True)
            buffer_mb = float(st.slider("Onboard buffer per satellite (MB)", 128, 4096, 768, step=128))
            rl_episodes = int(st.slider("Q-learning episodes", 50, 1000, 300, step=50))
        st.markdown("---")
        st.markdown("**What's real here**")
        st.markdown("<span class='real'>Real:</span> today's NASA fires, real satellite orbits, real Sentinel-2 "
                    "images, the AI model and its results.<br><span class='sim'>Simulated:</span> the six Argus "
                    "CubeSats (designed, not launched).", unsafe_allow_html=True)
        st.markdown(f"[Source code on GitHub]({REPO_URL})")

    if source.startswith("Saved") and snap is not None:
        result, logs = snap["result"], snap["logs"]
        source_label, source_ok = f"Saved run · {snap['meta']['created_utc']}", True
    else:
        live_mode = source.startswith("Fresh")
        runner = st.cache_resource(show_spinner=False)(compute)
        with st.spinner("Running Argus: live data, orbits, AI training, edge benchmark and downlink (3 to 5 min) ..."):
            result, logs = runner(live_mode, offline if live_mode else False, hours, seed, fast, buffer_mb,
                                  rl_episodes, use_real)
        source_label = "Fresh run · today's NASA data" if live_mode else "Scenario · November fire season"
        source_ok = live_mode

    o, s, e, ob, sch = result.orbit, result.segmentation, result.edge, result.onboard, result.scheduling
    ms = {m.policy: m for m in sch.metrics}
    heur, fifo = ms["Urgency heuristic"], ms["FIFO baseline"]
    cov = result.coverage if (result.coverage and result.coverage.fires) else None
    covsum = cov.summary() if cov else None
    lv = result.live
    planes = sorted({sp.ltan_hours for sp in result.config.satellites}) if hasattr(result.config, "satellites") else []
    ltan = " / ".join(f"{int(h):02d}:{int(round((h % 1) * 60)):02d}" for h in planes) or "15:30 / 18:30 / 21:30"

    # ------------------------------------------------------------- header
    epoch = f"{lv.epoch:%d %b %Y %H:%M} UTC" if lv else f"{o.epoch_utc:%d %b %Y %H:%M} UTC"
    chips = [f"<span class='chip'><b>Argus-{len(o.tracks)}</b> CubeSats · {len(planes) or 3} orbital planes</span>",
             f"<span class='chip'>Sun-synchronous · crossing times <b class='mono'>{ltan}</b></span>",
             f"<span class='chip'>{len(o.stations)} ground stations</span>",
             "<span class='chip'>SDG 9 · also 11, 13, 15</span>"]
    if lv:
        chips.insert(2, f"<span class='chip'><b class='mono'>{lv.hotspots:,}</b> NASA fire hotspots in 24 h · "
                        f"<b class='mono'>{len(o.zones)}</b> fires watched</span>")
    beacon = "ok" if source_ok else "warn"
    st.markdown(f"""
<div class="argus-top">
  <div>
    <div class="argus-title">ARGUS <span>·</span> <span style="color:#fff;font-weight:600;letter-spacing:.02em;font-size:1.25rem">mission console</span></div>
    <div class="argus-sub">Onboard-AI wildfire early warning from CubeSats, flying in the evening hours today's fire satellites miss.</div>
    <div>{''.join(chips)}</div>
  </div>
  <div class="status">
    <div class="row"><span class="beacon {beacon}"></span><b>{source_label}</b></div>
    <div class="row"><span class="beacon" style="background:#5AA9E6"></span>Fire data epoch <span class="mono">{epoch}</span></div>
    <div class="row"><span class="beacon" style="background:#7D8AA3"></span>Orbits: {o.backend}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------- KPIs
    st.markdown("<div class='section-label'>Headline results</div>", unsafe_allow_html=True)
    k = st.columns(4)
    if covsum:
        a, t = covsum["argus_first_alert_h"], covsum["existing_first_alert_h"]
        k[0].metric("First warning of a fire", f"{a:.1f} h", f"{a - t:+.1f} h vs today's {t:.1f} h ({(a - t) / t:+.0%})",
                    delta_color="inverse", help="Average time from start of the window to the first alert, over today's "
                    "real fires, with and without the six Argus satellites (simulation).")
    else:
        k[0].metric("Satellites", f"{len(o.tracks)}")
    k[1].metric("Urgent alerts on time", f"{heur.hp_timely}/{heur.hp_alerts}",
                f"first-come-first-served: {fifo.hp_timely}/{fifo.hp_alerts}", delta_color="off", delta_arrow="off",
                help="Alerts for real fires near people (WUI risk >= 0.6) delivered within 3 hours.")
    k[2].metric("False alarms (fire-free scenes)", _pct(s.test_metrics["tile_false_alarm_rate"]),
                f"physics test: {_pct(s.baseline_metrics.get('tile_false_alarm_rate', float('nan')))}",
                delta_color="off", delta_arrow="off")
    k[3].metric("AI model on the satellite", f"{e.deployed_variant.size_bytes_compressed / 1e3:.0f} kB",
                e.deployed_name, delta_color="off", delta_arrow="off")
    k = st.columns(4)
    k[0].metric("First notice via relay", f"{heur.hp_event_first_notice_min:.1f} min"
                if not math.isnan(heur.hp_event_first_notice_min) else "n/a",
                f"ground stations only: {heur.hp_event_first_alert_min:.0f} min"
                if not math.isnan(heur.hp_event_first_alert_min) else None, delta_color="off", delta_arrow="off")
    k[1].metric("Fire-mapping accuracy (IoU)", f"{s.test_metrics['pixel_iou']:.2f}",
                f"{s.test_metrics['pixel_iou'] - s.baseline_metrics.get('pixel_iou', 0):+.2f} vs physics test")
    k[2].metric("Raw image → alert", f"{s.payload['reduction_vs_raw']:.0f}× smaller",
                f"{s.payload['alert_kb']:.0f} kB alert", delta_color="off", delta_arrow="off")
    if covsum:
        k[3].metric("Longest blind gap", f"{covsum['combined_max_gap_h']:.1f} h",
                    f"{covsum['combined_max_gap_h'] - covsum['existing_max_gap_h']:+.1f} h vs today", delta_color="inverse")
    else:
        k[3].metric("Data vs downlink capacity", f"{sch.load_ratio:.1f}×")

    tabs = st.tabs(["Overview", "Live fire map", "Coverage", "ArgusNet model", "Onboard hardware", "Downlink",
                    "Captures", "Log"])

    # ------------------------------------------------------------- overview
    with tabs[0]:
        st.markdown("#### The gap: fires start when satellites aren't looking")
        st.markdown(_gap_svg(), unsafe_allow_html=True)
        li = (s.real or {}).get("late_ignitions", {}) if s.real else {}
        if li.get("afternoon_hotspots"):
            st.caption(f"Measured from real data: {int(li['not_burning_at_morning_pass'])} of "
                       f"{int(li['afternoon_hotspots'])} afternoon fires were not burning when Sentinel-2 imaged the "
                       f"same place that morning. They started between today's satellite passes.")
        c = st.columns(3)
        c[0].markdown(_card("Evening orbits", f"Three orbital planes crossing at {ltan}, two satellites each, watch "
                                              "the hours when fires peak.", "15:30 → 21:30", EMBER), unsafe_allow_html=True)
        c[1].markdown(_card("AI on board", f"{s.model_name} finds fire pixels on the satellite itself, so no raw "
                                           "image has to reach the ground first.",
                            f"{e.deployed_variant.size_bytes_compressed / 1e3:.0f} kB", AMBER), unsafe_allow_html=True)
        c[2].markdown(_card("Smart downlink", f"A {result.config.scheduler.relay_packet_bytes}-byte flash alert goes out "
                                              "through a satellite relay; fires near people are sent first.",
                            f"{heur.hp_event_first_notice_min:.0f} min" if not math.isnan(heur.hp_event_first_notice_min)
                            else "relay", GREEN), unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Pipeline</div>", unsafe_allow_html=True)
        st.markdown(
            "1. **Live fires** from NASA FIRMS are grouped into fire events and scored by risk to nearby towns.  \n"
            "2. **Orbits** of today's fire satellites (CelesTrak) and the Argus constellation are propagated with SGP4.  \n"
            "3. **Detect**: ArgusNet marks fire pixels in each capture.  \n"
            f"4. **Shrink**: a {s.payload.get('raw_scene_mb', 39.3):.1f} MB scene becomes a {s.payload['alert_kb']:.0f} kB alert product.  \n"
            "5. **Prioritise**: the downlink scheduler sends the most urgent alerts first; flash alerts go via relay.")

    # ------------------------------------------------------------- live map
    with tabs[1]:
        rows: list[dict[str, Any]] = []
        if lv:
            st.markdown(f"#### {lv.hotspots:,} NASA FIRMS hotspots in the last 24 h  ·  epoch {lv.epoch:%Y-%m-%d %H:%MZ}")
            for la, lo, frp in lv.hotspot_points:
                rows.append({"lat": la, "lon": lo, "color": "#ff4b1f", "size": 3000.0 + 300.0 * min(frp, 300.0)})
        else:
            st.markdown("#### Scenario watch zones (choose a live data source for today's fires)")
        for z in o.zones:
            rows.append({"lat": z.lat_deg, "lon": z.lon_deg, "color": AMBER if z.wui_risk >= 0.6 else "#ffe08a",
                         "size": max(z.radius_km, 25.0) * 1500.0})
        for gs in o.stations:
            rows.append({"lat": gs.lat_deg, "lon": gs.lon_deg, "color": BLUE, "size": 90000.0})
        st.map(pd.DataFrame(rows), latitude="lat", longitude="lon", color="color", size="size")
        st.caption("Red: NASA fire hotspots (sample) · amber: fires Argus watches near people (risk ≥ 0.6) · "
                   "pale yellow: other watched fires · blue: ground stations")
        if lv:
            st.markdown("##### Fires Argus is watching, ranked")
            st.dataframe(pd.DataFrame([{
                "fire": f"#{c_.cluster_id}", "nearest city": c_.nearest_city,
                "dist km": round(c_.city_dist_km) if math.isfinite(c_.city_dist_km) else None,
                "people exposed": int(c_.exposed_pop), "risk to people": round(c_.wui_risk, 2),
                "hotspots": c_.n_hotspots, "fire power MW": round(c_.frp_total_mw), "extent km": round(c_.extent_km, 1),
                "lat": round(c_.lat, 3), "lon": round(c_.lon, 3)} for c_ in lv.clusters[:40]]),
                hide_index=True, height=360,
                column_config={"risk to people": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")})
            st.caption("; ".join(f"{k_}: {v}" for k_, v in lv.feeds.items()) + f" | {lv.cities}")

    # ------------------------------------------------------------- coverage
    with tabs[2]:
        st.markdown(f"#### Argus-{len(o.tracks)} ground tracks ({o.backend})")
        rows = []
        for i, (name, tr) in enumerate(o.tracks.items()):
            step = max(1, len(tr.times_s) // 500)
            for la, lo in zip(tr.lat_deg[::step], tr.lon_deg[::step]):
                rows.append({"lat": float(la), "lon": float(lo), "color": SAT_COLORS[i % len(SAT_COLORS)], "size": 15000.0})
        st.map(pd.DataFrame(rows), latitude="lat", longitude="lon", color="color", size="size")
        if cov:
            st.markdown("##### Revisit gaps over today's real fires: today's fire satellites vs + Argus")
            st.caption("Operational sensors from live CelesTrak orbits: " + ", ".join(cov.sensors_used))
            st.dataframe(pd.DataFrame([{
                "fire": f.name, "risk to people": round(f.wui_risk, 2), "looks today": f.existing_looks,
                "+ Argus looks": f.argus_looks, "longest gap today (h)": round(f.existing_max_gap_h, 1),
                "with Argus (h)": round(f.combined_max_gap_h, 1),
                "first warning today (h)": round(f.existing_first_alert_h, 1),
                "with Argus (h) ": round(f.argus_first_alert_h, 1)} for f in cov.fires]), hide_index=True)
            if cov.scaling:
                sc = pd.DataFrame([{"constellation": "today only", "mean longest gap (h)": covsum["existing_max_gap_h"],
                                    "mean first warning (h)": covsum["existing_first_alert_h"]}]
                                  + [{"constellation": x["constellation"], "mean longest gap (h)": x["mean_max_gap_h"],
                                      "mean first warning (h)": x["mean_first_alert_h"]} for x in cov.scaling])
                st.markdown("##### Constellation size: how much each extra satellite helps")
                st.bar_chart(sc.set_index("constellation"), stack=False, color=[GREY, EMBER])
                st.caption("Same three orbital planes, more satellites per plane. Most of the gain comes by 12 satellites.")
        c1, c2 = st.columns(2)
        c1.markdown("##### Satellites")
        c1.dataframe(pd.DataFrame(o.satellite_summary()).round(2), hide_index=True)
        c2.markdown("##### Ground stations")
        c2.dataframe(pd.DataFrame(o.station_summary()).round(2), hide_index=True)

    # ------------------------------------------------------------- model
    with tabs[3]:
        st.markdown(f"#### {s.model_name}: {s.n_params:,} parameters vs the physics threshold test")
        keys = [("pixel IoU", "pixel_iou"), ("pixel F1", "pixel_f1"), ("precision", "pixel_precision"),
                ("recall", "pixel_recall"), ("tile detection rate", "tile_detection_rate"),
                ("tile false-alarm rate", "tile_false_alarm_rate"), ("burn-area error", "area_rel_error")]
        c1, c2 = st.columns([1, 1])
        c1.dataframe(pd.DataFrame([{"metric": n, "ArgusNet": round(s.test_metrics[k_], 3),
                                    "physics test": round(s.baseline_metrics.get(k_, float("nan")), 3)}
                                   for n, k_ in keys]), hide_index=True)
        fa = pd.DataFrame({"ArgusNet": {d["condition"]: d["false_alarm_rate"] for d in s.distractor_breakdown},
                           "physics test": {d["condition"]: d["false_alarm_rate"] for d in s.baseline_breakdown}})
        c2.markdown("**False-alarm rate by distractor** (simulated test, exact labels)")
        c2.bar_chart(fa.fillna(0.0), stack=False, color=[EMBER, GREY], height=260)
        st.markdown("**Training curve**")
        st.line_chart(pd.DataFrame(s.history).set_index("epoch")[["loss", "val_iou"]], color=[GREY, EMBER], height=220)
        if s.real:
            rv = s.real
            st.markdown(f"#### Real Sentinel-2 images: {rv['chips']} held-out chips, {rv['scenes']} scenes, unseen fire events")
            st.dataframe(pd.DataFrame([
                {"metric": "IoU vs physics pseudo-labels", "ArgusNet": round(rv["model"]["pseudo_iou"], 3),
                 "physics test": 1.0},
                {"metric": "FIRMS hotspots burning at S2 pass found", "ArgusNet": rv["model"]["hotspot_recall"],
                 "physics test": rv["baseline"]["hotspot_recall"]},
                {"metric": "detections confirmed by FIRMS", "ArgusNet": rv["model"]["firms_confirmed_rate"],
                 "physics test": rv["baseline"]["firms_confirmed_rate"]},
                {"metric": "false alarms on FIRMS-quiet chips", "ArgusNet": rv["model"]["clean_false_alarm_rate"],
                 "physics test": rv["baseline"]["clean_false_alarm_rate"]},
            ]), hide_index=True)
            st.markdown("**Real chips**: false colour (B12/B8A/B04), physics label (red), ArgusNet (cyan), "
                        "probability; yellow crosses = NASA FIRMS hotspots within 24 h")
            for smp in s.real_samples:
                cols = st.columns(4)
                cols[0].image(_overlay(smp["tile"], None, (0, 0, 0), smp["hotspots"]), caption=smp["title"][:60])
                cols[1].image(_overlay(smp["tile"], smp["label"], (255, 40, 40), smp["hotspots"]), caption="physics label")
                cols[2].image(_overlay(smp["tile"], smp["prob"] >= 0.5, (0, 255, 255), smp["hotspots"]), caption="ArgusNet")
                cols[3].image(_heat(smp["prob"]), caption=f"max p {float(smp['prob'].max()):.2f}")
        with st.expander("Simulated test tiles: ground truth (red) vs ArgusNet (cyan)"):
            for smp in s.samples:
                cols = st.columns(4)
                cols[0].image(_overlay(smp["tile"], None, (0, 0, 0)), caption=smp["tags"])
                cols[1].image(_overlay(smp["tile"], smp["mask"], (255, 40, 40)), caption=f"truth {smp['gt_px']} px")
                cols[2].image(_overlay(smp["tile"], smp["pred_mask"], (0, 255, 255)), caption=f"ArgusNet {smp['pred_px']} px")
                cols[3].image(_heat(smp["prob"]), caption=f"conf {smp['confidence']:.2f} -> {smp['product']}")

    # ------------------------------------------------------------- edge
    with tabs[4]:
        dv = e.deployed_variant
        c = st.columns(4)
        c[0].metric("Deployed model", f"{dv.size_bytes_compressed / 1e3:.0f} kB", dv.precision, delta_color="off", delta_arrow="off")
        c[1].metric("Accuracy kept (IoU)", f"{dv.iou:.3f}", f"{dv.iou_delta:+.3f} vs full model", delta_color="off", delta_arrow="off")
        c[2].metric("Host latency per tile", f"{dv.host_latency_ms:.2f} ms")
        c[3].metric("Working memory", f"{dv.memory_bytes / 1e3:.0f} kB")
        st.dataframe(pd.DataFrame([{
            "variant": v.name, "precision": v.precision, "params": v.params_total, "nonzero": v.params_nonzero,
            "stored kB": round(v.size_bytes_compressed / 1e3, 1), "host ms/tile": round(v.host_latency_ms, 2),
            "memory kB": round(v.memory_bytes / 1e3), "MMAC": round(v.macs / 1e6, 1), "IoU": round(v.iou, 3),
            "dIoU": round(v.iou_delta, 3)} for v in e.variants]), hide_index=True)
        st.markdown("**Modelled onboard hardware** (from operation counts and published throughput)")
        st.dataframe(pd.DataFrame([{
            "variant": v.name, "hardware": h.profile, "ms/tile": round(h.latency_ms_per_tile, 2),
            "mJ/tile": round(h.energy_mj_per_tile, 1), "s/scene": round(h.scene_latency_s, 2),
            "J/scene": round(h.scene_energy_j, 1)} for v in e.variants for h in v.hardware]), hide_index=True)
        st.caption(f"Deployed: {e.deployed_name} · {e.deploy_reason}  |  ONNX: {e.onnx_status}")
        en = result.energy
        st.markdown("**Energy per scene: thinking on board vs sending data down (J)**")
        st.bar_chart(pd.DataFrame({"energy per scene (J)": {
            "downlink raw": en["raw_j"], "downlink lossless": en["compressed_j"], "downlink alert": en["alert_j"],
            "onboard ArgusNet": dv.hardware[0].scene_energy_j if dv.hardware else 0.0}}), color=EMBER, height=260)

    # ------------------------------------------------------------- downlink
    with tabs[5]:
        st.markdown(f"#### Bandwidth-aware downlink · data offered is {sch.load_ratio:.1f}× what the passes can carry")
        alerts = [p for p in ob.products if p.kind == "ALERT"]
        scenes = [p for p in ob.products if p.kind != "ALERT"]
        relay_b = result.config.scheduler.relay_packet_bytes
        q = st.columns(3)
        q[0].markdown(_card("<span class='pill p1'>P1</span>Flash alerts via relay",
                            f"{sch.relay.sent if sch.relay else 0} packets of {relay_b} B: location, size, confidence, "
                            f"risk. Mean capture-to-ground "
                            f"{(sch.relay.mean_latency_s / 60 if sch.relay else float('nan')):.1f} min.",
                            f"{sch.relay.sent if sch.relay else 0}", EMBER), unsafe_allow_html=True)
        q[1].markdown(_card("<span class='pill p2'>P2</span>Fire alert products",
                            f"Fire mask, context chip and quicklook, avg "
                            f"{(np.mean([p.size_bytes for p in alerts]) / 1e3 if alerts else 0):.0f} kB. Delivered by the "
                            f"risk scheduler: {heur.alerts_delivered}/{heur.alerts}.",
                            f"{len(alerts)}", AMBER), unsafe_allow_html=True)
        q[2].markdown(_card("<span class='pill p3'>P3</span>Archive scenes (deferred)",
                            f"Lossless scenes and cloudy quicklooks, {sum(p.size_bytes for p in scenes) / 1e6:,.0f} MB "
                            f"in total. Sent only when the pass has room: "
                            f"{heur.scenes_delivered}/{len(scenes)}.",
                            f"{len(scenes)}", GREY), unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Next ground-station contacts (simulated passes)</div>",
                    unsafe_allow_html=True)
        nxt = sorted(o.passes, key=lambda g: g.start_s)[:10]
        st.dataframe(pd.DataFrame([{
            "AOS": _hhmm(g.start_s), "LOS": _hhmm(g.end_s), "satellite": g.sat_name, "station": g.station_name,
            "band": g.band, "minutes": round(g.duration_s / 60, 1), "max elevation °": round(g.max_elevation_deg, 1),
            "capacity MB": round(g.capacity_bytes / 1e6, 1)} for g in nxt]), hide_index=True)
        st.caption("AOS/LOS = acquisition / loss of signal, as time since the start of the simulated day.")
        st.markdown("<div class='section-label'>Scheduler comparison</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.bar_chart(pd.DataFrame({"urgent alerts on time": {m.policy: m.hp_timely for m in sch.metrics},
                                   "urgent alerts total": {m.policy: m.hp_alerts for m in sch.metrics}}),
                     stack=False, color=[EMBER, GREY], height=280, horizontal=True)
        c2.bar_chart(pd.DataFrame({"mean urgent-alert wait (min)": {m.policy: m.hp_mean_latency_min
                                                                    for m in sch.metrics}}), color=AMBER, height=280, horizontal=True)
        with st.expander("All scheduler metrics"):
            st.dataframe(pd.DataFrame([m.as_row() for m in sch.metrics]).set_index("policy").T, height=420)
            if sch.validation_returns:
                st.bar_chart(pd.DataFrame({"held-out mean mission value": sch.validation_returns}), color=EMBER, horizontal=True)
            st.write("Q-learning class ranking: " + " > ".join(n for n, _ in sch.rl_class_ranking))
        st.caption(f"{int(ob.stats['alerts'])} alerts, {int(ob.stats['scenes'])} scenes, "
                   f"{int(ob.stats.get('quicklooks', 0))} cloudy scenes sent as quicklooks "
                   f"({ob.stats.get('cloud_saved_mb', 0):,.0f} MB saved). Relay speed is an assumption for an "
                   f"Iridium-SBD-class service.")

    with tabs[6]:
        df = pd.DataFrame(ob.detections)
        only = st.checkbox("Only detections", value=True)
        st.dataframe((df[df["detected"]] if only else df).round(3), hide_index=True, height=500)

    with tabs[7]:
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
