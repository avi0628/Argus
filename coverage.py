"""Revisit-gap analysis: how often is each live fire observed by today's operational fire
sensors (live CelesTrak TLEs) versus with the Argus constellation added?

A fire counts as observed when the satellite's sub-point is within the sensor's swath
half-width (Argus: its off-nadir field of regard). Reflective-band imagers (Sentinel-2,
Landsat) only count in daylight; thermal/SWIR fire sensors (VIIRS, MODIS, Argus) count day
and night. Time-to-first-alert adds the delivery path: global FIRMS near-real-time
products (<= 3 h after observation) vs. Argus onboard detection + relay flash alert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import numpy as np

from . import deps
from .geo import SECONDS_PER_DAY, datetime_to_jd, ecef_to_geodetic, great_circle_km, sun_ecef_unit, up_vector
from .live import FIRE_SENSORS, FireCluster
from .orbit import AnalyticJ2Propagator, SatelliteTrack, SkyfieldSGP4Propagator

Logger = Callable[[str], None]
DAYLIGHT_ONLY = ("SENTINEL", "LANDSAT")


@dataclass
class FireCoverage:
    name: str
    lat: float
    lon: float
    wui_risk: float
    existing_looks: int
    argus_looks: int
    existing_max_gap_h: float
    combined_max_gap_h: float
    existing_first_look_h: float
    argus_first_look_h: float
    existing_first_alert_h: float
    argus_first_alert_h: float


@dataclass
class CoverageResult:
    sensors_used: list[str]
    sensors_missing: list[str]
    backend: str
    window_h: float
    fires: list[FireCoverage] = field(default_factory=list)
    scaling: list[dict[str, float]] = field(default_factory=list)  # constellation-size trade study

    def summary(self) -> dict[str, float]:
        if not self.fires:
            return {}

        def m(attr: str) -> float:
            vals = [getattr(f, attr) for f in self.fires if np.isfinite(getattr(f, attr))]
            return float(np.mean(vals)) if vals else float("nan")

        return {k: m(k) for k in ("existing_looks", "argus_looks", "existing_max_gap_h", "combined_max_gap_h",
                                  "existing_first_alert_h", "argus_first_alert_h")}


def _propagator(name: str, l1: str, l2: str, epoch: datetime, epoch_jd: float, ts: Any) -> Any:
    if ts is not None:
        from skyfield.api import EarthSatellite

        return SkyfieldSGP4Propagator(name, EarthSatellite(l1, l2, name, ts), ts, epoch)
    return AnalyticJ2Propagator.from_tle(name, l1, l2, epoch_jd)


def _looks(sub_lat: np.ndarray, sub_lon: np.ndarray, times_s: np.ndarray, lat: float, lon: float,
           reach_km: float, daylight: np.ndarray | None) -> np.ndarray:
    """Mid-times (s) of contiguous visibility runs."""
    vis = great_circle_km(sub_lat, sub_lon, lat, lon) <= reach_km
    if daylight is not None:
        vis &= daylight
    if not vis.any():
        return np.zeros(0)
    edges = np.diff(np.concatenate(([0], vis.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return (times_s[starts] + times_s[ends]) / 2.0


def _max_gap_h(looks: np.ndarray, window_s: float) -> float:
    pts = np.concatenate(([0.0], np.sort(looks), [window_s]))
    return float(np.max(np.diff(pts)) / 3600.0)


def _argus_looks(tracks: dict[str, SatelliteTrack], times: np.ndarray, lat: float, lon: float,
                 reach_km: float) -> np.ndarray:
    looks = []
    for tr in tracks.values():
        la = np.interp(times, tr.times_s, tr.lat_deg)
        lo = np.degrees(np.interp(times, tr.times_s, np.unwrap(np.radians(tr.lon_deg))))
        looks.append(_looks(la, ((lo + 180.0) % 360.0) - 180.0, times, lat, lon, reach_km, None))
    return np.sort(np.concatenate(looks)) if looks else np.zeros(0)


def analyse_coverage(fires: list[FireCluster], tles: dict[str, tuple[str, str]], argus_tracks: dict[str, SatelliteTrack],
                     epoch: datetime, window_h: float, argus_reach_km: float, step_s: float = 60.0,
                     nrt_latency_h: float = 3.0, argus_alert_latency_h: float = 0.05,
                     log: Logger = print,
                     scaling: dict[str, dict[str, SatelliteTrack]] | None = None) -> CoverageResult:
    epoch_jd = datetime_to_jd(epoch)
    times = np.arange(0.0, window_h * 3600.0 + 1e-9, step_s)
    ts = None
    backend = "analytic-J2 from TLE"
    if deps.SKYFIELD_AVAILABLE:
        try:
            from skyfield.api import load

            ts = load.timescale(builtin=True)
            backend = "skyfield-SGP4"
        except Exception:
            ts = None
    used, missing = [], []
    sub: dict[str, tuple[np.ndarray, np.ndarray, float, bool]] = {}
    for name, reach in FIRE_SENSORS.items():
        if name not in tles:
            missing.append(name)
            continue
        try:
            prop = _propagator(name, *tles[name], epoch, epoch_jd, ts)
            la, lo, _alt = ecef_to_geodetic(prop.ecef_km(times))
            sub[name] = (la, lo, reach, any(k in name for k in DAYLIGHT_ONLY))
            used.append(name)
        except Exception as exc:
            missing.append(f"{name} ({type(exc).__name__})")
    sun = sun_ecef_unit(epoch_jd + times / SECONDS_PER_DAY)
    window_s = float(times[-1])
    result = CoverageResult(used, missing, backend, window_h)
    op_looks: list[np.ndarray] = []
    for f in fires:
        sun_el = np.degrees(np.arcsin(np.clip(sun @ up_vector(f.lat, f.lon), -1, 1)))
        day = sun_el > 5.0
        existing = [_looks(la, lo, times, f.lat, f.lon, reach, day if daylit else None)
                    for la, lo, reach, daylit in sub.values()]
        ex = np.sort(np.concatenate(existing)) if existing else np.zeros(0)
        op_looks.append(ex)
        ar = _argus_looks(argus_tracks, times, f.lat, f.lon, argus_reach_km)
        first_ex = float(ex[0] / 3600.0) if ex.size else float("inf")
        first_ar = float(ar[0] / 3600.0) if ar.size else float("inf")
        result.fires.append(FireCoverage(
            name=f.nearest_city if f.nearest_city != "-" else f"fire {f.cluster_id}", lat=f.lat, lon=f.lon,
            wui_risk=f.wui_risk, existing_looks=int(ex.size), argus_looks=int(ar.size),
            existing_max_gap_h=_max_gap_h(ex, window_s), combined_max_gap_h=_max_gap_h(np.concatenate([ex, ar]), window_s),
            existing_first_look_h=first_ex, argus_first_look_h=first_ar,
            existing_first_alert_h=first_ex + nrt_latency_h,
            argus_first_alert_h=min(first_ar + argus_alert_latency_h, first_ex + nrt_latency_h),
        ))
    for name, tracks in (scaling or {}).items():  # constellation-size trade study
        gaps, alerts, looks = [], [], []
        for f, ex in zip(fires, op_looks):
            ar = _argus_looks(tracks, times, f.lat, f.lon, argus_reach_km)
            gaps.append(_max_gap_h(np.concatenate([ex, ar]), window_s))
            fe = ex[0] / 3600.0 + nrt_latency_h if ex.size else float("inf")
            alerts.append(min(fe, ar[0] / 3600.0 + argus_alert_latency_h) if ar.size else fe)
            looks.append(ar.size)
        result.scaling.append({"constellation": name, "satellites": len(tracks), "looks_per_fire": float(np.mean(looks)),
                               "mean_max_gap_h": float(np.mean(gaps)), "worst_gap_h": float(np.max(gaps)),
                               "mean_first_alert_h": float(np.mean([a for a in alerts if np.isfinite(a)] or [np.nan]))})
    log(f"  coverage: {len(used)} operational fire sensors ({backend}) vs +{len(argus_tracks)} Argus sats "
        f"over {len(fires)} live fires" + (f"; scaling study {len(result.scaling)} configs" if scaling else ""))
    return result


def scaled_constellations(base: tuple, per_plane: tuple[int, ...] = (2, 4, 8)) -> dict[str, tuple]:
    """Same planes (LTANs/altitudes) as ``base`` with k evenly phased satellites per plane."""
    from .config import SatelliteSpec

    planes: dict[float, float] = {}
    for s in base:
        planes.setdefault(s.ltan_hours, s.altitude_km)
    out: dict[str, tuple] = {}
    for k in per_plane:
        sats = [SatelliteSpec(f"S{p}-{j}", alt, ltan, 360.0 * j / k + 120.0 * p / k)
                for p, (ltan, alt) in enumerate(planes.items()) for j in range(k)]
        out[f"Argus-{len(sats)}"] = tuple(sats)
    return out
