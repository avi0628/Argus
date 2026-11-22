"""Stage 1 - orbit & downlink pass simulation.

* Multi-satellite sun-synchronous LEO CubeSat constellation.
* Propagation with Skyfield/SGP4 (TLEs generated from the design elements, or read
  from a user TLE file) and an offline analytic J2-perturbed Keplerian fallback.
* Global ground-station network with elevation masks, topocentric geometry,
  pass windows, an adaptive-coding-and-modulation (ACM) link budget and
  first-come-first-served antenna / transmitter contention.
* Imaging captures over wildfire-prone WUI watch zones (day and night) plus daytime
  background survey swaths with distractors (cloud, sun glint, smoke/haze).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import deps
from .config import GroundStation, OrbitConfig, SatelliteSpec, WatchZone
from .geo import (
    JD_SGP4_EPOCH0,
    J2_EARTH,
    MU_EARTH_KM3_S2,
    R_EARTH_KM,
    SECONDS_PER_DAY,
    datetime_to_jd,
    ecef_to_geodetic,
    eci_to_ecef,
    geodetic_to_ecef,
    gmst_rad,
    great_circle_km,
    orbital_period_min,
    raan_for_ltan_deg,
    solve_kepler,
    sso_inclination_deg,
    sun_ecef_unit,
    topocentric,
    up_vector,
)

# (minimum link margin [dB], fraction of the station's max data rate) - DVB-S2-like ACM ladder
ACM_TABLE: tuple[tuple[float, float], ...] = (
    (3.0, 1.00),
    (1.0, 0.75),
    (-1.0, 0.50),
    (-3.5, 0.33),
    (-7.0, 0.20),
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class SatelliteTrack:
    name: str
    times_s: np.ndarray
    ecef_km: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    alt_km: np.ndarray
    sun_elevation_deg: np.ndarray
    inclination_deg: float
    period_min: float


@dataclass(frozen=True)
class PassSlot:
    t0_s: float
    t1_s: float
    capacity_bytes: float
    rate_mbps: float
    elevation_deg: float


@dataclass
class GroundPass:
    sat_name: str
    station_name: str
    band: str
    start_s: float
    end_s: float
    max_elevation_deg: float
    tca_s: float
    min_range_km: float
    slots: list[PassSlot]
    truncated: bool = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def capacity_bytes(self) -> float:
        return float(sum(s.capacity_bytes for s in self.slots))

    @property
    def mean_rate_mbps(self) -> float:
        return self.capacity_bytes * 8.0 / 1e6 / self.duration_s if self.duration_s > 0 else 0.0

    def clipped(self, new_start_s: float) -> GroundPass | None:
        """Copy of the pass starting at ``new_start_s`` (antenna handed over late)."""
        slots: list[PassSlot] = []
        for s in self.slots:
            if s.t1_s <= new_start_s:
                continue
            if s.t0_s < new_start_s:
                frac = (s.t1_s - new_start_s) / max(s.t1_s - s.t0_s, 1e-9)
                s = PassSlot(new_start_s, s.t1_s, s.capacity_bytes * frac, s.rate_mbps, s.elevation_deg)
            slots.append(s)
        if not slots:
            return None
        return replace(self, start_s=new_start_s, slots=slots, truncated=True,
                       max_elevation_deg=max(s.elevation_deg for s in slots))

    def scaled(self, factor: float) -> GroundPass:
        """Copy with capacity scaled (used for domain-randomised RL training episodes)."""
        slots = [PassSlot(s.t0_s, s.t1_s, s.capacity_bytes * factor, s.rate_mbps * factor, s.elevation_deg)
                 for s in self.slots]
        return replace(self, slots=slots)


@dataclass
class FireEvent:
    event_id: int
    zone_name: str
    start_s: float
    end_s: float
    initial_area_ha: float
    growth_per_hour: float
    max_area_ha: float

    def area_at(self, t_s: float) -> float:
        """Active (flaming) area in hectares; exponential growth, capped, 0 when inactive."""
        if t_s < self.start_s or t_s > self.end_s:
            return 0.0
        hours = (t_s - self.start_s) / 3600.0
        return float(min(self.max_area_ha, self.initial_area_ha * math.exp(self.growth_per_hour * hours)))


@dataclass
class Capture:
    capture_id: int
    sat_name: str
    time_s: float
    lat_deg: float
    lon_deg: float
    kind: str  # "watch_zone" | "background"
    zone_name: str | None
    wui_risk: float
    population_density: float
    sun_elevation_deg: float
    fire_present: bool
    fire_area_ha: float
    fire_event_id: int | None
    cloud_cover: float
    glint: bool
    smoke: bool
    haze: float
    burn_scar: bool
    urban: bool
    illumination: float
    seed: int

    @property
    def is_daytime(self) -> bool:
        return self.sun_elevation_deg > 0.0


@dataclass
class OrbitResult:
    backend: str
    backend_note: str
    epoch_utc: datetime
    duration_s: float
    horizon_s: float
    tracks: dict[str, SatelliteTrack]
    passes: list[GroundPass]
    contention: dict[str, int]
    captures: list[Capture]
    fire_events: list[FireEvent]
    tle_lines: dict[str, tuple[str, str]]
    stations: tuple[GroundStation, ...]
    zones: tuple[WatchZone, ...]

    @property
    def satellite_names(self) -> list[str]:
        return list(self.tracks.keys())

    @property
    def total_capacity_bytes(self) -> float:
        return float(sum(p.capacity_bytes for p in self.passes))

    def passes_for(self, sat_name: str) -> list[GroundPass]:
        return [p for p in self.passes if p.sat_name == sat_name]

    def station_summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for gs in self.stations:
            ps = [p for p in self.passes if p.station_name == gs.name]
            contact = sum(p.duration_s for p in ps)
            cap = sum(p.capacity_bytes for p in ps)
            rows.append({
                "station": gs.name, "region": gs.region, "lat_deg": gs.lat_deg, "lon_deg": gs.lon_deg,
                "band": gs.band, "mask_deg": gs.min_elevation_deg, "max_rate_mbps": gs.max_rate_mbps,
                "passes": len(ps), "contact_min": contact / 60.0,
                "mean_rate_mbps": cap * 8.0 / 1e6 / contact if contact > 0 else 0.0,
                "capacity_mb": cap / 1e6,
                "mean_max_elev_deg": float(np.mean([p.max_elevation_deg for p in ps])) if ps else 0.0,
            })
        return rows

    def satellite_summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, tr in self.tracks.items():
            ps = self.passes_for(name)
            caps = [c for c in self.captures if c.sat_name == name]
            rows.append({
                "satellite": name, "mean_alt_km": float(np.mean(tr.alt_km)),
                "inclination_deg": tr.inclination_deg, "period_min": tr.period_min,
                "passes": len(ps), "contact_min": sum(p.duration_s for p in ps) / 60.0,
                "capacity_mb": sum(p.capacity_bytes for p in ps) / 1e6,
                "captures": len(caps), "zone_captures": sum(c.kind == "watch_zone" for c in caps),
            })
        return rows

    def capture_summary(self, hp_wui_threshold: float = 0.6) -> dict[str, int]:
        caps = self.captures
        active_events = {c.fire_event_id for c in caps if c.fire_event_id is not None}
        return {
            "total": len(caps),
            "watch_zone": sum(c.kind == "watch_zone" for c in caps),
            "background": sum(c.kind == "background" for c in caps),
            "fire_present": sum(c.fire_present for c in caps),
            "fire_high_priority": sum(c.fire_present and c.wui_risk >= hp_wui_threshold for c in caps),
            "day": sum(c.is_daytime for c in caps),
            "night": sum(not c.is_daytime for c in caps),
            "cloudy": sum(c.cloud_cover > 0.05 for c in caps),
            "glint": sum(c.glint for c in caps),
            "smoke_or_haze": sum(c.smoke or c.haze > 0 for c in caps),
            "fire_events_observed": len(active_events),
        }


# ---------------------------------------------------------------------------
# Propagators
# ---------------------------------------------------------------------------
class Propagator(Protocol):
    name: str
    inclination_deg: float
    period_min: float

    def ecef_km(self, t_s: np.ndarray) -> np.ndarray: ...


class AnalyticJ2Propagator:
    """Secular-J2-perturbed Keplerian propagator: offline fallback for SGP4.

    Includes nodal regression (sun-synchronous precession), apsidal rotation and
    the J2 mean-motion correction; ignores drag and short-period terms.
    """

    backend = "analytic-J2"

    def __init__(self, spec: SatelliteSpec, epoch_jd: float) -> None:
        self.name = spec.name
        self.epoch_jd = epoch_jd
        self.a_km = R_EARTH_KM + spec.altitude_km
        self.ecc = spec.eccentricity
        incl_deg = (spec.inclination_deg if spec.inclination_deg is not None
                    else sso_inclination_deg(spec.altitude_km, spec.eccentricity))
        self.inclination_deg = float(incl_deg)
        self.incl = math.radians(incl_deg)
        self.raan0 = math.radians(raan_for_ltan_deg(spec.ltan_hours, epoch_jd))
        self.argp0 = math.radians(spec.arg_perigee_deg)
        self.m0 = math.radians(spec.mean_anomaly_deg)
        n0 = math.sqrt(MU_EARTH_KM3_S2 / self.a_km**3)
        p = self.a_km * (1.0 - self.ecc**2)
        k = J2_EARTH * (R_EARTH_KM / p) ** 2
        cos_i = math.cos(self.incl)
        self.raan_dot = -1.5 * n0 * k * cos_i
        self.argp_dot = 0.75 * n0 * k * (5.0 * cos_i**2 - 1.0)
        self.m_dot = n0 * (1.0 + 0.75 * k * math.sqrt(1.0 - self.ecc**2) * (3.0 * cos_i**2 - 1.0))
        self.period_min = 2.0 * math.pi / (self.m_dot + self.argp_dot) / 60.0
        self.element_offset_s = 0.0  # seconds from the element epoch to the simulation epoch

    @classmethod
    def from_tle(cls, name: str, line1: str, line2: str, sim_epoch_jd: float) -> AnalyticJ2Propagator:
        """Offline fallback for real satellites: TLE mean elements propagated with secular J2."""
        yy = int(line1[18:20])
        doy = float(line1[20:32])
        year = 2000 + yy if yy < 57 else 1900 + yy
        tle_jd = datetime_to_jd(datetime(year, 1, 1, tzinfo=sim_epoch_tz())) + doy - 1.0
        incl, raan = float(line2[8:16]), float(line2[17:25])
        ecc = float("0." + line2[26:33].strip())
        argp, mean_anom = float(line2[34:42]), float(line2[43:51])
        rev_per_day = float(line2[52:63])
        n_rad_s = rev_per_day * 2.0 * math.pi / SECONDS_PER_DAY
        a_km = (MU_EARTH_KM3_S2 / n_rad_s**2) ** (1.0 / 3.0)
        spec = SatelliteSpec(name, a_km - R_EARTH_KM, 12.0, mean_anom, ecc, argp, incl)
        prop = cls(spec, sim_epoch_jd)
        prop.raan0 = math.radians(raan)
        prop.element_offset_s = (sim_epoch_jd - tle_jd) * SECONDS_PER_DAY
        return prop

    def ecef_km(self, t_s: np.ndarray) -> np.ndarray:
        t_sim = np.asarray(t_s, dtype=np.float64)
        t = t_sim + self.element_offset_s  # orbital elements evolve from their own epoch
        mean_anom = np.mod(self.m0 + self.m_dot * t, 2.0 * math.pi)
        ecc_anom = solve_kepler(mean_anom, self.ecc)
        nu = 2.0 * np.arctan2(math.sqrt(1.0 + self.ecc) * np.sin(ecc_anom / 2.0),
                              math.sqrt(1.0 - self.ecc) * np.cos(ecc_anom / 2.0))
        r = self.a_km * (1.0 - self.ecc * np.cos(ecc_anom))
        u = self.argp0 + self.argp_dot * t + nu
        raan = self.raan0 + self.raan_dot * t
        cu, su = np.cos(u), np.sin(u)
        co, so = np.cos(raan), np.sin(raan)
        ci, si = math.cos(self.incl), math.sin(self.incl)
        eci = np.stack([r * (co * cu - so * su * ci), r * (so * cu + co * su * ci), r * (su * si)], axis=-1)
        return eci_to_ecef(eci, gmst_rad(self.epoch_jd + t_sim / SECONDS_PER_DAY))


class SkyfieldSGP4Propagator:
    """SGP4 via Skyfield; positions returned in the ITRS (Earth-fixed) frame."""

    backend = "skyfield-SGP4"

    def __init__(self, name: str, earth_satellite: Any, timescale: Any, epoch_utc: datetime) -> None:
        self.name = name
        self.sat = earth_satellite
        self.ts = timescale
        self.epoch_utc = epoch_utc
        model = earth_satellite.model
        self.inclination_deg = float(math.degrees(model.inclo))
        self.period_min = float(2.0 * math.pi / model.no_kozai)  # no_kozai is rad/min

    def ecef_km(self, t_s: np.ndarray) -> np.ndarray:
        from skyfield.framelib import itrs

        e = self.epoch_utc
        seconds = e.second + e.microsecond * 1e-6 + np.asarray(t_s, dtype=np.float64)
        t = self.ts.utc(e.year, e.month, e.day, e.hour, e.minute, seconds)
        xyz = self.sat.at(t).frame_xyz(itrs).km
        return np.asarray(xyz, dtype=np.float64).T


def sim_epoch_tz() -> timezone:
    return timezone.utc


def read_tle_file(path: str | Path) -> list[tuple[str, str, str]]:
    """Parse 2-line or 3-line (name + 2 lines) TLE files."""
    lines = [ln.rstrip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[tuple[str, str, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            out.append((f"SAT-{len(out) + 1}", lines[i], lines[i + 1]))
            i += 2
        elif i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            out.append((lines[i].strip().lstrip("0 ").strip() or f"SAT-{len(out) + 1}", lines[i + 1], lines[i + 2]))
            i += 3
        else:
            i += 1
    return out


def link_rate_mbps(station: GroundStation, range_km: float, reference_range_km: float) -> float:
    """ACM data rate: free-space-loss margin vs. the reference range selects a modcod step."""
    margin_db = 20.0 * math.log10(reference_range_km / max(range_km, 1.0))
    for threshold_db, fraction in ACM_TABLE:
        if margin_db >= threshold_db:
            return station.max_rate_mbps * fraction
    return 0.0


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of contiguous True runs."""
    f = np.asarray(flags, dtype=np.int8)
    if f.size == 0 or not f.any():
        return []
    edges = np.diff(np.concatenate(([0], f, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _interp_crossing(t0: float, t1: float, v0: float, v1: float, level: float) -> float:
    if abs(v1 - v0) < 1e-12:
        return float(t1)
    frac = min(1.0, max(0.0, (level - v0) / (v1 - v0)))
    return float(t0 + frac * (t1 - t0))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class OrbitAndDownlinkSimulator:
    """Propagates the constellation, finds ground passes and generates imaging captures."""

    def __init__(self, cfg: OrbitConfig, seed: int = 0) -> None:
        if cfg.backend.lower() not in ("auto", "skyfield", "analytic"):
            raise ValueError(f"unknown orbit backend {cfg.backend!r} (use auto|skyfield|analytic)")
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.epoch_jd = datetime_to_jd(cfg.epoch_utc)
        self.duration_s = cfg.duration_hours * 3600.0
        self.horizon_s = (cfg.duration_hours + cfg.downlink_tail_hours) * 3600.0
        self.times_s = np.arange(0.0, self.horizon_s + 1e-9, cfg.time_step_s)
        self.tle_lines: dict[str, tuple[str, str]] = {}
        self.backend_name, self.backend_note, self.propagators = self._build_propagators()

    # -- propagator construction ------------------------------------------
    def _build_propagators(self) -> tuple[str, str, list[Propagator]]:
        want = self.cfg.backend.lower()
        note = ""
        if want in ("auto", "skyfield"):
            if deps.SKYFIELD_AVAILABLE:
                try:
                    props = self._build_skyfield()
                    src = f"TLE file {self.cfg.tle_file}" if self.cfg.tle_file else "TLEs generated from design elements"
                    return SkyfieldSGP4Propagator.backend, f"SGP4 via Skyfield ({src})", props
                except Exception as exc:  # pragma: no cover - defensive
                    note = f"Skyfield failed ({type(exc).__name__}: {exc}); "
            else:
                note = "Skyfield/SGP4 not installed; "
        if self.cfg.tle_file:
            note += "TLE file ignored by analytic backend; "
        props: list[Propagator] = [AnalyticJ2Propagator(s, self.epoch_jd) for s in self.cfg.satellites]
        return AnalyticJ2Propagator.backend, note + "analytic J2-perturbed Keplerian propagator", props

    def _build_skyfield(self) -> list[Propagator]:
        from sgp4.api import WGS72, Satrec
        from skyfield.api import EarthSatellite, load

        ts = load.timescale(builtin=True)
        props: list[Propagator] = []
        if self.cfg.tle_file:
            entries = read_tle_file(self.cfg.tle_file)
            if not entries:
                raise ValueError(f"no TLEs found in {self.cfg.tle_file}")
            for name, l1, l2 in entries:
                sat = EarthSatellite(l1, l2, name, ts)
                props.append(SkyfieldSGP4Propagator(name, sat, ts, self.cfg.epoch_utc))
                self.tle_lines[name] = (l1, l2)
            return props
        for k, spec in enumerate(self.cfg.satellites):
            incl = (spec.inclination_deg if spec.inclination_deg is not None
                    else sso_inclination_deg(spec.altitude_km, spec.eccentricity))
            a_km = R_EARTH_KM + spec.altitude_km
            satrec = Satrec()
            satrec.sgp4init(
                WGS72, "i", 90001 + k, self.epoch_jd - JD_SGP4_EPOCH0, spec.bstar, 0.0, 0.0,
                spec.eccentricity, math.radians(spec.arg_perigee_deg), math.radians(incl),
                math.radians(spec.mean_anomaly_deg), math.sqrt(MU_EARTH_KM3_S2 / a_km**3) * 60.0,
                math.radians(raan_for_ltan_deg(spec.ltan_hours, self.epoch_jd)),
            )
            sat = EarthSatellite.from_satrec(satrec, ts)
            sat.name = spec.name
            props.append(SkyfieldSGP4Propagator(spec.name, sat, ts, self.cfg.epoch_utc))
            try:
                from sgp4.exporter import export_tle

                self.tle_lines[spec.name] = tuple(export_tle(satrec))  # type: ignore[assignment]
            except Exception:
                pass
        return props

    # -- stage pipeline -----------------------------------------------------
    def run(self) -> OrbitResult:
        tracks = self.propagate()
        raw = self.compute_passes(tracks)
        passes, contention = self._resolve_contention(raw)
        captures, events = self.generate_captures(tracks)
        return OrbitResult(
            backend=self.backend_name, backend_note=self.backend_note, epoch_utc=self.cfg.epoch_utc,
            duration_s=self.duration_s, horizon_s=self.horizon_s, tracks=tracks, passes=passes,
            contention=contention, captures=captures, fire_events=events, tle_lines=dict(self.tle_lines),
            stations=self.cfg.stations, zones=self.cfg.zones,
        )

    def propagate(self) -> dict[str, SatelliteTrack]:
        jd = self.epoch_jd + self.times_s / SECONDS_PER_DAY
        sun = sun_ecef_unit(jd)
        tracks: dict[str, SatelliteTrack] = {}
        for prop in self.propagators:
            r = prop.ecef_km(self.times_s)
            lat, lon, alt = ecef_to_geodetic(r)
            sun_el = np.degrees(np.arcsin(np.clip(np.sum(up_vector(lat, lon) * sun, axis=-1), -1.0, 1.0)))
            period = prop.period_min if prop.period_min > 0 else orbital_period_min(R_EARTH_KM + float(np.mean(alt)))
            tracks[prop.name] = SatelliteTrack(prop.name, self.times_s, r, lat, lon, alt, sun_el,
                                               float(prop.inclination_deg), float(period))
        return tracks

    def compute_passes(self, tracks: dict[str, SatelliteTrack]) -> list[GroundPass]:
        out: list[GroundPass] = []
        for tr in tracks.values():
            for gs in self.cfg.stations:
                site = geodetic_to_ecef(gs.lat_deg, gs.lon_deg, gs.alt_m / 1000.0)
                el, _az, rng_km = topocentric(tr.ecef_km, gs.lat_deg, gs.lon_deg, site)
                out.extend(self._extract_passes(tr.name, gs, tr.times_s, el, rng_km))
        out.sort(key=lambda p: (p.start_s, p.sat_name))
        return out

    def _extract_passes(self, sat_name: str, gs: GroundStation, times: np.ndarray,
                        el: np.ndarray, rng_km: np.ndarray) -> list[GroundPass]:
        cfg = self.cfg
        dt = cfg.time_step_s
        mask = gs.min_elevation_deg
        passes: list[GroundPass] = []
        last = len(times) - 1
        for s, e in _runs(el >= mask):
            t_start = float(times[s]) if s == 0 else _interp_crossing(times[s - 1], times[s], el[s - 1], el[s], mask)
            t_end = float(times[e]) if e == last else _interp_crossing(times[e], times[e + 1], el[e], el[e + 1], mask)
            if t_end - t_start < cfg.min_pass_duration_s:
                continue
            slots: list[PassSlot] = []
            for k in range(s, e + 1):
                lo = t_start if k == s else float(times[k]) - dt / 2.0
                hi = t_end if k == e else float(times[k]) + dt / 2.0
                if hi <= lo:
                    continue
                rate = link_rate_mbps(gs, float(rng_km[k]), cfg.reference_range_km)
                cap = rate * 1e6 / 8.0 * cfg.link_efficiency * (hi - lo)
                slots.append(PassSlot(lo, hi, cap, rate, float(el[k])))
            if not slots:
                continue
            k_max = s + int(np.argmax(el[s:e + 1]))
            passes.append(GroundPass(sat_name, gs.name, gs.band, t_start, t_end, float(el[k_max]),
                                     float(times[k_max]), float(np.min(rng_km[s:e + 1])), slots))
        return passes

    def _resolve_contention(self, raw: list[GroundPass]) -> tuple[list[GroundPass], dict[str, int]]:
        """One pass per station antenna and one station per satellite transmitter (greedy FCFS)."""
        min_dur = self.cfg.min_pass_duration_s
        stats = {"raw_passes": len(raw), "station_truncated": 0, "station_lost": 0,
                 "sat_truncated": 0, "sat_lost": 0}
        stage1: list[GroundPass] = []
        for gs in self.cfg.stations:
            busy = [-math.inf] * max(1, gs.antennas)
            for p in sorted((q for q in raw if q.station_name == gs.name), key=lambda q: q.start_s):
                k = int(np.argmin(busy))
                if busy[k] <= p.start_s:
                    stage1.append(p)
                    busy[k] = p.end_s
                    continue
                clipped = p.clipped(busy[k]) if p.end_s - busy[k] >= min_dur else None
                if clipped is None:
                    stats["station_lost"] += 1
                    continue
                stats["station_truncated"] += 1
                stage1.append(clipped)
                busy[k] = clipped.end_s
        final: list[GroundPass] = []
        for sat in sorted({p.sat_name for p in stage1}):
            busy_until = -math.inf
            for p in sorted((q for q in stage1 if q.sat_name == sat), key=lambda q: q.start_s):
                if p.start_s >= busy_until:
                    final.append(p)
                    busy_until = p.end_s
                    continue
                clipped = p.clipped(busy_until) if p.end_s - busy_until >= min_dur else None
                if clipped is None:
                    stats["sat_lost"] += 1
                    continue
                stats["sat_truncated"] += 1
                final.append(clipped)
                busy_until = clipped.end_s
        final.sort(key=lambda p: (p.start_s, p.sat_name))
        stats["passes"] = len(final)
        return final, stats

    # -- imaging captures -----------------------------------------------------
    def _sample_fire_events(self) -> list[FireEvent]:
        rng = self.rng
        lookback_s = 48.0 * 3600.0
        events: list[FireEvent] = []
        for zone in self.cfg.zones:
            if zone.live_area_ha > 0:  # live FIRMS fire: burning now, keeps burning through the window
                events.append(FireEvent(
                    event_id=len(events), zone_name=zone.name, start_s=-lookback_s / 2.0,
                    end_s=self.duration_s + lookback_s, initial_area_ha=zone.live_area_ha * 0.7,
                    growth_per_hour=float(rng.uniform(0.0, 0.03)), max_area_ha=min(95.0, zone.live_area_ha * 2.0),
                ))
            lam = zone.fire_events_per_day * (self.duration_s + lookback_s) / SECONDS_PER_DAY
            for _ in range(int(rng.poisson(lam))):
                start = float(rng.uniform(-lookback_s, self.duration_s))
                events.append(FireEvent(
                    event_id=len(events), zone_name=zone.name, start_s=start,
                    end_s=start + float(rng.uniform(10.0, 72.0)) * 3600.0,
                    initial_area_ha=float(np.clip(rng.lognormal(math.log(3.0), 0.9), 0.3, 40.0)),
                    growth_per_hour=float(rng.uniform(0.01, 0.12)),
                    max_area_ha=float(rng.uniform(40.0, 100.0)),
                ))
        return events

    def _scene_conditions(self, sun_el: float, fire: bool, cloud_prob: float, glint_prob: float,
                          urban_prob: float) -> dict[str, Any]:
        rng = self.rng
        day = sun_el > 0.0
        if rng.random() < cloud_prob:
            cloud = float(rng.uniform(0.15, 0.75))
        else:
            cloud = float(rng.uniform(0.0, 0.1)) if rng.random() < 0.3 else 0.0
        return {
            "cloud_cover": cloud,
            "glint": bool(day and sun_el > 20.0 and rng.random() < glint_prob),
            "smoke": bool(fire and rng.random() < 0.6),
            "haze": float(rng.uniform(0.05, 0.3)) if rng.random() < 0.2 else 0.0,
            "burn_scar": bool(rng.random() < (0.7 if fire else 0.12)),
            "urban": bool(rng.random() < urban_prob),
            "illumination": float(np.clip(1.1 * math.sin(math.radians(sun_el)), 0.15, 1.0)) if day else 0.01,
        }

    def _risk_at(self, lat: float, lon: float) -> tuple[float, float]:
        """WUI risk / population proxy decaying with distance from the nearest watch zone."""
        if not self.cfg.zones:
            return 0.0, 0.0
        zl = np.array([[z.lat_deg, z.lon_deg, z.radius_km, z.wui_risk, z.population_density]
                       for z in self.cfg.zones])
        d = great_circle_km(lat, lon, zl[:, 0], zl[:, 1])
        w = np.exp(-np.maximum(0.0, d - zl[:, 2]) / 300.0)
        return float(np.max(zl[:, 3] * w)), float(np.max(zl[:, 4] * w))

    def generate_captures(self, tracks: dict[str, SatelliteTrack]) -> tuple[list[Capture], list[FireEvent]]:
        cfg = self.cfg
        rng = self.rng
        events = self._sample_fire_events()
        by_zone: dict[str, list[FireEvent]] = {}
        for ev in events:
            by_zone.setdefault(ev.zone_name, []).append(ev)
        zones = cfg.zones
        zlat = np.array([z.lat_deg for z in zones])
        zlon = np.array([z.lon_deg for z in zones])
        zreach = np.array([z.radius_km for z in zones]) + cfg.field_of_regard_km
        caps: list[Capture] = []
        n_sats = max(1, len(tracks))
        for si, tr in enumerate(tracks.values()):
            valid = tr.times_s <= self.duration_s
            times, lat, lon, sun = tr.times_s[valid], tr.lat_deg[valid], tr.lon_deg[valid], tr.sun_elevation_deg[valid]
            any_zone = np.zeros(len(times), dtype=bool)
            if zones:
                dist = great_circle_km(lat[:, None], lon[:, None], zlat[None, :], zlon[None, :])
                inside = dist <= zreach[None, :]
                any_zone = inside.any(axis=1)
                for zi, zone in enumerate(zones):
                    last_t = -math.inf
                    for s, e in _runs(inside[:, zi]):
                        k = s + int(np.argmin(dist[s:e + 1, zi]))
                        t = float(times[k])
                        if t - last_t < cfg.zone_revisit_debounce_s:
                            continue
                        last_t = t
                        active = [(ev.area_at(t), ev) for ev in by_zone.get(zone.name, [])]
                        active = [(a, ev) for a, ev in active if a > 0.0]
                        area, event = max(active, key=lambda x: x[0]) if active else (0.0, None)
                        fire = event is not None
                        cond = self._scene_conditions(float(sun[k]), fire, zone.cloud_prob, 0.04,
                                                      0.35 if zone.wui_risk >= 0.5 else 0.05)
                        caps.append(Capture(
                            -1, tr.name, t, zone.lat_deg, zone.lon_deg, "watch_zone", zone.name,
                            zone.wui_risk, zone.population_density, float(sun[k]), fire, area,
                            event.event_id if event is not None else None, seed=int(rng.integers(0, 2**31 - 1)),
                            **cond,
                        ))
            offset = si * cfg.background_interval_s / n_sats
            for tb in np.arange(offset, self.duration_s, cfg.background_interval_s):
                k = int(np.clip(np.searchsorted(times, tb), 0, len(times) - 1))
                if sun[k] < cfg.background_min_sun_elev_deg or any_zone[k]:
                    continue
                wui, pop = self._risk_at(float(lat[k]), float(lon[k]))
                fire = bool(rng.random() < cfg.background_fire_prob)
                event_id: int | None = None
                area = 0.0
                if fire:
                    area = float(np.clip(rng.lognormal(math.log(2.0), 0.9), 0.2, 60.0))
                    event_id = len(events)
                    events.append(FireEvent(event_id, "background", float(times[k]) - 3600.0,
                                            float(times[k]) + 6 * 3600.0, area, 0.0, area))
                cond = self._scene_conditions(float(sun[k]), fire, cfg.background_cloud_prob,
                                              cfg.background_glint_prob, 0.12)
                caps.append(Capture(
                    -1, tr.name, float(times[k]), float(lat[k]), float(lon[k]), "background", None,
                    wui, pop, float(sun[k]), fire, area, event_id, seed=int(rng.integers(0, 2**31 - 1)),
                    **cond,
                ))
        caps.sort(key=lambda c: (c.time_s, c.sat_name))
        caps = [replace(c, capture_id=i) for i, c in enumerate(caps)]
        return caps, events
