"""Live data layer: NASA FIRMS active fires, GeoNames population exposure, CelesTrak TLEs
and the Earth Search (AWS) Sentinel-2 L2A STAC catalogue.

Every download is cached under ``data/cache``; with ``offline=True`` (or when the
network fails) the most recent cached copy is used, so Argus always runs.
No API keys are required.
"""
from __future__ import annotations

import csv
import io
import json
import math
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .geo import great_circle_km

USER_AGENT = "Argus-wildfire-research/1.0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = PROJECT_ROOT / "data" / "cache"
Logger = Callable[[str], None]

FIRMS_FEEDS: dict[str, str] = {
    "VIIRS NOAA-20": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_24h.csv",
    "VIIRS NOAA-21": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-21-viirs-c2/csv/J2_VIIRS_C2_Global_24h.csv",
    "MODIS Terra/Aqua": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_24h.csv",
}
# 7-day archives: used to pair Sentinel-2 scenes with same-day fire detections for training data
FIRMS_ARCHIVE_FEEDS: dict[str, str] = {
    "MODIS Terra/Aqua 7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/MODIS_C6_1_Global_7d.csv",
    "VIIRS NOAA-20 7d": "https://firms.modaps.eosdis.nasa.gov/data/active_fire/noaa-20-viirs-c2/csv/J1_VIIRS_C2_Global_7d.csv",
}
GEONAMES_URL ="https://download.geonames.org/export/dump/cities15000.zip"
CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"
STAC_URL = "https://earth-search.aws.element84.com/v1/search"

# Operational fire-observing satellites (CelesTrak names) and their swath half-widths [km]
FIRE_SENSORS: dict[str, float] = {
    "NOAA 20 (JPSS-1)": 1530.0, "NOAA 21 (JPSS-2)": 1530.0, "SUOMI NPP": 1530.0,
    "TERRA": 1165.0, "AQUA": 1165.0,
    "SENTINEL-2A": 145.0, "SENTINEL-2B": 145.0, "SENTINEL-2C": 145.0,
    "LANDSAT 8": 92.5, "LANDSAT 9": 92.5,
}


def _get(url: str, timeout: float = 90.0, data: bytes | None = None, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cached_fetch(url: str, name: str, cache_dir: Path, max_age_h: float, offline: bool,
                 log: Logger | None = None) -> tuple[bytes | None, str]:
    """Return (bytes, provenance). Fresh cache -> cache; else download; on failure -> stale cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    age_h = (time.time() - path.stat().st_mtime) / 3600.0 if path.exists() else math.inf
    if path.exists() and (offline or age_h <= max_age_h):
        return path.read_bytes(), f"cache ({age_h:.1f} h old)"
    if offline:
        return None, "offline, not cached"
    try:
        data = _get(url)
        path.write_bytes(data)
        return data, "live download"
    except Exception as exc:
        if log:
            log(f"  ! {name}: download failed ({type(exc).__name__}: {exc})")
        if path.exists():
            return path.read_bytes(), f"stale cache ({age_h:.1f} h old)"
        return None, f"unavailable ({type(exc).__name__})"


# ---------------------------------------------------------------------------
# FIRMS active fires
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hotspot:
    lat: float
    lon: float
    frp_mw: float
    confidence: str
    acquired_utc: str  # ISO 8601
    satellite: str
    daynight: str
    source: str

    @property
    def time(self) -> datetime:
        return datetime.fromisoformat(self.acquired_utc)


def _parse_firms(raw: bytes, source: str) -> list[Hotspot]:
    out: list[Hotspot] = []
    for row in csv.DictReader(io.StringIO(raw.decode("utf-8", errors="replace"))):
        try:
            t = row.get("acq_time", "0000").zfill(4)
            when = datetime.strptime(f"{row['acq_date']} {t}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
            out.append(Hotspot(float(row["latitude"]), float(row["longitude"]), float(row.get("frp") or 0.0),
                               str(row.get("confidence", "")), when.isoformat(), str(row.get("satellite", "")),
                               str(row.get("daynight", "")), source))
        except (KeyError, ValueError):
            continue
    return out


def load_firms(cache_dir: Path = DEFAULT_CACHE, offline: bool = False, max_age_h: float = 3.0,
               log: Logger | None = None, feeds: dict[str, str] | None = None) -> tuple[list[Hotspot], dict[str, str]]:
    hotspots: list[Hotspot] = []
    meta: dict[str, str] = {}
    for name, url in (feeds or FIRMS_FEEDS).items():
        raw, prov = cached_fetch(url, url.rsplit("/", 1)[-1], cache_dir, max_age_h, offline, log)
        rows = _parse_firms(raw, name) if raw else []
        meta[name] = f"{len(rows):,} hotspots ({prov})"
        hotspots.extend(rows)
    return hotspots, meta


# ---------------------------------------------------------------------------
# Population exposure (GeoNames cities >= 15,000 inhabitants)
# ---------------------------------------------------------------------------
@dataclass
class Cities:
    lat: np.ndarray
    lon: np.ndarray
    pop: np.ndarray
    names: list[str]
    provenance: str = ""

    def exposure(self, lat: float, lon: float, radius_km: float = 50.0) -> tuple[float, str, float]:
        """(distance-weighted population within radius, nearest city, its distance km)."""
        if len(self.pop) == 0:
            return 0.0, "-", math.inf
        box = (np.abs(self.lat - lat) < radius_km / 100.0 + 0.5) & (np.abs(self.lon - lon) < 3.0)
        if not box.any():
            return 0.0, "-", math.inf
        idx = np.flatnonzero(box)
        d = great_circle_km(lat, lon, self.lat[idx], self.lon[idx])
        near = d <= radius_km
        weighted = float(np.sum(self.pop[idx][near] * np.exp(-d[near] / 15.0)))
        j = int(np.argmin(d))
        return weighted, self.names[idx[j]], float(d[j])


def load_cities(cache_dir: Path = DEFAULT_CACHE, offline: bool = False, log: Logger | None = None) -> Cities:
    raw, prov = cached_fetch(GEONAMES_URL, "cities15000.zip", cache_dir, 24.0 * 30, offline, log)
    if not raw:
        return Cities(np.zeros(0), np.zeros(0), np.zeros(0), [], prov)
    lat, lon, pop, names = [], [], [], []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("cities15000.txt") as fh:
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                p = line.split("\t")
                if len(p) > 14:
                    lat.append(float(p[4]))
                    lon.append(float(p[5]))
                    pop.append(float(p[14] or 0))
                    names.append(f"{p[1]} ({p[8]})")
    return Cities(np.array(lat), np.array(lon), np.array(pop), names, f"{len(names):,} cities ({prov})")


def wui_risk_from_exposure(weighted_pop: float) -> float:
    """0..1: 100 people -> 0.33, 10k -> 0.67, 1M -> 1.0 (log scale)."""
    return float(np.clip(math.log10(1.0 + max(weighted_pop, 0.0)) / 6.0, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Fire clusters (fire events)
# ---------------------------------------------------------------------------
@dataclass
class FireCluster:
    cluster_id: int
    lat: float
    lon: float
    n_hotspots: int
    frp_total_mw: float
    frp_max_mw: float
    extent_km: float
    first_utc: str
    last_utc: str
    sources: list[str] = field(default_factory=list)
    exposed_pop: float = 0.0
    nearest_city: str = "-"
    city_dist_km: float = math.inf
    wui_risk: float = 0.0
    peak_lat: float = 0.0
    peak_lon: float = 0.0

    @property
    def est_active_area_ha(self) -> float:
        """Crude active-flame area from FRP (~0.1-0.3 MW per 100 m^2 of flaming front)."""
        return float(np.clip(self.frp_total_mw / 15.0, 0.3, 90.0))


def cluster_hotspots(hotspots: list[Hotspot], link_km: float = 4.0, min_hotspots: int = 1) -> list[FireCluster]:
    """Single-linkage clustering on a lat/lon grid (O(n)); clusters ~= individual fire events."""
    if not hotspots:
        return []
    cell = link_km / 111.0
    lat = np.array([h.lat for h in hotspots])
    lon = np.array([h.lon for h in hotspots])
    gy = np.floor(lat / cell).astype(np.int64)
    gx = np.floor(lon / (cell / np.maximum(np.cos(np.radians(lat)), 0.1))).astype(np.int64)
    cells: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(gy.tolist(), gx.tolist())):
        cells.setdefault(key, []).append(i)
    label: dict[tuple[int, int], int] = {}
    comp = 0
    for start in cells:
        if start in label:
            continue
        stack = [start]
        label[start] = comp
        while stack:
            cy, cx = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (cy + dy, cx + dx)
                    if nb in cells and nb not in label:
                        label[nb] = comp
                        stack.append(nb)
        comp += 1
    members: dict[int, list[int]] = {}
    for key, idx in cells.items():
        members.setdefault(label[key], []).extend(idx)
    out: list[FireCluster] = []
    for idx in members.values():
        if len(idx) < min_hotspots:
            continue
        hs = [hotspots[i] for i in idx]
        frp = np.array([h.frp_mw for h in hs])
        w = np.maximum(frp, 1e-3)
        clat, clon = float(np.average(lat[idx], weights=w)), float(np.average(lon[idx], weights=w))
        peak = hs[int(np.argmax(frp))]
        ext = float(np.max(great_circle_km(clat, clon, lat[idx], lon[idx]))) if len(idx) > 1 else 0.4
        times = sorted(h.acquired_utc for h in hs)
        out.append(FireCluster(-1, clat, clon, len(hs), float(frp.sum()), float(frp.max()), ext, times[0], times[-1],
                               sorted({h.source for h in hs}), peak_lat=peak.lat, peak_lon=peak.lon))
    out.sort(key=lambda c: -c.frp_total_mw)
    for i, c in enumerate(out):
        c.cluster_id = i
    return out


def cluster_by_day(hotspots: list[Hotspot], daytime_only: bool = True, link_km: float = 4.0) -> list[FireCluster]:
    """Cluster separately per UTC day -> 'fire-day' events (for pairing with same-day imagery)."""
    by_day: dict[str, list[Hotspot]] = {}
    for h in hotspots:
        if daytime_only and h.daynight.upper() != "D":
            continue
        by_day.setdefault(h.acquired_utc[:10], []).append(h)
    out: list[FireCluster] = []
    for day in sorted(by_day):
        out.extend(cluster_hotspots(by_day[day], link_km))
    out.sort(key=lambda c: -c.frp_total_mw)
    for i, c in enumerate(out):
        c.cluster_id = i
    return out


def attach_exposure(clusters: list[FireCluster], cities: Cities, top_n: int | None = None) -> None:
    for c in clusters[:top_n] if top_n else clusters:
        c.exposed_pop, c.nearest_city, c.city_dist_km = cities.exposure(c.lat, c.lon)
        c.wui_risk = wui_risk_from_exposure(c.exposed_pop)


# ---------------------------------------------------------------------------
# CelesTrak TLEs
# ---------------------------------------------------------------------------
def load_tles(group: str = "resource", cache_dir: Path = DEFAULT_CACHE, offline: bool = False,
              log: Logger | None = None) -> tuple[dict[str, tuple[str, str]], str]:
    raw, prov = cached_fetch(CELESTRAK_URL.format(group=group), f"celestrak_{group}.tle", cache_dir, 12.0, offline, log)
    tles: dict[str, tuple[str, str]] = {}
    if raw:
        lines = [ln.rstrip() for ln in raw.decode("utf-8", errors="replace").splitlines() if ln.strip()]
        for i in range(0, len(lines) - 2, 3):
            if lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
                tles[lines[i].strip()] = (lines[i + 1], lines[i + 2])
    return tles, f"{len(tles)} TLEs ({prov})"


def load_fire_sensor_tles(cache_dir: Path = DEFAULT_CACHE, offline: bool = False,
                          log: Logger | None = None) -> tuple[dict[str, tuple[str, str]], str]:
    """Earth-resources + weather groups (Landsat, Sentinel-2, Terra/Aqua, NOAA-20/21, Suomi NPP)."""
    merged: dict[str, tuple[str, str]] = {}
    provs = []
    for group in ("resource", "weather"):
        tles, prov = load_tles(group, cache_dir, offline, log)
        merged.update(tles)
        provs.append(f"{group}: {prov}")
    return merged, "; ".join(provs)


# ---------------------------------------------------------------------------
# Sentinel-2 L2A STAC search (Earth Search on AWS)
# ---------------------------------------------------------------------------
def stac_search(lat: float, lon: float, start: datetime, end: datetime, max_cloud: float = 70.0,
                limit: int = 6, cache_dir: Path = DEFAULT_CACHE, offline: bool = False) -> list[dict[str, Any]]:
    key = f"stac_{lat:.3f}_{lon:.3f}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.json"
    path = cache_dir / "stac" / key
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        return []
    body = {
        "collections": ["sentinel-2-l2a"], "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "datetime": f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}", "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    res = json.loads(_get(STAC_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}))
    feats = res.get("features", [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feats), encoding="utf-8")
    return feats


def utc_now_hour() -> datetime:
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def cluster_to_dict(c: FireCluster) -> dict[str, Any]:
    d = asdict(c)
    d["est_active_area_ha"] = c.est_active_area_ha
    return d


def hotspot_window(hotspots: list[Hotspot]) -> tuple[datetime, datetime]:
    times = [h.time for h in hotspots]
    return (min(times), max(times)) if times else (utc_now_hour() - timedelta(days=1), utc_now_hour())
