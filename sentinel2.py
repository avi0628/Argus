"""Real Sentinel-2 L2A fire-chip dataset built from live NASA FIRMS detections.

For the strongest current fire clusters the builder finds recent Sentinel-2 scenes
(Earth Search STAC on AWS), reads 64x64 chips at 20 m (B04 from the 20 m overview,
B8A, B11, B12 and the SCL scene classification) straight out of the cloud-optimized
GeoTIFFs, pseudo-labels active fire with the physics test, and records which FIRMS
VIIRS/MODIS hotspots fall inside each chip as an independent check.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cog import COGReader, latlon_to_utm
from .fire_index import pseudo_label
from .live import DEFAULT_CACHE, FireCluster, Hotspot, stac_search

Logger = Callable[[str], None]
CHIP = 64
RES = 20.0
SCL_CLOUD = (8, 9, 10)  # cloud medium/high probability, thin cirrus
SCL_NODATA = (0, 1)  # no data, saturated/defective


@dataclass
class RealDataset:
    tiles: np.ndarray  # (N, 4, 64, 64) reflectance: Red, B8A, B11, B12
    labels: np.ndarray  # (N, 64, 64) uint8 pseudo-labels
    ignore: np.ndarray  # (N, 64, 64) bool
    scl: np.ndarray  # (N, 64, 64) uint8
    meta: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.meta)

    def split(self, seed: int = 0, val: float = 0.2, test: float = 0.25) -> dict[str, np.ndarray]:
        """Group split by fire cluster (no scene/region leakage between train/val/test)."""
        groups = sorted({m["group"] for m in self.meta})
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)
        n = len(groups)
        n_test, n_val = max(1, round(n * test)), max(1, round(n * val))
        g_test, g_val = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
        idx = np.arange(len(self.meta))
        grp = np.array([m["group"] for m in self.meta])
        return {"test": idx[np.isin(grp, list(g_test))], "val": idx[np.isin(grp, list(g_val))],
                "train": idx[~np.isin(grp, list(g_test | g_val))]}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), tiles=self.tiles.astype(np.float16), labels=self.labels,
                            ignore=self.ignore, scl=self.scl)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> RealDataset | None:
        npz, js = path.with_suffix(".npz"), path.with_suffix(".json")
        if not (npz.exists() and js.exists()):
            return None
        with np.load(npz) as d:
            return RealDataset(d["tiles"].astype(np.float32), d["labels"], d["ignore"], d["scl"],
                               json.loads(js.read_text(encoding="utf-8")))


def location_group(lat: float, lon: float) -> str:
    """Split key: chips of the same fire location (any day, any pass) share a group."""
    return f"{round(lat * 4) / 4:.2f},{round(lon * 4) / 4:.2f}"


def merge_datasets(a: RealDataset, b: RealDataset) -> RealDataset:
    keep = [i for i, m in enumerate(b.meta)
            if not any(m["scene"] == n["scene"] and abs(m["lat"] - n["lat"]) < 0.01 and abs(m["lon"] - n["lon"]) < 0.01
                       for n in a.meta)]
    return RealDataset(np.concatenate([a.tiles, b.tiles[keep]]), np.concatenate([a.labels, b.labels[keep]]),
                       np.concatenate([a.ignore, b.ignore[keep]]), np.concatenate([a.scl, b.scl[keep]]),
                       a.meta + [b.meta[i] for i in keep])


def _asset_scale(asset: dict[str, Any]) -> tuple[float, float]:
    rb = (asset.get("raster:bands") or [{}])[0]
    return float(rb.get("scale", 1e-4) or 1e-4), float(rb.get("offset", 0.0) or 0.0)


def _epsg(item: dict[str, Any]) -> int:
    p = item["properties"]
    code = p.get("proj:epsg") or p.get("proj:code")
    return int(str(code).split(":")[-1])


def read_chip(item: dict[str, Any], lat: float, lon: float, cache_dir: Path,
              max_cloud: float = 0.5) -> dict[str, Any] | None:
    """Read one 64x64 / 20 m chip centred on (lat, lon); None if cloudy, nodata or off-granule."""
    tile_cache = cache_dir / "cog"
    e, n = latlon_to_utm(lat, lon, _epsg(item))
    a = item["assets"]
    tr = a["swir22"]["proj:transform"]
    col0 = int(round((e - tr[2]) / RES)) - CHIP // 2
    row0 = int(round((tr[5] - n) / RES)) - CHIP // 2
    shape = a["swir22"].get("proj:shape", [5490, 5490])
    if row0 < 0 or col0 < 0 or row0 + CHIP > shape[0] or col0 + CHIP > shape[1]:
        return None
    scl = COGReader(a["scl"]["href"], tile_cache).read_window(row0, col0, CHIP, CHIP).astype(np.uint8)
    nodata = np.isin(scl, SCL_NODATA)
    cloud = np.isin(scl, SCL_CLOUD)
    if nodata.mean() > 0.05 or cloud.mean() > max_cloud:
        return None
    bands = []
    for key in ("red", "nir08", "swir16", "swir22"):
        reader = COGReader(a[key]["href"], tile_cache)
        level = 0
        if key == "red":  # 10 m band: use the 2x overview (20 m) so all bands share one grid
            level = next(i for i, lv in enumerate(reader.levels) if lv.width == shape[1])
        dn = reader.read_window(row0, col0, CHIP, CHIP, level).astype(np.float32)
        scale, offset = _asset_scale(a[key])
        refl = np.where(dn > 0, dn * scale + offset, 0.0)
        bands.append(np.clip(refl, 0.0, 1.6))
    tile = np.stack(bands).astype(np.float32)
    return {"tile": tile, "scl": scl, "invalid": nodata, "cloud_frac": float(cloud.mean()),
            "row0": row0, "col0": col0, "epsg": _epsg(item), "x0": tr[2], "y0": tr[5]}


def _hotspots_in_chip(chip: dict[str, Any], hotspots: list[Hotspot], scene_time: datetime) -> list[dict[str, Any]]:
    out = []
    for h in hotspots:
        e, n = latlon_to_utm(h.lat, h.lon, chip["epsg"])
        r = (chip["y0"] - n) / RES - chip["row0"]
        c = (e - chip["x0"]) / RES - chip["col0"]
        if -10 <= r < CHIP + 10 and -10 <= c < CHIP + 10:
            out.append({"row": r, "col": c, "frp": h.frp_mw, "source": h.source,
                        "dt_h": (h.time - scene_time).total_seconds() / 3600.0})
    return out


def build_real_dataset(fire_days: list[FireCluster], hotspots: list[Hotspot], cache_dir: Path = DEFAULT_CACHE,
                       target_chips: int = 80, max_candidates: int = 160, window_h: float = 8.0,
                       workers: int = 8, seed: int = 0, log: Logger = print) -> RealDataset:
    """Pair same-day Sentinel-2 scenes with 'fire-day' clusters (see ``live.cluster_by_day``)
    and cut a chip on the fire plus an offset chip ~2.5 km away in the same scene."""
    seen: set[tuple[int, int]] = set()
    cands: list[FireCluster] = []
    for c in fire_days:  # strongest first, one per 1-degree cell for geographic diversity
        key = (int(math.floor(c.lat)), int(math.floor(c.lon)))
        if key in seen or c.frp_total_mw < 15.0:
            continue
        seen.add(key)
        cands.append(c)
        if len(cands) >= max_candidates:
            break
    log(f"  {len(cands)} candidate fire-day clusters (strongest per 1-degree cell)")
    lat_all = np.array([h.lat for h in hotspots])
    lon_all = np.array([h.lon for h in hotspots])

    def nearby(c: FireCluster) -> list[Hotspot]:
        idx = np.flatnonzero((np.abs(lat_all - c.peak_lat) < 0.08) & (np.abs(lon_all - c.peak_lon) < 0.1))
        return [hotspots[i] for i in idx]

    def work(c: FireCluster) -> list[dict[str, Any]]:
        rng = np.random.default_rng(seed + c.cluster_id)  # per-thread generator
        t_first, t_last = datetime.fromisoformat(c.first_utc), datetime.fromisoformat(c.last_utc)
        try:
            items = stac_search(c.peak_lat, c.peak_lon, t_first - timedelta(hours=window_h),
                                t_last + timedelta(hours=window_h), cache_dir=cache_dir)
        except Exception as exc:
            return [{"error": f"STAC {type(exc).__name__}: {exc}"}]
        if not items:
            return []
        items.sort(key=lambda it: abs((datetime.fromisoformat(it["properties"]["datetime"].replace("Z", "+00:00"))
                                       - t_first).total_seconds()))
        near_hs = nearby(c)
        rows: list[dict[str, Any]] = []
        for item in items[:2]:
            scene_time = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00"))
            offsets = [(0.0, 0.0)]
            ang = rng.uniform(0, 2 * math.pi)
            offsets.append((0.022 * math.sin(ang), 0.022 * math.cos(ang) / max(math.cos(math.radians(c.lat)), 0.2)))
            got = False
            for k, (dlat, dlon) in enumerate(offsets):
                try:
                    chip = read_chip(item, c.peak_lat + dlat, c.peak_lon + dlon, cache_dir)
                except Exception as exc:
                    rows.append({"error": f"COG {item['id']}: {type(exc).__name__}: {exc}"})
                    continue
                if chip is None:
                    continue
                got = True
                label, ignore = pseudo_label(chip["tile"], chip["invalid"])
                rows.append({
                    "tile": chip["tile"], "label": label, "ignore": ignore, "scl": chip["scl"],
                    "meta": {"group": location_group(c.peak_lat, c.peak_lon), "kind": "centre" if k == 0 else "offset",
                             "scene": item["id"], "datetime": item["properties"]["datetime"],
                             "lat": c.peak_lat + dlat, "lon": c.peak_lon + dlon, "cluster_frp_mw": c.frp_total_mw,
                             "cluster_last_utc": c.last_utc, "nearest_city": c.nearest_city,
                             "exposed_pop": c.exposed_pop, "cloud_frac": chip["cloud_frac"],
                             "fire_px": int(label.sum()),
                             "hotspots": _hotspots_in_chip(chip, near_hs, scene_time)},
                })
            if got:
                break
        return rows

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(cands), workers * 2):  # batches: stop once we have enough chips
            futures = [pool.submit(work, c) for c in cands[start:start + workers * 2]]
            for fut in as_completed(futures):
                done += 1
                for r in fut.result():
                    if "error" in r:
                        errors.append(r["error"])
                    else:
                        rows.append(r)
            n_fire = sum(r["meta"]["fire_px"] > 0 for r in rows)
            log(f"    {done}/{len(cands)} clusters searched: {len(rows)} chips ({n_fire} with visible fire) "
                f"[{time.perf_counter() - t0:.0f}s]")
            if len(rows) >= target_chips:
                break
    if errors:
        log(f"  {len(errors)} chip reads failed (skipped), e.g. {errors[0][:200]}")
    if not rows:
        raise RuntimeError("no usable Sentinel-2 chips found")
    rows.sort(key=lambda r: (r["meta"]["group"], r["meta"]["kind"]))
    return RealDataset(np.stack([r["tile"] for r in rows]), np.stack([r["label"] for r in rows]),
                       np.stack([r["ignore"] for r in rows]), np.stack([r["scl"] for r in rows]),
                       [r["meta"] for r in rows])
