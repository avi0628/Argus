"""Synthetic multispectral (Sentinel-2 / Landsat-like) fire tiles and real-data hooks.

Bands (index order everywhere in this package)::

    0  Red     (S2 B04, 665 nm)
    1  NIR     (S2 B08, 842 nm)
    2  SWIR-1  (S2 B11, 1610 nm)
    3  SWIR-2  (S2 B12, 2190 nm)

Active fires are dominated by *emitted* radiance in SWIR-2 > SWIR-1 >> NIR (the
basis of Sentinel-2/Landsat active-fire algorithms and of NASA FIRMS hotspots),
so the generator injects sub-pixel fire emission on top of reflective land cover.
Distractors that commonly cause false alarms are modelled explicitly: clouds and
cloud shadows, specular sun glint over water, smoke plumes, haze, burn scars,
bright urban roofs and night-time (emission-only) acquisitions.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from . import deps
from .config import WatchZone

BAND_NAMES: tuple[str, ...] = ("Red (B04)", "NIR (B8A)", "SWIR-1 (B11)", "SWIR-2 (B12)")
N_BANDS = 4
N_FEATURES = 8
SENSOR_MAX = 1.6  # reflectance-equivalent saturation
QUANT_LEVELS = 4095  # 12-bit


def _spec(*values: float) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


SPECTRA: dict[str, np.ndarray] = {
    "vegetation": _spec(0.04, 0.32, 0.19, 0.09),
    "dry_grass": _spec(0.13, 0.24, 0.33, 0.25),
    "soil": _spec(0.21, 0.27, 0.35, 0.29),
    "water": _spec(0.03, 0.02, 0.010, 0.006),
    "urban": _spec(0.17, 0.21, 0.25, 0.23),
    "hot_roof": _spec(0.30, 0.34, 0.42, 0.38),
    "burn_scar": _spec(0.06, 0.09, 0.21, 0.20),
    "cloud": _spec(0.62, 0.66, 0.46, 0.30),
    "smoke": _spec(0.24, 0.20, 0.09, 0.05),
}
FIRE_EMISSION = _spec(0.004, 0.05, 0.42, 1.05)  # per unit sub-pixel fire fraction
SMOKE_EXTINCTION = _spec(0.85, 0.60, 0.22, 0.10)  # smoke is nearly transparent in SWIR
HAZE_EXTINCTION = _spec(1.00, 0.70, 0.30, 0.15)
GLINT_SHAPE = _spec(0.95, 1.00, 1.04, 1.08)  # spectrally flat specular reflection


@dataclass(frozen=True)
class TileSpec:
    fire_px: int = 0
    cloud_cover: float = 0.0
    glint: bool = False
    smoke: bool = False
    haze: float = 0.0
    burn_scar: bool = False
    urban: bool = False
    illumination: float = 1.0
    seed: int = 0

    @property
    def has_fire(self) -> bool:
        return self.fire_px > 0

    @property
    def is_night(self) -> bool:
        return self.illumination < 0.1

    def tags(self) -> list[str]:
        out = []
        if self.has_fire:
            out.append(f"fire({self.fire_px}px)")
        if self.cloud_cover > 0.05:
            out.append(f"cloud{int(self.cloud_cover * 100)}%")
        if self.glint:
            out.append("glint")
        if self.smoke:
            out.append("smoke")
        if self.haze > 0:
            out.append("haze")
        if self.burn_scar:
            out.append("scar")
        if self.urban:
            out.append("urban")
        if self.is_night:
            out.append("night")
        return out or ["clear"]


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + h, dx:dx + w]
    return out


class MultispectralTileGenerator:
    """Procedural 4-band tiles with pixel-accurate active-fire masks."""

    def __init__(self, tile_size: int = 64, gsd_m: float = 30.0) -> None:
        if tile_size < 16 or tile_size % 4:
            raise ValueError("tile_size must be a multiple of 4 and >= 16")
        self.size = tile_size
        self.gsd_m = gsd_m
        yy, xx = np.mgrid[0:tile_size, 0:tile_size]
        self._yy = yy.astype(np.float32)
        self._xx = xx.astype(np.float32)
        f = np.fft.fftfreq(tile_size)
        self._f2 = (f[:, None] ** 2 + f[None, :] ** 2).astype(np.float64)

    @property
    def pixel_area_ha(self) -> float:
        return self.gsd_m**2 / 1e4

    def smooth_field(self, rng: np.random.Generator, sigma_px: float) -> np.ndarray:
        """Zero-mean, unit-variance Gaussian random field (FFT Gaussian filter)."""
        noise = rng.standard_normal((self.size, self.size))
        kernel = np.exp(-2.0 * math.pi**2 * sigma_px**2 * self._f2)
        field = np.real(np.fft.ifft2(np.fft.fft2(noise) * kernel))
        std = field.std()
        return ((field - field.mean()) / (std if std > 1e-12 else 1.0)).astype(np.float32)

    def _unit_field(self, rng: np.random.Generator, sigma_px: float) -> np.ndarray:
        f = self.smooth_field(rng, sigma_px)
        return (f - f.min()) / max(float(f.max() - f.min()), 1e-6)

    # -- public API -----------------------------------------------------------
    def random_spec(self, rng: np.random.Generator, p_fire: float = 0.5) -> TileSpec:
        n = self.size
        fire = rng.random() < p_fire
        night = rng.random() < 0.15
        return TileSpec(
            fire_px=int(np.clip(rng.lognormal(math.log(35.0), 1.0), 2, 0.3 * n * n)) if fire else 0,
            cloud_cover=float(rng.uniform(0.1, 0.8)) if rng.random() < 0.35 else 0.0,
            glint=bool((not night) and rng.random() < 0.2),
            smoke=bool(fire and rng.random() < 0.55),
            haze=float(rng.uniform(0.05, 0.35)) if rng.random() < 0.2 else 0.0,
            burn_scar=bool(rng.random() < (0.6 if fire else 0.2)),
            urban=bool(rng.random() < 0.2),
            illumination=float(rng.uniform(0.005, 0.02)) if night else float(rng.uniform(0.3, 1.0)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    def spec_from_capture(self, capture: Any) -> TileSpec:
        """Map an orbital ``Capture`` (Stage 1) onto a tile specification."""
        fire_px = 0
        if capture.fire_present:
            fire_px = int(np.clip(round(capture.fire_area_ha / self.pixel_area_ha), 1, 0.35 * self.size**2))
        return TileSpec(
            fire_px=fire_px, cloud_cover=float(capture.cloud_cover), glint=bool(capture.glint),
            smoke=bool(capture.smoke), haze=float(capture.haze), burn_scar=bool(capture.burn_scar),
            urban=bool(capture.urban), illumination=float(capture.illumination), seed=int(capture.seed),
        )

    def dataset(self, n: int, rng: np.random.Generator,
                p_fire: float = 0.5) -> tuple[np.ndarray, np.ndarray, list[TileSpec]]:
        specs = [self.random_spec(rng, p_fire) for _ in range(n)]
        x, y = self.generate_many(specs)
        return x, y, specs

    def generate_many(self, specs: list[TileSpec]) -> tuple[np.ndarray, np.ndarray]:
        n = self.size
        x = np.empty((len(specs), N_BANDS, n, n), dtype=np.float32)
        y = np.empty((len(specs), n, n), dtype=np.uint8)
        for i, spec in enumerate(specs):
            x[i], y[i] = self.generate(spec)
        return x, y

    def generate(self, spec: TileSpec) -> tuple[np.ndarray, np.ndarray]:
        """Return (tile[4, H, W] reflectance-equivalent float32, fire mask[H, W] uint8)."""
        rng = np.random.default_rng(spec.seed)
        n = self.size
        yy, xx = self._yy, self._xx

        # --- surface reflectance: soft mixture of land-cover classes -----------
        names = ("vegetation", "dry_grass", "soil")
        logits = np.stack([self.smooth_field(rng, rng.uniform(3.0, 10.0)) * rng.uniform(1.5, 3.5)
                           + rng.normal(0.0, 0.8) for _ in names])
        weights = np.exp(logits - logits.max(axis=0, keepdims=True))
        weights /= weights.sum(axis=0, keepdims=True)
        surface = np.einsum("kij,kb->bij", weights, np.stack([SPECTRA[nm] for nm in names])).astype(np.float32)
        surface *= (1.0 + 0.05 * rng.standard_normal((1, n, n))).astype(np.float32)

        water = np.zeros((n, n), dtype=np.float32)
        if spec.glint or rng.random() < 0.15:
            wf = self.smooth_field(rng, rng.uniform(5.0, 12.0))
            thr = np.quantile(wf, 1.0 - rng.uniform(0.25, 0.55))
            water = (1.0 / (1.0 + np.exp(-(wf - thr) * 8.0))).astype(np.float32)
            surface = surface * (1.0 - water) + water * SPECTRA["water"][:, None, None]

        industrial = np.zeros((n, n), dtype=np.float32)
        if spec.urban:
            uf = self.smooth_field(rng, rng.uniform(4.0, 8.0))
            urban = (uf > np.quantile(uf, 1.0 - rng.uniform(0.15, 0.35))) & (water < 0.5)
            surface[:, urban] = 0.3 * surface[:, urban] + 0.7 * SPECTRA["urban"][:, None]
            roofs = urban & (rng.random((n, n)) < rng.uniform(0.05, 0.2))
            surface[:, roofs] = SPECTRA["hot_roof"][:, None] * float(rng.uniform(0.9, 1.25))
            if rng.random() < 0.3 and urban.any():
                # industrial heat source (flare stack / kiln): fire-like SWIR, but NOT a wildfire
                cand = np.flatnonzero(urban.ravel())
                hy, hx = divmod(int(cand[rng.integers(cand.size)]), n)
                for _ in range(int(rng.integers(1, 4))):
                    py = int(np.clip(hy + rng.integers(-1, 2), 0, n - 1))
                    px = int(np.clip(hx + rng.integers(-1, 2), 0, n - 1))
                    industrial[py, px] = float(rng.uniform(0.2, 0.7))

        cy, cx = (float(v) for v in rng.uniform(10.0, n - 10.0, size=2))
        if spec.burn_scar:
            off = rng.normal(0.0, 6.0, size=2) if spec.has_fire else rng.uniform(-20.0, 20.0, size=2)
            sy, sx = cy + off[0], cx + off[1]
            radius = rng.uniform(6.0, 18.0)
            rr = np.sqrt((yy - sy) ** 2 + ((xx - sx) * rng.uniform(0.5, 1.5)) ** 2) / radius
            scar = np.clip(1.5 - rr + 0.3 * self.smooth_field(rng, 2.0), 0.0, 1.0)[None]
            surface = surface * (1.0 - scar) + scar * SPECTRA["burn_scar"][:, None, None]

        # --- active fire (emitted radiance, sub-pixel fire fraction) ------------
        fire_frac = np.zeros((n, n), dtype=np.float32)
        mask = np.zeros((n, n), dtype=bool)
        if spec.has_fire:
            fire_frac, mask = self._fire_shape(rng, spec.fire_px, cy, cx)
        intensity = float(rng.uniform(0.3, 1.6))  # smouldering .. intense flaming
        emission = (fire_frac * intensity + industrial * float(rng.uniform(0.5, 1.3)))[None] \
            * FIRE_EMISSION[:, None, None]

        # --- specular sun glint over water --------------------------------------
        reflective = surface
        if spec.glint and float(water.max()) > 0.5:
            idx = int(np.argmax(water + 0.1 * rng.random((n, n))))
            gy, gx = divmod(idx, n)
            r = rng.uniform(5.0, 14.0)
            core = np.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / (2.0 * r * r)) * float(rng.uniform(0.25, 0.8))
            core = np.clip(core * (1.0 + 0.3 * rng.standard_normal((n, n))), 0.0, None)
            sparkle = (rng.random((n, n)) < 0.03) * rng.uniform(0.2, 1.2, (n, n))
            # glint colour varies with sea state / aerosols; a SWIR-tilted glint mimics fire ratios
            tilt = GLINT_SHAPE * (1.0 + float(rng.uniform(0.0, 0.6)) * _spec(0.0, 0.1, 0.4, 0.8))
            reflective = reflective + ((core + sparkle) * water)[None] * tilt[:, None, None]

        transmission = np.ones((N_BANDS, n, n), dtype=np.float32)
        # --- smoke plume drifting downwind of the fire ---------------------------
        if spec.smoke:
            phi = rng.uniform(0.0, 2.0 * math.pi)
            along = (xx - cx) * math.cos(phi) + (yy - cy) * math.sin(phi)
            across = -(xx - cx) * math.sin(phi) + (yy - cy) * math.cos(phi)
            width = 3.0 + 0.25 * np.clip(along, 0.0, None)
            plume = (np.exp(-0.5 * (across / width) ** 2) * (along > -3.0)
                     * np.exp(-np.clip(along, 0.0, None) / rng.uniform(25.0, 60.0)))
            density = np.clip(plume * (0.7 + 0.3 * self._unit_field(rng, 3.0)) * rng.uniform(0.4, 0.85), 0.0, 0.9)
            ext = density[None] * SMOKE_EXTINCTION[:, None, None]
            reflective = reflective * (1.0 - ext) + ext * SPECTRA["smoke"][:, None, None]
            transmission *= 1.0 - ext

        # --- haze -----------------------------------------------------------------
        if spec.haze > 0.0:
            h = spec.haze * (0.7 + 0.3 * self._unit_field(rng, 10.0))
            ext = h[None] * HAZE_EXTINCTION[:, None, None]
            reflective = reflective * (1.0 - ext) + ext * 0.18
            transmission *= 1.0 - 0.5 * ext

        # --- clouds and cloud shadows ---------------------------------------------
        if spec.cloud_cover > 0.0:
            cf = self.smooth_field(rng, rng.uniform(4.0, 10.0))
            thr = np.quantile(cf, 1.0 - spec.cloud_cover)
            opacity = np.clip((cf - thr) * rng.uniform(1.5, 3.5), 0.0, 1.0) ** 0.7
            shift = rng.integers(2, 6, size=2)
            shadow = np.roll(opacity, (int(shift[0]), int(shift[1])), axis=(0, 1))
            reflective = reflective * (1.0 - 0.55 * shadow[None])
            texture = 0.9 + 0.2 * self._unit_field(rng, 2.0)
            reflective = reflective * (1.0 - opacity[None]) + (opacity * texture)[None] * SPECTRA["cloud"][:, None, None]
            transmission *= 1.0 - opacity[None]
            mask &= opacity < 0.6  # fire under thick cloud is not observable

        emission = emission * transmission
        mask &= emission[3] > 0.02
        illum = float(np.clip(spec.illumination, 0.005, 1.2))
        signal = reflective * illum + emission
        # CubeSat pushbroom artefacts: band-to-band misregistration between focal planes
        if rng.random() < 0.3:  # SWIR focal plane offset vs. VNIR
            shift = (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
            signal[2:] = np.roll(signal[2:], shift, axis=(1, 2))
            mask = np.roll(mask, shift, axis=(0, 1))
        if rng.random() < 0.15:  # residual SWIR-2 vs SWIR-1 offset: false SWIR ratios at bright edges
            shift = (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
            signal[3] = np.roll(signal[3], shift, axis=(0, 1))
            mask = np.roll(mask, shift, axis=(0, 1))
        noise = (rng.normal(0.0, 0.005, signal.shape)
                 + rng.normal(0.0, 1.0, signal.shape) * 0.006 * np.sqrt(np.clip(signal, 0.0, None)))
        if rng.random() < 0.3:  # detector column striping (non-uniformity)
            noise += rng.normal(0.0, 0.006, (N_BANDS, 1, n))
        out = np.clip(signal + noise, 0.0, SENSOR_MAX)
        out = np.round(out / SENSOR_MAX * QUANT_LEVELS) / QUANT_LEVELS * SENSOR_MAX
        return out.astype(np.float32), mask.astype(np.uint8)

    def _fire_shape(self, rng: np.random.Generator, n_px: int, cy: float,
                    cx: float) -> tuple[np.ndarray, np.ndarray]:
        n = self.size
        n_px = int(np.clip(n_px, 1, int(0.35 * n * n)))
        theta = rng.uniform(0.0, math.pi)
        major = max(1.5, math.sqrt(n_px / math.pi) * rng.uniform(1.0, 2.0))
        ratio = rng.uniform(0.3, 1.0)
        dy, dx = self._yy - cy, self._xx - cx
        u = dx * math.cos(theta) + dy * math.sin(theta)
        v = -dx * math.sin(theta) + dy * math.cos(theta)
        rr = np.sqrt((u / major) ** 2 + (v / (major * ratio)) ** 2)
        noise = self.smooth_field(rng, rng.uniform(1.0, 2.5))
        if n_px > 30 and rng.random() < 0.4:  # flaming front: partial ring
            ang = np.arctan2(v, u)
            score = -np.abs(rr - 1.0) * 3.0 + 0.9 * np.cos(ang - rng.uniform(-math.pi, math.pi)) + 0.35 * noise
        else:  # compact / elongated blob
            score = -rr * 2.0 + 0.35 * noise
        flat = score.ravel()
        top = np.argpartition(-flat, n_px - 1)[:n_px]
        mask = np.zeros(n * n, dtype=bool)
        mask[top] = True
        mask = mask.reshape(n, n)
        sel = score[mask]
        lo, hi = float(sel.min()), float(sel.max())
        frac = np.zeros((n, n), dtype=np.float32)
        frac[mask] = 0.25 + 0.75 * (score[mask] - lo) / (hi - lo + 1e-6)
        if rng.random() < 0.3:  # spot fires ahead of the front
            for _ in range(int(rng.integers(1, 4))):
                sy = int(np.clip(cy + rng.normal(0.0, major + 3.0), 0, n - 1))
                sx = int(np.clip(cx + rng.normal(0.0, major + 3.0), 0, n - 1))
                mask[sy, sx] = True
                frac[sy, sx] = max(float(frac[sy, sx]), float(rng.uniform(0.3, 0.8)))
        halo = _dilate(mask) & ~mask  # unlabeled mixed pixels: hard boundary cases
        frac[halo] = rng.uniform(0.02, 0.12, int(halo.sum()))
        return frac, mask


# ---------------------------------------------------------------------------
# Network input features (shared by every model backend)
# ---------------------------------------------------------------------------
def spectral_features(tiles: np.ndarray) -> np.ndarray:
    """(N, 4, H, W) reflectance -> (N, 8, H, W) physics-informed features.

    Channels: 4 log-scaled bands; NBR = (NIR - SWIR2) / (NIR + SWIR2); the normalised
    SWIR difference (SWIR2 - SWIR1) / (SWIR2 + SWIR1), strongly positive for flaming
    pixels and ~0 for glint, cloud and bright soil; and the log band ratios
    log(B12/B11) and log(B12/B8A) used by operational threshold algorithms (Murphy 2016).
    """
    t = np.asarray(tiles, dtype=np.float32)
    if t.ndim == 3:
        t = t[None]
    nir, s1, s2 = t[:, 1], t[:, 2], t[:, 3]
    eps = np.float32(0.02)
    nbr = (nir - s2) / (nir + s2 + eps)
    swir_diff = (s2 - s1) / (s2 + s1 + eps)
    r12_11 = np.clip(np.log((s2 + eps) / (s1 + eps)), -2.0, 2.0) / 2.0
    r12_8 = np.clip(np.log((s2 + eps) / (nir + eps)), -2.0, 2.0) / 2.0
    bands = np.log1p(t * 10.0) / np.float32(math.log1p(16.0))
    return np.concatenate([bands, nbr[:, None], swir_diff[:, None], r12_11[:, None], r12_8[:, None]],
                          axis=1).astype(np.float32)


def false_color(tile: np.ndarray) -> np.ndarray:
    """SWIR-2 / NIR / Red composite as uint8 (H, W, 3): fires glow red-orange."""
    rgb = np.stack([tile[3], tile[1], tile[0]], axis=-1)
    rgb = np.clip(rgb / 0.6, 0.0, 1.0) ** 0.8
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Real-data hooks (GeoTIFF / NumPy tiles, NASA FIRMS hotspot CSV)
# ---------------------------------------------------------------------------
def load_tile_file(path: str | Path, reflectance_scale: float | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a real tile with bands ordered Red, NIR, SWIR-1, SWIR-2.

    Supports ``.npy`` (bands-first or bands-last array), ``.npz`` (keys ``tile`` and
    optional ``mask``) and GeoTIFF (``.tif``/``.tiff``, requires rasterio).
    Integer digital numbers (e.g. Sentinel-2 L1C/L2A) are scaled by 1e-4.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    mask: np.ndarray | None = None
    if suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".npz":
        with np.load(p) as data:
            arr = data["tile"] if "tile" in data.files else data[data.files[0]]
            mask = np.asarray(data["mask"]).astype(np.uint8) if "mask" in data.files else None
    elif suffix in (".tif", ".tiff"):
        if not deps.RASTERIO_AVAILABLE:
            raise ImportError("reading GeoTIFF tiles requires rasterio (pip install rasterio)")
        import rasterio

        with rasterio.open(p) as src:
            arr = src.read()
    else:
        raise ValueError(f"unsupported tile format: {p.name}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{p.name}: expected a 3-D array, got shape {arr.shape}")
    if arr.shape[0] < N_BANDS <= arr.shape[-1]:
        arr = np.moveaxis(arr, -1, 0)
    if arr.shape[0] < N_BANDS:
        raise ValueError(f"{p.name}: need >= 4 bands (Red, NIR, SWIR-1, SWIR-2)")
    arr = arr[:N_BANDS]
    scale = reflectance_scale if reflectance_scale is not None else (1e-4 if float(arr.max()) > 4.0 else 1.0)
    return np.clip(arr * scale, 0.0, SENSOR_MAX).astype(np.float32), mask


def tile_patches(tile: np.ndarray, mask: np.ndarray | None,
                 size: int) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """Split a (4, H, W) tile into zero-padded size x size patches."""
    _, h, w = tile.shape
    ph, pw = -h % size, -w % size
    t = np.pad(tile, ((0, 0), (0, ph), (0, pw)))
    m = np.pad(mask, ((0, ph), (0, pw))) if mask is not None else None
    out: list[tuple[np.ndarray, np.ndarray | None]] = []
    for y in range(0, t.shape[1], size):
        for x in range(0, t.shape[2], size):
            out.append((t[:, y:y + size, x:x + size], None if m is None else m[y:y + size, x:x + size]))
    return out


def iter_tile_dir(directory: str | Path) -> Iterator[tuple[str, np.ndarray, np.ndarray | None]]:
    for p in sorted(Path(directory).iterdir()):
        if p.suffix.lower() in (".npy", ".npz", ".tif", ".tiff"):
            tile, mask = load_tile_file(p)
            yield p.name, tile, mask


@dataclass(frozen=True)
class FirmsHotspot:
    lat_deg: float
    lon_deg: float
    frp_mw: float
    confidence: str
    acquired: str


def load_firms_csv(path: str | Path) -> list[FirmsHotspot]:
    """Parse a NASA FIRMS active-fire CSV (MODIS or VIIRS columns)."""
    out: list[FirmsHotspot] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                frp = float(row.get("frp") or 0.0)
            except ValueError:
                frp = 0.0
            out.append(FirmsHotspot(lat, lon, frp, str(row.get("confidence", "")),
                                    f"{row.get('acq_date', '')} {row.get('acq_time', '')}".strip()))
    return out


def firms_to_watch_zones(hotspots: list[FirmsHotspot], cluster_km: float = 150.0, max_zones: int = 12,
                         window_days: float = 1.0) -> tuple[WatchZone, ...]:
    """Greedy FRP-weighted clustering of FIRMS hotspots into Stage-1 watch zones.

    WUI risk / population are unknown from FIRMS alone and default to mid values;
    join with a population raster (e.g. GHSL/WorldPop) for operational use.
    """
    from .geo import great_circle_km

    clusters: list[list[float]] = []  # lat, lon, count, frp
    for h in sorted(hotspots, key=lambda q: -q.frp_mw):
        best, best_d = -1, cluster_km
        for i, c in enumerate(clusters):
            d = float(great_circle_km(h.lat_deg, h.lon_deg, c[0], c[1]))
            if d <= best_d:
                best, best_d = i, d
        if best < 0:
            clusters.append([h.lat_deg, h.lon_deg, 1.0, h.frp_mw])
        else:
            clusters[best][2] += 1.0
            clusters[best][3] += h.frp_mw
    clusters.sort(key=lambda c: -c[2])
    zones = []
    for i, (lat, lon, count, _frp) in enumerate(clusters[:max_zones]):
        zones.append(WatchZone(
            f"FIRMS cluster {i + 1} ({lat:.1f},{lon:.1f})", lat, lon, max(80.0, cluster_km), 0.5, 50.0,
            min(3.0, 0.3 + count / (window_days * 20.0)), 0.3, "FIRMS-derived",
        ))
    return tuple(zones)
