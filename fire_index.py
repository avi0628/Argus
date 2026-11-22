"""Physics-based active-fire test for Sentinel-2 / Landsat-class SWIR bands.

Adapted from Murphy et al. (2016, Remote Sensing of Environment 175, "HOTMAP"/GOLI-style
Landsat-8 active-fire algorithm) using the Sentinel-2 band mapping
B12 <-> L8 B7 (SWIR-2), B11 <-> B6 (SWIR-1), B8A <-> B5 (narrow NIR):

* unambiguous fire:  B12/B11 >= 1.4  and  B12/B8A >= 1.4  and  B12 >= 0.15
* potential fire:    relaxed ratios, or very bright SWIR-1 (B11/B8A >= 2 and B11 >= 0.5),
                     accepted only when 8-connected (2 px) to an unambiguous fire pixel

Argus uses it two ways: (1) as the classical baseline the neural model must beat, and
(2) to pseudo-label real Sentinel-2 chips for domain adaptation (with an ignore band
around uncertain pixels).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-4


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape[-2:]
    for _ in range(iterations):
        p = np.pad(out, [(0, 0)] * (out.ndim - 2) + [(1, 1), (1, 1)])
        acc = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                acc |= p[..., dy:dy + h, dx:dx + w]
        out = acc
    return out


def physics_fire_masks(tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(fire, potential) boolean masks for a (4, H, W) or (N, 4, H, W) reflectance array
    ordered Red, NIR(B8A), SWIR-1(B11), SWIR-2(B12)."""
    t = np.asarray(tile, dtype=np.float32)
    nir, s1, s2 = t[..., 1, :, :], t[..., 2, :, :], t[..., 3, :, :]
    r12_11 = s2 / (s1 + EPS)
    r12_8 = s2 / (nir + EPS)
    unambiguous = (r12_11 >= 1.4) & (r12_8 >= 1.4) & (s2 >= 0.15)
    potential = (((r12_11 >= 1.15) & (r12_8 >= 1.15) & (s2 >= 0.10))
                 | ((s1 / (nir + EPS) >= 2.0) & (s1 >= 0.5))
                 | ((s2 >= 1.0) & (r12_11 >= 1.0)))
    fire = unambiguous | (potential & _dilate(unambiguous, 2))
    return fire, potential & ~fire


def physics_fire_proba(tiles: np.ndarray) -> np.ndarray:
    """Baseline 'probability' map (hard 0/1) for (N, 4, H, W) tiles."""
    fire, potential = physics_fire_masks(tiles)
    return np.where(fire, 1.0, np.where(potential, 0.3, 0.0)).astype(np.float32)


def pseudo_label(tile: np.ndarray, invalid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(label uint8, ignore bool) for a real chip: ignore 2-px boundary band, potential pixels, nodata."""
    fire, potential = physics_fire_masks(tile)
    ignore = (_dilate(fire, 2) & ~fire) | potential
    if invalid is not None:
        ignore |= invalid
    return fire.astype(np.uint8), ignore


def cloud_fraction(tile: np.ndarray) -> float:
    """Onboard spectral cloud test (bright, spectrally flat in VNIR, SWIR-1 darker than NIR)."""
    red, nir, s1 = tile[0], tile[1], tile[2]
    bright = (red > 0.25) & (nir > 0.25)
    flat = np.abs(red - nir) < 0.3 * np.maximum(nir, EPS)
    cloud = bright & flat & (s1 < 0.9 * nir)
    return float(cloud.mean())


class PhysicsBaseline:
    """Segmenter-compatible wrapper (``predict_proba``) around the threshold algorithm."""

    name = "Physics threshold (Murphy-2016 style)"
    backend = "numpy"

    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        return physics_fire_proba(tiles)
