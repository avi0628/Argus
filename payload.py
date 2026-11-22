"""Downlink payload sizing: raw scene vs. lossless-compressed scene vs. metadata-assisted alert."""
from __future__ import annotations

from typing import Any

import numpy as np

from .config import PayloadConfig


class PayloadModel:
    """Byte budgets for the products a satellite can queue for downlink.

    * raw scene         - scene_px^2 x bands x bits
    * compressed scene  - CCSDS-123-class lossless compression of the raw scene
    * alert (metadata-assisted) - JSON metadata + run-length-encoded fire mask
      + full-resolution context chip around the detection + low-res quicklook
    """

    def __init__(self, cfg: PayloadConfig, tile_size: int, gsd_m: float) -> None:
        self.cfg = cfg
        self.tile_size = tile_size
        self.gsd_m = gsd_m

    @property
    def raw_scene_bytes(self) -> int:
        c = self.cfg
        return int(c.scene_px**2 * c.bands * c.bits_per_sample / 8)

    @property
    def compressed_scene_bytes(self) -> int:
        return int(self.raw_scene_bytes / self.cfg.lossless_compression_ratio)

    @property
    def tiles_per_scene(self) -> int:
        return max(1, (self.cfg.scene_px // self.tile_size) ** 2)

    @property
    def scene_width_km(self) -> float:
        return self.cfg.scene_px * self.gsd_m / 1000.0

    @staticmethod
    def rle_bytes(mask: np.ndarray) -> int:
        """Run-length encoding of a binary mask: (uint16 start, uint16 length) per run + header."""
        flat = np.asarray(mask, dtype=np.int8).ravel()
        runs = int(np.count_nonzero(np.diff(np.concatenate(([0], flat, [0]))) == 1))
        return 8 + 4 * runs

    def context_chip_bytes(self) -> int:
        c = self.cfg
        return int(c.roi_context_px**2 * c.bands * c.bits_per_sample / 8 / c.lossless_compression_ratio)

    def quicklook_bytes(self) -> int:
        return int(self.cfg.quicklook_px**2 * self.cfg.quicklook_bytes_per_px)

    def alert_bytes(self, pred_mask: np.ndarray) -> int:
        return int(self.cfg.metadata_bytes + self.rle_bytes(pred_mask) + self.context_chip_bytes()
                   + self.quicklook_bytes())

    def describe(self, example_mask: np.ndarray | None = None) -> dict[str, Any]:
        mask = example_mask if example_mask is not None else np.zeros((self.tile_size, self.tile_size), np.uint8)
        alert = self.alert_bytes(mask)
        return {
            "scene_px": self.cfg.scene_px,
            "scene_width_km": self.scene_width_km,
            "raw_scene_mb": self.raw_scene_bytes / 1e6,
            "compressed_scene_mb": self.compressed_scene_bytes / 1e6,
            "alert_kb": alert / 1e3,
            "alert_breakdown_kb": {
                "metadata": self.cfg.metadata_bytes / 1e3,
                "rle_mask": self.rle_bytes(mask) / 1e3,
                "context_chip": self.context_chip_bytes() / 1e3,
                "quicklook": self.quicklook_bytes() / 1e3,
            },
            "reduction_vs_raw": self.raw_scene_bytes / alert,
            "reduction_vs_compressed": self.compressed_scene_bytes / alert,
            "tiles_per_scene": self.tiles_per_scene,
        }
