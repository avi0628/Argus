"""Stage 2 - ArgusNet multispectral fire segmentation.

Training data = procedural synthetic tiles (exact labels, rare distractors on demand)
+ real Sentinel-2 L2A chips cut around live NASA FIRMS fires (physics pseudo-labels with
an ignore band). Evaluation compares ArgusNet against the classical physics threshold
algorithm on (a) synthetic ground truth and (b) real imagery, where NASA FIRMS hotspots
provide an independent check. Confidence is calibrated by temperature scaling.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import deps
from .config import SimulationConfig
from .fire_index import PhysicsBaseline
from .metrics import (detect, expected_calibration_error, firms_agreement, late_ignition_stats, segmentation_metrics,
                      tile_confidence)
from .models_numpy import QuantizedSpectralMLP, SpectralMLP
from .payload import PayloadModel
from .sentinel2 import RealDataset
from .tiles import N_FEATURES, MultispectralTileGenerator, TileSpec, iter_tile_dir, spectral_features, tile_patches

Logger = Callable[[str], None]
REAL_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "real_s2_chips"


# ---------------------------------------------------------------------------
# Inference facade
# ---------------------------------------------------------------------------
class Segmenter(ABC):
    """Uniform inference API over torch / quantized torch / NumPy / physics models."""

    name: str = "segmenter"
    backend: str = "none"

    @abstractmethod
    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """(N, 4, H, W) reflectance tiles -> (N, H, W) fire probability."""


class TorchSegmenter(Segmenter):
    backend = "torch"

    def __init__(self, module: Any, name: str, temperature: float = 1.0) -> None:
        self.module = module
        self.name = name
        self.temperature = temperature

    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        from .models_torch import predict_proba_torch

        return predict_proba_torch(self.module, spectral_features(tiles), batch_size, self.temperature)


class NumpySegmenter(Segmenter):
    backend = "numpy"

    def __init__(self, model: SpectralMLP | QuantizedSpectralMLP, name: str) -> None:
        self.model = model
        self.name = name

    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        return self.model.predict_proba(tiles, batch_size)


class BaselineSegmenter(Segmenter):
    backend = "physics"

    def __init__(self) -> None:
        self.impl = PhysicsBaseline()
        self.name = self.impl.name

    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        return self.impl.predict_proba(tiles)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    s_train: list[TileSpec]
    x_val: np.ndarray
    y_val: np.ndarray
    s_val: list[TileSpec]
    x_test: np.ndarray
    y_test: np.ndarray
    s_test: list[TileSpec]


@dataclass
class RealSplit:
    tiles: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    meta: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.meta)


@dataclass
class SegmentationResult:
    backend: str
    model_name: str
    segmenter: Segmenter
    raw_model: Any
    history: list[dict[str, float]]
    test_metrics: dict[str, float]
    distractor_breakdown: list[dict[str, Any]]
    data: DatasetSplits
    train_time_s: float
    n_params: int
    payload: dict[str, Any]
    samples: list[dict[str, Any]] = field(default_factory=list)
    real_tiles: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    baseline_breakdown: list[dict[str, Any]] = field(default_factory=list)
    real: dict[str, Any] = field(default_factory=dict)  # real Sentinel-2 evaluation (model vs baseline)
    real_test: RealSplit | None = None
    real_samples: list[dict[str, Any]] = field(default_factory=list)
    calibration: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0


DISTRACTORS: tuple[tuple[str, Callable[[TileSpec], bool]], ...] = (
    ("clear sky", lambda s: not (s.cloud_cover > 0 or s.glint or s.smoke or s.haze > 0 or s.urban or s.is_night)),
    ("cloud / shadow", lambda s: s.cloud_cover > 0.05),
    ("sun glint", lambda s: s.glint),
    ("smoke plume", lambda s: s.smoke),
    ("haze", lambda s: s.haze > 0),
    ("burn scar", lambda s: s.burn_scar),
    ("urban / hot roofs / industry", lambda s: s.urban),
    ("night (emission only)", lambda s: s.is_night),
    ("small fire (< 25 px)", lambda s: 0 < s.fire_px < 25),
)


def distractor_breakdown(prob: np.ndarray, masks: np.ndarray, specs: list[TileSpec], threshold: float,
                         min_px: int, topk: int) -> list[dict[str, Any]]:
    detected, _c, _n = detect(prob, threshold, min_px, topk)
    has_fire = masks.reshape(len(masks), -1).sum(axis=1) > 0
    rows = []
    for name, pred in DISTRACTORS:
        sel = np.array([pred(s) for s in specs], dtype=bool)
        fire_sel, nofire_sel = sel & has_fire, sel & ~has_fire
        rows.append({
            "condition": name,
            "fire_tiles": int(fire_sel.sum()),
            "detection_rate": float(detected[fire_sel].mean()) if fire_sel.any() else float("nan"),
            "nofire_tiles": int(nofire_sel.sum()),
            "false_alarm_rate": float(detected[nofire_sel].mean()) if nofire_sel.any() else float("nan"),
        })
    return rows


def make_datasets(cfg: SimulationConfig, log: Logger) -> DatasetSplits:
    mc = cfg.model
    gen = MultispectralTileGenerator(mc.tile_size, mc.gsd_m)
    rng = np.random.default_rng(cfg.seed + 101)
    t0 = time.perf_counter()
    xtr, ytr, s_tr = gen.dataset(mc.n_train, rng, mc.p_fire)
    xva, yva, s_va = gen.dataset(mc.n_val, rng, mc.p_fire)
    xte, yte, s_te = gen.dataset(mc.n_test, rng, mc.p_fire)
    log(f"  synthetic: {mc.n_train}/{mc.n_val}/{mc.n_test} train/val/test tiles "
        f"({mc.tile_size}x{mc.tile_size} px, {mc.gsd_m:.0f} m) in {time.perf_counter() - t0:.1f}s")
    return DatasetSplits(xtr, ytr, s_tr, xva, yva, s_va, xte, yte, s_te)


def load_real_splits(cfg: SimulationConfig, log: Logger) -> dict[str, RealSplit] | None:
    if not cfg.model.use_real_data:
        return None
    ds = RealDataset.load(Path(cfg.model.real_dataset_path) if cfg.model.real_dataset_path else REAL_DATASET_PATH)
    if ds is None or len(ds) < 6:
        log("  real Sentinel-2 dataset not found - run `python argus_cli.py build-dataset` (synthetic-only training)")
        return None
    sp = ds.split(cfg.seed)
    out = {k: RealSplit(ds.tiles[i], ds.labels[i], ~ds.ignore[i], [ds.meta[j] for j in i]) for k, i in sp.items()}
    n_fire = sum(m["fire_px"] > 0 for m in ds.meta)
    log(f"  real Sentinel-2 L2A: {len(ds)} chips from {len({m['scene'] for m in ds.meta})} scenes "
        f"({n_fire} with active fire) -> {len(out['train'])}/{len(out['val'])}/{len(out['test'])} "
        f"train/val/test, split by fire event")
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_segmentation(cfg: SimulationConfig, log: Logger = print) -> SegmentationResult:
    mc = cfg.model
    data = make_datasets(cfg, log)
    real = load_real_splits(cfg, log)
    use_torch = deps.TORCH_AVAILABLE and mc.use_torch
    note = ""
    if mc.use_torch and not deps.TORCH_AVAILABLE:
        note = f"torch unavailable ({deps.TORCH_ERROR}); using NumPy fallback"

    # combined train / val sets (real chips oversampled; ignore-band pixels get zero weight)
    x_tr, y_tr = data.x_train, data.y_train
    w_tr = np.ones(y_tr.shape, dtype=np.float32)
    x_va, y_va = data.x_val, data.y_val
    v_va = np.ones(y_va.shape, dtype=bool)
    if real is not None:
        k = max(1, mc.real_oversample)
        rt = real["train"]
        x_tr = np.concatenate([x_tr] + [rt.tiles] * k)
        y_tr = np.concatenate([y_tr] + [rt.labels] * k)
        w_tr = np.concatenate([w_tr] + [rt.valid.astype(np.float32) * mc.real_weight] * k)
        x_va = np.concatenate([x_va, real["val"].tiles])
        y_va = np.concatenate([y_va, real["val"].labels])
        v_va = np.concatenate([v_va, real["val"].valid])

    t0 = time.perf_counter()
    temperature = 1.0
    if use_torch:
        import torch

        from .models_torch import FireDSUNet, count_params, fit_temperature, predict_logits_torch, train_fire_unet

        torch.manual_seed(cfg.seed)  # seed BEFORE construction: weight init must be reproducible
        model = FireDSUNet(N_FEATURES, mc.base_channels)
        log(f"  training ArgusNet (DS-U-Net + squeeze-excitation, {count_params(model)[0]:,} params) on "
            f"{len(x_tr)} tiles with focal + soft-Dice loss")
        history = train_fire_unet(
            model, spectral_features(x_tr), y_tr, spectral_features(x_va), y_va,
            epochs=mc.epochs, batch_size=mc.batch_size, lr=mc.learning_rate, alpha=mc.focal_alpha,
            gamma=mc.focal_gamma, dice_weight=mc.dice_weight, seed=cfg.seed, threads=mc.train_threads,
            w_train=w_tr, val_valid=v_va, log=log,
        )
        if mc.calibrate:
            temperature = fit_temperature(predict_logits_torch(model, spectral_features(x_va)), y_va, v_va,
                                          mc.confidence_topk)
        segmenter: Segmenter = TorchSegmenter(model, "ArgusNet fp32", temperature)
        backend, model_name, n_params, raw = "torch", "ArgusNet (DS-U-Net + SE attention)", count_params(model)[0], model
    else:
        mlp = SpectralMLP(hidden=24, seed=cfg.seed)
        log(f"  training NumPy SpectralMLP ({mlp.num_params()} params) with weighted BCE + soft-Dice loss")
        history = mlp.fit(x_tr, y_tr, x_va, y_va, epochs=max(4, mc.epochs // 2 + 2), seed=cfg.seed, log=log,
                          ignore=w_tr <= 0)
        segmenter = NumpySegmenter(mlp, "SpectralMLP fp32")
        backend, model_name, n_params, raw = "numpy", "SpectralMLP (per-pixel, NumPy)", mlp.num_params(), mlp
    train_time = time.perf_counter() - t0

    thr, mpx, topk = mc.detection_threshold, mc.min_fire_pixels, mc.confidence_topk
    prob = segmenter.predict_proba(data.x_test)
    metrics = segmentation_metrics(prob, data.y_test, thr, mpx, topk)
    breakdown = distractor_breakdown(prob, data.y_test, data.s_test, thr, mpx, topk)
    base = BaselineSegmenter()
    base_prob = base.predict_proba(data.x_test)
    base_metrics = segmentation_metrics(base_prob, data.y_test, thr, mpx, topk)
    base_breakdown = distractor_breakdown(base_prob, data.y_test, data.s_test, thr, mpx, topk)

    calibration: dict[str, float] = {"temperature": temperature}
    has_fire = data.y_test.reshape(len(data.y_test), -1).sum(axis=1) > 0
    if use_torch:
        from .models_torch import predict_proba_torch

        raw_prob = predict_proba_torch(raw, spectral_features(data.x_test), 64, 1.0)
        calibration["tile_ece_before"] = expected_calibration_error(tile_confidence(raw_prob, topk), has_fire)
    calibration["tile_ece_after"] = expected_calibration_error(tile_confidence(prob, topk), has_fire)

    payload_model = PayloadModel(cfg.payload, mc.tile_size, mc.gsd_m)
    samples = _pick_samples(prob, data, thr, topk, payload_model, mc.gsd_m, mpx)
    example_mask = samples[0]["pred_mask"] if samples else None
    result = SegmentationResult(
        backend=backend, model_name=model_name, segmenter=segmenter, raw_model=raw, history=history,
        test_metrics=metrics, distractor_breakdown=breakdown, data=data, train_time_s=train_time,
        n_params=int(n_params), payload=payload_model.describe(example_mask), samples=samples, note=note,
        baseline_metrics=base_metrics, baseline_breakdown=base_breakdown, calibration=calibration,
        temperature=temperature,
    )
    if real is not None and len(real["test"]):
        result.real_test = real["test"]
        result.real, result.real_samples = evaluate_real(segmenter, base, real["test"], cfg)
    if mc.tiles_dir:
        result.real_tiles = score_real_tiles(segmenter, mc.tiles_dir, mc.tile_size, thr, mpx, topk, log)
    return result


def evaluate_real(model: Segmenter, base: Segmenter, rt: RealSplit,
                  cfg: SimulationConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Model vs physics baseline on held-out real Sentinel-2 chips (unseen fire events)."""
    mc = cfg.model
    pm, pb = model.predict_proba(rt.tiles), base.predict_proba(rt.tiles)
    out: dict[str, Any] = {"chips": len(rt), "fire_chips": int(sum(m["fire_px"] > 0 for m in rt.meta)),
                           "scenes": len({m["scene"] for m in rt.meta}),
                           "late_ignitions": late_ignition_stats(rt.labels, rt.meta)}
    for name, p in (("model", pm), ("baseline", pb)):
        seg = segmentation_metrics(p, rt.labels, mc.detection_threshold, mc.min_fire_pixels, mc.confidence_topk,
                                   valid=rt.valid)
        agr = firms_agreement(p, rt.meta, mc.detection_threshold)
        out[name] = {"pseudo_iou": seg["pixel_iou"], "pseudo_f1": seg["pixel_f1"], **agr}
    samples = []
    order = sorted(range(len(rt)), key=lambda i: -(rt.meta[i]["fire_px"] + 5 * len(rt.meta[i]["hotspots"])))
    for i in order[:8]:
        m = rt.meta[i]
        samples.append({"tile": rt.tiles[i], "label": rt.labels[i], "prob": pm[i], "baseline": pb[i],
                        "hotspots": [h for h in m["hotspots"] if abs(h["dt_h"]) <= 24.0],
                        "title": f"{m['scene']} | {m.get('nearest_city', '-')} | {m['datetime'][:16]}"})
    return out, samples


def _pick_samples(prob: np.ndarray, data: DatasetSplits, thr: float, topk: int, payload: PayloadModel,
                  gsd_m: float, min_px: int) -> list[dict[str, Any]]:
    """A handful of illustrative test tiles: fires plus the main distractor types."""
    detected, conf, npx = detect(prob, thr, min_px, topk)
    picks: list[int] = []
    wanted: list[Callable[[TileSpec], bool]] = [
        lambda s: s.has_fire and s.smoke, lambda s: s.has_fire and s.is_night,
        lambda s: s.has_fire and s.fire_px < 25, lambda s: (not s.has_fire) and s.glint,
        lambda s: (not s.has_fire) and s.cloud_cover > 0.3, lambda s: (not s.has_fire) and s.urban,
    ]
    for cond in wanted:
        for i, s in enumerate(data.s_test):
            if i not in picks and cond(s):
                picks.append(i)
                break
    rows = []
    px_ha = gsd_m**2 / 1e4
    for i in picks:
        pred = prob[i] >= thr
        rows.append({
            "index": i, "tags": ", ".join(data.s_test[i].tags()), "tile": data.x_test[i],
            "mask": data.y_test[i], "prob": prob[i], "pred_mask": pred.astype(np.uint8),
            "gt_px": int(data.y_test[i].sum()), "pred_px": int(npx[i]), "confidence": float(conf[i]),
            "detected": bool(detected[i]), "area_ha": float(npx[i] * px_ha),
            "product": "ALERT" if detected[i] else "SCENE",
            "payload_kb": (payload.alert_bytes(pred) if detected[i] else payload.compressed_scene_bytes) / 1e3,
        })
    return rows


def score_real_tiles(segmenter: Segmenter, directory: str, tile_size: int, thr: float, min_px: int,
                     topk: int, log: Logger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for name, tile, mask in iter_tile_dir(directory):
            patches = tile_patches(tile, mask, tile_size)
            x = np.stack([p[0] for p in patches])
            prob = segmenter.predict_proba(x)
            detected, conf, npx = detect(prob, thr, min_px, topk)
            row: dict[str, Any] = {"file": name, "patches": len(patches), "detections": int(detected.sum()),
                                   "max_confidence": float(conf.max()), "fire_px": int(npx.sum())}
            if mask is not None:
                y = np.stack([p[1] for p in patches])
                row["pixel_iou"] = segmentation_metrics(prob, y, thr, min_px, topk)["pixel_iou"]
            rows.append(row)
        log(f"  scored {len(rows)} real tile file(s) from {directory}")
    except Exception as exc:
        log(f"  real-tile scoring skipped: {type(exc).__name__}: {exc}")
    return rows
