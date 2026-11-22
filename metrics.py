"""Segmentation / detection metrics shared by all model backends."""
from __future__ import annotations

import math

import numpy as np


def tile_confidence(prob: np.ndarray, topk: int = 8) -> np.ndarray:
    """Per-tile detection confidence = mean of the top-k pixel fire probabilities."""
    flat = np.asarray(prob, dtype=np.float32).reshape(len(prob), -1)
    k = int(max(1, min(topk, flat.shape[1])))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)


def pixel_iou(prob: np.ndarray, masks: np.ndarray, threshold: float = 0.5,
              valid: np.ndarray | None = None) -> float:
    pred = np.asarray(prob) >= threshold
    gt = np.asarray(masks) > 0
    ok = np.ones_like(gt) if valid is None else np.asarray(valid, dtype=bool)
    tp = float(np.sum(pred & gt & ok))
    union = float(np.sum((pred | gt) & ok))
    return tp / union if union > 0 else 1.0


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Tile-level ECE: |accuracy - confidence| averaged over confidence bins."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return float(ece)


def firms_agreement(prob: np.ndarray, meta: list[dict], threshold: float = 0.5, pre_h: float = 2.5,
                    post_h: float = 1.0, radius_px: float = 20.0, confirm_dt_h: float = 12.0) -> dict[str, float]:
    """Independent check against NASA FIRMS hotspots (VIIRS 375 m / MODIS 1 km).

    * hotspot recall: hotspots already burning at the Sentinel-2 pass (observed between ``pre_h``
      before and ``post_h`` after it) inside the chip with a predicted fire pixel within ``radius_px``
      (20 px = 400 m). Later hotspots are excluded: those fires may have ignited after the pass.
    * FIRMS-confirmed detections: chips the model flags (>= 3 px) that have a FIRMS hotspot within
      ``radius_px`` of a predicted fire pixel at any time within +-``confirm_dt_h``
    * clean-chip false-alarm rate: chips with no hotspot within +-24 h whose prediction contains fire
    """
    pred = np.asarray(prob) >= threshold
    hit = total = 0
    flagged = confirmed = 0
    clean = clean_fa = 0
    for i, m in enumerate(meta):
        hs = m.get("hotspots", [])
        inside = [h for h in hs if 0 <= h["row"] < pred.shape[1] and 0 <= h["col"] < pred.shape[2]]
        rr, cc = np.nonzero(pred[i])

        def near(h: dict) -> bool:
            return bool(rr.size) and float(np.min((rr - h["row"]) ** 2 + (cc - h["col"]) ** 2)) <= radius_px**2

        for h in inside:
            if -pre_h <= h["dt_h"] <= post_h:
                total += 1
                hit += int(near(h))
        if rr.size >= 3:
            flagged += 1
            confirmed += int(any(abs(h["dt_h"]) <= confirm_dt_h and near(h) for h in hs))
        if not any(abs(h["dt_h"]) <= 24.0 for h in hs):
            clean += 1
            clean_fa += int(rr.size >= 3)
    return {"hotspots": float(total), "hotspot_recall": hit / total if total else math.nan,
            "flagged_chips": float(flagged), "firms_confirmed_rate": confirmed / flagged if flagged else math.nan,
            "clean_chips": float(clean), "clean_false_alarm_rate": clean_fa / clean if clean else math.nan}


def late_ignition_stats(labels: np.ndarray, meta: list[dict], radius_px: float = 20.0) -> dict[str, float]:
    """Hotspots seen by afternoon VIIRS (+1.5..+4 h after the morning Sentinel-2 overpass) where the
    Sentinel-2 image shows no fire within 400 m: fires that ignited between the two overpasses."""
    late = missing = 0
    for i, m in enumerate(meta):
        rr, cc = np.nonzero(labels[i])
        for h in m.get("hotspots", []):
            if 1.5 <= h["dt_h"] <= 4.0 and 0 <= h["row"] < labels.shape[1] and 0 <= h["col"] < labels.shape[2]:
                late += 1
                if not rr.size or float(np.min((rr - h["row"]) ** 2 + (cc - h["col"]) ** 2)) > radius_px**2:
                    missing += 1
    return {"afternoon_hotspots": float(late), "not_burning_at_morning_pass": float(missing),
            "late_ignition_share": missing / late if late else math.nan}


def detect(prob: np.ndarray, threshold: float, min_pixels: int, topk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile-level decision: (detected[N], confidence[N], predicted fire pixels[N])."""
    conf = tile_confidence(prob, topk)
    npx = (np.asarray(prob) >= threshold).reshape(len(prob), -1).sum(axis=1)
    return (conf >= threshold) & (npx >= min_pixels), conf, npx


def segmentation_metrics(prob: np.ndarray, masks: np.ndarray, threshold: float = 0.5,
                         min_pixels: int = 3, topk: int = 8, valid: np.ndarray | None = None) -> dict[str, float]:
    pred = np.asarray(prob) >= threshold
    gt = np.asarray(masks) > 0
    ok = np.ones_like(gt) if valid is None else np.asarray(valid, dtype=bool)
    tp = float(np.sum(pred & gt & ok))
    fp = float(np.sum(pred & ~gt & ok))
    fn = float(np.sum(~pred & gt & ok))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 1.0
    detected, _conf, npx = detect(prob, threshold, min_pixels, topk)
    has_fire = gt.reshape(len(gt), -1).sum(axis=1) > 0
    gt_px = gt.reshape(len(gt), -1).sum(axis=1)
    both = detected & has_fire
    area_err = float(np.mean(np.abs(npx[both] - gt_px[both]) / gt_px[both])) if both.any() else math.nan
    return {
        "pixel_iou": iou,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": f1,
        "tile_detection_rate": float(detected[has_fire].mean()) if has_fire.any() else math.nan,
        "tile_false_alarm_rate": float(detected[~has_fire].mean()) if (~has_fire).any() else math.nan,
        "tile_accuracy": float(np.mean(detected == has_fire)),
        "area_rel_error": area_err,
        "n_tiles": float(len(gt)),
        "n_fire_tiles": float(has_fire.sum()),
    }
