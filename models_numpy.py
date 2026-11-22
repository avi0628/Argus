"""NumPy-only fallback fire segmenter (used when PyTorch is unavailable).

A per-pixel spectral-contextual MLP: 6 spectral features + 3x3 neighbourhood means,
local max of the SWIR difference and SWIR-2 contrast against a 9x9 background
(14 features) -> 24 ReLU units -> sigmoid. Trained with Adam on weighted BCE +
soft Dice. Supports magnitude pruning, INT8 weight quantization and ONNX export
via ``onnx.helper`` when the ``onnx`` package is installed.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import numpy as np

from . import deps
from .metrics import pixel_iou
from .tiles import N_FEATURES, spectral_features

N_PIXEL_FEATURES = 2 * N_FEATURES + 2  # features + 3x3 means + local max SWIR diff + SWIR-2 contrast


def box_mean(x: np.ndarray, radius: int) -> np.ndarray:
    """(2r+1)^2 mean filter over the last two axes (integral image, edge padding)."""
    if radius <= 0:
        return np.asarray(x, dtype=np.float32).copy()
    k = 2 * radius + 1
    lead = [(0, 0)] * (x.ndim - 2)
    p = np.pad(np.asarray(x, dtype=np.float64), lead + [(radius, radius), (radius, radius)], mode="edge")
    s = np.pad(p.cumsum(-1).cumsum(-2), lead + [(1, 0), (1, 0)])
    h, w = x.shape[-2:]
    total = s[..., k:k + h, k:k + w] - s[..., 0:h, k:k + w] - s[..., k:k + h, 0:w] + s[..., 0:h, 0:w]
    return (total / (k * k)).astype(np.float32)


def max_filter3(x: np.ndarray) -> np.ndarray:
    lead = [(0, 0)] * (x.ndim - 2)
    p = np.pad(x, lead + [(1, 1), (1, 1)], mode="edge")
    h, w = x.shape[-2:]
    out = p[..., 0:h, 0:w].copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(out, p[..., dy:dy + h, dx:dx + w], out=out)
    return out


def pixel_features(feat6: np.ndarray) -> np.ndarray:
    """(N, F, H, W) spectral features -> (N, H, W, 2F+2) per-pixel feature vectors."""
    m3 = box_mean(feat6, 1)
    mx = max_filter3(feat6[:, 5:6])
    contrast = feat6[:, 3:4] - box_mean(feat6[:, 3:4], 4)
    f = np.concatenate([feat6, m3, mx, contrast], axis=1)
    return np.moveaxis(f, 1, -1).astype(np.float32)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class SpectralMLP:
    """Tiny per-pixel MLP segmenter (~400 parameters)."""

    weight_keys = ("W1", "W2")

    def __init__(self, hidden: int = 24, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        f = N_PIXEL_FEATURES
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, np.sqrt(2.0 / f), (f, hidden)).astype(np.float32),
            "b1": np.zeros(hidden, dtype=np.float32),
            "W2": rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, 1)).astype(np.float32),
            "b2": np.array([-2.0], dtype=np.float32),
        }
        self.mu = np.zeros(f, dtype=np.float32)
        self.sigma = np.ones(f, dtype=np.float32)
        self.normalised = False

    # -- inference --------------------------------------------------------------
    def _weights(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        return p["W1"], p["b1"], p["W2"], p["b2"]

    def predict_features(self, x: np.ndarray) -> np.ndarray:
        w1, b1, w2, b2 = self._weights()
        h = np.maximum((x - self.mu) / self.sigma @ w1 + b1, 0.0)
        return _sigmoid((h @ w2)[:, 0] + b2[0])

    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        tiles = np.asarray(tiles, dtype=np.float32)
        n, _, h, w = tiles.shape
        out = np.empty((n, h, w), dtype=np.float32)
        for i in range(0, n, batch_size):
            f = pixel_features(spectral_features(tiles[i:i + batch_size]))
            m = f.shape[0]
            out[i:i + m] = self.predict_features(f.reshape(-1, N_PIXEL_FEATURES)).reshape(m, h, w)
        return out

    # -- training -----------------------------------------------------------------
    @staticmethod
    def _sample_pixels(tiles: np.ndarray, masks: np.ndarray, rng: np.random.Generator,
                       neg_fraction: float, hard_k: int = 24,
                       ignore: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for i in range(0, len(tiles), 64):
            f = pixel_features(spectral_features(tiles[i:i + 64]))
            for j in range(f.shape[0]):
                fj = f[j].reshape(-1, N_PIXEL_FEATURES)
                yj = masks[i + j].reshape(-1) > 0
                keep = np.ones_like(yj) if ignore is None else ~ignore[i + j].reshape(-1)
                if ignore is not None:  # drop ignore-band pixels of pseudo-labelled real chips
                    yj = yj & keep
                neg = np.flatnonzero(~yj & keep)
                take = neg[rng.random(neg.size) < neg_fraction]
                if neg.size:
                    k = min(hard_k, neg.size)
                    hard_ratio = neg[np.argpartition(-fj[neg, 5], k - 1)[:k]]  # high SWIR difference
                    hard_bright = neg[np.argpartition(-fj[neg, 3], k - 1)[:k]]  # bright SWIR-2 (glint/roofs)
                    take = np.unique(np.concatenate([take, hard_ratio, hard_bright]))
                idx = np.concatenate([np.flatnonzero(yj), take])
                xs.append(fj[idx])
                ys.append(yj[idx].astype(np.float32))
        return np.concatenate(xs), np.concatenate(ys)

    def _loss_and_grads(self, x: np.ndarray, y: np.ndarray, pos_weight: float,
                        dice_weight: float) -> tuple[float, dict[str, np.ndarray]]:
        w1, b1, w2, b2 = self._weights()
        hpre = x @ w1 + b1
        h = np.maximum(hpre, 0.0)
        p = _sigmoid((h @ w2)[:, 0] + b2[0])
        m = float(len(y))
        wt = np.where(y > 0, pos_weight, 1.0)
        eps = 1e-7
        bce = float(-(wt * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).mean())
        gz = wt * (p - y) / m
        inter = float((p * y).sum())
        s = float(p.sum() + y.sum() + 1.0)
        dice = 1.0 - (2.0 * inter + 1.0) / s
        gp_dice = -(2.0 * y * s - (2.0 * inter + 1.0)) / s**2
        gz = gz + dice_weight * gp_dice * p * (1.0 - p)
        gh = gz[:, None] @ w2.T
        gh[hpre <= 0] = 0.0
        grads = {"W1": x.T @ gh, "b1": gh.sum(0), "W2": h.T @ gz[:, None], "b2": np.array([gz.sum()])}
        return bce + dice_weight * dice, grads

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *,
            epochs: int = 8, lr: float = 0.01, batch_size: int = 4096, pos_weight: float = 3.0,
            dice_weight: float = 1.0, neg_fraction: float = 0.06, seed: int = 0,
            masks: dict[str, np.ndarray] | None = None, ignore: np.ndarray | None = None,
            log: Callable[[str], None] | None = None) -> list[dict[str, float]]:
        rng = np.random.default_rng(seed)
        feats, labels = self._sample_pixels(x_train, y_train, rng, neg_fraction, ignore=ignore)
        if not self.normalised:
            self.mu = feats.mean(axis=0).astype(np.float32)
            self.sigma = (feats.std(axis=0) + 1e-6).astype(np.float32)
            self.normalised = True
        xs = ((feats - self.mu) / self.sigma).astype(np.float32)
        m1 = {k: np.zeros_like(v) for k, v in self.params.items()}
        m2 = {k: np.zeros_like(v) for k, v in self.params.items()}
        b1, b2, step = 0.9, 0.999, 0
        history: list[dict[str, float]] = []
        best_iou, best = -1.0, copy.deepcopy(self.params)
        for ep in range(epochs):
            perm = rng.permutation(len(xs))
            losses = []
            for i in range(0, len(xs), batch_size):
                idx = perm[i:i + batch_size]
                loss, grads = self._loss_and_grads(xs[idx], labels[idx], pos_weight, dice_weight)
                losses.append(loss)
                step += 1
                for k, g in grads.items():
                    m1[k] = b1 * m1[k] + (1 - b1) * g
                    m2[k] = b2 * m2[k] + (1 - b2) * g * g
                    upd = lr * (m1[k] / (1 - b1**step)) / (np.sqrt(m2[k] / (1 - b2**step)) + 1e-8)
                    self.params[k] = (self.params[k] - upd).astype(np.float32)
                if masks:
                    for k, mk in masks.items():
                        self.params[k] *= mk
            iou = pixel_iou(self.predict_proba(x_val), y_val)
            history.append({"epoch": float(ep + 1), "loss": float(np.mean(losses)), "val_iou": iou})
            if log:
                log(f"    epoch {ep + 1:2d}/{epochs}  loss {np.mean(losses):.4f}  val IoU {iou:.3f}")
            if iou > best_iou:
                best_iou, best = iou, copy.deepcopy(self.params)
        self.params = best
        return history

    # -- compression ----------------------------------------------------------------
    def copy(self) -> SpectralMLP:
        return copy.deepcopy(self)

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def nonzero_params(self) -> int:
        return int(sum(np.count_nonzero(v) for v in self.params.values()))

    def pruned(self, amount: float) -> tuple[SpectralMLP, dict[str, np.ndarray]]:
        """Global magnitude pruning of the weight matrices; returns (model, masks)."""
        m = self.copy()
        all_w = np.concatenate([np.abs(m.params[k]).ravel() for k in self.weight_keys])
        thr = float(np.quantile(all_w, amount))
        masks = {k: (np.abs(m.params[k]) > thr).astype(np.float32) for k in self.weight_keys}
        for k, mk in masks.items():
            m.params[k] *= mk
        return m, masks

    def save(self, path: str | Path) -> int:
        np.savez(path, mu=self.mu, sigma=self.sigma, **self.params)
        return Path(path).stat().st_size


class QuantizedSpectralMLP:
    """INT8 (per-output-channel symmetric) weights, FP32 biases/activations."""

    def __init__(self, src: SpectralMLP) -> None:
        self.q: dict[str, np.ndarray] = {}
        self.scales: dict[str, np.ndarray] = {}
        for k in SpectralMLP.weight_keys:
            w = src.params[k]
            scale = np.abs(w).max(axis=0) / 127.0
            scale[scale == 0] = 1.0
            self.q[k] = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
            self.scales[k] = scale.astype(np.float32)
        self.b1 = src.params["b1"].copy()
        self.b2 = src.params["b2"].copy()
        self.mu = src.mu.copy()
        self.sigma = src.sigma.copy()

    def dequantized(self) -> SpectralMLP:
        m = SpectralMLP()
        m.params = {"W1": self.q["W1"].astype(np.float32) * self.scales["W1"], "b1": self.b1,
                    "W2": self.q["W2"].astype(np.float32) * self.scales["W2"], "b2": self.b2}
        m.mu, m.sigma, m.normalised = self.mu, self.sigma, True
        return m

    def predict_proba(self, tiles: np.ndarray, batch_size: int = 64) -> np.ndarray:
        return self.dequantized().predict_proba(tiles, batch_size)

    def num_params(self) -> int:
        return int(sum(v.size for v in self.q.values()) + self.b1.size + self.b2.size)

    def nonzero_params(self) -> int:
        return int(sum(np.count_nonzero(v) for v in self.q.values()) + np.count_nonzero(self.b1) + 1)

    def save(self, path: str | Path) -> int:
        np.savez(path, mu=self.mu, sigma=self.sigma, b1=self.b1, b2=self.b2,
                 W1_q=self.q["W1"], W1_s=self.scales["W1"], W2_q=self.q["W2"], W2_s=self.scales["W2"])
        return Path(path).stat().st_size


def export_mlp_onnx(model: SpectralMLP, path: str | Path) -> tuple[bool, str]:
    """Export the per-pixel MLP (input: [N, 14] pixel features) with ``onnx.helper``."""
    if not deps.ONNX_AVAILABLE:
        return False, "onnx not installed (pip install onnx onnxruntime) - export skipped"
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        p = model.params
        inits = [
            numpy_helper.from_array(model.mu.astype(np.float32), "mu"),
            numpy_helper.from_array(model.sigma.astype(np.float32), "sigma"),
            numpy_helper.from_array(p["W1"].astype(np.float32), "W1"),
            numpy_helper.from_array(p["b1"].astype(np.float32), "b1"),
            numpy_helper.from_array(p["W2"].astype(np.float32), "W2"),
            numpy_helper.from_array(p["b2"].astype(np.float32), "b2"),
        ]
        nodes = [
            helper.make_node("Sub", ["features", "mu"], ["c"]),
            helper.make_node("Div", ["c", "sigma"], ["z"]),
            helper.make_node("MatMul", ["z", "W1"], ["h0"]),
            helper.make_node("Add", ["h0", "b1"], ["h1"]),
            helper.make_node("Relu", ["h1"], ["h"]),
            helper.make_node("MatMul", ["h", "W2"], ["o0"]),
            helper.make_node("Add", ["o0", "b2"], ["logit"]),
            helper.make_node("Sigmoid", ["logit"], ["fire_prob"]),
        ]
        graph = helper.make_graph(
            nodes, "argus_spectral_mlp",
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["N", N_PIXEL_FEATURES])],
            [helper.make_tensor_value_info("fire_prob", TensorProto.FLOAT, ["N", 1])],
            inits,
        )
        onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, str(path))
        return True, f"exported per-pixel MLP to {Path(path).name}"
    except Exception as exc:
        return False, f"ONNX export failed: {type(exc).__name__}: {exc}"
