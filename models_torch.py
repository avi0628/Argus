"""PyTorch depthwise-separable (MobileNet-style) U-Net for active-fire segmentation.

This module imports torch at module level and must only be imported after
checking ``argus.deps.TORCH_AVAILABLE``.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import deps
from .metrics import pixel_iou

try:
    import torch.ao.quantization as tq

    QUANT_API_AVAILABLE = True
except Exception:  # stripped / very old torch builds
    tq = None  # type: ignore[assignment]
    QUANT_API_AVAILABLE = False

try:
    import torch.nn.utils.prune as torch_prune

    PRUNE_API_AVAILABLE = True
except Exception:
    torch_prune = None  # type: ignore[assignment]
    PRUNE_API_AVAILABLE = False

try:
    from torch.ao.nn.quantized import FloatFunctional as _FloatFunctional
except Exception:
    try:
        from torch.nn.quantized import FloatFunctional as _FloatFunctional  # type: ignore[no-redef]
    except Exception:
        _FloatFunctional = None  # type: ignore[assignment,misc]


class _CatOp(nn.Module):
    """Stand-in for FloatFunctional when the quantization API is unavailable."""

    def cat(self, xs: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
        return torch.cat(xs, dim=dim)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


def _make_cat() -> nn.Module:
    return _FloatFunctional() if _FloatFunctional is not None else _CatOp()


def _quant_stub() -> nn.Module:
    return tq.QuantStub() if QUANT_API_AVAILABLE else nn.Identity()


def _dequant_stub() -> nn.Module:
    return tq.DeQuantStub() if QUANT_API_AVAILABLE else nn.Identity()


class ConvBNReLU(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each with BatchNorm + ReLU (fusable for INT8)."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvBNReLU(cin, cin, 3, stride, groups=cin)
        self.pw = ConvBNReLU(cin, cout, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SqueezeExcite(nn.Module):
    """Channel attention (Hu et al. 2018); quantization-friendly (fused conv-ReLU, QFunctional mul)."""

    def __init__(self, ch: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2d(hidden, ch, 1)
        self.gate = nn.Sigmoid()
        self.mul = _make_cat()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return self.mul.mul(x, w.expand_as(x))


class DSConvSE(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.ds = DSConv(cin, cout, stride)
        self.se = SqueezeExcite(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.ds(x))


class FireDSUNet(nn.Module):
    """ArgusNet: 3-level depthwise-separable U-Net with squeeze-excitation attention in the
    bottleneck and decoder. Input (N, 8, H, W) with H, W divisible by 4."""

    def __init__(self, in_channels: int = 8, base: int = 16) -> None:
        super().__init__()
        b = base
        self.quant = _quant_stub()
        self.dequant = _dequant_stub()
        self.stem = ConvBNReLU(in_channels, b, 3)
        self.enc1 = DSConv(b, b)
        self.down2 = DSConv(b, 2 * b, stride=2)
        self.enc2 = DSConv(2 * b, 2 * b)
        self.down3 = DSConv(2 * b, 4 * b, stride=2)
        self.enc3 = DSConvSE(4 * b, 4 * b)
        self.dec2 = DSConvSE(4 * b + 2 * b, 2 * b)
        self.dec1 = DSConvSE(2 * b + b, b)
        self.head = nn.Conv2d(b, 1, 1)
        self.cat2 = _make_cat()
        self.cat1 = _make_cat()
        nn.init.constant_(self.head.bias, -3.0)  # prior: fire pixels are rare

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(self.down2(s1))
        bott = self.enc3(self.down3(s2))
        u2 = F.interpolate(bott, scale_factor=2.0, mode="nearest")
        d2 = self.dec2(self.cat2.cat([u2, s2], dim=1))
        u1 = F.interpolate(d2, scale_factor=2.0, mode="nearest")
        d1 = self.dec1(self.cat1.cat([u1, s1], dim=1))
        return self.dequant(self.head(d1))

    def fusion_groups(self) -> list[list[str]]:
        groups = [[f"{n}.conv", f"{n}.bn", f"{n}.act"] for n, m in self.named_modules() if isinstance(m, ConvBNReLU)]
        groups += [[f"{n}.fc1", f"{n}.act"] for n, m in self.named_modules() if isinstance(m, SqueezeExcite)]
        return groups


# ---------------------------------------------------------------------------
# Loss, training, inference
# ---------------------------------------------------------------------------
def focal_dice_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0,
                    dice_weight: float = 1.0, weight: torch.Tensor | None = None) -> tuple[torch.Tensor, float, float]:
    """Pixel focal loss + batch-level soft Dice; ``weight`` (0 = ignore) masks uncertain pseudo-label pixels."""
    w = torch.ones_like(target) if weight is None else weight
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = (w * alpha_t * (1.0 - p_t).pow(gamma) * bce).sum() / w.sum().clamp_min(1.0)
    pw = prob * w
    inter = (pw * target).sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (pw.sum() + (target * w).sum() + 1.0)
    return focal * 10.0 + dice_weight * dice, float(focal.detach()), float(dice.detach())


def predict_logits_torch(model: nn.Module, feats: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(feats), batch_size):
            out.append(model(torch.from_numpy(np.ascontiguousarray(feats[i:i + batch_size])))[:, 0].numpy())
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0,) + feats.shape[2:], np.float32)


def predict_proba_torch(model: nn.Module, feats: np.ndarray, batch_size: int = 64,
                        temperature: float = 1.0) -> np.ndarray:
    logits = predict_logits_torch(model, feats, batch_size)
    return (1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -30.0, 30.0)))).astype(np.float32)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, valid: np.ndarray | None = None,
                    topk: int = 8) -> float:
    """Temperature scaling (Guo et al. 2017) of the *alert* confidence: grid-search T minimising the
    tile-level NLL of confidence = mean of the top-k pixel probabilities vs. "tile contains fire"."""
    z = np.asarray(logits, dtype=np.float64).reshape(len(logits), -1)
    if valid is not None:
        z = np.where(np.asarray(valid).reshape(len(z), -1), z, -30.0)
    y = (np.asarray(labels).reshape(len(z), -1).sum(axis=1) > 0).astype(np.float64)
    if y.min() == y.max():
        return 1.0
    k = int(max(1, min(topk, z.shape[1])))
    top = np.partition(z, -k, axis=1)[:, -k:]
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.3), np.log(6.0), 80)):
        conf = (1.0 / (1.0 + np.exp(-np.clip(top / t, -30, 30)))).mean(axis=1)
        nll = -np.mean(y * np.log(conf + 1e-7) + (1 - y) * np.log(1 - conf + 1e-7))
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return best_t


def train_fire_unet(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                    y_val: np.ndarray, *, epochs: int, batch_size: int, lr: float, alpha: float, gamma: float,
                    dice_weight: float, seed: int, threads: int = 4, w_train: np.ndarray | None = None,
                    val_valid: np.ndarray | None = None,
                    log: Callable[[str], None] | None = None) -> list[dict[str, float]]:
    """Train in place (features already computed). Restores the best validation-IoU weights.

    ``w_train`` (N, H, W) per-pixel loss weights (0 = ignore); ``val_valid`` masks validation pixels.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, threads))
    xt = torch.from_numpy(np.ascontiguousarray(x_train))
    yt = torch.from_numpy(np.ascontiguousarray(y_train[:, None].astype(np.float32)))
    wt = None if w_train is None else torch.from_numpy(np.ascontiguousarray(w_train[:, None].astype(np.float32)))
    n = len(xt)
    steps = max(1, math.ceil(n / batch_size))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.25)
    history: list[dict[str, float]] = []
    best_iou, best_state = -1.0, copy.deepcopy(model.state_dict())
    try:
        for ep in range(epochs):
            model.train()
            perm = torch.from_numpy(rng.permutation(n))
            tot = foc = dic = 0.0
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = xt[idx], yt[idx]
                wb = None if wt is None else wt[idx]
                k = int(rng.integers(4))
                if k:
                    xb, yb = torch.rot90(xb, k, (2, 3)), torch.rot90(yb, k, (2, 3))
                    wb = None if wb is None else torch.rot90(wb, k, (2, 3))
                if rng.random() < 0.5:
                    xb, yb = torch.flip(xb, (3,)), torch.flip(yb, (3,))
                    wb = None if wb is None else torch.flip(wb, (3,))
                loss, f_val, d_val = focal_dice_loss(model(xb), yb, alpha, gamma, dice_weight, wb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot, foc, dic = tot + float(loss.detach()), foc + f_val, dic + d_val
            iou = pixel_iou(predict_proba_torch(model, x_val), y_val, valid=val_valid)
            history.append({"epoch": float(ep + 1), "loss": tot / steps, "focal": foc / steps,
                            "dice": dic / steps, "val_iou": iou})
            if log:
                log(f"    epoch {ep + 1:2d}/{epochs}  loss {tot / steps:.4f}  (focal {foc / steps:.4f}, "
                    f"dice {dic / steps:.3f})  val IoU {iou:.3f}")
            if iou > best_iou:
                best_iou, best_state = iou, copy.deepcopy(model.state_dict())
    finally:
        torch.set_num_threads(old_threads)
    model.load_state_dict(best_state)
    model.eval()
    return history


# ---------------------------------------------------------------------------
# Edge compression helpers
# ---------------------------------------------------------------------------
def conv_layers(model: nn.Module) -> list[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, non-zero) parameters of an FP32 model."""
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += int(torch.count_nonzero(p.detach()))
    return total, nonzero


def count_conv_macs(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    macs = [0]

    def hook(m: nn.Conv2d, _inp: Any, out: torch.Tensor) -> None:
        kh, kw = m.kernel_size
        macs[0] += int(out.numel()) * (m.in_channels // m.groups) * kh * kw

    handles = [m.register_forward_hook(hook) for m in conv_layers(model)]
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(input_shape))
    finally:
        for h in handles:
            h.remove()
    return macs[0]


def activation_bytes(model: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Activation memory upper bound: input + every conv output, no buffer reuse.

    Only convolution modules (FP32 ``Conv2d`` or fused/quantized ``Conv*`` modules) are
    counted, because BatchNorm/ReLU are fused into the convolution on the flight target.
    """
    total = [0]

    def hook(_m: nn.Module, _inp: Any, out: Any) -> None:
        if isinstance(out, torch.Tensor):
            total[0] += int(out.numel()) * int(out.element_size())

    convs = [m for m in model.modules() if not list(m.children()) and "Conv" in type(m).__name__]
    handles = [m.register_forward_hook(hook) for m in convs]
    try:
        with torch.no_grad():
            model(torch.zeros(input_shape))
    finally:
        for h in handles:
            h.remove()
    return total[0] + int(np.prod(input_shape)) * 4


def apply_global_l1_pruning(model: nn.Module, amount: float) -> list[tuple[nn.Module, str]]:
    if not PRUNE_API_AVAILABLE:
        raise RuntimeError("torch.nn.utils.prune unavailable")
    params = [(m, "weight") for m in conv_layers(model)]
    torch_prune.global_unstructured(params, pruning_method=torch_prune.L1Unstructured, amount=amount)
    return params


def finalize_pruning(params: list[tuple[nn.Module, str]]) -> None:
    for module, name in params:
        torch_prune.remove(module, name)


def _pick_quant_engine() -> str:
    engines = list(torch.backends.quantized.supported_engines)
    for eng in ("x86", "fbgemm", "onednn", "qnnpack"):
        if eng in engines:
            return eng
    raise RuntimeError(f"no supported quantized engine in {engines}")


def quantize_static_int8(model: FireDSUNet, calib_feats: np.ndarray) -> tuple[nn.Module, str]:
    """Eager-mode post-training static INT8 quantization (Conv+BN+ReLU fusion, per-channel weights)."""
    if not QUANT_API_AVAILABLE:
        raise RuntimeError("torch.ao.quantization unavailable")
    engine = _pick_quant_engine()
    torch.backends.quantized.engine = engine
    m = copy.deepcopy(model).cpu().eval()
    m.qconfig = tq.get_default_qconfig(engine)
    tq.fuse_modules(m, m.fusion_groups(), inplace=True)
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        for i in range(0, len(calib_feats), 16):
            m(torch.from_numpy(np.ascontiguousarray(calib_feats[i:i + 16])))
    tq.convert(m, inplace=True)
    return m, engine


def fake_quantize_weights_int8(model: nn.Module) -> nn.Module:
    """Fallback: per-output-channel symmetric INT8 weight rounding (simulated quantization)."""
    m = copy.deepcopy(model).eval()
    with torch.no_grad():
        for conv in conv_layers(m):
            w = conv.weight
            scale = w.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12) / 127.0
            conv.weight.copy_(torch.round(w / scale).clamp(-127, 127) * scale)
    return m


def export_onnx(model: nn.Module, path: str | Path, input_shape: tuple[int, ...]) -> tuple[bool, str]:
    """Export to ONNX (opset 17). Tries the TorchScript exporter first, then the dynamo exporter."""
    if not deps.ONNX_AVAILABLE:
        return False, "onnx not installed (pip install onnx onnxruntime) - export skipped"
    dummy = torch.zeros(input_shape)
    kwargs: dict[str, Any] = {
        "input_names": ["features"], "output_names": ["fire_logits"], "opset_version": 17,
        "dynamic_axes": {"features": {0: "batch"}, "fire_logits": {0: "batch"}},
    }
    errors: list[str] = []
    model = copy.deepcopy(model).eval()
    for extra in ({"dynamo": False}, {}, {"dynamo": True}):
        try:
            torch.onnx.export(model, (dummy,), str(path), **kwargs, **extra)
            if Path(path).exists():
                import onnx

                onnx.checker.check_model(onnx.load(str(path)))
                return True, f"exported FP32 graph to {Path(path).name} (opset 17)"
        except TypeError as exc:  # older torch without the ``dynamo`` keyword
            errors.append(f"TypeError: {exc}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return False, "ONNX export failed: " + " | ".join(e[:160] for e in errors)


def onnx_runtime_check(path: str | Path, model: nn.Module, feats: np.ndarray,
                       runs: int = 20) -> tuple[float | None, float | None, str]:
    """(median ORT latency ms or None, max |logit diff| vs torch, status).

    Parity runs on a small batch, which also exercises the dynamic batch axis. Uses
    onnxruntime when installed, otherwise onnx's reference evaluator.
    """
    from .onnx_utils import ort_latency_ms, parity_status, run_onnx

    try:
        xb = np.ascontiguousarray(feats[:4])
        out, engine = run_onnx(path, {"features": xb})
        with torch.no_grad():
            ref = model.eval()(torch.from_numpy(xb)).numpy()
        diff = float(np.max(np.abs(out - ref)))
        latency = ort_latency_ms(path, {"features": np.ascontiguousarray(feats[:1])}, runs)
        return latency, diff, parity_status(diff, engine, len(xb))
    except Exception as exc:
        return None, None, f"ONNX parity check failed: {type(exc).__name__}: {exc}"
