"""Stage 3 - edge compression (pruning, INT8 quantization, ONNX export) and flight-hardware benchmark.

Host measurements (latency, file size, IoU) are real. Flight-computer numbers
(Jetson Nano 5 W mode, Raspberry Pi 4 / CM4) are *modelled* from the measured
MAC count and published-class sustained throughput / power figures in
``config.HardwareProfile``; replace them with on-target measurements when available.
"""
from __future__ import annotations

import time
import tracemalloc
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import deps
from .config import HardwareProfile, SimulationConfig
from .metrics import segmentation_metrics
from .models_numpy import N_PIXEL_FEATURES, QuantizedSpectralMLP, SpectralMLP, export_mlp_onnx, pixel_features
from .payload import PayloadModel
from .segmentation import NumpySegmenter, SegmentationResult, Segmenter, TorchSegmenter
from .tiles import N_FEATURES, spectral_features

Logger = Callable[[str], None]


@dataclass
class HardwareEstimate:
    profile: str
    latency_ms_per_tile: float
    energy_mj_per_tile: float
    scene_latency_s: float
    scene_energy_j: float


@dataclass
class VariantBenchmark:
    name: str
    precision: str
    params_total: int
    params_nonzero: int
    size_bytes: int
    size_bytes_compressed: int
    host_latency_ms: float
    host_latency_p90_ms: float
    memory_bytes: int
    macs: int
    metrics: dict[str, float]
    iou_delta: float = 0.0
    hardware: list[HardwareEstimate] = field(default_factory=list)
    path: str | None = None
    notes: str = ""

    @property
    def iou(self) -> float:
        return float(self.metrics.get("pixel_iou", float("nan")))


@dataclass
class EdgeResult:
    backend: str
    variants: list[VariantBenchmark]
    deployed_name: str
    deployed: Segmenter
    deploy_reason: str
    onnx_path: str | None
    onnx_status: str
    onnx_latency_ms: float | None
    quant_engine: str | None
    tiles_per_scene: int
    notes: list[str] = field(default_factory=list)

    def variant(self, name: str) -> VariantBenchmark:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(name)

    @property
    def deployed_variant(self) -> VariantBenchmark:
        return self.variant(self.deployed_name)

    def deployed_scene_latency_s(self) -> float:
        hw = self.deployed_variant.hardware
        return hw[0].scene_latency_s if hw else 0.0


def _time_fn(fn: Callable[[], Any], runs: int, warmup: int = 10) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples)), float(np.percentile(samples, 90))


def _compressed_size(path: Path) -> int:
    return len(zlib.compress(path.read_bytes(), 9))


def hardware_estimates(macs: int, precision: str, tiles_per_scene: int,
                       profiles: tuple[HardwareProfile, ...]) -> list[HardwareEstimate]:
    out = []
    for hw in profiles:
        gmacs = hw.effective_gmacs_int8 if precision == "int8" else hw.effective_gmacs_fp32
        lat_ms = macs / (gmacs * 1e9) * 1e3 + hw.per_inference_overhead_ms
        energy_mj = hw.active_power_w * lat_ms  # W * ms = mJ
        out.append(HardwareEstimate(hw.short_name, lat_ms, energy_mj, lat_ms * tiles_per_scene / 1e3,
                                    energy_mj * tiles_per_scene / 1e3))
    return out


class EdgeCompressor:
    """Builds compressed variants of the trained model, benchmarks them and selects one to fly."""

    def __init__(self, cfg: SimulationConfig, output_dir: Path, log: Logger = print) -> None:
        self.cfg = cfg
        self.out = output_dir
        self.models_dir = output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.tiles_per_scene = PayloadModel(cfg.payload, cfg.model.tile_size, cfg.model.gsd_m).tiles_per_scene

    def run(self, seg: SegmentationResult) -> EdgeResult:
        if seg.backend == "torch":
            result = self._run_torch(seg)
        else:
            result = self._run_numpy(seg)
        base_iou = result.variants[0].iou
        for v in result.variants:
            v.iou_delta = v.iou - base_iou
        return result

    def _metrics(self, segmenter: Segmenter, seg: SegmentationResult) -> dict[str, float]:
        mc = self.cfg.model
        prob = segmenter.predict_proba(seg.data.x_test)
        return segmentation_metrics(prob, seg.data.y_test, mc.detection_threshold, mc.min_fire_pixels,
                                    mc.confidence_topk)

    def _select(self, variants: list[VariantBenchmark]) -> tuple[str, str]:
        base = variants[0]
        ok = [v for v in variants if v.iou >= base.iou - self.cfg.edge.max_iou_drop]
        best = min(ok, key=lambda v: (v.size_bytes_compressed, v.host_latency_ms))
        reason = (f"smallest stored model within {self.cfg.edge.max_iou_drop:.0%}-pt IoU of FP32 "
                  f"(IoU {best.iou:.3f} vs {base.iou:.3f}; {best.size_bytes_compressed / 1e3:.1f} kB vs "
                  f"{base.size_bytes_compressed / 1e3:.1f} kB)")
        return best.name, reason

    # ------------------------------------------------------------------ torch
    def _run_torch(self, seg: SegmentationResult) -> EdgeResult:
        import copy

        import torch

        from . import models_torch as mt

        ec, mc = self.cfg.edge, self.cfg.model
        notes: list[str] = []
        base = seg.raw_model.eval()
        shape = (1, N_FEATURES, mc.tile_size, mc.tile_size)
        macs = mt.count_conv_macs(base, shape)
        x1 = torch.from_numpy(np.ascontiguousarray(spectral_features(seg.data.x_test[:1])))  # real tile
        feats_val = spectral_features(seg.data.x_val)

        @contextmanager
        def threads(n: int) -> Iterator[None]:
            # flush-to-zero like ARM NEON / GPU flight targets; x86 denormals otherwise skew timings
            old = torch.get_num_threads()
            torch.set_num_threads(max(1, n))
            ftz = torch.set_flush_denormal(True)
            try:
                yield
            finally:
                torch.set_num_threads(old)
                if ftz:
                    torch.set_flush_denormal(False)

        def bench(name: str, module: Any, precision: str, counts_from: Any, note: str = "",
                  size_override: tuple[int, int] | None = None) -> VariantBenchmark:
            path = self.models_dir / f"{name.lower().replace(' ', '_').replace('+', '')}.pt"
            if size_override is None:
                torch.save(module.state_dict(), path)
                size, csize = path.stat().st_size, _compressed_size(path)
            else:
                size, csize = size_override
            total, nonzero = mt.count_params(counts_from)
            module.eval()
            with threads(ec.bench_threads), torch.no_grad():
                med, p90 = _time_fn(lambda: module(x1), ec.bench_runs)
                act = mt.activation_bytes(module, shape)
            seg_wrap = TorchSegmenter(module, name, seg.temperature)
            v = VariantBenchmark(
                name=name, precision=precision, params_total=total, params_nonzero=nonzero, size_bytes=size,
                size_bytes_compressed=csize, host_latency_ms=med, host_latency_p90_ms=p90,
                memory_bytes=int(size + act), macs=macs, metrics=self._metrics(seg_wrap, seg),
                hardware=hardware_estimates(macs, precision, self.tiles_per_scene, ec.hardware),
                path=str(path) if size_override is None else None, notes=note,
            )
            self.log(f"  {name:<22s} {size / 1e3:7.1f} kB  host {med:6.2f} ms/tile  IoU {v.iou:.3f}")
            return v

        variants = [bench("FP32 baseline", base, "fp32", base)]
        segmenters: dict[str, Segmenter] = {"FP32 baseline": TorchSegmenter(base, "FP32 baseline", seg.temperature)}

        # 1) global L1 unstructured pruning + short fine-tune
        pruned = copy.deepcopy(base)
        try:
            params = mt.apply_global_l1_pruning(pruned, ec.prune_amount)
            if ec.finetune_epochs > 0:
                mt.train_fire_unet(
                    pruned, spectral_features(seg.data.x_train), seg.data.y_train, feats_val, seg.data.y_val,
                    epochs=ec.finetune_epochs, batch_size=mc.batch_size, lr=mc.learning_rate * 0.25,
                    alpha=mc.focal_alpha, gamma=mc.focal_gamma, dice_weight=mc.dice_weight,
                    seed=self.cfg.seed + 1, threads=mc.train_threads,
                )
            mt.finalize_pruning(params)
            name = f"Pruned {int(ec.prune_amount * 100)}%"
            variants.append(bench(name, pruned, "fp32", pruned,
                                  f"global L1 unstructured, {ec.finetune_epochs} fine-tune epoch(s)"))
            segmenters[name] = TorchSegmenter(pruned, name, seg.temperature)
        except Exception as exc:
            notes.append(f"pruning skipped: {type(exc).__name__}: {exc}")
            pruned = base

        # 2) INT8 post-training static quantization of the pruned model
        engine: str | None = None
        if ec.quantize:
            calib = feats_val[: ec.calibration_tiles]
            name = "Pruned + INT8 PTQ" if pruned is not base else "INT8 PTQ"
            try:
                qmodel, engine = mt.quantize_static_int8(pruned, calib)
                variants.append(bench(name, qmodel, "int8", pruned,
                                      f"static PTQ, per-channel weights, fused Conv-BN-ReLU, engine={engine}"))
                segmenters[name] = TorchSegmenter(qmodel, name, seg.temperature)
            except Exception as exc:
                notes.append(f"static INT8 PTQ failed ({type(exc).__name__}: {exc}); using simulated INT8 weights")
                fq = mt.fake_quantize_weights_int8(pruned)
                total, nonzero = mt.count_params(fq)
                est = int(nonzero + 4 * total // 64)  # int8 weights + per-channel scales (approx.)
                name = "INT8 weights (simulated)"
                variants.append(bench(name, fq, "int8", fq, "fake-quantized weights; size estimated",
                                      size_override=(est, est)))
                segmenters[name] = TorchSegmenter(fq, name, seg.temperature)

        # 3) ONNX export of the FP32 (pruned) graph for TensorRT / ONNX Runtime on the flight computer
        onnx_path: str | None = None
        onnx_latency: float | None = None
        if ec.export_onnx:
            target = self.models_dir / "argusnet.onnx"
            ok, status = mt.export_onnx(pruned, target, shape)
            if ok:
                onnx_path = str(target)
                onnx_latency, _diff, check_status = mt.onnx_runtime_check(target, pruned, feats_val, ec.bench_runs)
                status += f"; {check_status}"
        else:
            status = "ONNX export disabled"

        name, reason = self._select(variants)
        return EdgeResult("torch", variants, name, segmenters[name], reason, onnx_path, status, onnx_latency,
                          engine, self.tiles_per_scene, notes)

    # ------------------------------------------------------------------ numpy
    def _run_numpy(self, seg: SegmentationResult) -> EdgeResult:
        ec, mc = self.cfg.edge, self.cfg.model
        base: SpectralMLP = seg.raw_model
        n = mc.tile_size
        macs = (N_PIXEL_FEATURES * 24 + 24) * n * n  # dense per-pixel MLP
        tile1 = seg.data.x_test[:1]

        def bench(name: str, model: SpectralMLP | QuantizedSpectralMLP, precision: str,
                  note: str = "") -> VariantBenchmark:
            path = self.models_dir / f"{name.lower().replace(' ', '_').replace('+', '')}.npz"
            size = model.save(path)
            med, p90 = _time_fn(lambda: model.predict_proba(tile1), ec.bench_runs)
            tracemalloc.start()
            model.predict_proba(tile1)
            _cur, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            v = VariantBenchmark(
                name=name, precision=precision, params_total=base.num_params(),
                params_nonzero=model.nonzero_params(), size_bytes=size, size_bytes_compressed=_compressed_size(path),
                host_latency_ms=med, host_latency_p90_ms=p90, memory_bytes=int(peak), macs=macs,
                metrics=self._metrics(NumpySegmenter(model, name), seg),
                hardware=hardware_estimates(macs, precision, self.tiles_per_scene, ec.hardware),
                path=str(path), notes=note,
            )
            self.log(f"  {name:<22s} {size / 1e3:7.1f} kB  host {med:6.2f} ms/tile  IoU {v.iou:.3f}")
            return v

        variants = [bench("FP32 baseline", base, "fp32")]
        segmenters: dict[str, Segmenter] = {"FP32 baseline": NumpySegmenter(base, "FP32 baseline")}
        pruned, masks = base.pruned(ec.prune_amount)
        if ec.finetune_epochs > 0:
            pruned.fit(seg.data.x_train, seg.data.y_train, seg.data.x_val, seg.data.y_val,
                       epochs=ec.finetune_epochs, lr=0.003, seed=self.cfg.seed + 1, masks=masks)
        pname = f"Pruned {int(ec.prune_amount * 100)}%"
        variants.append(bench(pname, pruned, "fp32", "global magnitude pruning + masked fine-tune"))
        segmenters[pname] = NumpySegmenter(pruned, pname)
        if ec.quantize:
            q = QuantizedSpectralMLP(pruned)
            qname = "Pruned + INT8 weights"
            variants.append(bench(qname, q, "int8", "per-channel symmetric INT8 weights"))
            segmenters[qname] = NumpySegmenter(q, qname)
        onnx_path: str | None = None
        status = "ONNX export disabled"
        if ec.export_onnx:
            target = self.models_dir / "argus_spectral_mlp.onnx"
            ok, status = export_mlp_onnx(pruned, target)
            onnx_path = str(target) if ok else None
            if ok:
                from .onnx_utils import parity_status, run_onnx

                try:
                    f = pixel_features(spectral_features(tile1)).reshape(-1, N_PIXEL_FEATURES).astype(np.float32)
                    out, engine = run_onnx(target, {"features": f})
                    diff = float(np.max(np.abs(out[:, 0] - pruned.predict_features(f))))
                    status += f"; {parity_status(diff, engine, 1)}"
                except Exception as exc:
                    status += f"; ONNX parity check failed: {type(exc).__name__}: {exc}"
        name, reason = self._select(variants)
        return EdgeResult("numpy", variants, name, segmenters[name], reason, onnx_path, status, None, None,
                          self.tiles_per_scene, ["torch unavailable or disabled: NumPy model compressed instead"])


def downlink_energy_comparison(payload: PayloadModel, rate_mbps: float, tx_power_w: float,
                               alert_bytes: int) -> dict[str, float]:
    """Transmitter energy (J) to downlink one scene in each product format at ``rate_mbps``."""
    def joules(nbytes: float) -> float:
        return nbytes * 8.0 / (rate_mbps * 1e6) * tx_power_w

    return {
        "rate_mbps": rate_mbps,
        "raw_j": joules(payload.raw_scene_bytes),
        "compressed_j": joules(payload.compressed_scene_bytes),
        "alert_j": joules(alert_bytes),
    }
