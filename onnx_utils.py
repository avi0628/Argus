"""ONNX execution helpers: onnxruntime when installed, else onnx's pure-Python reference evaluator."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import deps


def run_onnx(path: str | Path, feed: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    """Run an ONNX graph and return (first output, engine name)."""
    if deps.ORT_AVAILABLE:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return np.asarray(sess.run(None, feed)[0]), "onnxruntime"
    if deps.ONNX_AVAILABLE:
        import onnx
        from onnx.reference import ReferenceEvaluator

        return np.asarray(ReferenceEvaluator(onnx.load(str(path))).run(None, feed)[0]), "onnx.reference"
    raise RuntimeError("neither onnxruntime nor onnx is installed")


def ort_latency_ms(path: str | Path, feed: dict[str, np.ndarray], runs: int = 20) -> float | None:
    """Median onnxruntime CPU latency; None when onnxruntime is unavailable (the reference
    evaluator is a correctness tool, its speed is not representative)."""
    if not deps.ORT_AVAILABLE:
        return None
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for _ in range(3):
        sess.run(None, feed)
    samples = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        sess.run(None, feed)
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples))


def parity_status(diff: float, engine: str, n: int, tol: float = 1e-3) -> str:
    verdict = "PASSED" if diff < tol else "DRIFT"
    note = "" if engine == "onnxruntime" else " (onnxruntime unavailable: latency not measured)"
    return f"parity {verdict} via {engine} on {n} tile(s), max |diff| {diff:.2e}{note}"
