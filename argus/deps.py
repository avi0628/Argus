"""Optional-dependency registry.

Only NumPy is required. Every heavier dependency is imported inside its own
try/except block so the end-to-end simulation always runs out of the box:

* torch       -> depthwise-separable U-Net, pruning, INT8 quantization, ONNX export
                 (fallback: NumPy per-pixel spectral MLP with pruning + INT8 weights)
* skyfield    -> SGP4 propagation from TLEs (fallback: analytic J2-perturbed Keplerian)
* onnx        -> ONNX export / model checking (skipped with a clear message when missing)
* onnxruntime -> ONNX latency benchmark and numerical parity check
* streamlit   -> optional dashboard (``dashboard.py``)

Set the environment variable ``ARGUS_DISABLE=torch,skyfield`` to force the
fallback code paths even when the libraries are installed.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np

_DISABLED = {s.strip().lower() for s in (os.environ.get("ARGUS_DISABLE", "") + ","
                                         + os.environ.get("FIRESAT_DISABLE", "")).split(",") if s.strip()}


def _disabled(name: str) -> None:
    if name in _DISABLED:
        raise ImportError(f"{name} disabled via ARGUS_DISABLE")


# --- PyTorch ---------------------------------------------------------------
try:
    _disabled("torch")
    import torch as _torch

    TORCH_AVAILABLE: bool = True
    TORCH_VERSION: str | None = str(_torch.__version__)
    TORCH_ERROR: str | None = None
except Exception as _exc:  # ImportError, or OSError when native libraries fail to load
    TORCH_AVAILABLE = False
    TORCH_VERSION = None
    TORCH_ERROR = f"{type(_exc).__name__}: {_exc}"

# --- Skyfield + SGP4 --------------------------------------------------------
try:
    _disabled("skyfield")
    import sgp4 as _sgp4  # noqa: F401
    import skyfield as _skyfield
    from skyfield.api import EarthSatellite as _EarthSatellite  # noqa: F401

    SKYFIELD_AVAILABLE: bool = True
    _sf_version = getattr(_skyfield, "__version__", None) or ".".join(
        str(v) for v in getattr(_skyfield, "VERSION", ("?",))
    )
    SKYFIELD_VERSION: str | None = str(_sf_version)
    SKYFIELD_ERROR: str | None = None
except Exception as _exc:
    SKYFIELD_AVAILABLE = False
    SKYFIELD_VERSION = None
    SKYFIELD_ERROR = f"{type(_exc).__name__}: {_exc}"

# --- ONNX -------------------------------------------------------------------
try:
    _disabled("onnx")
    import onnx as _onnx

    ONNX_AVAILABLE: bool = True
    ONNX_VERSION: str | None = str(_onnx.__version__)
except Exception:
    ONNX_AVAILABLE = False
    ONNX_VERSION = None

# --- ONNX Runtime -----------------------------------------------------------
try:
    _disabled("onnxruntime")
    import onnxruntime as _ort

    ORT_AVAILABLE: bool = True
    ORT_VERSION: str | None = str(_ort.__version__)
except Exception:
    ORT_AVAILABLE = False
    ORT_VERSION = None


# --- Light-weight probes (not imported: slow / side effects) ----------------
def _has_module(name: str) -> bool:
    if name in _DISABLED:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


STREAMLIT_AVAILABLE: bool = _has_module("streamlit")
PANDAS_AVAILABLE: bool = _has_module("pandas")
RASTERIO_AVAILABLE: bool = _has_module("rasterio")


def dependency_rows() -> list[tuple[str, str, str]]:
    """(package, version-or-status, role) rows for the report."""
    return [
        ("numpy", str(np.__version__), "required"),
        ("torch", TORCH_VERSION or "missing", "DS-U-Net / pruning / INT8 / ONNX" if TORCH_AVAILABLE
         else "fallback: NumPy spectral MLP"),
        ("skyfield+sgp4", SKYFIELD_VERSION or "missing", "SGP4 orbits from TLEs" if SKYFIELD_AVAILABLE
         else "fallback: analytic J2 propagator"),
        ("onnx", ONNX_VERSION or "missing", "ONNX export" if ONNX_AVAILABLE else "ONNX export skipped"),
        ("onnxruntime", ORT_VERSION or "missing", "ONNX benchmark + parity" if ORT_AVAILABLE
         else ("parity via onnx.reference, no latency" if ONNX_AVAILABLE else "ONNX checks skipped")),
        ("streamlit", "installed" if STREAMLIT_AVAILABLE else "missing", "optional dashboard"),
    ]


def short_status() -> str:
    return " | ".join(f"{name} {ver}" for name, ver, _ in dependency_rows())
