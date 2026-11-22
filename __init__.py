"""ARGUS: onboard-AI wildfire early warning from CubeSats with bandwidth-aware downlinking.

Stages
------
0. ``live`` / ``sentinel2`` / ``cog`` - live NASA FIRMS fires, GeoNames exposure, CelesTrak TLEs,
   real Sentinel-2 L2A chips read straight from cloud-optimized GeoTIFFs
1. ``orbit`` / ``coverage`` - constellation, ground passes, link budget, captures, revisit-gap analysis
2. ``segmentation`` - ArgusNet (DS-U-Net + attention) vs. the physics threshold baseline
3. ``edge`` - pruning, INT8 quantization, ONNX export, flight-hardware SWaP benchmark
4. ``scheduler`` - FIFO vs. urgency heuristic vs. Q-learning, plus relay flash alerts

Primary track: SDG 9 (Industry, Innovation & Infrastructure); co-benefits SDG 13, 11, 15.
"""
from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
