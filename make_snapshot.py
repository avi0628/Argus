"""Save a full Argus run so the dashboard can open instantly.

    python make_snapshot.py            # live data from data/cache (same settings as `run --live --offline --deterministic`)
    python make_snapshot.py --online   # fetch fresh live data first

Writes outputs/snapshot.pkl.gz, which argus_dashboard.py loads by default.
"""
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from argus.config import SimulationConfig  # noqa: E402
from argus.pipeline import run_pipeline  # noqa: E402

SNAPSHOT = ROOT / "outputs" / "snapshot.pkl.gz"


def sanitize(obj, seen=None):
    """Make the saved run portable: drop live PyTorch objects (models, tensors) and turn OS-specific paths
    (a WindowsPath cannot be unpickled on Linux) into plain strings."""
    import dataclasses
    from pathlib import PurePath

    seen = set() if seen is None else seen
    if id(obj) in seen:
        return obj
    seen.add(id(obj))

    def clean(v):
        mod = type(v).__module__ or ""
        if mod.startswith("torch") or type(v).__name__ in ("TorchSegmenter", "FireDSUNet"):
            return None
        if isinstance(v, PurePath):
            return str(v)
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, tuple) and not hasattr(v, "_fields"):
            return tuple(clean(x) for x in v)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            sanitize(v, seen)
        return v

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            try:
                object.__setattr__(obj, f.name, clean(getattr(obj, f.name)))
            except Exception:
                pass
    for k, v in list(getattr(obj, "__dict__", {}).items()):
        try:
            setattr(obj, k, clean(v))
        except Exception:
            pass
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true", help="download fresh live data instead of using data/cache")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epoch", default="2026-11-22T11:00",
                    help="mission start (UTC) for the cached data; default reproduces the run reported in the paper "
                         "and slides. Use 'now' for the current hour.")
    args = ap.parse_args()

    if args.epoch != "now":
        from argus import live as live_mod

        fixed = datetime.fromisoformat(args.epoch).replace(tzinfo=timezone.utc)
        live_mod.utc_now_hour = lambda: fixed  # the pipeline takes its epoch from this

    cfg = SimulationConfig(seed=args.seed)
    cfg.model.train_threads = 1
    cfg.live.enabled = True
    cfg.live.offline = not args.online
    cfg.output_dir = str(ROOT / "outputs" / "snapshot_run")
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg)
        print(msg, flush=True)

    result = run_pipeline(cfg, log=log)
    # the dashboard never reads the training arrays; dropping them keeps the file small
    result.segmentation.data = None
    result.segmentation.real_test = None
    # nor the live PyTorch objects (keeps the file loadable across Python / PyTorch versions)
    sanitize(result)
    ep = result.live.epoch if result.live else result.orbit.epoch_utc
    meta = {"created_utc": f"{ep:%d %b %Y %H:%M} UTC",
            "settings": "live NASA data (cached), full training, deterministic, seed %d" % args.seed}
    with gzip.open(SNAPSHOT, "wb", compresslevel=6) as fh:
        pickle.dump({"result": result, "logs": logs, "meta": meta}, fh, protocol=4)
    print(f"\nSaved {SNAPSHOT} ({SNAPSHOT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
