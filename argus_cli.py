#!/usr/bin/env python3
"""ARGUS - onboard-AI wildfire early warning from CubeSats.

Commands (run from any working directory):
  python argus_cli.py                     full mission run on the built-in fire-season scenario
  python argus_cli.py run --live          drive the mission with today's NASA FIRMS fires + live TLEs
  python argus_cli.py run --live --offline   same, but only from data/cache (no network)
  python argus_cli.py live                print today's live fire picture (FIRMS + population exposure)
  python argus_cli.py build-dataset       download real Sentinel-2 chips around recent fires (training data)
  python argus_cli.py dashboard           launch the Streamlit dashboard
Add --fast for a quick smoke run, --deterministic for bit-reproducible training.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from argus import deps  # noqa: E402
from argus.config import SimulationConfig  # noqa: E402

COMMANDS = ("run", "live", "build-dataset", "dashboard")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv.insert(0, "run")  # `python argus_cli.py --fast` == `python argus_cli.py run --fast`
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the full four-stage mission simulation")
    r.add_argument("--live", action="store_true", help="use today's NASA FIRMS fires, GeoNames, CelesTrak TLEs")
    r.add_argument("--offline", action="store_true", help="never touch the network (use data/cache)")
    r.add_argument("--fast", action="store_true", help="smaller datasets / fewer epochs")
    r.add_argument("--deterministic", action="store_true", help="bit-reproducible training (single thread)")
    r.add_argument("--hours", type=float, default=None, help="imaging window in hours (default 24)")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--backend", choices=("auto", "skyfield", "analytic"), default="auto")
    r.add_argument("--tle-file", default=None, help="TLE file for a custom constellation (Skyfield backend)")
    r.add_argument("--no-torch", action="store_true", help="force the NumPy segmentation fallback")
    r.add_argument("--no-real", action="store_true", help="train on synthetic tiles only")
    r.add_argument("--no-relay", action="store_true", help="disable relay flash alerts")
    r.add_argument("--epochs", type=int, default=None)
    r.add_argument("--rl-episodes", type=int, default=None)
    r.add_argument("--buffer-mb", type=float, default=None)
    r.add_argument("--zones", type=int, default=None, help="live mode: number of fires to task")
    r.add_argument("--firms-csv", default=None, help="NASA FIRMS CSV to derive watch zones (non-live mode)")
    r.add_argument("--tiles-dir", default=None, help="directory of real .npy/.npz/.tif tiles to score")
    r.add_argument("--output-dir", default=None, help="default: <project>/outputs")
    r.add_argument("--json", default=None, help="summary path (default: <output-dir>/summary.json)")
    r.add_argument("--quiet", action="store_true", help="print only the final report")

    lv = sub.add_parser("live", help="print today's live fire picture")
    lv.add_argument("--offline", action="store_true")
    lv.add_argument("--top", type=int, default=20)

    b = sub.add_parser("build-dataset", help="build data/real_s2_chips.* from Sentinel-2 + FIRMS (7-day archive)")
    b.add_argument("--target", type=int, default=90, help="stop after this many chips")
    b.add_argument("--candidates", type=int, default=220, help="max fire-day clusters to search")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--append", action="store_true", help="merge with the existing dataset")

    sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> SimulationConfig:
    cfg = SimulationConfig(seed=args.seed)
    if args.fast:
        cfg.apply_fast_mode()
    if args.hours is not None:
        cfg.orbit.duration_hours = max(1.0, args.hours)
    cfg.orbit.backend = args.backend
    cfg.orbit.tle_file = args.tle_file
    cfg.model.use_torch = not args.no_torch
    cfg.model.use_real_data = not args.no_real
    cfg.model.tiles_dir = args.tiles_dir
    if args.deterministic:
        cfg.model.train_threads = 1  # multi-threaded CPU reductions are not bit-reproducible
    if args.epochs is not None:
        cfg.model.epochs = max(1, args.epochs)
    if args.rl_episodes is not None:
        cfg.scheduler.rl_episodes = max(1, args.rl_episodes)
    if args.buffer_mb is not None:
        cfg.scheduler.buffer_capacity_mb = max(1.0, args.buffer_mb)
    cfg.scheduler.relay_enabled = not args.no_relay
    cfg.live.enabled = args.live or args.offline
    cfg.live.offline = args.offline
    if args.zones is not None:
        cfg.live.n_zones = max(1, args.zones)
    cfg.output_dir = str(Path(args.output_dir).resolve()) if args.output_dir else str(ROOT / "outputs")
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    from argus.pipeline import run_pipeline, save_summary
    from argus.report import render_report

    cfg = build_config(args)
    log = (lambda _msg: None) if args.quiet else (lambda msg: print(msg, flush=True))
    t0 = time.perf_counter()
    result = run_pipeline(cfg, log=log, firms_csv=args.firms_csv)
    print(render_report(result))
    json_path = Path(args.json).resolve() if args.json else Path(cfg.output_dir) / "summary.json"
    save_summary(result, json_path)
    print(f"\n  Outputs: models in {Path(cfg.output_dir) / 'models'} | JSON summary {json_path}")
    print(f"  Total runtime {time.perf_counter() - t0:.1f}s")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    import math

    from argus import live
    from argus.report import table

    hs, feeds = live.load_firms(offline=args.offline, log=print)
    for k, v in feeds.items():
        print(f"  NASA FIRMS {k}: {v}")
    if not hs:
        print("  no FIRMS data available")
        return 1
    cities = live.load_cities(offline=args.offline, log=print)
    clusters = live.cluster_hotspots(hs)
    live.attach_exposure(clusters, cities, top_n=1000)
    print(f"  {len(hs):,} hotspots -> {len(clusters):,} fire clusters | population: {cities.provenance}")
    print("\n  Strongest fires (fire radiative power):")
    print(table(["fire", "lat", "lon", "hotspots", "FRP MW", "nearest city", "dist km", "WUI risk"],
                [[f"#{c.cluster_id}", f"{c.lat:.2f}", f"{c.lon:.2f}", c.n_hotspots, c.frp_total_mw, c.nearest_city[:30],
                  c.city_dist_km if math.isfinite(c.city_dist_km) else "-", c.wui_risk] for c in clusters[:args.top]]))
    exposed = sorted(clusters[:1000], key=lambda c: -(c.wui_risk * math.log1p(c.frp_total_mw)))
    print("\n  Fires threatening people (WUI risk x intensity):")
    print(table(["fire", "lat", "lon", "FRP MW", "nearest city", "dist km", "exposed pop", "WUI risk"],
                [[f"#{c.cluster_id}", f"{c.lat:.2f}", f"{c.lon:.2f}", c.frp_total_mw, c.nearest_city[:30],
                  c.city_dist_km if math.isfinite(c.city_dist_km) else "-", int(c.exposed_pop), c.wui_risk]
                 for c in exposed[:args.top]]))
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from argus import live
    from argus.segmentation import REAL_DATASET_PATH
    from argus.sentinel2 import RealDataset, build_real_dataset, merge_datasets

    t0 = time.perf_counter()
    hs, feeds = live.load_firms(feeds=live.FIRMS_ARCHIVE_FEEDS, max_age_h=12, log=print)
    for k, v in feeds.items():
        print(f"  NASA FIRMS {k}: {v}")
    fire_days = live.cluster_by_day(hs)
    live.attach_exposure(fire_days, live.load_cities(log=print), top_n=args.candidates * 4)
    ds = build_real_dataset(fire_days, hs, target_chips=args.target, max_candidates=args.candidates,
                            workers=args.workers)
    old = RealDataset.load(REAL_DATASET_PATH) if args.append else None
    if old is not None:
        ds = merge_datasets(old, ds)
    ds.save(REAL_DATASET_PATH)
    print(f"  saved {len(ds)} chips ({sum(m['fire_px'] > 0 for m in ds.meta)} with active fire, "
          f"{len({m['scene'] for m in ds.meta})} scenes) to {REAL_DATASET_PATH}.npz/.json "
          f"in {time.perf_counter() - t0:.0f}s")
    return 0


def cmd_dashboard() -> int:
    if not deps.STREAMLIT_AVAILABLE:
        print("Streamlit is not installed. Install it with:  pip install streamlit pandas")
        return 1
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "argus_dashboard.py")]
    print("Launching:", " ".join(cmd))
    return subprocess.call(cmd)


def main(argv: list[str] | None = None) -> int:
    try:  # keep the report printable on legacy Windows consoles
        sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass
    for cat in (UserWarning, DeprecationWarning, FutureWarning):
        warnings.filterwarnings("ignore", category=cat)
    args = parse_args(argv)
    if args.command == "live":
        return cmd_live(args)
    if args.command == "build-dataset":
        return cmd_build_dataset(args)
    if args.command == "dashboard":
        return cmd_dashboard()
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
