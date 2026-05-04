#!/usr/bin/env python3
"""Build y_obs.npy from flow_distances.csv.

The downstream optimizer (Optimization/setup1.py, scripts/run_rw_experiments.py)
expects each ref directory to contain:
    y_obs.npy   : float64 array, shape (8,), max-particle x distance in cm
                  for frames 1..8.

This script reads <ref_dir>/flow_distances.csv (produced by
Calibration/extract_flow_distance.py) and writes <ref_dir>/y_obs.npy.

Usage
-----
    # Single ref directory
    python -m Calibration.build_y_obs --dir data/new_real_world_experiments/ref_Chuno_2.7_2.5_1

    # Batch over a parent directory (each subdir with flow_distances.csv)
    python -m Calibration.build_y_obs --root data/new_real_world_experiments
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COL = "distance_cm"
REQUIRED_FRAMES = list(range(1, 9))  # 1..8 inclusive


def build_one(ref_dir: Path, *, overwrite: bool = False) -> Path | None:
    csv_path = ref_dir / "flow_distances.csv"
    out_path = ref_dir / "y_obs.npy"
    if not csv_path.is_file():
        print(f"[skip] {ref_dir.name}: no flow_distances.csv", file=sys.stderr)
        return None
    if out_path.is_file() and not overwrite:
        print(f"[skip] {ref_dir.name}: y_obs.npy already exists (use --overwrite)", file=sys.stderr)
        return None

    df = pd.read_csv(csv_path)
    if REQUIRED_COL not in df.columns:
        raise ValueError(f"{csv_path}: missing column '{REQUIRED_COL}'; got {list(df.columns)}")
    if "frame_index" not in df.columns:
        raise ValueError(f"{csv_path}: missing column 'frame_index'")

    df = df.set_index("frame_index").sort_index()
    missing = [f for f in REQUIRED_FRAMES if f not in df.index]
    if missing:
        raise ValueError(f"{csv_path}: missing frames {missing}; have {list(df.index)}")

    y = df.loc[REQUIRED_FRAMES, REQUIRED_COL].to_numpy(dtype=np.float64)
    if y.shape != (8,):
        raise ValueError(f"{csv_path}: expected 8 frames, got shape {y.shape}")

    np.save(out_path, y)
    print(f"[ok] {ref_dir.name}: wrote y_obs.npy  values={np.array2string(y, precision=4)}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", type=Path, help="Single ref dir containing flow_distances.csv")
    g.add_argument("--root", type=Path, help="Parent dir; process every subdir with flow_distances.csv")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing y_obs.npy")
    args = ap.parse_args()

    if args.dir is not None:
        build_one(args.dir, overwrite=args.overwrite)
        return

    root: Path = args.root
    if not root.is_dir():
        sys.exit(f"--root not a directory: {root}")
    n_ok = n_skip = 0
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        result = build_one(d, overwrite=args.overwrite)
        if result is None:
            n_skip += 1
        else:
            n_ok += 1
    print(f"\nDone. wrote={n_ok}, skipped={n_skip}")


if __name__ == "__main__":
    main()
