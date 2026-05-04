"""Merge per-machine worker outputs into a unified infill pool, then
optionally regenerate the uniform train/val/test split combining old
data + new infill.

Usage:
  # Merge all machines' outputs into one CSV
  python merge_worker_outputs.py \\
      --inputs outputs/machine0/infill_machine0.csv \\
               outputs/machine1/infill_machine1.csv \\
               outputs/machine2/infill_machine2.csv \\
      --out final/infill_pool.csv

  # Then regenerate uniform split with old+new combined
  python merge_worker_outputs.py --regenerate-splits \\
      --infill-csv final/infill_pool.csv \\
      --B 4 --Q 100 --out-dir final/uniform_b4q100_v2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]


def merge_inputs(inputs: list[Path], out: Path):
    parts = []
    for p in inputs:
        if not p.is_file():
            print(f"  [missing] {p}")
            continue
        df = pd.read_csv(p)
        parts.append(df)
        print(f"  + {p.name}: {len(df):,} rows")
    if not parts:
        print("no inputs found.")
        return
    merged = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"\nmerged: {len(merged):,} rows → {out}")


def regenerate_splits(infill_csv: Path, B: int, Q: int, out_dir: Path,
                       train_frac: float = 0.80, val_frac: float = 0.10,
                       seed: int = 42):
    src = PIPE / "data" / "synthetic_splits_v2"
    parts = []
    for fn in ["train_v10_v2.csv", "val_v10_v2.csv", "test_v10_v2.csv"]:
        p = src / fn
        if p.is_file():
            parts.append(pd.read_csv(p))
    pool = pd.concat(parts, ignore_index=True)
    print(f"old pool: {len(pool):,} rows")

    if infill_csv.is_file():
        infill = pd.read_csv(infill_csv)
        keep = [c for c in pool.columns if c in infill.columns]
        infill = infill[keep]
        pool = pd.concat([pool, infill], ignore_index=True)
        print(f"+ infill: {len(infill):,}")
        print(f"combined: {len(pool):,}")

    feats = {
        "n":     pool.n.to_numpy(),
        "log_eta": np.log(pool.eta.clip(1e-12).to_numpy()),
        "log_sy":  np.log(pool.sigma_y.clip(1e-12).to_numpy()),
        "W":     pool.width.to_numpy(),
        "H":     pool.height.to_numpy(),
    }
    edges = {k: np.linspace(v.min(), v.max() + 1e-9, B + 1) for k, v in feats.items()}
    bin_idx = np.zeros((len(pool), 5), dtype=np.int32)
    for d, k in enumerate(["n", "log_eta", "log_sy", "W", "H"]):
        bin_idx[:, d] = np.digitize(feats[k], edges[k][1:-1])
    flat = np.zeros(len(pool), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= B
    pool["_cell"] = flat

    rng = np.random.default_rng(seed)
    n_train = int(round(Q * train_frac))
    n_val = int(round(Q * val_frac))
    n_test = Q - n_train - n_val

    train_rows, val_rows, test_rows = [], [], []
    cells_used = 0
    for cell, gdf in pool.groupby("_cell"):
        if len(gdf) < Q:
            continue
        idx = rng.permutation(len(gdf))[:Q]
        chunk = gdf.iloc[idx].reset_index(drop=True)
        train_rows.append(chunk.iloc[:n_train])
        val_rows.append(chunk.iloc[n_train: n_train + n_val])
        test_rows.append(chunk.iloc[n_train + n_val:])
        cells_used += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        if not rows:
            continue
        df = pd.concat(rows, ignore_index=True).drop(columns=["_cell"])
        df["split"] = name
        df.to_csv(out_dir / f"{name}_uniform.csv", index=False)
    print(f"\ncells used: {cells_used:,}")
    print(f"train: {sum(len(r) for r in train_rows):,}")
    print(f"val:   {sum(len(r) for r in val_rows):,}")
    print(f"test:  {sum(len(r) for r in test_rows):,}")
    print(f"\nwrote to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", type=Path, default=None,
                    help="Per-machine output CSVs to merge")
    ap.add_argument("--out", type=Path, default=None,
                    help="Merged infill pool CSV path")
    ap.add_argument("--regenerate-splits", action="store_true")
    ap.add_argument("--infill-csv", type=Path, default=None,
                    help="Combined infill CSV (output of --inputs merge)")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--Q", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write the new uniform splits")
    a = ap.parse_args()

    if a.inputs:
        if a.out is None:
            ap.error("--out required when using --inputs")
        merge_inputs(a.inputs, a.out)

    if a.regenerate_splits:
        if a.infill_csv is None or a.out_dir is None:
            ap.error("--infill-csv and --out-dir required for --regenerate-splits")
        regenerate_splits(a.infill_csv, a.B, a.Q, a.out_dir)


if __name__ == "__main__":
    main()
