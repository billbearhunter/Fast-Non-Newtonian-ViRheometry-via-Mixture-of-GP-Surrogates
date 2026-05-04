"""Generate a 5D-uniform sub-sample of the existing train+val+test pool.

For each occupied cell of a B**5 grid (B=4 default) with at least Q points
(default 50), randomly sample exactly Q points. Cells with <Q are dropped.

Splits the result into train/val/test using stratified sampling within each
cell so all three sets have the same uniform 5D coverage.

Output: pipeline_v10/data/synthetic_splits_uniform_b4q50/{train,val,test}_uniform.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4, help="bins per axis")
    ap.add_argument("--Q", type=int, default=50, help="per-cell quota")
    ap.add_argument("--train-frac", type=float, default=0.80)
    ap.add_argument("--val-frac",   type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path,
                    default=PIPE / "data" / "synthetic_splits_uniform_b4q50")
    a = ap.parse_args()

    src = PIPE / "data" / "synthetic_splits_v2"
    parts = []
    for fn in ["train_v10_v2.csv", "val_v10_v2.csv", "test_v10_v2.csv"]:
        p = src / fn
        if p.is_file():
            d = pd.read_csv(p)
            d["_orig_split"] = fn.split("_")[0]   # 'train', 'val', 'test'
            parts.append(d)
            print(f"  loaded {fn}: {len(d):,} rows")
    df = pd.concat(parts, ignore_index=True)
    print(f"\nFull pool: {len(df):,} points")

    # 5D bin assignment
    feats = [
        df.n.to_numpy(),
        np.log(df.eta.clip(1e-12).to_numpy()),
        np.log(df.sigma_y.clip(1e-12).to_numpy()),
        df.width.to_numpy(),
        df.height.to_numpy(),
    ]
    edges = [np.linspace(f.min(), f.max() + 1e-9, a.B + 1) for f in feats]
    bin_idx = np.zeros((len(df), 5), dtype=np.int32)
    for d, (f, e) in enumerate(zip(feats, edges)):
        bin_idx[:, d] = np.digitize(f, e[1:-1])
    flat = np.zeros(len(df), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= a.B
    df["_cell"] = flat

    rng = np.random.default_rng(a.seed)

    # Sample Q per cell, stratify into train/val/test
    train_rows = []
    val_rows = []
    test_rows = []
    n_train = int(round(a.Q * a.train_frac))
    n_val = int(round(a.Q * a.val_frac))
    n_test = a.Q - n_train - n_val

    cells_used = 0
    for cell, gdf in df.groupby("_cell"):
        if len(gdf) < a.Q:
            continue
        idx = rng.permutation(len(gdf))[:a.Q]
        chunk = gdf.iloc[idx].reset_index(drop=True)
        train_rows.append(chunk.iloc[:n_train])
        val_rows.append(chunk.iloc[n_train: n_train + n_val])
        test_rows.append(chunk.iloc[n_train + n_val:])
        cells_used += 1

    train = pd.concat(train_rows, ignore_index=True).drop(columns=["_cell", "_orig_split"])
    val   = pd.concat(val_rows,   ignore_index=True).drop(columns=["_cell", "_orig_split"])
    test  = pd.concat(test_rows,  ignore_index=True).drop(columns=["_cell", "_orig_split"])
    train["split"] = "train"
    val["split"]   = "val"
    test["split"]  = "test"

    a.out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(a.out_dir / "train_uniform.csv", index=False)
    val.to_csv(a.out_dir / "val_uniform.csv", index=False)
    test.to_csv(a.out_dir / "test_uniform.csv", index=False)

    print(f"\n=== B={a.B}, Q={a.Q} ===")
    print(f"  cells used (>= Q):  {cells_used:,} / {a.B ** 5:,}")
    print(f"  train: {len(train):,}  ({n_train}/cell × {cells_used} cells)")
    print(f"  val:   {len(val):,}    ({n_val}/cell × {cells_used} cells)")
    print(f"  test:  {len(test):,}   ({n_test}/cell × {cells_used} cells)")
    print(f"  total: {len(train) + len(val) + len(test):,}")
    print(f"\nwrote to {a.out_dir}")


if __name__ == "__main__":
    main()
