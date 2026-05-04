"""Sub-sample current train+val+test pool to a 5D-uniform dataset
without adding any new sims. Report the resulting size for several
grid resolutions and per-cell quotas.

5D axes: n, log eta, log sigma_y, W, H
The function reports how many cells are non-empty at each resolution,
and what dataset size we'd retain at various per-cell quotas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]


def load_pool() -> pd.DataFrame:
    """Concat all train + val + test (clean and v2)."""
    paths = [
        PIPE / "data" / "synthetic_splits_v2" / "train_v10_v2.csv",
        PIPE / "data" / "synthetic_splits_v2" / "val_v10_v2.csv",
        PIPE / "data" / "synthetic_splits_v2" / "test_v10_v2.csv",
    ]
    parts = []
    for p in paths:
        if p.is_file():
            df = pd.read_csv(p)
            df["_src"] = p.stem
            parts.append(df)
            print(f"  loaded {p.name}: {len(df):,} rows")
    df_all = pd.concat(parts, ignore_index=True)
    return df_all


def bin_cell_counts(df: pd.DataFrame, B: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Assign each row to a 5D cell using B bins per axis.

    Bin edges: equal-width on n, log eta, log sigma_y, W, H over the data range.
    Returns: (cell_counts (1D length B^5), edges_per_axis).
    """
    n_v = df.n.to_numpy()
    le_v = np.log(df.eta.clip(1e-12).to_numpy())
    ls_v = np.log(df.sigma_y.clip(1e-12).to_numpy())
    W_v = df.width.to_numpy()
    H_v = df.height.to_numpy()
    feats = [n_v, le_v, ls_v, W_v, H_v]
    edges = [np.linspace(f.min(), f.max() + 1e-9, B + 1) for f in feats]
    bin_idx = np.zeros((len(df), 5), dtype=np.int32)
    for d, (f, e) in enumerate(zip(feats, edges)):
        bin_idx[:, d] = np.digitize(f, e[1:-1])  # values in [0, B-1]
    # flat cell id = sum (bin_idx[:, d] * B^d)
    flat = np.zeros(len(df), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= B
    counts = np.bincount(flat, minlength=B ** 5)
    return counts, edges


def analyze(df: pd.DataFrame, B: int):
    counts, _edges = bin_cell_counts(df, B)
    n_cells = B ** 5
    nonempty = (counts > 0).sum()
    quantiles = np.percentile(counts[counts > 0], [10, 25, 50, 75, 90, 95, 99])
    print(f"\n=== B={B}  (B^5 = {n_cells:,} cells) ===")
    print(f"  cells non-empty:  {nonempty:,} / {n_cells:,}  ({100*nonempty/n_cells:.1f}%)")
    print(f"  cells with >= 1:    {(counts >= 1).sum():,}")
    print(f"  cells with >= 5:    {(counts >= 5).sum():,}")
    print(f"  cells with >= 20:   {(counts >= 20).sum():,}")
    print(f"  cells with >= 50:   {(counts >= 50).sum():,}")
    print(f"  cells with >= 100:  {(counts >= 100).sum():,}")
    print(f"  cells with >= 500:  {(counts >= 500).sum():,}")
    print(f"  per-cell percentiles (non-empty cells):")
    print(f"    p10={quantiles[0]:.0f}, p25={quantiles[1]:.0f}, "
          f"p50={quantiles[2]:.0f}, p75={quantiles[3]:.0f}, "
          f"p90={quantiles[4]:.0f}, p99={quantiles[6]:.0f}")
    print(f"    max cell count: {counts.max():,}")

    # Uniform dataset size at various quotas Q
    print(f"\n  Uniform dataset size at per-cell quota Q (only cells with >=Q kept):")
    print(f"   {'Q':>5}  {'cells_used':>11}  {'total_pts':>11}   '#sims_to_fill_to_Q'")
    for Q in [1, 5, 10, 20, 50, 100, 200, 500, 1000]:
        cells_used = (counts >= Q).sum()
        total_pts = cells_used * Q
        # to fill non-empty cells to Q, need extra = sum(max(0, Q - count))
        deficit = np.maximum(0, Q - counts[counts > 0]).sum()
        print(f"   {Q:>5}  {cells_used:>11,}  {total_pts:>11,}   "
              f"{deficit:>10,} more sims to fill all non-empty cells to {Q}")


def main():
    print("Loading full pool (train + val + test):")
    df = load_pool()
    print(f"\nTotal pool: {len(df):,} points\n")
    print(f"  range n      : [{df.n.min():.3f}, {df.n.max():.3f}]")
    print(f"  range eta    : [{df.eta.min():.3f}, {df.eta.max():.3f}]")
    print(f"  range sigma_y: [{df.sigma_y.min():.3f}, {df.sigma_y.max():.3f}]")
    print(f"  range W      : [{df.width.min():.3f}, {df.width.max():.3f}]")
    print(f"  range H      : [{df.height.min():.3f}, {df.height.max():.3f}]")

    for B in [3, 4, 5, 6]:
        analyze(df, B)


if __name__ == "__main__":
    main()
