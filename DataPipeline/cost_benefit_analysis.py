"""Cost-benefit curves for different infill strategies.

Question: instead of strict "B=4 Q=100 fill all 866 non-empty cells = 40k new sims",
can we get to a usable uniform pool with fewer sims?

Strategies analyzed:
  S1 (cheap): "include densest cells first"
              For target uniform total N at quota Q, sort cells by current count
              descending, include cells until K*Q >= N.
              Minimizes new sims but helps ONLY main-diagonal materials.

  S2 (corner-friendly): "include sparsest non-empty cells first"
              Sort cells by current count ASCENDING, include cells until K*Q >= N.
              Costs more new sims but fixes corner materials.

  S3 (current plan): "fill ALL non-empty cells to Q"
              All 866 cells included. Highest cost but most coverage.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]


def load_cell_counts(B: int) -> np.ndarray:
    parts = []
    for fn in ["train_v10_v2.csv", "val_v10_v2.csv", "test_v10_v2.csv"]:
        p = PIPE / "data" / "synthetic_splits_v2" / fn
        if p.is_file():
            parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    feats = [
        df.n.to_numpy(),
        np.log(df.eta.clip(1e-12).to_numpy()),
        np.log(df.sigma_y.clip(1e-12).to_numpy()),
        df.width.to_numpy(),
        df.height.to_numpy(),
    ]
    edges = [np.linspace(f.min(), f.max() + 1e-9, B + 1) for f in feats]
    bin_idx = np.zeros((len(df), 5), dtype=np.int32)
    for d, (f, e) in enumerate(zip(feats, edges)):
        bin_idx[:, d] = np.digitize(f, e[1:-1])
    flat = np.zeros(len(df), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= B
    return np.bincount(flat, minlength=B ** 5)


def evaluate(counts: np.ndarray, Q: int, target_total: int, strategy: str) -> dict:
    """Compute min new sims to achieve `target_total` uniform points at Q per cell."""
    nonempty = counts[counts > 0]
    if strategy == "cheap":
        # cells with most points first (small or zero deficit)
        order = np.argsort(-counts)
    elif strategy == "corner":
        # cells with fewest points first (large deficit, but covers corners)
        # but must have at least 1 point (avoid empty cells where sim may fail)
        positive = np.where(counts > 0)[0]
        order = positive[np.argsort(counts[positive])]
    else:  # current plan: include ALL nonempty
        order = np.where(counts > 0)[0]

    cumulative_uniform = 0
    cumulative_new_sims = 0
    cells_used = 0
    for c in order:
        cur = int(counts[c])
        if cur == 0 and strategy == "cheap":
            break  # don't fill empty cells
        deficit = max(0, Q - cur)
        cumulative_new_sims += deficit
        cumulative_uniform += Q
        cells_used += 1
        if cumulative_uniform >= target_total and strategy != "current_all":
            break
    return {
        "Q": Q, "target_total": target_total, "strategy": strategy,
        "cells_used": cells_used,
        "uniform_total": cumulative_uniform,
        "new_sims": cumulative_new_sims,
        "wall_h_per_machine_x3": cumulative_new_sims * 2.0 / 3 / 3600,
        "wall_h_master_3x_workers_1x": cumulative_new_sims * 2.0 * 0.6 / 3.0 / 3600,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    a = ap.parse_args()

    counts = load_cell_counts(a.B)
    nonempty = (counts > 0).sum()
    total_pool = counts.sum()
    print(f"B={a.B} grid: {a.B**5:,} cells, {nonempty:,} non-empty, total {total_pool:,} pts")
    print()

    rows = []
    for Q in [30, 50, 75, 100, 150]:
        for tgt in [50_000, 100_000, 150_000]:
            for strat in ["cheap", "corner"]:
                rows.append(evaluate(counts, Q, tgt, strat))
        # also "fill all" for comparison
        rows.append({**evaluate(counts, Q, 10**12, "current_all"),
                       "target_total": -1})

    df = pd.DataFrame(rows)

    print("=" * 110)
    print("STRATEGY: 'cheap' = include cells with HIGHEST current_count first (min sims)")
    print("STRATEGY: 'corner' = include cells with LOWEST current_count first (corners)")
    print("STRATEGY: 'current_all' = include ALL 866 nonempty cells (current plan)")
    print("=" * 110)

    print(f"\n{'Q':>4} {'target':>9} {'strategy':>12} {'cells':>6} {'uniform_actual':>15} {'new_sims':>10} {'wall_h(3x@2s)':>14}")
    print("-" * 110)
    for _, r in df.iterrows():
        tgt_str = f"{r.target_total:,}" if r.target_total > 0 else "ALL"
        print(f"{r.Q:>4} {tgt_str:>9} {r.strategy:>12}  "
              f"{r.cells_used:>5,} {r.uniform_total:>15,} "
              f"{r.new_sims:>10,} {r.wall_h_per_machine_x3:>13.1f}")

    # Headline comparison: current plan vs cheaper alternatives at same target
    print()
    print("=" * 110)
    print("HEADLINE - at target ~100k uniform")
    print("=" * 110)
    print(f"{'plan':>50} {'sims':>8} {'wall_h':>8} {'fixes corners?':>15}")
    print("-" * 110)
    plans = [
        ("Q=100, fill ALL 866 nonempty cells (current)", 100, -1, "current_all"),
        ("Q=100, fill DENSEST cells to 100k uniform",     100, 100_000, "cheap"),
        ("Q=100, fill SPARSEST cells to 100k uniform",    100, 100_000, "corner"),
        ("Q=75,  fill DENSEST cells to 100k uniform",      75, 100_000, "cheap"),
        ("Q=75,  fill SPARSEST cells to 100k uniform",     75, 100_000, "corner"),
        ("Q=50,  fill DENSEST cells to 100k uniform",      50, 100_000, "cheap"),
        ("Q=50,  fill SPARSEST cells to 100k uniform",     50, 100_000, "corner"),
        ("Q=100, fill DENSEST cells to 150k uniform",     100, 150_000, "cheap"),
        ("Q=100, fill SPARSEST cells to 150k uniform",    100, 150_000, "corner"),
    ]
    for label, Q, tgt, strat in plans:
        r = evaluate(counts, Q, tgt, strat)
        if strat == "cheap":
            corner_fix = "no - main diagonal only"
        elif strat == "corner":
            corner_fix = "yes - covers low-sy cells"
        else:
            corner_fix = "yes - full coverage"
        print(f"{label:>50}  {r['new_sims']:>7,}  {r['wall_h_per_machine_x3']:>6.1f}h   {corner_fix}")


if __name__ == "__main__":
    main()
