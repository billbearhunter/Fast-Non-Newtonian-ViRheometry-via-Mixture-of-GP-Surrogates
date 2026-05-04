"""Regenerate a cell_plan reflecting only the work that's still pending,
counting completion across ALL workers' output CSVs.

Use case: machine 1 finishes its 8k assigned sims early, you want it to
help master. Run this to produce `cell_plan_remaining.csv` covering only
unfilled deficit, then start the free machine on that new plan.

Usage:
  # After machine 1 finishes:
  python regen_remaining_plan.py \\
      --plan plan/cell_plan.csv \\
      --done-csvs outputs/machine0/infill_machine0.csv \\
                  outputs/machine1/infill_machine1.csv \\
      --machines 1 \\
      --out plan/cell_plan_remaining.csv

  # Then on the free machine:
  python run_worker.py --machine-id 0 \\
      --plan plan/cell_plan_remaining.csv \\
      --out-dir outputs/machine1_helper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True,
                    help="original cell_plan.csv")
    ap.add_argument("--done-csvs", nargs="+", type=Path, required=True,
                    help="all worker output CSVs to count toward completion")
    ap.add_argument("--machines", type=int, default=1,
                    help="number of free machines to assign remaining work to")
    ap.add_argument("--machine-weights", type=str, default=None,
                    help="comma-separated weights, e.g. '1,1' for 2 free machines")
    ap.add_argument("--out", type=Path, required=True,
                    help="output cell_plan_remaining.csv path")
    ap.add_argument("--skip-cells-touched-in", nargs="*", type=Path, default=None,
                    help="CSV(s) whose cell_ids should be EXCLUDED from the new plan "
                         "(reserve those cells for the original worker — avoids "
                         "duplication when a free worker takes over master's queue).")
    ap.add_argument("--include-machine-ids", nargs="+", type=int, default=None,
                    help="Only consider cells originally assigned to these machine_ids "
                         "in the input plan. E.g. '--include-machine-ids 0' lets a free "
                         "worker take ONLY from master's slice, leaving other machines' "
                         "untouched cells alone.")
    a = ap.parse_args()

    plan = pd.read_csv(a.plan)
    print(f"loaded plan: {len(plan)} cells, total deficit {plan.deficit.sum():,}")

    # Optional: filter to specific machine_ids' cells before doing anything else
    if a.include_machine_ids:
        keep = set(int(m) for m in a.include_machine_ids)
        before = len(plan)
        plan = plan[plan.machine_id.isin(keep)].copy().reset_index(drop=True)
        print(f"  filtered to machine_ids {sorted(keep)}: "
              f"{len(plan)} cells (from {before})")

    # Aggregate completion across all done CSVs
    per_cell_done = pd.Series(0, index=plan.cell_id, name="done_n", dtype=int)
    for csv in a.done_csvs:
        if not csv.is_file():
            print(f"  [missing] {csv}")
            continue
        try:
            df = pd.read_csv(csv, usecols=["cell_id"])
            counts = df.cell_id.value_counts()
            for cid, n in counts.items():
                if cid in per_cell_done.index:
                    per_cell_done.loc[cid] += int(n)
            print(f"  + {csv.name}: {len(df):,} rows ({len(counts)} unique cells)")
        except Exception as e:
            print(f"  [error reading {csv.name}: {e}]")

    plan = plan.merge(per_cell_done.reset_index(), on="cell_id", how="left").fillna(0)
    plan["done_n"] = plan["done_n"].astype(int)
    plan["deficit_remaining"] = (plan.deficit - plan.done_n).clip(lower=0).astype(int)

    print(f"\ntotal completed: {plan.done_n.sum():,}")
    print(f"total remaining: {plan.deficit_remaining.sum():,}")
    print(f"cells fully done: {(plan.deficit_remaining == 0).sum()} / {len(plan)}")
    print(f"cells partially: {((plan.deficit_remaining > 0) & (plan.done_n > 0)).sum()}")
    print(f"cells untouched: {(plan.done_n == 0).sum()}")

    # Drop fully-done cells; keep only those needing more sims
    remain = plan[plan.deficit_remaining > 0].copy()
    if len(remain) == 0:
        print("\nNothing left to do!")
        return

    # Optionally exclude cells that are currently being worked on by another
    # worker (so the helper picks ONLY untouched cells, leaving in-progress
    # ones for the original worker to finish).
    if a.skip_cells_touched_in:
        excluded_cells = set()
        for csv in a.skip_cells_touched_in:
            if not csv.is_file():
                continue
            try:
                df = pd.read_csv(csv, usecols=["cell_id"])
                touched = set(df.cell_id.unique().tolist())
                excluded_cells.update(touched)
                print(f"\n  reserving {len(touched)} cells from {csv.name} "
                      f"(in-progress by original worker)")
            except Exception as e:
                print(f"  [error reading {csv.name}: {e}]")
        before = len(remain)
        remain = remain[~remain.cell_id.isin(excluded_cells)].copy()
        print(f"  excluded {before - len(remain)} in-progress cells; "
              f"{len(remain)} cells available for helper")
        if len(remain) == 0:
            print("\nNo untouched cells available! Original worker still has work; "
                  "either wait or let helper join master's queue with possible overlap.")
            return

    # Re-cost based on remaining deficit
    remain["deficit"] = remain.deficit_remaining
    remain["est_cost"] = remain.deficit * remain.est_particles

    # Greedy bin-pack onto free machines
    if a.machine_weights:
        weights = np.array([float(x) for x in a.machine_weights.split(",")],
                            dtype=np.float64)
        if len(weights) != a.machines:
            ap.error(f"--machine-weights must have {a.machines} values")
    else:
        weights = np.ones(a.machines, dtype=np.float64)
    weight_share = weights / weights.sum()
    target_cost = remain.est_cost.sum() * weight_share

    remain = remain.sort_values("est_cost", ascending=False).reset_index(drop=True)
    machine_loads = np.zeros(a.machines, dtype=np.float64)
    new_assign = np.zeros(len(remain), dtype=np.int32)
    for i, row in remain.iterrows():
        deficit_ratio = (target_cost - machine_loads) / np.maximum(target_cost, 1.0)
        m = int(np.argmax(deficit_ratio))
        new_assign[i] = m
        machine_loads[m] += float(row.est_cost)
    remain["machine_id"] = new_assign

    # Drop helper cols; mirror original plan format
    keep_cols = [c for c in plan.columns
                  if c not in ("done_n", "deficit_remaining")]
    remain = remain[keep_cols]
    remain = remain.sort_values(["machine_id", "cell_id"]).reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    remain.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")

    print(f"\nper-machine remaining load:")
    for m in range(a.machines):
        sub = remain[remain.machine_id == m]
        ref_hours = sub.deficit.sum() * 2.0 * (sub.est_particles.mean() / 350_000) / 3600
        est_hours = ref_hours / weights[m]
        print(f"  machine {m} (weight {weights[m]:.1f}): "
              f"{len(sub):>4} cells  "
              f"{sub.deficit.sum():>6,} sims  "
              f"~{est_hours:.1f}h")


if __name__ == "__main__":
    main()
