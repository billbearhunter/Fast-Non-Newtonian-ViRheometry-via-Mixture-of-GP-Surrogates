"""Regenerate cell_plan.csv for a NEW target Q, accounting for what's
already been completed by workers.

Why this is needed: the original `generate_cell_plan.py` builds the
deficit only from the static `synthetic_splits_v2/{train,val,test}` pool.
After Round-2 starts, workers add new sims that aren't yet in those CSVs.
If you want to upgrade the target from Q=100 to Q=150, the new deficit
per cell should be:
    new_deficit = max(0, Q_new - (original_count + worker_count))

This script does exactly that: combines pool + worker outputs into a
single counts-per-cell, then emits a fresh plan at the new Q.

Usage:
    python regen_plan_for_target_Q.py \\
        --Q 150 \\
        --machines 3 --machine-weights 3,1,1 \\
        --worker-csvs outputs/machine0/infill_machine0.csv \\
                       outputs/machine0/infill_machine1.csv \\
                       outputs/machine0/infill_machine2.csv \\
        --out plan/cell_plan_q150.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--Q", type=int, required=True,
                    help="new target points-per-cell (e.g. 150)")
    ap.add_argument("--machines", type=int, default=3)
    ap.add_argument("--machine-weights", type=str, default=None,
                    help="comma-separated weights, e.g. '3,1,1'")
    ap.add_argument("--worker-csvs", nargs="+", type=Path, default=None,
                    help="worker output CSVs to count toward already-done")
    ap.add_argument("--src-dir", type=Path,
                    default=PIPE / "data" / "synthetic_splits_v2")
    ap.add_argument("--out", type=Path, required=True,
                    help="output cell_plan*.csv path")
    a = ap.parse_args()

    # Load synthetic_splits + worker outputs
    parts = []
    for fn in ["train_v10_v2.csv", "val_v10_v2.csv", "test_v10_v2.csv"]:
        p = a.src_dir / fn
        if p.is_file():
            parts.append(pd.read_csv(p))
    pool = pd.concat(parts, ignore_index=True)
    print(f"loaded synthetic_splits: {len(pool):,}")

    if a.worker_csvs:
        keep = ["n", "eta", "sigma_y", "width", "height"]
        for c in a.worker_csvs:
            d = pd.read_csv(c)
            d = d[[k for k in keep if k in d.columns]]
            pool = pd.concat([pool, d], ignore_index=True)
            print(f"  + {c.name}: {len(d):,}")
    print(f"combined pool: {len(pool):,}")

    # Bin
    feats = {
        "n":     pool.n.to_numpy(),
        "log_eta": np.log(pool.eta.clip(1e-12).to_numpy()),
        "log_sy":  np.log(pool.sigma_y.clip(1e-12).to_numpy()),
        "W":     pool.width.to_numpy(),
        "H":     pool.height.to_numpy(),
    }
    edges = {k: np.linspace(v.min(), v.max() + 1e-9, a.B + 1)
             for k, v in feats.items()}
    bin_idx = np.zeros((len(pool), 5), dtype=np.int32)
    for d, k in enumerate(["n", "log_eta", "log_sy", "W", "H"]):
        bin_idx[:, d] = np.digitize(feats[k], edges[k][1:-1])
    flat = np.zeros(len(pool), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= a.B
    counts = np.bincount(flat, minlength=a.B ** 5)

    # Build plan
    PAD, Z, DX, SAMPLES = 0.5, 4.15, 0.126, 2

    def particles(W, H):
        return ((W + PAD) * (H + PAD) * Z) / (DX ** 3) * (SAMPLES ** 3)

    rows = []
    n_cells = a.B ** 5
    for cell in range(n_cells):
        idx = np.array([(cell // (a.B ** d)) % a.B for d in range(5)],
                        dtype=np.int32)
        cur = int(counts[cell])
        if cur == 0:
            continue   # don't fill empty cells
        deficit = max(0, a.Q - cur)
        if deficit == 0:
            continue
        W_mid = 0.5 * (edges["W"][idx[3]] + edges["W"][idx[3] + 1])
        H_mid = 0.5 * (edges["H"][idx[4]] + edges["H"][idx[4] + 1])
        est_p = particles(W_mid, H_mid)
        rows.append({
            "cell_id": cell,
            "i_n": int(idx[0]), "i_log_eta": int(idx[1]),
            "i_log_sy": int(idx[2]), "i_W": int(idx[3]), "i_H": int(idx[4]),
            "n_lo": float(edges["n"][idx[0]]),
            "n_hi": float(edges["n"][idx[0] + 1]),
            "log_eta_lo": float(edges["log_eta"][idx[1]]),
            "log_eta_hi": float(edges["log_eta"][idx[1] + 1]),
            "log_sy_lo": float(edges["log_sy"][idx[2]]),
            "log_sy_hi": float(edges["log_sy"][idx[2] + 1]),
            "W_lo": float(edges["W"][idx[3]]),
            "W_hi": float(edges["W"][idx[3] + 1]),
            "H_lo": float(edges["H"][idx[4]]),
            "H_hi": float(edges["H"][idx[4] + 1]),
            "current_n": cur,
            "target_n": a.Q,
            "deficit": deficit,
            "est_particles": float(est_p),
            "est_cost": float(deficit * est_p),
        })
    plan = pd.DataFrame(rows)
    print(f"\ncells needing infill: {len(plan)} / {(counts > 0).sum()} non-empty")
    print(f"total deficit (sims): {plan.deficit.sum():,}")

    # Greedy bin-pack
    if a.machine_weights:
        weights = np.array([float(x) for x in a.machine_weights.split(",")],
                            dtype=np.float64)
    else:
        weights = np.ones(a.machines, dtype=np.float64)
    weight_share = weights / weights.sum()
    target_cost = plan.est_cost.sum() * weight_share

    plan = plan.sort_values("est_cost", ascending=False).reset_index(drop=True)
    machine_loads = np.zeros(a.machines, dtype=np.float64)
    machine_assign = np.zeros(len(plan), dtype=np.int32)
    for i, row in plan.iterrows():
        deficit_ratio = (target_cost - machine_loads) / np.maximum(target_cost, 1.0)
        m = int(np.argmax(deficit_ratio))
        machine_assign[i] = m
        machine_loads[m] += float(row.est_cost)
    plan["machine_id"] = machine_assign

    plan = plan.sort_values(["machine_id", "cell_id"]).reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(a.out, index=False)
    print(f"\nwrote: {a.out}")
    print(f"per-machine load (weights={list(weights)}):")
    base_p = 350_000
    for m in range(a.machines):
        sub = plan[plan.machine_id == m]
        avg_p = sub.est_particles.mean()
        ref_h = sub.deficit.sum() * 2.0 * (avg_p / base_p) / 3600
        est_h = ref_h / weights[m]
        print(f"  m{m} (w={weights[m]:.1f}): "
              f"{len(sub):>4} cells / {sub.deficit.sum():>6,} sims / "
              f"~{est_h:.1f}h on this machine")


if __name__ == "__main__":
    main()
