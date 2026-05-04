"""Generate the cell-level infill plan for distributed worker machines.

Logic:
  1. Load full pool (train + val + test from synthetic_splits_v2)
  2. Bin into B^5 grid (default B=4) — same axes as diagnose_uniformity.py
  3. For each cell, compute current_count and deficit = max(0, Q - current_count)
  4. Drop cells that are entirely empty (current_count == 0) UNLESS --include-empty
  5. Estimate per-cell compute cost = deficit × estimated_particle_count(W_mid, H_mid)
     - sim wall time scales linearly with particle count
     - W=H=7 cell ~ 12× the cost of W=H=2 cell
  6. Greedy bin-pack cells onto N_MACHINES, balanced by COST (not by deficit).
     If --machine-weights is provided (e.g. "3,2,2"), the strongest machine
     gets a proportionally larger cost share.
  7. Save plan/cell_plan.csv:
        cell_id, parameter ranges, current_n, target_n, deficit,
        est_particles_per_sim, est_cost (= deficit × particles),
        machine_id

Usage:
  # Equal machines:
  python generate_cell_plan.py --B 4 --Q 100 --machines 3

  # Strong master (3×) + 2 weaker workers:
  python generate_cell_plan.py --B 4 --Q 100 --machines 3 --machine-weights 3,1,1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]   # pipeline_v10/


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--Q", type=int, default=100,
                    help="Target points per non-empty cell")
    ap.add_argument("--machines", type=int, default=3,
                    help="Number of worker machines to split across")
    ap.add_argument("--machine-weights", type=str, default=None,
                    help="Comma-separated relative compute weights per machine "
                         "(e.g. '3,1,1' = master is 3x stronger than workers). "
                         "If omitted, all machines treated as equal.")
    ap.add_argument("--include-empty", action="store_true",
                    help="Also include currently empty cells (risky — sim may fail)")
    ap.add_argument("--src-dir", type=Path,
                    default=PIPE / "data" / "synthetic_splits_v2")
    ap.add_argument("--out-csv", type=Path,
                    default=PIPE / "data_infill" / "plan" / "cell_plan.csv")
    a = ap.parse_args()

    # Load full pool
    parts = []
    for fn in ["train_v10_v2.csv", "val_v10_v2.csv", "test_v10_v2.csv"]:
        p = a.src_dir / fn
        if p.is_file():
            parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    print(f"loaded full pool: {len(df):,} pts")

    # Compute axis edges from full pool
    feats = {
        "n":     df.n.to_numpy(),
        "log_eta": np.log(df.eta.clip(1e-12).to_numpy()),
        "log_sy":  np.log(df.sigma_y.clip(1e-12).to_numpy()),
        "W":     df.width.to_numpy(),
        "H":     df.height.to_numpy(),
    }
    edges = {k: np.linspace(v.min(), v.max() + 1e-9, a.B + 1) for k, v in feats.items()}

    # Bin assignment
    bin_idx = np.zeros((len(df), 5), dtype=np.int32)
    for d, k in enumerate(["n", "log_eta", "log_sy", "W", "H"]):
        bin_idx[:, d] = np.digitize(feats[k], edges[k][1:-1])
    flat = np.zeros(len(df), dtype=np.int64)
    mult = 1
    for d in range(5):
        flat += bin_idx[:, d] * mult
        mult *= a.B
    counts = np.bincount(flat, minlength=a.B ** 5)

    # ---- Particle-count cost model ----
    # HeadlessSimulatorMLS allocates fields for the full cuboid:
    #   particles ≈ ((W+pad) * (H+pad) * Z) / dx^3 * samples_per_dim^3
    # Constants from setting.xml: pad=0.5, Z=4.15, dx=0.126, samples=2
    PAD = 0.5
    Z = 4.15
    DX = 0.126
    SAMPLES_PER_DIM = 2
    def particles(W: float, H: float) -> float:
        return ((W + PAD) * (H + PAD) * Z) / (DX ** 3) * (SAMPLES_PER_DIM ** 3)

    # Assemble cell plan
    n_cells = a.B ** 5
    rows = []
    for cell in range(n_cells):
        # decode multi-index (i_n, i_e, i_s, i_W, i_H)
        idx = np.array([(cell // (a.B ** d)) % a.B for d in range(5)], dtype=np.int32)
        cur = int(counts[cell])
        if cur == 0 and not a.include_empty:
            continue
        deficit = max(0, a.Q - cur)
        if deficit == 0:
            continue   # cell already at quota
        W_mid = 0.5 * (edges["W"][idx[3]] + edges["W"][idx[3] + 1])
        H_mid = 0.5 * (edges["H"][idx[4]] + edges["H"][idx[4] + 1])
        est_particles = particles(W_mid, H_mid)
        est_cost = deficit * est_particles      # proportional to wall time
        rows.append({
            "cell_id":      cell,
            "i_n":          int(idx[0]),
            "i_log_eta":    int(idx[1]),
            "i_log_sy":     int(idx[2]),
            "i_W":          int(idx[3]),
            "i_H":          int(idx[4]),
            "n_lo":         float(edges["n"][idx[0]]),
            "n_hi":         float(edges["n"][idx[0] + 1]),
            "log_eta_lo":   float(edges["log_eta"][idx[1]]),
            "log_eta_hi":   float(edges["log_eta"][idx[1] + 1]),
            "log_sy_lo":    float(edges["log_sy"][idx[2]]),
            "log_sy_hi":    float(edges["log_sy"][idx[2] + 1]),
            "W_lo":         float(edges["W"][idx[3]]),
            "W_hi":         float(edges["W"][idx[3] + 1]),
            "H_lo":         float(edges["H"][idx[4]]),
            "H_hi":         float(edges["H"][idx[4] + 1]),
            "current_n":    cur,
            "target_n":     a.Q,
            "deficit":      deficit,
            "est_particles": float(est_particles),
            "est_cost":     float(est_cost),
        })

    plan = pd.DataFrame(rows)
    print(f"\ntotal cells needing infill: {len(plan):,}")
    print(f"total deficit (sims to run): {plan.deficit.sum():,}")
    print(f"total est_cost (sims × particles): {plan.est_cost.sum():,.0f}")
    print(f"  particle-count range per sim: [{plan.est_particles.min():,.0f}, "
          f"{plan.est_particles.max():,.0f}]   "
          f"({plan.est_particles.max()/plan.est_particles.min():.1f}x ratio)")

    # ---- Parse machine weights ----
    if a.machine_weights:
        weights = np.array([float(x) for x in a.machine_weights.split(",")],
                            dtype=np.float64)
        if len(weights) != a.machines:
            ap.error(f"--machine-weights must have {a.machines} values")
    else:
        weights = np.ones(a.machines, dtype=np.float64)
    weight_share = weights / weights.sum()
    target_cost = plan.est_cost.sum() * weight_share   # target per machine

    # ---- Greedy bin-pack by COST, biased by relative capacity ----
    # Sort cells by est_cost descending (largest first → assign to least-saturated machine)
    plan = plan.sort_values("est_cost", ascending=False).reset_index(drop=True)
    machine_loads = np.zeros(a.machines, dtype=np.float64)
    machine_assign = np.zeros(len(plan), dtype=np.int32)
    for i, row in plan.iterrows():
        # pick machine whose current load is FURTHEST below its weighted target
        deficit_ratio = (target_cost - machine_loads) / np.maximum(target_cost, 1.0)
        m = int(np.argmax(deficit_ratio))
        machine_assign[i] = m
        machine_loads[m] += float(row.est_cost)
    plan["machine_id"] = machine_assign

    plan = plan.sort_values(["machine_id", "cell_id"]).reset_index(drop=True)
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(a.out_csv, index=False)
    print(f"\nwrote plan: {a.out_csv}")
    print(f"\nper-machine load:")
    print(f"  weights = {weights.tolist()}  (relative compute capacity)")
    print()
    total_cost = plan.est_cost.sum()
    for m in range(a.machines):
        sub = plan[plan.machine_id == m]
        cost_share = sub.est_cost.sum() / total_cost * 100
        target_share = weight_share[m] * 100
        # Estimate wall time:
        # base = 2 s/sim on a "1x-weight" reference machine doing 350k particles
        # actual time per sim = base * (particles / 350k) / weight
        base_particles_per_sim = 350_000   # mid-range W=H=4.5 reference
        avg_particles = sub.est_particles.mean()
        time_factor = avg_particles / base_particles_per_sim
        ref_hours = sub.deficit.sum() * 2.0 * time_factor / 3600   # if machine were 1x
        est_hours = ref_hours / weights[m]
        print(f"  machine {m} (weight {weights[m]:.1f}): "
              f"{len(sub):>4} cells  "
              f"{sub.deficit.sum():>6,} sims  "
              f"avg {avg_particles/1000:>5.0f}k particles/sim  "
              f"cost share {cost_share:>5.1f}% (target {target_share:.1f}%)  "
              f"~{est_hours:.1f}h on this machine")


if __name__ == "__main__":
    main()
