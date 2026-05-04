"""Worker script: each remote machine runs this with its --machine-id.

Reads cell_plan.csv (filter to assigned machine_id), iterates over assigned
cells, runs LHS within each cell's parameter box, executes MLS-MPM sim per
sample, appends results to a per-machine output CSV. Resumable — already-
finished cells are skipped on restart.

Usage on a worker machine:
  python run_worker.py --machine-id 0 --plan plan/cell_plan.csv --out-dir outputs/machine0
  python run_worker.py --machine-id 1 ...
  python run_worker.py --machine-id 2 ...

Output CSV columns:
  cell_id, n, eta, sigma_y, width, height,
  x_01..x_08 (y_max per frame),
  sim_seconds
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PIPE = Path(__file__).resolve().parents[0]    # pipeline_v10/
INFILL = Path(__file__).resolve().parents[0]  # DataPipeline/ (was data_infill/)
INFILL_DIR = Path(__file__).resolve().parents[0]  # this script's dir

REPO = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(REPO))

from DataPipeline.headless_mls import HeadlessSimulatorMLS  # noqa: E402


def lhs(n: int, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube samples in box [lo, hi]."""
    d = len(lo)
    u = rng.uniform(0.0, 1.0, size=(n, d))
    perm = np.argsort(rng.uniform(0.0, 1.0, size=(n, d)), axis=0)
    u = (perm + u) / n
    return lo[None, :] + (hi[None, :] - lo[None, :]) * u


def already_done(out_csv: Path, cell_id: int) -> int:
    """Return how many rows already exist for this cell."""
    if not out_csv.is_file():
        return 0
    try:
        df = pd.read_csv(out_csv, usecols=["cell_id"])
        return int((df.cell_id == cell_id).sum())
    except Exception:
        return 0


def append_rows(out_csv: Path, header: list[str], rows: list[dict]):
    write_header = not out_csv.is_file()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-id", type=int, required=True)
    ap.add_argument("--plan",       type=Path, required=True,
                    help="path to cell_plan.csv")
    ap.add_argument("--out-dir",    type=Path, required=True,
                    help="per-machine output dir; CSV will be written here")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="Flush rows to disk every N sims (resilience to crashes)")
    ap.add_argument("--max-sims",   type=int, default=0,
                    help="Cap total sims this run (0 = run all assigned)")
    ap.add_argument("--arch",       type=str, default="cuda",
                    help="taichi arch: cuda or cpu")
    a = ap.parse_args()

    plan = pd.read_csv(a.plan)
    plan = plan[plan.machine_id == a.machine_id].reset_index(drop=True)
    if len(plan) == 0:
        print(f"machine {a.machine_id}: no assigned cells")
        return

    a.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = a.out_dir / f"infill_machine{a.machine_id}.csv"
    header = ["cell_id", "n", "eta", "sigma_y", "width", "height",
              "x_01", "x_02", "x_03", "x_04", "x_05", "x_06", "x_07", "x_08",
              "sim_seconds"]

    rng = np.random.default_rng(a.seed + a.machine_id * 1000)

    print(f"machine {a.machine_id}: {len(plan)} cells, "
          f"{plan.deficit.sum():,} sims to run")
    print(f"output → {out_csv}")
    print(f"init MLS-MPM headless simulator (arch={a.arch})...")
    sim = HeadlessSimulatorMLS(arch=a.arch)

    t0 = time.time()
    n_done_total = 0
    sim_budget = a.max_sims if a.max_sims > 0 else int(1e12)

    for ci, cell in plan.iterrows():
        cell_id = int(cell.cell_id)
        deficit = int(cell.deficit)
        existing = already_done(out_csv, cell_id)
        need = max(0, deficit - existing)
        if need == 0:
            continue
        if n_done_total >= sim_budget:
            print(f"  [budget reached: {a.max_sims} sims]")
            break
        need = min(need, sim_budget - n_done_total)

        # Per-cell LHS
        lo = np.array([cell.n_lo, cell.log_eta_lo, cell.log_sy_lo,
                       cell.W_lo, cell.H_lo], dtype=np.float64)
        hi = np.array([cell.n_hi, cell.log_eta_hi, cell.log_sy_hi,
                       cell.W_hi, cell.H_hi], dtype=np.float64)
        cand = lhs(need, lo, hi, rng)
        cand_phys = np.column_stack([
            cand[:, 0],            # n
            np.exp(cand[:, 1]),    # eta
            np.exp(cand[:, 2]),    # sigma_y
            cand[:, 3],            # W
            cand[:, 4],            # H
        ])

        rows = []
        for i in range(len(cand_phys)):
            n_, eta_, sy_, W_, H_ = cand_phys[i]
            ts = time.time()
            try:
                y8 = sim.run(float(n_), float(eta_), float(sy_),
                              float(W_), float(H_))
                dt = time.time() - ts
                rows.append({
                    "cell_id": cell_id,
                    "n":  float(n_), "eta": float(eta_), "sigma_y": float(sy_),
                    "width": float(W_), "height": float(H_),
                    **{f"x_{j+1:02d}": float(y8[j]) for j in range(8)},
                    "sim_seconds": dt,
                })
            except Exception as e:
                print(f"  [FAIL cell {cell_id} sim {i+1}/{need}: {e}]")
                continue
            n_done_total += 1

            if (i + 1) % a.checkpoint_every == 0:
                append_rows(out_csv, header, rows)
                rows = []
                elapsed = time.time() - t0
                rate = n_done_total / max(elapsed, 1e-6)
                print(f"  cell {cell_id} ({ci+1}/{len(plan)}) "
                      f"  done {i+1}/{need}  "
                      f"  total sims={n_done_total}  "
                      f"  elapsed={elapsed:.0f}s  "
                      f"  {rate:.2f} sims/s")

        if rows:
            append_rows(out_csv, header, rows)
        # End of cell

    elapsed = time.time() - t0
    print(f"\n=== machine {a.machine_id} done ===")
    print(f"  total sims this run: {n_done_total}")
    print(f"  elapsed: {elapsed:.0f}s = {elapsed/3600:.2f}h")
    if n_done_total > 0:
        print(f"  avg: {elapsed/n_done_total:.2f}s/sim")


if __name__ == "__main__":
    main()
