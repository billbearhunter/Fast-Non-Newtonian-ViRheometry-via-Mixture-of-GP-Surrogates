"""
L0 Sim Differential Reference Panel
=====================================

在 prescribed time-varying θ(t) 下用 MPM forward 生成 synthetic dam-break
观测,跑我们的 pipeline 反演,与 ground truth (T_k*, σ_Y∞*) 对比。

Goal: validate that the pipeline correctly recovers a *time-varying*
θ trajectory in the absence of any real-world preparation noise.

Usage
-----
    python scripts/run_synthetic_freshness.py \\
        --prescribed-csv data/freshness_2026-05-09/L0_sim/prescribed_panel.csv \\
        --out-root       data/freshness_2026-05-09/L0_sim/ \\
        --state-root     Models/yshape_mogp_production \\
        --seed           42

Inputs
------
prescribed_panel.csv with columns:
    id, T_k_min, sigma_inf_pa, sigma_0_pa, n, eta_inf_pas

Outputs
-------
out_root/
├── trajectory_<id>/
│   ├── t000_setup1/  y_obs.npy + settings.xml + theta_hat.json
│   ├── t000_setup2/  y_obs.npy + settings.xml + theta_hat.json
│   ├── t020_setup1/  ...
│   ├── ... (3 time × 2 setup = 6 dirs)
│   └── avrami_fit.json   <- recovered (T_k_hat, sigma_inf_hat) ± 1σ
└── recovery_summary.csv   <- 12 rows: prescribed vs recovered

Computational cost
------------------
12 trajectory × 3 time × 2 setup = 72 MPM forward calls
Plus 36 single-setup inverse + 36 joint inverse
Estimated wall-clock: ~6 GPU-h on a single consumer GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def avrami_first_order(t_min, sigma_inf, T_k_min, sigma_0=0.0):
    """First-order Avrami / build-up: σ_Y(t) = σ_0 + (σ_∞ - σ_0)(1 - exp(-t/T_k))"""
    return sigma_0 + (sigma_inf - sigma_0) * (1.0 - np.exp(-t_min / T_k_min))


def write_settings_xml(out_dir: Path, W: float, H: float, rho: float = 1.0):
    """Write a minimal settings.xml that the existing pipeline can consume."""
    xml = f'''<?xml version="1.0"?>
<Optimizer>
  <path root_dir_path="{out_dir.resolve()}" />
  <setup RHO="{rho}" H="{H:.4f}" W="{W:.4f}" />
  <cuboid min="-0.15 -0.15 -0.15" max="{W:.4f} {H:.4f} {W+0.15:.4f}" density="{rho}" cell_samples_per_dim="2" vel="0.0 0.0 0.0" omega="0.0 0.0 0.0" />
  <static_box min="-100 -1 -100" max="100 0 100" boundary_behavior="sticking"/>
  <static_box min="-1 0 0" max="0 20 {W:.4f}" boundary_behavior="sticking"/>
  <static_box min="-1 0 -0.3" max="{W:.4f} 20 0" boundary_behavior="sticking"/>
  <static_box min="-1 0 {W:.4f}" max="{W:.4f} 20 {W+0.3:.4f}" boundary_behavior="sticking"/>
</Optimizer>
'''
    (out_dir / "settings.xml").write_text(xml, encoding="utf-8")


def run_mpm_forward(theta_t, W, H, out_dir: Path) -> np.ndarray:
    """Call Simulation/run_sim_only.py and return the 8-D flow-front vector."""
    n, eta, sigma_y = theta_t
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "_mpm_y.csv"

    cmd = [
        sys.executable, "-m", "Simulation.run_sim_only",
        "--n", f"{n}",
        "--eta", f"{eta}",
        "--sigma_y", f"{sigma_y}",
        "--W", f"{W:.4f}",
        "--H", f"{H:.4f}",
        "--out_csv", str(out_csv),
    ]
    print(f"  [MPM forward] θ=({n:.3f},{eta:.2f},{sigma_y:.2f}) (W,H)=({W:.2f},{H:.2f})")
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)

    df = pd.read_csv(out_csv)
    y_obs = df[[f"x_0{i}" for i in range(1, 9)]].iloc[0].to_numpy(dtype=float)
    np.save(out_dir / "y_obs.npy", y_obs)
    write_settings_xml(out_dir, W=W, H=H)
    return y_obs


def run_first_setup_inverse(setup1_dir: Path, state_root: Path, density: float = 1.0):
    """Call estimate_first_setup. Returns (theta_hat, proposed_W2, proposed_H2)."""
    cmd = [
        sys.executable, "-m", "Optimization.estimate_first_setup",
        "--state-root", str(state_root),
        "-f", str(setup1_dir),
        "--density", f"{density}",
        "--shape-cma-restarts", "5",
        "--shape-report-ci",
    ]
    print(f"  [single inv] {setup1_dir.name}")
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)

    theta_hat_path = setup1_dir / "theta_hat.json"
    th = json.loads(theta_hat_path.read_text())
    proposed = th.get("proposed_setup_2", {})
    W2 = float(proposed.get("W", 0.0))
    H2 = float(proposed.get("H", 0.0))
    th_hat = (
        float(th["theta_hat"]["n"]),
        float(th["theta_hat"]["eta"]),
        float(th["theta_hat"]["sigma_y"]),
    )
    return th_hat, W2, H2


def run_joint_inverse(setup1_dir: Path, setup2_dir: Path):
    """Call estimate_joint_setup. Returns recovered θ̂_joint."""
    cmd = [
        sys.executable, "-m", "Optimization.estimate_joint_setup",
        "-f", str(setup1_dir),
        "-s", str(setup2_dir),
        "--shape-warm-start-from-first",
        "--shape-cma-restarts", "5",
        "--shape-report-ci",
    ]
    print(f"  [joint inv] {setup1_dir.name} + {setup2_dir.name}")
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)

    out_path = setup2_dir / "theta_hat_joint.json"
    if not out_path.exists():
        out_path = setup2_dir / "theta_hat.json"
    th = json.loads(out_path.read_text())
    return (
        float(th["theta_hat"]["n"]),
        float(th["theta_hat"]["eta"]),
        float(th["theta_hat"]["sigma_y"]),
    )


# ----------------------------------------------------------------------
# Main panel loop
# ----------------------------------------------------------------------

def process_trajectory(row, out_root: Path, state_root: Path, rng: np.random.Generator):
    """Process one prescribed trajectory: 3 time points × joint pair, then Avrami fit."""
    traj_id = int(row["id"])
    T_k_true = float(row["T_k_min"])
    sigma_inf_true = float(row["sigma_inf_pa"])
    sigma_0_true = float(row["sigma_0_pa"])
    n_true = float(row["n"])
    eta_inf_true = float(row["eta_inf_pas"])

    traj_root = out_root / f"trajectory_{traj_id:02d}"
    traj_root.mkdir(parents=True, exist_ok=True)

    # Sample one random setup-1 geometry per trajectory (kept fixed across t_i,
    # mirrors the lab session protocol)
    W1 = float(rng.uniform(2.0, 7.0))
    H1 = float(rng.uniform(2.0, 7.0))

    time_points = [0.0, 20.0, 40.0]
    sigma_y_traj_true = [
        avrami_first_order(t, sigma_inf_true, T_k_true, sigma_0_true)
        for t in time_points
    ]

    sigma_y_traj_recovered = []

    # We propose setup-2 once at t=0 (to mirror the lab pre-commit).
    proposed_W2, proposed_H2 = None, None

    for t_i, sigma_y_t in zip(time_points, sigma_y_traj_true):
        theta_t = (n_true, eta_inf_true, sigma_y_t)
        tag = f"t{int(t_i):03d}"

        s1_dir = traj_root / f"{tag}_setup1"
        s2_dir = traj_root / f"{tag}_setup2"

        # Setup-1 forward + inverse
        run_mpm_forward(theta_t, W1, H1, out_dir=s1_dir)

        if t_i == 0.0:
            # First time: run inverse + proposer to lock setup-2 geometry
            th_hat_1, proposed_W2, proposed_H2 = run_first_setup_inverse(
                s1_dir, state_root=state_root, density=1.0
            )
            print(f"    [traj {traj_id}] proposed setup-2 = "
                  f"({proposed_W2:.2f}, {proposed_H2:.2f})")
        else:
            # Subsequent time points: still need single-setup inverse on setup-1
            # so joint can warm-start; proposer call ignored
            run_first_setup_inverse(s1_dir, state_root=state_root, density=1.0)

        # Setup-2 forward (using the proposed (W2, H2) from t=0)
        run_mpm_forward(theta_t, proposed_W2, proposed_H2, out_dir=s2_dir)

        # Joint inverse
        th_joint = run_joint_inverse(s1_dir, s2_dir)
        sigma_y_traj_recovered.append(th_joint[2])

    # Avrami fit on recovered σ_Y(t_i)
    try:
        popt, pcov = curve_fit(
            lambda t, s_inf, T: avrami_first_order(t, s_inf, T, sigma_0=sigma_0_true),
            time_points,
            sigma_y_traj_recovered,
            p0=[sigma_inf_true, T_k_true],
            bounds=([1.0, 1.0], [1000.0, 1000.0]),
        )
        sigma_inf_hat, T_k_hat = popt
        sigma_inf_se = float(np.sqrt(pcov[0, 0]))
        T_k_se = float(np.sqrt(pcov[1, 1]))
    except Exception as exc:
        print(f"    [traj {traj_id}] Avrami fit failed: {exc}")
        sigma_inf_hat, T_k_hat = np.nan, np.nan
        sigma_inf_se, T_k_se = np.nan, np.nan

    # Save per-trajectory summary
    fit_summary = {
        "trajectory_id": traj_id,
        "prescribed": {
            "T_k_min": T_k_true,
            "sigma_inf_pa": sigma_inf_true,
            "sigma_0_pa": sigma_0_true,
            "n": n_true,
            "eta_inf_pas": eta_inf_true,
        },
        "setup_1": {"W": W1, "H": H1},
        "setup_2_proposed": {"W": proposed_W2, "H": proposed_H2},
        "time_points_min": time_points,
        "sigma_y_true_pa": sigma_y_traj_true,
        "sigma_y_recovered_pa": sigma_y_traj_recovered,
        "avrami_fit": {
            "T_k_hat_min": T_k_hat,
            "T_k_hat_se_min": T_k_se,
            "sigma_inf_hat_pa": sigma_inf_hat,
            "sigma_inf_hat_se_pa": sigma_inf_se,
        },
        "errors_pct": {
            "T_k": abs(T_k_hat - T_k_true) / T_k_true * 100.0
                if not np.isnan(T_k_hat) else None,
            "sigma_inf": abs(sigma_inf_hat - sigma_inf_true) / sigma_inf_true * 100.0
                if not np.isnan(sigma_inf_hat) else None,
        },
    }
    (traj_root / "avrami_fit.json").write_text(json.dumps(fit_summary, indent=2))

    return fit_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prescribed-csv", type=Path, required=True,
                    help="CSV with columns id, T_k_min, sigma_inf_pa, sigma_0_pa, n, eta_inf_pas")
    ap.add_argument("--out-root", type=Path, required=True,
                    help="Output dir for synthetic panel data")
    ap.add_argument("--state-root", type=Path,
                    default=Path("Models/yshape_mogp_production"),
                    help="GP surrogate bank root")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(args.prescribed_csv)

    print(f"Loaded {len(panel)} prescribed trajectories from {args.prescribed_csv}")
    summaries = []
    for _, row in panel.iterrows():
        try:
            summary = process_trajectory(row, args.out_root, args.state_root, rng)
            summaries.append(summary)
        except Exception as exc:
            print(f"  [traj {row['id']}] FAILED: {exc}")
            summaries.append({"trajectory_id": int(row["id"]), "error": str(exc)})

    # Write recovery summary CSV
    rows = []
    for s in summaries:
        if "avrami_fit" not in s:
            continue
        rows.append({
            "trajectory_id": s["trajectory_id"],
            "T_k_true_min": s["prescribed"]["T_k_min"],
            "T_k_hat_min": s["avrami_fit"]["T_k_hat_min"],
            "T_k_err_pct": s["errors_pct"]["T_k"],
            "sigma_inf_true_pa": s["prescribed"]["sigma_inf_pa"],
            "sigma_inf_hat_pa": s["avrami_fit"]["sigma_inf_hat_pa"],
            "sigma_inf_err_pct": s["errors_pct"]["sigma_inf"],
            "n_true": s["prescribed"]["n"],
            "eta_inf_true": s["prescribed"]["eta_inf_pas"],
            "setup1_W": s["setup_1"]["W"],
            "setup1_H": s["setup_1"]["H"],
            "setup2_W": s["setup_2_proposed"]["W"],
            "setup2_H": s["setup_2_proposed"]["H"],
        })
    pd.DataFrame(rows).to_csv(args.out_root / "recovery_summary.csv", index=False)
    print(f"\nWrote {args.out_root / 'recovery_summary.csv'} with {len(rows)} trajectories.")

    # Quick console report
    if rows:
        df = pd.DataFrame(rows)
        print("\n=== Recovery Summary ===")
        print(f"  Median T_k error      : {df['T_k_err_pct'].median():.1f}%")
        print(f"  Median σ_Y∞ error     : {df['sigma_inf_err_pct'].median():.1f}%")
        n_pass = ((df['T_k_err_pct'] < 30) & (df['sigma_inf_err_pct'] < 30)).sum()
        print(f"  Trajectories with both errors < 30%: {n_pass} / {len(df)}")


if __name__ == "__main__":
    main()
