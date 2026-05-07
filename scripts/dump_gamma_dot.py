"""Run MLS-MPM forward with a recovered θ̂ and dump per-frame γ̇ statistics.

Wraps `Simulation/simulation/AGTaichiMPM2.py`, which already records
per-particle γ̇ during simulation and dumps it as
``config_XX_gamma_dot.bin``.  This script:

  1. Reads a CMA-ES inverse output JSON (e.g. scripts/run_y8q_chuno.json)
     — extracts θ̂ for the requested mode/setup combo.
  2. Runs AGTaichiMPM2 with that θ̂ and the matching (W, H).
  3. Reads back the γ̇ binaries.
  4. Summarises per-frame γ̇ with quantile + histogram statistics so the
     numbers can be compared against rheometer γ̇ ∈ [0.1, 1000] s⁻¹.
  5. Writes ``gamma_dot_summary.json`` next to the .bin files.

Use cases:
  * Verify that the inverse's effective γ̇ range matches rheometer
    coverage (paper rebuttal "are you in HB regime at γ̇ ≲ 100 s⁻¹?").
  * Diagnose (n, η) ridge degeneracy: low-σ_y-dominated frames vs
    viscous-shear-dominated frames split at γ̇ ≈ (σ_y / η)^(1/n).
  * Drop into the freshness-experiment kinetics analysis: report γ̇(t)
    distribution as a derived observable.

Usage:
    # Recovered θ̂ from a JSON
    python scripts/dump_gamma_dot.py \
        --theta-json scripts/run_y8q_chuno_paper.json \
        --mode gp_fit_y8_quantile --setup setup1 \
        --W 2.5 --H 2.7 \
        --out gamma_dot_runs/chuno_setup1

    # Direct θ inputs (skip JSON)
    python scripts/dump_gamma_dot.py \
        --eta 8.12 --n 0.664 --sigmaY 19.65 \
        --W 4.0 --H 2.0 \
        --out gamma_dot_runs/chuno_setup2_joint
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
MPM_PY = REPO / "Simulation" / "simulation" / "AGTaichiMPM2.py"


def resolve_theta(args) -> tuple[float, float, float]:
    """Returns (eta, n, sigmaY)."""
    if args.eta is not None and args.n is not None and args.sigmaY is not None:
        return float(args.eta), float(args.n), float(args.sigmaY)
    if args.theta_json is None:
        raise SystemExit("Need either --theta-json or all three of --eta/--n/--sigmaY")
    blob = json.loads(args.theta_json.read_text())
    # Schema 1: y8q chuno-style { mode: { setup1: {theta: [n, eta, sy]}, joint: {...}, ... } }
    if args.mode and args.mode in blob:
        mblob = blob[args.mode]
        if args.setup not in mblob:
            raise SystemExit(f"--setup {args.setup!r} not in {list(mblob)}")
        target = mblob[args.setup]
        if isinstance(target, dict) and "theta" in target:
            theta = target["theta"]
            return float(theta[1]), float(theta[0]), float(theta[2])  # eta, n, sigmaY
    # Schema 2: prior-data style { material: { joint: {theta: [n, eta, sy]}, per_setup: {...} } }
    if args.material and args.material in blob:
        mat = blob[args.material]
        if args.setup == "joint" or args.setup is None:
            target = mat.get("joint")
        else:
            target = mat.get("per_setup", {}).get(args.setup)
        if isinstance(target, dict) and "theta" in target:
            theta = target["theta"]
            return float(theta[1]), float(theta[0]), float(theta[2])
    raise SystemExit(f"Could not extract θ̂ from {args.theta_json} via mode={args.mode}, "
                     f"material={args.material}, setup={args.setup}")


def summarise_frame(out_dir: Path, frame_idx: int, dt_s: float, fps: int,
                    bins_logspace: tuple[float, float, int]) -> dict | None:
    f_gdot = out_dir / f"config_{frame_idx:02d}_gamma_dot.bin"
    f_pos  = out_dir / f"config_{frame_idx:02d}_pos.bin"
    if not f_gdot.exists() or not f_pos.exists():
        return None
    gdot = np.fromfile(f_gdot, dtype=np.float32)
    pos = np.fromfile(f_pos, dtype=np.float32).reshape(-1, 3)
    n = gdot.size
    if n == 0:
        return {"frame": frame_idx, "n_particles": 0}

    pos_clean = pos[np.isfinite(gdot)]
    gdot_clean = gdot[np.isfinite(gdot)]
    nz = gdot_clean > 1e-9
    gdot_nz = gdot_clean[nz]
    n_nz = int(nz.sum())

    qs = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    quant_all = {f"q{int(q*100):02d}": float(np.quantile(gdot_clean, q)) for q in qs}
    quant_nz  = {f"q{int(q*100):02d}_nz": float(np.quantile(gdot_nz, q))
                 for q in qs} if n_nz else {}

    log_lo, log_hi, log_n = bins_logspace
    edges = np.logspace(log_lo, log_hi, log_n + 1)
    hist, _ = np.histogram(gdot_nz, bins=edges) if n_nz else (np.zeros(log_n), edges)

    return {
        "frame": frame_idx,
        "t_s": float(frame_idx) / fps if fps else None,
        "n_particles": int(n),
        "n_nonzero": n_nz,
        "frac_nonzero": float(n_nz) / max(n, 1),
        "min": float(gdot_clean.min()),
        "max": float(gdot_clean.max()),
        "median": float(np.median(gdot_clean)),
        "mean": float(gdot_clean.mean()),
        "quantiles_all": quant_all,
        "quantiles_nz": quant_nz,
        "hist_log10_edges": [float(e) for e in edges.tolist()],
        "hist_counts": [int(c) for c in hist.tolist()],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_argument_group("θ̂ source")
    src.add_argument("--theta-json", type=Path, default=None,
                     help="Inverse-output JSON (e.g. scripts/run_y8q_chuno_paper.json).")
    src.add_argument("--mode", type=str, default=None,
                     help="JSON top-level mode key (e.g. gp_fit_y8_quantile).")
    src.add_argument("--material", type=str, default=None,
                     help="JSON material key (prior-data schema).")
    src.add_argument("--setup", type=str, default="joint",
                     help='"setup1" / "setup2" / "joint" key inside mode/material.')
    src.add_argument("--eta", type=float, default=None)
    src.add_argument("--n", type=float, default=None)
    src.add_argument("--sigmaY", type=float, default=None)

    geo = ap.add_argument_group("setup geometry")
    geo.add_argument("--W", type=float, required=True, help="reservoir W [cm]")
    geo.add_argument("--H", type=float, required=True, help="reservoir H [cm]")
    geo.add_argument("--Z", type=float, default=4.3)

    sim = ap.add_argument_group("MPM args")
    sim.add_argument("--out", type=Path, required=True,
                     help="Output dir for .bin/.dat/.json files.")
    sim.add_argument("--max-frames", type=int, default=8)
    sim.add_argument("--fps", type=int, default=24)
    sim.add_argument("--dt", type=float, default=0.000075)
    sim.add_argument("--rho", type=float, default=1.025)
    sim.add_argument("--save-dat", action="store_true")
    sim.add_argument("--taichi-arch", type=str, default="cuda",
                     choices=["cuda", "cpu"])
    sim.add_argument("--skip-mpm", action="store_true",
                     help="Skip MPM run; only summarise existing dumps under --out.")

    sty = ap.add_argument_group("γ̇ summary")
    sty.add_argument("--hist-log-lo", type=float, default=-2.0,
                     help="log10(γ̇_min) for histogram (default −2.0 → γ̇=0.01 s⁻¹).")
    sty.add_argument("--hist-log-hi", type=float, default=4.0,
                     help="log10(γ̇_max) for histogram (default +4.0 → γ̇=10⁴ s⁻¹).")
    sty.add_argument("--hist-bins", type=int, default=24)

    args = ap.parse_args()
    eta, n, sy = resolve_theta(args)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[dump_gamma_dot] θ̂: eta={eta:.4f}  n={n:.4f}  sigmaY={sy:.4f}", flush=True)
    print(f"                   geometry W={args.W:.2f} H={args.H:.2f} Z={args.Z:.2f}",
          flush=True)

    if not args.skip_mpm:
        # Run as module so the in-package `from .xmlParser import ...` works
        # (Simulation/simulation/taichi.py shadows the real taichi package
        # when invoked as a free-standing script).
        cmd = [
            sys.executable, "-m", "Simulation.simulation.AGTaichiMPM2",
            "--eta", f"{eta:.6f}", "--n", f"{n:.6f}", "--sigmaY", f"{sy:.6f}",
            "--W", f"{args.W:.6f}", "--H", f"{args.H:.6f}", "--Z", f"{args.Z:.6f}",
            "--rho", f"{args.rho}",
            "--out", str(args.out.resolve()),
            "--max-frames", str(args.max_frames),
            "--fps", str(args.fps),
            "--dt", f"{args.dt}",
            "--taichi-arch", args.taichi_arch,
        ]
        if args.save_dat:
            cmd.append("--save-dat")
        print("[dump_gamma_dot] launching MPM:", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, check=False, cwd=str(REPO)).returncode
        if rc != 0:
            raise SystemExit(f"AGTaichiMPM2 failed with returncode={rc}")

    # ---- summarise frames ----
    summary = {
        "theta": {"eta": eta, "n": n, "sigmaY": sy},
        "geometry": {"W": args.W, "H": args.H, "Z": args.Z},
        "fps": args.fps, "dt_s": args.dt,
        "frames": [],
    }
    for f in range(0, args.max_frames + 1):
        rec = summarise_frame(args.out, f, args.dt, args.fps,
                              (args.hist_log_lo, args.hist_log_hi, args.hist_bins))
        if rec is not None:
            summary["frames"].append(rec)

    out_json = args.out / "gamma_dot_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[dump_gamma_dot] saved {out_json}\n")

    print(f"{'frame':>5} | {'t [s]':>6} | {'γ̇ med':>9} | {'γ̇ q05':>9} | "
          f"{'γ̇ q95':>9} | {'γ̇ max':>9} | {'frac>0':>6}")
    for r in summary["frames"]:
        if r["n_particles"] == 0:
            print(f"{r['frame']:>5} |   --   | (no particles)")
            continue
        q = r["quantiles_all"]
        print(f"{r['frame']:>5} | {r['t_s']:>6.3f} | {r['median']:>9.3f} | "
              f"{q['q05']:>9.3f} | {q['q95']:>9.3f} | {r['max']:>9.3f} | "
              f"{r['frac_nonzero']:>6.2%}")


if __name__ == "__main__":
    main()
