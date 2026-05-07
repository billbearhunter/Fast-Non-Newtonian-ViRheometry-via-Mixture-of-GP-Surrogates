"""Run MLS-MPM forward with a recovered θ̂ and dump per-frame γ̇ statistics.

Wraps `Simulation/simulation/AGTaichiMPM2.py`, which already records
per-particle γ̇ during simulation and dumps it as
``config_XX_gamma_dot.bin``.  Supports two modes:

  (1) Single θ̂ mode: read one (n, η, σ_y) from a JSON or CLI flags,
      run MPM once, write ``gamma_dot_summary.json``.

  (2) Trail-CSV mode (--trail-csv): read every CMA-ES candidate logged
      via Optimization/libs/selector.py's ``--shape-cma-trail-csv`` flag,
      run MPM for each row, append per-frame γ̇ summary statistics back
      to the CSV.  Used for the 老師 request "γ̇ trajectory through the
      whole optimization, not just the final θ̂".

Use cases:
  * Verify that the inverse's effective γ̇ range matches rheometer
    coverage (paper rebuttal "are you in HB regime at γ̇ ≲ 100 s⁻¹?").
  * Diagnose (n, η) ridge degeneracy: low-σ_y-dominated frames vs
    viscous-shear-dominated frames split at γ̇ ≈ (σ_y / η)^(1/n).
  * Drop into the freshness-experiment kinetics analysis: report γ̇(t)
    distribution as a derived observable.

Usage:
    # Single θ̂ from JSON
    python scripts/dump_gamma_dot.py \\
        --theta-json scripts/run_y8q_chuno_paper.json \\
        --mode gp_fit_y8_quantile --setup setup1 \\
        --W 2.5 --H 2.7 \\
        --out gamma_dot_runs/chuno_setup1

    # Single θ̂ direct
    python scripts/dump_gamma_dot.py \\
        --eta 8.12 --n 0.664 --sigmaY 19.65 \\
        --W 4.0 --H 2.0 \\
        --out gamma_dot_runs/chuno_setup2_joint

    # Trail-CSV: every CMA candidate gets its own MPM γ̇ pass
    python scripts/dump_gamma_dot.py \\
        --trail-csv runs/chuno_trail.csv \\
        --W 2.5 --H 2.7 \\
        --trail-out runs/chuno_trail_with_gamma_dot.csv \\
        --skip-mpm-existing   # don't re-run MPM if cached
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
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


def _run_mpm(eta: float, n: float, sy: float, W: float, H: float, Z: float,
              out_dir: Path, max_frames: int, fps: int, dt: float, rho: float,
              taichi_arch: str = "cuda", save_dat: bool = False,
              quiet: bool = False) -> int:
    """Run AGTaichiMPM2 once with given (θ, geometry); returns subprocess rc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "Simulation.simulation.AGTaichiMPM2",
        "--eta", f"{eta:.6f}", "--n", f"{n:.6f}", "--sigmaY", f"{sy:.6f}",
        "--W", f"{W:.6f}", "--H", f"{H:.6f}", "--Z", f"{Z:.6f}",
        "--rho", f"{rho}",
        "--out", str(out_dir.resolve()),
        "--max-frames", str(max_frames),
        "--fps", str(fps),
        "--dt", f"{dt}",
        "--taichi-arch", taichi_arch,
    ]
    if save_dat:
        cmd.append("--save-dat")
    if not quiet:
        print("[dump_gamma_dot] launching MPM:", " ".join(cmd), flush=True)
    if quiet:
        rc = subprocess.run(cmd, check=False, cwd=str(REPO),
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode
    else:
        rc = subprocess.run(cmd, check=False, cwd=str(REPO)).returncode
    return rc


def _summarise_run(out_dir: Path, max_frames: int, dt: float, fps: int,
                    hist: tuple[float, float, int]) -> dict:
    summary_frames = []
    for f in range(0, max_frames + 1):
        rec = summarise_frame(out_dir, f, dt, fps, hist)
        if rec is not None:
            summary_frames.append(rec)
    return {"frames": summary_frames}


def _aggregate_per_run(summary_frames: list[dict], drop_frame_0: bool = True) -> dict:
    """Collapse per-frame γ̇ stats into a per-run summary suitable for a CSV row."""
    frames = summary_frames
    if drop_frame_0:
        frames = [f for f in frames if f.get("frame", 0) != 0]
    if not frames:
        return {f"gamma_dot_{k}": None for k in
                ("med_overall", "q05_overall", "q95_overall", "max_overall",
                 "med_peak_frame", "med_peak_value")}
    # Pool all per-frame medians (one number per frame)
    meds = [f["median"] for f in frames if f.get("median") is not None]
    q05s = [f["quantiles_all"]["q05"] for f in frames if f.get("quantiles_all")]
    q95s = [f["quantiles_all"]["q95"] for f in frames if f.get("quantiles_all")]
    maxs = [f["max"] for f in frames if f.get("max") is not None]
    if not meds:
        return {f"gamma_dot_{k}": None for k in
                ("med_overall", "q05_overall", "q95_overall", "max_overall",
                 "med_peak_frame", "med_peak_value")}
    peak_idx = int(np.argmax(meds))
    peak_frame = frames[peak_idx]["frame"]
    return {
        "gamma_dot_med_overall": float(np.median(meds)),
        "gamma_dot_q05_overall": float(np.median(q05s)) if q05s else None,
        "gamma_dot_q95_overall": float(np.median(q95s)) if q95s else None,
        "gamma_dot_max_overall": float(np.max(maxs)) if maxs else None,
        "gamma_dot_med_peak_frame": int(peak_frame),
        "gamma_dot_med_peak_value": float(meds[peak_idx]),
    }


def _theta_hash(eta: float, n: float, sy: float) -> str:
    """Stable short hash for caching MPM dumps by θ.  4 decimals = 1e-4 precision."""
    s = f"{eta:.4f}_{n:.4f}_{sy:.4f}"
    return hashlib.md5(s.encode()).hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    src = ap.add_argument_group("θ̂ source (single-run mode)")
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

    trail = ap.add_argument_group("trail-CSV mode")
    trail.add_argument("--trail-csv", type=Path, default=None,
                       help="Input CSV from selector --shape-cma-trail-csv "
                            "(columns: inverse_id, restart, generation, "
                            "candidate_idx, n, eta, sigma_y, ..., loss).  "
                            "If set, runs MPM once per row and appends γ̇ "
                            "summary columns.")
    trail.add_argument("--trail-out", type=Path, default=None,
                       help="Output CSV with γ̇ columns appended (default: "
                            "<trail-csv-stem>_gamma_dot.csv).")
    trail.add_argument("--trail-cache-root", type=Path, default=None,
                       help="Per-row MPM dump cache directory.  Each unique θ "
                            "gets its own subdir keyed by hash, so re-runs of "
                            "the same θ skip MPM.  Default: "
                            "<trail-csv-parent>/_mpm_cache/.")
    trail.add_argument("--trail-filter", type=str, default=None,
                       help="Optional Python expression over CSV columns to "
                            "filter rows.  E.g. 'restart == 0' or "
                            "'generation % 5 == 0' (subsamples).")
    trail.add_argument("--trail-keep-bin", action="store_true",
                       help="Keep raw .bin per-particle dumps (default: cleaned "
                            "after each row to keep cache footprint bounded).")

    geo = ap.add_argument_group("setup geometry")
    geo.add_argument("--W", type=float, required=True, help="reservoir W [cm]")
    geo.add_argument("--H", type=float, required=True, help="reservoir H [cm]")
    geo.add_argument("--Z", type=float, default=4.3)

    sim = ap.add_argument_group("MPM args")
    sim.add_argument("--out", type=Path, default=None,
                     help="Output dir for .bin/.dat/.json files (single-run only).")
    sim.add_argument("--max-frames", type=int, default=8)
    sim.add_argument("--fps", type=int, default=24)
    sim.add_argument("--dt", type=float, default=0.000075)
    sim.add_argument("--rho", type=float, default=1.025)
    sim.add_argument("--save-dat", action="store_true")
    sim.add_argument("--taichi-arch", type=str, default="cuda",
                     choices=["cuda", "cpu"])
    sim.add_argument("--skip-mpm", action="store_true",
                     help="Single-run only: skip MPM, summarise existing dumps.")

    sty = ap.add_argument_group("γ̇ summary")
    sty.add_argument("--hist-log-lo", type=float, default=-2.0,
                     help="log10(γ̇_min) for histogram (default −2.0 → γ̇=0.01 s⁻¹).")
    sty.add_argument("--hist-log-hi", type=float, default=4.0,
                     help="log10(γ̇_max) for histogram (default +4.0 → γ̇=10⁴ s⁻¹).")
    sty.add_argument("--hist-bins", type=int, default=24)

    args = ap.parse_args()
    hist_cfg = (args.hist_log_lo, args.hist_log_hi, args.hist_bins)

    if args.trail_csv:
        # ---------------------- trail-CSV (batch) mode ----------------------
        return run_trail_mode(args, hist_cfg)

    # ---------------------- single-run mode ----------------------
    if args.out is None:
        raise SystemExit("--out is required in single-run mode")
    eta, n, sy = resolve_theta(args)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[dump_gamma_dot] θ̂: eta={eta:.4f}  n={n:.4f}  sigmaY={sy:.4f}", flush=True)
    print(f"                   geometry W={args.W:.2f} H={args.H:.2f} Z={args.Z:.2f}",
          flush=True)
    if not args.skip_mpm:
        rc = _run_mpm(eta, n, sy, args.W, args.H, args.Z, args.out,
                       args.max_frames, args.fps, args.dt, args.rho,
                       args.taichi_arch, args.save_dat)
        if rc != 0:
            raise SystemExit(f"AGTaichiMPM2 failed with returncode={rc}")

    summary = {
        "theta": {"eta": eta, "n": n, "sigmaY": sy},
        "geometry": {"W": args.W, "H": args.H, "Z": args.Z},
        "fps": args.fps, "dt_s": args.dt,
        **_summarise_run(args.out, args.max_frames, args.dt, args.fps, hist_cfg),
    }
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


def run_trail_mode(args, hist_cfg):
    in_csv = args.trail_csv
    out_csv = args.trail_out or in_csv.with_name(f"{in_csv.stem}_gamma_dot.csv")
    cache_root = args.trail_cache_root or in_csv.parent / "_mpm_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        in_rows = list(reader)
        in_fields = list(reader.fieldnames or [])
    if not in_rows:
        raise SystemExit(f"Trail CSV {in_csv} is empty")

    new_fields = [
        "gamma_dot_med_overall", "gamma_dot_q05_overall",
        "gamma_dot_q95_overall", "gamma_dot_max_overall",
        "gamma_dot_med_peak_frame", "gamma_dot_med_peak_value",
        "mpm_cached", "mpm_rc",
    ]
    out_fields = in_fields + new_fields

    # Optional filter
    if args.trail_filter:
        flt_expr = args.trail_filter
        in_rows_typed = []
        for r in in_rows:
            try:
                env = {k: (float(v) if k in {"n", "eta", "sigma_y", "z_n",
                                              "z_log_eta", "z_log_sigma_y", "loss"}
                            else int(v) if k in {"restart", "generation", "candidate_idx"}
                            else v) for k, v in r.items()}
                if eval(flt_expr, {"__builtins__": {}}, env):
                    in_rows_typed.append(r)
            except Exception as e:
                print(f"[trail-mode] filter eval err on row: {e}; skipping", flush=True)
        in_rows = in_rows_typed
    print(f"[trail-mode] {len(in_rows)} rows to process  cache={cache_root}",
          flush=True)
    print(f"[trail-mode] geometry W={args.W:.2f} H={args.H:.2f} Z={args.Z:.2f}",
          flush=True)

    fout = out_csv.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=out_fields)
    writer.writeheader()

    t0 = time.time()
    n_cached_hit = 0
    for idx, row in enumerate(in_rows):
        try:
            eta = float(row["eta"]); n_v = float(row["n"]); sy = float(row["sigma_y"])
        except Exception as e:
            print(f"[trail-mode] row {idx}: bad θ ({e}); skip"); continue
        h = _theta_hash(eta, n_v, sy)
        cache_dir = cache_root / f"theta_{h}"
        # Cache check: presence of all expected gamma_dot.bin files
        cached = all((cache_dir / f"config_{f:02d}_gamma_dot.bin").exists()
                      for f in range(args.max_frames + 1))
        if not cached:
            rc = _run_mpm(eta, n_v, sy, args.W, args.H, args.Z, cache_dir,
                          args.max_frames, args.fps, args.dt, args.rho,
                          args.taichi_arch, args.save_dat, quiet=True)
        else:
            rc = 0
            n_cached_hit += 1

        agg = {f"gamma_dot_{k}": None for k in
               ("med_overall", "q05_overall", "q95_overall", "max_overall",
                "med_peak_frame", "med_peak_value")}
        if rc == 0:
            srun = _summarise_run(cache_dir, args.max_frames, args.dt, args.fps, hist_cfg)
            agg = _aggregate_per_run(srun["frames"], drop_frame_0=True)

        out_row = {**row, **agg, "mpm_cached": int(cached), "mpm_rc": rc}
        writer.writerow(out_row)
        fout.flush()

        if not args.trail_keep_bin and not cached and rc == 0:
            for fbin in cache_dir.glob("*.bin"):
                try: fbin.unlink()
                except OSError: pass

        if (idx + 1) % 10 == 0 or idx == len(in_rows) - 1:
            elapsed = time.time() - t0
            avg = elapsed / (idx + 1)
            eta_s = avg * (len(in_rows) - idx - 1)
            print(f"  [{idx+1:5d}/{len(in_rows)}]  cached={n_cached_hit}  "
                  f"avg {avg:.2f}s/row  ETA {eta_s/60:.1f} min  "
                  f"last γ̇_med_peak={agg['gamma_dot_med_peak_value']}", flush=True)

    fout.close()
    elapsed = time.time() - t0
    print(f"\n[trail-mode] done.  {len(in_rows)} rows in {elapsed/60:.1f} min "
          f"(cached={n_cached_hit}, MPM={len(in_rows) - n_cached_hit})", flush=True)
    print(f"[trail-mode] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
