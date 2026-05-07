"""Production-grade y8q + GP-aware-likelihood inverse on Chuno setup1 + setup2 + joint.

Reproduces the y8q breakthrough joint result σ_y diff +6.4 % vs rheometer truth
(see docs/PIPELINE_2026-05-07.md).  Compares legacy `gp_fit` against
`gp_fit_y8_quantile` sub-routing, K=1, log_nuisance_gp loss, default frame_w.

Cleaned-up replacement of scripts/v2_chuno_y8_quantile_test.py.  Output JSON has
the same schema so existing callers can swap.

Usage:
    python scripts/run_y8q_chuno.py
    python scripts/run_y8q_chuno.py --out scripts/run_y8q_chuno.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

# Force UTF-8 stdout (Windows default cp932 chokes on θ̂ / σ_y / ✓ / ✗)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as V10
from Optimization.libs import selector as SHA


# Chuno rheometer truth (25 °C fit, surrogate-space units; divide by 10 for Pa)
TRUTH_CHUNO = (0.633, 10.51, 19.62)


def make_args(top_k: int, weight_mode: str) -> argparse.Namespace:
    """Production argset.  All dead-path flags forced to 0 / False so the
    archived code (Method A, Method B, σ_y prior, spillover, CI, etc.) is
    NEVER touched at runtime."""
    return argparse.Namespace(
        # ---- routing / re-rank ----
        shape_top_k=top_k,
        shape_l2_mode="hard", shape_l2_eps_cm=0.5,
        shape_eps_cap_by_primary=False,
        shape_weight_mode=weight_mode, shape_gpfit_oversample=3,
        shape_topk_softmax_tau=1.5,
        shape_per_sub_z_scale=False,
        # ---- loss ----
        loss_mode="log_nuisance_gp",
        sigma_bias=0.25, sigma_trend=0.20, sigma_y_min=5.0,
        shape_gp_noise_log_floor=0.20,    # production default (Occam-bias suppressor)
        shape_disable_calib=True,         # production default (calib × low floor amplified bias)
        shape_cma_restarts=1,             # paper-grade: pass 5 via CLI
        # ---- CMA-ES ----
        shape_sigma0=0.25, shape_max_iter=30, shape_popsize=12,
        shape_cma_seed_offset=0,
        # ---- bbox ----
        shape_box_pad=0.10, shape_support_weight=0.25,
        # ---- frame_w ----
        shape_frame_w=None,           # default [0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0]
        shape_info_frame_w=False,
        # ---- UQ ----
        shape_report_ci=True,        # report 95% CI + per-dim identifiability flag
    )


def run_setup(label: str, ref_dir: Path, W: float, H: float, args, geo, xs, ys,
              device, dtype, truth: tuple[float, float, float]) -> tuple:
    y_obs = np.load(ref_dir / "y_obs.npy").astype(np.float64).ravel()
    print(f"\n--- {label}  (y_obs[7]={y_obs[7]:.3f}) ---", flush=True)
    item = SHA.prepare_setup_shape(ref_dir.name, W, H, y_obs,
                                    geo, xs, ys, device, dtype, args)
    primary = item["members"][0]
    bbox = (np.exp(primary.z_lo[2]), np.exp(primary.z_hi[2]))
    in_truth = "✓" if bbox[0] <= truth[2] <= bbox[1] else "✗"
    z_score = abs(y_obs[7] - primary.y8_med) / max(primary.y8_std, 1e-6)
    print(f"  routed sub_{primary.sub_id}  bbox σ_y=[{bbox[0]:.2f},{bbox[1]:.2f}] {in_truth}truth"
          f"  y8_med={primary.y8_med:.2f}±{primary.y8_std:.2f}  z-score={z_score:.2f}",
          flush=True)

    t0 = time.time()
    res = SHA.inverse_single_shape(item, args, device, dtype)
    elapsed = time.time() - t0
    th = (res["theta_n"], res["theta_eta"], res["theta_sy"])
    nd = 100 * (th[0] - truth[0]) / truth[0]
    ed = 100 * (th[1] - truth[1]) / truth[1]
    sd = 100 * (th[2] - truth[2]) / truth[2]
    print(f"  θ̂=({th[0]:.4f},{th[1]:.3f},{th[2]:.3f})  {elapsed:.1f}s  "
          f"diff: n {nd:+.1f}%  η {ed:+.1f}%  σ_y {sd:+.1f}%", flush=True)
    if "theta_ci_95" in res:
        ci = res["theta_ci_95"]
        idf = res["identifiable"]
        print(f"  CI 95%:  n=[{ci['n'][0]:.4f},{ci['n'][1]:.4f}]{'✓' if idf['n'] else '?'}  "
              f"η=[{ci['eta'][0]:.3f},{ci['eta'][1]:.3f}]{'✓' if idf['eta'] else '?'}  "
              f"σ_y=[{ci['sigma_y'][0]:.2f},{ci['sigma_y'][1]:.2f}]{'✓' if idf['sigma_y'] else '?'}",
              flush=True)
    return item, res, th, primary


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path,
                    default=REPO / "scripts" / "run_y8q_chuno.json",
                    help="Output JSON path")
    ap.add_argument("--state-root", type=Path,
                    default=REPO / "Models" / "v10_yshape_planB",
                    help="Production model bank")
    ap.add_argument("--data-root", type=Path,
                    default=REPO / "data" / "new_real_world_experiments",
                    help="Real-world experiments root")
    cli = ap.parse_args()

    V10.STATE_ROOT = cli.state_root.resolve()
    geo, xs, ys = V10.load_runtime()
    device, dtype = V10.HC.DEVICE, V10.HC.DTYPE

    setup1_dir = cli.data_root / "ref_Chuno_2.7_2.5_1"
    setup2_dir = cli.data_root / "ref_Chuno_2.0_4.0_2"

    summary: dict = {}
    for label, mode in [("legacy gp_fit", "gp_fit"),
                         ("PRODUCTION y8_quantile", "gp_fit_y8_quantile")]:
        print(f"\n{'='*72}\n=== {label}  weight_mode={mode}  K=1 ===\n{'='*72}",
              flush=True)
        args = make_args(top_k=1, weight_mode=mode)

        item1, res1, th1, p1 = run_setup("setup1 (W=2.5, H=2.7)", setup1_dir, 2.5, 2.7,
                                          args, geo, xs, ys, device, dtype, TRUTH_CHUNO)
        item2, res2, th2, p2 = run_setup("setup2 (W=4.0, H=2.0)", setup2_dir, 4.0, 2.0,
                                          args, geo, xs, ys, device, dtype, TRUTH_CHUNO)

        print(f"\n--- joint K=1+K=1 ---", flush=True)
        t0 = time.time()
        res_j = SHA.inverse_double_shape(item1, item2, args, device, dtype)
        ej = time.time() - t0
        thj = (res_j["theta_n"], res_j["theta_eta"], res_j["theta_sy"])
        nd = 100 * (thj[0] - TRUTH_CHUNO[0]) / TRUTH_CHUNO[0]
        ed = 100 * (thj[1] - TRUTH_CHUNO[1]) / TRUTH_CHUNO[1]
        sd = 100 * (thj[2] - TRUTH_CHUNO[2]) / TRUTH_CHUNO[2]
        print(f"  θ̂=({thj[0]:.4f},{thj[1]:.3f},{thj[2]:.3f})  {ej:.1f}s  "
              f"diff: n {nd:+.1f}%  η {ed:+.1f}%  σ_y {sd:+.1f}%", flush=True)
        if "theta_ci_95" in res_j:
            ci = res_j["theta_ci_95"]
            idf = res_j["identifiable"]
            print(f"  CI 95%:  n=[{ci['n'][0]:.4f},{ci['n'][1]:.4f}]{'✓' if idf['n'] else '?'}  "
                  f"η=[{ci['eta'][0]:.3f},{ci['eta'][1]:.3f}]{'✓' if idf['eta'] else '?'}  "
                  f"σ_y=[{ci['sigma_y'][0]:.2f},{ci['sigma_y'][1]:.2f}]{'✓' if idf['sigma_y'] else '?'}",
                  flush=True)

        summary[mode] = {
            "setup1": {
                "sub_id": int(p1.sub_id),
                "bbox": [float(np.exp(p1.z_lo[2])), float(np.exp(p1.z_hi[2]))],
                "y8_med": float(p1.y8_med), "y8_std": float(p1.y8_std),
                "theta": list(th1),
                "in_truth": bool(p1.z_lo[2] <= np.log(TRUTH_CHUNO[2]) <= p1.z_hi[2]),
            },
            "setup2": {
                "sub_id": int(p2.sub_id),
                "bbox": [float(np.exp(p2.z_lo[2])), float(np.exp(p2.z_hi[2]))],
                "y8_med": float(p2.y8_med), "y8_std": float(p2.y8_std),
                "theta": list(th2),
                "in_truth": bool(p2.z_lo[2] <= np.log(TRUTH_CHUNO[2]) <= p2.z_hi[2]),
            },
            "joint": {
                "theta": list(thj), "elapsed_s": float(ej),
                "n_diff": nd, "eta_diff": ed, "sy_diff": sd,
                "theta_ci_95": res_j.get("theta_ci_95"),
                "identifiable": res_j.get("identifiable"),
            },
        }

    cli.out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {cli.out}\n")
    print("=== Joint Summary ===")
    print(f"{'mode':<22} {'s1_sub':<8} {'s1_y8_med':>10} {'s2_sub':<8} {'s2_y8_med':>10} "
          f"{'joint θ̂':<28}  {'n%':>6} {'η%':>6} {'σ_y%':>7}")
    for mode, d in summary.items():
        s1 = d["setup1"]; s2 = d["setup2"]; j = d["joint"]
        t = j["theta"]
        th_s = f"({t[0]:.3f}, {t[1]:.2f}, {t[2]:.2f})"
        print(f"{mode:<22} sub_{s1['sub_id']:<3} {s1['y8_med']:>10.2f}  "
              f"sub_{s2['sub_id']:<3} {s2['y8_med']:>10.2f}  "
              f"{th_s:<28}  {j['n_diff']:>+5.1f} {j['eta_diff']:>+5.1f} {j['sy_diff']:>+6.1f}")


if __name__ == "__main__":
    main()
