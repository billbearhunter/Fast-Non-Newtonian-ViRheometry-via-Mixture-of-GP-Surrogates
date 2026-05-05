"""2nd-setup Herschel-Bulkley joint inverse.

Mirror of prev-work `6_optimization/herschel_bulkley_parameter_estimation_for_2nd_setup.py`,
but with surrogate replacing MPM and y8 replacing silhouettes.

Pipeline:
    1. Load both `<ref_dir>/y_obs.npy` files; parse (H, W) from each dir name.
    2. Joint inverse: CMA-ES + σY two-stage prior on (setup1, setup2).
    3. Write `<second_dir>/theta_hat.json` with the joint θ̂.

Usage:
    python -m Optimization.setup2 \
        -f data/real_world_experiments/ref_Sesame_4.5_4.0_1 \
        -s data/real_world_experiments/ref_Sesame_2.0_2.0_2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as V10
from Optimization.libs import selector as SHA
from Optimization.setup1 import parse_ref_dir


def estimate(first_dir: Path, second_dir: Path, args) -> dict:
    info_a = parse_ref_dir(first_dir)
    info_b = parse_ref_dir(second_dir)
    y_a = np.load(first_dir / "y_obs.npy").astype(np.float64).ravel()
    y_b = np.load(second_dir / "y_obs.npy").astype(np.float64).ravel()

    geo_router, xs, ys = V10.load_runtime()
    device, dtype = V10.HC.DEVICE, V10.HC.DTYPE

    item_a = SHA.prepare_setup_shape(first_dir.name, info_a["W"], info_a["H"], y_a,
                                     geo_router, xs, ys, device, dtype, args)
    item_b = SHA.prepare_setup_shape(second_dir.name, info_b["W"], info_b["H"], y_b,
                                     geo_router, xs, ys, device, dtype, args)
    res = SHA.inverse_double_shape(item_a, item_b, args, device, dtype)
    if res is None:
        raise RuntimeError("joint inverse failed: no admitted experts")

    out = {
        "material": info_a["material"],
        "setup_1": {"H": info_a["H"], "W": info_a["W"], "name": first_dir.name},
        "setup_2": {"H": info_b["H"], "W": info_b["W"], "name": second_dir.name},
        "theta_hat": {
            "n": float(res["theta_n"]),
            "eta": float(res["theta_eta"]),
            "sigma_y": float(res["theta_sy"]),
        },
        "objective": float(res["objective"]),
        "opt_s": float(res["opt_s"]),
        "sy_anchor": float(res.get("sy_anchor", float("nan"))),
        "sy_single_a": float(res.get("sy_single_a", float("nan"))),
        "sy_single_b": float(res.get("sy_single_b", float("nan"))),
    }
    if "theta_ci_95" in res:
        out["theta_ci_95"] = res["theta_ci_95"]
        out["z_std"] = res.get("z_std")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-f", "--first-dir", type=Path, required=True)
    ap.add_argument("-s", "--second-dir", type=Path, required=True)
    ap.add_argument("--state-root", type=Path, default=V10.STATE_ROOT,
                    help="Trained MoGP-rBCM bank root")
    ap.add_argument("--loss-mode", choices=["linear", "log_nuisance", "log_nuisance_gp", "mog_likelihood"], default="log_nuisance")
    ap.add_argument("--sigma-bias", type=float, default=0.25)
    ap.add_argument("--sigma-trend", type=float, default=0.20)
    ap.add_argument("--sigma-y-min", type=float, default=5.0)
    ap.add_argument("--inverse-mode", choices=["shape"], default="shape")
    ap.add_argument("--shape-warm-start-from-first", action="store_true",
                    help="Read <first_dir>/theta_hat.json (must exist; "
                         "produced by a prior setup1 run) and use its θ̂ "
                         "as joint CMA-ES warm-start x0.  When combined "
                         "with --shape-sy-prior-weight > 0 also uses "
                         "setup1's σY directly as the prior anchor "
                         "(skipping setup2's expensive internal stage-1 "
                         "single inverses).  Workflow: setup1 -K=3 → "
                         "writes theta_hat.json → setup2 -K=1 with this "
                         "flag picks up the warm-start cheaply.")
    SHA.add_argparse_args(ap)
    args = ap.parse_args()

    V10.STATE_ROOT = args.state_root.resolve()
    if args.shape_warm_start_from_first:
        ws_path = args.first_dir.resolve() / "theta_hat.json"
        if not ws_path.is_file():
            sys.exit(f"--shape-warm-start-from-first: missing {ws_path}; run setup1 first")
        ws_data = json.loads(ws_path.read_text())
        ws_th = ws_data.get("theta_hat", {})
        if not all(k in ws_th for k in ("n", "eta", "sigma_y")):
            sys.exit(f"{ws_path}: missing theta_hat.{{n,eta,sigma_y}}")
        args.shape_warm_start_z = np.array([
            float(ws_th["n"]),
            float(np.log(max(float(ws_th["eta"]), 1e-9))),
            float(np.log(max(float(ws_th["sigma_y"]), 1e-9))),
        ], dtype=np.float64)
        print(f"[warm-start] x0 from {ws_path}: theta = "
              f"(n={ws_th['n']:.4f}, eta={ws_th['eta']:.4f}, "
              f"sigma_y={ws_th['sigma_y']:.4f})")
    out = estimate(args.first_dir.resolve(), args.second_dir.resolve(), args)
    (args.second_dir / "theta_hat.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    th = out["theta_hat"]
    if "theta_ci_95" in out:
        ci = out["theta_ci_95"]
        print(
            f"\n=== Joint θ̂ MAP and 95 % credible intervals (Laplace approx.) ==="
            f"\n  n        = {th['n']:>8.4f}    CI 95%: [{ci['n'][0]:>8.4f}, {ci['n'][1]:>8.4f}]"
            f"\n  η        = {th['eta']:>8.4f}    CI 95%: [{ci['eta'][0]:>8.4f}, {ci['eta'][1]:>8.4f}]"
            f"\n  σ_y      = {th['sigma_y']:>8.4f}    CI 95%: [{ci['sigma_y'][0]:>8.4f}, {ci['sigma_y'][1]:>8.4f}]"
        )


if __name__ == "__main__":
    main()
