"""Forward-predict hold-out y_obs from a known θ̂ + (W, H) using the surrogate.

Usage:
    python scripts/holdout_forward.py \\
        --theta-json freshness/data/ref_Katakuriko_2.0_6.5_2/theta_hat.json \\
        --holdout-dir freshness/data/ref_Katakuriko_2.9_4.2_3 \\
        --state-root Models/yshape_mogp_production

Reports per-frame predicted y_hat, actual y_obs, and residual stats.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as ENGINE
from Optimization.libs import selector as SHA


_REF_RE = re.compile(r"^ref_(?P<material>.+)_(?P<H>[0-9.]+)_(?P<W>[0-9.]+)_(?P<idx>[0-9]+)$")


def parse_ref_dir(p: Path) -> dict:
    m = _REF_RE.match(p.name)
    if not m:
        raise ValueError(f"can't parse {p.name}")
    return {"material": m.group("material"),
            "H": float(m.group("H")),
            "W": float(m.group("W")),
            "idx": int(m.group("idx"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta-json", type=Path, required=True,
                    help="Path to theta_hat.json containing 'theta_hat': {n, eta, sigma_y}")
    ap.add_argument("--holdout-dir", type=Path, required=True,
                    help="ref_<material>_<H>_<W>_<idx>/ with y_obs.npy")
    ap.add_argument("--state-root", type=Path, default=ENGINE.STATE_ROOT)
    # routing args (must match estimate_*_setup)
    ap.add_argument("--shape-l2-mode", default="eps")
    ap.add_argument("--shape-l2-eps-cm", type=float, default=0.5)
    ap.add_argument("--shape-eps-cap-by-primary", action="store_true", default=True)
    ap.add_argument("--shape-top-k", type=int, default=10)
    ap.add_argument("--shape-weight-mode", default="gp_fit")
    ap.add_argument("--shape-gpfit-oversample", type=int, default=1)
    ap.add_argument("--shape-noise-floor-cm", type=float, default=0.05)
    args = ap.parse_args()

    ENGINE.STATE_ROOT = args.state_root.resolve()

    # 1. Load θ̂
    with open(args.theta_json) as f:
        meta = json.load(f)
    th = meta["theta_hat"]
    n, eta, sigma_y = float(th["n"]), float(th["eta"]), float(th["sigma_y"])
    print(f"[theta] n={n:.4f}  η={eta:.3f}  σ_y={sigma_y:.3f}")

    # 2. Hold-out (W, H, y_obs)
    info = parse_ref_dir(args.holdout_dir)
    W, H = info["W"], info["H"]
    y_obs_path = args.holdout_dir / "y_obs.npy"
    if not y_obs_path.is_file():
        # fall back to output/y_obs.npy
        y_obs_path = args.holdout_dir / "output" / "y_obs.npy"
    y_obs = np.load(y_obs_path).astype(np.float64).ravel()
    print(f"[holdout] {args.holdout_dir.name}  W={W}cm  H={H}cm")
    print(f"[holdout] y_obs = {np.array2string(y_obs, precision=4)}")

    # 3. Route to find experts at this (W, H)
    geo_router, xs, ys = ENGINE.load_runtime()
    device, dtype = ENGINE.HC.DEVICE, ENGINE.HC.DTYPE
    item = SHA.prepare_setup_for_inverse(
        args.holdout_dir.name, W, H, y_obs,
        geo_router, xs, ys, device, dtype, args,
    )
    members = item["members"]
    weights = item.get("weights", None)
    print(f"[route] gid={item.get('gid')}  members={len(members)}")

    # 4. Forward predict at θ̂ for each member
    z = np.array([n, np.log(eta), np.log(sigma_y)], dtype=np.float64)
    ctx = ENGINE.make_context(xs, ys, W, H, y_obs, device, dtype)

    y_preds = []
    sigma_preds = []
    for m in members:
        y_h, s_h = SHA._predict_one(z, m, ctx)
        y_preds.append(np.asarray(y_h, dtype=np.float64))
        sigma_preds.append(np.asarray(s_h, dtype=np.float64))

    y_preds_arr = np.stack(y_preds, axis=0)             # (n_members, 8)
    sigma_preds_arr = np.stack(sigma_preds, axis=0)     # (n_members, 8)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        y_pred = (y_preds_arr * w).sum(axis=0) / w.sum()
    else:
        y_pred = y_preds_arr.mean(axis=0)

    # 5. Residuals
    resid = y_pred - y_obs
    rel_resid = resid / np.maximum(np.abs(y_obs), 1e-6)
    print()
    print("frame   y_pred    y_obs    Δ        |Δ|/|y|")
    for i in range(8):
        print(f"  {i}    {y_pred[i]:7.4f}  {y_obs[i]:7.4f}  {resid[i]:+7.4f}   {rel_resid[i]*100:+6.2f}%")
    print()
    abs_resid = np.abs(resid)
    rel_abs = abs_resid / np.maximum(np.abs(y_obs), 1e-6)
    print(f"abs |Δ|:  median={np.median(abs_resid):.4f}  p90={np.quantile(abs_resid,0.9):.4f}  cm")
    print(f"rel |Δ|:  median={np.median(rel_abs)*100:.2f}%  p90={np.quantile(rel_abs,0.9)*100:.2f}%")

    # Save
    out_path = args.holdout_dir / "holdout_forward.json"
    out = {
        "holdout_setup": {"W": W, "H": H, "name": args.holdout_dir.name},
        "theta_hat_source": str(args.theta_json),
        "theta_hat": {"n": n, "eta": eta, "sigma_y": sigma_y},
        "y_obs": y_obs.tolist(),
        "y_pred": y_pred.tolist(),
        "residual_cm": resid.tolist(),
        "residual_relative": rel_resid.tolist(),
        "abs_residual_median_cm": float(np.median(abs_resid)),
        "abs_residual_p90_cm": float(np.quantile(abs_resid, 0.9)),
        "rel_residual_median": float(np.median(rel_abs)),
        "rel_residual_p90": float(np.quantile(rel_abs, 0.9)),
        "n_members": len(members),
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
