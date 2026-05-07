"""Bank-wide GP precision audit via 90/10 hold-out.

For every sub in every gid, refits the GP on 90 % of its training data
(re-using stored hyperparameters; no Adam re-fit) and predicts on the 10 %
hold-out.  Reports per-sub:

  * val_rel_wrms = √(mean((y_pred − y_obs)² × frame_w²) / mean(y_obs² × frame_w²))
    = the same metric as `relative_wrms` in selector.py / val_rel_wrms_median
  * val_mse = mean squared error on hold-out
  * mean_var = average GP-variance on hold-out
  * c_k = √(MSE / mean_var)  (post-hoc variance calibration)
  * n_train, n_val

Aggregates: per-gid median + p90; bank-wide median + p90.

This is the GP precision metric you need to argue paper-quality
surrogate accuracy.

Usage:
    python scripts/audit_gp_precision.py --bank Models/v10_yshape_planB
    python scripts/audit_gp_precision.py --bank ... --gids 0,5,10
    python scripts/audit_gp_precision.py --bank ... --out scripts/gp_audit.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as V10
from surrogate.experts import ExactExpert


FRAME_W = np.array([0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0], dtype=np.float64)


def _holdout_metrics(ckpt: dict, xs, ys, device, dtype,
                     holdout_frac: float = 0.10, seed: int = 13) -> dict:
    """Returns (val_rel_wrms, val_mse, mean_var, c_k, n_train, n_val) or None."""
    X_phys = np.asarray(ckpt["X_phys"], dtype=np.float64)
    Y_phys = np.asarray(ckpt["Y_phys"], dtype=np.float64)
    n = X_phys.shape[0]
    if n < 50:
        return {"val_rel_wrms": None, "val_mse": None, "mean_var": None,
                "c_k": 1.0, "n_train": n, "n_val": 0, "skipped": True}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(5, int(round(n * holdout_frac)))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]

    X_trn, Y_trn = X_phys[trn_idx], Y_phys[trn_idx]
    X_val, Y_val = X_phys[val_idx], Y_phys[val_idx]

    X_s_trn = torch.tensor(xs.transform(X_trn), dtype=dtype, device=device)
    Y_s_trn = torch.tensor(ys.transform(Y_trn), dtype=dtype, device=device)
    X_s_val = torch.tensor(xs.transform(X_val), dtype=dtype, device=device)

    exp = ExactExpert(X_s_trn, Y_s_trn,
                      kernel_name=ckpt.get("kernel_name", "matern25_ard")).to(device)
    exp.set_train_data(X_s_trn, Y_s_trn)
    exp.load_state_dict(ckpt["state_dict"])
    exp.eval()

    with torch.no_grad():
        mu_s, var_s = exp.predict(X_s_val)
    y_mean_t = torch.tensor(ys.mean, dtype=dtype, device=device)
    Y_pred = (mu_s + y_mean_t).cpu().numpy()
    Var_pred = var_s.cpu().numpy()

    err = (Y_pred - Y_val) * FRAME_W
    scale = max(float(np.mean((Y_val * FRAME_W) ** 2)), 1e-9)
    val_rel_wrms = float(np.sqrt(np.mean(err * err) / scale))

    mse = float(np.mean((Y_pred - Y_val) ** 2))
    mvar = float(np.mean(Var_pred))
    c_k = float(np.sqrt(mse / mvar)) if mvar > 1e-20 else 1.0

    return {
        "val_rel_wrms": val_rel_wrms,
        "val_mse": mse,
        "mean_var": mvar,
        "c_k": c_k,
        "n_train": int(len(trn_idx)),
        "n_val": int(n_val),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--gids", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write detailed JSON here.")
    args = ap.parse_args()

    V10.STATE_ROOT = args.bank.resolve()
    geo, xs, ys = V10.load_runtime()
    device, dtype = V10.HC.DEVICE, V10.HC.DTYPE

    if args.gids:
        gids = [int(g) for g in args.gids.split(",")]
    else:
        gids = sorted([int(p.name.split("_")[-1])
                       for p in args.bank.glob("state_gid_*")
                       if p.is_dir()])

    out_data: dict = {"per_gid": {}, "bank": {}}
    all_rwrms: list[float] = []
    all_ck: list[float] = []
    all_n_train: list[int] = []
    t0 = time.time()
    for gid in gids:
        gd = args.bank / f"state_gid_{gid}"
        exp_dir = gd / "experts"
        if not exp_dir.exists():
            continue
        gid_recs: list[dict] = []
        for pt in sorted(exp_dir.glob("sub_*.pt")):
            ckpt = torch.load(pt, map_location="cpu", weights_only=False)
            m = _holdout_metrics(ckpt, xs, ys, device, dtype)
            gid_recs.append({
                "sub_id": int(ckpt.get("sub_id", -1)),
                "file": pt.name,
                **m,
            })
        if not gid_recs:
            continue
        rwrms = [r["val_rel_wrms"] for r in gid_recs if r.get("val_rel_wrms") is not None]
        cks = [r["c_k"] for r in gid_recs if r.get("c_k") is not None]
        nts = [r["n_train"] for r in gid_recs]
        agg = {
            "n_subs": len(gid_recs),
            "rel_wrms_median": float(np.median(rwrms)) if rwrms else None,
            "rel_wrms_p90": float(np.percentile(rwrms, 90)) if rwrms else None,
            "rel_wrms_max": float(np.max(rwrms)) if rwrms else None,
            "c_k_median": float(np.median(cks)) if cks else None,
            "c_k_p90": float(np.percentile(cks, 90)) if cks else None,
            "n_train_median": int(np.median(nts)),
        }
        out_data["per_gid"][gid] = {"agg": agg, "subs": gid_recs}
        all_rwrms.extend(rwrms)
        all_ck.extend(cks)
        all_n_train.extend(nts)
        elapsed = time.time() - t0
        print(f"gid {gid:2d}: n_subs={agg['n_subs']:3d}  "
              f"rel_wrms median/p90/max = {agg['rel_wrms_median']:.3f}/"
              f"{agg['rel_wrms_p90']:.3f}/{agg['rel_wrms_max']:.3f}   "
              f"c_k median = {agg['c_k_median']:.3f}   ({elapsed:.0f}s)", flush=True)

    bank_agg = {
        "n_subs": len(all_rwrms),
        "rel_wrms_median": float(np.median(all_rwrms)) if all_rwrms else None,
        "rel_wrms_p90": float(np.percentile(all_rwrms, 90)) if all_rwrms else None,
        "rel_wrms_p99": float(np.percentile(all_rwrms, 99)) if all_rwrms else None,
        "rel_wrms_max": float(np.max(all_rwrms)) if all_rwrms else None,
        "c_k_median": float(np.median(all_ck)) if all_ck else None,
        "c_k_p90": float(np.percentile(all_ck, 90)) if all_ck else None,
        "n_train_median": int(np.median(all_n_train)) if all_n_train else None,
    }
    out_data["bank"] = bank_agg
    print()
    print("=== Bank-wide GP precision audit ===")
    print(f"  n_subs evaluated: {bank_agg['n_subs']}")
    print(f"  val_rel_wrms median = {bank_agg['rel_wrms_median']:.3f}")
    print(f"  val_rel_wrms p90    = {bank_agg['rel_wrms_p90']:.3f}")
    print(f"  val_rel_wrms p99    = {bank_agg['rel_wrms_p99']:.3f}")
    print(f"  val_rel_wrms max    = {bank_agg['rel_wrms_max']:.3f}")
    print(f"  c_k median (calibration factor) = {bank_agg['c_k_median']:.3f}")
    print(f"  n_train median = {bank_agg['n_train_median']}")
    print(f"  total wall-time: {time.time() - t0:.0f}s")

    # Worst 10 subs
    flat = []
    for gid, dd in out_data["per_gid"].items():
        for r in dd["subs"]:
            if r.get("val_rel_wrms") is not None:
                flat.append({"gid": gid, **r})
    flat.sort(key=lambda r: -r["val_rel_wrms"])
    print(f"\n=== Worst 10 subs by val_rel_wrms ===")
    for r in flat[:10]:
        print(f"  gid {r['gid']:2d}  sub_{r['sub_id']:04d}  "
              f"rel_wrms={r['val_rel_wrms']:.3f}  c_k={r['c_k']:.3f}  n_train={r['n_train']}")

    if args.out:
        args.out.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
