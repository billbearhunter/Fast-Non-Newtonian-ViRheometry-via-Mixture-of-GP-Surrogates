"""Post-hoc GP variance calibration via random hold-out.

The original `calibrate_subs.py` reads pre-stored val tensors from the
checkpoint, but the v10_yshape_planB checkpoints have empty val tensors
(training pipeline at the time skipped val storage).  This script does
its own random 90/10 hold-out per sub: refits the GP on 90 % using the
already-trained hyperparameters (no Adam re-fit), predicts on the 10 %,
and computes:

    c_k = sqrt(MSE_val / mean_GP_var_val)

c_k > 1 → GP overconfident.  At inference, σ_eff = c_k * σ_GP recovers
correct heteroscedastic likelihood (Type-II ML).

Usage:
    python scripts/calibrate_subs_holdout.py --bank Models/v10_yshape_planB --gids 0
    python scripts/calibrate_subs_holdout.py --bank Models/v10_yshape_planB --dry-run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as V10
from surrogate.experts import ExactExpert


def compute_c_k(ckpt: dict, xs, ys, device, dtype, holdout_frac: float = 0.10,
                seed: int = 13) -> float:
    """Random hold-out calibration.  Refits GP train data to 90 % subset,
    predicts on 10 %, returns c_k = √(MSE / mean_var).

    Returns 1.0 if the sub is too small to split reliably (n_train < 50)."""
    X_phys = np.asarray(ckpt["X_phys"], dtype=np.float64)
    Y_phys = np.asarray(ckpt["Y_phys"], dtype=np.float64)
    n = X_phys.shape[0]
    if n < 50:
        return 1.0

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(5, int(round(n * holdout_frac)))
    val_idx = perm[:n_val]
    trn_idx = perm[n_val:]

    X_phys_trn = X_phys[trn_idx]
    Y_phys_trn = Y_phys[trn_idx]
    X_phys_val = X_phys[val_idx]
    Y_phys_val = Y_phys[val_idx]

    # Scaled (matches inference pipeline)
    X_s_trn = torch.tensor(xs.transform(X_phys_trn), dtype=dtype, device=device)
    Y_s_trn = torch.tensor(ys.transform(Y_phys_trn), dtype=dtype, device=device)
    X_s_val = torch.tensor(xs.transform(X_phys_val), dtype=dtype, device=device)
    Y_phys_val_t = torch.tensor(Y_phys_val, dtype=dtype, device=device)

    exp = ExactExpert(X_s_trn, Y_s_trn, kernel_name=ckpt.get("kernel_name", "matern25_ard")).to(device)
    exp.set_train_data(X_s_trn, Y_s_trn)
    exp.load_state_dict(ckpt["state_dict"])
    exp.eval()

    with torch.no_grad():
        mu_s, var_s = exp.predict(X_s_val)
    # OutputScaler is mean-only (engine.make_context only uses ys.mean).
    # mu_s is scaled-output (Y - y_mean); var_s is in same units.
    y_mean_t = torch.tensor(ys.mean, dtype=dtype, device=device)
    Y_pred = (mu_s + y_mean_t).cpu().numpy()
    Var_pred = var_s.cpu().numpy()

    mse = float(np.mean((Y_pred - Y_phys_val) ** 2))
    mvar = float(np.mean(Var_pred))
    if mvar < 1e-20:
        return 1.0
    return float(np.sqrt(mse / mvar))


def calibrate_gid(bank: Path, gid: int, dry_run: bool, xs, ys, device, dtype) -> list[dict]:
    gid_dir = bank / f"state_gid_{gid}"
    exp_dir = gid_dir / "experts"
    if not exp_dir.exists():
        return []
    records = []
    for pt_path in sorted(exp_dir.glob("sub_*.pt")):
        ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
        c_k = compute_c_k(ckpt, xs, ys, device, dtype)
        prev = float(ckpt.get("calib_scale", float("nan")))
        records.append({
            "file": pt_path.name,
            "sub_id": int(ckpt.get("sub_id", -1)),
            "n_train": int(ckpt.get("n_train", 0)),
            "calib_scale": c_k,
            "prev_calib": prev,
        })
        if not dry_run:
            ckpt["calib_scale"] = c_k
            torch.save(ckpt, pt_path)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--gids", type=str, default=None,
                    help="Comma-separated gids (default: all)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--holdout-frac", type=float, default=0.10)
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

    all_records: list[dict] = []
    for gid in gids:
        recs = calibrate_gid(args.bank, gid, args.dry_run, xs, ys, device, dtype)
        if not recs:
            continue
        c_arr = np.array([r["calib_scale"] for r in recs])
        print(f"\n=== gid={gid}  ({len(recs)} subs) ===")
        print(f"  c_k:  min={c_arr.min():.3f}  median={np.median(c_arr):.3f}  "
              f"max={c_arr.max():.3f}  mean={c_arr.mean():.3f}")
        for r in recs:
            flag = ""
            if r["calib_scale"] > 2.0:
                flag = "  ⚠ very overconfident"
            elif r["calib_scale"] > 1.5:
                flag = "  ⚠ overconfident"
            elif r["calib_scale"] < 0.5:
                flag = "  ⚠ very conservative"
            print(f"    sub_{r['sub_id']:04d} (n={r['n_train']}):  c_k={r['calib_scale']:.3f}{flag}")
        all_records.extend(recs)

    if args.dry_run:
        print(f"\n[DRY RUN] No files modified.")
    else:
        print(f"\nSaved calib_scale to {len(all_records)} checkpoint(s).")


if __name__ == "__main__":
    main()
