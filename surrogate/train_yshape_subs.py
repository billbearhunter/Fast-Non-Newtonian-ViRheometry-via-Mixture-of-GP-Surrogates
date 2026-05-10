"""Train one ExactExpert per (gid, sub_id) using the y-shape partition.

Reads:
  Models/v10_yshape_v1/state_gid_<g>/sub_assignments.csv
  Models/v10_yshape_v1/state_gid_<g>/infill_*.csv  (auto-merged)

Writes:
  Models/v10_yshape_v1/state_gid_<g>/experts/sub_<sub_id:04d>.pt

Usage:
  python -m surrogate.train_yshape_subs --gids 5 --max-n 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PIPE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs.engine import load_runtime  # noqa: E402
from surrogate import config as HC  # noqa: E402
from surrogate.experts import ExactExpert  # noqa: E402


def gather_sub_data(state_dir: Path, sub_id: int) -> dict:
    """Returns dict with separate train/val splits for the sub.

    Reads sub_assignments.csv (which after partition contains both train+val
    rows with "split" column), filters to sub_id, and partitions into
    train/val based on the split column. If split column is absent, all
    rows are treated as training.
    """
    df = pd.read_csv(state_dir / "sub_assignments.csv")
    keep_cols = list(df.columns)
    for inf in sorted(state_dir.glob("infill_*.csv")):
        d = pd.read_csv(inf)
        d = d[[c for c in keep_cols if c in d.columns]]
        df = pd.concat([df, d], ignore_index=True)
    df = df[df.sub_id == sub_id]
    if len(df) == 0:
        return {"X_train": np.empty((0, 5)), "Y_train": np.empty((0, 8)),
                "X_val": np.empty((0, 5)), "Y_val": np.empty((0, 8))}
    if "split" in df.columns:
        df_train = df[df.split == "train"]
        df_val = df[df.split == "val"]
    else:
        df_train = df
        df_val = df.iloc[:0]
    cols_X = ["n", "eta", "sigma_y", "width", "height"]
    cols_Y = [f"x_{i:02d}" for i in range(1, 9)]
    # Return physical (n, eta, sigma_y, W, H) — log transform applied later in
    # train_one_sub just before xs.transform, so X_phys saved in checkpoint
    # remains physical for z_support_from_X / member_nearest_z in engine.py.
    return {
        "X_train": df_train[cols_X].to_numpy(dtype=np.float64),
        "Y_train": df_train[cols_Y].to_numpy(dtype=np.float64),
        "X_val":   df_val[cols_X].to_numpy(dtype=np.float64),
        "Y_val":   df_val[cols_Y].to_numpy(dtype=np.float64),
    }


def _phys_to_log(X: np.ndarray) -> np.ndarray:
    """Physical (n, eta, sigma_y, W, H) → (n, log_eta, log_sigma_y, W, H).

    The V6 InputScaler (xs) was fit on log(eta) and log(sigma_y).  Applying
    this transform before xs.transform() aligns training with the inference
    path in engine.py::predict_one(), where z = (n, log_eta, log_sigma_y) is
    passed directly through xs.  X_phys stored in the checkpoint stays in
    physical units so that z_support_from_X / member_nearest_z in engine.py
    can still apply their own log conversion.
    """
    X = X.copy()
    X[:, 1] = np.log(np.clip(X[:, 1], 1e-12, None))  # log(eta)
    X[:, 2] = np.log(np.clip(X[:, 2], 1e-12, None))  # log(sigma_y)
    return X


def train_one_sub(state_dir: Path, sub_id: int, xs, ys, device, dtype,
                   max_n: int, n_iters: int, lr: float) -> dict | None:
    data = gather_sub_data(state_dir, sub_id)
    X_train, Y_train = data["X_train"], data["Y_train"]
    X_val, Y_val = data["X_val"], data["Y_val"]
    if len(X_train) < 20:
        return None
    if len(X_train) > max_n:
        rng = np.random.default_rng(7 * sub_id + 1)
        idx = rng.choice(len(X_train), max_n, replace=False)
        X_train = X_train[idx]
        Y_train = Y_train[idx]
    # Log-transform eta/sigma_y before InputScaler (xs was fit on log-space).
    # X_train keeps physical values for saving in checkpoint (X_phys).
    X_s = torch.tensor(xs.transform(_phys_to_log(X_train)), dtype=dtype, device=device)
    Y_s = torch.tensor(ys.transform(Y_train), dtype=dtype, device=device)
    exp = ExactExpert(X_s, Y_s, kernel_name="matern25_ard").to(device)
    exp.set_train_data(X_s, Y_s)
    t0 = time.time()
    try:
        final_loss = exp.fit(n_iters=n_iters, lr=lr, verbose=False)
    except Exception as e:
        _emsg = str(e).lower()
        if any(k in _emsg for k in ("svd", "singular", "linalg", "converge", "ill-conditioned")):
            # Small subs in log-space can cluster tightly → near-rank-deficient kernel
            # → SVD failure in Matern 2.5 add_low_rank path. Retry with tiny jitter.
            print(f"    [warn] sub {sub_id}: SVD/linalg error (N={len(X_train)}), retry with noise")
            X_s2 = X_s + torch.randn_like(X_s) * 1e-4
            exp = ExactExpert(X_s2, Y_s, kernel_name="matern25_ard").to(device)
            exp.set_train_data(X_s2, Y_s)
            try:
                final_loss = exp.fit(n_iters=n_iters, lr=lr, verbose=False)
            except Exception as e2:
                print(f"    [warn] sub {sub_id}: retry also failed ({e2}), skipping")
                return None
        else:
            raise
    dt = time.time() - t0
    exp.eval()

    # ---- Validation forward pass ----
    val_metrics = {"n_val": int(len(X_val))}
    Y_val_pred = np.empty((0, 8))
    val_rmse = float("nan")
    val_rel_wrms = float("nan")
    Y_val_var = np.empty((0, 8))
    if len(X_val) > 0:
        with torch.no_grad():
            X_val_s = torch.tensor(xs.transform(_phys_to_log(X_val)), dtype=dtype, device=device)
            mu_s, var_s = exp.predict(X_val_s)
            Y_val_pred = mu_s.cpu().numpy() + ys.mean
            # GP posterior variance in scaled output space (diagonal per output dim)
            Y_val_var = var_s.cpu().numpy()   # shape (N_val, D_Y)
        err = Y_val_pred - Y_val
        val_rmse = float(np.sqrt(np.mean(err * err)))
        FRAME_W = np.array([0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0])
        per_row = []
        for k in range(len(Y_val)):
            scale = max(float(np.mean((Y_val[k] * FRAME_W) ** 2)), 1e-9)
            d = (Y_val_pred[k] - Y_val[k]) * FRAME_W
            per_row.append(float(np.sqrt(np.mean(d * d) / scale)))
        val_rel_wrms = float(np.median(per_row))
        # GP calibration: mean std vs mean absolute error (lower = better calibrated)
        mean_gp_std = float(np.sqrt(Y_val_var).mean())
        mean_abs_err = float(np.abs(err).mean())
        val_metrics.update({
            "val_rmse": val_rmse,
            "val_rel_wrms_median": val_rel_wrms,
            "val_rel_wrms_p90": float(np.quantile(per_row, 0.9)),
            "val_gp_std_mean": mean_gp_std,
            "val_abs_err_mean": mean_abs_err,
            "val_calibration_ratio": mean_gp_std / (mean_abs_err + 1e-9),
        })

    return {
        # ---- backward-compatible fields (same shape as basin_gp pt) ----
        "state_dict": exp.state_dict(),
        "X_phys": X_train, "Y_phys": Y_train,  # train data
        "kernel_name": "matern25_ard",
        "sub_id": int(sub_id),
        "n_train": int(len(X_train)),
        "n_iters": int(n_iters),
        "final_loss": float(final_loss),
        "fit_seconds": float(dt),
        # ---- new: val data + forward metrics + GP confidence ----
        "X_val_phys": X_val, "Y_val_phys": Y_val,
        "Y_val_pred": Y_val_pred,
        "Y_val_var": Y_val_var,   # GP posterior variance (scaled space, shape N_val×8)
        **val_metrics,
    }


def train_gid(state_dir: Path, gid: int, xs, ys, device, dtype,
               max_n: int, n_iters: int, lr: float, force: bool) -> list[dict]:
    import pickle
    with open(state_dir / "state.pkl", "rb") as f:
        state = pickle.load(f)
    out_dir = state_dir / "experts"
    out_dir.mkdir(exist_ok=True)
    rows = []
    # Support both legacy two-layer state (has "bins") and flat KMeans state
    if state.get("partition_type") == "flat_kmeans_separate":
        sub_ids = sorted(s["sub_id"]
                         for c in state["flat_clusters"] for s in c["subs"])
    else:
        sub_ids = sorted(s["sub_id"] for b in state["bins"] for s in b["subs"])
    print(f"\n=== gid {gid}: {len(sub_ids)} subs ===")
    for sid in sub_ids:
        out_path = out_dir / f"sub_{sid:04d}.pt"
        if out_path.exists() and not force:
            print(f"  sub {sid:>3d}: exists, skip")
            rows.append({"sub_id": sid, "status": "skip"})
            continue
        try:
            ck = train_one_sub(state_dir, sid, xs, ys, device, dtype, max_n, n_iters, lr)
        except Exception as e:
            import traceback
            print(f"  sub {sid:>3d}: EXCEPTION ({type(e).__name__}: {e}), skipping", flush=True)
            traceback.print_exc()
            rows.append({"sub_id": sid, "status": f"exception: {type(e).__name__}"})
            # Try to release memory before continuing
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
        if ck is None:
            print(f"  sub {sid:>3d}: too few points, skip")
            rows.append({"sub_id": sid, "status": "empty"})
            continue
        torch.save(ck, out_path)
        val_str = ""
        if ck.get("n_val", 0) > 0:
            val_str = f"  val_rwrms={ck['val_rel_wrms_median']:.3f} (n_val={ck['n_val']})"
        print(f"  sub {sid:>3d}: N={ck['n_train']:>4d}  loss={ck['final_loss']:+.3f}  "
              f"{ck['fit_seconds']:.1f}s{val_str}")
        rows.append({"sub_id": sid, "status": "ok",
                      "N": ck["n_train"], "loss": ck["final_loss"],
                      "seconds": ck["fit_seconds"],
                      "n_val": ck.get("n_val", 0),
                      "val_rel_wrms": ck.get("val_rel_wrms_median", float("nan"))})
        # Release GPU memory between subs to prevent cumulative OOM
        # (PyTorch's caching allocator doesn't free promptly otherwise; we've
        # observed crashes after ~10-12 subs with N≈1000 each on 17 GB GPUs).
        del ck
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path,
                    default=PIPE / "Models" / "v10_yshape_v1")
    ap.add_argument("--gids", type=str, default="all",
                    help="Comma-separated gids or 'all'")
    ap.add_argument("--max-n", type=int, default=1000)
    ap.add_argument("--n-iters", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if sub_<id>.pt already exists")
    args = ap.parse_args()

    geo_router, xs, ys = load_runtime()
    device, dtype = HC.DEVICE, HC.DTYPE

    gid_dirs = sorted(args.bank.glob("state_gid_*"))
    if args.gids != "all":
        wanted = {int(g) for g in args.gids.split(",")}
        gid_dirs = [d for d in gid_dirs if int(d.name.split("_")[-1]) in wanted]

    t_total = time.time()
    for gd in gid_dirs:
        gid = int(gd.name.split("_")[-1])
        train_gid(gd, gid, xs, ys, device, dtype, args.max_n, args.n_iters, args.lr,
                   force=args.force)
    print(f"\ntotal: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
