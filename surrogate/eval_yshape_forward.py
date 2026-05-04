"""Forward-error + sub-routing-accuracy evaluation for y-shape arch.

Three reports:
  1. Routing accuracy on val (truth_sub vs routed_sub from y_obs)
     Truth assigned during partition fit. Routed via length-bin + PCA + KMeans.
     Self-consistency check: should be near 100% in dense regions.

  2. Routing accuracy on test (truth not in partition fit)
     truth_sub = partition's nearest-shape lookup if hadn't been in fit.
     "matches expected" = routed sub's training theta-bbox contains test theta.

  3. Forward predictive error on test:
     For each test row, route y_obs -> sub. Use sub's GP to predict y at the
     known truth (n, eta, sy, W, H). Compare predicted y vs actual.
     Reports per-frame RMSE and frame-weighted relative error.

Usage:
  python -m surrogate.eval_yshape_forward --gids 5
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PIPE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs.engine import load_runtime  # noqa: E402
from surrogate.densify_yshape import route_y_to_sub  # noqa: E402
from surrogate import config as HC  # noqa: E402
from surrogate.experts import ExactExpert  # noqa: E402

FRAME_W = np.array([0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0])


def load_sub_gp(state_dir: Path, sub_id: int, xs, ys, device, dtype):
    p = state_dir / "experts" / f"sub_{sub_id:04d}.pt"
    if not p.exists():
        return None, None, None
    ck = torch.load(p, weights_only=False, map_location=device)
    X_phys = np.asarray(ck["X_phys"], dtype=np.float64)
    Y_phys = np.asarray(ck["Y_phys"], dtype=np.float64)
    X_s = torch.tensor(xs.transform(X_phys), dtype=dtype, device=device)
    Y_s = torch.tensor(ys.transform(Y_phys), dtype=dtype, device=device)
    exp = ExactExpert(X_s, Y_s, kernel_name=ck.get("kernel_name", "matern25_ard")).to(device)
    exp.set_train_data(X_s, Y_s)
    exp.load_state_dict(ck["state_dict"])
    exp.eval()
    return exp, X_phys, Y_phys


def predict_y_at_theta(exp, x_phys: np.ndarray, xs, ys, device, dtype) -> np.ndarray:
    x_t = torch.tensor(xs.transform(x_phys.reshape(1, 5)), dtype=dtype, device=device)
    with torch.no_grad():
        mu_s, _ = exp.predict(x_t)
    return (mu_s.cpu().numpy() + ys.mean).ravel()[:8]


def routing_accuracy(df: pd.DataFrame, state: dict, label: str) -> dict:
    """Route each row's y_obs and compare against truth_sub if available."""
    Y = df[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
    routed = []
    for k in range(len(df)):
        _bin, sid = route_y_to_sub(Y[k], state)
        routed.append(int(sid))
    df = df.copy()
    df["routed_sub"] = routed
    if "sub_id" in df.columns:
        match = (df.routed_sub == df.sub_id).sum()
        acc = match / len(df)
        print(f"  [{label}] routing match: {match}/{len(df)} = {100*acc:.1f}%")
        # mismatches breakdown
        miss = df[df.routed_sub != df.sub_id]
        if len(miss):
            top_miss = miss.groupby(["sub_id", "routed_sub"]).size().sort_values(ascending=False).head(5)
            print(f"  [{label}] top routing-mismatch pairs (truth_sub -> routed_sub: count):")
            for (t, r), c in top_miss.items():
                print(f"      {int(t)} -> {int(r)}: {c}")
        return {"label": label, "n": len(df), "match": int(match),
                "acc": float(acc), "df": df}
    else:
        print(f"  [{label}] no truth sub_id column; routing only")
        return {"label": label, "n": len(df), "df": df}


def forward_error(df: pd.DataFrame, state: dict, state_dir: Path,
                   xs, ys, device, dtype, label: str = "test") -> dict:
    """For each row: route y_obs -> sub -> predict y at truth -> compare to actual y."""
    Y = df[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
    X = df[["n", "eta", "sigma_y", "width", "height"]].to_numpy(dtype=np.float64)
    cache = {}
    rms_per_frame = []
    rel_per_row = []
    routed_subs = []
    for k in range(len(df)):
        _bin, sid = route_y_to_sub(Y[k], state)
        routed_subs.append(sid)
        if sid not in cache:
            exp, _, _ = load_sub_gp(state_dir, sid, xs, ys, device, dtype)
            cache[sid] = exp
        exp = cache[sid]
        if exp is None:
            continue
        y_pred = predict_y_at_theta(exp, X[k], xs, ys, device, dtype)
        err = (y_pred - Y[k]) * FRAME_W
        scale = max(float(np.mean((Y[k] * FRAME_W) ** 2)), 1e-9)
        rel_per_row.append(float(np.sqrt(np.mean(err * err) / scale)))
        rms_per_frame.append((y_pred - Y[k]) ** 2)
    if not rel_per_row:
        print(f"  [{label}] no rows could be evaluated (no GPs?)")
        return {}
    rms_per_frame = np.sqrt(np.mean(np.stack(rms_per_frame), axis=0))
    rel = np.array(rel_per_row)
    print(f"\n  [{label} forward predictive error] N_eval = {len(rel)}")
    print(f"    median rel-wrms: {np.median(rel):.4f}   mean: {rel.mean():.4f}   p90: {np.quantile(rel, 0.9):.4f}")
    print(f"    per-frame RMSE (cm): " + ", ".join(f"f{i+1}={rms_per_frame[i]:.3f}" for i in range(8)))
    return {"label": label, "n": len(rel),
            "rel_median": float(np.median(rel)),
            "rel_mean": float(rel.mean()),
            "rel_p90": float(np.quantile(rel, 0.9)),
            "rms_per_frame": rms_per_frame.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path,
                    default=PIPE / "Models" / "v10_yshape_v1")
    ap.add_argument("--val-csv", type=Path,
                    default=PIPE / "data" / "synthetic_splits_clean" / "val_v10_clean.csv")
    ap.add_argument("--test-csv", type=Path,
                    default=PIPE / "data" / "synthetic_splits_clean" / "test_v10_clean.csv")
    ap.add_argument("--gids", type=str, required=True,
                    help="Comma-separated gids to evaluate")
    a = ap.parse_args()

    geo_router, xs, ys = load_runtime()
    device, dtype = HC.DEVICE, HC.DTYPE
    wanted = {int(g) for g in a.gids.split(",")}

    val = pd.read_csv(a.val_csv)
    test = pd.read_csv(a.test_csv)
    val["gid"] = geo_router.predict(val[["width", "height"]].to_numpy()).astype(int)
    test["gid"] = geo_router.predict(test[["width", "height"]].to_numpy()).astype(int)

    for gid in sorted(wanted):
        sd = a.bank / f"state_gid_{gid}"
        with open(sd / "state.pkl", "rb") as f:
            state = pickle.load(f)
        # Recover val sub_ids from sub_assignments + infill (training labels)
        train_assign = pd.read_csv(sd / "sub_assignments.csv")
        for inf in sorted(sd.glob("infill_*.csv")):
            d = pd.read_csv(inf)
            d = d[[c for c in train_assign.columns if c in d.columns]]
            train_assign = pd.concat([train_assign, d], ignore_index=True)

        val_g = val[val.gid == gid].reset_index(drop=True)
        test_g = test[test.gid == gid].reset_index(drop=True)
        # Add truth_sub for val rows by matching (n, eta, sigma_y) to train_assign
        # (val was included in partition fit so it IS labeled)
        val_keys = list(zip(val_g["n"].round(6), val_g["eta"].round(4),
                              val_g["sigma_y"].round(4),
                              val_g["width"].round(4), val_g["height"].round(4)))
        ass_keys = {(round(r["n"], 6), round(r["eta"], 4), round(r["sigma_y"], 4),
                       round(r["width"], 4), round(r["height"], 4)): int(r["sub_id"])
                     for _, r in train_assign.iterrows()}
        val_g["sub_id"] = [ass_keys.get(k, -1) for k in val_keys]
        n_unmatched = (val_g.sub_id == -1).sum()
        if n_unmatched:
            print(f"\n[gid {gid}] val: {n_unmatched}/{len(val_g)} val rows had no labeled sub (likely floating-point key mismatch); excluding")
            val_g = val_g[val_g.sub_id != -1].reset_index(drop=True)

        print(f"\n=== gid {gid}  val_rows={len(val_g)}  test_rows={len(test_g)} ===")
        if len(val_g):
            routing_accuracy(val_g, state, label="val")
        if len(test_g):
            forward_error(test_g, state, sd, xs, ys, device, dtype, label="test")


if __name__ == "__main__":
    main()
