"""
Surrogate Fidelity Evaluation — ORACLE routing (paper §7.2).

For each row in `data/canonical/test.csv`, look up the row's original
(gid, sub_id) by exact-value join against the per-gid `sub_assignments.csv`
files, then run the assigned sub's exact-GP forward at the row's true θ
and compare ŷ to ground-truth y.

This is the *fitting* fidelity, isolated from production routing error.
Reports the GP regression quality per sub on truly held-out data.

Differs from `eval_surrogate_fidelity.py` (deployment routing) in that
each test row is forwarded by its training-time sub assignment, not by
the production gate. The two scripts together separate:
  - GP fit error  (this script — oracle routing)
  - routing error (this script vs the deployment one — gap)

Usage
-----
    python scripts/eval_surrogate_fidelity_oracle.py \\
        --test-csv  data/canonical/test.csv \\
        --state-root Models/yshape_mogp_production \\
        --out-dir   data/eval_oracle/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Optimization.libs import engine, selector  # noqa: E402


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-csv", type=Path,
                    default=REPO / "data" / "canonical" / "test.csv")
    ap.add_argument("--state-root", type=Path, default=engine.STATE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "eval_oracle")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    engine.STATE_ROOT = args.state_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float64
    print(f"[oracle] state-root = {engine.STATE_ROOT}")
    print(f"[oracle] device     = {device}")

    # ── Build oracle index by joining test.csv ↔ sub_assignments.csv ────
    t0 = time.perf_counter()
    state_root = args.state_root.resolve()
    parts = []
    for d in sorted(state_root.glob("state_gid_*")):
        gid = int(d.name.replace("state_gid_", ""))
        f = d / "sub_assignments.csv"
        if f.exists():
            df_g = pd.read_csv(f)
            df_g["gid"] = gid
            parts.append(df_g)
    sa = pd.concat(parts, ignore_index=True)

    test = pd.read_csv(args.test_csv)
    if args.max_rows is not None:
        test = test.head(args.max_rows)
    n_rows = len(test)

    key_cols = ["n", "eta", "sigma_y", "width", "height"] + [f"x_0{i}" for i in range(1, 9)]
    sa[key_cols] = sa[key_cols].round(8)
    test_key = test.copy()
    test_key[key_cols] = test_key[key_cols].round(8)
    merged = test_key.merge(
        sa[key_cols + ["gid", "sub_id"]],
        on=key_cols, how="left",
    )
    n_matched = int(merged["sub_id"].notna().sum())
    print(f"[oracle] matched {n_matched}/{n_rows} rows to (gid, sub_id) "
          f"in {time.perf_counter()-t0:.1f}s")
    if n_matched < n_rows:
        unmatched = merged["sub_id"].isna().sum()
        print(f"[oracle] WARNING: {unmatched} rows could not be matched")

    # ── Group rows by (gid, sub_id), batch-forward per sub ─────────────
    geo_router, xs, ys = engine.load_runtime()  # for xs/ys; geo_router unused

    y_cols = [f"x_0{i}" for i in range(1, 9)]
    y_true_all = test_key[y_cols].to_numpy(dtype=np.float64)
    y_pred_all = np.zeros_like(y_true_all)
    sigma_pred_all = np.zeros_like(y_true_all)
    sub_ids_all = merged["sub_id"].to_numpy()
    gids_all = merged["gid"].to_numpy()
    fail_mask = np.zeros(n_rows, dtype=bool)

    # Pre-compute (n, log eta, log sigma_y) z arrays
    n_t = test_key["n"].to_numpy(dtype=np.float64)
    eta_t = test_key["eta"].to_numpy(dtype=np.float64)
    sy_t = test_key["sigma_y"].to_numpy(dtype=np.float64)
    W_t = test_key["width"].to_numpy(dtype=np.float64)
    H_t = test_key["height"].to_numpy(dtype=np.float64)
    z_all = np.column_stack([
        n_t,
        np.log(np.maximum(eta_t, 1e-12)),
        np.log(np.maximum(sy_t, 1e-12)),
    ])

    # Group by (gid, sub_id)
    groups = merged.groupby(["gid", "sub_id"]).indices  # dict (gid,sub_id) -> array idx
    print(f"[oracle] {len(groups)} unique (gid, sub_id) groups to forward")

    t1 = time.perf_counter()
    n_done = 0
    for (gid, sub_id), idxs in groups.items():
        if pd.isna(sub_id):
            fail_mask[idxs] = True
            continue
        gid = int(gid)
        sub_id = int(sub_id)
        sd = state_root / f"state_gid_{gid}"
        try:
            exp, _ = selector._load_sub_pt(sd, sub_id, xs, ys, device, dtype)
        except Exception as exc:
            print(f"  [(gid={gid}, sub={sub_id})] load fail: {exc}")
            fail_mask[idxs] = True
            continue

        # Each row in this group has its own (W, H); build the (M, 5) input
        # in scaled space.
        M = len(idxs)
        x_log = np.column_stack([z_all[idxs], W_t[idxs], H_t[idxs]])  # (M, 5)
        x_log_t = torch.tensor(x_log, dtype=dtype, device=device)
        x_mean = torch.tensor(xs.mean, dtype=dtype, device=device)
        x_std = torch.tensor(xs.std, dtype=dtype, device=device)
        y_mean = torch.tensor(ys.mean, dtype=dtype, device=device)
        X_s = (x_log_t - x_mean) / x_std
        with torch.no_grad():
            mu_s, var_s = exp.predict(X_s)
        y_hat = (mu_s + y_mean).cpu().numpy()
        sigma_hat = torch.sqrt(torch.clamp(var_s, min=1e-12)).cpu().numpy()
        y_pred_all[idxs] = y_hat
        sigma_pred_all[idxs] = sigma_hat
        # Free GPU memory
        del exp, x_log_t, X_s, mu_s, var_s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_done += M
        if n_done % 5000 < M:
            elapsed = time.perf_counter() - t1
            rate = n_done / max(elapsed, 1e-3)
            eta_sec = (n_rows - n_done) / max(rate, 1e-6)
            print(f"  [{n_done}/{n_rows}]  rate={rate:.1f} rows/s  ETA={eta_sec/60:.1f} min")

    elapsed = time.perf_counter() - t1
    n_ok = int((~fail_mask).sum())
    print(f"\n[oracle] {n_ok}/{n_rows} forwards in {elapsed/60:.1f} min")

    keep = ~fail_mask
    y_true_ok = y_true_all[keep]
    y_pred_ok = y_pred_all[keep]
    sigma_pred_ok = sigma_pred_all[keep]
    sub_ids_ok = sub_ids_all[keep]
    gids_ok = gids_all[keep]

    # ── Aggregate ─────────────────────────────────────────────────────
    abs_err = np.abs(y_pred_ok - y_true_ok)
    mae_per_dim = abs_err.mean(axis=0)
    mae_overall = float(abs_err.mean())
    row_max = abs_err.max(axis=1)
    p50, p95, p99, p999 = np.percentile(row_max, [50, 95, 99, 99.9])
    r2_per_dim = np.array(
        [_r2(y_true_ok[:, d], y_pred_ok[:, d]) for d in range(8)],
        dtype=np.float64,
    )
    r2_overall = _r2(y_true_ok.ravel(), y_pred_ok.ravel())

    summary = {
        "routing_mode": "oracle (training-time sub_id from sub_assignments.csv)",
        "n_rows_total": n_rows,
        "n_rows_succeeded": n_ok,
        "n_unique_subs": int(np.unique(sub_ids_ok.astype(int)).size),
        "n_unique_gids": int(np.unique(gids_ok.astype(int)).size),
        "MAE_per_dim_cm": mae_per_dim.tolist(),
        "MAE_overall_cm": mae_overall,
        "R2_per_dim": r2_per_dim.tolist(),
        "R2_overall": r2_overall,
        "row_max_err_p50_cm": float(p50),
        "row_max_err_p95_cm": float(p95),
        "row_max_err_p99_cm": float(p99),
        "row_max_err_p999_cm": float(p999),
        "elapsed_min": elapsed / 60.0,
    }
    out_summary = args.out_dir / "surrogate_fidelity_oracle_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))

    print("\n=== Oracle-routing aggregate ===")
    print(f"  MAE overall        : {mae_overall:.4f} cm")
    print(f"  R2  overall        : {r2_overall:.6f}")
    print(f"  Row-max p50/95/99  : {p50:.4f} / {p95:.4f} / {p99:.4f} cm")
    print(f"  Distinct subs      : {summary['n_unique_subs']} / 1396")
    print(f"  Distinct gids      : {summary['n_unique_gids']} / 25")
    print(f"  -> {out_summary}")

    # ── Per-sub R² + MAE table ───────────────────────────────────────
    rows = []
    for (gid, sub_id), idxs in merged.groupby(["gid", "sub_id"]).indices.items():
        if pd.isna(sub_id):
            continue
        gid = int(gid); sub_id = int(sub_id)
        sel = np.zeros(n_rows, dtype=bool); sel[idxs] = True
        sel = sel & keep
        if int(sel.sum()) < 5:
            continue
        ytt = y_true_all[sel]
        ypp = y_pred_all[sel]
        rows.append({
            "gid": gid, "sub_id": sub_id,
            "n_samples": int(sel.sum()),
            "R2_overall": _r2(ytt.ravel(), ypp.ravel()),
            "MAE_overall_cm": float(np.mean(np.abs(ytt - ypp))),
        })
    df_r2 = pd.DataFrame(rows).sort_values("R2_overall", ascending=False)
    out_r2 = args.out_dir / "expert_r2_oracle.csv"
    df_r2.to_csv(out_r2, index=False)
    if len(df_r2) > 0:
        print(f"  Per-sub R2: median={df_r2['R2_overall'].median():.4f}  "
              f"p5={df_r2['R2_overall'].quantile(0.05):.4f}  "
              f"p95={df_r2['R2_overall'].quantile(0.95):.4f}")
    print(f"  -> {out_r2}")

    out_parity = args.out_dir / "parity_data_oracle.npz"
    np.savez_compressed(
        out_parity,
        y_true=y_true_ok, y_pred=y_pred_ok, sigma_pred=sigma_pred_ok,
        sub_ids=sub_ids_ok, gids=gids_ok,
    )
    print(f"  -> {out_parity}")


if __name__ == "__main__":
    main()
