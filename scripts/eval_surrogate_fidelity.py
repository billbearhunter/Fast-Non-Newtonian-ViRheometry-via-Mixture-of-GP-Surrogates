"""
Surrogate Fidelity Evaluation (paper §7.2).

For each row in `data/canonical/test.csv` (42,584 held-out MPM samples):
  1. Use the production router (geometry gid + length bin + PCA/K-means
     shape + y_8 marginal-likelihood rerank) on the true y_obs to pick
     a primary sub-expert.
  2. Run the routed sub's exact-GP forward at the *true* θ.
  3. Compare predicted ŷ to ground-truth y from MPM.

Reports
-------
  - Aggregate MAE per output dim, overall R², row-max p50/p95/p99/p999
  - Per-expert R² distribution across the experts that get visited
  - Parity arrays for fig:parity (true vs predicted)

Usage
-----
    python scripts/eval_surrogate_fidelity.py \\
        --test-csv data/canonical/test.csv \\
        --out-dir  data/eval/ \\
        --state-root Models/yshape_mogp_production \\
        [--max-rows 42584]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import pickle  # noqa: E402
from collections import OrderedDict  # noqa: E402

from Optimization.libs import engine, selector  # noqa: E402


# ----------------------------------------------------------------------
# In-process caches.
# state.pkl is small (~MB) per gid → keep all 25 in RAM unconditionally.
# sub .pt loaded onto GPU is BIG (~600 MB each because of GPyTorch kernel
# state) → use bounded LRU to avoid CUDA OOM.
# ----------------------------------------------------------------------

_STATE_CACHE: dict = {}
_SUB_PT_LRU_MAX = 24  # ~14 GB at 600 MB/sub; tune for your GPU


class _LRUDict(OrderedDict):
    def __init__(self, max_size: int):
        super().__init__()
        self.max_size = int(max_size)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            evicted_key, evicted = self.popitem(last=False)
            # GPyTorch ExactExpert keeps GPU tensors; deletion + empty_cache
            # frees CUDA memory.
            del evicted
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


_SUB_PT_CACHE = _LRUDict(_SUB_PT_LRU_MAX)


def _install_caches(sub_lru_max: int = 24):
    """Monkey-patch selector to cache state.pkl + sub .pt loads."""
    global _SUB_PT_CACHE
    _SUB_PT_CACHE = _LRUDict(sub_lru_max)

    # Cache state.pkl reads (unbounded — small).
    original_load = pickle.load

    def cached_pickle_load(f):
        path = getattr(f, "name", None)
        if path is not None and path.endswith("state.pkl"):
            if path not in _STATE_CACHE:
                _STATE_CACHE[path] = original_load(f)
            return _STATE_CACHE[path]
        return original_load(f)

    selector.pickle.load = cached_pickle_load

    # LRU cache for sub .pt (bounded — GPU memory).
    original_load_sub_pt = selector._load_sub_pt

    def cached_load_sub_pt(sd, sub_id, xs, ys, device, dtype):
        key = (str(sd), int(sub_id), str(device))
        if key in _SUB_PT_CACHE:
            return _SUB_PT_CACHE[key]
        loaded = original_load_sub_pt(sd, sub_id, xs, ys, device, dtype)
        _SUB_PT_CACHE[key] = loaded
        return loaded

    selector._load_sub_pt = cached_load_sub_pt


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _make_args(weight_mode: str = "y8_quantile") -> SimpleNamespace:
    """Production routing defaults (mirrors selector.add_argparse_args)."""
    return SimpleNamespace(
        shape_top_k=1,
        shape_l2_mode="hard",
        shape_l2_eps_cm=0.5,
        shape_weight_mode=weight_mode,
        shape_gpfit_oversample=3,
        shape_eps_cap_by_primary=False,
        shape_topk_softmax_tau=1.5,
        shape_frame_w=None,
        shape_per_sub_z_scale=False,
        shape_box_pad=0.10,
    )


def _to_z_phys(n: float, eta: float, sigma_y: float) -> np.ndarray:
    """Physical θ → z = (n, log η, log σ_Y)."""
    return np.array([
        float(n),
        float(np.log(max(eta, 1e-12))),
        float(np.log(max(sigma_y, 1e-12))),
    ], dtype=np.float64)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    if y_true.size == 0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-csv", type=Path,
                    default=REPO / "data" / "canonical" / "test.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "eval")
    ap.add_argument("--state-root", type=Path,
                    default=engine.STATE_ROOT)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Subsample N rows (default: all). Random shuffle to "
                         "avoid gid-order bias in canonical/test.csv.")
    ap.add_argument("--shuffle-seed", type=int, default=42,
                    help="Seed for random shuffle when --max-rows < total.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--sub-cache-max", type=int, default=24,
                    help="Max sub .pt entries cached on GPU (LRU). "
                         "Each sub holds ~600 MB GPU memory; cap at 24 keeps "
                         "headroom on a 16 GB GPU.")
    ap.add_argument("--weight-mode", type=str, default="y8_quantile",
                    choices=["pca", "gp_fit", "y8_quantile"],
                    help="Sub-level routing re-rank: y8_quantile (production, "
                         "tuned for real-sim gap robustness), gp_fit (legacy, "
                         "best raw-y match), pca (no rerank, PCA distance).")
    args = ap.parse_args()

    engine.STATE_ROOT = args.state_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float64

    print(f"[eval_fidelity] state-root = {engine.STATE_ROOT}")
    print(f"[eval_fidelity] test-csv   = {args.test_csv}")
    print(f"[eval_fidelity] device     = {device}")
    _install_caches(sub_lru_max=args.sub_cache_max)
    print(f"[eval_fidelity] caches: state.pkl unbounded, sub .pt LRU max={args.sub_cache_max}")
    geo_router, xs, ys = engine.load_runtime()
    routing_args = _make_args(weight_mode=args.weight_mode)
    print(f"[eval_fidelity] weight-mode = {args.weight_mode}")

    df = pd.read_csv(args.test_csv)
    # canonical/test.csv is concatenated per-gid (first 25k rows are gid=0,
    # then gid=1 etc.). Always shuffle so a sub-sample is gid-balanced.
    df = df.sample(frac=1.0, random_state=args.shuffle_seed).reset_index(drop=True)
    if args.max_rows is not None:
        df = df.head(args.max_rows)
    n_rows = len(df)
    print(f"[eval_fidelity] {n_rows} test rows loaded (shuffled, seed={args.shuffle_seed})")

    y_cols = [f"x_0{i}" if i < 10 else f"x_{i}" for i in range(1, 9)]
    y_true_all = df[y_cols].to_numpy(dtype=np.float64)

    # Storage
    y_pred_all = np.zeros_like(y_true_all)
    sigma_pred_all = np.zeros_like(y_true_all)
    sub_ids = np.full(n_rows, -1, dtype=np.int64)
    gids = np.full(n_rows, -1, dtype=np.int64)
    fail_mask = np.zeros(n_rows, dtype=bool)

    # Cache routed setups by (gid, sub_id) — but each row has different y_obs
    # so the routing must run per-row.  The dominant cost is GP forward;
    # routing itself (~50 ms per row) stays manageable.
    t0 = time.perf_counter()
    for i, row in enumerate(df.itertuples(index=False)):
        n_t = float(row.n)
        eta_t = float(row.eta)
        sigma_y_t = float(row.sigma_y)
        W = float(row.width)
        H = float(row.height)
        y_obs = y_true_all[i]
        try:
            item = selector.prepare_setup_for_inverse(
                f"row{i}", W, H, y_obs, geo_router, xs, ys, device, dtype, routing_args
            )
            primary = item["members"][0]
            ctx = item["ctx"]
            z_true = _to_z_phys(n_t, eta_t, sigma_y_t)
            y_hat, sig = selector.predict_full_batched(primary, [z_true], ctx)
            y_pred_all[i] = y_hat[0]
            sigma_pred_all[i] = sig[0]
            sub_ids[i] = int(primary.sub_id)
            gids[i] = int(item["gid"])
        except Exception as exc:
            fail_mask[i] = True
            if i < 20 or (i % args.progress_every) == 0:
                print(f"  [row {i}] FAILED: {exc}")

        if (i + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta_sec = (n_rows - (i + 1)) / max(rate, 1e-6)
            print(f"  [{i+1}/{n_rows}]  rate={rate:.1f} rows/s  ETA={eta_sec/60:.1f} min")

    elapsed = time.perf_counter() - t0
    n_ok = int((~fail_mask).sum())
    print(f"\n[eval_fidelity] {n_ok}/{n_rows} rows succeeded in {elapsed/60:.1f} min")
    print(f"  ({n_rows - n_ok} failures, mostly out-of-box geometries)")

    # Drop failures
    keep = ~fail_mask
    y_true_ok = y_true_all[keep]
    y_pred_ok = y_pred_all[keep]
    sigma_pred_ok = sigma_pred_all[keep]
    sub_ids_ok = sub_ids[keep]
    gids_ok = gids[keep]

    # ----- Aggregate metrics ---------------------------------------------
    abs_err = np.abs(y_pred_ok - y_true_ok)
    mae_per_dim = abs_err.mean(axis=0)
    mae_overall = float(abs_err.mean())
    row_max_err = abs_err.max(axis=1)
    p50, p95, p99, p999 = np.percentile(row_max_err, [50, 95, 99, 99.9])
    r2_per_dim = np.array(
        [_r2(y_true_ok[:, d], y_pred_ok[:, d]) for d in range(8)],
        dtype=np.float64,
    )
    r2_overall = _r2(y_true_ok.ravel(), y_pred_ok.ravel())

    summary = {
        "n_rows_total": n_rows,
        "n_rows_succeeded": n_ok,
        "elapsed_min": elapsed / 60.0,
        "MAE_per_dim_cm": mae_per_dim.tolist(),
        "MAE_overall_cm": mae_overall,
        "R2_per_dim": r2_per_dim.tolist(),
        "R2_overall": r2_overall,
        "row_max_err_p50_cm": float(p50),
        "row_max_err_p95_cm": float(p95),
        "row_max_err_p99_cm": float(p99),
        "row_max_err_p999_cm": float(p999),
        "n_unique_subs_visited": int(np.unique(sub_ids_ok).size),
        "n_unique_gids_visited": int(np.unique(gids_ok).size),
    }
    out_summary = args.out_dir / "surrogate_fidelity_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Aggregate ===")
    print(f"  MAE overall           : {mae_overall:.4f} cm")
    print(f"  R2  overall           : {r2_overall:.6f}")
    print(f"  Row-max err p50/95/99 : {p50:.3f} / {p95:.3f} / {p99:.3f} cm")
    print(f"  Distinct subs visited : {summary['n_unique_subs_visited']} / 1396")
    print(f"  -> {out_summary}")

    # ----- Parity data (for fig:parity) ----------------------------------
    out_parity = args.out_dir / "parity_data.npz"
    np.savez_compressed(
        out_parity,
        y_true=y_true_ok,
        y_pred=y_pred_ok,
        sigma_pred=sigma_pred_ok,
        sub_ids=sub_ids_ok,
        gids=gids_ok,
    )
    print(f"  → {out_parity} ({y_true_ok.size * 8 / 1e6:.1f} MB scale)")

    # ----- Per-expert R² (for fig:expert-r2) -----------------------------
    sub_r2_rows = []
    unique = np.unique(sub_ids_ok)
    for sub_id in unique:
        sel = (sub_ids_ok == sub_id)
        if int(sel.sum()) < 5:
            continue  # too few samples to compute meaningful R²
        ytt = y_true_ok[sel]
        ypp = y_pred_ok[sel]
        sub_r2_rows.append({
            "sub_id": int(sub_id),
            "gid": int(gids_ok[sel][0]),
            "n_samples": int(sel.sum()),
            "R2_overall": _r2(ytt.ravel(), ypp.ravel()),
            "MAE_overall_cm": float(np.mean(np.abs(ytt - ypp))),
        })
    df_r2 = pd.DataFrame(sub_r2_rows).sort_values("R2_overall", ascending=False)
    out_r2 = args.out_dir / "expert_r2.csv"
    df_r2.to_csv(out_r2, index=False)
    if len(df_r2) > 0:
        print(f"  R2 per expert: median={df_r2['R2_overall'].median():.4f}  "
              f"p5={df_r2['R2_overall'].quantile(0.05):.4f}")
    print(f"  -> {out_r2}")


if __name__ == "__main__":
    main()
