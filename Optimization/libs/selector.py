"""Y-shape selector — deterministic L1/L2/L3 routing + single-sub CMA-ES inverse.

Architecture:
  L1: geo router on (W, H) -> gid
  L2: log-equal-width y8 bin lookup -> bin_id
  L3: PCA shape projection + nearest KMeans centroid -> sub_id
  ↓
  ONE sub GP is loaded; CMA-ES runs the inverse within its training z-bbox.

Design notes:
  * No admission scoring, no top-K, no rerank. Routing is fully deterministic.
  * Per-sub trust + sigma calibration are read directly from the sub's .pt
    (val_rel_wrms_median saved during training). No magic weights.
  * Soft-route fallback (admit top-2 subs by shape distance ratio) is OPTIONAL
    and only triggers when the picked sub has high val error or the second-
    nearest centroid is comparably close.

Public entry points:
  prepare_setup_shape(setup_name, W, H, y_obs, geo_router, xs, ys, device, dtype, args)
  inverse_single_shape(item, args, device, dtype)
  inverse_double_shape(item_a, item_b, args, device, dtype)
  add_argparse_args(ap)
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cma
import numpy as np
import torch

import Optimization.libs.engine as engine
from Optimization.libs.engine import (
    clamp_z, to_z, z_to_theta, make_context, bounds_log_np,
)
from surrogate.densify_yshape import route_y_to_sub
from surrogate.experts import ExactExpert


# ---------------------------------------------------------------------------
# Inline utilities (formerly imported from selector_abd, no longer needed)
# ---------------------------------------------------------------------------
FRAME_W_NP = np.array([0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0], dtype=np.float64)
Z_SCALE_NP = np.array([0.20, 0.90, 1.20], dtype=np.float64)


def _resolve_frame_w(args) -> np.ndarray:
    """Return per-frame weights honoring --shape-frame-w override.

    args.shape_frame_w (if set, list of 8 floats) replaces the default
    [0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0] for the loss frame
    weighting.  Use it to e.g. zero out f1-f2 (surface-tension dominated
    initial frames) and concentrate the fit on f3-f8.
    """
    fw = getattr(args, "shape_frame_w", None)
    if fw is None:
        return FRAME_W_NP
    arr = np.asarray(fw, dtype=np.float64)
    if arr.shape != (8,):
        raise ValueError(f"--shape-frame-w must have 8 floats, got shape {arr.shape}")
    return arr


def relative_wrms(y_hat: np.ndarray, y_obs: np.ndarray) -> float:
    """Frame-weighted relative residual; matches engine.setup_loss linear."""
    err = (y_hat - y_obs) * FRAME_W_NP
    scale = max(float(np.mean((y_obs * FRAME_W_NP) ** 2)), 1e-9)
    return float(np.sqrt(np.mean(err * err) / scale))


def predict_full_batched(member, z_list, ctx) -> tuple[np.ndarray, np.ndarray]:
    """Batched GP forward: K z-points → (y_hat, sigma) shape (K, 8) each."""
    if len(z_list) == 0:
        return np.empty((0, 8)), np.empty((0, 8))
    x_mean, x_std, y_mean, _y_target, W_t, H_t, *_ = ctx
    z_arr = np.asarray(z_list, dtype=np.float64).reshape(-1, 3)
    z_t = torch.tensor(z_arr, dtype=ctx[0].dtype, device=ctx[0].device)
    K = z_t.shape[0]
    W_col = W_t.view(1, 1).expand(K, 1)
    H_col = H_t.view(1, 1).expand(K, 1)
    x_log = torch.cat([z_t, W_col, H_col], dim=1)
    X_s = (x_log - x_mean) / x_std
    mu_s, var_s = member.exp.predict(X_s)
    y_hat = (mu_s + y_mean).detach().cpu().numpy()
    sigma_s = torch.sqrt(torch.clamp(var_s, min=1e-12)).detach().cpu().numpy()
    return y_hat, sigma_s


# ---------------------------------------------------------------------------
# SubMember
# ---------------------------------------------------------------------------

@dataclass
class SubMember:
    """A single y-shape sub: GP + bbox + val-derived trust / sigma cal."""
    exp: ExactExpert
    sub_id: int
    bin_id: int
    basin_k: int  # alias = bin_id (for compatibility with old code that reads m.basin_k)
    z_mu: np.ndarray
    z_nn: np.ndarray
    z_lo: np.ndarray
    z_hi: np.ndarray
    z_scale: np.ndarray
    n_train: int
    final_loss: float
    val_rel_wrms_median: float = 0.0
    val_rel_wrms_p90: float = 0.0
    n_val: int = 0
    trust: float = 1.0
    sigma_cal: float = 0.0
    # Post-hoc variance calibration scalar c_k (stored in checkpoint by
    # scripts/calibrate_subs.py).  sigma_eff = calib_scale * sigma_gp.
    # c_k > 1 → GP overconfident; c_k < 1 → GP conservative.
    # Defaults to 1.0 (no-op) for checkpoints that pre-date calibration.
    calib_scale: float = 1.0
    # legacy fields some helpers expect
    p_basin: float = 1.0
    meta: dict = field(default_factory=dict)
    # Y_8 quantile criterion: median + std of training y[7] distribution.
    # Used by gp_fit_y8_quantile re-rank to score |y_obs[7] - y8_med| / y8_std.
    y8_med: float = 0.0
    y8_std: float = 1.0


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _z_bbox_from_X(X_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute (z_mu, z_lo, z_hi, z_scale) over training X."""
    n = X_phys[:, 0]
    log_eta = np.log(np.clip(X_phys[:, 1], 1e-12, None))
    log_sy = np.log(np.clip(X_phys[:, 2], 1e-12, None))
    z = np.column_stack([n, log_eta, log_sy])
    z_mu = z.mean(axis=0)
    z_lo = z.min(axis=0)
    z_hi = z.max(axis=0)
    std = z.std(axis=0, ddof=1) if len(z) > 1 else np.array([0.05, 0.5, 0.5])
    z_scale = np.maximum(std, np.array([0.035, 0.12, 0.12]))
    return z_mu, z_lo, z_hi, z_scale


def _z_nn_to_y(X_phys: np.ndarray, Y_phys: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """Nearest training z to y_obs, frame-weighted."""
    d = np.sqrt(np.mean(((Y_phys - y_obs[None, :]) * FRAME_W_NP[None, :]) ** 2, axis=1))
    z_phys = X_phys[int(np.argmin(d)), :3]
    return clamp_z(to_z(z_phys))


def _load_sub_pt(sd: Path, sub_id: int, xs, ys, device, dtype) -> tuple[ExactExpert, dict]:
    p = sd / "experts" / f"sub_{sub_id:04d}.pt"
    if not p.exists():
        raise FileNotFoundError(f"missing sub GP: {p}")
    ck = torch.load(p, weights_only=False, map_location=device)
    X_phys = np.asarray(ck["X_phys"], dtype=np.float64)
    Y_phys = np.asarray(ck["Y_phys"], dtype=np.float64)
    X_s = torch.tensor(xs.transform(X_phys), dtype=dtype, device=device)
    Y_s = torch.tensor(ys.transform(Y_phys), dtype=dtype, device=device)
    exp = ExactExpert(X_s, Y_s, kernel_name=ck.get("kernel_name", "matern25_ard")).to(device)
    exp.set_train_data(X_s, Y_s)
    exp.load_state_dict(ck["state_dict"])
    exp.eval()
    return exp, ck


def _build_sub_member(sd: Path, sub_id: int, bin_id: int, y_obs: np.ndarray,
                      xs, ys, device, dtype) -> SubMember:
    exp, ck = _load_sub_pt(sd, sub_id, xs, ys, device, dtype)
    X_phys = np.asarray(ck["X_phys"], dtype=np.float64)
    Y_phys = np.asarray(ck["Y_phys"], dtype=np.float64)
    z_mu, z_lo, z_hi, z_scale = _z_bbox_from_X(X_phys)
    z_nn = _z_nn_to_y(X_phys, Y_phys, y_obs)
    # Y_8 quantile statistics for sub-level routing (gp_fit_y8_quantile mode).
    y8_arr = Y_phys[:, 7] if Y_phys.shape[1] >= 8 else np.array([1.0])
    y8_med = float(np.median(y8_arr)) if y8_arr.size > 0 else 1.0
    y8_std = float(np.std(y8_arr, ddof=1)) if y8_arr.size > 1 else 1.0
    y8_std = max(y8_std, 0.1)  # floor to avoid divide-by-zero on degenerate subs
    val_rwrms = float(ck.get("val_rel_wrms_median", 0.0))
    val_p90 = float(ck.get("val_rel_wrms_p90", 0.0))
    n_val = int(ck.get("n_val", 0))
    trust = float(np.exp(-val_rwrms)) if val_rwrms > 0 else 1.0
    calib_scale = float(ck.get("calib_scale", 1.0))  # from calibrate_subs.py
    return SubMember(
        exp=exp, sub_id=int(sub_id), bin_id=int(bin_id), basin_k=int(bin_id),
        z_mu=z_mu, z_nn=z_nn, z_lo=z_lo, z_hi=z_hi, z_scale=z_scale,
        n_train=int(ck.get("n_train", len(X_phys))),
        final_loss=float(ck.get("final_loss", 0.0)),
        val_rel_wrms_median=val_rwrms,
        val_rel_wrms_p90=val_p90,
        n_val=n_val,
        trust=trust,
        sigma_cal=val_rwrms,
        calib_scale=calib_scale,
        p_basin=1.0,
        meta={"sub_id": int(sub_id), "bin_id": int(bin_id),
              "trust_score": trust, "mu_z": z_mu,
              "cov_z": np.diag(z_scale ** 2), "pi": 1.0},
        y8_med=y8_med, y8_std=y8_std,
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _setup_loss_log_nuisance_gp_np(y_hat, sigma_pred, y_obs, frame_w,
                                     sigma_bias, sigma_trend, sigma_obs_log) -> float:
    """GP-aware log_nuisance: per-frame variance from GP posterior.

    Replaces the implicit uniform per-frame noise with a delta-method
    log-space variance that combines GP epistemic uncertainty and a
    measurement-noise floor:
        σ_log_f² = (σ_pred_f / y_hat_f)² + σ_obs_log²

    The frame weight then becomes w_f = frame_w_f² / σ_log_f².  Frames
    where the GP is *more confident* contribute more to the loss
    (high precision); frames where it is unsure are downweighted.

    Returns proper −log p(y_obs | θ) up to constant, including the
    log-determinant 0.5 Σ log(σ_log_f²).  Without that term, CMA-ES
    would game the loss by drifting z to regions of *high* GP
    uncertainty (which inflates σ_log_f² and shrinks w_f and resid²/var).
    Normalised by frame count to keep magnitude comparable to the
    other loss modes.
    """
    eps = 1e-9
    yh = np.clip(y_hat, eps, None)
    r = np.log(yh) - np.log(np.clip(y_obs, eps, None))
    n = y_hat.size
    # delta method: log-y std ≈ σ_y / y
    sigma_log = np.asarray(sigma_pred) / yh
    var_f = sigma_log * sigma_log + sigma_obs_log * sigma_obs_log
    # combined per-frame weights = frame importance² × 1/var_log
    w = (frame_w * frame_w) / var_f
    t = np.linspace(-1.0, 1.0, n)
    X = np.stack([np.ones_like(t), t], axis=1)
    WX = X * w[:, None]
    A = X.T @ WX + np.diag([1.0 / sigma_bias ** 2, 1.0 / sigma_trend ** 2])
    b = X.T @ (w * r)
    gamma = np.linalg.solve(A, b)
    resid = r - X @ gamma
    nll = 0.5 * float(np.sum(w * resid * resid)) + 0.5 * float(np.sum(np.log(var_f)))
    return nll / n


def _predict_one(z, member, ctx):
    y, s = predict_full_batched(member, [z], ctx)
    return y[0], s[0]


def _support_penalty(z, member: SubMember) -> float:
    over = np.maximum(z - member.z_hi, 0.0)
    under = np.maximum(member.z_lo - z, 0.0)
    return float(np.sum(((over + under) / member.z_scale) ** 2))


def _hessian_ci(loss_fn, z_best, h_scale: np.ndarray, alpha: float = 0.05,
                  h_factor: float = 0.10, min_eig_frac: float = 1e-2):
    """Laplace-approx 95 % CI from numerical Hessian at MAP (Type-II ML posterior).

    Approximates posterior near z_best as N(z_best, H⁻¹) with
    H = ∇²(−log p(z|y)).  Central differences with step h_i = h_factor · z_scale_i.

    Returns
    -------
    z_std :    (3,) sqrt of H⁻¹ diag in z-space.
    theta_ci : list[(lo, hi)] length 3 — 95 % CI on (n, η, σ_y) (exp-link transform).
    flags :    list[bool] length 3 — True when direction is well-identified
               (eigenvalue at floor → False = under-identified).
    """
    z_best = np.asarray(z_best, dtype=np.float64)
    n = z_best.size
    f0 = loss_fn(z_best)
    h = np.maximum(np.minimum(np.asarray(h_scale) * h_factor, 0.30), 5e-3)
    H = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (loss_fn(z_best + ei) + loss_fn(z_best - ei) - 2.0 * f0) / (h[i] * h[i])
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = h[i]
            ej = np.zeros(n); ej[j] = h[j]
            f_pp = loss_fn(z_best + ei + ej)
            f_pm = loss_fn(z_best + ei - ej)
            f_mp = loss_fn(z_best - ei + ej)
            f_mm = loss_fn(z_best - ei - ej)
            v = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = v
    H = 0.5 * (H + H.T)
    z_std = np.full(n, float("nan"))
    flags = [True] * n
    try:
        evals, evecs = np.linalg.eigh(H)
        max_eig = float(np.abs(evals).max())
        floor_eig = max(min_eig_frac * max(max_eig, 1.0), 1e-6)
        below_floor = evals < floor_eig
        evals_pd = np.where(below_floor, floor_eig, evals)
        cov = (evecs / evals_pd) @ evecs.T
        diag = np.clip(np.diag(cov), 0.0, None)
        z_std = np.sqrt(diag)
        # Per-dim identifiability: direction i is under-identified if its
        # dominant eigenvector mode hit the floor.  Project: dim_to_mode =
        # |evecs|² (each row is a dim; sum over modes weights its support).
        dim_floor_weight = (evecs ** 2) @ below_floor.astype(np.float64)
        flags = [bool(w < 0.5) for w in dim_floor_weight]
    except np.linalg.LinAlgError:
        pass
    z_q = 1.96
    z_lo_c = np.maximum(z_best - z_q * z_std, np.array([0.0, -3.0, 0.0]))
    z_hi_c = np.minimum(z_best + z_q * z_std, np.array([1.0,  6.0, 7.0]))
    theta_ci = [
        (float(z_lo_c[0]), float(z_hi_c[0])),                    # n
        (float(np.exp(z_lo_c[1])), float(np.exp(z_hi_c[1]))),    # η
        (float(np.exp(z_lo_c[2])), float(np.exp(z_hi_c[2]))),    # σ_y
    ]
    return z_std, theta_ci, flags


# ---------------------------------------------------------------------------
# CMA-ES
# ---------------------------------------------------------------------------

def _run_cma(loss_fn, x0, lo, hi, sigma0=0.25, max_iter=30, popsize=12, seed=7):
    opts = {"bounds": [list(lo), list(hi)], "popsize": popsize, "verbose": -9,
            "tolfun": 1e-6, "tolx": 1e-5, "maxiter": max_iter, "seed": seed}
    es = cma.CMAEvolutionStrategy(list(x0), float(sigma0), opts)
    n_evals = 0
    while not es.stop():
        xs = es.ask()
        losses = [loss_fn(np.asarray(x, dtype=np.float64)) for x in xs]
        es.tell(xs, losses)
        n_evals += len(xs)
    return np.asarray(es.result.xbest, dtype=np.float64), float(es.result.fbest), n_evals


def _run_cma_multistart(loss_fn, x0, lo, hi, sigma0=0.25, max_iter=30, popsize=12,
                         base_seed=7, n_restarts=1):
    """Multi-start CMA-ES.  Run `n_restarts` times with seeds [base_seed, base_seed+100, ...],
    return best result + theta dispersion across restarts (truth-blind ridge-degeneracy
    diagnostic: large dispersion = loss landscape has multiple basins, ML estimate
    not unique).

    n_restarts=1 = legacy single-start (no overhead).  n_restarts>1 multiplies CMA cost
    proportionally; dispersion array is None when n_restarts==1.
    """
    if n_restarts <= 1:
        z, l, n_ev = _run_cma(loss_fn, x0, lo, hi, sigma0, max_iter, popsize, base_seed)
        return z, l, n_ev, None
    results = []
    total_evals = 0
    for k in range(n_restarts):
        z_k, l_k, n_k = _run_cma(loss_fn, x0, lo, hi, sigma0, max_iter, popsize,
                                  base_seed + 100 * k)
        results.append((z_k, l_k))
        total_evals += n_k
    results.sort(key=lambda r: r[1])
    z_best, l_best = results[0]
    z_arr = np.stack([r[0] for r in results])           # (n_restarts, 3) z-space
    z_dispersion = z_arr.std(axis=0)                    # 3 — std over restarts
    return z_best, l_best, total_evals, z_dispersion


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Flat KMeans routing (partition_type='flat_kmeans_separate')
# ---------------------------------------------------------------------------

def _norm_curve_one(y_obs: np.ndarray, nc_version: str = "v1") -> np.ndarray:
    """(8,) → (7,) shape descriptor.

    v1 (legacy):     (Y[1:8] - Y[0]) / Y[7]                — subtract frame1, scale by frame8
    v2 (range-norm): (Y[1:8] - Y[0]) / (Y[7] - Y[0])      — affine-invariant, sim-real robust

    The version is chosen by state["nc_version"] in callers; defaults to v1
    when missing for backward compatibility with old state.pkl files.
    """
    y1 = float(y_obs[0])
    if nc_version == "v2":
        span = max(float(y_obs[7]) - y1, 1e-6)
        return (y_obs[1:8] - y1) / span
    # v1 (default)
    y8 = max(float(y_obs[7]), 1e-6)
    return (y_obs[1:8] - y1) / y8


def _project_flat(y_obs: np.ndarray, state: dict) -> np.ndarray:
    """Project y_obs into flat KMeans feature space.

    Separate PCA (pca.n_features_in_==7):  [log_y8_s*w, pc1, pc2, pc3]  (4D)
    Joint   PCA (pca.n_features_in_==8):   pca([log_y8_s*w, nc_1..7])   (4D)
    """
    nc_version = state.get("nc_version", "v1")
    nc   = _norm_curve_one(y_obs, nc_version=nc_version)           # (7,)
    sc_nc = state["flat_scaler_nc"]
    pca   = state["flat_pca"]
    sc_y8 = state["flat_scaler_y8"]
    w     = float(state.get("y8_weight", 1.0))

    nc_s     = (nc - sc_nc.mean_) / sc_nc.scale_                  # (7,)
    log_y8   = float(np.log(max(float(y_obs[7]), 1e-6)))
    log_y8_s = float((log_y8 - sc_y8.mean_[0]) / sc_y8.scale_[0]) * w

    if pca.n_features_in_ == 8:
        # Joint PCA: PCA was fitted on [log_y8_s, nc_s] together
        raw = np.concatenate([[log_y8_s], nc_s])                   # (8,)
        feat = (raw - pca.mean_) @ pca.components_.T               # (n_pc,)
    else:
        # Separate PCA: PCA fitted on nc_s only, log_y8_s prepended
        pcs  = (nc_s - pca.mean_) @ pca.components_.T             # (n_pc,)
        feat = np.concatenate([[log_y8_s], pcs])                   # (1+n_pc,)
    return feat


def _route_flat_kmeans(y_obs: np.ndarray, state: dict,
                        k: int) -> list[tuple]:
    """Route y_obs via flat KMeans.

    Returns list of (cluster_id, sub_id, dist) tuples, sorted by ascending
    distance to cluster centroid.  When a cluster has a sigma_y split (two
    sub-experts), BOTH sub_ids are returned for that cluster so the top-K
    CMA-ES sees both candidates and picks via GP-fit at z_best.

    k = number of *clusters* to pull from (not number of subs).
    """
    feat = _project_flat(y_obs, state)
    centroids = np.asarray(state["flat_centroids"])           # (K, d)
    dists = np.linalg.norm(centroids - feat, axis=1)
    order = np.argsort(dists)

    cands: list[tuple] = []
    clusters_seen = 0
    for ci in order:
        cluster = state["flat_clusters"][int(ci)]
        d = float(dists[ci])
        for sub_entry in cluster["subs"]:
            cands.append((int(ci), int(sub_entry["sub_id"]), d))
        clusters_seen += 1
        if clusters_seen >= max(1, k):
            break
    return cands


def _pca_dist_in_bin(y_obs: np.ndarray, bin_obj: dict) -> tuple[np.ndarray, list[tuple]]:
    """Project y_obs into a single bin's PCA space and compute per-sub distances.

    Returns (y_obs_pc, [(sub_id, pca_dist), ...]).
    """
    from surrogate.densify_yshape import norm_curve_one
    nc = norm_curve_one(y_obs)
    nc_s = (nc - bin_obj["scaler"]["mean"]) / bin_obj["scaler"]["std"]
    pc = bin_obj["pca"]["components"] @ (nc_s - bin_obj["pca"]["mean"])
    out = []
    for s in bin_obj["subs"]:
        c_pc = np.asarray(s["centroid_pc"], dtype=np.float64)
        d = float(np.linalg.norm(c_pc - pc))
        out.append((int(s["sub_id"]), d))
    return pc, out


def _route_topk_subs(y_obs: np.ndarray, state: dict, k: int,
                      mode: str = "hard", eps_cm: float = 0.0,
                      eps_cap_by_primary: bool = False) -> list[tuple]:
    """Return top-K (bin_id, sub_id, pca_dist) sorted by ascending PCA distance.

    Plan E: instead of top-1, return up to K candidate subs.

    Dispatches to flat KMeans routing when state['partition_type'] ==
    'flat_kmeans_separate'. The --shape-l2-mode flag is ignored in flat mode.

    L2-gate modes (legacy two-layer partition only):
      hard      - legacy: only subs in the bin that contains y_obs[7]
      eps       - Fix A: also include neighbour bin subs when y_obs[7] is
                  within eps_cm of an edge of the primary bin
      cross_bin - C1: ignore L2; per-bin top-1 sub, ranked across bins
      hybrid    - C3: primary bin full subs + every other bin's top-1

    Note: PCA distances are NOT comparable across bins (each bin has its
    own PCA basis & scaler). When mode != hard we still sort by raw PCA
    dist as a coarse heuristic.
    """
    # Flat KMeans path (new partition)
    if state.get("partition_type") == "flat_kmeans_separate":
        return _route_flat_kmeans(y_obs, state, k)

    # Legacy two-layer (L2 bin + L3 KMeans) path
    edges = state["length_edges"]
    y8 = float(y_obs[7])
    # Primary bin via legacy hard lookup (always needed as anchor)
    bin_id = -1
    for b in state["bins"]:
        if b["y8_lo"] <= y8 < b["y8_hi"]:
            bin_id = int(b["bin_id"]); break
    if bin_id < 0:
        bin_id = int(state["bins"][0]["bin_id"]) if y8 < edges[0] \
                  else int(state["bins"][-1]["bin_id"])

    # Decide which bins to draw candidates from
    all_bins = list(state["bins"])
    bin_by_id = {int(b["bin_id"]): b for b in all_bins}
    if mode == "hard":
        cand_bin_ids = [bin_id]
    elif mode == "eps":
        primary = bin_by_id[bin_id]
        cand_bin_ids = [bin_id]
        # Spillover when y_obs[7] sits within eps_cm of a bin edge
        if (y8 - float(primary["y8_lo"])) < eps_cm and (bin_id - 1) in bin_by_id:
            cand_bin_ids.append(bin_id - 1)
        if (float(primary["y8_hi"]) - y8) < eps_cm and (bin_id + 1) in bin_by_id:
            cand_bin_ids.append(bin_id + 1)
    elif mode == "cross_bin":
        cand_bin_ids = sorted(bin_by_id.keys())  # all bins
    elif mode == "hybrid":
        cand_bin_ids = sorted(bin_by_id.keys())  # all bins, but we'll cap non-primary to top-1
    elif mode == "soft_attn":
        # Improvement (4): soft binning attention. Pull from ALL bins but
        # add bin-distance penalty to each sub's PCA distance.  Replaces
        # hard L2 + ε spillover with a smooth fall-off.
        cand_bin_ids = sorted(bin_by_id.keys())
    else:
        raise ValueError(f"unknown --shape-l2-mode={mode}")

    # Collect candidates per bin in that bin's own PCA space
    primary_n = len(bin_by_id[bin_id]["subs"])
    # Improvement (4): soft attention bin penalty
    soft_attn_gamma = 0.0
    if mode == "soft_attn":
        # default attention sharpness: γ=2 in log(y8) units
        soft_attn_gamma = 2.0  # could expose as flag if needed
    cands: list[tuple] = []
    for bid in cand_bin_ids:
        b = bin_by_id[bid]
        _, per_sub = _pca_dist_in_bin(y_obs, b)
        per_sub.sort(key=lambda x: x[1])
        bin_penalty = 0.0
        if mode == "soft_attn":
            log_y8 = float(np.log(max(y8, 1e-3)))
            log_bc = float(np.log(max(0.5 * (b["y8_lo"] + b["y8_hi"]), 1e-3)))
            bin_penalty = soft_attn_gamma * abs(log_y8 - log_bc)
        if mode == "cross_bin":
            # C1: each bin contributes its top-1
            sid, d = per_sub[0]
            cands.append((bid, sid, d))
        elif mode == "hybrid" and bid != bin_id:
            # C3: non-primary bins contribute top-1
            sid, d = per_sub[0]
            cands.append((bid, sid, d))
        else:
            # hard / eps / hybrid-primary / soft_attn: keep all subs in bin.
            # Fix A v2: when eps spillover from a non-primary bin would
            # contribute more subs than the primary bin (which biases the
            # ensemble toward whichever side has more subs — empirically
            # observed on Okonomiyaki where bin_2 spillover added 9 subs
            # vs bin_3 primary's 7, dragging σ_y to a high-σ_y compromise),
            # cap the spillover at primary_n top-PCA subs.
            if eps_cap_by_primary and mode == "eps" and bid != bin_id and len(per_sub) > primary_n:
                per_sub = per_sub[:primary_n]
            for sid, d in per_sub:
                cands.append((bid, sid, d + bin_penalty))

    cands.sort(key=lambda x: x[2])  # sort by PCA dist (cross-bin: heuristic only)
    return cands[:max(1, k)]


def prepare_setup_shape(setup_name, W, H, y_obs, geo_router, xs, ys,
                         device, dtype, args) -> dict | None:
    """Deterministic L1/L2/L3 routing → top-K SubMembers (K=1 = top-1)."""
    gid = int(geo_router.predict([[W, H]])[0])
    sd = engine.STATE_ROOT / f"state_gid_{gid}"
    if not (sd / "state.pkl").is_file():
        raise FileNotFoundError(f"missing trained state for gid={gid}: {sd}")
    if not (sd / "experts").is_dir():
        raise FileNotFoundError(f"no experts dir for gid={gid}: run train_yshape_subs.py")
    with open(sd / "state.pkl", "rb") as f:
        state = pickle.load(f)
    top_k = max(1, int(getattr(args, "shape_top_k", 1)))
    l2_mode = str(getattr(args, "shape_l2_mode", "hard"))
    eps_cm = float(getattr(args, "shape_l2_eps_cm", 0.5))
    weight_mode = str(getattr(args, "shape_weight_mode", "pca"))
    ctx = make_context(xs, ys, W, H, y_obs, device, dtype)

    # Selection (Option D): when weight_mode='gp_fit', oversample candidates
    # by PCA distance, then re-rank by GP-fit at z_nn and keep top-K. This
    # makes K=1 actually pick the single best-fitting sub instead of the
    # PCA-best (which on Chuno A is sub_15 with the largest GP error).
    #   pca   : legacy. PCA-distance ranking AND PCA-distance softmax weights.
    #   gp_fit: GP-fit-at-z_nn ranking AND softmax weights.  Cross-bin
    #           comparable.  Selection cost ≈ K × (sub-load 50–200 ms +
    #           1 GP forward).  For oversample=3 this adds at most ~1 s
    #           of routing time, dominated by sub-loading I/O.
    over_factor = int(getattr(args, "shape_gpfit_oversample", 3))
    fetch_k = top_k * over_factor if weight_mode in ("gp_fit", "gp_fit_y8_quantile") else top_k
    eps_cap = bool(getattr(args, "shape_eps_cap_by_primary", False))
    cands = _route_topk_subs(y_obs, state, fetch_k, mode=l2_mode, eps_cm=eps_cm,
                              eps_cap_by_primary=eps_cap)
    if not cands:
        if state.get("partition_type") == "flat_kmeans_separate":
            cands = _route_flat_kmeans(y_obs, state, 1)
        else:
            bin_id, sub_id = route_y_to_sub(y_obs, state)
            cands = [(bin_id, sub_id, 0.0)]
    over_members = []
    over_pca_dists = []
    for bin_id, sub_id, d in cands:
        try:
            m = _build_sub_member(sd, sub_id, bin_id, y_obs, xs, ys, device, dtype)
            over_members.append(m); over_pca_dists.append(d)
        except FileNotFoundError:
            continue

    if weight_mode == "gp_fit_y8_quantile":
        # Y_8 quantile criterion (Bayes-evidence-like sub selection).
        # Score = |y_obs[7] - y8_med_k| / y8_std_k.  Picks the sub for which
        # y_obs[7] is most "typical" (central in training Y_8 distribution).
        # Relies on cluster-level routing already filtering shape match.
        y_obs_y8 = float(y_obs[7])
        fit_dists = [abs(y_obs_y8 - m.y8_med) / max(m.y8_std, 1e-6)
                     for m in over_members]
        order = np.argsort(np.asarray(fit_dists))
        keep = order[:max(1, top_k)]
        members = [over_members[i] for i in keep]
        d_kept = np.asarray([fit_dists[i] for i in keep], dtype=np.float64)
    elif weight_mode == "gp_fit":
        # Legacy ranking: ||GP(z_nn) - y_obs|| in raw y-space.  Kept for
        # A/B comparison against gp_fit_y8_quantile.
        fit_dists = []
        for m in over_members:
            y_pred, _ = _predict_one(m.z_nn, m, ctx)
            fit_dists.append(float(np.linalg.norm(y_pred - np.asarray(y_obs, dtype=np.float64))))
        order = np.argsort(np.asarray(fit_dists))
        keep = order[:max(1, top_k)]
        members = [over_members[i] for i in keep]
        d_kept = np.asarray([fit_dists[i] for i in keep], dtype=np.float64)
    elif weight_mode == "pca":
        members = over_members[:max(1, top_k)]
        d_kept = np.asarray(over_pca_dists[:max(1, top_k)], dtype=np.float64)
    else:
        raise ValueError(f"unknown --shape-weight-mode={weight_mode}")

    if len(members) > 1:
        tau = float(getattr(args, "shape_topk_softmax_tau", 1.5))
        ws = np.exp(-(d_kept - d_kept.min()) / max(tau, 1e-3))
        ws = ws / ws.sum()
    else:
        ws = np.array([1.0])

    frame_w_used = _resolve_frame_w(args)
    return {
        "setup": setup_name, "W": W, "H": H, "y_obs": y_obs,
        "gid": gid, "state_dir": sd,
        "bin_id": int(members[0].bin_id), "sub_id": int(members[0].sub_id),
        "members": members, "member_weights": ws.tolist(), "ctx": ctx,
        "frame_w": frame_w_used,
    }


def _make_box_for_pair(ma: SubMember, mb: SubMember, fb_lo, fb_hi,
                        pad: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Joint search box. Prefer intersection; if empty, fall back to union."""
    inter_lo = np.maximum(ma.z_lo, mb.z_lo)
    inter_hi = np.minimum(ma.z_hi, mb.z_hi)
    if np.all(inter_lo < inter_hi - 1e-3):
        lo, hi = inter_lo, inter_hi
    else:
        # bboxes don't overlap → use union with padding
        lo = np.minimum(ma.z_lo, mb.z_lo) - pad * Z_SCALE_NP
        hi = np.maximum(ma.z_hi, mb.z_hi) + pad * Z_SCALE_NP
    lo = np.maximum(lo, fb_lo)
    hi = np.minimum(hi, fb_hi)
    return lo, hi


def inverse_single_shape(item, args, device, dtype) -> dict | None:
    if not item["members"]:
        return None
    members = item["members"]
    weights = np.asarray(item.get("member_weights", [1.0]*len(members)), dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-9)
    primary = members[0]
    ctx = item["ctx"]
    fb_lo_orig = ctx[6].detach().cpu().numpy().astype(np.float64)
    fb_hi = ctx[7].detach().cpu().numpy().astype(np.float64)
    fb_lo = fb_lo_orig.copy()
    if args.sigma_y_min > 0:
        fb_lo[2] = max(fb_lo[2], float(np.log(args.sigma_y_min)))
    pad = float(getattr(args, "shape_box_pad", 0.10))
    # Plan E: when top-K > 1, expand bbox to UNION of all members.
    z_lo_union = np.min(np.stack([m.z_lo for m in members]), axis=0)
    z_hi_union = np.max(np.stack([m.z_hi for m in members]), axis=0)
    zs_pad = Z_SCALE_NP
    lo = np.maximum(z_lo_union - pad * zs_pad, fb_lo)
    hi = np.minimum(z_hi_union + pad * zs_pad, fb_hi)
    # Guard: sigma_y_min floor can push lo[2] above hi[2] for lo-half subs
    # (σy-split). Fall back to global lower bound (no floor) on dim 2 only.
    if lo[2] >= hi[2] - 1e-3:
        lo[2] = max(z_lo_union[2] - pad * zs_pad[2], fb_lo_orig[2])
    x0 = np.maximum(np.minimum(primary.z_nn, hi - 1e-3), lo + 1e-3)

    sup_w = float(getattr(args, "shape_support_weight", 0.25))
    fw = item.get("frame_w", FRAME_W_NP)
    sigma_obs_log = float(getattr(args, "shape_gp_noise_log_floor", 0.05))

    n_frames = float(len(item["y_obs"]))
    log_w = np.log(np.maximum(weights, 1e-12))

    def loss(z):
        # Compute per-member NLL (= -log p_m(y|θ) / n_frames) and support penalty.
        losses = np.empty(len(members))
        sup_arr = np.empty(len(members))
        for i, m in enumerate(members):
            y_hat, sig = _predict_one(z, m, ctx)
            sig_calib = sig * m.calib_scale       # post-hoc GP variance calibration
            losses[i] = _setup_loss_log_nuisance_gp_np(y_hat, sig_calib, item["y_obs"], fw,
                                                       args.sigma_bias, args.sigma_trend,
                                                       sigma_obs_log)
            sup_arr[i] = _support_penalty(z, m)
        # Bayes-correct mixture marginal NLL (Method-B-style logsumexp):
        #   −log Σ_m π_m p_m(y|θ) = −logsumexp_m [ log π_m + log p_m(y|θ) ]
        # log p_m(y|θ) = −losses[m] · n_frames (un-normalize from per-frame avg).
        # K=1: collapses to losses[0]; this branch is taken in production y8q.
        if len(members) == 1:
            l_total = float(losses[0])
        else:
            log_terms = log_w - losses * n_frames
            mx = float(log_terms.max())
            l_total = -(mx + float(np.log(np.sum(np.exp(log_terms - mx))))) / n_frames
        sup_total = float(np.dot(weights, sup_arr))
        return l_total + sup_w * sup_total

    t0 = time.time()
    n_restarts = max(1, int(getattr(args, "shape_cma_restarts", 1)))
    z_best, l_best, n_ev, z_disp = _run_cma_multistart(
        loss, x0, lo, hi,
        sigma0=getattr(args, "shape_sigma0", 0.25),
        max_iter=getattr(args, "shape_max_iter", 30),
        popsize=getattr(args, "shape_popsize", 12),
        base_seed=7 + primary.sub_id + int(getattr(args, "shape_cma_seed_offset", 0)),
        n_restarts=n_restarts,
    )
    theta = z_to_theta(z_best)
    # Plan E: pick the member whose forward GP best matches y_obs at z_best
    best_idx = 0
    best_rwrms = float("inf")
    for i, m in enumerate(members):
        y_h, _ = _predict_one(z_best, m, ctx)
        rw = float(relative_wrms(y_h, item["y_obs"]))
        if rw < best_rwrms:
            best_rwrms = rw; best_idx = i
    sel = members[best_idx]
    y_hat, sigma_hat = _predict_one(z_best, sel, ctx)
    raw_wrms = relative_wrms(y_hat, item["y_obs"])

    # Hessian Laplace CI (Type-II ML posterior approximation).  Optional
    # (--shape-report-ci); 9 extra GP evals.  Wide CI on a dim flagged
    # !identifiable means that direction is under-identified by the data alone.
    ci_block = {}
    if bool(getattr(args, "shape_report_ci", False)):
        try:
            z_std, theta_ci, ci_flags = _hessian_ci(loss, z_best, primary.z_scale)
            ci_block = {
                "z_std": z_std.tolist(),
                "theta_ci_95": {
                    "n":       list(theta_ci[0]),
                    "eta":     list(theta_ci[1]),
                    "sigma_y": list(theta_ci[2]),
                },
                "identifiable": {
                    "n": ci_flags[0], "eta": ci_flags[1], "sigma_y": ci_flags[2]
                },
            }
        except Exception as e:
            print(f"[hessian_ci] failed: {e}")

    out = {
        "theta_n": theta[0], "theta_eta": theta[1], "theta_sy": theta[2],
        "objective": l_best,
        "raw_wrms_total": raw_wrms,
        "flow_total": 0.0,
        "sigma_total": float(np.sqrt(np.mean((sigma_hat * FRAME_W_NP) ** 2))),
        "support_total": _support_penalty(z_best, sel),
        "trust_total": float(sel.trust),
        "prior_total": 0.0,
        "posterior_score": l_best,
        "sub_ids": [sel.sub_id], "basins": [sel.bin_id],
        "start_kind": "shape_cma_topk" if len(members) > 1 else "shape_cma",
        "opt_s": time.time() - t0, "n_rows": n_ev,
        "val_rel_wrms_median": sel.val_rel_wrms_median,
    }
    if z_disp is not None:
        out["z_dispersion"] = z_disp.tolist()       # truth-blind ridge-degeneracy diagnostic
    out.update(ci_block)
    return out


def inverse_double_shape(item_a, item_b, args, device, dtype) -> dict | None:
    if not item_a["members"] or not item_b["members"]:
        return None
    members_a = item_a["members"]
    members_b = item_b["members"]
    ws_a = np.asarray(item_a.get("member_weights", [1.0]*len(members_a)), dtype=np.float64)
    ws_b = np.asarray(item_b.get("member_weights", [1.0]*len(members_b)), dtype=np.float64)
    ws_a = ws_a / max(ws_a.sum(), 1e-9)
    ws_b = ws_b / max(ws_b.sum(), 1e-9)
    ma = members_a[0]
    mb = members_b[0]
    ctx_a, ctx_b = item_a["ctx"], item_b["ctx"]
    fb_lo_orig = ctx_a[6].detach().cpu().numpy().astype(np.float64)
    fb_hi = ctx_a[7].detach().cpu().numpy().astype(np.float64)
    fb_lo = fb_lo_orig.copy()
    if args.sigma_y_min > 0:
        fb_lo[2] = max(fb_lo[2], float(np.log(args.sigma_y_min)))
    pad = float(getattr(args, "shape_box_pad", 0.10))
    # Plan E: union bbox over all top-K members (both setups)
    if len(members_a) > 1 or len(members_b) > 1:
        all_lo = np.stack([m.z_lo for m in (*members_a, *members_b)])
        all_hi = np.stack([m.z_hi for m in (*members_a, *members_b)])
        lo = np.maximum(all_lo.min(axis=0) - pad * Z_SCALE_NP, fb_lo)
        hi = np.minimum(all_hi.max(axis=0) + pad * Z_SCALE_NP, fb_hi)
    else:
        lo, hi = _make_box_for_pair(ma, mb, fb_lo, fb_hi, pad=pad)
    # Guard: sigma_y_min floor can push lo[2] above hi[2] for lo-half subs
    # (σy-split). Fall back to global lower bound (no floor) on dim 2 only.
    if lo[2] >= hi[2] - 1e-3:
        all_lo_union = (all_lo if len(members_a) > 1 or len(members_b) > 1
                        else np.stack([ma.z_lo, mb.z_lo]))
        lo[2] = max(all_lo_union.min(axis=0)[2] - pad * Z_SCALE_NP[2], fb_lo_orig[2])
    # Warm-start: when setup2 is invoked with --shape-warm-start-from-first,
    # setup2.py reads <first_dir>/theta_hat.json and writes the unpacked z
    # into args.shape_warm_start_z.  Use it as CMA-ES x0 instead of the
    # midpoint of the two members' z_nn.  Lets a strong setup1 K=3 result
    # warm a fast joint K=1 / K=2 inverse.
    ws_z = getattr(args, "shape_warm_start_z", None)
    if ws_z is not None:
        x0 = clamp_z(np.asarray(ws_z, dtype=np.float64))
    else:
        x0 = clamp_z(0.5 * (ma.z_nn + mb.z_nn))
    x0 = np.maximum(np.minimum(x0, hi - 1e-3), lo + 1e-3)
    sup_w = float(getattr(args, "shape_support_weight", 0.25))

    fw_a = item_a.get("frame_w", FRAME_W_NP)
    fw_b = item_b.get("frame_w", FRAME_W_NP)
    sigma_obs_log = float(getattr(args, "shape_gp_noise_log_floor", 0.05))

    n_frames_a = float(len(item_a["y_obs"]))
    n_frames_b = float(len(item_b["y_obs"]))
    log_ws_a = np.log(np.maximum(ws_a, 1e-12))
    log_ws_b = np.log(np.maximum(ws_b, 1e-12))

    def _setup_nll(z, members_list, ctx_, y_obs_, fw_, n_frames_, log_ws_, ws_):
        """Per-setup mixture marginal NLL (Bayes-correct logsumexp over K members)."""
        losses = np.empty(len(members_list))
        sup_arr = np.empty(len(members_list))
        for i, m in enumerate(members_list):
            y_hat, sig = _predict_one(z, m, ctx_)
            sig_calib = sig * m.calib_scale
            losses[i] = _setup_loss_log_nuisance_gp_np(y_hat, sig_calib, y_obs_, fw_,
                                                       args.sigma_bias, args.sigma_trend,
                                                       sigma_obs_log)
            sup_arr[i] = _support_penalty(z, m)
        if len(members_list) == 1:
            nll = float(losses[0])
        else:
            log_terms = log_ws_ - losses * n_frames_
            mx = float(log_terms.max())
            nll = -(mx + float(np.log(np.sum(np.exp(log_terms - mx))))) / n_frames_
        sup_total = float(np.dot(ws_, sup_arr))
        return nll, sup_total

    def loss(z):
        # Plan E: per-setup mixture marginal likelihood (Bayes-correct over its top-K).
        l_a, sup_a = _setup_nll(z, members_a, ctx_a, item_a["y_obs"], fw_a,
                                 n_frames_a, log_ws_a, ws_a)
        l_b, sup_b = _setup_nll(z, members_b, ctx_b, item_b["y_obs"], fw_b,
                                 n_frames_b, log_ws_b, ws_b)
        sup = 0.5 * (sup_a + sup_b)
        # Setups conditionally independent given θ: −log p(y_a,y_b|θ) = l_a + l_b.
        return l_a + l_b + sup_w * sup

    t0 = time.time()
    n_restarts = max(1, int(getattr(args, "shape_cma_restarts", 1)))
    z_best, l_best, n_ev, z_disp = _run_cma_multistart(
        loss, x0, lo, hi,
        sigma0=getattr(args, "shape_sigma0", 0.25),
        max_iter=getattr(args, "shape_max_iter", 30),
        popsize=getattr(args, "shape_popsize", 12),
        base_seed=7 + ma.sub_id * 100 + mb.sub_id,
        n_restarts=n_restarts,
    )
    theta = z_to_theta(z_best)
    # Plan E: pick best-fitting member per setup at z_best
    def _pick_best(members_list, ctx, y_obs):
        best_i = 0; best_rw = float("inf")
        for i, m in enumerate(members_list):
            yh, _ = _predict_one(z_best, m, ctx)
            rw = float(relative_wrms(yh, y_obs))
            if rw < best_rw:
                best_rw = rw; best_i = i
        return best_i, best_rw
    sel_a_i, _ = _pick_best(members_a, ctx_a, item_a["y_obs"])
    sel_b_i, _ = _pick_best(members_b, ctx_b, item_b["y_obs"])
    sel_a = members_a[sel_a_i]
    sel_b = members_b[sel_b_i]
    y_a, sa = _predict_one(z_best, sel_a, ctx_a)
    y_b, sb = _predict_one(z_best, sel_b, ctx_b)
    raw_wrms = (relative_wrms(y_a, item_a["y_obs"])
                + relative_wrms(y_b, item_b["y_obs"]))
    sigma_total = (float(np.sqrt(np.mean((sa * FRAME_W_NP) ** 2)))
                   + float(np.sqrt(np.mean((sb * FRAME_W_NP) ** 2))))
    sup = _support_penalty(z_best, sel_a) + _support_penalty(z_best, sel_b)
    topk_tag = "_topk" if (len(members_a)>1 or len(members_b)>1) else ""

    # Hessian Laplace CI (joint posterior).  Step-size uses the smaller
    # per-dim z_scale across the two routed primaries so finite differences
    # stay inside both subs' trust regions.
    ci_block = {}
    if bool(getattr(args, "shape_report_ci", False)):
        try:
            zs_for_h = np.minimum(ma.z_scale, mb.z_scale)
            z_std, theta_ci, ci_flags = _hessian_ci(loss, z_best, zs_for_h)
            ci_block = {
                "z_std": z_std.tolist(),
                "theta_ci_95": {
                    "n":       list(theta_ci[0]),
                    "eta":     list(theta_ci[1]),
                    "sigma_y": list(theta_ci[2]),
                },
                "identifiable": {
                    "n": ci_flags[0], "eta": ci_flags[1], "sigma_y": ci_flags[2]
                },
            }
        except Exception as e:
            print(f"[hessian_ci] failed: {e}")

    out = {
        "theta_n": theta[0], "theta_eta": theta[1], "theta_sy": theta[2],
        "objective": l_best,
        "raw_wrms_total": raw_wrms,
        "flow_total": 0.0,
        "sigma_total": sigma_total,
        "support_total": sup,
        "trust_total": float(0.5 * (sel_a.trust + sel_b.trust)),
        "prior_total": 0.0,
        "posterior_score": l_best,
        "sub_ids": [sel_a.sub_id, sel_b.sub_id],
        "basins": [sel_a.bin_id, sel_b.bin_id],
        "start_kind": "shape_cma_double" + topk_tag,
        "opt_s": time.time() - t0,
        "n_rows": n_ev,
        "pair_spread": float(np.sqrt(np.mean(((sel_a.z_mu - sel_b.z_mu) / Z_SCALE_NP) ** 2))),
        "n_pairs": 1,
        "val_rel_wrms_a": sel_a.val_rel_wrms_median,
        "val_rel_wrms_b": sel_b.val_rel_wrms_median,
    }
    if z_disp is not None:
        out["z_dispersion"] = z_disp.tolist()
    out.update(ci_block)
    return out


def add_argparse_args(ap) -> None:
    """Production y8q + GP-aware-likelihood inverse args.

    Dead paths (Method A/B priors, σ_y two-stage prior, Hessian CI,
    info_frame_w, long_y8_spillover, sy_split_equal_weight, etc.) were
    archived to docs/archive_2026-05-07/selector_full_2026-05-07.py on
    2026-05-07. Use that snapshot if any of those code paths are needed
    for retrospective comparison.
    """
    # ---- CMA-ES ----
    ap.add_argument("--shape-sigma0", type=float, default=0.25,
                    help="CMA-ES initial step in z-space")
    ap.add_argument("--shape-max-iter", type=int, default=30,
                    help="CMA-ES max generations")
    ap.add_argument("--shape-popsize", type=int, default=12,
                    help="CMA-ES population size")
    ap.add_argument("--shape-cma-seed-offset", type=int, default=0,
                    help="Offset added to CMA-ES seed; vary across runs to "
                         "probe identifiability-degenerate θ̂ clusters.")
    ap.add_argument("--shape-cma-restarts", type=int, default=1,
                    help="Number of CMA-ES multi-starts (different seeds, take best). "
                         "n>1 also reports z_dispersion = std(θ_k) across restarts — "
                         "large dispersion = loss landscape has multiple basins → "
                         "ridge-degeneracy diagnostic.  Default 1 = legacy single-start. "
                         "5 is a reasonable paper-grade choice (5× CMA cost).")
    # ---- inverse bbox ----
    ap.add_argument("--shape-box-pad", type=float, default=0.10,
                    help="z-space pad applied to sub training bbox before CMA-ES")
    ap.add_argument("--shape-support-weight", type=float, default=0.25,
                    help="Penalty on z outside sub training bbox")
    # ---- routing ----
    ap.add_argument("--shape-top-k", type=int, default=1,
                    help="Top-K candidate subs per setup. Loss = softmax-weighted "
                         "consensus across the K GPs. 1 = top-1, 2-3 = ensemble.")
    ap.add_argument("--shape-topk-softmax-tau", type=float, default=1.5,
                    help="Softmax temperature on routing scores. Smaller = "
                         "sharper top-1 weight; larger = more even consensus.")
    ap.add_argument("--shape-l2-mode", type=str, default="hard",
                    choices=["hard", "eps"],
                    help="L2 (length-bin) gate behaviour for legacy two-layer "
                         "partitions: 'hard' = strict bin lookup; "
                         "'eps' = pull neighbour bin's subs when y_obs[7] is "
                         "within --shape-l2-eps-cm of a bin edge.")
    ap.add_argument("--shape-l2-eps-cm", type=float, default=0.5,
                    help="Distance (cm) from y_obs[7] to a bin edge below "
                         "which the neighbour bin is pulled in (--shape-l2-mode=eps).")
    ap.add_argument("--shape-eps-cap-by-primary", action="store_true",
                    help="Cap eps-spillover contribution to ≤ primary bin "
                         "sub count to avoid the denser neighbour bin "
                         "dominating the ensemble.")
    ap.add_argument("--shape-weight-mode", type=str, default="gp_fit_y8_quantile",
                    choices=["pca", "gp_fit", "gp_fit_y8_quantile"],
                    help="Sub-level routing re-rank: "
                         "'gp_fit_y8_quantile' (production) = pick sub for which "
                         "y_obs[7] is most central in training Y_8 distribution "
                         "(Bayes-evidence-like); "
                         "'gp_fit' (legacy) = ||GP(z_nn) - y_obs|| in raw y-space; "
                         "'pca' = top-K by PCA centroid distance only.")
    ap.add_argument("--shape-gpfit-oversample", type=int, default=3,
                    help="When --shape-weight-mode in {gp_fit, gp_fit_y8_quantile}, "
                         "oversample factor for the PCA pre-filter: routing "
                         "fetches K*factor candidates, then re-ranks.")
    # ---- loss ----
    ap.add_argument("--shape-gp-noise-log-floor", type=float, default=0.05,
                    help="GP-aware likelihood: log-space measurement noise "
                         "floor σ_obs_log added in quadrature to delta-method "
                         "log-y std (σ_pred / y_hat). Used by "
                         "--loss-mode=log_nuisance_gp. 0.05 ≈ 5%% relative "
                         "noise floor; raise (0.10–0.20) if the GP is over-"
                         "confident on real data and CMA-ES collapses into "
                         "low-σ_pred regions.")
    ap.add_argument("--shape-frame-w", type=float, nargs=8, default=None,
                    metavar=("F1","F2","F3","F4","F5","F6","F7","F8"),
                    help="Override per-frame loss weight (default "
                         "[0.2,0.2,0.35,0.5,0.8,1.0,1.4,2.0]). Use to "
                         "down-weight frames dominated by un-modeled "
                         "physics (e.g. f1-f2 surface-tension initial "
                         "yield, drop_f1: 0.0 0.2 0.35 ...). Pass 8 floats.")
    # ---- UQ output ----
    ap.add_argument("--shape-report-ci", action="store_true",
                    help="Type-II ML Laplace 95%% CI on (n, η, σ_y) from numerical "
                         "Hessian at MAP (9 extra GP evals).  Output dict gains "
                         "'theta_ci_95' + 'identifiable' (per-dim flag).  An "
                         "under-identified direction (eigenvalue at floor) means "
                         "the data alone cannot pin θ in that direction — joint "
                         "two-setup typically narrows σ_y CI dramatically.")
