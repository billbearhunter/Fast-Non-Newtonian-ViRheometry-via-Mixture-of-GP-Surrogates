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


def _info_frame_w(member, z_ref: np.ndarray, ctx, h: float = 0.01) -> np.ndarray:
    """Improvement (5): info-weighted frame loss.

    Per-frame loss weight ∝ ‖∂y_i/∂z‖ at z_ref.  Frames where the surrogate
    is sensitive to θ get more weight; frames that are flat (uninformative)
    get less.  Computed via finite difference (3 extra GP forward calls,
    sub-millisecond cost).  Returns 8-vector normalised so that sum=8 to
    keep the loss scale comparable to the legacy fixed weights.
    """
    y0, _ = _predict_one(z_ref, member, ctx)
    J = np.zeros((8, 3))
    for d in range(3):
        zp = z_ref.copy()
        zp[d] += h
        yp, _ = _predict_one(zp, member, ctx)
        J[:, d] = (yp - y0) / h
    sens = np.linalg.norm(J, axis=1)
    # Avoid zero weights — floor at 5% of max sensitivity
    sens = np.maximum(sens, 0.05 * sens.max())
    return sens / sens.sum() * 8.0  # normalise so sum=8 (same as legacy)


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

def _setup_loss_log_nuisance_np(y_hat, y_obs, frame_w, sigma_bias, sigma_trend) -> float:
    eps = 1e-9
    r = np.log(np.clip(y_hat, eps, None)) - np.log(np.clip(y_obs, eps, None))
    n = y_hat.size
    t = np.linspace(-1.0, 1.0, n)
    X = np.stack([np.ones_like(t), t], axis=1)
    w = frame_w * frame_w
    WX = X * w[:, None]
    A = X.T @ WX + np.diag([1.0 / sigma_bias ** 2, 1.0 / sigma_trend ** 2])
    b = X.T @ (w * r)
    gamma = np.linalg.solve(A, b)
    resid = r - X @ gamma
    return float(np.mean(w * resid * resid))


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


def _setup_loss_linear_np(y_hat, y_obs, frame_w) -> float:
    scale = max(float(np.mean((y_obs * frame_w) ** 2)), 1e-9)
    err = (y_hat - y_obs) * frame_w
    return float(np.mean(err * err) / scale)


def _predict_one(z, member, ctx):
    y, s = predict_full_batched(member, [z], ctx)
    return y[0], s[0]


def _gaussian_prior(z, member: SubMember) -> float:
    """Method A: per-sub Gaussian prior centered at sub's training mean.

    Returns ½ ‖(z − z_mu) / z_scale‖² (negative log-density of N(z_mu,
    diag(z_scale²)) up to constant).  Used as a soft anchor pulling z
    toward the sub's training centroid.  When summed over ensemble
    members with their weights wᵢ, the sum is equivalent to a single
    Gaussian prior centered at the precision-weighted mean of all
    z_mu_i, with covariance (Σᵢ wᵢ Λᵢ⁻¹)⁻¹.
    """
    return float(0.5 * np.sum(((z - member.z_mu) / member.z_scale) ** 2))


def _mog_prior_neg_log(z, members, weights) -> float:
    """Method B: Mixture-of-Gaussians prior — Bayesian marginalisation
    over the routed cluster ensemble.

        −log p(z) = − log Σ_k π_k · 𝒩(z; μ_k, diag(σ_k²))
                  = − logsumexp_k [log π_k − ½‖(z−μ_k)/σ_k‖² − Σ_j log σ_kj]
                  (up to z-independent constant ½ d log 2π)

    π_k are taken to be the routing ensemble weights (data-driven
    cluster posterior).  Compared to Method A's weighted-average
    quadratic prior, the logsumexp lets one cluster *dominate softly*
    when z is near its mode while never zeroing out the others.

    Why MoG is more honest than Method A's mean-anchor:
      * Top-K routing may include wrong clusters; under Method A every
        member contributes a quadratic pull (weighted average).  Under
        MoG, far-away wrong clusters receive vanishing posterior
        responsibility automatically (their term drops out of the
        logsumexp), so the prior is robust to routing errors.
      * Eliminates the two-stage "pick K then prior" — there is just
        one Bayes prior with cluster-membership marginalised out.
      * For y_obs near a cluster boundary (between two centroids), the
        MoG smoothly interpolates between modes; Method A's
        weighted-average centroid would land in a low-density valley.
    """
    log_pi = np.log(np.maximum(np.asarray(weights, dtype=np.float64), 1e-12))
    z_arr = np.asarray(z, dtype=np.float64)
    log_terms = np.empty(len(members), dtype=np.float64)
    for i, (m, lp) in enumerate(zip(members, log_pi)):
        d = (z_arr - m.z_mu) / m.z_scale
        mahala = 0.5 * float(np.dot(d, d))
        log_norm = float(np.sum(np.log(m.z_scale)))   # ½ log det Σ_k
        log_terms[i] = lp - mahala - log_norm
    mx = float(log_terms.max())
    log_p = mx + float(np.log(np.sum(np.exp(log_terms - mx))))
    return -log_p


def _support_penalty(z, member: SubMember) -> float:
    over = np.maximum(z - member.z_hi, 0.0)
    under = np.maximum(member.z_lo - z, 0.0)
    return float(np.sum(((over + under) / member.z_scale) ** 2))


def _hessian_ci(loss_fn, z_best, h_scale: np.ndarray, alpha: float = 0.05,
                  h_factor: float = 0.10, min_eig_frac: float = 1e-3):
    """Method C: Laplace-approx 95% CI from numerical Hessian at MAP.

    Approximates the posterior near z_best as N(z_best, H⁻¹) where
    H = ∇²(−log p(z|y)).  CI is computed by central differences with
    step h_i = h_factor · z_scale_i.  The exp-link transform z → θ for
    (η, σ_y) maps Gaussian-on-z to log-normal-on-θ; CI bounds are
    obtained by transforming the z-bounds.

    Numerical safeguards:
      * Symmetrise H = ½ (H + Hᵀ).
      * Eigenvalues below `min_eig_frac · max|eval|` (or below 0) are
        replaced by that floor — represents an under-identified
        direction with a wide-but-finite CI.  Without this floor a
        single non-positive eigenvalue (from CMA-ES not converging
        exactly to the local minimum, or from the support-penalty kink
        if z_best is near the bbox) would explode the covariance to
        machine-infinity.
      * CI bounds are intersected with a sane physical range
        (n ∈ [0, 1], log η ∈ [-3, 6], log σ_y ∈ [0, 7]) to avoid
        nonsense outputs when the data alone cannot identify a
        direction.

    Returns
    -------
    z_std : np.ndarray (3,)
        Posterior std on z (sqrt of H⁻¹ diag).
    theta_ci : list[(lo, hi)] length 3 — 95 % CI on θ.
    H : np.ndarray (3, 3) — the symmetrised Hessian.
    """
    z_best = np.asarray(z_best, dtype=np.float64)
    n = z_best.size
    f0 = loss_fn(z_best)
    h = np.maximum(np.minimum(np.asarray(h_scale) * h_factor, 0.30), 5e-3)
    H = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        f_pi = loss_fn(z_best + ei)
        f_mi = loss_fn(z_best - ei)
        H[i, i] = (f_pi + f_mi - 2.0 * f0) / (h[i] * h[i])
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
    evals_out = np.zeros(n)
    try:
        evals, evecs = np.linalg.eigh(H)
        evals_out = evals.copy()
        # robust positive-definite floor: keep at least min_eig_frac × max
        max_eig = float(np.abs(evals).max())
        floor_eig = max(min_eig_frac * max(max_eig, 1.0), 1e-6)
        evals_pd = np.where(evals < floor_eig, floor_eig, evals)
        cov = (evecs / evals_pd) @ evecs.T
        diag = np.clip(np.diag(cov), 0.0, None)
        z_std = np.sqrt(diag)
    except np.linalg.LinAlgError:
        pass
    from scipy.stats import norm
    z_q = float(norm.ppf(1.0 - alpha / 2.0))
    z_lo = z_best - z_q * z_std
    z_hi = z_best + z_q * z_std
    # Clamp to physical box so the report is interpretable when a direction
    # is genuinely under-identified by the data alone.
    z_box_lo = np.array([0.0, -3.0, 0.0])
    z_box_hi = np.array([1.0,  6.0, 7.0])
    z_lo_c = np.maximum(z_lo, z_box_lo)
    z_hi_c = np.minimum(z_hi, z_box_hi)
    theta_ci = [
        (float(z_lo_c[0]), float(z_hi_c[0])),                    # n
        (float(np.exp(z_lo_c[1])), float(np.exp(z_hi_c[1]))),    # η
        (float(np.exp(z_lo_c[2])), float(np.exp(z_hi_c[2]))),    # σ_y
    ]
    return z_std, theta_ci, H


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
    fetch_k = top_k * over_factor if weight_mode in ("gp_fit", "gp_fit_shape", "gp_fit_combined", "gp_fit_y8_quantile", "inv_var") else top_k
    eps_cap = bool(getattr(args, "shape_eps_cap_by_primary", False))
    cands = _route_topk_subs(y_obs, state, fetch_k, mode=l2_mode, eps_cm=eps_cm,
                              eps_cap_by_primary=eps_cap)
    if not cands:
        if state.get("partition_type") == "flat_kmeans_separate":
            # Flat fallback: just use the single nearest cluster's first sub
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
            continue  # skip missing experts

    if weight_mode == "gp_fit_y8_quantile":
        # Y_8 quantile criterion (Bayes-evidence-like sub selection).
        # Score = |y_obs[7] - y8_med_k| / y8_std_k  where (y8_med_k, y8_std_k)
        # are the median and std of sub_k's training Y_8 distribution.
        # Picks the sub for which y_obs[7] is most "typical" (central in training
        # Y_8 distribution).  Ignores absolute shape — relies on cluster-level
        # routing having already filtered shape match.
        y_obs_y8 = float(y_obs[7])
        fit_dists = []
        for m in over_members:
            score = abs(y_obs_y8 - m.y8_med) / max(m.y8_std, 1e-6)
            fit_dists.append(score)
        order = np.argsort(np.asarray(fit_dists))
        keep = order[:max(1, top_k)]
        members = [over_members[i] for i in keep]
        d_kept = np.asarray([fit_dists[i] for i in keep], dtype=np.float64)
    elif weight_mode in ("gp_fit", "gp_fit_shape", "gp_fit_combined"):
        # Score every oversampled candidate by GP-fit at its own z_nn and
        # keep the K best.  Then derive softmax weights from the same scores.
        #
        # gp_fit          : Euclidean ||y_pred - y_obs|| (legacy). Sim-real Y_8 bias →
        #                    favours high σ_y subs.
        # gp_fit_shape    : range-norm shape only. Affine-invariant; ignores absolute
        #                    Y_8 magnitude completely → can flip too far to low σ_y.
        # gp_fit_combined : shape (7D) + λ·log(Y_8) (1D) in joint metric. The λ
        #                    (`shape_gp_fit_y8_factor`) controls how much absolute
        #                    Y_8 information enters. λ=0 = pure shape, λ=∞ ≈ pure Y_8.
        #                    Default λ=1.0 ≈ shape & log_y8 contribute equally.
        mode = weight_mode
        y8_factor = float(getattr(args, "shape_gp_fit_y8_factor", 1.0))
        eps = 1e-6

        def _shape(y):
            y = np.asarray(y, dtype=np.float64)
            span = max(float(y[7] - y[0]), eps)
            return (y[1:8] - y[0]) / span

        def _logy8(y):
            return float(np.log(max(float(y[7]), eps)))

        if mode == "gp_fit":
            y_obs_ref = np.asarray(y_obs, dtype=np.float64)
        elif mode == "gp_fit_shape":
            y_obs_ref = _shape(y_obs)
        else:  # gp_fit_combined
            y_obs_shape = _shape(y_obs)
            y_obs_logy8 = _logy8(y_obs)

        fit_dists = []
        for m in over_members:
            y_pred, _ = _predict_one(m.z_nn, m, ctx)
            if mode == "gp_fit":
                d = float(np.linalg.norm(y_pred - y_obs_ref))
            elif mode == "gp_fit_shape":
                d = float(np.linalg.norm(_shape(y_pred) - y_obs_ref))
            else:  # gp_fit_combined
                ds = _shape(y_pred) - y_obs_shape  # 7D shape diff
                dy = _logy8(y_pred) - y_obs_logy8  # 1D log-y8 diff
                d = float(np.sqrt(np.sum(ds * ds) + (y8_factor * dy) ** 2))
            fit_dists.append(d)
        order = np.argsort(np.asarray(fit_dists))
        keep = order[:max(1, top_k)]
        members = [over_members[i] for i in keep]
        d_kept = np.asarray([fit_dists[i] for i in keep], dtype=np.float64)
    elif weight_mode == "inv_var":
        # Improvement (1) — BLUE inverse-variance weighting.
        # Each sub's GP returns a frame-wise predictive σ at z_nn; aggregate
        # to a scalar σ_total² and weight ∝ 1/σ_total².  Selects top-K by
        # smallest σ_total² (most-confident GPs at z_nn) and uses the same
        # scores as multiplicative weights (NO softmax — true BLUE).
        # Theory: weighted least squares with inverse-variance weights is
        # the minimum-variance unbiased linear estimator (Gauss-Markov).
        sigma_totals = []
        for m in over_members:
            _, sigma = _predict_one(m.z_nn, m, ctx)
            # Apply per-sub calibration scalar so σ_eff ≈ actual RMSE.
            # calib_scale=1.0 when checkpoint pre-dates calibrate_subs.py.
            sigma_eff = np.asarray(sigma) * m.calib_scale
            sigma_totals.append(float(np.sqrt(np.mean(sigma_eff ** 2))))
        sig_arr = np.asarray(sigma_totals, dtype=np.float64)
        order = np.argsort(sig_arr)  # ascending: smallest sigma first
        keep = order[:max(1, top_k)]
        members = [over_members[i] for i in keep]
        sig_kept = sig_arr[keep]
        # BLUE weights = (1/σ²) / Σ(1/σ²)
        inv_var = 1.0 / np.maximum(sig_kept ** 2, 1e-12)
        d_kept = sig_kept  # informational only (for downstream callers)
    elif weight_mode == "pca":
        members = over_members[:max(1, top_k)]
        d_kept = np.asarray(over_pca_dists[:max(1, top_k)], dtype=np.float64)
    else:
        raise ValueError(f"unknown --shape-weight-mode={weight_mode}")

    if weight_mode == "inv_var" and len(members) > 1:
        ws = inv_var / inv_var.sum()
    elif len(members) > 1:
        tau = float(getattr(args, "shape_topk_softmax_tau", 1.5))
        ws = np.exp(-(d_kept - d_kept.min()) / max(tau, 1e-3))
        ws = ws / ws.sum()
    else:
        ws = np.array([1.0])

    # σ_y split equal-weight (flat partition only).
    # Each cluster's σ_y range is quantile-split into 1-4 sub-experts
    # (low/mid/high σ_y bins).  When multiple subs from the same cluster
    # appear in the ensemble, they are different σ_y bins of the same
    # (length, shape) regime → force them equal-weighted to remove the
    # bias from gp_fit / pca / inv_var weighting that would otherwise
    # favour whichever σ_y bin happens to fit y_obs at z_nn best.  The
    # cluster's overall ensemble weight (sum across its sub-bins) is
    # preserved; only the internal split is equalised.
    if (bool(getattr(args, "shape_sy_split_equal_weight", False))
            and state.get("partition_type") == "flat_kmeans_separate"
            and len(members) > 1):
        from collections import defaultdict
        by_cluster = defaultdict(list)
        for i, m in enumerate(members):
            by_cluster[int(m.bin_id)].append(i)
        for ci, idxs in by_cluster.items():
            if len(idxs) >= 2:
                cluster_total = float(sum(ws[i] for i in idxs))
                share = cluster_total / len(idxs)
                for i in idxs:
                    ws[i] = share
        s = ws.sum()
        if s > 0:
            ws = ws / s
    # Long-y8 spillover (flat partition only). Force-add a "long-flow"
    # cluster's σ_y-low sub to the ensemble to recover OLD length-bin's
    # bin_4 → bin_3 spillover effect: low-σ_y physical constraint pulled
    # in for boundary-y8 queries (e.g. Chuno y8=5.32 → bring in cluster 5
    # y8≈8 σ_y_low sub to anchor σ_y down).
    long_y8_factor = float(getattr(args, "shape_long_y8_spillover_factor", 0.0))
    if (long_y8_factor > 0
            and state.get("partition_type") == "flat_kmeans_separate"
            and y_obs is not None):
        y8_query = float(y_obs[7])
        threshold = float(getattr(args, "shape_long_y8_spillover_ratio", 1.5)) * y8_query
        # Cache cluster y8 medians (load each cluster's first sub's training y8)
        if "_cluster_y8_med_cache" not in state:
            cache = []
            for cluster in state["flat_clusters"]:
                first_sid = int(cluster["subs"][0]["sub_id"])
                p = sd / "experts" / f"sub_{first_sid:04d}.pt"
                try:
                    ck = torch.load(p, weights_only=False, map_location="cpu")
                    cache.append(float(np.median(np.asarray(ck["Y_phys"])[:, 7])))
                except FileNotFoundError:
                    cache.append(0.0)
            state["_cluster_y8_med_cache"] = cache
        cluster_y8_meds = state["_cluster_y8_med_cache"]
        used = {int(m.bin_id) for m in members}
        candidates = [ci for ci, y8m in enumerate(cluster_y8_meds)
                       if y8m > threshold and ci not in used]
        if candidates:
            # Pick the nearest long-y8 cluster (smallest centroid distance)
            from Optimization.libs.selector import _project_flat as _proj
            feat = _proj(y_obs, state)
            centroids = np.asarray(state["flat_centroids"])
            cdists = [float(np.linalg.norm(centroids[ci] - feat)) for ci in candidates]
            ci_pick = candidates[int(np.argmin(cdists))]
            # Pick its σ_y-low sub (subs are stored in σ_y bin order, [0]=lowest)
            cluster = state["flat_clusters"][ci_pick]
            sid_pick = int(cluster["subs"][0]["sub_id"])
            try:
                m_extra = _build_sub_member(sd, sid_pick, ci_pick, y_obs, xs, ys, device, dtype)
                members.append(m_extra)
                ws_new = np.append(ws, long_y8_factor)
                ws_new = ws_new / ws_new.sum()
                ws = ws_new
            except FileNotFoundError:
                pass
    # Resolve frame weights: legacy override or info-weighted (improvement 5)
    if bool(getattr(args, "shape_info_frame_w", False)) and len(members) >= 1:
        frame_w_used = _info_frame_w(members[0], members[0].z_nn, ctx)
    else:
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
    # Plan E: when top-K > 1, expand bbox to UNION of all members (so CMA
    # can search across all candidate subs)
    z_lo_union = np.min(np.stack([m.z_lo for m in members]), axis=0)
    z_hi_union = np.max(np.stack([m.z_hi for m in members]), axis=0)
    # Improvement (6): use per-sub z_scale (training-data std per dim) instead
    # of fixed Z_SCALE_NP for bbox padding.  union over members' z_scale gives
    # a width adaptive to the routed subs (wider when subs span more variance).
    if bool(getattr(args, "shape_per_sub_z_scale", False)):
        zs_pad = np.max(np.stack([m.z_scale for m in members]), axis=0)
    else:
        zs_pad = Z_SCALE_NP
    lo = np.maximum(z_lo_union - pad * zs_pad, fb_lo)
    hi = np.minimum(z_hi_union + pad * zs_pad, fb_hi)
    # Guard: sigma_y_min floor can push lo[2] above hi[2] for lo-half subs
    # (σy-split). Fall back to global lower bound (no floor) on dim 2 only.
    if lo[2] >= hi[2] - 1e-3:
        lo[2] = max(z_lo_union[2] - pad * zs_pad[2], fb_lo_orig[2])
    x0 = np.maximum(np.minimum(primary.z_nn, hi - 1e-3), lo + 1e-3)

    sup_w = float(getattr(args, "shape_support_weight", 0.25))
    loss_mode = args.loss_mode

    # Plan C: σY-prior for single mode, anchored at the sub-training nearest
    # neighbour to y_obs (member.z_nn) rather than the sub bbox mean
    # (member.z_mu).  z_nn answers "what σY produced data most similar to
    # y_obs?" — much more informative than the bbox centroid.
    sy_prior_w_single = float(getattr(args, "shape_sy_prior_single_weight", 0.0))
    sy_anchor_mode = str(getattr(args, "shape_sy_prior_single_anchor", "z_nn"))
    log_sy_anchor_single = None
    log_sy_prior_std_single = 0.5
    if sy_prior_w_single > 0:
        if sy_anchor_mode == "z_mu":
            log_sy_anchor_single = float(primary.z_mu[2])
        else:
            log_sy_anchor_single = float(primary.z_nn[2])
        log_sy_prior_std_single = max(
            float((primary.z_hi[2] - primary.z_lo[2]) / 4.0),
            float(getattr(args, "shape_sy_prior_min_std", 0.4)),
        )

    fw = item.get("frame_w", FRAME_W_NP)
    use_mog = (loss_mode == "mog_likelihood")
    sigma_obs = float(getattr(args, "shape_mog_sigma_obs", 0.05))
    sigma_obs_log = float(getattr(args, "shape_gp_noise_log_floor", 0.05))
    log_priors = np.log(np.maximum(weights, 1e-12))
    # Method A: per-sub Gaussian prior weight (default 0 = off; 1.0 = MAP)
    gauss_w = float(getattr(args, "shape_gaussian_prior_weight", 0.0))
    # Method B: Mixture-of-Gaussians prior weight (default 0 = off; 1.0 = MAP)
    mog_pri_w = float(getattr(args, "shape_mog_prior_weight", 0.0))
    def loss(z):
        if use_mog:
            # Bayesian Mixture-of-GPs negative log marginal likelihood.
            #   p(y|z) = Σᵢ wᵢ · 𝒩(y; μᵢ(z), σᵢ²(z) + σ_obs²)
            #   loss   = −logsumexp_i [log wᵢ + log 𝒩_i]
            # frame_w used to scale per-frame contributions inside log 𝒩.
            log_lls = []
            sup_total = 0.0
            for m, w_log in zip(members, log_priors):
                y_hat, sigma = _predict_one(z, m, ctx)
                var = np.asarray(sigma) ** 2 + sigma_obs ** 2  # (8,)
                resid2 = (y_hat - item["y_obs"]) ** 2
                # frame_w weights the diagonal Mahalanobis distance (still a
                # valid log-likelihood up to constant since fw acts like an
                # additional precision multiplier per frame)
                ll = -0.5 * np.sum(fw * (resid2 / var)) - 0.5 * np.sum(np.log(var))
                log_lls.append(w_log + ll)
                sup_total += float(np.exp(w_log)) * _support_penalty(z, m)
            log_lls_arr = np.asarray(log_lls)
            mx = float(log_lls_arr.max())
            log_p = mx + float(np.log(np.sum(np.exp(log_lls_arr - mx))))
            prior = 0.0
            if log_sy_anchor_single is not None:
                prior = sy_prior_w_single * ((z[2] - log_sy_anchor_single) / log_sy_prior_std_single) ** 2
            gp_total = sum(float(np.exp(w_log)) * _gaussian_prior(z, m)
                            for m, w_log in zip(members, log_priors))
            mog_pri = _mog_prior_neg_log(z, members, weights) if mog_pri_w > 0 else 0.0
            return -log_p + sup_w * sup_total + gauss_w * gp_total + mog_pri_w * mog_pri + prior
        # Legacy paths (linear / log_nuisance / log_nuisance_gp) below
        l_total = 0.0
        sup_total = 0.0
        gp_total = 0.0  # Method A: weighted Gaussian prior across ensemble
        for m, w in zip(members, weights):
            y_hat, sig = _predict_one(z, m, ctx)
            if loss_mode == "log_nuisance_gp":
                lk = _setup_loss_log_nuisance_gp_np(y_hat, sig, item["y_obs"], fw,
                                                     args.sigma_bias, args.sigma_trend,
                                                     sigma_obs_log)
            elif loss_mode == "log_nuisance":
                lk = _setup_loss_log_nuisance_np(y_hat, item["y_obs"], fw,
                                                  args.sigma_bias, args.sigma_trend)
            else:
                lk = _setup_loss_linear_np(y_hat, item["y_obs"], fw)
            l_total += float(w) * lk
            sup_total += float(w) * _support_penalty(z, m)
            gp_total += float(w) * _gaussian_prior(z, m)
        prior = 0.0
        if log_sy_anchor_single is not None:
            prior = sy_prior_w_single * ((z[2] - log_sy_anchor_single) / log_sy_prior_std_single) ** 2
        mog_pri = _mog_prior_neg_log(z, members, weights) if mog_pri_w > 0 else 0.0
        return l_total + sup_w * sup_total + gauss_w * gp_total + mog_pri_w * mog_pri + prior

    t0 = time.time()
    z_best, l_best, n_ev = _run_cma(
        loss, x0, lo, hi,
        sigma0=getattr(args, "shape_sigma0", 0.25),
        max_iter=getattr(args, "shape_max_iter", 30),
        popsize=getattr(args, "shape_popsize", 12),
        seed=7 + primary.sub_id + int(getattr(args, "shape_cma_seed_offset", 0)),
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
    # Method C: Hessian-based 95 % CI at MAP (Laplace approximation)
    z_std, theta_ci, _Hmat = (np.full(3, float("nan")),
                               [(float("nan"), float("nan"))]*3, None)
    if bool(getattr(args, "shape_report_ci", False)):
        try:
            z_std, theta_ci, _Hmat = _hessian_ci(loss, z_best, primary.z_scale)
        except Exception as _e:
            print(f"[hessian_ci] failed: {_e}")
    return {
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
        "z_std": z_std.tolist(),
        "theta_ci_95": {
            "n": list(theta_ci[0]),
            "eta": list(theta_ci[1]),
            "sigma_y": list(theta_ci[2]),
        },
    }


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
    # into args.shape_warm_start_z.  Use it as CMA-ES x0 (and as the σY
    # anchor below if --shape-sy-prior-weight > 0) instead of the legacy
    # midpoint of the two members' z_nn.  Lets a strong setup1 K=3 result
    # warm a fast joint K=1 / K=2 inverse.
    ws_z = getattr(args, "shape_warm_start_z", None)
    if ws_z is not None:
        x0 = clamp_z(np.asarray(ws_z, dtype=np.float64))
    else:
        x0 = clamp_z(0.5 * (ma.z_nn + mb.z_nn))
    x0 = np.maximum(np.minimum(x0, hi - 1e-3), lo + 1e-3)
    sup_w = float(getattr(args, "shape_support_weight", 0.25))
    loss_mode = args.loss_mode

    # ---- σY two-stage prior (V1-paper style: anchor σY across setups) ----
    sy_prior_w = float(getattr(args, "shape_sy_prior_weight", 0.0))
    log_sy_anchor = None
    log_sy_prior_std = None
    sy_a_ret = sy_b_ret = sy_anchor_ret = float("nan")
    sat_lo = float(getattr(args, "shape_sy_sat_lo", 8.0))
    sat_hi = float(getattr(args, "shape_sy_sat_hi", 380.0))
    sat_factor = float(getattr(args, "shape_sy_prior_sat_factor", 0.0))
    if sy_prior_w > 0 and ws_z is not None:
        # Warm-start path: use the σY of the supplied warm-start θ̂ as the
        # anchor directly. Skips the expensive internal stage-1 single
        # inverses; relies on the caller having run setup1 (e.g. K=3) and
        # producing a trustworthy σY.  Mimics res_a/res_b structure for
        # downstream code by routing both setups to the same supplied σY.
        sy_anchor_ret = float(np.exp(ws_z[2]))
        log_sy_anchor = float(ws_z[2])
        log_sy_prior_std = max(
            float(getattr(args, "shape_sy_prior_min_std", 0.4)),
            0.5,  # widely permissive when no disagreement signal available
        )
        sy_a_ret = sy_anchor_ret
        sy_b_ret = sy_anchor_ret
    elif sy_prior_w > 0:
        # Stage-1: single inverse on each setup
        try:
            res_a = inverse_single_shape(item_a, args, device, dtype)
            res_b = inverse_single_shape(item_b, args, device, dtype)
        except Exception as e:
            res_a = res_b = None
        if res_a and res_b:
            sy_a_ret = res_a["theta_sy"]
            sy_b_ret = res_b["theta_sy"]
            log_sy_a = np.log(max(sy_a_ret, 1e-6))
            log_sy_b = np.log(max(sy_b_ret, 1e-6))
            # Saturation guard
            sat_a = (sy_a_ret < sat_lo) or (sy_a_ret > sat_hi)
            sat_b = (sy_b_ret < sat_lo) or (sy_b_ret > sat_hi)

            # Plan D: confidence-weighted anchor (no truth needed).
            # Each single is weighted by (1/raw_wrms) × penalty(saturated).
            # This downweights unreliable single estimates without disabling
            # them entirely.  Falls back to equal-weighted geometric mean if
            # weights cannot be computed.
            anchor_mode = str(getattr(args, "shape_sy_anchor_mode", "weighted"))
            sat_penalty = float(getattr(args, "shape_sy_anchor_sat_penalty", 0.05))
            if anchor_mode == "weighted":
                # Plan D: 1/raw_wrms × sat_penalty
                rw_a = float(res_a.get("raw_wrms_total", 1.0))
                rw_b = float(res_b.get("raw_wrms_total", 1.0))
                w_a = 1.0 / max(rw_a, 1e-3)
                w_b = 1.0 / max(rw_b, 1e-3)
                if sat_a:
                    w_a *= sat_penalty
                if sat_b:
                    w_b *= sat_penalty
                ws = w_a + w_b
                if ws <= 0:
                    log_sy_anchor = 0.5 * (log_sy_a + log_sy_b)
                else:
                    log_sy_anchor = (w_a * log_sy_a + w_b * log_sy_b) / ws
            elif anchor_mode == "boundary":
                # Plan F: weight = 1.0 if NOT at boundary, sat_penalty if at
                # boundary.  Drops the 1/raw_wrms factor that hurt Plan D —
                # only the boundary signal matters.
                w_a = sat_penalty if sat_a else 1.0
                w_b = sat_penalty if sat_b else 1.0
                ws = w_a + w_b
                log_sy_anchor = (w_a * log_sy_a + w_b * log_sy_b) / ws
            else:
                # Legacy "equal" anchor (Plan B path).
                if sat_a or sat_b:
                    sy_prior_w *= sat_factor
                log_sy_anchor = 0.5 * (log_sy_a + log_sy_b)

            sy_anchor_ret = float(np.exp(log_sy_anchor))
            # data-driven prior std: if singles agree → tight; if disagree → loose
            disagree = abs(log_sy_a - log_sy_b)
            log_sy_prior_std = max(float(getattr(args, "shape_sy_prior_min_std", 0.4)),
                                     0.5 * disagree)

    fw_a = item_a.get("frame_w", FRAME_W_NP)
    fw_b = item_b.get("frame_w", FRAME_W_NP)
    gauss_w = float(getattr(args, "shape_gaussian_prior_weight", 0.0))
    mog_pri_w = float(getattr(args, "shape_mog_prior_weight", 0.0))
    sigma_obs_log = float(getattr(args, "shape_gp_noise_log_floor", 0.05))
    def loss(z):
        # Plan E: weighted consensus across top-K members for each setup
        l_a_total = sup_a_total = gp_a_total = 0.0
        for m, w in zip(members_a, ws_a):
            y_a, sig_a = _predict_one(z, m, ctx_a)
            if loss_mode == "log_nuisance_gp":
                la = _setup_loss_log_nuisance_gp_np(y_a, sig_a, item_a["y_obs"], fw_a,
                                                      args.sigma_bias, args.sigma_trend,
                                                      sigma_obs_log)
            elif loss_mode == "log_nuisance":
                la = _setup_loss_log_nuisance_np(y_a, item_a["y_obs"], fw_a,
                                                   args.sigma_bias, args.sigma_trend)
            else:
                la = _setup_loss_linear_np(y_a, item_a["y_obs"], fw_a)
            l_a_total += float(w) * la
            sup_a_total += float(w) * _support_penalty(z, m)
            gp_a_total += float(w) * _gaussian_prior(z, m)
        l_b_total = sup_b_total = gp_b_total = 0.0
        for m, w in zip(members_b, ws_b):
            y_b, sig_b = _predict_one(z, m, ctx_b)
            if loss_mode == "log_nuisance_gp":
                lb = _setup_loss_log_nuisance_gp_np(y_b, sig_b, item_b["y_obs"], fw_b,
                                                      args.sigma_bias, args.sigma_trend,
                                                      sigma_obs_log)
            elif loss_mode == "log_nuisance":
                lb = _setup_loss_log_nuisance_np(y_b, item_b["y_obs"], fw_b,
                                                   args.sigma_bias, args.sigma_trend)
            else:
                lb = _setup_loss_linear_np(y_b, item_b["y_obs"], fw_b)
            l_b_total += float(w) * lb
            sup_b_total += float(w) * _support_penalty(z, m)
            gp_b_total += float(w) * _gaussian_prior(z, m)
        sup = 0.5 * (sup_a_total + sup_b_total)
        # Bayes-correct prior: a single z-prior, not summed over both
        # ensembles.  members_a and members_b may route to different
        # sub-experts; we average their views of the prior so the prior
        # contribution stays comparable to single-setup magnitude.
        gp = 0.5 * (gp_a_total + gp_b_total)
        # Method B (MoG): logsumexp over the union of routed clusters
        # from both setups.  Sharing one MoG between setups is the
        # Bayes-correct way (single prior on z); we union the routed
        # ensembles so a cluster that is dominant under setup A
        # competes against clusters dominant under setup B in one
        # mixture.  When members_a == members_b the union equals each
        # ensemble alone.
        mog_pri = 0.0
        if mog_pri_w > 0:
            members_union = list(members_a) + list(members_b)
            ws_union = np.concatenate([0.5 * ws_a, 0.5 * ws_b])  # already-normalised
            ws_union = ws_union / max(ws_union.sum(), 1e-9)
            mog_pri = _mog_prior_neg_log(z, members_union, ws_union)
        prior = 0.0
        if log_sy_anchor is not None:
            prior = ((z[2] - log_sy_anchor) / log_sy_prior_std) ** 2
        # Bayes-correct likelihood: −log p(y_a, y_b | z) = −log p(y_a|z)
        # − log p(y_b|z) = l_a_total + l_b_total (assuming setups
        # conditionally independent given θ).  Previously this was
        # 0.5·(l_a+l_b) which halved the data Hessian, leaving joint CI
        # no narrower than single — the 1/2 made the joint behave like a
        # weighted-average of the two single losses rather than a proper
        # multi-data posterior.
        return (l_a_total + l_b_total + sup_w * sup + gauss_w * gp
                + mog_pri_w * mog_pri + sy_prior_w * prior)

    t0 = time.time()
    z_best, l_best, n_ev = _run_cma(
        loss, x0, lo, hi,
        sigma0=getattr(args, "shape_sigma0", 0.25),
        max_iter=getattr(args, "shape_max_iter", 30),
        popsize=getattr(args, "shape_popsize", 12),
        seed=7 + ma.sub_id * 100 + mb.sub_id,
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
    # Method C: Hessian-based 95 % CI at MAP (Laplace approximation).
    # Use the smaller per-coord z_scale across the two routed primaries
    # so the finite-difference step h stays inside both trust regions.
    z_std, theta_ci, _Hmat = (np.full(3, float("nan")),
                               [(float("nan"), float("nan"))]*3, None)
    if bool(getattr(args, "shape_report_ci", False)):
        try:
            zs_for_h = np.minimum(ma.z_scale, mb.z_scale)
            z_std, theta_ci, _Hmat = _hessian_ci(loss, z_best, zs_for_h)
        except Exception as _e:
            print(f"[hessian_ci] failed: {_e}")
    return {
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
        "sy_single_a": sy_a_ret,
        "sy_single_b": sy_b_ret,
        "sy_anchor": sy_anchor_ret,
        "z_std": z_std.tolist(),
        "theta_ci_95": {
            "n": list(theta_ci[0]),
            "eta": list(theta_ci[1]),
            "sigma_y": list(theta_ci[2]),
        },
    }


def add_argparse_args(ap) -> None:
    ap.add_argument("--shape-sigma0", type=float, default=0.25,
                    help="CMA-ES initial step in z-space")
    ap.add_argument("--shape-max-iter", type=int, default=30,
                    help="CMA-ES max generations")
    ap.add_argument("--shape-popsize", type=int, default=12,
                    help="CMA-ES population size")
    ap.add_argument("--shape-box-pad", type=float, default=0.10,
                    help="z-space pad applied to sub training bbox before CMA-ES")
    ap.add_argument("--shape-support-weight", type=float, default=0.25,
                    help="Penalty on z outside sub training bbox")
    ap.add_argument("--shape-sy-prior-weight", type=float, default=0.0,
                    help="V1-paper-style σY two-stage prior: λ in "
                         "λ × ((log σY - log σY_anchor)/σ)². 0 = off.")
    ap.add_argument("--shape-sy-prior-min-std", type=float, default=0.4,
                    help="Floor on prior std (log space). When the two single "
                         "estimates of σY agree closely the prior would be "
                         "infinitely tight; this floor keeps it physically "
                         "useful (e.g. 0.4 ≈ 50%).")
    ap.add_argument("--shape-sy-sat-lo", type=float, default=8.0,
                    help="σY saturation lower bound (default 8 Pa internal). "
                         "Single estimates at/below get treated as unreliable.")
    ap.add_argument("--shape-sy-sat-hi", type=float, default=380.0,
                    help="σY saturation upper bound (default 380).")
    ap.add_argument("--shape-sy-prior-sat-factor", type=float, default=0.0,
                    help="Multiplier on prior weight when either single σY "
                         "estimate is saturated. 0 = skip prior entirely, "
                         "0.2 = weak prior, 1.0 = no guard.")
    ap.add_argument("--shape-sy-prior-single-weight", type=float, default=0.0,
                    help="Plan C: σY prior weight for single mode. "
                         "Helps stabilise single-setup σY estimates "
                         "against ridge degeneracy.")
    ap.add_argument("--shape-sy-prior-single-anchor", type=str,
                    default="z_nn", choices=["z_nn", "z_mu"],
                    help="Plan C anchor: 'z_nn' (default, σY of nearest "
                         "training y to y_obs) or 'z_mu' (sub training "
                         "mean σY). z_nn is more data-informative.")
    ap.add_argument("--shape-sy-anchor-mode", type=str,
                    default="equal",
                    choices=["weighted", "equal", "boundary"],
                    help="σY anchor mode: 'equal' = Plan B geometric mean "
                         "(production default), 'weighted' = Plan D "
                         "(1/raw_wrms × sat_penalty), 'boundary' = Plan F "
                         "(weight 1 if not saturated, sat_penalty if at "
                         "boundary).")
    ap.add_argument("--shape-sy-anchor-sat-penalty", type=float,
                    default=0.05,
                    help="Plan D: weight multiplier when a single σY is "
                         "saturated. 0.05 = heavy downweight, 1.0 = no "
                         "penalty.")
    ap.add_argument("--shape-top-k", type=int, default=1,
                    help="Plan E: top-K candidate subs per setup based on "
                         "PCA distance.  Loss = softmax-weighted consensus "
                         "across all K GPs at the same z. Wrong-routed "
                         "subs auto-downweight (their forward GP fails to "
                         "match y_obs). 1 = legacy top-1, 3 = recommended.")
    ap.add_argument("--shape-topk-softmax-tau", type=float, default=1.5,
                    help="Plan E: softmax temperature on PCA distance. "
                         "Smaller = sharper top-1 weight. Larger = more "
                         "even consensus.")
    # ---- L2 (length-bin) gate softening (Fix A / C1 / C3) ----
    # Background: L2 gate uses real y_obs[7] against bin edges that were
    # learned on SIM data. With known sim>real bias (e.g. surface tension
    # absent in MPM), real y_obs[7] near a bin upper edge gets routed to
    # the lower bin even though the truth-θ would lie in the upper bin.
    ap.add_argument("--shape-l2-mode", type=str, default="hard",
                    choices=["hard", "eps", "cross_bin", "hybrid", "soft_attn"],
                    help="L2 (length-bin) gate behaviour: "
                         "'hard' = legacy strict bin lookup; "
                         "'eps' = Fix A — when y_obs[7] is within "
                         "--shape-l2-eps-cm of a bin edge, also pull all "
                         "subs from the neighbour bin into the candidate "
                         "pool; "
                         "'cross_bin' = C1 — ignore L2, take each bin's "
                         "top-1 sub (per-bin PCA), rank meta by GP "
                         "fidelity at the sub's z_nn; "
                         "'hybrid' = C3 — primary bin's full subs (legacy) "
                         "+ each non-primary bin's top-1 sub.")
    ap.add_argument("--shape-l2-eps-cm", type=float, default=0.5,
                    help="Fix A: distance (cm) from y_obs[7] to a bin edge "
                         "below which the neighbour bin is also pulled in.")
    ap.add_argument("--shape-per-sub-z-scale", action="store_true",
                    help="Improvement (6): use union of routed subs' "
                         "per-training-data z_scale for the CMA-ES bbox "
                         "padding, instead of the fixed Z_SCALE_NP "
                         "constant [0.20, 0.90, 1.20].  Adapts the box "
                         "size to the routed members' actual variance.")
    ap.add_argument("--shape-info-frame-w", action="store_true",
                    help="Improvement (5): replace fixed --shape-frame-w "
                         "with info-weighted Jacobian magnitude ∂y/∂z at "
                         "primary member's z_nn (3 extra GP calls). Frames "
                         "where the surrogate is most sensitive to θ get "
                         "more weight in the CMA-ES loss.")
    ap.add_argument("--shape-gaussian-prior-weight", type=float, default=0.0,
                    help="Method A: Gaussian prior weight λ added to "
                         "the CMA-ES loss as λ·Σᵢwᵢ·½‖(z−z_mu_i)/z_scale_i‖² "
                         "where (z_mu_i, z_scale_i) come from each routed "
                         "sub's training data.  Equivalent to a single "
                         "Gaussian prior centered at the precision-weighted "
                         "mean of all sub means, with covariance "
                         "(Σᵢwᵢ Λ_i⁻¹)⁻¹.  λ=1.0 = standard MAP, smaller "
                         "= weaker pull, 0 = off (legacy).  Recommended: "
                         "1.0 to anchor σ_y away from extreme bbox edges.")
    ap.add_argument("--shape-mog-prior-weight", type=float, default=0.0,
                    help="Method B: Mixture-of-Gaussians prior weight "
                         "λ added to the CMA-ES loss as "
                         "−λ · logsumexp_k [log π_k − ½‖(z−μ_k)/σ_k‖² "
                         "− Σ_j log σ_kj] where π_k = routing ensemble "
                         "weights and (μ_k, σ_k) come from each routed "
                         "sub's training data.  Compared to Method A's "
                         "weighted-average quadratic prior, the "
                         "logsumexp lets one cluster softly dominate "
                         "when z is near its mode and is robust to "
                         "wrong-cluster routing (far clusters drop out "
                         "of the logsumexp automatically).  λ=1.0 = "
                         "standard MAP, 0 = off.  Use INSTEAD of "
                         "--shape-gaussian-prior-weight, not in "
                         "addition (combining double-counts the prior).")
    ap.add_argument("--shape-long-y8-spillover-factor", type=float, default=0.0,
                    help="Flat partition only. Forces a 'long-flow' cluster's "
                         "σ_y-low sub into the ensemble with this weight share "
                         "to recover OLD length-bin's bin_4 → bin_3 spillover. "
                         "Long-flow cluster = nearest with y8_med > ratio × "
                         "y_obs[7] (default ratio=1.5). 0 = off, 0.2 = mild, "
                         "0.5 = strong σ_y anchor.")
    ap.add_argument("--shape-long-y8-spillover-ratio", type=float, default=1.5,
                    help="Multiplier on y_obs[7] for the long-y8 cluster threshold.")
    ap.add_argument("--shape-sy-split-equal-weight", action="store_true",
                    help="Flat partition only: force the two σ_y-low/high "
                         "split sub-experts of each routed cluster to have "
                         "equal ensemble weights (50/50 within their pair). "
                         "Recovers the symmetric σ_y averaging that the OLD "
                         "length-bin sy model had via bin spillover.  "
                         "gp_fit / inv_var / pca weighting all naturally "
                         "bias one half — this re-centres the σ_y ensemble.")
    ap.add_argument("--shape-mog-sigma-obs", type=float, default=0.05,
                    help="Observation noise σ (cm) added to GP variance "
                         "in --loss-mode=mog_likelihood. Prevents singular "
                         "Σ when GP is over-confident. 0.05 = ~5%% of "
                         "typical y_obs scale.")
    ap.add_argument("--shape-gp-noise-log-floor", type=float, default=0.05,
                    help="GP-aware likelihood: log-space measurement noise "
                         "floor σ_obs_log added in quadrature to delta-method "
                         "log-y std (σ_pred / y_hat). Used by "
                         "--loss-mode=log_nuisance_gp. 0.05 ≈ 5%% relative "
                         "noise floor; raise it (0.10–0.20) if the GP is "
                         "over-confident on real data and CMA-ES collapses "
                         "into low-σ_pred regions.")
    ap.add_argument("--shape-report-ci", action="store_true",
                    help="Method C: report Laplace-approx 95 %% CI on θ "
                         "computed from the numerical Hessian of the "
                         "negative-log-posterior at the MAP point. Adds 9 "
                         "extra GP evals (3 diag + 3 off-diag central "
                         "differences). Wide CI on σ_y under single-setup "
                         "indicates the well-known identifiability "
                         "degeneracy; joint setup CI should narrow it.")
    ap.add_argument("--shape-cma-seed-offset", type=int, default=0,
                    help="Improvement (3): offset added to CMA-ES seed. "
                         "Run setup1 with different offsets (0, 100, 200, "
                         "300, 400) to detect identifiability-degenerate "
                         "θ̂ clusters in surrogate loss landscape.")
    ap.add_argument("--shape-eps-cap-by-primary", action="store_true",
                    help="Fix A v2: when --shape-l2-mode=eps spillover is "
                         "active, cap the neighbour bin's contribution to "
                         "≤ primary bin sub count (top by PCA).  Avoids "
                         "the case where a denser neighbour bin dominates "
                         "the ensemble and biases σ_y the wrong direction "
                         "(e.g. Okonomiyaki: bin_3 primary 7 subs vs bin_2 "
                         "spillover 9 subs all biased high σ_y).")
    ap.add_argument("--shape-weight-mode", type=str, default="pca",
                    choices=["pca", "gp_fit", "inv_var"],
                    help="Option E: how to weight ensemble members. "
                         "'pca' = legacy softmax on PCA distance to sub "
                         "centroid in y-shape space; "
                         "'gp_fit' = softmax on ||GP(z_nn) - y_obs|| in "
                         "raw y-space — directly punishes subs whose GP "
                         "cannot predict y_obs even at its own nearest "
                         "training θ. Cross-bin comparable (unlike PCA "
                         "distance). Adds K cheap GP evals at routing time.")
    ap.add_argument("--shape-gpfit-oversample", type=int, default=3,
                    help="Option D: when --shape-weight-mode=gp_fit, "
                         "oversample factor for the PCA pre-filter. "
                         "Routing fetches K * factor candidates by PCA "
                         "distance, then re-ranks by GP-fit at z_nn and "
                         "keeps the top-K.  Larger = more thorough but "
                         "+(50-200ms × extra subs) routing overhead.")
    ap.add_argument("--shape-frame-w", type=float, nargs=8, default=None,
                    metavar=("F1","F2","F3","F4","F5","F6","F7","F8"),
                    help="Override per-frame loss weight (default "
                         "[0.2,0.2,0.35,0.5,0.8,1.0,1.4,2.0]). Use to "
                         "down-weight frames dominated by un-modeled "
                         "physics (e.g. f1-f2 surface-tension initial "
                         "yield). Pass exactly 8 floats.")
