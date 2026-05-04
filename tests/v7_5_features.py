"""v7.5 feature + routing helpers.

Implements:
  - build_phi_v8_route       : 5d routing feature (shape + magnitude + geo)
  - make_real_covariance_v8  : 5d real-noise covariance for BGM_real
  - late_weighted_y_distance : frame-weighted distance for y-closest candidate routing
  - regularize_vbgmm         : eigenvalue floor + condition cap for VBGMM components
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from surrogate.features import _B_LEG_PINV


# ============== CONSTANTS (per v7.4 LOCKED) ==============
V8_BLOCK_DIMS = [3, 1, 1]   # shape (3) + magnitude (1) + geometry (1)
W_MAG_DEFAULT = 2.0
FRAME_WEIGHTS = np.array([0.2, 0.3, 0.5, 0.7, 1.0, 1.0, 1.0, 1.5])

EIG_FLOOR = 1e-4
COND_CAP = 1e4
N_MIN_PER_SUB = 20

VBGMM_K_MAX = 8
VBGMM_WEIGHT_PRIOR = 0.1
VBGMM_ACTIVE_THRESHOLD = 0.05


# ============== Layer 2 routing feature ==============

def build_phi_v8_route(df_or_y, W=None, H=None, eps: float = 1e-9) -> np.ndarray:
    """5d routing feature: [y5/y8, y7/y8, c_3, log(y8), log(W/H)].

    Accepts:
      - DataFrame with x_01..x_08, width, height (training mode, N rows)
      - 1-D array y (8 values) + W, H scalars     (inference mode)

    Returns: (N, 5) array (RAW, not standardized).
    """
    if isinstance(df_or_y, pd.DataFrame):
        Y = df_or_y[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
        W_arr = df_or_y["width"].to_numpy(dtype=np.float64)
        H_arr = df_or_y["height"].to_numpy(dtype=np.float64)
    else:
        Y = np.asarray(df_or_y, dtype=np.float64).reshape(1, -1)
        W_arr = np.full(1, W, dtype=np.float64)
        H_arr = np.full(1, H, dtype=np.float64)

    y8 = Y[:, [-1]] + eps                       # (N, 1)
    Y_norm = Y / y8                              # (N, 8)

    # Legendre c_3 (3rd-order curvature on normalized shape)
    c_leg = Y_norm @ _B_LEG_PINV.T               # (N, 4)

    out = np.column_stack([
        Y_norm[:, 4],                            # [0] y_5/y_8
        Y_norm[:, 6],                            # [1] y_7/y_8
        c_leg[:, 2],                             # [2] legendre c_3
        np.log(np.abs(Y[:, -1]) + eps),          # [3] log_y_8 (magnitude)
        np.log((W_arr + eps) / (H_arr + eps)),   # [4] log(W/H)
    ])
    return out


# ============== Real-noise covariance R for 5d ==============

def make_real_covariance_v8(w_mag: float = W_MAG_DEFAULT,
                             scale: float = 1.0) -> np.ndarray:
    """Real-measurement noise covariance in standardized + boosted space.

    Each diagonal entry: noise as fraction of standardized std.
    log_y_8 entry incorporates W_MAG boost (boost amplifies its noise too).
    """
    diag = np.array([
        0.30**2,                # [0] y_5/y_8: ~30% of std
        0.15**2,                # [1] y_7/y_8: closer to 1, more stable
        0.40**2,                # [2] c_3: legendre coefficients are noisier
        (0.20 * w_mag)**2,      # [3] log_y_8: ~20% × boost
        0.05**2,                # [4] log(W/H): W, H precisely measured
    ])
    return np.diag(diag * scale)


# ============== Late-weighted y-distance (Layer 2 candidate routing) ==============

def late_weighted_y_distance(y_obs: np.ndarray,
                              basin_y_mean: np.ndarray,
                              basin_y_std: np.ndarray,
                              eps: float = 1e-6) -> float:
    """Frame-weighted z-distance (per Codex Caution 3).

    Equal-weight 8-frame distance would re-introduce early-frame noise.
    Use FRAME_WEIGHTS to emphasize late frames.

    Args:
        y_obs:       (8,) observed flow profile
        basin_y_mean: (8,) basin's training y mean
        basin_y_std: (8,) basin's training y std

    Returns: scalar weighted distance.
    """
    sigma = np.maximum(basin_y_std, eps)
    z_diff = (y_obs - basin_y_mean) / sigma
    return float(np.sqrt(np.sum((FRAME_WEIGHTS * z_diff) ** 2)))


def y_closest_basins(y_obs: np.ndarray,
                       basin_y_mean: np.ndarray,
                       basin_y_std: np.ndarray,
                       active_basins: np.ndarray = None,
                       top_k: int = 3) -> np.ndarray:
    """Return top-K basin indices by late-weighted y-distance.

    Args:
        basin_y_mean: (K_basins, 8)
        basin_y_std:  (K_basins, 8)
        active_basins: optional (K_basins,) bool mask
    """
    K = basin_y_mean.shape[0]
    dists = np.full(K, np.inf)
    for k in range(K):
        if active_basins is not None and not active_basins[k]:
            continue
        dists[k] = late_weighted_y_distance(y_obs, basin_y_mean[k], basin_y_std[k])
    return np.argsort(dists)[:top_k]


# ============== Layer 3: VBGMM split + regularize ==============

def to_z(df_or_arr) -> np.ndarray:
    """(n, log eta, log sigma_y). Accepts DataFrame or (N, 3+) array."""
    if isinstance(df_or_arr, pd.DataFrame):
        n  = df_or_arr["n"].to_numpy(dtype=np.float64)
        e  = df_or_arr["eta"].to_numpy(dtype=np.float64)
        sy = df_or_arr["sigma_y"].to_numpy(dtype=np.float64)
    else:
        a = np.asarray(df_or_arr, dtype=np.float64)
        n, e, sy = a[:, 0], a[:, 1], a[:, 2]
    return np.column_stack([n, np.log(np.maximum(e, 1e-9)),
                              np.log(np.maximum(sy, 1e-9))])


def split_basin_vbgmm(df_basin: pd.DataFrame,
                       K_max: int = VBGMM_K_MAX,
                       weight_prior: float = VBGMM_WEIGHT_PRIOR,
                       active_threshold: float = VBGMM_ACTIVE_THRESHOLD,
                       seed: int = 42) -> dict:
    """VBGMM in z_theta. Auto-K via Dirichlet process.

    Returns dict with:
        mu_z (K_a, 3), cov_z (K_a, 3, 3) regularized,
        pi (K_a,), membership (N, K_a) soft, hard_assign (N,),
        N_per_sub (K_a,), fallback_only (K_a,) bool,
        K_active (int), K_max_used (int).
    """
    z = to_z(df_basin)
    N = z.shape[0]

    # Need at least N_min for VBGMM fit
    if N < 30:
        # Too small to split — single sub
        return _single_sub_fallback(z, df_basin)

    # Adapt K_max to data
    K_max_eff = min(K_max, max(2, N // 30))

    try:
        vbgm = BayesianGaussianMixture(
            n_components=K_max_eff,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=weight_prior,
            covariance_type='full',
            max_iter=200,
            random_state=seed,
            reg_covar=1e-4,
        ).fit(z)
    except Exception:
        return _single_sub_fallback(z, df_basin)

    weights = vbgm.weights_
    active = weights > active_threshold
    if active.sum() == 0:
        return _single_sub_fallback(z, df_basin)

    means = vbgm.means_[active]
    covs_raw = vbgm.covariances_[active]
    pis = weights[active] / weights[active].sum()
    membership_full = vbgm.predict_proba(z)[:, active]   # (N, K_a)
    hard = membership_full.argmax(axis=1)                 # (N,)

    # Regularize each cov
    covs_reg = []
    fallback_only = []
    N_per_sub = []
    for s_idx in range(active.sum()):
        Σ = covs_raw[s_idx].copy()
        Σ_reg = _regularize_cov(Σ)
        covs_reg.append(Σ_reg)
        N_s = int((hard == s_idx).sum())
        N_per_sub.append(N_s)
        fallback_only.append(N_s < N_MIN_PER_SUB)

    return {
        'mu_z':          means,
        'cov_z':         np.array(covs_reg),
        'pi':            pis,
        'membership':    membership_full,
        'hard_assign':   hard,
        'N_per_sub':     np.array(N_per_sub),
        'fallback_only': np.array(fallback_only, dtype=bool),
        'K_active':      int(active.sum()),
        'K_max_used':    K_max_eff,
        'method':        'vbgmm',
    }


def _single_sub_fallback(z: np.ndarray, df_basin: pd.DataFrame) -> dict:
    """Single sub fallback when VBGMM cannot fit."""
    N = z.shape[0]
    mu = z.mean(0)
    if N >= 3:
        Σ = np.cov(z.T) + 1e-3 * np.eye(3)
    else:
        Σ = np.eye(3)
    Σ_reg = _regularize_cov(Σ)
    return {
        'mu_z':          mu[None, :],
        'cov_z':         Σ_reg[None, :, :],
        'pi':            np.array([1.0]),
        'membership':    np.ones((N, 1)),
        'hard_assign':   np.zeros(N, dtype=int),
        'N_per_sub':     np.array([N]),
        'fallback_only': np.array([N < N_MIN_PER_SUB], dtype=bool),
        'K_active':      1,
        'K_max_used':    1,
        'method':        'fallback_single',
    }


def _regularize_cov(Σ: np.ndarray,
                     eig_floor: float = EIG_FLOOR,
                     cond_cap: float = COND_CAP) -> np.ndarray:
    """Floor small eigenvalues + cap condition number."""
    eigvals, eigvecs = np.linalg.eigh(Σ)
    eigvals = np.maximum(eigvals, eig_floor)
    eigvals = np.maximum(eigvals, eigvals.max() / cond_cap)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
