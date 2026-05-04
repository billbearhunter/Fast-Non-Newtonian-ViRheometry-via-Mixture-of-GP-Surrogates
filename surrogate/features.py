"""
surrogate/features.py
=====================
Feature engineering for MoE gating and GP inputs.

Previously duplicated in:
  - DataPipeline/moe_utils.py  (build_phi, build_input_features)
  - Optimization/libs/moe_core.py  (build_phi, build_input_phi)
  - Optimization/soft_interpolate.py  (build_phi)
"""

import numpy as np


def build_phi(df_or_y, W=None, H=None, eps: float = 1e-8) -> np.ndarray:
    """Build the 18-dim gating feature vector phi.

    Accepts either:
    - a DataFrame with columns x_01..x_08, width, height  (training mode)
    - a 1-D array y (8 values) + W, H scalars             (inference mode)

    Returns: (N, 18) array
    """
    import pandas as pd

    if isinstance(df_or_y, pd.DataFrame):
        df = df_or_y
        Y  = df[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
        W_arr = df["width"].to_numpy(dtype=np.float64)
        H_arr = df["height"].to_numpy(dtype=np.float64)
    else:
        Y     = np.asarray(df_or_y, dtype=np.float64).reshape(1, -1)
        W_arr = np.full(1, W, dtype=np.float64)
        H_arr = np.full(1, H, dtype=np.float64)

    denom  = Y[:, [-1]] + eps
    Y_norm = Y / denom                                        # (8)
    dY     = np.diff(Y_norm, axis=1)                          # (7)
    scale  = np.log(np.abs(Y[:, -1]) + eps).reshape(-1, 1)    # (1)
    g1     = np.log(np.sqrt(W_arr * H_arr) + eps).reshape(-1, 1)   # (1)
    g2     = np.log((W_arr + eps) / (H_arr + eps)).reshape(-1, 1)   # (1)

    return np.hstack([Y_norm, dY, scale, g1, g2])   # 8+7+1+1+1 = 18


# ──────────────────────────────────────────────────────────────────────────────
# v6 feature: shape + Legendre dynamics + arrest/decay + geometry-aware scale
# See docs/v6_mathematical_theory.md §3.1
# ──────────────────────────────────────────────────────────────────────────────

# Legendre design matrix on the 8 frame indices (cached at import).
# t_i ∈ [-1, 1] uniformly spaced for i = 1..8.
# We need P_l(t) for l = 1..4 evaluated at each t_i.
_T_LEGENDRE = np.linspace(-1.0, 1.0, 8)

def _legendre_basis_matrix() -> np.ndarray:
    """Return (8, 4) matrix B with columns P_1(t), P_2(t), P_3(t), P_4(t)."""
    t = _T_LEGENDRE
    # Standard Legendre polynomials
    P1 = t
    P2 = 0.5  * (3.0 * t**2 - 1.0)
    P3 = 0.5  * (5.0 * t**3 - 3.0 * t)
    P4 = 0.125 * (35.0 * t**4 - 30.0 * t**2 + 3.0)
    return np.stack([P1, P2, P3, P4], axis=-1)   # (8, 4)


_B_LEG = _legendre_basis_matrix()
# Pre-compute Moore-Penrose pseudoinverse for projection: c = (B^T B)^-1 B^T y
_B_LEG_PINV = np.linalg.pinv(_B_LEG)              # (4, 8)


def build_phi_v6(df_or_y, W=None, H=None, eps: float = 1e-8) -> np.ndarray:
    """Build the 27-dim v6 gating feature vector.

    Composition (matches docs/v6_mathematical_theory.md §3.1):
      [0:8]   y / y_8                              (8)  shape baseline
      [8:15]  diff(y / y_8)                        (7)  velocity-shape baseline
      [15]    log(y_8)                             (1)  final-extent log scale
      [16:20] Legendre coefficients c_1..c_4       (4)  polynomial dynamics
      [20]    endpoint_slope = (y_8 - y_7) / y_8   (1)  arrest descriptor
      [21]    late_motion_ratio = (y_8 - y_6)/y_8  (1)
      [22]    early_motion_ratio = (y_3 - y_1)/y_8 (1)
      [23]    decay_ratio = log(mean(v_5..v_7) / mean(v_1..v_2))  (1)
      [24]    log(sqrt(W H))                       (1)  geometry scale
      [25]    log(W / H)                           (1)  aspect ratio
      [26]    log(y_8 / sqrt(W H))                 (1)  geometry-aware scale

    Total: 27 dims.

    Accepts the same shapes as build_phi:
      - DataFrame with x_01..x_08, width, height  (training mode)
      - 1-D array y (8 values) + W, H scalars     (inference mode)

    Returns: (N, 27) array
    """
    import pandas as pd

    if isinstance(df_or_y, pd.DataFrame):
        df = df_or_y
        Y  = df[[f"x_{i:02d}" for i in range(1, 9)]].to_numpy(dtype=np.float64)
        W_arr = df["width"].to_numpy(dtype=np.float64)
        H_arr = df["height"].to_numpy(dtype=np.float64)
    else:
        Y     = np.asarray(df_or_y, dtype=np.float64).reshape(1, -1)
        W_arr = np.full(1, W, dtype=np.float64)
        H_arr = np.full(1, H, dtype=np.float64)

    N = Y.shape[0]
    y8 = Y[:, [-1]] + eps                                # (N,1)
    Y_norm = Y / y8                                       # (N,8)
    dY = np.diff(Y_norm, axis=1)                          # (N,7)
    log_y8 = np.log(np.abs(Y[:, -1]) + eps).reshape(-1, 1)   # (N,1)

    # Legendre coefficients c_1..c_4 fitted to Y_norm against P_l(t_i)
    # c (N,4) = Y_norm (N,8) @ B_pinv.T (8,4)
    c_leg = Y_norm @ _B_LEG_PINV.T                        # (N,4)

    # Arrest descriptors
    v = np.diff(Y, axis=1)                                # (N,7) raw velocities
    endpoint_slope = (Y[:, [-1]] - Y[:, [-2]]) / y8       # = v_7/y_8
    late_motion = (Y[:, [-1]] - Y[:, [-3]]) / y8           # = (v_6+v_7)/y_8
    early_motion = (Y[:, [2]] - Y[:, [0]]) / y8            # = (v_1+v_2)/y_8

    mean_late_v  = np.mean(v[:, 4:7], axis=1, keepdims=True) + eps  # mean(v_5,v_6,v_7)
    mean_early_v = np.mean(v[:, 0:2], axis=1, keepdims=True) + eps  # mean(v_1,v_2)
    # Use absolute values so log is defined even with tiny negative numerical noise
    decay_ratio = np.log(np.abs(mean_late_v) + eps) - np.log(np.abs(mean_early_v) + eps)

    # Geometry scale
    g1 = np.log(np.sqrt(W_arr * H_arr) + eps).reshape(-1, 1)
    g2 = np.log((W_arr + eps) / (H_arr + eps)).reshape(-1, 1)
    g3 = np.log((Y[:, -1] + eps) / (np.sqrt(W_arr * H_arr) + eps)).reshape(-1, 1)

    return np.hstack([
        Y_norm,                          # 8
        dY,                              # 7
        log_y8,                          # 1
        c_leg,                           # 4
        endpoint_slope,                  # 1
        late_motion,                     # 1
        early_motion,                    # 1
        decay_ratio,                     # 1
        g1,                              # 1
        g2,                              # 1
        g3,                              # 1
    ])   # total 27


# Indices into phi_v6 for ablation slicing.
PHI_V6_BLOCKS = {
    # Coarse blocks
    "shape_norm":   (0, 8),    # y / y_8 all 8 frames
    "shape_diff":   (8, 15),   # diff(y/y_8) all 7 differences
    "log_y8":       (15, 16),  # log y_8
    "legendre":     (16, 20),  # c_1..c_4
    "arrest":       (20, 24),  # endpoint, late_motion, early_motion, decay_ratio
    "geo":          (24, 27),  # log sqrt(WH), log W/H, log(y_8/sqrt(WH))

    # Fine blocks for "late_geo" ablation (drops early frames where real
    # video extraction has highest noise — splash, threshold ambiguity,
    # frame-rate quantisation around release moment).
    "shape_norm_late": (4, 8),   # y_5/y_8, y_6/y_8, y_7/y_8, y_8/y_8 — converged extent ratios
    "shape_diff_late": (11, 15), # Δ(y_5..y_8) — late-stage velocity shape
    "legendre_high":   (18, 20), # c_3, c_4 only (higher-order arrest / overshoot)
    "arrest_no_early": (20, 22), # endpoint_slope, late_motion_ratio (drop early_motion)
    "decay":           (23, 24), # decay_ratio (still useful — log ratio of late/early v)

    # v6.1 strict per docs/v6_1_integrated_plan.md §5.1:
    # split the old "geo" into pure-setup (no y_8) and scale-geometry (contains y_8).
    "pure_setup":      (24, 26), # log sqrt(WH), log(W/H) — purely (W,H) determined
    "scale_geo":       (26, 27), # log(y_8 / sqrt(WH)) — interaction of scale and geometry
}

# Semantic-block grouping per docs/v6_1_integrated_plan.md §5.2.
# Each entry is a list of (offset, length) pairs that compose ONE block in
# the v6.1 block-diagonal covariance. Ordering matters because the dryrun
# script slices in this order.
#
# Block layout for `late_geo_v61` ablation:
#   late      : 13 dims (shape + arrest + decay + legendre_high)
#   scale     :  1 dim  (log y_8)
#   setup     :  2 dims (log √(WH), log(W/H))
#   scalegeo  :  1 dim  (log(y_8/√(WH)))
#   PP block (training-only) lives outside phi entirely, dimension 2.
# Total phi dim = 17, joint feature dim = 19.
PHI_V6_BLOCK_GROUPS = {
    "late_geo": {
        "phi_blocks": [
            ("shape_norm_late", "shape_diff_late",
             "log_y8", "legendre_high", "arrest_no_early", "decay", "geo"),
        ],   # single coarse block (back-compat with the original late_geo dry-run)
    },
    "late_geo_v61": {
        "phi_blocks": [
            ("shape_norm_late", "shape_diff_late",
             "legendre_high", "arrest_no_early", "decay"),   # late shape (13D)
            ("log_y8",),                                       # scale (1D)
            ("pure_setup",),                                   # setup (2D)
            ("scale_geo",),                                    # scale-geo (1D)
        ],
    },
}

# Ablation presets: which blocks to include.
PHI_V6_ABLATIONS = {
    "base":              ["shape_norm", "shape_diff", "log_y8", "geo"],          # ~v5 + g3
    "base_plus_legendre": ["shape_norm", "shape_diff", "log_y8", "legendre", "geo"],
    "base_plus_arrest":  ["shape_norm", "shape_diff", "log_y8", "arrest", "geo"],
    "full_v6":           ["shape_norm", "shape_diff", "log_y8",
                          "legendre", "arrest", "geo"],
    # "late_geo": drop early-frame noise-prone features. Keeps converged
    # extent (y_8 and y_5..y_8 ratios), late-stage diffs, higher-order
    # Legendre coefs, late arrest descriptors, decay_ratio, and geometry.
    # 4 + 4 + 1 + 2 + 2 + 1 + 3 = 17 dims.
    "late_geo":          ["shape_norm_late", "shape_diff_late", "log_y8",
                          "legendre_high", "arrest_no_early", "decay", "geo"],
    # "late_geo_v61": v6.1 STRICT spec per docs/v6_1_integrated_plan.md §5.1.
    # Same physical content as late_geo, but the "geo" coarse block is split
    # into pure_setup (no y_8) and scale_geo (contains y_8) so the §5.2
    # block-diagonal covariance can keep absolute-flow-scale signal in its
    # own block. Total dim = 13 + 1 + 2 + 1 = 17. (Same dimensionality as
    # late_geo, just re-blocked.)
    "late_geo_v61":      ["shape_norm_late", "shape_diff_late",
                          "legendre_high", "arrest_no_early", "decay",
                          "log_y8", "pure_setup", "scale_geo"],
}


def slice_phi_v6(phi_v6: np.ndarray, ablation: str) -> np.ndarray:
    """Extract a sub-feature corresponding to an ablation preset."""
    if ablation not in PHI_V6_ABLATIONS:
        raise ValueError(f"Unknown ablation {ablation!r}; "
                         f"choose from {list(PHI_V6_ABLATIONS)}")
    blocks = PHI_V6_ABLATIONS[ablation]
    parts = []
    for name in blocks:
        lo, hi = PHI_V6_BLOCKS[name]
        parts.append(phi_v6[:, lo:hi])
    return np.concatenate(parts, axis=1)


def build_input_features(df_or_params, W=None, H=None, eps: float = 1e-8) -> np.ndarray:
    """Build 5D input feature vector for input-space GMM gating.

    Features: (n, log(eta), log(sigma_y), W, H)

    Accepts either:
    - a DataFrame with columns n, eta, sigma_y, width, height  (training mode)
    - an array of shape (N, 3) with [n, eta, sigma_y] + W, H   (inference mode)

    Returns: (N, 5) array
    """
    import pandas as pd

    if isinstance(df_or_params, pd.DataFrame):
        df = df_or_params
        return np.column_stack([
            df["n"].values.astype(np.float64),
            np.log(df["eta"].values.astype(np.float64) + eps),
            np.log(df["sigma_y"].values.astype(np.float64) + eps),
            df["width"].values.astype(np.float64),
            df["height"].values.astype(np.float64),
        ])
    else:
        p = np.asarray(df_or_params, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(1, -1)
        return np.column_stack([
            p[:, 0],
            np.log(p[:, 1] + eps),
            np.log(p[:, 2] + eps),
            np.full(len(p), W, dtype=np.float64),
            np.full(len(p), H, dtype=np.float64),
        ])
