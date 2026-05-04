"""Analytical Hessian-based active setup selection.

Ported from prev-work `libs/mechanism.py` (Hamamichi et al. 2023).

The math is purely analytic — Poiseuille flow loss has a closed-form
Hessian over (η, n, σ_y) at any candidate (W, H). The proposer:

    1. For each already-observed setup s_i, compute the rescaled Hessian
       H_i and its principal eigenvector q_i (= the parameter direction
       that loss is *least* sensitive to → max posterior uncertainty).
    2. Sweep candidate (W, H) on a 1cm grid over the box.
    3. Pick the candidate whose own q_dash is most orthogonal to the
       union of {q_i}: minimise Σ_i |q_i · q_dash|² over candidates.

This gives the next physical experiment that maximally constrains the
direction(s) currently uncertain — no MPM or surrogate forward passes
required.

CGS units (cm, g, s) throughout, matching prev-work convention.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


# ─── prev-work CGS constants (libs/const.py CGS class) ──────────────────────
MIN_ETA, MAX_ETA = 0.001, 300.0
MIN_N, MAX_N = 0.3, 1.0
MIN_SIGMA_Y, MAX_SIGMA_Y = 0.0, 400.0
MIN_W, MAX_W = 2.0, 7.0
MIN_H, MAX_H = 2.0, 7.0
EXTENT_ETA = np.array([MIN_ETA, MAX_ETA])
EXTENT_N = np.array([MIN_N, MAX_N])
EXTENT_SIGMA_Y = np.array([MIN_SIGMA_Y, MAX_SIGMA_Y])

_EPS = 1e-4

# Theta_C polynomial coefficients (prev-work libs/compare_loss.py).
# Verbatim — used only inside `mat_hw_to_PL`.
_THETA_C = np.array([
    -1.37321779e+01,  8.14275110e-03, -1.09765321e+00, -8.00907812e-04,
     4.94833928e+00, -2.37988743e+00, -3.02972824e-03, -3.29177317e-01,
     3.79945757e-04,  2.93519945e-03, -2.55175315e-03, -2.57625369e-04,
    -1.09840507e-03,  1.04878754e+00, -1.27088316e-05,  3.57459552e-02,
    -3.86847818e-04,  1.06611942e+00, -1.72646031e-02,  4.11885913e-05,
     5.11073420e+00, -1.26161737e+00, -1.24167310e-04,  8.26656079e-03,
    -5.16803036e-08, -1.38268842e-05, -1.06641050e-05, -8.82504323e+01,
    -7.30494656e+00, -7.78391030e-03,  2.28734558e-05, -2.84809983e-05,
     2.59136974e-08,  5.31670213e-04,  2.13557773e+01,  4.34277901e-04,
    -1.92921774e-03, -6.04553328e-05,  4.17722294e+00,  2.49944076e-05,
     9.16347259e-03,  2.60463552e+01,  7.04927334e-05,  7.35523906e-06,
    -1.36230114e-05,  2.25393983e+01,  6.08146911e-07, -3.07427873e-01,
     1.29895721e-02, -9.46927385e-06, -1.90529838e-06, -4.84522719e-10,
     2.79472840e-02,  7.56647218e-02,  4.18255477e-08, -9.91701259e+00,
     6.77000023e-06,  1.72139168e-09,  1.83779067e-08,  4.39811697e-04,
     6.66829013e-08, -3.57121832e-04, -1.03725625e-04, -1.59476660e-09,
     2.57842478e-05,  5.92792423e-05,
])

_C_EXT_LO, _C_EXT_HI = 5.0, 500.0
_PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])


# ─── data classes (prev-work libs/setup.py, param.py) ───────────────────────

@dataclass
class Setup:
    H: float       # cm
    W: float       # cm
    weight: float = 1.0


@dataclass
class Param:
    eta: float     # CGS poise
    n: float
    sigmaY: float  # CGS dyne/cm²


# ─── prev-work libs/conversion_function.py + compare_loss.mat_hw_to_PL ──────

def _f_mat_scalar_compact_with_inverse(M, HW, theta, order=2):
    """Compact symmetric polynomial expansion used by the H/W → P/L surrogate."""
    MHW = np.concatenate([M, HW])
    MHW_inv = np.concatenate([
        MHW,
        np.array([1.0 / MHW[0], 1.0 / MHW[1], 1.0 / (MHW[2] + _EPS),
                  1.0 / MHW[3], 1.0 / MHW[4]]),
    ])
    ret = theta[0]
    prev = 1
    for k in range(1, order + 1):
        prod_t = _PRIMES.copy()
        prod = MHW_inv.copy()
        for _ in range(k - 1):
            prod_t = np.tensordot(prod_t, _PRIMES, axes=0)
            prod = np.tensordot(prod, MHW_inv, axes=0)
        prod_t = prod_t.reshape(-1)
        prod = prod.reshape(-1)
        _, idx, counts = np.unique(prod_t, return_index=True, return_counts=True)
        prod_uni = prod[idx] * counts
        n_terms = prod_uni.size
        ret += float(np.dot(theta[prev:prev + n_terms], prod_uni)) / math.factorial(k)
        prev += n_terms
    return ret


def mat_hw_to_PL(eta_mks: float, n: float, sigmaY_mks: float, Hcm: float, Wcm: float) -> Tuple[float, float]:
    """(η_mks, n, σ_y_mks, H_cm, W_cm) → (P, L) Poiseuille parameters (CGS-style return)."""
    HW = np.array([Hcm * 0.01, Wcm * 0.01])
    rate = _f_mat_scalar_compact_with_inverse([eta_mks, n, sigmaY_mks], HW, _THETA_C, 2)
    C_pred = rate * (_C_EXT_HI - _C_EXT_LO) + _C_EXT_LO
    P0 = 2500.0
    L0 = C_pred / P0
    return P0, L0


# ─── analytical Hessian (prev-work libs/mechanism.Mechanism.singleHessian) ──

def _setup_valid(P: float, L: float, m: Param) -> bool:
    if P * L - m.sigmaY <= 0.0:
        return False
    eta, _, sigmaY = m.eta, m.n, m.sigmaY
    l = sigmaY / P
    W = P * (L - l) / eta
    return not (math.isnan(W) or W <= 0.0)


def single_hessian(m: Param, P: float, L: float) -> np.ndarray:
    if not _setup_valid(P, L, m):
        raise ValueError(f"invalid setup for params {m}")
    eta, n, sigmaY = m.eta, m.n, m.sigmaY
    l = sigmaY / P
    W = P * (L - l) / eta

    S = math.pow(W, (n + 1.0) / n) * eta / (P * (n + 1.0))
    A_eta = -S / eta
    A_n = S * (1.0 / (n + 1.0) - math.log(W) / n)
    B_n = S / n
    A_sy = -math.pow(W, 1.0 / n) / P

    C1 = (1 + n) ** 2 / ((1 + 2 * n) * (2 + 3 * n))
    C2 = n * n * (3 + 5 * n) * (1 + n) / ((1 + 2 * n) ** 2 * (2 + 3 * n) ** 2)
    C3 = (2 + 3 * n) / (2 * (1 + n) * (1 + 2 * n))
    C4 = n ** 3 / ((2 + 3 * n) ** 3)
    C5 = n * n * (3 + 4 * n) / (4 * (1 + n) ** 2 * (1 + 2 * n) ** 2)
    C6 = 1.0 / ((1 + n) * (2 + n))

    H_ee = 2 * A_eta * A_eta * l + 4 * A_eta * A_eta * (L - l) * C1
    H_en = 2 * A_eta * A_n * l + 4 * A_eta * A_n * (L - l) * C1 - 2 * A_eta * B_n * (L - l) * C2
    H_es = 2 * A_eta * A_sy * l + 2 * A_eta * A_sy * (L - l) * C3
    H_nn = 2 * A_n * A_n * l + 4 * B_n * B_n * (L - l) * C4 - 4 * A_n * B_n * (L - l) * C2 + 4 * A_n * A_n * (L - l) * C1
    H_ns = 2 * A_n * A_sy * l + 2 * A_n * A_sy * (L - l) * C3 - 2 * B_n * A_sy * (L - l) * C5
    H_ss = 2 * A_sy * A_sy * l + 4 * A_sy * A_sy * (L - l) * C6
    return np.array([[H_ee, H_en, H_es], [H_en, H_nn, H_ns], [H_es, H_ns, H_ss]])


def _principal_normal_rescaled(H: np.ndarray) -> np.ndarray:
    """Eigenvector of largest eigenvalue, rescaled to unit-box coordinates."""
    s, Q = np.linalg.eig(H)
    n = np.ravel(Q[:, np.argmax(s)])
    A = np.diag([
        EXTENT_ETA[1] - EXTENT_ETA[0],
        EXTENT_N[1] - EXTENT_N[0],
        EXTENT_SIGMA_Y[1] - EXTENT_SIGMA_Y[0],
    ])
    n_tilde = A @ n
    return n_tilde / np.linalg.norm(n_tilde)


# ─── public API ─────────────────────────────────────────────────────────────

def propose_next_setup(
    m_hat_cgs: Param,
    observed_setups: Sequence[Setup],
    grid_step_cm: float = 0.1,
) -> Setup:
    """Pick (W*, H*) that is most orthogonal to all observed setups' max-uncertainty direction.

    Args:
        m_hat_cgs: best-fit Param after CMA on existing setup(s) (CGS units).
        observed_setups: list of already-run Setup(H, W, weight) (CGS, H/W in cm).
        grid_step_cm: candidate (W, H) grid resolution (default 1 mm).

    Returns:
        Setup(H*, W*, weight=1.0) for the recommended next experiment.
    """
    # 1. Each observed setup's principal uncertainty direction.
    qs: List[np.ndarray] = []
    for s in observed_setups:
        P0, L0 = mat_hw_to_PL(m_hat_cgs.eta * 0.1, m_hat_cgs.n,
                              m_hat_cgs.sigmaY * 0.1, s.H, s.W)
        P, L = P0 / 10.0, L0 * 100.0
        qs.append(_principal_normal_rescaled(single_hessian(m_hat_cgs, P, L)))

    # 2. Sweep candidate grid; minimise sum of squared projections.
    Hs = np.arange(MIN_H, MAX_H + 1e-9, grid_step_cm)
    Ws = np.arange(MIN_W, MAX_W + 1e-9, grid_step_cm)
    best_proj = sys.float_info.max
    best = Setup(MIN_H, MIN_W, 1.0)
    for Hcm in Hs:
        for Wcm in Ws:
            P0, L0 = mat_hw_to_PL(m_hat_cgs.eta * 0.1, m_hat_cgs.n,
                                  m_hat_cgs.sigmaY * 0.1, float(Hcm), float(Wcm))
            P, L = P0 / 10.0, L0 * 100.0
            try:
                if not _setup_valid(P, L, m_hat_cgs):
                    continue
                q_dash = _principal_normal_rescaled(single_hessian(m_hat_cgs, P, L))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            proj_sq = sum(float(np.dot(q, q_dash)) ** 2 for q in qs)
            if proj_sq < best_proj:
                best_proj = proj_sq
                best = Setup(float(Hcm), float(Wcm), 1.0)
    return best


__all__ = [
    "Setup", "Param", "mat_hw_to_PL", "single_hessian",
    "propose_next_setup",
]
