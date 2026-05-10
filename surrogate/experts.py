"""ExactGP and SparseGP (SVGP) experts for hierarchical rBCM-MoGP.

Each (g, k) cluster owns one expert: D_Y independent single-output GPs
sharing the same 5-D input but with their own kernel + noise hyperparameters.

ExactExpert  — exact GP, O(N³) Cholesky, N ≤ 800 (GPyTorch default cap).
SparseExpert — SVGP with M inducing points, O(NM²) ELBO, no N cap.
               Use when sub N > 800 to exploit all available data.

After the refactor (2026-04-19), this module is drastically simplified:
  * No hyperprior / KL contributions
  * No joint-model dependencies — each expert is self-contained
  * Exposes standalone `fit()` for independent MLE training and
    `predict()` returning (mean, variance) needed by rBCM
"""
from __future__ import annotations
import itertools
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import gpytorch

from surrogate.gp_base import SingleOutputExactGP, make_kernel
from surrogate import config as HC


# ════════════════════════════════════════════════════════════════════════════
# Per-expert bundle: D_Y independent ExactGPs
# ════════════════════════════════════════════════════════════════════════════

class ExactExpert(nn.Module):
    """One expert for one (g, k) cluster.

    Holds D_Y independent ExactGPs. Trained by calling `.fit(...)` which
    runs standard Adam on ExactMarginalLogLikelihood, independently per
    output dimension but sharing the same (X, Y) cluster.

    Prediction returns (mean, variance) of shape (B, D_Y) each, which is
    what rBCM aggregation needs.
    """

    def __init__(self,
                 X_train: torch.Tensor,          # (N_k, D_X)
                 Y_train: torch.Tensor,          # (N_k, D_Y)
                 kernel_name: str = HC.KERNEL):
        super().__init__()
        assert X_train.ndim == 2 and Y_train.ndim == 2
        assert X_train.shape[0] == Y_train.shape[0]

        self.D_Y = Y_train.shape[1]
        self.kernel_name = kernel_name
        # Store training data on the target device. We deliberately do NOT
        # mark these as buffers so they don't bloat state_dict; they're
        # saved separately by the outer model.
        self.register_buffer("_X_train", X_train.to(HC.DEVICE).to(HC.DTYPE),
                             persistent=False)
        self.register_buffer("_Y_train", Y_train.to(HC.DEVICE).to(HC.DTYPE),
                             persistent=False)

        self.models = nn.ModuleList()
        self.likes  = nn.ModuleList()
        for j in range(self.D_Y):
            lk = gpytorch.likelihoods.GaussianLikelihood().to(HC.DEVICE).to(HC.DTYPE)
            m  = SingleOutputExactGP(self._X_train, self._Y_train[:, j], lk,
                                     kernel_name=kernel_name
                                     ).to(HC.DEVICE).to(HC.DTYPE)
            self.models.append(m)
            self.likes.append(lk)

        # Polynomial-residual correction (optional, added post-hoc by a
        # patcher that calls `fit_poly_residual`). Stored outside
        # state_dict (see outer model save/load) to avoid strict-load
        # headaches with pre-poly checkpoints.
        #   poly_powers : (n_basis, D_X) int — degree-wise exponent matrix
        #   poly_coef   : (n_basis, D_Y) float — ridge-fit coefficients,
        #                 applied as μ ← μ + Φ(X_scaled) @ poly_coef
        # Both stored in SCALED input / SCALED output space — poly lives
        # between the expert's GP mean and the outer OutputScaler.
        self.poly_powers: Optional[torch.Tensor] = None
        self.poly_coef:   Optional[torch.Tensor] = None

    # ── Training data access (for rebuilding after checkpoint load) ────────
    def set_train_data(self, X: torch.Tensor, Y: torch.Tensor):
        """Re-attach training data to all D_Y sub-models.

        Used when loading from a checkpoint where train data is saved
        separately from state_dict.
        """
        X = X.to(HC.DEVICE).to(HC.DTYPE)
        Y = Y.to(HC.DEVICE).to(HC.DTYPE)
        self._X_train = X
        self._Y_train = Y
        for j in range(self.D_Y):
            self.models[j].set_train_data(inputs=X, targets=Y[:, j],
                                          strict=False)

    # ── MLE fit (standalone, one expert at a time) ─────────────────────────
    def fit(self, n_iters: int = HC.EXPERT_N_ITERS,
            lr: float = HC.EXPERT_LR,
            tol: float = HC.EXPERT_EARLY_STOP_TOL,
            verbose: bool = False) -> float:
        """Fit kernel + noise hyperparams by Adam on ExactMLL.

        Returns final mean neg-MLL across outputs. Trains all D_Y output
        GPs jointly (shared optimiser step) for a single cluster.
        """
        params: List[torch.nn.Parameter] = []
        for j in range(self.D_Y):
            self.models[j].train()
            self.likes[j].train()
            params += list(self.models[j].parameters())
            params += list(self.likes[j].parameters())
        opt = torch.optim.Adam(params, lr=lr)
        mlls = [gpytorch.mlls.ExactMarginalLogLikelihood(self.likes[j],
                                                         self.models[j])
                for j in range(self.D_Y)]

        last = float("inf")
        plateau = 0
        final_loss = float("nan")
        for it in range(n_iters):
            opt.zero_grad()
            loss = torch.zeros((), dtype=HC.DTYPE, device=HC.DEVICE)
            for j in range(self.D_Y):
                out = self.models[j](self._X_train)
                loss = loss - mlls[j](out, self._Y_train[:, j])
            loss.backward()
            opt.step()
            cur = float(loss.detach()) / self.D_Y
            if verbose and (it % 20 == 0 or it == n_iters - 1):
                print(f"      iter {it:3d}  neg_mll_mean={cur:.4f}")
            if abs(last - cur) < tol:
                plateau += 1
                if plateau >= 10:
                    final_loss = cur
                    break
            else:
                plateau = 0
            last = cur
            final_loss = cur
        return final_loss

    # ── Prediction (returns mean + variance for rBCM) ──────────────────────
    # NOTE: No @torch.no_grad() here — Change 4's gradient-based inverse
    # needs autograd to flow from the test input X back to (n, eta, σy).
    # All production callers go through HVIMoGPrBCMPredictor.predict which
    # supplies a @torch.no_grad context, so behaviour is unchanged there.
    def predict(self, X: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) each of shape (B, D_Y).

        If `poly_powers` / `poly_coef` are attached (post-hoc Change-3
        patch), the mean is bias-corrected by adding `Φ(X) @ poly_coef`
        where Φ is the monomial basis matching `poly_powers`. Variance
        is returned unchanged — the GP variance already captures the
        uncertainty the poly is deterministically removing.
        """
        X = X.to(HC.DEVICE).to(HC.DTYPE)
        means, variances = [], []
        with gpytorch.settings.fast_pred_var():
            for j in range(self.D_Y):
                self.models[j].eval()
                self.likes[j].eval()
                p = self.likes[j](self.models[j](X))
                means.append(p.mean)
                variances.append(p.variance)
        mu  = torch.stack(means, dim=-1)
        var = torch.stack(variances, dim=-1)
        if self.poly_powers is not None and self.poly_coef is not None:
            mu = mu + self._apply_poly(X)
        return mu, var

    # ── Polynomial-residual correction helpers (Change 3) ──────────────────
    def _apply_poly(self, X: torch.Tensor) -> torch.Tensor:
        """Given (B, D_X) scaled inputs, return (B, D_Y) poly correction.

        Φ[b, j] = ∏_k X[b, k] ** poly_powers[j, k]
        correction = Φ @ poly_coef   # (B, n_basis) @ (n_basis, D_Y)
        """
        pp = self.poly_powers.to(X.device).to(X.dtype)            # (J, D_X)
        # Broadcast pow: (B, 1, D_X) ** (1, J, D_X) → (B, J, D_X) → prod
        Phi = (X.unsqueeze(1) ** pp.unsqueeze(0)).prod(dim=-1)    # (B, J)
        return Phi @ self.poly_coef.to(X.device).to(X.dtype)

    @staticmethod
    def _make_poly_powers(D_X: int, degree: int) -> np.ndarray:
        """Generate (n_basis, D_X) exponent matrix for total-degree ≤ degree
        polynomial features. Includes constant term (row of zeros)."""
        rows = []
        for total in range(degree + 1):
            for combo in itertools.combinations_with_replacement(range(D_X), total):
                p = [0] * D_X
                for c in combo:
                    p[c] += 1
                rows.append(p)
        return np.asarray(rows, dtype=np.int64)

    def fit_poly_residual(self, degree: int = 2, alpha: float = 1e-3,
                          verbose: bool = False) -> float:
        """Fit ridge-regression polynomial residual on stored training data.

        Runs once, post-GP-training: computes residual = Y_train − μ_gp(X_train)
        in scaled output space, fits ridge with degree-`degree` polynomial
        features of X_train (scaled input space), stores the coefficients
        on `self` so subsequent `predict` calls apply the correction.

        Intercept (power-zero basis) is NOT regularised (alpha=0 on its
        diagonal entry) so the poly is free to absorb a constant bias.

        Returns
        -------
        float : mean squared residual *after* poly correction, across all
        D_Y outputs — i.e. the residual variance not captured by poly.
        """
        if self._X_train is None or self._X_train.numel() == 0:
            if verbose:
                print("  [poly] empty expert — skipping")
            return float("nan")
        D_X = self._X_train.shape[1]
        powers = self._make_poly_powers(D_X, degree)   # (J, D_X)

        # GP mean on training points (scaled space, no poly applied yet)
        X = self._X_train
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mus = []
            for j in range(self.D_Y):
                self.models[j].eval()
                self.likes[j].eval()
                mus.append(self.likes[j](self.models[j](X)).mean.cpu().numpy())
        mu_gp = np.stack(mus, axis=1)                       # (N, D_Y)
        Y_np  = self._Y_train.detach().cpu().numpy()
        resid = Y_np - mu_gp                                # (N, D_Y)

        # Polynomial design matrix
        X_np = X.detach().cpu().numpy().astype(np.float64)
        Phi = np.ones((X_np.shape[0], powers.shape[0]), dtype=np.float64)
        for j, p in enumerate(powers):
            for k, pk in enumerate(p):
                if pk > 0:
                    Phi[:, j] = Phi[:, j] * (X_np[:, k] ** int(pk))

        # Ridge: coef = (Φ^T Φ + α I)^-1 Φ^T resid, with α_intercept = 0
        PtP = Phi.T @ Phi
        reg = alpha * np.eye(PtP.shape[0])
        reg[0, 0] = 0.0
        coef = np.linalg.solve(PtP + reg, Phi.T @ resid)    # (J, D_Y)

        # Store on module
        dev = X.device
        dt  = X.dtype
        self.poly_powers = torch.as_tensor(powers, dtype=torch.long, device=dev)
        self.poly_coef   = torch.as_tensor(coef,   dtype=dt,         device=dev)

        fit_resid = resid - Phi @ coef
        msr = float(np.mean(fit_resid ** 2))
        if verbose:
            msr_raw = float(np.mean(resid ** 2))
            print(f"  [poly] N={X_np.shape[0]}  D_X={D_X}  deg={degree}  "
                  f"basis={powers.shape[0]}  msr raw={msr_raw:.4e} -> "
                  f"corrected={msr:.4e}")
        return msr

    # ── Free the per-model prediction_strategy cache (OOM mitigation) ──────
    def clear_prediction_cache(self):
        """Drop gpytorch's memoized prediction_strategy on each sub-model.

        ExactGP lazily builds a (N_train × N_train) Cholesky factor the
        first time `model(X_test)` is called and keeps it alive for reuse
        across subsequent test queries. When HVIMoGP_rBCM iterates through
        hundreds of experts, those caches stack up on GPU and OOM (~400
        MB/expert × 500 experts = 200 GB). Call this after each expert's
        predict() to release that state.
        """
        # gpytorch's @cached decorator stores results on the object's
        # _memoize_cache dict (keyed by (method, args)). Dropping the
        # prediction_strategy attribute alone is insufficient — the
        # memoize cache keeps references alive — so we clear both.
        for m, lk in zip(self.models, self.likes):
            m.prediction_strategy = None
            if hasattr(m, "_memoize_cache"):
                m._memoize_cache.clear()
            # Likelihood also memoizes (e.g. marginal covariance factor).
            if hasattr(lk, "_memoize_cache"):
                lk._memoize_cache.clear()

    # ── Prior variance at any x (for rBCM correction term) ─────────────────
    @torch.no_grad()
    def prior_variance(self) -> torch.Tensor:
        """Return prior predictive variance per output dim (shape (D_Y,)).

        For a stationary kernel k(x,x) = outputscale + noise, this is
        constant in x. We include noise so σ²_prior matches the scale
        produced by `predict` (which includes likelihood noise).
        """
        vs = []
        for j in range(self.D_Y):
            m, lk = self.models[j], self.likes[j]
            # outputscale is the amplitude of the base kernel
            os = m.covar_module.outputscale
            ns = lk.noise
            vs.append(os + ns)
        return torch.stack(vs).reshape(-1)


# ════════════════════════════════════════════════════════════════════════════
# Sparse GP expert — no N cap, uses inducing points (SVGP / ELBO)
# ════════════════════════════════════════════════════════════════════════════

class _SingleOutputSVGP(gpytorch.models.ApproximateGP):
    """Single-output SVGP with learnable inducing locations."""

    def __init__(self, inducing_points: torch.Tensor, kernel_name: str = "matern25_ard"):
        vdist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        vstrat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, vdist, learn_inducing_locations=True
        )
        super().__init__(vstrat)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = make_kernel(kernel_name)
        self.kernel_name  = kernel_name

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class SparseExpert(nn.Module):
    """SVGP expert: same interface as ExactExpert but no N=800 Cholesky cap.

    Uses M inducing points (default: min(N//4, 300)), initialised by K-means
    on X_train.  Training minimises the variational ELBO with mini-batching.
    Prediction returns (mean, variance) of shape (B, D_Y), compatible with
    the rBCM aggregator and engine.py's load_sub_gp / predict_one pipeline.

    Recommended usage:
        exp = SparseExpert(X_s, Y_s, kernel_name="matern25_ard", n_inducing=200)
        exp.set_train_data(X_s, Y_s)      # attach full dataset
        exp.fit(n_iters=400, lr=0.02)
        exp.eval()
        mu, var = exp.predict(X_query)
    """

    def __init__(self,
                 X_train: torch.Tensor,   # (N, D_X)
                 Y_train: torch.Tensor,   # (N, D_Y)
                 kernel_name: str = HC.KERNEL,
                 n_inducing: int = 200):
        super().__init__()
        assert X_train.ndim == 2 and Y_train.ndim == 2
        assert X_train.shape[0] == Y_train.shape[0]

        self.D_Y = Y_train.shape[1]
        self.kernel_name = kernel_name
        N = X_train.shape[0]
        M = min(n_inducing, N)
        self.n_inducing = M

        # Store full training data (not registered as buffer — saved externally)
        self._X_train = X_train.to(HC.DEVICE).to(HC.DTYPE)
        self._Y_train = Y_train.to(HC.DEVICE).to(HC.DTYPE)

        # Initialise inducing points via K-means (or subsample if N≤M)
        Z = self._init_inducing(self._X_train, M)

        self.models: nn.ModuleList = nn.ModuleList()
        self.likes:  nn.ModuleList = nn.ModuleList()
        for _ in range(self.D_Y):
            lk = gpytorch.likelihoods.GaussianLikelihood().to(HC.DEVICE).to(HC.DTYPE)
            m  = _SingleOutputSVGP(Z.clone(), kernel_name=kernel_name
                                   ).to(HC.DEVICE).to(HC.DTYPE)
            self.models.append(m)
            self.likes.append(lk)

    @staticmethod
    def _init_inducing(X: torch.Tensor, M: int) -> torch.Tensor:
        """K-means initialisation of inducing points on CPU, returned on X.device."""
        N = X.shape[0]
        if M >= N:
            return X.clone()
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=M, n_init=5, random_state=42, max_iter=100)
            km.fit(X.detach().cpu().numpy())
            Z = torch.tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)
        except Exception:
            # Fallback: random subsample
            idx = torch.randperm(N, device=X.device)[:M]
            Z = X[idx].clone()
        return Z

    # ── Training data attachment (mirrors ExactExpert API) ──────────────────
    def set_train_data(self, X: torch.Tensor, Y: torch.Tensor):
        """Attach / replace training data. Does NOT reinitialise inducing pts."""
        self._X_train = X.to(HC.DEVICE).to(HC.DTYPE)
        self._Y_train = Y.to(HC.DEVICE).to(HC.DTYPE)

    # ── ELBO training ────────────────────────────────────────────────────────
    def fit(self,
            n_iters:    int   = 400,
            lr:         float = 0.02,
            batch_size: int   = 256,
            tol:        float = HC.EXPERT_EARLY_STOP_TOL,
            verbose:    bool  = False) -> float:
        """Train D_Y SVGPs via VariationalELBO + Adam with mini-batching.

        Returns final mean neg-ELBO per output (comparable in sign to
        ExactExpert's neg-MLL so callers can log it uniformly).
        """
        N = self._X_train.shape[0]
        bs = min(batch_size, N)

        params: List[torch.nn.Parameter] = []
        mlls: List[gpytorch.mlls.VariationalELBO] = []
        for j in range(self.D_Y):
            self.models[j].train()
            self.likes[j].train()
            params += list(self.models[j].parameters())
            params += list(self.likes[j].parameters())
            mlls.append(gpytorch.mlls.VariationalELBO(
                self.likes[j], self.models[j], num_data=N))

        opt = torch.optim.Adam(params, lr=lr)

        last = float("inf")
        plateau = 0
        final_loss = float("nan")
        perm = torch.randperm(N, device=HC.DEVICE)
        ptr = 0

        for it in range(n_iters):
            # Cyclic mini-batch (reshuffle when exhausted)
            if ptr + bs > N:
                perm = torch.randperm(N, device=HC.DEVICE)
                ptr = 0
            idx = perm[ptr: ptr + bs]
            ptr += bs
            Xb = self._X_train[idx]
            Yb = self._Y_train[idx]

            opt.zero_grad()
            loss = torch.zeros((), dtype=HC.DTYPE, device=HC.DEVICE)
            for j in range(self.D_Y):
                out  = self.models[j](Xb)
                loss = loss - mlls[j](out, Yb[:, j])
            loss.backward()
            opt.step()

            cur = float(loss.detach()) / self.D_Y
            if verbose and (it % 50 == 0 or it == n_iters - 1):
                print(f"      iter {it:4d}  neg_elbo_mean={cur:.4f}")
            if abs(last - cur) < tol:
                plateau += 1
                if plateau >= 15:
                    final_loss = cur
                    break
            else:
                plateau = 0
            last = cur
            final_loss = cur
        return final_loss

    # ── Prediction (same interface as ExactExpert) ───────────────────────────
    def predict(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, variance) each of shape (B, D_Y)."""
        X = X.to(HC.DEVICE).to(HC.DTYPE)
        means, variances = [], []
        with gpytorch.settings.fast_pred_var():
            for j in range(self.D_Y):
                self.models[j].eval()
                self.likes[j].eval()
                p = self.likes[j](self.models[j](X))
                means.append(p.mean)
                variances.append(p.variance)
        return torch.stack(means, dim=-1), torch.stack(variances, dim=-1)

    # ── Stubs for API compatibility with ExactExpert ─────────────────────────
    def clear_prediction_cache(self):
        pass  # ApproximateGP has no persistent Cholesky cache

    @torch.no_grad()
    def prior_variance(self) -> torch.Tensor:
        vs = []
        for j in range(self.D_Y):
            os = self.models[j].covar_module.outputscale
            ns = self.likes[j].noise
            vs.append(os + ns)
        return torch.stack(vs).reshape(-1)
