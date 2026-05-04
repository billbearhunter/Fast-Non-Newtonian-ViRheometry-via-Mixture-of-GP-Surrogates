"""Deterministic grid-based geo router.

Replaces sklearn BayesianGaussianMixture for layer-1 (W, H) routing.

Why grid instead of BGM?
  (W, H) is exact, exogenous (we set the test setup); there is no latent
  cluster assignment to infer. BGM uses Mahalanobis density of Gaussian
  components, which is biased by component covariance shape — corners of
  the (W, H) box (e.g. W=7, H=7) end up routed to whichever Gaussian's
  Σ has the longest tail toward that corner, NOT the closest center.

  A deterministic axis-aligned grid:
    1. Guarantees every (W, H) in domain has unambiguous geo assignment
    2. Center of corner cell is at most √(0.5² + 0.5²) ≈ 0.71 from any
       (W, H) inside that cell (vs >2.0 with BGM)
    3. Routing is just integer arithmetic — fast, robust, debuggable
    4. Survives data backfill: cell boundaries don't shift when we add
       more samples — so newly-collected MPM data goes where you expect

Compatibility:
  HVIMoGP_rBCM.attach_router() expects (geo_gmm, geo_scaler, phi_gmms,
  phi_scalers) where geo_gmm has .predict() and .predict_proba(). We mimic
  the BGM interface so existing route() / predict() code works unchanged.
"""
from __future__ import annotations
import numpy as np


class GridGeoRouter:
    """Axis-aligned grid partition of (W, H) plane.

    Parameters
    ----------
    w_edges : (Nw+1,) array
        Bin edges along W. Must be sorted ascending.
    h_edges : (Nh+1,) array
        Bin edges along H. Must be sorted ascending.

    Geo IDs are linearised as `wi * Nh + hi`, so increasing geo_id walks
    H first then W — matches the natural reading order.
    """
    def __init__(self, w_edges, h_edges):
        self.w_edges = np.asarray(w_edges, dtype=np.float64)
        self.h_edges = np.asarray(h_edges, dtype=np.float64)
        assert np.all(np.diff(self.w_edges) > 0), "w_edges must be ascending"
        assert np.all(np.diff(self.h_edges) > 0), "h_edges must be ascending"
        self.nw = len(self.w_edges) - 1
        self.nh = len(self.h_edges) - 1
        self.n_components = self.nw * self.nh

        # Cell centers (used by inspection / debugging)
        cx = 0.5 * (self.w_edges[:-1] + self.w_edges[1:])
        cy = 0.5 * (self.h_edges[:-1] + self.h_edges[1:])
        gx, gy = np.meshgrid(cx, cy, indexing="ij")
        # Shape (n_components, 2): each row is (W_center, H_center)
        self.means_ = np.stack([gx.ravel(), gy.ravel()], axis=-1)

        # For BGM-like API: uniform weights (no concept of mixture proportion)
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)

    def _bin_indices(self, wh):
        wh = np.atleast_2d(np.asarray(wh, dtype=np.float64))
        wi = np.clip(np.searchsorted(self.w_edges, wh[:, 0], side="right") - 1,
                     0, self.nw - 1)
        hi = np.clip(np.searchsorted(self.h_edges, wh[:, 1], side="right") - 1,
                     0, self.nh - 1)
        return wi, hi

    def predict(self, wh):
        """Return integer geo id per row. wh is shape (N, 2)."""
        wi, hi = self._bin_indices(wh)
        return (wi * self.nh + hi).astype(np.int64)

    def predict_proba(self, wh):
        """Return one-hot soft assignment shape (N, n_components)."""
        ids = self.predict(wh)
        proba = np.zeros((len(ids), self.n_components), dtype=np.float64)
        proba[np.arange(len(ids)), ids] = 1.0
        return proba

    def __repr__(self):
        return (f"GridGeoRouter(w_edges={self.w_edges.tolist()}, "
                f"h_edges={self.h_edges.tolist()}, "
                f"n_components={self.n_components})")


class IdentityScaler:
    """No-op scaler that mimics sklearn's transform/inverse_transform API.

    Used in place of StandardScaler when we don't want any normalisation
    (e.g. with GridGeoRouter, since (W, H) is already in physical units
    that the grid edges are defined in).
    """
    def __init__(self, n_features=2):
        self.mean_ = np.zeros(n_features, dtype=np.float64)
        self.scale_ = np.ones(n_features, dtype=np.float64)
        self.var_ = np.ones(n_features, dtype=np.float64)
        self.n_features_in_ = n_features

    def transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float64)

    def fit(self, X):
        return self

    def fit_transform(self, X):
        return self.transform(X)

    def __repr__(self):
        return f"IdentityScaler(n_features={self.n_features_in_})"
