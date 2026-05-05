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
    # legacy fields some helpers expect
    p_basin: float = 1.0
    meta: dict = field(default_factory=dict)


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
    val_rwrms = float(ck.get("val_rel_wrms_median", 0.0))
    val_p90 = float(ck.get("val_rel_wrms_p90", 0.0))
    n_val = int(ck.get("n_val", 0))
    trust = float(np.exp(-val_rwrms)) if val_rwrms > 0 else 1.0
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
        p_basin=1.0,
        meta={"sub_id": int(sub_id), "bin_id": int(bin_id),
              "trust_score": trust, "mu_z": z_mu,
              "cov_z": np.diag(z_scale ** 2), "pi": 1.0},
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


def _setup_loss_linear_np(y_hat, y_obs, frame_w) -> float:
    scale = max(float(np.mean((y_obs * frame_w) ** 2)), 1e-9)
    err = (y_hat - y_obs) * frame_w
    return float(np.mean(err * err) / scale)


def _predict_one(z, member, ctx):
    y, s = predict_full_batched(member, [z], ctx)
    return y[0], s[0]


def _support_penalty(z, member: SubMember) -> float:
    over = np.maximum(z - member.z_hi, 0.0)
    under = np.maximum(member.z_lo - z, 0.0)
    return float(np.sum(((over + under) / member.z_scale) ** 2))


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
                      mode: str = "hard", eps_cm: float = 0.0) -> list[tuple]:
    """Return top-K (bin_id, sub_id, pca_dist) sorted by ascending PCA distance.

    Plan E: instead of top-1, return up to K candidate subs.

    L2-gate modes (selector argparse: --shape-l2-mode):
      hard      — legacy: only subs in the bin that contains y_obs[7]
      eps       — Fix A: also include neighbour bin subs when y_obs[7] is
                  within `eps_cm` of an edge of the primary bin
      cross_bin — C1: ignore L2; per-bin top-1 sub, ranked across bins
      hybrid    — C3: primary bin full subs + every other bin's top-1

    Note: PCA distances are NOT comparable across bins (each bin has its
    own PCA basis & scaler). When mode != hard we still sort by raw PCA
    dist as a coarse heuristic — CMA-ES + Plan E softmax then forms the
    consensus, and the post-CMA-ES Plan-E rerank picks the winning sub by
    GP fit at the converged z (so cross-bin incomparability is mostly
    absorbed downstream).
    """
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
    else:
        raise ValueError(f"unknown --shape-l2-mode={mode}")

    # Collect candidates per bin in that bin's own PCA space
    cands: list[tuple] = []
    for bid in cand_bin_ids:
        b = bin_by_id[bid]
        _, per_sub = _pca_dist_in_bin(y_obs, b)
        per_sub.sort(key=lambda x: x[1])
        if mode == "cross_bin":
            # C1: each bin contributes its top-1
            sid, d = per_sub[0]
            cands.append((bid, sid, d))
        elif mode == "hybrid" and bid != bin_id:
            # C3: non-primary bins contribute top-1
            sid, d = per_sub[0]
            cands.append((bid, sid, d))
        else:
            # hard / eps / hybrid-primary: keep all subs in this bin
            for sid, d in per_sub:
                cands.append((bid, sid, d))

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
    fetch_k = top_k * over_factor if weight_mode == "gp_fit" else top_k
    cands = _route_topk_subs(y_obs, state, fetch_k, mode=l2_mode, eps_cm=eps_cm)
    if not cands:
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

    if weight_mode == "gp_fit":
        # Score every oversampled candidate by GP-fit at its own z_nn and
        # keep the K best.  Then derive softmax weights from the same scores.
        fit_dists = []
        for m in over_members:
            y_pred, _ = _predict_one(m.z_nn, m, ctx)
            fit_dists.append(float(np.linalg.norm(y_pred - y_obs)))
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
    return {
        "setup": setup_name, "W": W, "H": H, "y_obs": y_obs,
        "gid": gid, "state_dir": sd,
        "bin_id": int(members[0].bin_id), "sub_id": int(members[0].sub_id),
        "members": members, "member_weights": ws.tolist(), "ctx": ctx,
        "frame_w": _resolve_frame_w(args),
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
    lo = np.maximum(z_lo_union - pad * Z_SCALE_NP, fb_lo)
    hi = np.minimum(z_hi_union + pad * Z_SCALE_NP, fb_hi)
    # Guard: sigma_y_min floor can push lo[2] above hi[2] for lo-half subs
    # (σy-split). Fall back to global lower bound (no floor) on dim 2 only.
    if lo[2] >= hi[2] - 1e-3:
        lo[2] = max(z_lo_union[2] - pad * Z_SCALE_NP[2], fb_lo_orig[2])
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
    def loss(z):
        l_total = 0.0
        sup_total = 0.0
        for m, w in zip(members, weights):
            y_hat, _sig = _predict_one(z, m, ctx)
            if loss_mode == "log_nuisance":
                lk = _setup_loss_log_nuisance_np(y_hat, item["y_obs"], fw,
                                                  args.sigma_bias, args.sigma_trend)
            else:
                lk = _setup_loss_linear_np(y_hat, item["y_obs"], fw)
            l_total += float(w) * lk
            sup_total += float(w) * _support_penalty(z, m)
        prior = 0.0
        if log_sy_anchor_single is not None:
            prior = sy_prior_w_single * ((z[2] - log_sy_anchor_single) / log_sy_prior_std_single) ** 2
        return l_total + sup_w * sup_total + prior

    t0 = time.time()
    z_best, l_best, n_ev = _run_cma(
        loss, x0, lo, hi,
        sigma0=getattr(args, "shape_sigma0", 0.25),
        max_iter=getattr(args, "shape_max_iter", 30),
        popsize=getattr(args, "shape_popsize", 12),
        seed=7 + primary.sub_id,
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
    if sy_prior_w > 0:
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
    def loss(z):
        # Plan E: weighted consensus across top-K members for each setup
        l_a_total = sup_a_total = 0.0
        for m, w in zip(members_a, ws_a):
            y_a, _ = _predict_one(z, m, ctx_a)
            if loss_mode == "log_nuisance":
                la = _setup_loss_log_nuisance_np(y_a, item_a["y_obs"], fw_a,
                                                   args.sigma_bias, args.sigma_trend)
            else:
                la = _setup_loss_linear_np(y_a, item_a["y_obs"], fw_a)
            l_a_total += float(w) * la
            sup_a_total += float(w) * _support_penalty(z, m)
        l_b_total = sup_b_total = 0.0
        for m, w in zip(members_b, ws_b):
            y_b, _ = _predict_one(z, m, ctx_b)
            if loss_mode == "log_nuisance":
                lb = _setup_loss_log_nuisance_np(y_b, item_b["y_obs"], fw_b,
                                                   args.sigma_bias, args.sigma_trend)
            else:
                lb = _setup_loss_linear_np(y_b, item_b["y_obs"], fw_b)
            l_b_total += float(w) * lb
            sup_b_total += float(w) * _support_penalty(z, m)
        sup = 0.5 * (sup_a_total + sup_b_total)
        prior = 0.0
        if log_sy_anchor is not None:
            prior = ((z[2] - log_sy_anchor) / log_sy_prior_std) ** 2
        return 0.5 * (l_a_total + l_b_total) + sup_w * sup + sy_prior_w * prior

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
                    choices=["hard", "eps", "cross_bin", "hybrid"],
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
    ap.add_argument("--shape-weight-mode", type=str, default="pca",
                    choices=["pca", "gp_fit"],
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
