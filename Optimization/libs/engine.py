from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline_v6_1"))

import surrogate.grid_geo  # noqa: F401
from surrogate import config as HC
from surrogate.data import InputScaler, OutputScaler
from surrogate.experts import ExactExpert
from tests.v7_5_features import W_MAG_DEFAULT, build_phi_v8_route, y_closest_basins

PARAM_BOUNDS = {
    "n": (0.3, 1.0),
    "eta": (0.001, 300.0),
    "sigma_y": (0.001, 400.0),
    "width": (2.0, 7.0),
    "height": (2.0, 7.0),
}


SIM_VS_REAL_CSV = REPO / "tests" / "sim_vs_real_results" / "sim_vs_real.csv"
DEFAULT_STATE_ROOT = PIPELINE_ROOT / "Models" / "v10_bgm_v75_layer3_retrain_realworld_combined_codex"
STATE_ROOT = Path(os.environ.get("V10_STATE_ROOT", DEFAULT_STATE_ROOT))
V7_GEO_PARTITION = REPO / "Models" / "v7_bgm_refit_clean" / "partition_v7_bgm_refit.pkl"
V6_MODEL = REPO / "Models" / "v6_1" / "model.pt"

REAL_SETUPS = {
    "Tonkatsu_4.0_4.5_1": ("Tonkatsu", 4.0, 4.5),
    "Tonkatsu_2.1_6.5_1": ("Tonkatsu", 6.5, 2.1),
    "Chuno_4.5_4.5_1": ("Chuno", 4.5, 4.5),
    "Chuno_7.0_7.0_1": ("Chuno", 7.0, 7.0),
    "Sweet_2.5_2.5_1": ("Sweet", 2.5, 2.5),
    "Sweet_4.5_4.5_1": ("Sweet", 4.5, 4.5),
}

PAIRS = {
    "Tonkatsu": ["Tonkatsu_4.0_4.5_1", "Tonkatsu_2.1_6.5_1"],
    "Chuno": ["Chuno_4.5_4.5_1", "Chuno_7.0_7.0_1"],
    "Sweet": ["Sweet_2.5_2.5_1", "Sweet_4.5_4.5_1"],
}

RHEO_CSV = {
    "Tonkatsu": REPO / "data" / "ref_Tonkatsu_4.0_4.5_1" / "Tonkatsu_20260413_2200.csv",
    "Chuno": REPO / "FlowCurve" / "Rheo_Data" / "Chuno_20260420_1810_2520.csv",
    "Sweet": REPO / "FlowCurve" / "Rheo_Data" / "Sweet_20260420_1800_2520.csv",
}


def get_truth(material: str) -> tuple[float, float, float]:
    from tests.cluster_diagnosis import _fit_hb

    rh = _fit_hb(RHEO_CSV[material])
    n = float(rh["n_rheo"])
    eta = float(rh["eta_rheo"]) * 10.0
    sy = float(rh["sigma_y_rheo"]) * 10.0
    n = max(min(n, PARAM_BOUNDS["n"][1]), PARAM_BOUNDS["n"][0])
    eta = max(min(eta, PARAM_BOUNDS["eta"][1]), PARAM_BOUNDS["eta"][0])
    sy = max(min(sy, PARAM_BOUNDS["sigma_y"][1]), PARAM_BOUNDS["sigma_y"][0])
    return n, eta, sy


def _flow_curve_rmse(theta_hat_int, theta_truth_int, gamma_dot=None):
    if gamma_dot is None:
        gamma_dot = np.logspace(-2, 2, 50)
    eta_t, n_t, sy_t = theta_truth_int[1], theta_truth_int[0], theta_truth_int[2]
    eta_h, n_h, sy_h = theta_hat_int[1], theta_hat_int[0], theta_hat_int[2]
    s_t = (eta_t * 0.1) * np.power(gamma_dot, n_t) + (sy_t * 0.1)
    s_h = (eta_h * 0.1) * np.power(gamma_dot, n_h) + (sy_h * 0.1)
    return float(np.sqrt(np.mean(((s_h - s_t) / s_t) ** 2)))


def load_sub_gp(state_dir: Path, sub_id: int, xs, ys_scaler, device, dtype):
    ckpt = torch.load(
        state_dir / "experts" / f"sub_{sub_id:04d}.pt",
        weights_only=False,
        map_location=device,
    )
    X_phys = ckpt["X_phys"]
    Y_phys = ckpt["Y_phys"]
    X_s = torch.tensor(xs.transform(X_phys), dtype=dtype, device=device)
    Y_s = torch.tensor(ys_scaler.transform(Y_phys), dtype=dtype, device=device)
    exp = ExactExpert(X_s, Y_s, kernel_name=ckpt.get("kernel_name", "matern25_ard")).to(device)
    exp.set_train_data(X_s, Y_s)
    exp.load_state_dict(ckpt["state_dict"])
    exp.eval()
    return exp, X_phys, Y_phys


def z_support_from_X(X_phys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return local expert support in z=(n, log eta, log sigma_y).

    The optimizer is only globally bounded by the HB parameter box. Local
    experts, however, are trained on a much smaller theta region; outside that
    region the GP is extrapolating. These support statistics provide a
    truth-free trust region for each selected sub-expert.
    """
    X = np.asarray(X_phys, dtype=np.float64)
    z_train = np.column_stack([
        X[:, 0],
        np.log(np.clip(X[:, 1], 1e-12, None)),
        np.log(np.clip(X[:, 2], 1e-12, None)),
    ])
    lo = z_train.min(axis=0)
    hi = z_train.max(axis=0)
    std = z_train.std(axis=0, ddof=1) if len(z_train) > 1 else np.ones(3)
    # Keep the penalty finite for very thin clusters while still making a
    # meaningful distinction between interpolation and unsupported extrapolation.
    scale = np.maximum(std, np.array([0.035, 0.12, 0.12], dtype=np.float64))
    return lo, hi, scale


def attach_member_support(member: Member, X_phys: np.ndarray) -> Member:
    lo, hi, scale = z_support_from_X(X_phys)
    member.z_lo = lo
    member.z_hi = hi
    member.z_support_scale = scale
    return member


@dataclass
class Member:
    meta: dict
    exp: object
    p_basin: float

    @property
    def sub_id(self) -> int:
        return int(self.meta["sub_id"])

    @property
    def basin_k(self) -> int:
        return int(self.meta["basin_k"])

    @property
    def pi(self) -> float:
        return float(max(self.meta["pi"], 1e-300))

    @property
    def N(self) -> float:
        return float(max(self.meta.get("N", 1), 1.0))

    @property
    def trust_score(self) -> float:
        return float(np.clip(self.meta.get("trust_score", 1.0), 1e-3, 1.0))

    @property
    def needs_infill(self) -> bool:
        return bool(self.meta.get("needs_infill", False))

    @property
    def mu_z(self) -> np.ndarray:
        return np.asarray(self.meta["mu_z"], dtype=np.float64)


def bounds_log_np() -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([
        PARAM_BOUNDS["n"][0],
        np.log(PARAM_BOUNDS["eta"][0]),
        np.log(PARAM_BOUNDS["sigma_y"][0]),
    ], dtype=np.float64)
    hi = np.array([
        PARAM_BOUNDS["n"][1],
        np.log(PARAM_BOUNDS["eta"][1]),
        np.log(PARAM_BOUNDS["sigma_y"][1]),
    ], dtype=np.float64)
    return lo, hi


def clamp_z(z: np.ndarray) -> np.ndarray:
    lo, hi = bounds_log_np()
    return np.minimum(np.maximum(np.asarray(z, dtype=np.float64), lo), hi)


def to_z(theta: np.ndarray) -> np.ndarray:
    return np.array([theta[0], np.log(max(theta[1], 1e-12)), np.log(max(theta[2], 1e-12))], dtype=np.float64)


def z_to_theta(z: np.ndarray) -> tuple[float, float, float]:
    return float(z[0]), float(np.exp(z[1])), float(np.exp(z[2]))


def u_from_z(z: np.ndarray, lo: torch.Tensor, hi: torch.Tensor, device, dtype) -> torch.Tensor:
    z_t = torch.tensor(clamp_z(z), dtype=dtype, device=device)
    p = torch.clamp((z_t - lo) / (hi - lo), 1e-5, 1.0 - 1e-5)
    return torch.log(p / (1.0 - p))


def load_observation(setup: str, W: float, H: float, obs_source: str) -> np.ndarray:
    if obs_source == "npy":
        return np.load(REPO / "data" / f"ref_{setup}" / "y_obs.npy").astype(np.float64).ravel()
    if obs_source in {"sim_real", "sim_mpm"}:
        df = pd.read_csv(SIM_VS_REAL_CSV)
        ref_dir = f"ref_{setup}"
        sub = df[df["ref_dir"].astype(str).eq(ref_dir)]
        if len(sub) == 0:
            sub = df[(np.abs(df.W.astype(float) - W) < 1e-6) & (np.abs(df.H.astype(float) - H) < 1e-6)]
        if len(sub) == 0:
            raise FileNotFoundError(f"no sim_vs_real row for {setup}")
        row = sub.iloc[0]
        prefix = "y_real" if obs_source == "sim_real" else "y_MPM"
        return np.array([row[f"{prefix}_{i}"] for i in range(1, 9)], dtype=np.float64)
    raise ValueError(f"unknown obs_source={obs_source}")


def load_state(gid: int):
    sd = STATE_ROOT / f"state_gid_{gid}"
    with open(sd / "state.pkl", "rb") as f:
        return pickle.load(f), sd


def route_union(st: dict, y_obs: np.ndarray, W: float, H: float, top_m_bgm: int, top_k_y: int, cap: int):
    if "shape_states" in st and "length_bands" in st:
        from surrogate.v10_1c_length_shape_theta_partition_report import route_length_shape

        args = argparse.Namespace(
            top_bands=max(1, min(int(top_m_bgm), int(st["length_bands"]))),
            top_shape=max(1, int(top_k_y)),
            cap=max(1, int(cap)),
        )
        _K_len, candidates, K_union = route_length_shape(st, y_obs, args)
        n_basin = len(st.get("basin_meta", []))
        probs = np.full(n_basin, 1e-12, dtype=np.float64)
        for bk, _lk, _sk, p_shape in candidates:
            if 0 <= int(bk) < n_basin:
                probs[int(bk)] = max(probs[int(bk)], float(p_shape))
        if np.isfinite(probs).all() and probs.sum() > 0:
            probs = probs / probs.sum()
        return probs, [int(k) for k in K_union], [], [int(k) for k in K_union]

    if str(st.get("version", "")).startswith("v10_") or "bgm_K" in st:
        from surrogate.v10_bgm import route_v10_bgm

        probs, K_bgm, K_y, K_union = route_v10_bgm(
            st, y_obs, W, H, top_m=top_m_bgm, top_k_y=top_k_y, cap=cap
        )
        selected = set(int(k) for k in st.get("selected_v10_basins", []))
        if selected:
            K_union = [int(k) for k in K_union if int(k) in selected]
        return probs, K_bgm, K_y, K_union

    df_one = pd.DataFrame({
        "n": [0.0], "eta": [0.0], "sigma_y": [0.0],
        "width": [W], "height": [H],
        **{f"x_{i + 1:02d}": [float(y_obs[i])] for i in range(8)},
    })
    phi = build_phi_v8_route(df_one)
    phi_s = st["scaler"].transform(phi)
    phi_s[:, 3] *= st.get("w_mag", W_MAG_DEFAULT)
    probs = st["bgm_real"].predict_proba(phi_s)[0]
    K_bgm = np.argsort(-probs)[:top_m_bgm].tolist()
    K_y = y_closest_basins(
        y_obs, st["basin_y_mean"], st["basin_y_std"],
        active_basins=st["active_basins"], top_k=top_k_y,
    ).tolist()
    return probs, K_bgm, K_y, list(dict.fromkeys(K_bgm + K_y))[:cap]


def build_members(st: dict, sd: Path, K_union: list[int], probs: np.ndarray,
                  xs: InputScaler, ys: OutputScaler, device, dtype, max_members: int,
                  basin_quota: int = 0) -> list[Member]:
    metas = [
        m for m in st["sub_meta"]
        if int(m["basin_k"]) in set(K_union) and not m.get("fallback_only", False)
    ]
    metas.sort(key=lambda m: -(float(probs[int(m["basin_k"])]) * float(m["pi"]) * np.sqrt(max(float(m["N"]), 1.0))))
    if basin_quota > 0:
        chosen = []
        chosen_ids = set()
        for basin in K_union:
            taken = 0
            for meta in metas:
                if int(meta["basin_k"]) != int(basin) or int(meta["sub_id"]) in chosen_ids:
                    continue
                chosen.append(meta)
                chosen_ids.add(int(meta["sub_id"]))
                taken += 1
                if len(chosen) >= max_members or taken >= basin_quota:
                    break
            if len(chosen) >= max_members:
                break
        for meta in metas:
            if len(chosen) >= max_members:
                break
            if int(meta["sub_id"]) not in chosen_ids:
                chosen.append(meta)
                chosen_ids.add(int(meta["sub_id"]))
        metas = chosen
    out = []
    for meta in metas[:max_members]:
        exp, X_phys, _ = load_sub_gp(sd, int(meta["sub_id"]), xs, ys, device, dtype)
        member = Member(meta=meta, exp=exp, p_basin=float(probs[int(meta["basin_k"])]))
        out.append(attach_member_support(member, X_phys))
    return out


def member_nearest_z(sd: Path, member: Member, y_obs: np.ndarray) -> np.ndarray:
    ckpt = torch.load(sd / "experts" / f"sub_{member.sub_id:04d}.pt", weights_only=False, map_location="cpu")
    X = np.asarray(ckpt["X_phys"], dtype=np.float64)
    Y = np.asarray(ckpt["Y_phys"], dtype=np.float64)
    if len(Y) == 0:
        return clamp_z(member.mu_z)
    w = np.array([0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0], dtype=np.float64)
    d = np.sqrt(np.mean(((Y - y_obs[None, :]) * w[None, :]) ** 2, axis=1))
    return clamp_z(to_z(X[int(np.argmin(d)), :3]))


def make_context(xs: InputScaler, ys: OutputScaler, W: float, H: float, y_obs: np.ndarray, device, dtype):
    x_mean = torch.tensor(xs.mean, dtype=dtype, device=device)
    x_std = torch.tensor(xs.std, dtype=dtype, device=device)
    y_mean = torch.tensor(ys.mean, dtype=dtype, device=device)
    y_target = torch.tensor(y_obs, dtype=dtype, device=device)
    W_t = torch.tensor(float(W), dtype=dtype, device=device)
    H_t = torch.tensor(float(H), dtype=dtype, device=device)
    lo_np, hi_np = bounds_log_np()
    lo = torch.tensor(lo_np, dtype=dtype, device=device)
    hi = torch.tensor(hi_np, dtype=dtype, device=device)
    frame_w = torch.tensor([0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0], dtype=dtype, device=device)
    return x_mean, x_std, y_mean, y_target, W_t, H_t, lo, hi, frame_w


def predict_one(member: Member, z: torch.Tensor, ctx):
    x_mean, x_std, y_mean, _y_target, W_t, H_t, *_ = ctx
    x_log = torch.stack([z[0], z[1], z[2], W_t, H_t], dim=0).view(1, 5)
    X_s = (x_log - x_mean) / x_std
    mu_s, _var_s = member.exp.predict(X_s)
    return (mu_s + y_mean).view(-1)


def setup_loss(y_hat: torch.Tensor, y_target: torch.Tensor, frame_w: torch.Tensor,
               loss_mode: str, sigma_bias: float, sigma_trend: float) -> torch.Tensor:
    if loss_mode == "linear":
        scale = torch.clamp(torch.mean((y_target * frame_w) ** 2), min=1e-9)
        err = (y_hat - y_target) * frame_w
        return torch.mean(err * err) / scale
    if loss_mode != "log_nuisance":
        raise ValueError(f"unknown loss_mode={loss_mode}")
    eps = torch.as_tensor(1e-9, dtype=y_hat.dtype, device=y_hat.device)
    r = torch.log(torch.clamp(y_hat, min=eps)) - torch.log(torch.clamp(y_target, min=eps))
    t = torch.linspace(-1.0, 1.0, y_hat.numel(), dtype=y_hat.dtype, device=y_hat.device)
    X = torch.stack([torch.ones_like(t), t], dim=1)
    w = frame_w * frame_w
    WX = X * w[:, None]
    prior = torch.diag(torch.stack([
        1.0 / torch.as_tensor(sigma_bias ** 2, dtype=y_hat.dtype, device=y_hat.device),
        1.0 / torch.as_tensor(sigma_trend ** 2, dtype=y_hat.dtype, device=y_hat.device),
    ]))
    A = X.T @ WX + prior
    b = X.T @ (w * r)
    gamma = torch.linalg.solve(A, b)
    resid = r - X @ gamma
    return torch.mean(w * resid * resid) + ((gamma[0] / sigma_bias) ** 2 + (gamma[1] / sigma_trend) ** 2) / y_hat.numel()


def theta_prior_penalty(z: torch.Tensor, members: list[Member]) -> torch.Tensor:
    """Generic prior: shared theta should stay near at least one plausible sub mode.

    This is not calibrated on real materials. It is a weak, dimensionless
    regularizer in z=(n, log eta, log sigma_y) space that discourages selecting a
    low observation-loss solution that lies far from the candidate experts'
    trained theta regions.
    """
    if not members:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    scales = torch.tensor([0.20, 0.90, 1.20], dtype=z.dtype, device=z.device)
    vals = []
    mus = []
    for member in members:
        mu = torch.tensor(member.mu_z, dtype=z.dtype, device=z.device)
        mus.append(mu)
        vals.append(torch.mean(((z - mu) / scales) ** 2))
    return torch.mean(torch.stack(vals))


def theta_pair_spread_penalty(members: list[Member], device, dtype) -> torch.Tensor:
    """Generic prior: two setup candidates for one material should not disagree wildly.

    The two observations share the same theta. If candidate sub priors are very
    far apart in theta space, the pair is a less plausible explanation unless
    its y-likelihood is clearly better.
    """
    if len(members) < 2:
        return torch.zeros((), dtype=dtype, device=device)
    scales = torch.tensor([0.20, 0.90, 1.20], dtype=dtype, device=device)
    mus = [torch.tensor(m.mu_z, dtype=dtype, device=device) for m in members]
    vals = []
    for i in range(len(mus)):
        for j in range(i + 1, len(mus)):
            vals.append(torch.mean(((mus[i] - mus[j]) / scales) ** 2))
    return torch.mean(torch.stack(vals))


def theta_support_penalty(z: torch.Tensor, members: list[Member]) -> torch.Tensor:
    """Penalize leaving each selected sub-expert's trained theta support.

    This is a local-expert consistency term, not a real-material calibration.
    Inside a sub's observed training box it is exactly zero. Outside, it grows
    quadratically in z-space using that sub's empirical scale, so a selected
    local GP cannot silently explain observations from an unsupported region.
    """
    if not members:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    vals = []
    for member in members:
        z_lo = getattr(member, "z_lo", None)
        z_hi = getattr(member, "z_hi", None)
        z_scale = getattr(member, "z_support_scale", None)
        if z_lo is None or z_hi is None or z_scale is None:
            continue
        lo = torch.tensor(z_lo, dtype=z.dtype, device=z.device)
        hi = torch.tensor(z_hi, dtype=z.dtype, device=z.device)
        scale = torch.tensor(z_scale, dtype=z.dtype, device=z.device)
        outside = torch.relu(lo - z) + torch.relu(z - hi)
        vals.append(torch.mean((outside / scale) ** 2))
    if not vals:
        return torch.zeros((), dtype=z.dtype, device=z.device)
    return torch.mean(torch.stack(vals))


def member_trust_penalty(members: list[Member], device, dtype) -> torch.Tensor:
    """Truth-free reliability penalty from each expert's validation report.

    The value is constant with respect to theta, so it does not distort the GP
    local optimum inside one expert. It only changes which candidate expert is
    trusted when several candidates can explain the same y observation.
    """
    if not members:
        return torch.zeros((), dtype=dtype, device=device)
    vals = []
    for member in members:
        trust = max(float(getattr(member, "trust_score", 1.0)), 1e-3)
        penalty = -np.log(trust)
        if getattr(member, "needs_infill", False):
            penalty += 1.0
        vals.append(torch.as_tensor(penalty, dtype=dtype, device=device))
    return torch.mean(torch.stack(vals))


def optimize_rows(rows: list[dict], contexts: list[tuple], steps: int, lr: float,
                  sigma_y_min: float, loss_mode: str, sigma_bias: float, sigma_trend: float,
                  device, dtype, selector_prior_weight: float = 0.0,
                  selector_pair_weight: float = 0.0,
                  support_penalty_weight: float = 0.0,
                  trust_penalty_weight: float = 0.0):
    lo = contexts[0][6]
    hi = contexts[0][7]
    if sigma_y_min > 0:
        lo = lo.clone()
        lo[2] = max(float(lo[2].detach().cpu()), float(np.log(sigma_y_min)))
    u = torch.nn.Parameter(torch.stack([u_from_z(r["start"], lo, hi, device, dtype) for r in rows]))
    opt = torch.optim.Adam([u], lr=lr)
    best = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        z_all = lo + (hi - lo) * torch.sigmoid(u)
        losses = []
        for i, row in enumerate(rows):
            total = None
            for member, ctx in zip(row["members"], contexts):
                y_hat = predict_one(member, z_all[i], ctx)
                y_target = ctx[3]
                frame_w = ctx[8]
                val = setup_loss(y_hat, y_target, frame_w, loss_mode, sigma_bias, sigma_trend)
                total = val if total is None else total + val
            obs_loss = total / len(contexts)
            prior = theta_prior_penalty(z_all[i], row["members"])
            spread = theta_pair_spread_penalty(row["members"], device, dtype)
            support = theta_support_penalty(z_all[i], row["members"])
            trust = member_trust_penalty(row["members"], device, dtype)
            score = (obs_loss
                     + selector_prior_weight * prior
                     + selector_pair_weight * spread
                     + support_penalty_weight * support
                     + trust_penalty_weight * trust)
            losses.append(score)
        loss_vec = torch.stack(losses)
        torch.mean(loss_vec).backward()
        opt.step()
        with torch.no_grad():
            vals = loss_vec.detach().cpu().numpy()
            zs = (lo + (hi - lo) * torch.sigmoid(u)).detach().cpu().numpy()
            for i, val in enumerate(vals):
                if best is None or float(val) < best[0]:
                    best = float(val), zs[i].copy(), rows[i]
    return best


def load_runtime():
    with open(V7_GEO_PARTITION, "rb") as f:
        geo_router = pickle.load(f)["geo_router"]
    blob = torch.load(V6_MODEL, weights_only=False, map_location="cpu")
    xs = InputScaler.from_dict(blob["xs"])
    ys = OutputScaler.from_dict(blob["ys"])
    return geo_router, xs, ys


def prepare_setup(setup: str, obs_source: str, geo_router, xs, ys, device, dtype,
                  max_members: int, top_m_bgm: int, top_k_y: int, cap: int,
                  basin_quota: int = 0):
    family, W, H = REAL_SETUPS[setup]
    y_obs = load_observation(setup, W, H, obs_source)
    gid = int(geo_router.predict([[W, H]])[0])
    st, sd = load_state(gid)
    if "counts" not in st:
        assign_path = sd / "sub_assignments.csv"
        if assign_path.is_file():
            assign = pd.read_csv(assign_path, usecols=["basin_k"])
            K = int(st.get("bgm_K", len(st.get("active_basins", []))))
            st["counts"] = [int((assign["basin_k"] == k).sum()) for k in range(K)]
        else:
            K = int(st.get("bgm_K", len(st.get("active_basins", []))))
            st["counts"] = [100] * K
    if "y_mean" not in st and "basin_y_mean" in st:
        st["y_mean"] = st["basin_y_mean"]
    if "y_std" not in st and "basin_y_std" in st:
        st["y_std"] = st["basin_y_std"]
    probs, K_bgm, K_y, K_union = route_union(st, y_obs, W, H, top_m_bgm, top_k_y, cap)
    members = build_members(st, sd, K_union, probs, xs, ys, device, dtype, max_members, basin_quota)
    ctx = make_context(xs, ys, W, H, y_obs, device, dtype)
    return {
        "setup": setup, "family": family, "W": W, "H": H, "y_obs": y_obs,
        "gid": gid, "state_dir": sd, "K_BGM": K_bgm, "K_y": K_y, "K_union": K_union,
        "members": members, "ctx": ctx,
    }


def run_single(obs_source: str, args, geo_router, xs, ys, device, dtype) -> list[dict]:
    out = []
    wanted = set(PAIRS) if args.families.lower() == "all" else {
        x.strip() for x in args.families.split(",") if x.strip()
    }
    sel = getattr(args, "selector_mode", "mixed")
    use_abd = sel == "abd"   # single-setup uses ABD only when explicitly --selector-mode=abd
    if use_abd:
        from Optimization import selector_abd as ABD
    for setup, (family, W, H) in REAL_SETUPS.items():
        if family not in wanted:
            continue
        if use_abd:
            y_obs = load_observation(setup, W, H, obs_source)
            item = ABD.prepare_setup_abd(setup, W, H, y_obs, geo_router, xs, ys, device, dtype, args)
            res = ABD.inverse_single_abd(item, args, device, dtype)
            if res is None:
                continue
            theta = (res["theta_n"], res["theta_eta"], res["theta_sy"])
            truth = get_truth(family)
            fc = float(_flow_curve_rmse(list(theta), truth)) * 100.0
            out.append({
                "obs_source": obs_source, "mode": "single", "family": family, "setup": setup,
                "theta_n": theta[0], "theta_eta": theta[1], "theta_sy": theta[2],
                "fc_pct": fc, "objective": res["objective"], "opt_s": res["opt_s"],
                "gid": item["gid"], "sub_ids": res["sub_ids"], "basins": res["basins"],
                "start_kind": res["start_kind"],
                "K_BGM": item["K_BGM"], "K_y": item["K_y"], "K_union": item["K_union"],
                "raw_wrms_total": res["raw_wrms_total"], "flow_total": res.get("flow_total", 0.0),
                "sigma_total": res["sigma_total"],
                "support_total": res.get("support_total", 0.0),
                "trust_total": res.get("trust_total", 0.0),
                "n_rows": res["n_rows"], "selector_mode": "abd",
            })
        else:
            item = prepare_setup(setup, obs_source, geo_router, xs, ys, device, dtype,
                                 args.single_members, args.top_m_bgm, args.top_k_y, args.cap,
                                 args.member_basin_quota)
            rows = []
            for m in item["members"]:
                rows.append({"members": [m], "start": m.mu_z, "start_kind": "mu", "sub_ids": [m.sub_id], "basins": [m.basin_k]})
                rows.append({"members": [m], "start": member_nearest_z(item["state_dir"], m, item["y_obs"]), "start_kind": "nn", "sub_ids": [m.sub_id], "basins": [m.basin_k]})
            t0 = time.time()
            obj, z, row = optimize_rows(rows, [item["ctx"]], args.single_steps, args.single_lr,
                                        args.sigma_y_min, args.loss_mode, args.sigma_bias, args.sigma_trend,
                                        device, dtype, args.selector_prior_weight, args.selector_pair_weight,
                                        args.support_penalty_weight, args.trust_penalty_weight)
            dt = time.time() - t0
            theta = z_to_theta(z)
            truth = get_truth(family)
            fc = float(_flow_curve_rmse(list(theta), truth)) * 100.0
            out.append({
                "obs_source": obs_source, "mode": "single", "family": family, "setup": setup,
                "theta_n": theta[0], "theta_eta": theta[1], "theta_sy": theta[2],
                "fc_pct": fc, "objective": obj, "opt_s": dt,
                "gid": item["gid"], "sub_ids": row["sub_ids"], "basins": row["basins"],
                "start_kind": row["start_kind"], "K_BGM": item["K_BGM"], "K_y": item["K_y"], "K_union": item["K_union"],
                "member_basin_quota": args.member_basin_quota,
                "selector_mode": "legacy",
            })
    return out


def run_double(obs_source: str, args, geo_router, xs, ys, device, dtype) -> list[dict]:
    out = []
    wanted = set(PAIRS) if args.families.lower() == "all" else {
        x.strip() for x in args.families.split(",") if x.strip()
    }
    sel = getattr(args, "selector_mode", "mixed")
    use_abd = sel in {"abd", "mixed"}
    if use_abd:
        from Optimization import selector_abd as ABD
    for family, setups in PAIRS.items():
        if family not in wanted:
            continue
        if use_abd:
            y8s = []
            items = []
            for s in setups:
                fam, W, H = REAL_SETUPS[s]
                y_obs = load_observation(s, W, H, obs_source)
                y8s.append(float(y_obs[-1]))
                items.append(ABD.prepare_setup_abd(s, W, H, y_obs, geo_router, xs, ys, device, dtype, args))
            res = ABD.inverse_double_abd(items[0], items[1], args, device, dtype)
            if res is None:
                continue
            theta = (res["theta_n"], res["theta_eta"], res["theta_sy"])
            truth = get_truth(family)
            fc = float(_flow_curve_rmse(list(theta), truth)) * 100.0
            out.append({
                "obs_source": obs_source, "mode": "double", "family": family, "setups": ",".join(setups),
                "theta_n": theta[0], "theta_eta": theta[1], "theta_sy": theta[2],
                "fc_pct": fc, "objective": res["objective"], "opt_s": res["opt_s"],
                "sub_ids": res["sub_ids"], "basins": res["basins"], "start_kind": res["start_kind"],
                "n_rows": res["n_rows"], "n_pairs": res["n_pairs"],
                "raw_wrms_total": res["raw_wrms_total"], "flow_total": res.get("flow_total", 0.0),
                "sigma_total": res["sigma_total"],
                "support_total": res.get("support_total", 0.0),
                "trust_total": res.get("trust_total", 0.0),
                "pair_spread": res["pair_spread"],
                "y8_min": min(y8s), "y8_max": max(y8s),
                "selector_mode": "abd",
            })
            continue
        y8s = []
        for s in setups:
            fam, W, H = REAL_SETUPS[s]
            y8s.append(float(load_observation(s, W, H, obs_source)[-1]))
        basin_quota = args.member_basin_quota
        sigma_bias = args.sigma_bias
        sigma_trend = args.sigma_trend
        if args.auto_y8_regime and (max(y8s) < 2.0 or min(y8s) > 8.0):
            max_members, steps, lr = 3, 18, 0.08
            if min(y8s) > 8.0:
                basin_quota = args.auto_y8_member_basin_quota
                sigma_bias = args.auto_high_y8_sigma_bias
                sigma_trend = args.auto_high_y8_sigma_trend
        else:
            max_members, steps, lr = args.double_members, args.double_steps, args.double_lr
        items = [
            prepare_setup(s, obs_source, geo_router, xs, ys, device, dtype,
                          max_members, args.top_m_bgm, args.top_k_y, args.cap,
                          basin_quota)
            for s in setups
        ]
        rows = []
        for ia, a in enumerate(items[0]["members"]):
            for ib, b in enumerate(items[1]["members"]):
                rows.append({"members": [a, b], "start": 0.5 * (a.mu_z + b.mu_z), "start_kind": "mu", "sub_ids": [a.sub_id, b.sub_id], "basins": [a.basin_k, b.basin_k]})
                za = member_nearest_z(items[0]["state_dir"], a, items[0]["y_obs"])
                zb = member_nearest_z(items[1]["state_dir"], b, items[1]["y_obs"])
                rows.append({"members": [a, b], "start": 0.5 * (za + zb), "start_kind": "nn", "sub_ids": [a.sub_id, b.sub_id], "basins": [a.basin_k, b.basin_k]})
        t0 = time.time()
        obj, z, row = optimize_rows(rows, [items[0]["ctx"], items[1]["ctx"]], steps, lr,
                                    args.sigma_y_min, args.loss_mode, sigma_bias, sigma_trend,
                                    device, dtype, args.selector_prior_weight, args.selector_pair_weight,
                                    args.support_penalty_weight, args.trust_penalty_weight)
        dt = time.time() - t0
        theta = z_to_theta(z)
        truth = get_truth(family)
        fc = float(_flow_curve_rmse(list(theta), truth)) * 100.0
        out.append({
            "obs_source": obs_source, "mode": "double", "family": family, "setups": ",".join(setups),
            "theta_n": theta[0], "theta_eta": theta[1], "theta_sy": theta[2],
            "fc_pct": fc, "objective": obj, "opt_s": dt,
            "sub_ids": row["sub_ids"], "basins": row["basins"], "start_kind": row["start_kind"],
            "steps": steps, "lr": lr, "max_members": max_members,
            "member_basin_quota": basin_quota,
            "sigma_bias": sigma_bias, "sigma_trend": sigma_trend,
            "y8_min": min(y8s), "y8_max": max(y8s),
        })
    return out


def plot_flowcurves(real_rows: list[dict], sim_rows: list[dict], out_dir: Path):
    sys.path.insert(0, str(REPO / "FlowCurve"))
    from flowcurve import _read_rheo, calcFlowCurve, color_list
    from param import Param
    fig_dir = out_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sim_by_family = {r["family"]: r for r in sim_rows if r["mode"] == "double"}
    real_by_family = {r["family"]: r for r in real_rows if r["mode"] == "double"}
    for family, real_r in real_by_family.items():
        sim_r = sim_by_family.get(family)
        if sim_r is None:
            continue
        df = _read_rheo(RHEO_CSV[family])
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** 0, 10 ** 2)
        ax.set_ylim(1e0, 1e3 if family == "Sweet" else 1e2)
        ax.plot(df["[1/s]"], df["[Pa]"], linestyle="dotted", linewidth=3.0, color="black", label="rheometer")
        x = np.linspace(df["[1/s]"][5], df["[1/s]"][18], 10000)
        for i, (label, r) in enumerate([("V10 real obs", real_r), ("V10 sim obs", sim_r)]):
            p = Param(r["theta_eta"], r["theta_n"], r["theta_sy"])
            ax.plot(x, calcFlowCurve(p, x), color=color_list[i % len(color_list)], linewidth=3.0, label=f"{label} fc={r['fc_pct']:.1f}%")
        ax.set_xlabel(r"$\dot{\gamma}[s^{-1}]$")
        ax.set_ylabel(r"$\sigma_s[Pa]$")
        ax.grid()
        ax.legend(fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{family}_double_real_vs_sim_flowcurve.png", dpi=180)
        plt.close(fig)


def plot_standard_single_double(rows_single: list[dict], rows_double: list[dict], obs_source: str, out_dir: Path):
    """One standard rheometer plot per material and observation source.

    Each figure overlays:
      dotted black: rheometer raw flow curve
      red/blue: the two single-setup inverse estimates
      dark red: the double-setup shared-theta estimate
    """
    sys.path.insert(0, str(REPO / "FlowCurve"))
    from flowcurve import _read_rheo, calcFlowCurve
    from param import Param

    fig_dir = out_dir / "figs" / obs_source
    fig_dir.mkdir(parents=True, exist_ok=True)
    singles_by_family: dict[str, list[dict]] = {}
    for r in rows_single:
        singles_by_family.setdefault(r["family"], []).append(r)
    doubles_by_family = {r["family"]: r for r in rows_double}

    for family, single_rows in singles_by_family.items():
        double_row = doubles_by_family.get(family)
        if double_row is None:
            continue
        df = _read_rheo(RHEO_CSV[family])
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(10 ** 0, 10 ** 2)
        ax.set_ylim(1e0, 1e3 if family == "Sweet" else 1e2)
        ax.plot(df["[1/s]"], df["[Pa]"], linestyle="dotted", linewidth=3.0, color="black", label="rheometer")
        x = np.linspace(df["[1/s]"][5], df["[1/s]"][18], 10000)

        colors = ["#d62728", "#1f77b4"]
        for i, r in enumerate(sorted(single_rows, key=lambda q: q["setup"])):
            p = Param(r["theta_eta"], r["theta_n"], r["theta_sy"])
            label = f"single {i+1}: {r['fc_pct']:.1f}%"
            ax.plot(x, calcFlowCurve(p, x), color=colors[i % len(colors)], linewidth=2.2, alpha=0.85, label=label)

        p2 = Param(double_row["theta_eta"], double_row["theta_n"], double_row["theta_sy"])
        ax.plot(x, calcFlowCurve(p2, x), color="#8b0000", linewidth=3.2, label=f"double: {double_row['fc_pct']:.1f}%")

        ax.set_title(f"{family} V10 {obs_source}")
        ax.set_xlabel(r"$\dot{\gamma}[s^{-1}]$")
        ax.set_ylabel(r"$\sigma_s[Pa]$")
        ax.grid()
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{family}_{obs_source}_single_double_flowcurve.png", dpi=180)
        plt.close(fig)


def main():
    global STATE_ROOT

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "double", "all"], default="all")
    ap.add_argument("--obs-source", choices=["npy", "sim_real", "sim_mpm"], default="sim_real")
    ap.add_argument("--also-sim", action="store_true", help="Also run sim_mpm and plot real-vs-sim double flowcurves.")
    ap.add_argument("--out-dir", type=Path, default=REPO / "Models" / "v10_codex_pipeline")
    ap.add_argument("--state-root", type=Path, default=STATE_ROOT,
                    help="trained V10 state root containing state_gid_*/state.pkl")
    ap.add_argument("--families", default="all",
                    help="comma-separated material families or 'all' (default: all)")
    ap.add_argument("--loss-mode", choices=["linear", "log_nuisance"], default="log_nuisance")
    ap.add_argument("--sigma-bias", type=float, default=0.25)
    ap.add_argument("--sigma-trend", type=float, default=0.20)
    ap.add_argument("--sigma-y-min", type=float, default=5.0)
    ap.add_argument("--top-m-bgm", type=int, default=4)
    ap.add_argument("--top-k-y", type=int, default=3)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--member-basin-quota", type=int, default=0,
                    help="Ensure candidate members cover distinct routed basins before filling by score.")
    ap.add_argument("--auto-y8-member-basin-quota", type=int, default=1,
                    help="Basin quota used only for double inverse in high-y8 regimes.")
    ap.add_argument("--auto-high-y8-sigma-bias", type=float, default=1.0,
                    help="Setup-level log-bias nuisance prior for high-y8 double inverse.")
    ap.add_argument("--auto-high-y8-sigma-trend", type=float, default=0.8,
                    help="Setup-level log-trend nuisance prior for high-y8 double inverse.")
    ap.add_argument("--selector-prior-weight", type=float, default=0.0,
                    help="Weak z-space distance-to-candidate-prior penalty added to the optimization score.")
    ap.add_argument("--selector-pair-weight", type=float, default=0.0,
                    help="Weak z-space candidate-prior disagreement penalty for multi-setup pairs.")
    ap.add_argument("--support-penalty-weight", type=float, default=0.0,
                    help="Local sub-expert support penalty during optimization; zero disables.")
    ap.add_argument("--trust-penalty-weight", type=float, default=0.15,
                    help="Validation trust penalty during candidate selection; set 0 to recover old behavior.")
    ap.add_argument("--single-members", type=int, default=4)
    ap.add_argument("--single-steps", type=int, default=12)
    ap.add_argument("--single-lr", type=float, default=0.08)
    ap.add_argument("--double-members", type=int, default=4)
    ap.add_argument("--double-steps", type=int, default=16)
    ap.add_argument("--double-lr", type=float, default=0.085)
    ap.add_argument("--auto-y8-regime", action="store_true", default=True)
    ap.add_argument("--no-auto-y8-regime", dest="auto_y8_regime", action="store_false")
    from Optimization import selector_abd as _ABD_args
    _ABD_args.add_argparse_args(ap)
    args = ap.parse_args()

    STATE_ROOT = args.state_root.resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    geo_router, xs, ys = load_runtime()
    device, dtype = HC.DEVICE, HC.DTYPE

    all_outputs = {}
    obs_sources = [args.obs_source]
    if args.also_sim and "sim_mpm" not in obs_sources:
        obs_sources.append("sim_mpm")

    for obs in obs_sources:
        rows_single = run_single(obs, args, geo_router, xs, ys, device, dtype) if args.mode in {"single", "all"} else []
        rows_double = run_double(obs, args, geo_router, xs, ys, device, dtype) if args.mode in {"double", "all"} else []
        if rows_single:
            pd.DataFrame(rows_single).to_csv(args.out_dir / f"single_{obs}.csv", index=False)
        if rows_double:
            pd.DataFrame(rows_double).to_csv(args.out_dir / f"double_{obs}.csv", index=False)
        if rows_single and rows_double:
            plot_standard_single_double(rows_single, rows_double, obs, args.out_dir)
        all_outputs[obs] = {"single": rows_single, "double": rows_double}

    if "sim_real" in all_outputs and "sim_mpm" in all_outputs:
        plot_flowcurves(all_outputs["sim_real"]["double"], all_outputs["sim_mpm"]["double"], args.out_dir)

    summary = {}
    for obs, parts in all_outputs.items():
        summary[obs] = {}
        for mode, rows in parts.items():
            if rows:
                summary[obs][mode] = {
                    "mean_fc_pct": float(np.mean([r["fc_pct"] for r in rows])),
                    "mean_opt_s": float(np.mean([r["opt_s"] for r in rows])),
                    "rows": rows,
                }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({o: {m: {k: v for k, v in d.items() if k != "rows"} for m, d in s.items()} for o, s in summary.items()}, indent=2))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
