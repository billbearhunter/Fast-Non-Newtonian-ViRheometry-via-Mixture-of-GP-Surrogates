"""
Synthetic Validation at Scale (paper §7.3) ─ Hamamichi-style 1→2 setup ablation.

Replicates the predecessor's §7.1 protocol on an N-case panel sampled from
`data/canonical/test.csv`, with 1 setup and joint 2-setup recovery scored by
the Hamamichi 2023 relative-error formula:

    E_rel(θ̂, θ*) = sqrt(
        ((n̂ − n*)/(n_max − n_min))² +
        ((η̂ − η*)/(η_max − η_min))² +
        ((σ̂_Y − σ_Y*)/(σ_Y_max − σ_Y_min))²
    )

Recovery is declared successful when E_rel ≤ 0.1 (Hamamichi's threshold).

Pipeline per case
-----------------
  1. setup-1 (W₁, H₁, y_obs⁽¹⁾) read directly from a row of canonical/test.csv;
     no MPM call needed.
  2. Single-setup inverse → θ̂₁; record E_rel(θ̂₁).
  3. Hessian-orthogonal proposer(θ̂₁, [setup-1]) → (W₂*, H₂*).
  4. Headless MLS-MPM forward at the *true* θ on (W₂*, H₂*) → y_obs⁽²⁾.
  5. Joint 2-setup inverse on (item₁, item₂) → θ̂₂; record E_rel(θ̂₂).

3-setup extension
-----------------
The current production codebase only exposes inverse_joint_setup(item_a,
item_b). A 3-setup variant requires a small extension to selector.py
(union of three sub bbox + summed GP-aware NLL); this script reports 1- and
2-setup numbers and leaves the 3-setup row TODO.

Usage
-----
    python scripts/run_synthetic_panel.py \\
        --test-csv data/canonical/test.csv \\
        --out-dir  data/eval/synthetic_panel/ \\
        --n-cases  300 \\
        --state-root Models/yshape_mogp_production \\
        --seed     42
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

from Optimization.libs import engine, selector  # noqa: E402
from Optimization.libs.mechanism import Param, Setup, propose_next_setup  # noqa: E402


# Hamamichi 2023 normalisation box (matches our training corpus, §6 paper)
N_MIN, N_MAX = 0.3, 1.0
ETA_MIN, ETA_MAX = 1e-3, 3.0e2
SIGMA_Y_MIN, SIGMA_Y_MAX = 1e-3, 4.0e2

# Headless MPM container parameters (match training set defaults)
RHO_DEFAULT = 1.2
DENSITY_DEFAULT = 1.2


def _make_inverse_args(routing_profile: str, restarts: int = 5, max_iter: int = 30) -> SimpleNamespace:
    """Inverse defaults for the synthetic panel.

    The synthetic panel is simulator-matched, so its default routing profile
    is sim_precision.  Real-video drivers should use real_robust.
    """
    return SimpleNamespace(
        # CMA-ES
        shape_sigma0=0.25,
        shape_max_iter=int(max_iter),
        shape_popsize=12,
        shape_cma_seed_offset=0,
        shape_cma_restarts=int(restarts),
        shape_cma_trail_csv=None,
        # bbox
        shape_box_pad=0.10,
        shape_support_weight=0.25,
        # routing
        shape_top_k=1,
        shape_topk_softmax_tau=1.5,
        shape_l2_mode="hard",
        shape_l2_eps_cm=0.5,
        shape_eps_cap_by_primary=False,
        shape_routing_profile=routing_profile,
        shape_weight_mode=None,
        shape_z_anchor=None,
        shape_gpfit_oversample=3,
        # loss
        shape_gp_noise_log_floor=0.20,
        shape_disable_calib=True,
        shape_frame_w=None,
        shape_per_sub_z_scale=False,
        shape_warm_start_z=None,
        shape_joint_pair_mode="mixture",
        shape_joint_max_pairs=0,
        # legacy / required by inverse_*_setup
        sigma_bias=0.25,
        sigma_trend=0.20,
        sigma_y_min=5.0,
        # CI reporting (off for speed; can re-enable per case)
        shape_report_ci=False,
    )


def _erel(theta_hat, theta_true) -> float:
    """Hamamichi 2023 normalised relative error."""
    n_h, eta_h, sy_h = theta_hat
    n_t, eta_t, sy_t = theta_true
    return float(np.sqrt(
        ((n_h - n_t) / (N_MAX - N_MIN)) ** 2
        + ((eta_h - eta_t) / (ETA_MAX - ETA_MIN)) ** 2
        + ((sy_h - sy_t) / (SIGMA_Y_MAX - SIGMA_Y_MIN)) ** 2
    ))


def _per_param_rel(theta_hat, theta_true) -> tuple[float, float, float]:
    n_h, eta_h, sy_h = theta_hat
    n_t, eta_t, sy_t = theta_true
    return (
        abs(n_h - n_t) / max(abs(n_t), 1e-9),
        abs(eta_h - eta_t) / max(abs(eta_t), 1e-9),
        abs(sy_h - sy_t) / max(abs(sy_t), 1e-9),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--test-csv", type=Path,
                    default=REPO / "data" / "canonical" / "test.csv")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "data" / "eval" / "synthetic_panel")
    ap.add_argument("--n-cases", type=int, default=300)
    ap.add_argument("--state-root", type=Path, default=engine.STATE_ROOT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--routing-profile", choices=["sim_precision", "real_robust", "joint_consistency"],
                    default="sim_precision",
                    help="Observation-evidence routing profile for setup-1 (and setup-2 by "
                         "default).")
    ap.add_argument("--routing-profile-setup2",
                    choices=["sim_precision", "real_robust", "joint_consistency"],
                    default=None,
                    help="Optional override for setup-2 routing only. When set to "
                         "'joint_consistency', setup-2 reranks the PCA-shortlist by closest "
                         "z-distance to the setup-1 posterior θ̂_1, keeping the joint inverse "
                         "from coupling two parameter-incompatible sub experts. Default "
                         "(None) inherits --routing-profile.")
    ap.add_argument("--setup2-top-k", type=int, default=1,
                    help="Setup-2 candidate experts to keep. Values >1 are meant "
                         "for --joint-pair-mode=per_pair: setup-2 keeps the top-M "
                         "observation-matched experts, then joint inverse tests each "
                         "candidate in its own tight box.")
    ap.add_argument("--setup2-oversample", type=int, default=10,
                    help="PCA prefilter multiplier when --setup2-top-k > 1. "
                         "The setup-2 candidate list is reranked by full-y nearest "
                         "neighbour before the top-M are kept.")
    ap.add_argument("--joint-pair-mode", choices=["mixture", "per_pair"],
                    default="mixture",
                    help="Joint inverse hypothesis handling. 'mixture' is the "
                         "existing soft expert mixture with union box; 'per_pair' "
                         "runs a separate tight-box joint inverse for each routed "
                         "expert pair and keeps the best joint evidence.")
    ap.add_argument("--joint-max-pairs", type=int, default=0,
                    help="Optional cap for --joint-pair-mode=per_pair. 0 = all "
                         "setup1/setup2 candidate pairs.")
    ap.add_argument("--joint-guard-setup1-wrms-factor", type=float, default=0.0,
                    help="Truth-blind consistency guard. If >0, reject the joint "
                         "update and keep setup-1 theta when the joint solution's "
                         "setup-1 WRMS is worse than factor * setup-1 single WRMS. "
                         "This prevents setup-2 from overwriting a well-explained "
                         "first observation.")
    ap.add_argument("--restarts", type=int, default=5,
                    help="CMA restarts for each inverse. Use 1 for smoke tests.")
    ap.add_argument("--max-iter", type=int, default=30,
                    help="CMA generations for each inverse. Use lower values for "
                         "quick smoke tests.")
    ap.add_argument("--skip-joint", action="store_true",
                    help="Setup-1 only: skip the proposer + MPM + joint 2-setup inverse. "
                         "Roughly halves runtime; useful for diagnosing setup-1 tail before "
                         "committing to a full panel.")
    ap.add_argument("--mpm-arch", type=str, default="cuda",
                    choices=["cuda", "cpu"],
                    help="Taichi backend for headless MLS-MPM")
    args = ap.parse_args()

    engine.STATE_ROOT = args.state_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = torch.float64

    print(f"[synth_panel] state-root  = {engine.STATE_ROOT}")
    print(f"[synth_panel] test-csv    = {args.test_csv}")
    print(f"[synth_panel] device      = {device}")
    print(f"[synth_panel] N cases     = {args.n_cases}")

    # ── Pipeline runtimes ────────────────────────────────────────────────
    geo_router, xs, ys = engine.load_runtime()
    inv_args = _make_inverse_args(args.routing_profile, restarts=args.restarts, max_iter=args.max_iter)
    setup2_profile = args.routing_profile_setup2 or args.routing_profile
    inv_args_setup2 = _make_inverse_args(setup2_profile, restarts=args.restarts, max_iter=args.max_iter)
    print(f"[synth_panel] routing s1={args.routing_profile}  s2={setup2_profile}  "
          f"s2_top_k={args.setup2_top_k}  joint_mode={args.joint_pair_mode}")

    # Headless MLS-MPM ─ instantiate ONCE (JIT warmup ~2s)
    from DataPipeline.headless_mls import HeadlessSimulatorMLS  # noqa: E402
    sim = HeadlessSimulatorMLS(
        max_W=7.0, max_H=7.0,
        rho=RHO_DEFAULT,
        arch=args.mpm_arch,
    )

    # ── Sample N cases from test.csv ─────────────────────────────────────
    df = pd.read_csv(args.test_csv)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n_cases, len(df)), replace=False)
    cases = df.iloc[idx].reset_index(drop=True)
    print(f"[synth_panel] sampled {len(cases)} cases from {len(df)} total")

    y_cols = [f"x_0{i}" for i in range(1, 9)]

    rows = []
    t0 = time.perf_counter()
    for i, row in enumerate(cases.itertuples(index=False)):
        n_t = float(row.n)
        eta_t = float(row.eta)
        sy_t = float(row.sigma_y)
        W1 = float(row.width)
        H1 = float(row.height)
        y_obs_1 = np.array([getattr(row, c) for c in y_cols], dtype=np.float64)
        case_record = {
            "case_id": int(idx[i]),
            "n_true": n_t, "eta_true": eta_t, "sigma_y_true": sy_t,
            "W1": W1, "H1": H1,
        }

        try:
            # ─── 1. setup-1 single inverse ─────────────────────────────
            item1 = selector.prepare_setup_for_inverse(
                f"case{i}_setup1", W1, H1, y_obs_1,
                geo_router, xs, ys, device, dtype, inv_args,
            )
            res1 = selector.inverse_single_setup(item1, inv_args, device, dtype)
            theta_hat_1 = (res1["theta_n"], res1["theta_eta"], res1["theta_sy"])
            erel_1 = _erel(theta_hat_1, (n_t, eta_t, sy_t))
            rel_n_1, rel_eta_1, rel_sy_1 = _per_param_rel(theta_hat_1, (n_t, eta_t, sy_t))
            case_record.update({
                "theta_hat_1_n": theta_hat_1[0],
                "theta_hat_1_eta": theta_hat_1[1],
                "theta_hat_1_sy": theta_hat_1[2],
                "erel_1": erel_1,
                "rel_n_1": rel_n_1, "rel_eta_1": rel_eta_1, "rel_sy_1": rel_sy_1,
            })

            if args.skip_joint:
                # Setup-1 only diagnostic: skip proposer + MPM + joint inverse.
                case_record["ok"] = True
            else:
                # ─── 2. propose setup-2 ────────────────────────────────────
                m_hat = Param(eta=theta_hat_1[1], n=theta_hat_1[0], sigmaY=theta_hat_1[2])
                next_s = propose_next_setup(m_hat, [Setup(H=H1, W=W1)], grid_step_cm=0.5)
                W2, H2 = float(next_s.W), float(next_s.H)

                # ─── 3. headless MLS-MPM forward at TRUE θ on (W2, H2) ──────
                t_mpm0 = time.perf_counter()
                y_obs_2 = sim.run(n_t, eta_t, sy_t, W2, H2)
                mpm_dt = time.perf_counter() - t_mpm0

                # ─── 4. setup-2 routing + joint 2-setup inverse ────────────
                # Setup-2 may use a different routing profile (e.g. joint_consistency
                # which reranks by z-distance to θ̂_1).  When the profile needs an
                # anchor we attach θ̂_1 in z-space.
                z_anchor_1 = np.array([
                    theta_hat_1[0],
                    np.log(max(theta_hat_1[1], 1e-12)),
                    np.log(max(theta_hat_1[2], 1e-12)),
                ], dtype=np.float64)
                inv_args_s2_case = SimpleNamespace(**vars(inv_args_setup2))
                inv_args_s2_case.shape_z_anchor = z_anchor_1
                if args.setup2_top_k > 1:
                    inv_args_s2_case.shape_weight_mode = "full_nn"
                    inv_args_s2_case.shape_top_k = int(args.setup2_top_k)
                    inv_args_s2_case.shape_gpfit_oversample = int(args.setup2_oversample)
                item2 = selector.prepare_setup_for_inverse(
                    f"case{i}_setup2", W2, H2, y_obs_2,
                    geo_router, xs, ys, device, dtype, inv_args_s2_case,
                )
                # Warm-start joint from setup-1 estimate
                joint_args = SimpleNamespace(**vars(inv_args))
                joint_args.shape_warm_start_z = z_anchor_1
                joint_args.shape_joint_pair_mode = args.joint_pair_mode
                joint_args.shape_joint_max_pairs = int(args.joint_max_pairs)
                res2 = selector.inverse_joint_setup(item1, item2, joint_args, device, dtype)
                guard_factor = float(args.joint_guard_setup1_wrms_factor)
                guard_rejected = False
                if guard_factor > 0.0 and "raw_wrms_a" in res2:
                    setup1_wrms = float(res1.get("raw_wrms_total", np.inf))
                    joint_s1_wrms = float(res2.get("raw_wrms_a", np.inf))
                    if joint_s1_wrms > guard_factor * max(setup1_wrms, 1e-12):
                        guard_rejected = True
                if guard_rejected:
                    theta_hat_2 = theta_hat_1
                else:
                    theta_hat_2 = (res2["theta_n"], res2["theta_eta"], res2["theta_sy"])
                erel_2 = _erel(theta_hat_2, (n_t, eta_t, sy_t))
                rel_n_2, rel_eta_2, rel_sy_2 = _per_param_rel(theta_hat_2, (n_t, eta_t, sy_t))
                case_record.update({
                    "W2": W2, "H2": H2,
                    "mpm_setup2_sec": mpm_dt,
                    "theta_hat_2_n": theta_hat_2[0],
                    "theta_hat_2_eta": theta_hat_2[1],
                    "theta_hat_2_sy": theta_hat_2[2],
                    "erel_2": erel_2,
                    "rel_n_2": rel_n_2, "rel_eta_2": rel_eta_2, "rel_sy_2": rel_sy_2,
                    "joint_pair_mode": args.joint_pair_mode,
                    "setup2_top_k": int(args.setup2_top_k),
                    "joint_n_pairs": int(res2.get("n_pairs", 1)),
                    "joint_guard_factor": guard_factor,
                    "joint_guard_rejected": bool(guard_rejected),
                    "setup1_raw_wrms": float(res1.get("raw_wrms_total", np.nan)),
                    "joint_raw_wrms_a": float(res2.get("raw_wrms_a", np.nan)),
                    "joint_raw_wrms_b": float(res2.get("raw_wrms_b", np.nan)),
                    "ok": True,
                })
                if "pair_choice" in res2:
                    case_record.update({
                        "joint_pair_sub_a": res2["pair_choice"].get("sub_a"),
                        "joint_pair_sub_b": res2["pair_choice"].get("sub_b"),
                        "joint_pair_ib": res2["pair_choice"].get("ib"),
                    })

        except Exception as exc:
            case_record["ok"] = False
            case_record["error"] = str(exc)
            print(f"  [case {i}] FAILED: {exc}")

        rows.append(case_record)

        if (i + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            eta_sec = (len(cases) - (i + 1)) / max(rate, 1e-6)
            ok_count = sum(1 for r in rows if r.get("ok"))
            print(f"  [{i+1}/{len(cases)}]  ok={ok_count}  "
                  f"rate={rate*60:.1f} cases/min  ETA={eta_sec/60:.1f} min")

    elapsed = time.perf_counter() - t0
    print(f"\n[synth_panel] completed in {elapsed/60:.1f} min")

    # ── Save per-case CSV + summary ─────────────────────────────────────
    df_out = pd.DataFrame(rows)
    out_csv = args.out_dir / "panel_recovery.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  → {out_csv}")

    ok = df_out[df_out.get("ok", False) == True]
    if len(ok) == 0:
        print("[synth_panel] no successful cases; aborting summary")
        return

    pass_1 = (ok["erel_1"] <= 0.1).sum()
    n_ok = len(ok)
    summary = {
        "n_cases": len(cases),
        "n_succeeded": n_ok,
        "elapsed_min": elapsed / 60.0,
        "skip_joint": bool(args.skip_joint),
        "routing_profile": args.routing_profile,
        "routing_profile_setup2": setup2_profile,
        "setup2_top_k": int(args.setup2_top_k),
        "joint_pair_mode": args.joint_pair_mode,
        "joint_max_pairs": int(args.joint_max_pairs),
        "joint_guard_setup1_wrms_factor": float(args.joint_guard_setup1_wrms_factor),
        "restarts": int(args.restarts),
        "max_iter": int(args.max_iter),
        "setup_1": {
            "pass_count": int(pass_1),
            "pass_rate": float(pass_1) / n_ok,
            "median_E_rel": float(ok["erel_1"].median()),
            "median_rel_n": float(ok["rel_n_1"].median()),
            "median_rel_eta": float(ok["rel_eta_1"].median()),
            "median_rel_sigma_y": float(ok["rel_sy_1"].median()),
        },
        "predecessor_baseline_2023_30case": {
            "pass_count_setup1": 11, "pass_count_setup2": 20,  # Hamamichi §7.1
            "panel_size": 30,
        },
    }
    if not args.skip_joint:
        pass_2 = (ok["erel_2"] <= 0.1).sum()
        summary["setup_2_joint"] = {
            "pass_count": int(pass_2),
            "pass_rate": float(pass_2) / n_ok,
            "median_E_rel": float(ok["erel_2"].median()),
            "median_rel_n": float(ok["rel_n_2"].median()),
            "median_rel_eta": float(ok["rel_eta_2"].median()),
            "median_rel_sigma_y": float(ok["rel_sy_2"].median()),
        }
        summary["mpm_setup2_median_sec"] = float(ok["mpm_setup2_sec"].median())
        summary["todo"] = "3-setup ablation requires extending selector.inverse_joint_setup to N items"
    out_json = args.out_dir / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Recovery Summary ===")
    print(f"  Setup-1 pass (E_rel <= 0.1):  {pass_1}/{n_ok}  "
          f"({pass_1/n_ok*100:.1f}%)  median E_rel={ok['erel_1'].median():.4f}")
    if not args.skip_joint:
        pass_2 = (ok["erel_2"] <= 0.1).sum()
        print(f"  Setup-2 (joint) pass:         {pass_2}/{n_ok}  "
              f"({pass_2/n_ok*100:.1f}%)  median E_rel={ok['erel_2'].median():.4f}")
    else:
        print(f"  Setup-2 (joint) pass:         skipped (--skip-joint)")
    print(f"  Hamamichi 2023 (30-case ref): 11/30 -> 20/30")
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
