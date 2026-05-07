"""Production-grade y8q + GP-aware-likelihood inverse on Hamamichi 2023 prior data.

Sweeps every material directory under ``data/real_world_experiments`` of the
form ``ref_<Material>_<H>_<W>_<setup_id>``.  For each material, runs setup1-
alone, setup2-alone, and joint inverse with paper-grade defaults
(`floor=0.20`, `disable_calib=True`, 5-restart CMA, Hessian Laplace CI).

Truth-blind UQ only — these prior materials lack rheometer truth in this repo
(except Lotion, where rheometer files are present but in a different schema).
The reported quantities (point estimate + 95 % CI + per-restart dispersion +
identifiability flag) are the same that a paper-grade Hamamichi-style report
needs.

Usage:
    python scripts/run_y8q_prior.py
    python scripts/run_y8q_prior.py --material Carbonara
    python scripts/run_y8q_prior.py --restarts 1
    python scripts/run_y8q_prior.py --out scripts/run_y8q_prior.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as V10
from Optimization.libs import selector as SHA


# Directory pattern: ref_<Material>_<H>_<W>_<setup_id>
DIR_RE = re.compile(r"^ref_(?P<mat>[^_]+)_(?P<H>[\d.]+)_(?P<W>[\d.]+)_(?P<sid>\d+)$")


def discover_materials(data_root: Path) -> dict[str, list[tuple[int, float, float, Path]]]:
    """Return {material -> [(setup_id, H, W, dir), ...]} sorted by setup_id."""
    out: dict[str, list[tuple[int, float, float, Path]]] = {}
    for child in sorted(data_root.iterdir()):
        if not child.is_dir():
            continue
        m = DIR_RE.match(child.name)
        if not m:
            continue
        mat = m.group("mat")
        sid = int(m.group("sid"))
        H = float(m.group("H"))
        W = float(m.group("W"))
        out.setdefault(mat, []).append((sid, H, W, child))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return out


def make_args(restarts: int, trail_csv: str | None = None) -> argparse.Namespace:
    """Production-grade argset (matches scripts/run_y8q_chuno.make_args, paper-grade restarts)."""
    return argparse.Namespace(
        # routing / re-rank
        shape_top_k=1,
        shape_l2_mode="hard", shape_l2_eps_cm=0.5,
        shape_eps_cap_by_primary=False,
        shape_weight_mode="gp_fit_y8_quantile", shape_gpfit_oversample=3,
        shape_topk_softmax_tau=1.5,
        shape_per_sub_z_scale=False,
        # loss
        loss_mode="log_nuisance_gp",
        sigma_bias=0.25, sigma_trend=0.20, sigma_y_min=5.0,
        shape_gp_noise_log_floor=0.20,
        shape_disable_calib=True,
        shape_cma_restarts=int(restarts),
        shape_cma_trail_csv=trail_csv,
        # CMA-ES
        shape_sigma0=0.25, shape_max_iter=30, shape_popsize=12,
        shape_cma_seed_offset=0,
        # bbox
        shape_box_pad=0.10, shape_support_weight=0.25,
        # frame_w
        shape_frame_w=None,
        shape_info_frame_w=False,
        # UQ
        shape_report_ci=True,
    )


def _ci_block(res: dict, key: str) -> tuple[list[float], bool] | tuple[None, None]:
    ci = res.get("theta_ci_95")
    idf = res.get("identifiable")
    if ci is None or idf is None:
        return None, None
    return list(ci[key]), bool(idf[key])


def _summarize_setup(label: str, ref_dir: Path, W: float, H: float, args, geo, xs, ys,
                     device, dtype, joint_only: bool = False) -> dict | None:
    y_obs = np.load(ref_dir / "y_obs.npy").astype(np.float64).ravel()
    if y_obs.shape != (8,):
        print(f"  [{label}] y_obs shape {y_obs.shape} != (8,) — skipped")
        return None
    print(f"\n--- {label}  W={W:.2f} H={H:.2f}  y_obs[7]={y_obs[7]:.3f} ---", flush=True)
    item = SHA.prepare_setup_shape(ref_dir.name, W, H, y_obs,
                                    geo, xs, ys, device, dtype, args)
    primary = item["members"][0]
    bbox_sy = (float(np.exp(primary.z_lo[2])), float(np.exp(primary.z_hi[2])))
    z_score = abs(y_obs[7] - primary.y8_med) / max(primary.y8_std, 1e-6)
    print(f"  routed sub_{primary.sub_id}  bbox σ_y=[{bbox_sy[0]:.2f},{bbox_sy[1]:.2f}]"
          f"  y8_med={primary.y8_med:.2f}±{primary.y8_std:.2f}  z-score={z_score:.2f}",
          flush=True)

    base = {
        "label": label,
        "dir": ref_dir.name,
        "W": float(W), "H": float(H),
        "y_obs": y_obs.tolist(),
        "sub_id": int(primary.sub_id),
        "bin_id": int(primary.bin_id),
        "bbox_sigma_y": list(bbox_sy),
        "y8_med": float(primary.y8_med),
        "y8_std": float(primary.y8_std),
        "y8_z_score": float(z_score),
        "item": item,
    }
    if joint_only:
        return base  # routing-only; skip the alone-inverse to save wall-time

    t0 = time.time()
    res = SHA.inverse_single_shape(item, args, device, dtype)
    elapsed = time.time() - t0
    th = (float(res["theta_n"]), float(res["theta_eta"]), float(res["theta_sy"]))
    print(f"  θ̂=({th[0]:.4f},{th[1]:.3f},{th[2]:.3f})  {elapsed:.1f}s", flush=True)
    n_ci, n_id = _ci_block(res, "n")
    e_ci, e_id = _ci_block(res, "eta")
    s_ci, s_id = _ci_block(res, "sigma_y")
    if n_ci is not None:
        print(f"  CI 95%:  n=[{n_ci[0]:.4f},{n_ci[1]:.4f}]{'✓' if n_id else '?'}  "
              f"η=[{e_ci[0]:.3f},{e_ci[1]:.3f}]{'✓' if e_id else '?'}  "
              f"σ_y=[{s_ci[0]:.2f},{s_ci[1]:.2f}]{'✓' if s_id else '?'}",
              flush=True)
    z_disp = res.get("z_dispersion")
    if z_disp is not None:
        print(f"  z-dispersion (std across restarts): "
              f"n={z_disp[0]*100:.1f}%  η={z_disp[1]*100:.1f}%  σ_y={z_disp[2]*100:.1f}%",
              flush=True)

    base.update({
        "theta": list(th),
        "theta_ci_95": {"n": n_ci, "eta": e_ci, "sigma_y": s_ci},
        "identifiable": {"n": n_id, "eta": e_id, "sigma_y": s_id},
        "z_dispersion": z_disp,
        "elapsed_s": float(elapsed),
    })
    return base


def _summarize_joint(item_a, item_b, args, device, dtype) -> dict:
    print(f"\n--- joint setup1 + setup2 ---", flush=True)
    t0 = time.time()
    res = SHA.inverse_double_shape(item_a, item_b, args, device, dtype)
    elapsed = time.time() - t0
    th = (float(res["theta_n"]), float(res["theta_eta"]), float(res["theta_sy"]))
    print(f"  θ̂=({th[0]:.4f},{th[1]:.3f},{th[2]:.3f})  {elapsed:.1f}s", flush=True)
    n_ci, n_id = _ci_block(res, "n")
    e_ci, e_id = _ci_block(res, "eta")
    s_ci, s_id = _ci_block(res, "sigma_y")
    if n_ci is not None:
        print(f"  CI 95%:  n=[{n_ci[0]:.4f},{n_ci[1]:.4f}]{'✓' if n_id else '?'}  "
              f"η=[{e_ci[0]:.3f},{e_ci[1]:.3f}]{'✓' if e_id else '?'}  "
              f"σ_y=[{s_ci[0]:.2f},{s_ci[1]:.2f}]{'✓' if s_id else '?'}",
              flush=True)
    z_disp = res.get("z_dispersion")
    if z_disp is not None:
        print(f"  z-dispersion (std across restarts): "
              f"n={z_disp[0]*100:.1f}%  η={z_disp[1]*100:.1f}%  σ_y={z_disp[2]*100:.1f}%",
              flush=True)
    return {
        "theta": list(th),
        "theta_ci_95": {"n": n_ci, "eta": e_ci, "sigma_y": s_ci},
        "identifiable": {"n": n_id, "eta": e_id, "sigma_y": s_id},
        "z_dispersion": z_disp,
        "elapsed_s": float(elapsed),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-root", type=Path,
                    default=REPO / "data" / "real_world_experiments")
    ap.add_argument("--state-root", type=Path,
                    default=REPO / "Models" / "v10_yshape_planB")
    ap.add_argument("--out", type=Path,
                    default=REPO / "scripts" / "run_y8q_prior.json")
    ap.add_argument("--restarts", type=int, default=5,
                    help="CMA-ES restarts (paper-grade=5; smoke-test=1)")
    ap.add_argument("--material", type=str, default=None,
                    help="Run only this material (e.g. Carbonara)")
    ap.add_argument("--joint-only", action="store_true",
                    help="Skip setup1-alone and setup2-alone inverses; only run joint. "
                         "Saves ~50%% of batch wall-time (setup-alone diagnostics still "
                         "implicit in joint via routed sub_id + bbox).")
    ap.add_argument("--cma-trail-csv", type=Path, default=None,
                    help="If set, append every CMA-ES candidate (θ + loss + restart "
                         "+ generation + candidate_idx) to this CSV.  Use later "
                         "with scripts/dump_gamma_dot.py --trail-csv to compute γ̇ "
                         "offline per candidate.")
    cli = ap.parse_args()

    V10.STATE_ROOT = cli.state_root.resolve()
    geo, xs, ys = V10.load_runtime()
    device, dtype = V10.HC.DEVICE, V10.HC.DTYPE

    materials = discover_materials(cli.data_root)
    if cli.material is not None:
        if cli.material not in materials:
            available = ", ".join(sorted(materials))
            raise SystemExit(f"--material {cli.material!r} not found. Available: {available}")
        materials = {cli.material: materials[cli.material]}

    args = make_args(cli.restarts, trail_csv=str(cli.cma_trail_csv) if cli.cma_trail_csv else None)
    print(f"\n{'='*72}\n=== Prior data y8q inverse  restarts={cli.restarts}  "
          f"materials={len(materials)} ===\n{'='*72}", flush=True)
    if cli.cma_trail_csv:
        print(f"CMA candidate trail → {cli.cma_trail_csv}", flush=True)

    summary: dict = {}
    t_global = time.time()
    for mat, setups in materials.items():
        print(f"\n{'#'*72}\n# {mat}  ({len(setups)} setups)\n{'#'*72}", flush=True)
        per_setup: dict = {}
        items_for_joint: list = []
        for sid, H, W, ref_dir in setups:
            label = f"setup{sid} (W={W:.2f},H={H:.2f})"
            try:
                d = _summarize_setup(label, ref_dir, W, H, args, geo, xs, ys, device, dtype,
                                      joint_only=cli.joint_only)
            except Exception as e:
                print(f"  [{label}] FAILED: {e}", flush=True)
                d = {"error": str(e), "dir": ref_dir.name, "W": W, "H": H}
            if d is None:
                continue
            item = d.pop("item", None)
            per_setup[f"setup{sid}"] = d
            if item is not None:
                items_for_joint.append((sid, item))

        # Joint = first two setups (setup1 + setup2 by convention)
        joint_d: dict | None = None
        if len(items_for_joint) >= 2:
            (_, item_a), (_, item_b) = items_for_joint[0], items_for_joint[1]
            try:
                joint_d = _summarize_joint(item_a, item_b, args, device, dtype)
            except Exception as e:
                print(f"  [joint] FAILED: {e}", flush=True)
                joint_d = {"error": str(e)}
        else:
            print(f"  [{mat}] only {len(items_for_joint)} setups loaded — skipping joint")

        summary[mat] = {"per_setup": per_setup, "joint": joint_d}

    cli.out.write_text(json.dumps(summary, indent=2))
    print(f"\n\nSaved {cli.out}  ({time.time()-t_global:.1f}s total)\n")

    # ----- summary table -----
    print(f"\n=== Joint inverse summary (truth-blind UQ) ===")
    print(f"{'material':<40} {'n̂':>7} {'η̂':>8} {'σ_yhat':>8}  {'σ_y CI 95%':<22}  {'σ_y disp':>9}  {'id'}")
    for mat, d in summary.items():
        j = d.get("joint")
        if j is None or "error" in j:
            print(f"{mat:<40} {'(joint failed)':>20}")
            continue
        t = j["theta"]
        ci = j["theta_ci_95"]["sigma_y"]
        idf = j["identifiable"]
        zd = j.get("z_dispersion")
        sy_disp = f"{zd[2]*100:>+.1f}%" if zd is not None else "-"
        ci_str = f"[{ci[0]:.2f},{ci[1]:.2f}]" if ci is not None else "-"
        flags = ("n" if idf and idf["n"] else "·") + \
                ("e" if idf and idf["eta"] else "·") + \
                ("s" if idf and idf["sigma_y"] else "·")
        print(f"{mat:<40} {t[0]:>7.4f} {t[1]:>8.3f} {t[2]:>8.3f}  {ci_str:<22}  "
              f"{sy_disp:>9}  {flags}")


if __name__ == "__main__":
    main()
