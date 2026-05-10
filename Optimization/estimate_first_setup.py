"""1st-setup Herschel-Bulkley parameter estimation + Hessian-based 2nd-setup proposal.

Mirror of prev-work `6_optimization/herschel_bulkley_parameter_estimation_for_1st_setup.py`,
but with the MoGP-rBCM surrogate replacing MPM and y8 flow distance replacing
binary silhouettes.

Pipeline:
    1. Load `<ref_dir>/y_obs.npy` (8 frames of flow distance), parse (H, W) from dir name.
    2. Inverse: y8-quantile routing + GP-aware likelihood over the Plan-B MoGP bank.
    3. Propose next setup (W*, H*) via analytical Poiseuille Hessian orthogonality.
    4. Create `data/real_world_experiments/ref_<material>_<H*>_<W*>_2/` and write
       `settings.xml` inside (same format as propose_initial_setup/out_xml.sh).
    5. Write `<ref_dir>/theta_hat.json` with theta_hat + proposed setup info.

Usage:
    python -m Optimization.estimate_first_setup \\
        --state-root Models/yshape_mogp_production \\
        -f data/real_world_experiments/ref_Sesame_4.2_4.0_1 \\
        --density 1.2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from Optimization.libs import engine as ENGINE
from Optimization.libs import selector as SHA
from Optimization.libs.mechanism import Param, Setup, propose_next_setup

_OUT_XML_SH = REPO / "propose_initial_setup" / "out_xml.sh"
_REF_DIR_RE = re.compile(r"^ref_(?P<material>.+)_(?P<H>[0-9.]+)_(?P<W>[0-9.]+)_(?P<idx>[0-9]+)$")


def parse_ref_dir(ref_dir: Path) -> dict:
    m = _REF_DIR_RE.match(ref_dir.name)
    if not m:
        raise ValueError(
            f"ref dir name must be ref_<material>_<H>_<W>_<idx>: {ref_dir.name}"
        )
    return {
        "material": m.group("material"),
        "H": float(m.group("H")),
        "W": float(m.group("W")),
        "idx": int(m.group("idx")),
    }


def _export_setup_xml(out_dir: Path, material: str, H: float, W: float,
                      rho: float, idx: int) -> Path:
    """Create <out_dir>/ and write settings.xml via out_xml.sh.

    Mirrors prev-work exportSecondSetupXML():
        arg_1  = ref dir path
        arg_2  = "settings.xml"
        arg_3  = optimizer root name (e.g. "<material>_2")
        arg_4  = H (cm)
        arg_5  = W (cm)
        arg_6  = rho
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_path = out_dir / "settings.xml"
    root_dir_name = str(out_dir.parent / f"{material}_{idx}")

    Hstr = str(round(H, 2))
    Wstr = str(round(W, 2))

    cmd = (
        f'sh "{_OUT_XML_SH}" '
        f'"{out_dir}" "settings.xml" '
        f'"{root_dir_name}" '
        f'{Hstr} {Wstr} {rho}'
    )
    ret = os.system(cmd)
    if ret != 0 or not settings_path.exists():
        # Fallback: write XML directly (Windows without bash/WSL)
        _write_xml_direct(settings_path, root_dir_name, H, W, rho)
    return settings_path


def _write_xml_direct(path: Path, root_dir_path: str, H: float, W: float, rho: float) -> None:
    """Direct Python XML writer — fallback when sh is unavailable."""
    Hs = str(round(H, 2))
    Ws = str(round(W, 2))
    xml = f"""<?xml version="1.0"?>
<Optimizer>
  <path
    root_dir_path="{root_dir_path}"
    GL_render_path="../libs/3D/GLRender3d/build/GLRender3d"
    mpm_path="../libs/3D/MPM3d/AGTaichiMPM.py"
    particle_skinner_path="../libs/ParticleSkinner3DTaichi.py"
    shell_script_dir_path="../libs/3D/shellScript3d"
    GL_emulation_render_path="../libs/3D/GLEmulationRender3d/build/GLEmulationRender3d"
  />

  <setup
    RHO="{rho}"
    H="{Hs}"
    W="{Ws}"
  />
  <cuboid min="-0.150000 -0.150000 -0.150000" max="{Ws} {Hs} 4.150000" density="{rho}" cell_samples_per_dim="2" vel="0.0 0.0 0.0" omega="0.0 0.0 0.0" />
  <static_box min="-100.000000 -1.000000 -100.000000" max="100.000000 0.000000 100.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 0.000000" max="0.000000 20.000000 4.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 -0.300000" max="{Ws} 20.000000 0.000000" boundary_behavior="sticking"/>
  <static_box min="-1.000000 0.000000 4.000000" max="{Ws} 20.000000 4.300000" boundary_behavior="sticking"/>

</Optimizer>
"""
    path.write_text(xml, encoding="utf-8")


def estimate_first_setup(ref_dir: Path, args) -> dict:
    """Single-setup inverse → θ̂ + create proposed-setup2 folder with settings.xml."""
    info = parse_ref_dir(ref_dir)
    y_obs_path = ref_dir / "y_obs.npy"
    if not y_obs_path.is_file():
        raise FileNotFoundError(f"missing {y_obs_path}")
    y_obs = np.load(y_obs_path).astype(np.float64).ravel()

    geo_router, xs, ys = ENGINE.load_runtime()
    device, dtype = ENGINE.HC.DEVICE, ENGINE.HC.DTYPE

    item = SHA.prepare_setup_for_inverse(
        ref_dir.name, info["W"], info["H"], y_obs,
        geo_router, xs, ys, device, dtype, args,
    )
    res = SHA.inverse_single_setup(item, args, device, dtype)
    if res is None:
        raise RuntimeError("inverse failed: no admitted experts")

    m_hat = Param(
        eta=float(res["theta_eta"]),
        n=float(res["theta_n"]),
        sigmaY=float(res["theta_sy"]),
    )

    # ── Hessian-based active setup2 proposal ────────────────────────────────
    next_s = propose_next_setup(
        m_hat,
        [Setup(H=info["H"], W=info["W"])],
        grid_step_cm=float(args.proposal_grid_cm),
    )
    next_idx = info["idx"] + 1
    next_name = (
        f"ref_{info['material']}_{round(next_s.H, 2)}_{round(next_s.W, 2)}_{next_idx}"
    )
    next_dir = Path(args.data_root) / next_name if args.data_root else ref_dir.parent / next_name

    # ── Create folder + settings.xml ────────────────────────────────────────
    xml_path = _export_setup_xml(next_dir, info["material"],
                                 next_s.H, next_s.W, args.density, next_idx)

    out = {
        "material": info["material"],
        "setup_1": {"H": info["H"], "W": info["W"], "name": ref_dir.name},
        "theta_hat": {
            "n": float(m_hat.n),
            "eta": float(m_hat.eta),
            "sigma_y": float(m_hat.sigmaY),
        },
        "objective": float(res["objective"]),
        "opt_s": float(res["opt_s"]),
        "proposed_setup_2": {
            "H": float(next_s.H),
            "W": float(next_s.W),
            "name": next_name,
            "dir": str(next_dir),
            "settings_xml": str(xml_path),
        },
    }
    if "theta_ci_95" in res:
        out["theta_ci_95"] = res["theta_ci_95"]
        out["z_std"] = res.get("z_std")
        out["identifiable"] = res.get("identifiable")
    if "z_dispersion" in res:
        out["z_dispersion"] = res.get("z_dispersion")

    (ref_dir / "theta_hat.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-f", "--first-dir", type=Path, required=True,
                    help="Path to ref_<material>_<H>_<W>_<idx>/ directory")
    ap.add_argument("-d", "--density", type=float, default=1.0,
                    help="Material density ρ (g/cm³) for settings.xml (default: 1.0)")
    ap.add_argument("--state-root", type=Path, default=ENGINE.STATE_ROOT,
                    help="Trained MoGP-rBCM bank root")
    ap.add_argument("--proposal-grid-cm", type=float, default=0.5,
                    help="Candidate (W, H) grid step for setup-2 proposal in cm (default: 0.5)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Root folder where proposed setup2 dir is created. "
                         "Default: same folder as <first-dir>")
    ap.add_argument("--loss-mode", choices=["log_nuisance_gp"], default="log_nuisance_gp",
                    help="GP-aware Type-II ML likelihood (production). "
                         "Old modes linear / log_nuisance / mog_likelihood "
                         "archived 2026-05-07 — see "
                         "docs/archive_2026-05-07/selector_full_2026-05-07.py.")
    ap.add_argument("--sigma-bias", type=float, default=0.25)
    ap.add_argument("--sigma-trend", type=float, default=0.20)
    ap.add_argument("--sigma-y-min", type=float, default=5.0)
    ap.add_argument("--inverse-mode", choices=["shape"], default="shape")
    SHA.add_argparse_args(ap)
    args = ap.parse_args()

    ENGINE.STATE_ROOT = args.state_root.resolve()
    out = estimate_first_setup(args.first_dir.resolve(), args)

    print(json.dumps(out, indent=2))
    th = out["theta_hat"]
    if "theta_ci_95" in out:
        ci = out["theta_ci_95"]
        print(
            f"\n=== θ̂ MAP and 95 % credible intervals (Laplace approx.) ==="
            f"\n  n        = {th['n']:>8.4f}    CI 95%: [{ci['n'][0]:>8.4f}, {ci['n'][1]:>8.4f}]"
            f"\n  η        = {th['eta']:>8.4f}    CI 95%: [{ci['eta'][0]:>8.4f}, {ci['eta'][1]:>8.4f}]"
            f"\n  σ_y      = {th['sigma_y']:>8.4f}    CI 95%: [{ci['sigma_y'][0]:>8.4f}, {ci['sigma_y'][1]:>8.4f}]"
        )
    print(
        f"\n→ Created: {out['proposed_setup_2']['dir']}"
        f"\n→ Written: {out['proposed_setup_2']['settings_xml']}"
            f"\n\nBefore running the joint inverse, place into the folder above:"
        f"\n  - y_obs.npy  (8-frame flow distance from 2nd experiment)"
        f"\n  - camera_params.xml  (calibration)"
        f"\n\nThen run:"
        f"\n  python -m Optimization.estimate_joint_setup -f {args.first_dir} -s {out['proposed_setup_2']['dir']}"
    )


if __name__ == "__main__":
    main()
