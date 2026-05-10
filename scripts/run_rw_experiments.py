"""Batch driver for the production y8-quantile + GP-aware inverse.

Interactive application use should prefer:

    python -m Optimization.estimate_first_setup ...
    python -m Optimization.estimate_joint_setup ...

Batch usage:

    python -m scripts.run_rw_experiments \
        --root data/new_real_world_experiments \
        --out-dir OptimizationResults/<run_name> \
        --materials all --mode all
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import Optimization.libs.engine as ENGINE
from Optimization.libs import selector as SHA


SETUP_RE = re.compile(r"^ref_(?P<material>.+)_(?P<H>[0-9.]+)_(?P<W>[0-9.]+)_(?P<idx>[0-9]+)$")


def cleanup_runtime() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def discover_experiments(root: Path) -> dict[str, list[dict]]:
    """Return material -> sorted setup folders that match ref_<material>_<H>_<W>_<idx>."""
    out: dict[str, list[dict]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = SETUP_RE.match(d.name)
        if not m:
            continue
        y_path = d / "y_obs.npy"
        out.setdefault(m.group("material"), []).append({
            "name": d.name,
            "path": d,
            "material": m.group("material"),
            "H": float(m.group("H")),
            "W": float(m.group("W")),
            "idx": int(m.group("idx")),
            "has_y_obs": y_path.is_file(),
        })
    for rows in out.values():
        rows.sort(key=lambda r: r["idx"])
    return out


def run_single_setups(materials, geo_router, xs, ys, device, dtype, args) -> tuple[list, list]:
    rows, skipped = [], []
    for mat, setups in materials.items():
        for setup in setups:
            if not setup["has_y_obs"]:
                skipped.append({
                    "mode": "single",
                    "material": mat,
                    "setup": setup["name"],
                    "reason": "missing y_obs.npy",
                })
                continue
            try:
                y_obs = np.load(setup["path"] / "y_obs.npy").astype(np.float64).ravel()
                item = SHA.prepare_setup_for_inverse(
                    setup["name"], float(setup["W"]), float(setup["H"]), y_obs,
                    geo_router, xs, ys, device, dtype, args,
                )
                res = SHA.inverse_single_setup(item, args, device, dtype)
            except Exception as exc:
                skipped.append({
                    "mode": "single",
                    "material": mat,
                    "setup": setup["name"],
                    "reason": str(exc),
                })
                continue
            if res is None:
                skipped.append({
                    "mode": "single",
                    "material": mat,
                    "setup": setup["name"],
                    "reason": "no admitted members",
                })
                continue
            rows.append({
                "mode": "single",
                "material": mat,
                "setup": setup["name"],
                "idx": setup["idx"],
                "H": setup["H"],
                "W": setup["W"],
                "gid": item["gid"],
                "theta_n": res["theta_n"],
                "theta_eta": res["theta_eta"],
                "theta_sy": res["theta_sy"],
                "objective": res["objective"],
                "opt_s": res["opt_s"],
                "sub_ids": json.dumps(res["sub_ids"]),
                "basins": json.dumps(res["basins"]),
                "y8": float(item["y_obs"][-1]),
                "raw_wrms_total": res["raw_wrms_total"],
                "theta_ci_95": json.dumps(res.get("theta_ci_95")),
                "identifiable": json.dumps(res.get("identifiable")),
                "z_dispersion": json.dumps(res.get("z_dispersion")),
            })
            cleanup_runtime()
    return rows, skipped


def run_joint_setups(materials, geo_router, xs, ys, device, dtype, args) -> tuple[list, list]:
    rows, skipped = [], []
    for mat, setups in materials.items():
        if len(setups) < 2:
            skipped.append({"mode": "joint", "material": mat, "reason": "less than two setups"})
            continue
        pair = setups[:2]
        if any(not setup["has_y_obs"] for setup in pair):
            skipped.append({"mode": "joint", "material": mat, "reason": "missing y_obs.npy"})
            continue
        try:
            items = []
            for setup in pair:
                y_obs = np.load(setup["path"] / "y_obs.npy").astype(np.float64).ravel()
                items.append(SHA.prepare_setup_for_inverse(
                    setup["name"], float(setup["W"]), float(setup["H"]), y_obs,
                    geo_router, xs, ys, device, dtype, args,
                ))
            res = SHA.inverse_joint_setup(items[0], items[1], args, device, dtype)
        except Exception as exc:
            skipped.append({"mode": "joint", "material": mat, "reason": str(exc)})
            continue
        if res is None:
            skipped.append({"mode": "joint", "material": mat, "reason": "no admitted members"})
            continue
        rows.append({
            "mode": "joint",
            "material": mat,
            "setup_a": pair[0]["name"],
            "setup_b": pair[1]["name"],
            "theta_n": res["theta_n"],
            "theta_eta": res["theta_eta"],
            "theta_sy": res["theta_sy"],
            "objective": res["objective"],
            "opt_s": res["opt_s"],
            "sub_ids": json.dumps(res["sub_ids"]),
            "basins": json.dumps(res["basins"]),
            "y8_a": float(items[0]["y_obs"][-1]),
            "y8_b": float(items[1]["y_obs"][-1]),
            "raw_wrms_total": res["raw_wrms_total"],
            "theta_ci_95": json.dumps(res.get("theta_ci_95")),
            "identifiable": json.dumps(res.get("identifiable")),
            "z_dispersion": json.dumps(res.get("z_dispersion")),
        })
        cleanup_runtime()
    return rows, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=PIPELINE_ROOT / "data" / "new_real_world_experiments")
    ap.add_argument("--out-dir", type=Path, default=PIPELINE_ROOT / "OptimizationResults" / "real_world_batch")
    ap.add_argument("--state-root", type=Path, default=ENGINE.STATE_ROOT)
    ap.add_argument("--materials", default="all", help="comma-separated material names or 'all'")
    ap.add_argument("--mode", choices=["single", "joint", "double", "all"], default="all",
                    help="'joint' is the canonical name; 'double' is accepted for compatibility")
    ap.add_argument("--loss-mode", choices=["log_nuisance_gp"], default="log_nuisance_gp")
    ap.add_argument("--sigma-bias", type=float, default=0.25)
    ap.add_argument("--sigma-trend", type=float, default=0.20)
    ap.add_argument("--sigma-y-min", type=float, default=5.0)
    ap.add_argument("--inverse-mode", choices=["shape"], default="shape")
    SHA.add_argparse_args(ap)
    args = ap.parse_args()

    ENGINE.STATE_ROOT = args.state_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    materials = discover_experiments(args.root)
    if args.materials.lower() != "all":
        keep = {m.strip() for m in args.materials.split(",") if m.strip()}
        materials = {m: rows for m, rows in materials.items() if m in keep}

    geo_router, xs, ys = ENGINE.load_runtime()
    device, dtype = ENGINE.HC.DEVICE, ENGINE.HC.DTYPE

    all_skipped = []
    single_rows, joint_rows = [], []
    if args.mode in {"single", "all"}:
        single_rows, skipped = run_single_setups(materials, geo_router, xs, ys, device, dtype, args)
        all_skipped.extend(skipped)
        pd.DataFrame(single_rows).to_csv(args.out_dir / "single_real_world.csv", index=False)
    if args.mode in {"joint", "double", "all"}:
        joint_rows, skipped = run_joint_setups(materials, geo_router, xs, ys, device, dtype, args)
        all_skipped.extend(skipped)
        pd.DataFrame(joint_rows).to_csv(args.out_dir / "joint_real_world.csv", index=False)
    pd.DataFrame(all_skipped).to_csv(args.out_dir / "skipped.csv", index=False)

    summary = {
        "root": str(args.root),
        "state_root": str(ENGINE.STATE_ROOT),
        "n_materials": len(materials),
        "n_single": len(single_rows),
        "n_joint": len(joint_rows),
        "n_skipped": len(all_skipped),
        "single_mean_s": float(np.mean([r["opt_s"] for r in single_rows])) if single_rows else None,
        "joint_mean_s": float(np.mean([r["opt_s"] for r in joint_rows])) if joint_rows else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
