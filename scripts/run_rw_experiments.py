"""Plan B production driver — run RW inverse on 12 materials with shape selector.

Single inverse mode: '--inverse-mode shape' (CMA-ES + σY two-stage prior).
This is the production path for the V3p2 bank.

Usage:
    python -m scripts.run_rw_experiments \
        --state-root pipeline_v10/Models/v10_yshape_v3p2_round2partial \
        --out-dir pipeline_v10/OptimizationResults/<run_name> \
        --materials all --mode all \
        --shape-sy-prior-weight 0.5 --shape-sy-sat-lo 2.0
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import Optimization.libs.engine as V10
from Optimization.libs import selector as SHA


SETUP_RE = re.compile(r"^ref_(?P<material>.+)_(?P<H>[0-9.]+)_(?P<W>[0-9.]+)_(?P<idx>[0-9]+)$")


def cleanup_runtime() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def discover(root: Path) -> dict[str, list[dict]]:
    """Walk root for ref_<material>_<H>_<W>_<idx>/y_obs.npy directories."""
    out: dict[str, list[dict]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        m = SETUP_RE.match(d.name)
        if not m:
            continue
        y_path = d / "y_obs.npy"
        out.setdefault(m.group("material"), []).append({
            "name": d.name, "path": d,
            "material": m.group("material"),
            "H": float(m.group("H")), "W": float(m.group("W")),
            "idx": int(m.group("idx")),
            "has_y_obs": y_path.is_file(),
        })
    for rows in out.values():
        rows.sort(key=lambda r: r["idx"])
    return out


def run_single(materials, geo_router, xs, ys, device, dtype, args) -> tuple[list, list]:
    rows, skipped = [], []
    for mat, setups in materials.items():
        for s in setups:
            if not s["has_y_obs"]:
                skipped.append({"mode": "single", "material": mat, "setup": s["name"], "reason": "missing y_obs.npy"})
                continue
            try:
                W, H = float(s["W"]), float(s["H"])
                y_obs = np.load(s["path"] / "y_obs.npy").astype(np.float64).ravel()
                item = SHA.prepare_setup_shape(s["name"], W, H, y_obs,
                                                 geo_router, xs, ys, device, dtype, args)
            except Exception as exc:
                skipped.append({"mode": "single", "material": mat, "setup": s["name"], "reason": str(exc)})
                continue
            res = SHA.inverse_single_shape(item, args, device, dtype)
            if res is None:
                skipped.append({"mode": "single", "material": mat, "setup": s["name"], "reason": "no admitted members"})
                continue
            rows.append({
                "mode": "single", "material": mat, "setup": s["name"], "idx": s["idx"],
                "H": s["H"], "W": s["W"], "gid": item["gid"],
                "theta_n": res["theta_n"], "theta_eta": res["theta_eta"], "theta_sy": res["theta_sy"],
                "objective": res["objective"], "opt_s": res["opt_s"],
                "sub_ids": res["sub_ids"], "basins": res["basins"],
                "y8": float(item["y_obs"][-1]),
                "raw_wrms_total": res["raw_wrms_total"],
                "selector_mode": "shape",
            })
            cleanup_runtime()
    return rows, skipped


def run_double(materials, geo_router, xs, ys, device, dtype, args) -> tuple[list, list]:
    rows, skipped = [], []
    for mat, setups in materials.items():
        if len(setups) < 2:
            skipped.append({"mode": "double", "material": mat, "reason": "less than two setups"})
            continue
        if any(not s["has_y_obs"] for s in setups[:2]):
            skipped.append({"mode": "double", "material": mat, "reason": "missing y_obs.npy"})
            continue
        try:
            items = []
            for s in setups[:2]:
                W, H = float(s["W"]), float(s["H"])
                y_obs = np.load(s["path"] / "y_obs.npy").astype(np.float64).ravel()
                items.append(SHA.prepare_setup_shape(s["name"], W, H, y_obs,
                                                       geo_router, xs, ys, device, dtype, args))
        except Exception as exc:
            skipped.append({"mode": "double", "material": mat, "reason": str(exc)})
            continue
        res = SHA.inverse_double_shape(items[0], items[1], args, device, dtype)
        if res is None:
            skipped.append({"mode": "double", "material": mat, "reason": "no admitted members"})
            continue
        rows.append({
            "mode": "double", "material": mat,
            "setup_A": setups[0]["name"], "setup_B": setups[1]["name"],
            "theta_n": res["theta_n"], "theta_eta": res["theta_eta"], "theta_sy": res["theta_sy"],
            "objective": res["objective"], "opt_s": res["opt_s"],
            "sub_ids": res["sub_ids"], "basins": res["basins"],
            "y8_A": float(items[0]["y_obs"][-1]), "y8_B": float(items[1]["y_obs"][-1]),
            "raw_wrms_total": res["raw_wrms_total"],
            "sy_anchor": res.get("sy_anchor", float("nan")),
            "sy_single_a": res.get("sy_single_a", float("nan")),
            "sy_single_b": res.get("sy_single_b", float("nan")),
            "selector_mode": "shape",
        })
        cleanup_runtime()
    return rows, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=PIPELINE_ROOT / "data" / "real_world_experiments")
    ap.add_argument("--out-dir", type=Path, default=PIPELINE_ROOT / "OptimizationResults" / "real_world_batch")
    ap.add_argument("--state-root", type=Path, default=V10.STATE_ROOT)
    ap.add_argument("--materials", default="all",
                    help="comma-separated material names or 'all'")
    ap.add_argument("--mode", choices=["single", "double", "all"], default="all")
    ap.add_argument("--loss-mode", choices=["linear", "log_nuisance"], default="log_nuisance")
    ap.add_argument("--sigma-bias", type=float, default=0.25)
    ap.add_argument("--sigma-trend", type=float, default=0.20)
    ap.add_argument("--sigma-y-min", type=float, default=5.0)
    ap.add_argument("--inverse-mode", choices=["shape"], default="shape",
                    help="Production: 'shape' = CMA-ES + σY two-stage prior (Plan B)")
    SHA.add_argparse_args(ap)
    args = ap.parse_args()

    V10.STATE_ROOT = args.state_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    materials = discover(args.root)
    if args.materials.lower() != "all":
        keep = {m.strip() for m in args.materials.split(",") if m.strip()}
        materials = {m: rows for m, rows in materials.items() if m in keep}
    geo_router, xs, ys = V10.load_runtime()
    device, dtype = V10.HC.DEVICE, V10.HC.DTYPE

    all_skipped = []
    single_rows, double_rows = [], []
    if args.mode in {"single", "all"}:
        single_rows, sk = run_single(materials, geo_router, xs, ys, device, dtype, args)
        all_skipped.extend(sk)
        pd.DataFrame(single_rows).to_csv(args.out_dir / "single_real_world.csv", index=False)
    if args.mode in {"double", "all"}:
        double_rows, sk = run_double(materials, geo_router, xs, ys, device, dtype, args)
        all_skipped.extend(sk)
        pd.DataFrame(double_rows).to_csv(args.out_dir / "double_real_world.csv", index=False)
    pd.DataFrame(all_skipped).to_csv(args.out_dir / "skipped.csv", index=False)
    summary = {
        "root": str(args.root),
        "state_root": str(V10.STATE_ROOT),
        "n_materials": len(materials),
        "n_single": len(single_rows),
        "n_double": len(double_rows),
        "n_skipped": len(all_skipped),
        "single_mean_s": float(np.mean([r["opt_s"] for r in single_rows])) if single_rows else None,
        "double_mean_s": float(np.mean([r["opt_s"] for r in double_rows])) if double_rows else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
