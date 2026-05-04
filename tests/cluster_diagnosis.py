"""Diagnose which cluster (expert) v2 routes each real material to,
and whether that cluster's underperformance is a data-coverage or a
model-training issue.

For every folder in old_result/ we:
  1. Parse ref/2setup/settings.xml → geometry (W, H).
  2. Fit HB to the rheometer CSV in ref/2setup/ → (η*, n*, σy*) ground truth.
  3. Read param_data.csv last row → old-method converged (η, n, σy).
  4. Forward-predict surrogate y at (θ_old, W, H) → proxy y_obs (closest
     available synthetic match to the real video observations).
  5. Route (W, H, y_proxy) through the frozen 2-level BGM gate → expert_id.
  6. Inspect: expert RMSE/MAE/NMSE (from per_expert_errors.csv); expert's
     training bounding box; whether rheometer truth θ* falls INSIDE that
     box (=> model/training issue) or OUTSIDE (=> data gap).

Internal unit convention: our surrogate uses η, σy scaled 10× compared to
physical MKS. Rheometer returns MKS. Converting: η_internal = η_rheo × 10,
σy_internal = σy_rheo × 10.
"""
from __future__ import annotations
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from surrogate import config as HC
from surrogate.predict import HVIMoGPrBCMPredictor
from FlowCurve.hb_fit import hb_model  # noqa
from scipy.optimize import curve_fit


def _fit_hb(csv_path: Path, idx_start: int = 5, idx_end: int = 19):
    """Adaptive Anton Paar CSV parser: scan header candidates for '[1/s]'.

    Pandas's default skip-blank-lines behaviour means the pandas-row-index
    of the unit-symbol header differs from the file-line index. Try
    header=5..12 until we find the row that has '[1/s]' as a column.
    """
    df = None
    hdr = None
    for h in range(4, 15):
        try:
            cand = pd.read_table(csv_path, header=h, encoding="UTF-16")
            if "[1/s]" in cand.columns and "[Pa]" in cand.columns:
                df = cand; hdr = h
                break
        except Exception:
            continue
    if df is None:
        raise ValueError(f"could not locate [1/s] header in {csv_path}")
    x = df["[1/s]"].iloc[idx_start:idx_end].astype(float)
    y = df["[Pa]"].iloc[idx_start:idx_end].astype(float)
    popt, _ = curve_fit(hb_model, x, y,
                        p0=[1.0, 0.5, float(y.iloc[0]) * 0.5],
                        bounds=([0, 0, 0], [np.inf, 1.0, np.inf]),
                        maxfev=10000)
    K, n, sy = popt
    return {"eta_rheo": float(K), "n_rheo": float(n), "sigma_y_rheo": float(sy),
            "header_row": hdr}


def _parse_setup_xml(path: Path):
    tree = ET.parse(path)
    setup = tree.find(".//setup")
    return float(setup.attrib["W"]), float(setup.attrib["H"])


def _find_rheo_csv(setup_dir: Path) -> Path | None:
    # Try any .csv that is NOT experiment_log.csv
    for p in setup_dir.iterdir():
        if p.suffix == ".csv" and p.name != "experiment_log.csv":
            return p
    return None


def _find_rheo_csv_by_name(material_name: str, rheo_root: Path) -> Path | None:
    """Fallback: find rheometer CSV in FlowCurve/Rheo_Data/ by material name prefix.

    Prefer CSVs with '2026' in the date (more recent). Case-insensitive.
    """
    base = material_name.rstrip("_0123456789").lower()  # e.g. Tonkatsu_2 -> tonkatsu
    cands = [p for p in rheo_root.glob("*.csv")
             if p.name.lower().startswith(base)]
    if not cands:
        return None
    # Prefer those with '2026' in the filename (newer measurements)
    recent = [p for p in cands if "2026" in p.name]
    return (recent[0] if recent else cands[0])


def _box_contains(box: dict, theta: np.ndarray) -> dict:
    """Return per-axis containment flags for (n, η, σy) — input in INTERNAL units."""
    n, eta, sy = float(theta[0]), float(theta[1]), float(theta[2])
    return {
        "in_n":   box["n"][0]       <= n    <= box["n"][1],
        "in_eta": box["eta"][0]     <= eta  <= box["eta"][1],
        "in_sy":  box["sigma_y"][0] <= sy   <= box["sigma_y"][1],
    }


def analyse_one(material_dir: Path, predictor: HVIMoGPrBCMPredictor,
                per_expert_err: pd.DataFrame):
    tag = material_dir.name
    print(f"\n{'='*78}")
    print(f"{tag}")
    print('='*78)

    # ── 1. geometry — try 2setup first, else 1setup ───────────────────
    xml2 = material_dir / "ref" / "2setup" / "settings.xml"
    xml1 = material_dir / "ref" / "1setup" / "settings.xml"
    setup_used = None
    if xml2.exists():
        W, H = _parse_setup_xml(xml2); setup_used = "2setup"
    elif xml1.exists():
        W, H = _parse_setup_xml(xml1); setup_used = "1setup"
    else:
        print(f"  skip (no settings.xml)"); return None
    print(f"  Using {setup_used} geometry: W = {W}  H = {H}")

    # ── 2. rheometer truth (fit HB to Anton Paar CSV) ──────────────────
    rheo_csv = _find_rheo_csv(material_dir / "ref" / setup_used)
    if rheo_csv is None:
        # fallback: search FlowCurve/Rheo_Data by material-name prefix
        rheo_root = REPO / "FlowCurve" / "Rheo_Data"
        rheo_csv = _find_rheo_csv_by_name(material_dir.name, rheo_root)
    if rheo_csv is None:
        print(f"  skip (no rheometer CSV for {material_dir.name})"); return None
    print(f"  Rheometer CSV: {rheo_csv.name}")
    try:
        rheo = _fit_hb(rheo_csv)
    except Exception as e:
        print(f"  [err] HB fit failed: {e}"); return None
    # Convert MKS -> internal (×10 for η and σy)
    eta_truth_internal = rheo["eta_rheo"] * 10.0
    sy_truth_internal  = rheo["sigma_y_rheo"] * 10.0
    n_truth            = rheo["n_rheo"]
    theta_truth = np.array([n_truth, eta_truth_internal, sy_truth_internal])
    print(f"  Rheometer truth (MKS):      η = {rheo['eta_rheo']:.3f} Pa·s   "
          f"n = {n_truth:.4f}   σ_y = {rheo['sigma_y_rheo']:.3f} Pa")
    print(f"  Rheometer truth (internal): η = {eta_truth_internal:.3f}   "
          f"n = {n_truth:.4f}   σ_y = {sy_truth_internal:.3f}")

    # ── 3. old-method converged params ─────────────────────────────────
    pd_csv = material_dir / "param_data.csv"
    if pd_csv.exists():
        last = pd.read_csv(pd_csv, header=None).iloc[-1].values.astype(float)
        # header-less; cols are (η, n, σy) per old-method convention
        eta_old, n_old, sy_old = float(last[0]), float(last[1]), float(last[2])
        theta_old = np.array([n_old, eta_old, sy_old])
        print(f"  Old-method final (internal): η = {eta_old:.3f}   "
              f"n = {n_old:.4f}   σ_y = {sy_old:.3f}")
    else:
        theta_old = None

    # ── 4. proxy y_obs via surrogate forward @ old-method's θ ──────────
    # Using old-method's θ as the "best available target for this video" —
    # surrogate's forward at that point gives the y most representative of
    # what v2 would see as observation.
    if theta_old is not None:
        X = np.array([[n_old, eta_old, sy_old, W, H]], dtype=np.float64)
        y_proxy = predictor.predict(X_raw=X, W=np.array([W]), H=np.array([H]),
                                     y_obs=None, top_k_phi=1,
                                     use_baseline=False, clear_cache=True)[0]
    else:
        # fallback: use truth
        X = np.array([[n_truth, eta_truth_internal, sy_truth_internal, W, H]], dtype=np.float64)
        y_proxy = predictor.predict(X_raw=X, W=np.array([W]), H=np.array([H]),
                                     y_obs=None, top_k_phi=1,
                                     use_baseline=False, clear_cache=True)[0]
    print(f"  Surrogate y_proxy: {np.array2string(y_proxy, precision=3)}")

    # ── 5. route through frozen v2 gate ─────────────────────────────────
    geo_id, eids, wts = predictor.route_for(W, H, y_proxy, top_k_phi=1)
    eid = int(eids[0])
    print(f"  Routed: geo = {geo_id}  expert = {eid}  p_top1 = {wts[0]:.4f}")

    # ── 6. expert performance ──────────────────────────────────────────
    row = per_expert_err[per_expert_err.expert_id == eid]
    if len(row) == 0:
        print(f"  [warn] expert {eid} not in per-expert error table")
        mae = rmse = n_test = None
    else:
        r = row.iloc[0]
        mae  = float(r.MAE)
        rmse = float(r.RMSE)
        n_test = int(r.n_test)
        print(f"  Expert {eid}: MAE = {mae:.4f} cm   RMSE = {rmse:.4f} cm   "
              f"n_test = {n_test}   training_N = {int(r.cluster_size)}")

    # ── 7. expert box & truth containment ──────────────────────────────
    boxes = predictor.expert_boxes_physical()
    box = boxes[eid]
    print(f"  Expert training box:")
    print(f"    n   ∈ [{box['n'][0]:.3f}, {box['n'][1]:.3f}]")
    print(f"    η   ∈ [{box['eta'][0]:.3f}, {box['eta'][1]:.3f}]")
    print(f"    σ_y ∈ [{box['sigma_y'][0]:.3f}, {box['sigma_y'][1]:.3f}]")

    cont_truth = _box_contains(box, theta_truth)
    cont_old   = _box_contains(box, theta_old) if theta_old is not None else None
    print(f"  Truth in box?  n={cont_truth['in_n']}  η={cont_truth['in_eta']}  σy={cont_truth['in_sy']}")
    if cont_old:
        print(f"  Old-θ in box?  n={cont_old['in_n']}  η={cont_old['in_eta']}  σy={cont_old['in_sy']}")

    # ── 8. verdict ──────────────────────────────────────────────────────
    truth_all_in = cont_truth['in_n'] and cont_truth['in_eta'] and cont_truth['in_sy']
    if not truth_all_in:
        verdict = "DATA GAP — rheometer truth is outside the routed expert's training box"
    elif mae is not None and mae > 0.05:
        verdict = f"MODEL LIMIT — truth inside box, but expert's MAE ({mae:.3f}) is above typical"
    else:
        verdict = "no obvious issue — expert trained well and covers truth"
    print(f"\n  >>> {verdict}")

    return dict(
        tag=tag, W=W, H=H,
        eta_rheo_mks=rheo["eta_rheo"], n_rheo=n_truth, sy_rheo_mks=rheo["sigma_y_rheo"],
        eta_truth_int=eta_truth_internal, sy_truth_int=sy_truth_internal,
        eta_old=theta_old[1] if theta_old is not None else None,
        n_old=theta_old[0] if theta_old is not None else None,
        sy_old=theta_old[2] if theta_old is not None else None,
        expert_id=eid, geo_id=int(geo_id),
        expert_MAE=mae, expert_RMSE=rmse, expert_n_test=n_test,
        box_n=box["n"], box_eta=box["eta"], box_sy=box["sigma_y"],
        truth_in_n=cont_truth["in_n"],
        truth_in_eta=cont_truth["in_eta"],
        truth_in_sy=cont_truth["in_sy"],
        verdict=verdict,
    )


def main():
    # Load v2 surrogate
    print(f"[load] {HC.DEFAULT_V2_MODEL}")
    predictor = HVIMoGPrBCMPredictor.load(HC.DEFAULT_V2_MODEL)

    # Latest per-expert error report
    per_expert_csvs = sorted(glob(
        str(REPO / "tests/per_expert_analysis/per_expert_*/per_expert_errors.csv")))
    if not per_expert_csvs:
        raise SystemExit("Run tests/per_expert_errors.py first.")
    per_expert = pd.read_csv(per_expert_csvs[-1])
    print(f"[load] per-expert errors: {per_expert_csvs[-1]}  ({len(per_expert)} experts)")

    # Iterate all old_result subdirs
    old_root = REPO / "old_result"
    material_dirs = [d for d in sorted(old_root.iterdir())
                     if d.is_dir() and (d / "ref").exists()]
    print(f"[info] materials: {[d.name for d in material_dirs]}")

    results = []
    for md in material_dirs:
        r = analyse_one(md, predictor, per_expert)
        if r is not None:
            results.append(r)

    # Summary table
    if results:
        out = pd.DataFrame(results)
        csv_out = REPO / "tests" / "cluster_diagnosis.csv"
        out.to_csv(csv_out, index=False, float_format="%.4g")
        print(f"\n[saved] {csv_out}")


if __name__ == "__main__":
    main()
