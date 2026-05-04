from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.cluster_diagnosis import _fit_hb  # noqa: E402


RHEO_ROOT = REPO / "FlowCurve" / "Rheo_Data"
DEFAULT_RESULTS = PIPELINE_ROOT / "OptimizationResults" / "real_world_batch_combined"


# Hamamichi et al. 2023, Table 1.  Values are the paper's physical
# Herschel-Bulkley parameters in the order eta, n, sigma_y.
PAPER_TABLE1 = {
    "MoisturizingMilk": {
        "paper_name": "Moisturizing milk",
        "setup1": (2.9, 5.9),
        "setup2": (3.2, 2.0),
        "eta": 1.27,
        "n": 0.87,
        "sigma_y": 14.75,
    },
    "JapanesePorkCutletSauce": {
        "paper_name": "Japanese pork cutlet sauce (Tonkatsu sauce)",
        "setup1": (6.0, 5.8),
        "setup2": (3.2, 2.0),
        "eta": 4.02,
        "n": 0.55,
        "sigma_y": 1.60,
    },
    "JapaneseThickenedWorcestershireSauce": {
        "paper_name": "Japanese thickened Worcestershire sauce (Chuno sauce)",
        "setup1": (2.1, 4.0),
        "setup2": (3.0, 2.0),
        "eta": 0.55,
        "n": 0.81,
        "sigma_y": 1.95,
    },
    "JapaneseCabbagePancakeSauce": {
        "paper_name": "Japanese cabbage pancake sauce (Okonomi sauce)",
        "setup1": (4.2, 3.8),
        "setup2": (2.9, 2.0),
        "eta": 2.62,
        "n": 0.76,
        "sigma_y": 11.58,
    },
    "Lotion": {
        "paper_name": "Lotion",
        "setup1": (5.1, 2.5),
        "setup2": (7.0, 7.0),
        "eta": 9.76,
        "n": 0.42,
        "sigma_y": 1.46,
    },
    "SweetBeanPaste": {
        "paper_name": "Sweet bean paste",
        "setup1": (4.8, 4.1),
        "setup2": (2.1, 2.0),
        "eta": 10.85,
        "n": 0.75,
        "sigma_y": 33.46,
    },
    "Mustard": {
        "paper_name": "Mustard",
        "setup1": (3.3, 6.6),
        "setup2": (2.3, 2.0),
        "eta": 4.94,
        "n": 0.85,
        "sigma_y": 27.65,
    },
    "Island": {
        "paper_name": "Thousand island dressing",
        "setup1": (3.3, 3.6),
        "setup2": (2.9, 2.0),
        "eta": 1.49,
        "n": 0.82,
        "sigma_y": 13.48,
    },
    "Cobb": {
        "paper_name": "Cobb salad dressing",
        "setup1": (5.5, 4.7),
        "setup2": (2.9, 2.0),
        "eta": 1.08,
        "n": 0.87,
        "sigma_y": 5.67,
    },
    "Sesame": {
        "paper_name": "Sesame dressing",
        "setup1": (4.0, 4.2),
        "setup2": (2.9, 2.0),
        "eta": 0.49,
        "n": 1.00,
        "sigma_y": 1.93,
    },
    "Pomodoro": {
        "paper_name": "Pomodoro sauce",
        "setup1": (3.2, 2.1),
        "setup2": (7.0, 7.0),
        "eta": 4.28,
        "n": 0.46,
        "sigma_y": 16.98,
    },
    "Carbonara": {
        "paper_name": "Carbonara sauce",
        "setup1": (6.4, 3.7),
        "setup2": (2.5, 2.0),
        "eta": 7.29,
        "n": 1.00,
        "sigma_y": 1.52,
    },
    "Congee": {
        "paper_name": "Congee",
        "setup1": (6.4, 5.7),
        "setup2": (2.1, 2.0),
        "eta": 18.17,
        "n": 0.50,
        "sigma_y": 22.90,
    },
}


# Prefer the paper-era 2023 rheometer CSVs for the materials explicitly
# described as rheometer-valid in Section 7.2.  SweetBeanPaste only has a
# 2026 Sweet CSV in this workspace, so it intentionally falls back to
# Hamamichi Table 1 unless a 2023 CSV is added later.
RHEO_MATCH = {
    "MoisturizingMilk": ("Nyueki_20230114_1259_23C.csv", "high", "Nyueki is the local Japanese label for moisturizing milk/emulsion."),
    "JapanesePorkCutletSauce": ("tonkatsu_20230113_2000_23C.csv", "high", "Paper synonym: Tonkatsu sauce."),
    "JapaneseThickenedWorcestershireSauce": ("Chuno_20230114_1458_23C.csv", "high", "Paper synonym: Chuno sauce."),
    "JapaneseCabbagePancakeSauce": ("Okonomiyaki_2023_0114_1627_23C.csv", "high", "Paper synonym: Okonomi sauce."),
    "Lotion": ("Lotion_20230114_1204_23C.csv", "high", "Exact local family match."),
}


def to_internal(n: float, eta_phys: float, sy_phys: float) -> tuple[float, float, float]:
    return float(n), float(eta_phys) * 10.0, float(sy_phys) * 10.0


def fc_rmse_pct(theta_hat_internal, theta_truth_internal, gamma_dot=None) -> float:
    if gamma_dot is None:
        gamma_dot = np.logspace(-2, 2, 50)
    th = np.asarray(theta_hat_internal, dtype=float)
    tt = np.asarray(theta_truth_internal, dtype=float)
    stress_hat = (th[1] * 0.1) * np.power(gamma_dot, th[0]) + (th[2] * 0.1)
    stress_truth = (tt[1] * 0.1) * np.power(gamma_dot, tt[0]) + (tt[2] * 0.1)
    return float(np.sqrt(np.mean(((stress_hat - stress_truth) / stress_truth) ** 2)) * 100.0)


def build_truth_rows() -> list[dict]:
    rows: list[dict] = []
    for material, paper in PAPER_TABLE1.items():
        paper_internal = to_internal(paper["n"], paper["eta"], paper["sigma_y"])
        row = {
            "material": material,
            "paper_name": paper["paper_name"],
            "paper_setup1_W": paper["setup1"][0],
            "paper_setup1_H": paper["setup1"][1],
            "paper_setup2_W": paper["setup2"][0],
            "paper_setup2_H": paper["setup2"][1],
            "hamamichi_table_eta_phys": paper["eta"],
            "hamamichi_table_n": paper["n"],
            "hamamichi_table_sigma_y_phys": paper["sigma_y"],
            "hamamichi_table_eta_internal": paper_internal[1],
            "hamamichi_table_sigma_y_internal": paper_internal[2],
            "truth_source": "hamamichi_table1",
            "truth_confidence": "fallback",
            "truth_n": paper_internal[0],
            "truth_eta": paper_internal[1],
            "truth_sy": paper_internal[2],
            "rheo_csv": "",
            "rheo_eta_phys": np.nan,
            "rheo_n": np.nan,
            "rheo_sigma_y_phys": np.nan,
            "rheo_header_row": np.nan,
            "fc_rheo_vs_hamamichi_pct": np.nan,
            "notes": "No matching local rheometer CSV assigned; using Hamamichi Table 1 as temporary truth.",
        }
        if material in RHEO_MATCH:
            csv_name, confidence, notes = RHEO_MATCH[material]
            csv_path = RHEO_ROOT / csv_name
            fit = _fit_hb(csv_path)
            rheo_internal = to_internal(fit["n_rheo"], fit["eta_rheo"], fit["sigma_y_rheo"])
            row.update(
                {
                    "truth_source": "rheometer_csv_fit",
                    "truth_confidence": confidence,
                    "truth_n": rheo_internal[0],
                    "truth_eta": rheo_internal[1],
                    "truth_sy": rheo_internal[2],
                    "rheo_csv": str(csv_path),
                    "rheo_eta_phys": fit["eta_rheo"],
                    "rheo_n": fit["n_rheo"],
                    "rheo_sigma_y_phys": fit["sigma_y_rheo"],
                    "rheo_header_row": fit["header_row"],
                    "fc_rheo_vs_hamamichi_pct": fc_rmse_pct(paper_internal, rheo_internal),
                    "notes": notes,
                }
            )
        rows.append(row)
    return rows


def attach_truth(result_csv: Path, truth: pd.DataFrame, out_csv: Path) -> None:
    if not result_csv.exists():
        return
    df = pd.read_csv(result_csv)
    merged = df.merge(
        truth[
            [
                "material",
                "truth_source",
                "truth_confidence",
                "truth_n",
                "truth_eta",
                "truth_sy",
                "paper_name",
                "rheo_csv",
                "notes",
            ]
        ],
        on="material",
        how="left",
    )
    fc = []
    for _, r in merged.iterrows():
        if pd.isna(r["truth_n"]):
            fc.append(np.nan)
        else:
            fc.append(
                fc_rmse_pct(
                    (r["theta_n"], r["theta_eta"], r["theta_sy"]),
                    (r["truth_n"], r["truth_eta"], r["truth_sy"]),
                )
            )
    merged["fc_pct_vs_assigned_truth"] = fc
    merged.to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    truth = pd.DataFrame(build_truth_rows()).sort_values("material")
    truth_csv = args.results_dir / "truth_map.csv"
    truth.to_csv(truth_csv, index=False)
    (args.results_dir / "truth_map.json").write_text(
        json.dumps(truth.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    attach_truth(
        args.results_dir / "double_real_world.csv",
        truth,
        args.results_dir / "double_real_world_with_truth.csv",
    )
    attach_truth(
        args.results_dir / "single_real_world.csv",
        truth,
        args.results_dir / "single_real_world_with_truth.csv",
    )

    print(f"wrote {truth_csv}")
    dbl = args.results_dir / "double_real_world_with_truth.csv"
    if dbl.exists():
        df = pd.read_csv(dbl)
        cols = [
            "material",
            "theta_n",
            "theta_eta",
            "theta_sy",
            "truth_source",
            "truth_confidence",
            "truth_n",
            "truth_eta",
            "truth_sy",
            "fc_pct_vs_assigned_truth",
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
