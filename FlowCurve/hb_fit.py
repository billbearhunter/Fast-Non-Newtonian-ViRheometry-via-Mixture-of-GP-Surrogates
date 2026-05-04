"""
hb_fit.py  –  Herschel-Bulkley fitting for Anton Paar rheometer data

Usage:
    python hb_fit.py --file sample.csv
    python hb_fit.py --file sample.csv --range 5 19
    python hb_fit.py --file sample.csv --out params.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def hb_model(gamma_dot, K, n, sigma_y):
    """Herschel-Bulkley: σ = K * γ̇^n + σ_Y"""
    return K * np.power(gamma_dot, n) + sigma_y


def fit_hb(gamma_dot, sigma, idx_start, idx_end):
    x = gamma_dot[idx_start:idx_end]
    y = sigma[idx_start:idx_end]

    popt, pcov = curve_fit(
        hb_model, x, y,
        p0=[1.0, 0.5, y.iloc[0] * 0.5],
        bounds=([0, 0, 0], [np.inf, 1, np.inf]),
        maxfev=10000,
    )
    perr = np.sqrt(np.diag(pcov))
    K, n, sigma_y = popt

    y_pred = hb_model(x, *popt)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    print("=" * 40)
    print("  Herschel-Bulkley Fit Result")
    print("=" * 40)
    print(f"  K      = {K:.4f} ± {perr[0]:.4f}  [Pa·s^n]")
    print(f"  n      = {n:.4f} ± {perr[1]:.4f}  [-]")
    print(f"  σ_Y    = {sigma_y:.4f} ± {perr[2]:.4f}  [Pa]")
    print(f"  R²     = {r2:.6f}")
    print(f"  Range  : index {idx_start}–{idx_end-1}  "
          f"(γ̇ = {x.iloc[0]:.3g} – {x.iloc[-1]:.3g} s⁻¹)")
    print("=" * 40)

    return {
        "eta": float(K),
        "n": float(n),
        "sigma_y": float(sigma_y),
        "r2": float(r2),
        "eta_err": float(perr[0]),
        "n_err": float(perr[1]),
        "sigma_y_err": float(perr[2]),
    }


def main():
    parser = argparse.ArgumentParser(description="HB fitting for Anton Paar CSV data.")
    parser.add_argument("--file",  required=True, help="Input CSV file path")
    parser.add_argument("--range", nargs=2, type=int, default=[5, 19],
                        metavar=("START", "END"),
                        help="Row index range for fitting (default: 5 19)")
    parser.add_argument("--out", default=None,
                        help="Save fitted parameters to JSON file")
    args = parser.parse_args()

    # Auto-detect header row (different export templates differ); same logic as flowcurve.py
    df = None
    for h in range(0, 12):
        try:
            cand = pd.read_table(args.file, header=h, encoding="UTF-16")
            if '[1/s]' in cand.columns and '[Pa]' in cand.columns:
                df = cand
                break
        except Exception:
            continue
    if df is None:
        raise RuntimeError(f"Could not locate header row with '[1/s]' and '[Pa]' in {args.file}")
    # Drop unit/comment rows that aren't numeric
    df = df[pd.to_numeric(df['[1/s]'], errors='coerce').notna()].reset_index(drop=True)
    df['[1/s]'] = df['[1/s]'].astype(float)
    df['[Pa]'] = df['[Pa]'].astype(float)
    result = fit_hb(df['[1/s]'], df['[Pa]'], args.range[0], args.range[1])

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()