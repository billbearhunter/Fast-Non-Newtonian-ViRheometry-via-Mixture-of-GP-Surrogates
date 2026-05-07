"""Compare the 12-material joint inverse against Hamamichi 2023 Table 1 truth.

Hamamichi 2023 Table 1 reports HB params (η, n, σ_y) for 13 commercial
materials. Their η and σ_y are in units 10× smaller than our surrogate
convention (CGS-style Pa·sⁿ × 10 / Pa × 10), so we multiply by 10 to
match before comparing. n is unitless and unchanged.

For each material we tabulate:
  Hamamichi reference (×10 for η, σ_y)
  our point estimate (n̂, η̂, σ̂_y)
  our 95% Hessian Laplace CI
  identifiability flag
  whether Hamamichi truth lies inside our 95% CI per parameter

Usage:
    python scripts/compare_prior_to_hamamichi.py
    python scripts/compare_prior_to_hamamichi.py --json scripts/run_y8q_prior_joint.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]


# Hamamichi 2023 Table 1, in their published units.
# Multiply η and σ_y by 10 for surrogate-space comparison.
HAMAMICHI = {
    # material_key      :  (eta_h,  n_h,   sy_h)
    "MoisturizingMilk":                       (1.27, 0.87, 14.75),
    # "JapanesePorkCutletSauce" not in our data folder — skipped
    "JapaneseThickenedWorcestershireSauce":   (0.55, 0.81,  1.95),
    "JapaneseCabbagePancakeSauce":            (2.62, 0.76, 11.58),
    "Lotion":                                 (9.76, 0.42,  1.46),
    "SweetBeanPaste":                         (10.85, 0.75, 33.46),
    "Mustard":                                (4.94, 0.85, 27.65),
    "Island":                                 (1.49, 0.82, 13.48),  # Thousand island dressing
    "Cobb":                                   (1.08, 0.87,  5.67),  # Cobb salad dressing
    "Sesame":                                 (0.49, 1.00,  1.93),  # Sesame dressing
    "Pomodoro":                               (4.28, 0.46, 16.98),
    "Carbonara":                              (7.29, 1.00,  1.52),
    "Congee":                                 (18.17, 0.50, 22.90),
}


def in_ci(val: float, ci: list[float] | None) -> str:
    if ci is None:
        return "?"
    return "✓" if (ci[0] <= val <= ci[1]) else "✗"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path,
                    default=REPO / "scripts" / "run_y8q_prior_joint.json")
    ap.add_argument("--md-out", type=Path, default=None,
                    help="If set, write a markdown table here.")
    cli = ap.parse_args()

    summary = json.loads(cli.json.read_text())

    rows = []
    n_in_ci = {"n": 0, "eta": 0, "sigma_y": 0}
    n_total = 0
    for mat, (eta_h, n_h, sy_h) in HAMAMICHI.items():
        d = summary.get(mat)
        if d is None:
            print(f"[skip] {mat} not in JSON")
            continue
        joint = d.get("joint")
        if joint is None or "error" in (joint or {}):
            print(f"[skip] {mat} joint failed")
            continue
        n_hat, eta_hat, sy_hat = joint["theta"]
        ci = joint.get("theta_ci_95") or {}
        idf = joint.get("identifiable") or {}
        # Hamamichi reference in surrogate units (×10 for η and σ_y)
        eta_ref = eta_h * 10.0
        sy_ref = sy_h * 10.0
        n_ref = n_h

        n_status = in_ci(n_ref, ci.get("n"))
        e_status = in_ci(eta_ref, ci.get("eta"))
        s_status = in_ci(sy_ref, ci.get("sigma_y"))
        n_in_ci["n"] += int(n_status == "✓")
        n_in_ci["eta"] += int(e_status == "✓")
        n_in_ci["sigma_y"] += int(s_status == "✓")
        n_total += 1

        rows.append({
            "mat": mat,
            "n_ref": n_ref, "eta_ref": eta_ref, "sy_ref": sy_ref,
            "n_hat": n_hat, "eta_hat": eta_hat, "sy_hat": sy_hat,
            "n_ci": ci.get("n"), "eta_ci": ci.get("eta"), "sy_ci": ci.get("sigma_y"),
            "n_id": bool(idf.get("n")), "eta_id": bool(idf.get("eta")),
            "sy_id": bool(idf.get("sigma_y")),
            "n_in": n_status, "eta_in": e_status, "sy_in": s_status,
        })

    # ----- console table -----
    print()
    print(f"{'Material':<40} | {'  n_ref/n̂  ':<22} | {'   η_ref/η̂  ':<24} | {'   σ_y_ref/σ̂_y  ':<26}")
    print("-" * 120)
    for r in rows:
        n_cell = f"{r['n_ref']:.2f}/{r['n_hat']:.2f} {r['n_in']}"
        e_cell = f"{r['eta_ref']:.1f}/{r['eta_hat']:.1f} {r['eta_in']}"
        s_cell = f"{r['sy_ref']:.1f}/{r['sy_hat']:.1f} {r['sy_in']}"
        print(f"{r['mat']:<40} | {n_cell:<22} | {e_cell:<24} | {s_cell:<26}")
    print("-" * 120)
    print(f"{'In-CI rate (Hamamichi truth ∈ ours 95% CI)':<40} | "
          f"  n: {n_in_ci['n']}/{n_total}  η: {n_in_ci['eta']}/{n_total}  "
          f"σ_y: {n_in_ci['sigma_y']}/{n_total}")

    # ----- optional markdown -----
    if cli.md_out is not None:
        lines = []
        lines.append("| Material | n_Hama | η_Hama×10 | σ_y_Hama×10 |"
                     " n̂ | η̂ | σ̂_y |"
                     " n in CI? | η in CI? | σ_y in CI? |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|")
        for r in rows:
            lines.append(
                f"| {r['mat']} | {r['n_ref']:.2f} | {r['eta_ref']:.1f} | {r['sy_ref']:.1f} |"
                f" {r['n_hat']:.3f} | {r['eta_hat']:.2f} | {r['sy_hat']:.2f} |"
                f" {r['n_in']} | {r['eta_in']} | {r['sy_in']} |"
            )
        lines.append("")
        lines.append(f"In-CI rate: n {n_in_ci['n']}/{n_total},"
                     f" η {n_in_ci['eta']}/{n_total},"
                     f" σ_y {n_in_ci['sigma_y']}/{n_total}")
        cli.md_out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSaved markdown to {cli.md_out}")


if __name__ == "__main__":
    main()
