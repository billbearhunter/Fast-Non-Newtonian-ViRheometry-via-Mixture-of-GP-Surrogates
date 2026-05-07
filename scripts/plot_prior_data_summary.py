"""Render a 4×3 grid of prior-data joint-inverse flow curves.

Reads ``scripts/run_y8q_prior_joint.json`` (or any file with the same schema)
and produces a paper-figure overview of HB flow curves recovered for the
12 Hamamichi 2023 prior materials. Each panel shows:
  * Solid line: HB curve at point estimate (n̂, η̂, σ̂_y)
  * Shaded band: HB curve evaluated at σ_y CI 95% bounds (visualises σ_y
    uncertainty; (n, η) bands are too wide to be useful per HB-ridge
    analysis — see DISCUSSION_2026-05-07 §1).
  * sub_id and identifiability flags annotated.

CGS convention follows ``FlowCurve/flowcurve.py``: η and σ_y are stored in
units that are 10× SI Pa·sⁿ / 10× SI Pa, so we multiply by 0.1 before
plotting σ_s in Pa.

Usage:
    python scripts/plot_prior_data_summary.py
    python scripts/plot_prior_data_summary.py --json scripts/run_y8q_prior_joint.json \
        --out FlowCurve/figs/prior_data_summary.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO = Path(__file__).resolve().parents[1]


def hb_curve(gamma_dot: np.ndarray, n: float, eta: float, sy: float) -> np.ndarray:
    """HB σ_s = (η/10) · γ̇^n + σ_y/10  (matches FlowCurve/flowcurve.calcFlowCurve)"""
    return (eta * 0.1) * np.power(gamma_dot, n) + (sy * 0.1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path,
                    default=REPO / "scripts" / "run_y8q_prior_joint.json")
    ap.add_argument("--out", type=Path,
                    default=REPO / "FlowCurve" / "figs" / "prior_data_summary.png")
    cli = ap.parse_args()

    summary = json.loads(cli.json.read_text())
    materials = list(summary.keys())
    n_mat = len(materials)
    ncols = 4
    nrows = (n_mat + ncols - 1) // ncols

    plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["font.size"] = 11
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows),
                              sharex=True, sharey=False)
    axes = axes.flatten() if nrows > 1 else axes
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    gamma = np.logspace(0, 2, 200)  # γ̇ ∈ [1, 100] s⁻¹

    for i, mat in enumerate(materials):
        ax = axes[i]
        d = summary[mat]
        joint = d.get("joint")
        if joint is None or "error" in (joint or {}):
            ax.text(0.5, 0.5, f"{mat}\n(joint failed)", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(mat, fontsize=10)
            continue
        n, eta, sy = joint["theta"]
        ci = joint.get("theta_ci_95", {}) or {}
        sy_ci = ci.get("sigma_y") if isinstance(ci, dict) else None
        idf = joint.get("identifiable") or {}

        y_pt = hb_curve(gamma, n, eta, sy)
        ax.plot(gamma, y_pt, color="C0", linewidth=2.0, zorder=3,
                label=fr"$\hat\theta$: n={n:.2f}, η={eta:.1f}, σ_y={sy:.1f} Pa")
        if sy_ci is not None and isinstance(sy_ci, list) and len(sy_ci) == 2:
            y_lo = hb_curve(gamma, n, eta, sy_ci[0])
            y_hi = hb_curve(gamma, n, eta, sy_ci[1])
            ax.fill_between(gamma, y_lo, y_hi, color="C0", alpha=0.18, zorder=1,
                            label=fr"σ_y 95% CI [{sy_ci[0]:.1f}, {sy_ci[1]:.1f}]")

        flags = ("n" if idf.get("n") else "·") + \
                ("η" if idf.get("eta") else "·") + \
                ("σ_y" if idf.get("sigma_y") else "·")
        sub_a = d["per_setup"]["setup1"]["sub_id"]
        sub_b = d["per_setup"]["setup2"]["sub_id"] if "setup2" in d["per_setup"] else None
        sub_str = f"sub_{sub_a}+{sub_b}" if sub_b is not None else f"sub_{sub_a}"
        ax.set_title(f"{mat}\n{sub_str}, id=[{flags}]", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 100)
        ax.set_ylim(0.5, 200)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")

    # Hide unused axes
    for j in range(n_mat, len(axes)):
        axes[j].axis("off")

    # Common labels
    for ax in axes[:n_mat]:
        ax.set_xlabel(r"$\dot\gamma\ [\mathrm{s^{-1}}]$", fontsize=10)
        ax.set_ylabel(r"$\sigma_s\ [\mathrm{Pa}]$", fontsize=10)

    fig.suptitle(f"Prior data (Hamamichi 2023, n={n_mat} materials) — "
                 f"camera-joint y8q paper-grade inverse",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cli.out, dpi=150)
    print(f"Saved {cli.out}")


if __name__ == "__main__":
    main()
