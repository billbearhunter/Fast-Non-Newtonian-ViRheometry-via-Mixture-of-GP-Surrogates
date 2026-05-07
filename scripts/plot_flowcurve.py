"""Plot Herschel-Bulkley flow curve overlays in `FlowCurve/flowcurve.py` style.

Reads a θ̂-summary JSON (produced by run_y8q_chuno.py or any compatible script)
and the rheometer CSV, then produces a 3-line plot:
  - black dotted: rheometer measurement
  - red:         rheometer truth fit
  - blue:        recovered θ̂ (default: joint y8q)

Usage:
    python scripts/plot_flowcurve.py
    python scripts/plot_flowcurve.py \
        --json scripts/run_y8q_chuno.json \
        --rheo FlowCurve/Rheo_Data/Chuno_20230114_1523_25C.csv \
        --out FlowCurve/figs/y8q_chuno_flowcurve.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Make FlowCurve/ importable so we can reuse `param.py` and the plotting
# convention (eta, sigmaY divided by 10) without copying it.
sys.path.insert(0, str(REPO / "FlowCurve"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from param import Param  # noqa: E402  (FlowCurve/param.py)


# Color order matches FlowCurve/flowcurve.py
COLORS = [(255/255, 0/255, 0/255), (0/255, 0/255, 255/255), (208/255, 2/255, 4/255)]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 30


# Chuno rheometer truth (25 °C fit, surrogate-space units)
TRUTH_CHUNO = (0.633, 10.51, 19.62)  # (n, η, σ_y); divide last two by 10 for Pa


def calc_flow_curve(p: Param, x: np.ndarray) -> np.ndarray:
    """HB flow curve.  Matches FlowCurve/flowcurve.py exactly:
    surrogate (n, η, σ_y) → physical: η_phys = η/10, σ_y_phys = σ_y/10."""
    eta = p.eta * 0.1
    sigmaY = p.sigmaY * 0.1
    return eta * np.power(x, p.n) + sigmaY


def read_rheo(file: Path) -> pd.DataFrame:
    """Auto-locate header row (the rheometer export template varies)."""
    for h in range(0, 12):
        try:
            df = pd.read_table(file, header=h, encoding="UTF-16")
            if "[1/s]" in df.columns and "[Pa]" in df.columns:
                return df
        except Exception:
            continue
    raise RuntimeError(f"Could not locate header row in {file}")


def plot_flow_curve(rheo_csv: Path, params: list[Param], out_path: Path,
                    extent_y: tuple[float, float] = (1e0, 1e2)):
    df = read_rheo(rheo_csv)
    fig, ax = plt.subplots()
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(10**0, 10**2); plt.ylim(*extent_y)

    ax.plot(df["[1/s]"], df["[Pa]"], linestyle="dotted", linewidth=3.0, color="black")
    x = np.linspace(df["[1/s]"][5], df["[1/s]"][18], 10000)
    for i, p in enumerate(params):
        y = calc_flow_curve(p, x)
        ax.plot(x, y, color=COLORS[i % len(COLORS)], linewidth=3.0)

    ax.set_xlabel(r"$\dot{\gamma}[s^{-1}]$", fontsize=31, labelpad=1.8)
    ax.set_ylabel(r"$\sigma_s[Pa]$", fontsize=31, labelpad=1.8)
    ax.grid()
    plt.subplots_adjust(left=0.19, right=0.95, bottom=0.235, top=0.93)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path,
                    default=REPO / "scripts" / "run_y8q_chuno.json",
                    help="Summary JSON with theta_hat (run_y8q_chuno.py format)")
    ap.add_argument("--mode", default="gp_fit_y8_quantile",
                    help="Which weight_mode key in JSON to plot (default: y8q)")
    ap.add_argument("--key", default="joint",
                    choices=["joint", "setup1", "setup2"],
                    help="Which θ̂ to plot (default: joint)")
    ap.add_argument("--rheo", type=Path,
                    default=REPO / "FlowCurve" / "Rheo_Data" / "Chuno_20230114_1523_25C.csv",
                    help="Rheometer CSV")
    ap.add_argument("--out", type=Path,
                    default=REPO / "FlowCurve" / "figs" / "y8q_chuno_flowcurve.png",
                    help="Output PNG")
    ap.add_argument("--truth", nargs=3, type=float,
                    default=list(TRUTH_CHUNO), metavar=("n", "eta", "sigma_y"),
                    help="Truth (n η σ_y) — drawn red")
    cli = ap.parse_args()

    summary = json.loads(cli.json.read_text())
    if cli.mode not in summary:
        sys.exit(f"--mode={cli.mode} not in {cli.json}; have {list(summary)}")
    block = summary[cli.mode][cli.key]
    th = block["theta"] if cli.key in ("setup1", "setup2") else block["theta"]
    print(f"truth      (red):  n={cli.truth[0]:.3f}  η={cli.truth[1]:.2f}  σ_y={cli.truth[2]:.2f}")
    print(f"{cli.mode}/{cli.key} (blue):  n={th[0]:.3f}  η={th[1]:.2f}  σ_y={th[2]:.2f}")

    params = [
        Param(eta=cli.truth[1], n=cli.truth[0], sigmaY=cli.truth[2]),  # red
        Param(eta=th[1],         n=th[0],         sigmaY=th[2]),         # blue
    ]
    plot_flow_curve(cli.rheo, params, cli.out)


if __name__ == "__main__":
    main()
