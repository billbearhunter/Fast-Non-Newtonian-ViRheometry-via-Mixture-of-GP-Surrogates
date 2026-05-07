"""Plot Herschel-Bulkley flow curve overlays via STRICT subprocess call to
FlowCurve/flowcurve.py.

Wraps a θ̂-summary JSON (run_y8q_chuno.py format) and the rheometer CSV,
then invokes `python FlowCurve/flowcurve.py --file ... --est ... --out ...`
exactly as the original script — no copy of plotting logic, no import of
Param.  The figure is produced by FlowCurve/flowcurve.py itself.

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
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Chuno rheometer truth (25 °C fit, surrogate-space units; flowcurve.py will divide by 10)
TRUTH_CHUNO = (0.633, 10.51, 19.62)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path,
                    default=REPO / "scripts" / "run_y8q_chuno.json")
    ap.add_argument("--mode", default="gp_fit_y8_quantile",
                    help="weight_mode key in JSON to plot (default y8q)")
    ap.add_argument("--key", default="joint",
                    choices=["joint", "setup1", "setup2"],
                    help="Which θ̂ to plot (default joint)")
    ap.add_argument("--rheo", type=Path,
                    default=REPO / "FlowCurve" / "Rheo_Data" / "Chuno_20230114_1523_25C.csv")
    ap.add_argument("--out", type=Path,
                    default=REPO / "FlowCurve" / "figs" / "y8q_chuno_flowcurve.png")
    ap.add_argument("--truth", nargs=3, type=float,
                    default=list(TRUTH_CHUNO), metavar=("n", "eta", "sigma_y"),
                    help="Truth (n η σ_y) — drawn red")
    ap.add_argument("--extent-y", nargs=2, type=float, default=[1.0, 100.0])
    cli = ap.parse_args()

    summary = json.loads(cli.json.read_text())
    if cli.mode not in summary:
        sys.exit(f"--mode={cli.mode} not in {cli.json}; have {list(summary)}")
    th = summary[cli.mode][cli.key]["theta"]

    n_t, eta_t, sy_t = cli.truth
    n_h, eta_h, sy_h = th[0], th[1], th[2]
    print(f"truth (red):       n={n_t:.3f}  η={eta_t:.2f}  σ_y={sy_t:.2f}")
    print(f"{cli.mode}/{cli.key} (blue):  n={n_h:.3f}  η={eta_h:.2f}  σ_y={sy_h:.2f}")

    # FlowCurve/flowcurve.py expects --est in order (eta, n, sigmaY).
    # First --est = red, second --est = blue (per FlowCurve/flowcurve.py:6 color_list).
    fc_dir = REPO / "FlowCurve"
    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(fc_dir.resolve()))
        except ValueError:
            return str(p)
    cmd = [
        sys.executable, "flowcurve.py",
        "--file", _rel(cli.rheo),
        "--est", f"{eta_t}", f"{n_t}", f"{sy_t}",         # red = truth
        "--est", f"{eta_h}", f"{n_h}", f"{sy_h}",         # blue = recovered
        "--out", _rel(cli.out),
        "--extent_y", f"{cli.extent_y[0]}", f"{cli.extent_y[1]}",
    ]
    # Ensure output directory exists
    cli.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[+] cd {fc_dir} && {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(cmd, cwd=str(fc_dir), check=True, env=env)
    print(f"Saved {cli.out}")


if __name__ == "__main__":
    main()
