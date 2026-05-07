"""Verify recovered θ̂ by running MPM and writing 8-frame displacement CSV.

This is the production replacement for the ad-hoc MPM-verify shell snippets
that produced `joint_y8q_setup1_mpm.csv` and `joint_y8q_setup2_mpm.csv`
yesterday.  Output schema (matches existing files):

    n,eta,sigma_y,width,height,x_01,x_02,x_03,x_04,x_05,x_06,x_07,x_08

Usage:
    python scripts/verify_mpm.py \\
        --theta 0.6989 7.911 20.867 \\
        --setup 2.5 2.7 \\
        --out scripts/joint_y8q_setup1_mpm.csv

Requires Taichi MPM stack (HeadlessSimulatorMLS).  Run in a process with the
GPU-enabled env (Taichi will lazily init on first call).  Single MPM run is
~minutes per setup — defer batch runs.

⚠ STATUS: scaffolded. Wire into surrogate.densify_yshape.HeadlessSimulatorMLS
   when first needed.  Until then, prefer running ad-hoc and dropping the
   resulting csv into scripts/.  Skipping in Phase 1 verification because
   joint_y8q_setup{1,2}_mpm.csv already exist on disk and don't need
   regeneration to verify the routing+inverse pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--theta", nargs=3, type=float, required=True,
                    metavar=("n", "eta", "sigma_y"))
    ap.add_argument("--setup", nargs=2, type=float, required=True,
                    metavar=("W", "H"))
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path")
    ap.add_argument("--density", type=float, default=1.0)
    args = ap.parse_args()

    # TODO: lazy-import HeadlessSimulatorMLS, run forward, extract 8-frame y[7]
    # See surrogate/densify_yshape.py for the API used during training-data
    # generation; the same simulator is the right verify back-end.
    sys.exit(
        "verify_mpm.py: stub — full MPM run not yet wired.  "
        "joint_y8q_setup{1,2}_mpm.csv already on disk; reproduce by running "
        "the y8q joint inverse + an inline Taichi MPM call (see yesterday's "
        "session 2026-05-06 transcript)."
    )


if __name__ == "__main__":
    main()
