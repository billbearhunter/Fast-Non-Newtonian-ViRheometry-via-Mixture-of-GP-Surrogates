"""Batch inverse pipeline entry point.

Thin wrapper around `scripts.run_rw_experiments`.
Runs the production y8-quantile + GP-aware inverse over many materials.

Example:
    python run_pipeline.py \
        --state-root Models/yshape_mogp_production \
        --out-dir OptimizationResults/test_run \
        --materials all --mode all

For single-material application use, prefer:
    python -m Optimization.estimate_first_setup -f <ref_dir>
    python -m Optimization.estimate_joint_setup -f <ref_dir1> -s <ref_dir2>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from scripts.run_rw_experiments import main


if __name__ == "__main__":
    main()
