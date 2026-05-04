"""Batch inverse pipeline entry point.

Thin wrapper that delegates to scripts.run_rw_experiments.
Runs Plan B inverse on all 12 RW materials (single + double mode).

Default config (override via CLI):
    --inverse-mode shape           # CMA-ES + σY two-stage prior
    --shape-popsize 12 --shape-max-iter 30
    --shape-sy-prior-weight 0.5 --shape-sy-prior-min-std 0.4
    --shape-sy-prior-sat-factor 0.0
    --shape-sy-sat-lo 2.0 --shape-sy-sat-hi 380.0  # Plan B: relaxed saturation guard

Example:
    python run_pipeline.py \\
        --state-root Models/v10_yshape_v3p2_round2partial \\
        --out-dir OptimizationResults/test_run \\
        --materials all --mode all

For single-material interactive use, see:
    python -m Optimization.setup1 -f <ref_dir>
    python -m Optimization.setup2 -f <ref_dir1> -s <ref_dir2>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from scripts.run_rw_experiments import main


if __name__ == "__main__":
    main()
