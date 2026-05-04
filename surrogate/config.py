"""Hyperparameters for the HVIMoGP-rBCM surrogate.

All tunables live here. Deliberately flat (no nested dicts) to make
sweeps / CLI overrides easy.

Layout after the 2026-04-22 flatten:
  * Section 1 — core paths, shared tensor config (DEVICE / DTYPE)
  * Section 2 — hierarchical rBCM model structure (K_geo, K_phi, kernel)
  * Section 3 — training / inference hyperparams

Paths in section 1 default to in-repo locations so a fresh clone / release
tarball Just Works. Override via environment variables or the relevant
script's CLI flag if you need to point elsewhere.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import torch

# ════════════════════════════════════════════════════════════════════════════
# 1) Paths, I/O columns, device
# ════════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).parent.parent.resolve()

# Training data (canonical v3 merged split). Overrideable via env.
MERGED_WS = Path(os.environ.get(
    "VIRHEO_TRAINING_DATA",
    ROOT / "TrainingData" / "moe_workspace_merged_v3_20260419",
))
MERGED_TRAIN_CSV = MERGED_WS / "train_merged.csv"
MERGED_VAL_CSV   = MERGED_WS / "val_merged.csv"
MERGED_TEST_CSV  = MERGED_WS / "test_merged.csv"

# Model checkpoints root
MODELS_ROOT = Path(os.environ.get("VIRHEO_MODELS", ROOT / "Models"))
DEFAULT_PARTITION_DIR = MODELS_ROOT / "full_partition"
DEFAULT_V2_MODEL      = MODELS_ROOT / "rbcm_v2" / "model.pt"
DEFAULT_V1_MODEL      = MODELS_ROOT / "rbcm_v1" / "model.pt"

# Optimization / inverse run output root
DEFAULT_OUT = Path(os.environ.get(
    "VIRHEO_OPT_RESULTS", ROOT / "OptimizationResults"
))

# I/O column contract
INPUT_COLS  = ["n", "eta", "sigma_y", "width", "height"]     # 5
OUTPUT_COLS = [f"x_{i:02d}" for i in range(1, 9)]            # 8
LOG_INPUTS  = ["eta", "sigma_y"]
D_X = len(INPUT_COLS)    # 5
D_Y = len(OUTPUT_COLS)   # 8

# Tensor config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64

# Numerical
JITTER  = 1e-5
LOG_EPS = 1e-8
SEED    = 42


# ════════════════════════════════════════════════════════════════════════════
# 2) Hierarchical partition structure (BIC-selected)
# ════════════════════════════════════════════════════════════════════════════
# 12 geometry groups on (W, H); per-geo K_phi from BIC scan stored in
# k_phi.json next to this file. Total experts = sum(K_PHI_PER_GEO) ≈ 540.
K_GEO = 12
_K_PHI_JSON = Path(__file__).with_name("k_phi.json")
if _K_PHI_JSON.exists():
    with open(_K_PHI_JSON, encoding="utf-8") as _f:
        _kphi_cfg = json.load(_f)
    K_PHI_PER_GEO: list[int] = _kphi_cfg["k_phi_per_geo"]
else:
    K_PHI_PER_GEO = [20] * K_GEO

assert len(K_PHI_PER_GEO) == K_GEO, \
    f"K_PHI_PER_GEO length {len(K_PHI_PER_GEO)} != K_GEO {K_GEO}"
N_EXPERTS = sum(K_PHI_PER_GEO)

# Phi feature dimension (obs-space; see surrogate.features.build_phi)
D_PHI = 18


# ════════════════════════════════════════════════════════════════════════════
# 3) Model / training / inference hyperparameters
# ════════════════════════════════════════════════════════════════════════════
# Expert kernel (ARD across 5 input dims)
KERNEL = "matern25_ard"

# Cluster size guardrails
MIN_CLUSTER_SIZE = 30        # below this -> null phi-expert (baseline only)
EXPERT_MAX_SIZE  = 2500      # cap via stratified subsample before Cholesky

# Per-expert MLE training
EXPERT_N_ITERS        = 100
EXPERT_LR             = 0.05
EXPERT_EARLY_STOP_TOL = 1e-4
EXPERT_LOG_EVERY      = 10

# Per-geo baseline experts (GRBCM communication subset)
BASELINE_ENABLED    = True
BASELINE_SUBSAMPLE  = 2000
BASELINE_N_ITERS    = 200
BASELINE_LR         = 0.05

# Inference (rBCM / GRBCM aggregation)
INFER_TOP_K_PHI = 1        # paper default: single best phi-expert
INFER_TOP_GEO   = 1        # hard geo routing
RBCM_BETA_FLOOR = 0.0
