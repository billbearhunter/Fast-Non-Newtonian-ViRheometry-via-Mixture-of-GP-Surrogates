# Fast Non-Newtonian ViRheometry via Mixture of GP Surrogates

Production inverse pipeline for recovering Herschel–Bulkley parameters
`(n, η, σ_y)` from real-world dam-break video. Replaces direct sim+CMA loops
(Hamamichi et al. 2023, ~35h/material) with a hierarchical Mixture-of-GP
surrogate + CMA-ES (~12s/material, ~4000–7000× faster).

## Highlights

```
Time:        ~6s single-setup, ~12s double-setup inverse
Median fc%:  25.7% on 12 RW materials (Plan B production config)
4-rheo fc%:  20.8% (better than Hamamichi v3 production 22.9%)
```

## Architecture (Y-shape MoGP-rBCM surrogate)

```
y_obs (8 frames) + (W, H)
  ↓
Layer 1: GridGeoRouter on (W, H)             → 5×5 grid → 25 gids
Layer 2: log-y8 equal-width bin              → k_len=5 bins (Plan B adaptive merge)
Layer 3: PCA(3) + KMeans on normalised curve → k_sh_max=5 shape clusters
  ↓
sub_id (gid, bin, shape) → ExactGP (Matern 2.5 ARD, max_n=1000)
  ↓
CMA-ES + σY two-stage prior (Plan B: sat_lo=2)
  ↓
θ̂ = (n, η, σY)
```

## Repo layout

```
.
├─ README.md
├─ STATUS_PLAN_B.md                # Production config + per-material fc%
├─ requirements.txt
├─ run_pipeline.py                 # Batch inverse runner (all 12 RW materials)
│
├─ propose_initial_setup/          # ⭐ Step 0: random setup1 proposal (prev-work style)
│  ├─ propose_initial_setup.py     # Random (W, H) → settings.xml
│  └─ out_xml.sh
│
├─ Optimization/                   # ⭐ Inverse pipeline (prev-work style)
│  ├─ setup1.py                    # 1st-setup inverse + Hessian-based setup2 proposal
│  ├─ setup2.py                    # Joint 2-setup inverse → final θ̂
│  └─ libs/                        # Internal helpers
│     ├─ engine.py                 # load_runtime (rBCM predictor + scalers + router)
│     ├─ selector.py               # Plan B: CMA-ES + σY two-stage prior
│     └─ mechanism.py              # Analytical Hessian + orthogonality setup selector
│
├─ surrogate/                      # ⭐ GP surrogate + training (merged vi_mogp)
│  ├─ partition_v10_yshape.py      # 3-layer Y-shape partition (Layer 1+2+3)
│  ├─ train_yshape_subs.py         # Per-sub ExactGP training
│  ├─ eval_yshape_forward.py       # Forward GP error + routing accuracy
│  ├─ densify_yshape.py            # route_y_to_sub helper
│  ├─ merge_tiny_subs.py           # Post-partition cleanup
│  ├─ experts.py / gp_base.py      # ExactGP (Matern 2.5 ARD)
│  ├─ model.py / predict.py        # HVIMoGP_rBCM model + inference wrapper
│  ├─ data.py                      # InputScaler, OutputScaler
│  ├─ grid_geo.py                  # GridGeoRouter (Layer 1)
│  └─ config.py / constants.py / features.py
│
├─ vi_mogp/                        # Pickle compatibility shim only
│  └─ __init__.py                  # Redirects vi_mogp.X → surrogate.X at import time
│
├─ Simulation/                     # MPM sim + render
│  ├─ main.py                      # sim+render entry
│  ├─ run_batch_sim.py / run_sim_only.py
│  └─ simulation/ GLRender3d/ ParticleSkinner3DTaichi/ config/
│
├─ DataPipeline/                   # ⭐ Round-2 5D-LHS infill data collection
│  ├─ headless_mls.py              # MPM sim core
│  ├─ run_worker.py                # Cell-by-cell worker (auto-resume)
│  ├─ generate_cell_plan.py        # B=4 grid × Q=150 plan
│  ├─ regen_plan_for_target_Q.py / regen_remaining_plan.py
│  ├─ merge_worker_outputs.py      # Worker CSV → pool
│  ├─ make_uniform_subsample.py    # Uniform LHS subsample
│  ├─ diagnose_uniformity.py / cost_benefit_analysis.py
│  ├─ build_merged_dataset.py / clean_min.py / dp_config.py
│  ├─ INFILL_README.md / STATUS.md
│  └─ plan/cell_plan_q150_2machines.csv
│
├─ Calibration/                    # Camera + ArUco calibration
├─ FlowCurve/                      # HB rheometer fit
│  ├─ hb_fit.py / flowcurve.py / param.py
│  └─ Rheo_Data/                   # Per-material rheometer CSVs
├─ tests/                          # Runtime utils (used by engine + scripts)
│  ├─ v7_5_features.py
│  └─ cluster_diagnosis.py
│
├─ scripts/                        # Batch drivers + benchmarks
│  ├─ run_rw_experiments.py        # Batch inverse: all 12 RW materials
│  ├─ build_rw_truth_map.py        # fc% computation vs rheometer/V1 truth
│  └─ bench_max_n_accuracy.py      # GP accuracy vs max_n benchmark
│
├─ Models/                         # Trained banks (358 MB)
│  ├─ v10_yshape_v3p2_round2partial/   # ⭐ Plan B production (k=5,5)
│  ├─ v10_yshape_v3p2_round2_combined/ # Round-2-combined bank
│  └─ v10_bgm_v75_..._codex/           # Engine runtime (load_runtime)
│
└─ data/                           # Datasets (191 MB)
   ├─ synthetic_splits_v2/
   ├─ synthetic_splits_v3p2_round2partial/
   ├─ synthetic_splits_round2_combined/
   └─ real_world_experiments/      # 12 RW materials × 2 setups
```

## Plan B production config

```bash
# Batch inverse: all 12 RW materials, both setups
python run_pipeline.py \
    --state-root Models/v10_yshape_v3p2_round2partial \
    --out-dir OptimizationResults/<run_name> \
    --materials all --mode all \
    --shape-popsize 12 --shape-max-iter 30 --shape-sigma0 0.25 \
    --shape-sy-prior-weight 0.5 --shape-sy-prior-min-std 0.4 \
    --shape-sy-prior-sat-factor 0.0 \
    --shape-sy-sat-lo 2.0 --shape-sy-sat-hi 380.0
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2a. Single-material inverse (prev-work style)
#     Step 0 — propose random 1st setup
python propose_initial_setup/propose_initial_setup.py -m Sesame -d 1.2

#     Step 1 — capture ref_Sesame_H_W_1/, then run inverse + get setup2 proposal
python -m Optimization.setup1 \
    --state-root Models/v10_yshape_v3p2_round2partial \
    -f data/real_world_experiments/ref_Sesame_4.2_4.0_1

#     Step 2 — capture proposed ref_Sesame_H*_W*_2/, then joint inverse
python -m Optimization.setup2 \
    --state-root Models/v10_yshape_v3p2_round2partial \
    -f data/real_world_experiments/ref_Sesame_4.2_4.0_1 \
    -s data/real_world_experiments/ref_Sesame_2.0_2.9_2

# 2b. Batch inverse: all 12 RW materials at once
python run_pipeline.py \
    --state-root Models/v10_yshape_v3p2_round2partial \
    --out-dir OptimizationResults/test_run \
    --materials all --mode all \
    --shape-sy-prior-weight 0.5 --shape-sy-sat-lo 2.0 \
    --shape-sy-prior-min-std 0.4

# 3. Compute fc% vs rheometer/paper truth
python -m scripts.build_rw_truth_map \
    --results OptimizationResults/test_run

# 4. (Optional) Render sim @ θ̂ for pixel snapdiff vs real
cd Simulation && python main.py \
    --n 0.92 --eta 76 --sigma_y 115 \
    --ref ../data/real_world_experiments/ref_Carbonara_3.7_6.4_1 \
    --out_dir results/test
```

## Data collection (Round-2 5D-LHS infill)

`DataPipeline/` is the canonical training-data collection workflow.
Plan a B=4 (n, η, σY, W, H) cell grid, run a per-cell worker (auto-resume),
merge and subsample to feed `surrogate/partition_v10_yshape.py`.

```bash
# 1. Generate cell plan
python DataPipeline/generate_cell_plan.py \
    --out DataPipeline/plan/cell_plan_q150.csv --target-q 150 --machines 2

# 2. Run worker (resumable)
python DataPipeline/run_worker.py \
    --plan DataPipeline/plan/cell_plan_q150.csv \
    --machine-id 0 --out data/round2_worker_m0.csv

# 3. Merge + uniform subsample → training pool
python DataPipeline/merge_worker_outputs.py \
    --inputs data/round2_worker_m0.csv data/round2_worker_m1.csv \
    --out data/synthetic_splits_round2_combined/pool.csv
python DataPipeline/make_uniform_subsample.py \
    --pool data/synthetic_splits_round2_combined/pool.csv \
    --out data/synthetic_splits_round2_combined/train.csv

# 4. Partition + train new bank
python -m surrogate.partition_v10_yshape \
    --data data/synthetic_splits_round2_combined \
    --out Models/v10_yshape_round2_combined
python -m surrogate.train_yshape_subs \
    --state-root Models/v10_yshape_round2_combined --max-n 1000
```

> Future work: replace uniform LHS with BO sparse-sub targeted infill.

## Per-material fc% (Plan B production, double mode)

| material | trust | fc% | comments |
|---|:-:|---:|---|
| **JTWS** (Chuno) | rheo | **5.6** ⭐⭐ | Best on rheo materials |
| **Sesame** | fall | **8.1** ⭐⭐ | σY-anchor lucky alignment |
| JCP | rheo | 20.2 | |
| Mustard | fall | 20.4 | |
| Island | fall | 20.4 | |
| **Lotion** | rheo | **21.4** ⭐ | Plan B saved (default 89.5%) |
| Pomodoro | fall | 30.0 | Plan B trade-off (default 5.3%) |
| Cobb | fall | 40.4 | |
| MoistMilk | rheo | 41.6 | |
| Congee | fall | 42.7 | |
| SweetBean | fall | 47.9 | |
| Carbonara | fall | 323.6 | sim-real gap (acknowledged limit) |

## Citation context

```
@inproceedings{xiong2026fastvirheometry,
  author = {Xiong, BillBear et al.},
  title = {Fast Non-Newtonian ViRheometry via Mixture of GP Surrogates},
  booktitle = {SIGGRAPH 2026},
  year = {2026},
}

@article{hamamichi2023nonnewtonian,
  author = {Hamamichi, Nagasawa, Okada, Seto, Yue},
  title = {Non-Newtonian ViRheometry via Similarity Analysis},
  journal = {ACM ToG (SIGGRAPH Asia)},
  year = {2023},
  doi = {10.1145/3618310},
}
```

See `STATUS_PLAN_B.md` for full algorithm details, ablation results, and
future-work roadmap.
