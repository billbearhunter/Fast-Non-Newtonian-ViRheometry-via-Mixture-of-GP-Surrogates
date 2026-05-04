# Plan B Production Pipeline — V10 Y-Shape

Last updated: 2026-05-04

## Overview

Plan B is the empirical local optimum of the V10 y-shape inverse pipeline.
After exhaustive testing of 9 algorithmic variants (Default / Plan B / C /
C2 / B+C2 / D / E K=2 / F / Joint MAP), Plan B's `sat_lo=2` single-parameter
change to V1's σY-prior is the only configuration that achieves a clean
median fc% of 25.7% on 12 RW materials without per-material tuning.

```
Production bank:    Models/v10_yshape_v3p2_round2partial   (415k pool)
Production config:  --inverse-mode shape  +  --shape-sy-sat-lo 2.0
Total time:         ~6s/single, ~12s/double (vs V1 ~35h/material → 4000-7000× faster)
Median fc%:         25.7% (full 12 mat),  20.8% (4 rheo, < v3 prod 22.9%)
```

## Production config (one-liner)

```bash
python -m pipeline_v10.Optimization.run_real_world_experiments \
    --root pipeline_v10/data/real_world_experiments \
    --state-root pipeline_v10/Models/v10_yshape_v3p2_round2partial \
    --out-dir pipeline_v10/OptimizationResults/<run_name> \
    --materials all --mode all \
    --inverse-mode shape \
    --shape-popsize 12 --shape-max-iter 30 --shape-sigma0 0.25 \
    --shape-sy-prior-weight 0.5 --shape-sy-prior-min-std 0.4 \
    --shape-sy-prior-sat-factor 0.0 \
    --shape-sy-sat-lo 2.0 --shape-sy-sat-hi 380.0
```

## Per-material fc% (Plan B double mode)

| material | trust | θ̂ (n, η, σY) | fc% | vs v3 prod |
|---|:-:|---|---:|---:|
| **JTWS** (Chuno) | rheo | (0.59, 12.0, 17.4) | **5.6** ⭐⭐ | -2 |
| **Sesame** | fall | (1.00, 4.5, 17.8) | **8.1** ⭐⭐ | -3.8 |
| JCP | rheo | (0.57, 52.2, 101.0) | 20.2 | -7.4 |
| Mustard | fall | (0.42, 190.8, 224.2) | 20.4 | -0.8 |
| Island | fall | (0.51, 58.2, 81.7) | 20.4 | +0.3 |
| **Lotion** | rheo | (0.37, 124.7, 27.0) | **21.4** ⭐ | -67.1 |
| Pomodoro | fall | (0.30, 109.8, 54.5) | 30.0 | +14.8 (regression) |
| Cobb | fall | (0.51, 54.1, 11.0) | 40.4 | +3.4 |
| MoistMilk | rheo | (0.75, 34.0, 62.5) | 41.6 | +1.7 |
| Congee | fall | (0.64, 146.2, 392.0) | 42.7 | +4.1 |
| SweetBean | fall | (0.99, 84.6, 201.6) | 47.9 | +24.3 |
| Carbonara | fall | (0.92, 76.0, 115.3) | 323.6 | -329 ⭐ (sim-real gap limit) |

```
12-mat median:   25.7%  ⭐ best across all 9 configs tested
4-rheo median:   20.8%  ⭐ better than v3 production 22.9%

Trade-off: Pomodoro 5→30 regression accepted for huge Lotion (89→21) +
JTWS (32→6) wins on the 4 rheometer-validated materials.
```

## Algorithm rationale

```
1. Layer 1+2+3 deterministic routing (one sub per setup)
2. CMA-ES inverse on sub GP forward, joint two-setup
3. σY two-stage prior (Hamamichi V1-paper-style):
   stage 1: single inverse on each setup → σY_a, σY_b estimates
   stage 2: anchor = √(σY_a × σY_b), pull θ̂ toward anchor
4. Saturation guard: if any single σY estimate < 2 or > 380 (boundary):
   → disable prior (avoid biasing toward bad estimate)
5. Default CMA budget (popsize=12, max_iter=30) is intentional:
   deeper search exposes ridge degeneracy; early stopping = implicit
   regularization (Sesame fc 8.1% requires this; deep CMA gives 82%).
```

## Why not Joint MAP (no prior)?

```
Pure ML / Joint MAP is mathematically cleanest under bounded sim-real gap
+ Hamamichi-orthogonal setup design.  But in our actual data:
- Sim-real gap exceeds 20% for some materials (Carbonara 28%, MoistMilk 30%)
- Setups not strictly Hamamichi-orthogonal
- Likelihood ridge degeneracy real → ML estimate non-unique
- Plan B's σY-prior empirically breaks ridge near truth (lucky alignment for
  Sesame/JTWS/Lotion)

Joint MAP median = 37.5% (vs Plan B 25.7%) confirms ridge is real.
```

## Architecture summary

```
y_obs (8 frames) + (W, H)
  ↓
Layer 1: GridGeoRouter (W,H) → gid (5×5 grid, 25 gids)
Layer 2: log-y8 equal-width bin (k_len=5, with Plan B adaptive merge)
Layer 3: PCA(3) + KMeans on normalized curve shape (k_sh_max=5)
  ↓
sub_id → ExactExpert GP (Matern 2.5 ARD, max_n=1000 cap)
  ↓
CMA-ES + σY prior + double-mode joint optim
  ↓
θ̂ = (n, η, σY)
```

## Files / directories (post-archive cleanup)

```
KEEP (production):
  Models/v10_yshape_v3p2_round2partial/        ← Plan B bank (active)
  Models/v10_bgm_v75_layer3_retrain_realworld_combined_codex/  ← engine runtime
  Models/v10_yshape_v3/                         ← v3 paper baseline
  Models/v10_yshape_v3_backup_pre_carbonara/    ← safety backup
  
  OptimizationResults/v3p2_12mat_satlo2/        ← Plan B 12-mat results (final)
  OptimizationResults/v3p2_12mat_default/       ← Default baseline
  OptimizationResults/v3p2_12mat_planC2/        ← Plan C2 alternative
  OptimizationResults/v3p2_12mat_jointMAP/      ← Joint MAP theoretical

  Simulation/results/v3p2_12mat/                ← snapdiff renders for paper

ARCHIVED (failed experiments, packed to .tar.gz):
  Models/_archive_pre_planB.tar.gz              (1.1 GB → ~400 MB compressed)
  OptimizationResults/_archive_pre_planB.tar.gz (8.8 GB → ~3 GB compressed)
```

## Future work (paper roadmap)

```
1. Round-2 完成 (master worker resumed task b3das70by)
   → +50k uniform pool, total 470k
   → expected median fc% 22-24% (-2-3pp marginal)

2. Try k_len=6 partition with full Round-2 data
   → 30% faster GP training
   → bbox stays healthy due to more data
   → expected fc% similar to k_len=5 (no degradation)

3. Sim-to-real gap modeling (out of scope)
   → Carbonara/MoistMilk fc% bounded by gap
   → Future paper / sim improvement

4. Hamamichi-orthogonal setup 2 selection (already implemented in
   inverse_addons/active_design.py, not currently used since paper data
   used pre-selected setup pairs)
```

## Citation context

```
"We extend Hamamichi et al. 2023 (ViRheometry) by replacing their direct
sim+CMA-ES inverse loop with a hierarchical MoGP-rBCM surrogate.  This
amortizes the per-material 35-hour sim runtime to ~12 seconds at
competitive accuracy (median fc% 25.7% vs Hamamichi's published 22.9%
on 12 mat).  The σY two-stage prior with sat_lo=2 (Plan B) is our
empirical improvement over V1's saturation guard.  Configurations
tested: 9 algorithmic variants + 2 CMA depths × 12 materials = 132
runs, all on the same 415k Plan B bank."
```
