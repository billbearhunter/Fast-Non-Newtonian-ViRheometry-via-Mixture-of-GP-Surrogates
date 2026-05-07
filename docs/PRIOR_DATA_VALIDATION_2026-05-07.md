# Prior data validation — Hamamichi 2023 dataset

Companion appendix to [`RESULTS_2026-05-07.md`](RESULTS_2026-05-07.md).
Demonstrates that the y8q + GP-aware-likelihood inverse runs end-to-end on
the 12 prior materials from Hamamichi 2023 (`data/real_world_experiments/`)
with paper-grade defaults. No rheometer truth is available for most prior
materials, so reporting is **truth-blind UQ only**: identifiability flags +
95 % Hessian Laplace CI + 5-restart dispersion.

---

## 1. Setup

- Inverse: joint setup1 + setup2, paper-grade
  (`floor=0.20`, `disable_calib=True`, 5-restart CMA, `--joint-only`).
- Driver: [`scripts/run_y8q_prior.py`](../scripts/run_y8q_prior.py).
- Reference output:
  [`scripts/run_y8q_prior_joint.json`](../scripts/run_y8q_prior_joint.json).
- Wall time: **95.6 s** for 12 materials (≈ 8 s per material on a single
  NVIDIA GPU).

---

## 2. Joint inverse summary

`identifiable` columns use `n / η / σ_y` flag where `·` = not identifiable.

| Material                                  |    n̂  |     η̂  |   σ̂_y   | σ_y CI 95%       | σ_y disp | id     |
|---|---:|---:|---:|---|---:|:---:|
| Carbonara                                 | 0.474 |  42.62 | 225.08 | [173.98, 291.18] |  +2.2 %  | n,η,σ |
| Cobb                                      | 0.705 |  31.36 |   6.27 | [1.00, 1096.63]  |  +0.3 %  | n,η,·  (low σ_y) |
| Congee                                    | 0.521 | 246.89 | 224.45 | [152.31, 330.75] |  +8.7 %  | n,η,σ |
| Island                                    | 0.487 |  34.31 | 194.66 | [159.71, 237.26] |  +0.0 %  | n,η,σ |
| JapaneseCabbagePancakeSauce               | 0.349 |  31.25 |  62.08 | [25.33, 152.13]  |  +0.2 %  | n,η,σ |
| JapaneseThickenedWorcestershireSauce      | 0.540 |  14.92 |  41.03 | [22.09, 76.21]   |  +0.0 %  | n,η,σ |
| Lotion                                    | 0.548 |  29.63 | 146.73 | [101.18, 212.78] | +19.2 %  | n,η,σ |
| MoisturizingMilk                          | 0.300 |  11.98 |  43.95 | [8.70, 222.07]   |  +0.1 %  | n,η,σ |
| Mustard                                   | 0.524 | 259.58 |  62.63 | [4.45, 881.90]   |  +0.0 %  | n,η,σ |
| Pomodoro                                  | 0.476 |  38.03 | 276.33 | [220.73, 345.92] |  +0.0 %  | n,η,σ |
| Sesame                                    | 0.717 |  12.71 |   0.57 | [1.00, 1096.63]  |  +0.1 %  | n,η,·  (low σ_y) |
| SweetBeanPaste                            | 0.496 | 154.25 | 270.71 | [199.12, 368.05] |  +0.1 %  | n,η,σ |

All 12 materials route to a routed sub_id within the 25-gid bank without
fallback. 10/12 materials report `identifiable[σ_y] = True`. The two
`σ_y` non-identifiable cases (**Cobb**, **Sesame**) have point estimates
on the lower edge of the routed sub bbox, consistent with materials whose
yield stress is below the σ_y_min = 5 Pa reporting floor — the inverse
correctly refuses to confidently distinguish σ_y from zero.

`n` is non-identifiable (CI [0, 1]) on all 12 materials, as expected from
the HB ridge analysis (DISCUSSION_2026-05-07 §1).

---

## 3. Visualisation

[`FlowCurve/figs/prior_data_summary.png`](../FlowCurve/figs/prior_data_summary.png):
4×3 grid of HB flow curves on γ̇ ∈ [1, 100] s⁻¹ with σ_y 95 % CI shaded.
Renderer: [`scripts/plot_prior_data_summary.py`](../scripts/plot_prior_data_summary.py).

---

## 4. Performance breakdown

| Mode                                     | wall time |  per-inverse avg |
|---|---:|---:|
| Full (setup1 + setup2 + joint, 5-restart) | 173.8 s | 4.83 s (36 inverses) |
| **`--joint-only` (joint only, 5-restart)** | **95.6 s** | **7.97 s (12 inverses)** |
| Speedup of `--joint-only` vs full         | 1.82× | — |

`--joint-only` skips per-setup-alone diagnostic inverses (the routed
sub_id, bbox, and y8 z-score are still reported via `prepare_setup_shape`
output, so identifiability is not lost). The headline σ_y point + CI come
from the joint inverse, which is unchanged.

The batched per-generation CMA evaluator (`engine._predict_full_batched`)
delivers the underlying ~8–13× per-inverse speedup vs the pre-batched
sequential path; on prior data with 12 materials this is what enables the
sub-100 s end-to-end batch.

---

## 5. Reproducibility

```bash
# Full run (3 inverses per material)
python scripts/run_y8q_prior.py --restarts 5 --out scripts/run_y8q_prior.json

# Joint-only run (paper-grade headline)
python scripts/run_y8q_prior.py --restarts 5 --joint-only \
    --out scripts/run_y8q_prior_joint.json

# Single material (e.g. Lotion)
python scripts/run_y8q_prior.py --material Lotion --restarts 5 --joint-only \
    --out scripts/run_y8q_prior_lotion.json

# Summary figure
python scripts/plot_prior_data_summary.py \
    --json scripts/run_y8q_prior_joint.json \
    --out FlowCurve/figs/prior_data_summary.png
```
