# Fast Non-Newtonian ViRheometry via Mixture of GP Surrogates

Production inverse pipeline for recovering Herschel-Bulkley parameters
`(n, eta, sigma_y)` from real-world dam-break video.

## Production Method

```text
model bank:        Models/yshape_mogp_production
routing:           y8_quantile
likelihood:        log_nuisance_gp
shape_top_k:       1
noise floor:       0.20
variance calib:    disabled
CMA-ES restarts:   5
CI reporting:      enabled
```

## Application Entry Points

```bash
# Step 1: estimate from setup 1 and propose setup 2.
python -m Optimization.estimate_first_setup \
    -f data/new_real_world_experiments/ref_<material>_<H>_<W>_1 \
    --density <rho>

# Step 2: joint inverse from setup 1 + setup 2.
python -m Optimization.estimate_joint_setup \
    -f data/new_real_world_experiments/ref_<material>_<H1>_<W1>_1 \
    -s data/new_real_world_experiments/ref_<material>_<H2>_<W2>_2
```

## Batch Driver

```bash
python run_pipeline.py \
    --root data/new_real_world_experiments \
    --out-dir OptimizationResults/<run_name> \
    --materials all --mode all
```

## Important Paths

```text
Optimization/
  estimate_first_setup.py      # application step 1
  estimate_joint_setup.py      # application step 2
  libs/
    engine.py                  # runtime loading and production bank path
    selector.py                # y8_quantile routing + GP-aware CMA-ES inverse
    mechanism.py               # Hessian-based next-setup proposal

Models/yshape_mogp_production/ # active production model bank
surrogate/                     # GP surrogate and training utilities
scripts/validate_chuno_inverse.py # Chuno validation fixture
scripts/validate_prior_inverse.py # prior-material validation fixture
docs/                          # method, results, discussion notes
```

The old top-level `vi_mogp/` package is removed. Legacy pickle aliases are
installed in memory by `Optimization.libs.engine` when needed.
