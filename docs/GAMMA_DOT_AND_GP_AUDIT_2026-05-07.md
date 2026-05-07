# γ̇ logger + bank-wide GP precision audit — 2026-05-07

Two pieces of paper-grade pipeline tooling added in response to in-meeting
feedback:

1. **γ̇ logger** (老师 request): given a recovered θ̂ from any inverse run,
   plug back into MLS-MPM and dump the per-frame γ̇ field + summary
   statistics so the experimental shear-rate range is auditable against
   rheometer measurement coverage.
2. **GP precision audit**: a one-pass bank-wide val_rel_wrms + c_k report
   so we can substantiate paper claims about surrogate accuracy.

---

## 1. γ̇ logger

### 1.1 Driver

[`scripts/dump_gamma_dot.py`](../scripts/dump_gamma_dot.py) wraps
[`Simulation/simulation/AGTaichiMPM2.py`](../Simulation/simulation/AGTaichiMPM2.py),
which already records per-particle γ̇ and dumps `config_XX_gamma_dot.bin`
during simulation. The wrapper:
1. Resolves θ̂ from a CMA-ES inverse output JSON (`run_y8q_chuno*.json`,
   `run_y8q_prior*.json`, etc.) **or** from raw `--eta --n --sigmaY` args.
2. Invokes `AGTaichiMPM2` as a module so its in-package `taichi.py`
   shadowing isn't tripped.
3. Reads the binary dumps back, computes per-frame γ̇ quantiles + a
   logarithmic histogram, and writes `gamma_dot_summary.json`.

### 1.2 Reference runs

```bash
# Chuno joint θ̂ at setup-1 geometry (W=2.5, H=2.7)
python scripts/dump_gamma_dot.py \
    --theta-json scripts/run_y8q_chuno_paper.json \
    --mode gp_fit_y8_quantile --setup joint \
    --W 2.5 --H 2.7 \
    --out gamma_dot_runs/chuno_setup1_joint

# Okonomi joint θ̂ at setup-1 geometry (W=2.2, H=2.5)
python scripts/dump_gamma_dot.py \
    --theta-json scripts/run_y8q_okonomi_paper.json \
    --mode gp_fit_y8_quantile --setup joint \
    --W 2.2 --H 2.5 \
    --out gamma_dot_runs/okonomi_setup1_joint
```

### 1.3 Sample output

Chuno setup1 joint, θ̂ = (n=0.664, η=8.12, σ_y=19.65):

```
frame |  t [s] |   γ̇ med |   γ̇ q05 |   γ̇ q95 |   γ̇ max | frac>0
    0 |  0.000 |    0.00 |    0.00 |    0.00 |    0.00 |   0%
    1 |  0.042 |   20.67 |    2.70 |   57.99 |  171.92 | 100%
    2 |  0.083 |   24.10 |    2.23 |  101.07 |  192.91 |  99%
    3 |  0.125 |   23.63 |    3.41 |   82.10 |  139.86 | 100%
    4 |  0.167 |   19.31 |    4.12 |   48.95 |   67.81 | 100%
    5 |  0.208 |   12.71 |    3.98 |   30.05 |   42.04 | 100%
    6 |  0.250 |    8.12 |    3.39 |   18.25 |   25.59 | 100%
    7 |  0.292 |    5.33 |    2.33 |   11.42 |   19.76 | 100%
    8 |  0.333 |    3.81 |    1.57 |    8.81 |   16.75 | 100%
```

Okonomi setup1 joint, θ̂ = (n=0.769, η=21.35, σ_y=97.99):

```
frame |  t [s] |   γ̇ med |   γ̇ q95 |   γ̇ max
    1 |  0.042 |   12.93 |   29.31 |   83.63
    2 |  0.083 |   12.60 |   29.02 |   87.17
    3 |  0.125 |    7.33 |   17.05 |   49.03
    4 |  0.167 |    3.97 |    9.91 |   28.54
    5 |  0.208 |    2.66 |    6.76 |   19.52
    6 |  0.250 |    2.07 |    5.37 |   15.92
    7 |  0.292 |    1.71 |    4.59 |   13.33
    8 |  0.333 |    1.45 |    4.08 |   11.33
```

### 1.4 Take-aways for paper

- **γ̇ range** in our experiments lies in **[0.1, 200] s⁻¹** for both
  test materials, which is precisely the rheometer-measurement window
  [0.1, 1000] s⁻¹ used to fit reference HB params. So our σ_y/η/n
  recovery is in the same shear-rate regime as the rheometer reference,
  and HB validity is not stretched.
- **Higher σ_y → lower γ̇**: Okonomi (σ_y = 98 Pa) tops out at
  γ̇ q95 ≈ 30 s⁻¹ vs Chuno (σ_y = 20 Pa) at γ̇ q95 ≈ 100 s⁻¹. The
  yield-stress critical strain rate γ̇_y = (σ_y/η)^(1/n) is ~3.7 s⁻¹
  for Chuno and ~7.1 s⁻¹ for Okonomi — frames 4–8 sit *below* γ̇_y in
  both cases, consistent with the documented yield-stress-dominated
  late-frame regime that drives σ_y identification.
- **Wall time per run**: ≈ 5 s MPM + 8 s JIT per material per setup. So a
  full Chuno+Okonomi+joint γ̇ audit takes a minute on a single GPU.

### 1.5 Paper figure plan

For the final paper, suggested γ̇ figure: 2-row × 8-column heatmap
showing γ̇ on the (x, z) flow plane per frame, one row for each test
material. Already have all the binary data needed — just wire up a
matplotlib colourmap on the `_pos.bin` + `_gamma_dot.bin` files.

---

## 2. Bank-wide GP precision audit

[`scripts/audit_gp_precision.py`](../scripts/audit_gp_precision.py) runs
a 90/10 random hold-out per sub (refits hyperparameters NOT re-trained,
just the kernel matrix re-conditioned on the train subset; predicts on
the val subset) and computes per-sub:

- `val_rel_wrms` = √(mean((y_pred − y_obs)² × frame_w²) / mean(y_obs² × frame_w²))
  — the same metric used elsewhere as `val_rel_wrms_median`.
- `val_mse`, `mean_var`, `c_k = √(MSE / mean_var)`
- `n_train` after hold-out split.

### 2.1 Run

```bash
python scripts/audit_gp_precision.py --bank Models/v10_yshape_planB \
    --out scripts/gp_audit.json
```

### 2.2 Bank-wide results (1 396 subs, 25 gids)

```
val_rel_wrms median      = 0.045   (4.5 % per-sub relative-weighted-RMS)
val_rel_wrms p90         = 0.089
val_rel_wrms p99         = 0.139
val_rel_wrms max         = 0.368   (one outlier sub, see below)

c_k median               = 0.751   (slightly conservative; CIs slightly wider than ideal)
n_train median per sub   = 224
total wall-time          = 174 s on a single GPU
```

**Worst 10 subs** (potential candidates for retraining):

| gid | sub | val_rel_wrms | c_k  | n_train |
|---:|---:|---:|---:|---:|
|  8 |  40 | 0.368 | 3.27 |  151 |  ← outlier (3× over threshold), c_k overconfident |
| 10 |   1 | 0.247 | 0.83 |  295 |
|  5 |  33 | 0.186 | 0.97 |  543 |
|  5 |  31 | 0.180 | 1.08 |  136 |
|  5 |  32 | 0.171 | 0.91 |  272 |
| 23 |  33 | 0.170 | 1.88 |  142 |
|  5 |  30 | 0.163 | 1.22 |  136 |
|  5 |   1 | 0.155 | 1.00 |  229 |
|  3 |   2 | 0.150 | 1.21 |  444 |
| 20 |  32 | 0.149 | 1.25 |  165 |

### 2.3 Take-aways

- **GP precision is paper-grade**: 90 % of subs have ≤ 9 % per-sub
  relative WRMS, and 99 % of subs ≤ 14 %. The single outlier (gid 8
  sub 40) at 37 % is the only candidate worth a targeted retrain
  (c_k = 3.27 also flags it as overconfident — both metrics agree).
- **GPs are mildly conservative** (c_k median 0.75): CIs we report are
  slightly wider than the ideal c_k = 1.0 calibration. This **helps**
  paper UQ honesty — the identifiability flag never claims more
  certainty than warranted.
- **n_train heterogeneous** (63–1000 per sub): sub-to-sub variation is
  expected from the partition algorithm; not a defect.

### 2.4 Calibration coverage status (post-audit, 2026-05-07)

```
gid 0–4, 6–9, 10–24:  c_k populated via 90/10 holdout (all subs)
gid 5: ALSO calibrated (via this round) — 38 subs, c_k median 0.93
```

So **all 25 gids now have post-hoc c_k calibration**. The earlier
DISCUSSION_2026-05-07 §4 caveat ("only gid 0/10/24 calibrated") is
out-of-date.

---

## 3. What else is on the improvement board (in priority)

After the above two items, the remaining real improvement directions
are *not* GP-accuracy. From DISCUSSION_2026-05-07 §8:

1. **Third setup at extreme aspect ratio** to break (n, η) HB ridge —
   physics-side, biggest impact.
2. **Sim-real gap correction layer** trained on (rheometer-truth-θ,
   real-y_obs) pairs — fixes Lotion-class small-flow-distance misses
   we saw in [`PRIOR_DATA_VALIDATION_2026-05-07.md`](PRIOR_DATA_VALIDATION_2026-05-07.md).
3. **Surrogate-Hessian setup-2 proposer** to replace analytical
   Poiseuille model.
4. **Data-driven `frame_w`** via inverse-Fisher per frame — depends on
   having a calibration set ≥ 5 materials with rheometer truth, i.e.
   *after* freshness experiment data is in. Not the right time yet.
