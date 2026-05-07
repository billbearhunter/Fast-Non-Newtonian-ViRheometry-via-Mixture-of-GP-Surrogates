# Freshness Experiment Design — 2026-05-07

Design specification for the freshness/kinetics experiment that demonstrates
the temporal-resolution advantage of camera-based inverse over conventional
rheometry. This document supersedes the earlier meeting-notes draft.

Companion to [`METHOD_2026-05-07.md`](METHOD_2026-05-07.md),
[`RESULTS_2026-05-07.md`](RESULTS_2026-05-07.md),
[`DISCUSSION_2026-05-07.md`](DISCUSSION_2026-05-07.md).

---

## 1. Hypothesis and primary outcome

**H1 (primary):** A non-Newtonian fluid undergoing time-varying microstructural
evolution exhibits monotonic σ_y kinetics that camera inverse can resolve at
≤ 5-min cadence.

**Primary outcome:** σ_y(t) curve over t ∈ [1, 60] min sampled at 7-9 time
points; comparison against a sparse rheometer reference at t ∈ {5, 30, 60}
min.

**Success criteria** (pre-registered):
1. ≥ 80 % of camera time points report `identifiable[σ_y] = True` (truth-blind
   identifiability flag from Hessian Laplace CI).
2. Camera σ_y(t) at the 3 rheometer-validation time points lies within
   ±25 % of the rheometer HB-fit σ_y.
3. σ_y(t) is monotonic (or matches a known non-monotonic pattern, see §3.4)
   to within camera dispersion bands.

Failing any criterion triggers root-cause analysis (§9), not silent retry.

---

## 2. Material selection — decision matrix

Three candidates (Option A in earlier notes; B and C deprecated, see §10):

| Candidate | Cold-soluble | Kinetics timescale | σ_y range (Pa) at typical conc. | Reproducibility | Recommendation |
|---|:---:|---|---|:---:|:---:|
| **Xanthan gum 0.5 % w/v** | ✓ | 5–30 min | 5–60 | high | **★ primary** |
| Guar gum 0.7 % w/v | ✓ | 10–60 min | 10–80 | high | secondary |
| ι-carrageenan 0.5 % w/v + KCl 50 mM | ✓ | 30–60 min | 20–150 | medium | tertiary |
| Agar 0.5 % w/v (heat-dissolved + cooled) | thermal | 5–20 min cooling-set | 50–300 | medium | excluded — protocol requires heat |
| Gelatin 2 % (heat-dissolved + cooled) | thermal | 30–120 min | 20–400 | medium | excluded — slow + temperature-sensitive |

**Primary choice rationale (xanthan):**
- **Cold-soluble**: no thermal protocol → no temperature-as-confound.
- **σ_y range 5–60 Pa**: comfortably inside calibrated GP coverage (gid 0
  σ_y_floor = 5 Pa; gid 0 covers σ_y up to ~100 Pa).
- **Kinetics 5–30 min**: rheometer cycle time (~30 min) matches the *full*
  experiment, so a single rheometer run captures only one "static" point —
  a clean demonstration of the temporal-resolution gap.
- **Food-grade, cheap, available**: standard kitchen ingredient. Single
  manufacturer batch eliminates vendor-variability confound.

---

## 3. Protocol — sample preparation

### 3.1 Reagents
- Xanthan gum, food-grade, single batch (record lot number).
- Distilled or filtered tap water, pre-equilibrated at lab temperature
  (24 ± 1 °C, recorded at start and end of each trial).

### 3.2 Recipe (per single time point)
- Mass xanthan: **0.500 g** (± 0.005 g, balance accuracy).
- Mass water:   **99.500 g** (± 0.05 g) → 0.5 % w/w.
- Container: 250 mL beaker, magnetic stirrer 200 rpm, stir bar 25 mm.

### 3.3 Mixing protocol
- t = 0: simultaneously start stir + start timer + drop xanthan slowly into
  vortex over 5 s (avoid clumping).
- Continue stir 60 s. Stop stir.
- Sample is now ready for camera release.

### 3.4 Time points
- Linear-then-log spacing: **t ∈ {1, 3, 5, 10, 20, 30, 45, 60} min** (8 pts).
- Each time point requires a **fresh batch** (xanthan re-disperses on
  re-stirring → cannot reuse). 8 batches per replicate trial.
- Replicates: **n = 3 independent trials per time point** total.
  - Per replicate, randomize time-point order to avoid correlated batch
    effects (e.g., trial-1 morning vs trial-3 afternoon).

### 3.5 Expected kinetics shape
Xanthan hydration: monotonic σ_y rise, plateau by ~30 min. Functional form:

```
σ_y(t) ≈ σ_y_∞ · (1 − exp(−t/τ))
```

with τ ∈ [3, 15] min depending on temperature/shear history. Fit `(σ_y_∞,
τ)` from the camera kinetics curve as a secondary outcome.

---

## 4. Camera setup geometry (W, H)

### 4.1 Routing requirement
Pre-experiment GP-coverage check (§5) determines geometry. Default plan:

| Setup | (W, H) cm | Routes to gid | Calibrated? |
|---|---|---|:---:|
| **Setup 1** | (2.5, 2.7) | gid 0  | ✓ (post-hoc c_k) |
| **Setup 2** | (4.5, 2.5) | gid 10 | ✓ (post-hoc c_k) |

Both gids are post-hoc-calibrated (see DISCUSSION §4), so confidence
intervals are trustworthy. Reuse `data/new_real_world_experiments_freshness/
ref_Chuno_10min_2.7_2.5_1/settings.xml` as the setup-1 template; create
analogous `ref_Xanthan_<t>min_2.5_4.5_2/settings.xml` for setup 2.

### 4.2 Per-time-point capture
- Release substrate prepared (clean acrylic plate with calibrated grid).
- Setup 1 (W=2.5, H=2.7) cuboid mould → release → 8 frames at fixed Δt.
- Repeat for setup 2 (W=4.5, H=2.5) on a *separate aliquot* of the same
  batch, within ≤ 60 s of setup 1 release.
- Camera calibration (`camera_params.xml`) **fixed across the day** — do
  not re-calibrate between time points.

### 4.3 Joint vs single-setup choice for kinetics
- **Joint (recommended)**: paper-grade σ_y identification (DISCUSSION §1).
  Cost: 2× release per time point, ~20 s inverse on GPU.
- **Single setup-1 only (fast mode)**: ~10 s inverse, but high-σ_y
  late-time points may show (n, η) ridge degeneracy (DISCUSSION §1).
- **Decision rule**: run joint for paper-grade kinetics curve (primary
  output). Optionally also report setup-1-alone curve for direct
  side-by-side cost/quality comparison.

---

## 5. Pre-experiment routing / coverage check

Before committing to the 8-time-point × 3-replicate × 2-setup protocol
(48 camera runs), validate GP coverage with a single mid-time-point trial:

```bash
# Pilot trial: t = 15 min (mid-kinetics), n = 1
# 1. Prepare batch per §3.3, wait 15 min
# 2. Capture 8 frames at setup 1 + setup 2
# 3. Run paper-grade joint inverse:
python scripts/run_y8q_prior.py \
    --data-root data/new_real_world_experiments_freshness \
    --material Xanthan15min \
    --restarts 5 \
    --out scripts/run_y8q_xanthan_pilot.json
```

**Pilot pass criteria**:
1. Both setups' routed sub bbox includes the pilot σ_y estimate.
2. y8 z-score < 1.5 for both setups (y_obs lies in training y8 distribution).
3. Joint σ_y identifiable, dispersion < 10 %.
4. Joint σ_y CI half-width < 50 % of point estimate.

If pilot fails, iterate: try lower/higher concentration to bring σ_y range
into coverage; or change setup 2 geometry to reach a different gid.

Pilot is also the first chance to validate the rheometer-side protocol
(§6); concurrent rheometer measurement at the pilot time point gives a
single anchor point for sanity-checking sub bbox.

---

## 6. Rheometer validation protocol

### 6.1 Sparse validation time points
- t = 5 min, 30 min, 60 min — three rheometer runs per replicate trial.
- One trial × 3 time points = 3 separate xanthan batches per rheometer day.
- Rheometer cycle time (loading + measurement + cleaning) ≈ 30 min →
  one rheometer can keep up if dedicated.

### 6.2 Rheometer settings
- Geometry: cone-plate 40 mm Ø, 1° angle, gap 28 µm.
- Temperature: 24 °C (matched to lab/camera temperature).
- Shear-rate sweep: γ̇ ∈ [0.1, 1000] s⁻¹, log-spaced 21 points,
  pre-shear 30 s at 100 s⁻¹ then 10 s rest.
- Fit HB model on γ̇ ∈ [1, 100] s⁻¹ (inverse-domain match).

### 6.3 Camera-rheometer comparison
At each of t = 5, 30, 60 min, plot:
- Camera-joint θ̂ point + 95 % CI.
- Rheometer HB fit point.
- Flow-curve overlay: rheometer curve + camera-implied HB curves at the 3
  validation time points.

Acceptance: rheometer truth lies inside camera 95 % CI at ≥ 2 of 3 points.

---

## 7. Statistical analysis plan

### 7.1 Per-time-point summary
For each (t, replicate), report:
- θ̂ = (n̂, η̂, σ̂_y) joint inverse point estimate.
- 95 % Hessian Laplace CI per parameter.
- 5-restart z-dispersion per parameter.
- Identifiability flag per parameter.
- Routed sub_id (setup 1 + setup 2), bbox, y8 z-score (per setup).

### 7.2 Across-replicates aggregation
- σ_y(t) primary estimate: median of 3 replicate point estimates per t.
- σ_y(t) error bar: max of (per-replicate CI half-width, replicate-spread).
  This conservatively reports the larger of inverse uncertainty and
  preparation variability.

### 7.3 Kinetic parameter fit
Fit `σ_y(t) = σ_y_∞ · (1 − exp(−t/τ)) + σ_y_0` with non-linear least squares
(weighted by error bars from §7.2). Report `(σ_y_∞, τ, σ_y_0)` ± 1-σ.

### 7.4 Temporal-resolution headline statistic
- "Camera resolves σ_y kinetics at Δt = 1 min cadence (8 pts in 60 min)
  with median CI half-width X Pa per point".
- "Rheometer in the same 60 min produces 1–2 reliable points".
- → Direct figure-of-merit: **points-per-hour ratio** ≈ 8/2 = 4× to 8/1 = 8×
  (depending on what counts as a "reliable" rheometer point at high cadence).

---

## 8. Pre-registration checklist (commit before pilot)

Lock these decisions in writing **before** running the pilot:

- [ ] Material: xanthan gum (or alternative if pilot fails).
- [ ] Concentration: 0.5 % w/w.
- [ ] Time points: {1, 3, 5, 10, 20, 30, 45, 60} min.
- [ ] Replicates: n = 3 trials, randomized time-point order.
- [ ] Setups: (W=2.5, H=2.7) + (W=4.5, H=2.5).
- [ ] Inverse: joint paper-grade (5-restart, floor=0.20, disable_calib).
- [ ] Rheometer validation points: t ∈ {5, 30, 60} min.
- [ ] Success criteria (§1).

After pre-registration, deviations require a written amendment with
rationale.

---

## 9. Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|---|---|---|
| GP coverage gap (pilot fails) | y8 z-score > 1.5 or sub bbox excludes σ_y | Re-pilot at adjusted concentration / setup geometry; consider a 4th candidate material. |
| HB ridge degeneracy at high σ_y plateau | dispersion[η] > 25 % at late time points | Already documented (DISCUSSION §1). σ_y still identified; report (n, η) as ridge-degenerate, not as bias. |
| Real-MPM gap dominates inverse residual | Row 4 ≈ Row 5 in snapdiff but Row 6 small | Already understood (DISCUSSION §3). Truth recovery preserved; flag in caption. |
| Replicate-to-replicate σ_y > 30 % spread | Cross-replicate variance high | Investigate batch preparation (water temp, stirring time, age of xanthan powder). Add 4th replicate. |
| Identifiability flips from True at t=5 to False at t=30 | Identifiability column heterogeneous over t | Likely sub-routing changes as σ_y crosses bin boundary. Report sub_id per t; if necessary, run joint inverse with K=2 (top-2 sub union) for affected points. |
| Rheometer-camera gap > 25 % at all 3 validation points | Validation acceptance criterion fails | Investigate sim-real gap calibration. Could indicate xanthan-specific physics (charge-screening, salt content of tap water) not modelled by Taichi MPM. |

---

## 10. Why options B and C are deprecated

**Option B (commercial + intervention)** was deprecated because the
intervention protocol is the variable being studied — but the inverse
itself does not benefit from intervention vs simple kinetics, and the
rheometer side becomes harder (you have to control intervention timing
on both sides).

**Option C (Chuno aging)** was deprecated as the *primary* design but
**retained as a secondary demonstration** if pilot succeeds and time
permits. Chuno_10min sample dir already exists
(`data/new_real_world_experiments_freshness/ref_Chuno_10min_2.7_2.5_1/`,
empty pending camera capture). Running both gives a "reagent" + "real
food" pair, strengthening the freshness narrative.

---

## 11. Deliverables

End-of-experiment outputs to produce:

```
data/new_real_world_experiments_freshness/
  ref_Xanthan_<t>min_2.7_2.5_1/    × 8 t-values × 3 replicates  ⇒ 24 dirs
  ref_Xanthan_<t>min_2.5_4.5_2/    × 8 t-values × 3 replicates  ⇒ 24 dirs
  ref_Xanthan_<t>min_rheo/...      × 3 t-values × 3 replicates  ⇒ 9 csvs

scripts/
  run_y8q_xanthan_kinetics.py        # batch inverse driver, exports kinetics JSON
  plot_xanthan_kinetics.py           # σ_y(t) curve + rheometer overlay

docs/figs/
  xanthan_kinetics_sigma_y.png       # primary headline figure
  xanthan_kinetics_flowcurves.png    # rheometer + camera flowcurves at validation pts
  xanthan_temporal_resolution.png    # bar chart: camera 8 pts/h vs rheo 1-2 pts/h
```

---

## 12. Status

- 2026-05-07: design pre-registered (this document).
- Pending: pilot trial (xanthan, t = 15 min) to validate GP coverage and
  rheometer protocol.
- Pending: full kinetics campaign once pilot passes.

Existing artefacts:
- `data/new_real_world_experiments_freshness/ref_Chuno_10min_2.7_2.5_1/
  settings.xml` (reusable as setup-1 template; ready for camera capture
  on Chuno-aging secondary experiment).
- gid 0 + gid 10 post-hoc calibrated (suitable for paper-grade output).
