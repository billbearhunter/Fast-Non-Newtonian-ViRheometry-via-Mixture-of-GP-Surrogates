# Discussion — Limitations and Future Work, 2026-05-07

Companion to [`METHOD_2026-05-07.md`](METHOD_2026-05-07.md) and
[`RESULTS_2026-05-07.md`](RESULTS_2026-05-07.md).

---

## 1. HB ridge for high-σ_y materials

For materials whose rheology is yield-stress-dominated at experimental
shear rates (γ̇ ≲ 100 s⁻¹, σ_y ≳ 50 Pa), the joint two-setup inverse
identifies σ_y uniquely but leaves (n, η) on the HB ridge.  Empirically:

```
Chuno    (σ_y =  19.62 Pa) :   joint identifies σ_y, η to within 25 %
Okonomi  (σ_y =  87.24 Pa) :   joint identifies σ_y; (n, η) ridge degenerate
                                (n diff +93 %, η diff −68 % at MAP;
                                 5-restart η dispersion 32 %)
```

This is a **fundamental property** of HB inverse with our (W, H) setup
choices, not a surrogate-training limitation.  We confirmed by direct
density analysis that GP training near rheometer truth is *denser* for
Okonomi than for Chuno.

**Mitigation** (not yet implemented):
- A third experimental setup with very different aspect ratio (e.g.,
  H/W ≫ 2 for shear-dominated profile, or extreme W/H for rapid
  spreading) would target the (n, η) ridge directly.
- Alternative: incorporate independent rheometer prior on n (e.g., from
  high-shear-rate measurement) as a Bayesian regulariser.

---

## 2. Poiseuille proposal mismatch

`propose_next_setup` (Hamamichi 2023, faithfully ported in
`Optimization/libs/mechanism.py`) computes the analytical HB Hessian
under **Poiseuille channel flow**.  Our experiment is gravity-driven
release-and-spread with a free surface.  The two physics differ:

```
Poiseuille:   pressure-driven steady flow in fixed channel
              no free surface, no transient, no spread
              closed-form Hessian over (η, n, σ_y)

Release-spread:  gravity-driven, free surface evolution
                 transient + spread to halt at finite Y_8
                 yield-stress-dominated quasi-static late stage
```

For low-σ_y materials (Chuno) the Poiseuille analytical proposal happens
to be near-optimal: the recommended (W, H) = (4, 2) successfully breaks
the (η, σ_y) ridge in real release-spread MPM physics.  For high-σ_y
materials (Okonomi), the recommended (W, H) = (7, 7) is squat geometry
matching setup 1's aspect ratio — orthogonal under Poiseuille's
linearised model but not orthogonal under release-spread MPM physics.

**Future work**: replace analytical Poiseuille proposer with an
MPM-or-surrogate Hessian-based proposer.  Computing ∇²ℓ over GP at
candidate (W, H) is feasible (one batched GP forward per candidate);
amortised over many materials, the fixed cost is acceptable.

---

## 3. Sim-real gap

Our forward simulator (Taichi MPM) does not model:
- surface tension (relevant at small (W, H) and at flow-front edges)
- transient release-gate dynamics (frame 1 systematic offset)
- non-isothermal effects (cooling during spread)

This manifests as a residual ~5–15 % MPM-vs-real discrepancy on flow
distance per frame, *irrespective of inverse accuracy*.  The
`σ_obs_log = 0.20` floor in our likelihood absorbs some of this as
nominal observation noise, but a systematic bias is not modelled.

**We report θ̂ that recovers rheology truth, *not* θ̂ that minimises
MPM-real residual** — the two are mutually exclusive given a sim-real
gap.  Snapdiff Row 4 (Real vs MPM(θ̂)) is the visualisation of the gap;
it does *not* indicate inverse error when accompanied by a small Row 6
(MPM(θ̂) vs MPM(truth)).

**Mitigation directions**:
- Add surface tension to the Taichi MPM (engineering effort, not
  conceptual).
- Train a sim-to-real correction layer: small NN that maps
  *MPM-y → real-y* learnt from paired
  (rheometer-truth-θ, real-y_obs) tuples across a calibration set of
  materials.

---

## 4. Calibration coverage of the production model bank

As of 2026-05-07, 20 of 25 production gids
(`Models/v10_yshape_planB/state_gid_*`) are v2-partitioned and trained:

```
Fully trained + post-hoc c_k calibrated:  gid 0, 10, 24
Fully trained + default c_k = 1.0:        gid 1-9, 11-19, 23
Partial training (interrupted):           gid 20  (23/61 experts)
Untouched (still v1 partition):           gid 21, 22
```

Materials whose (W, H) routes to gids 20-22 will fall back to a v1
partition or a partial v2 partition; inverse may be unreliable in those
geometric regions.  Resume of training is straightforward:

```
python scripts/retrain_unused_gids.py --gids 20,21,22 \
    --bank Models/v10_yshape_planB
```

---

## 5. Optimization defaults trade-off

The production defaults `σ_obs_log = 0.20` + `disable_calib` were chosen
empirically to give consistent paper-grade σ_y identification on both
Chuno and Okonomiyaki.  Alternative defaults
(`σ_obs_log = 0.05` + `calib_scale enabled`) give:

- Chuno  : σ_y diff +6.4 % (vs +0.2 % with new defaults)
            but MPM(θ̂) ≈ MPM(truth) at sub-pixel level
- Okonomi: σ_y diff −4.2 % (vs +12.1 % with new defaults)
            but n = +93 %, η = −63 % → flowcurve diverges 3× at
            γ̇ = 100 s⁻¹

The new defaults sacrifice ~5 % σ_y precision on Chuno to avoid the
catastrophic flowcurve mismatch on high-σ_y materials.  Backward
compatibility is preserved via CLI flags.

---

## 6. Single-restart vs 5-restart

Single-restart CMA (default) is sufficient for materials with unique
loss-landscape basins.  5-restart is needed when (n, η) is ridge
degenerate: empirically Okonomi shows 32 % η dispersion across restarts,
meaning the global minimum location changes from seed to seed.

**Practical recommendation**: always use 5-restart for paper-grade
output.  The dispersion metric tells you whether single-restart would
have been sufficient (a posteriori).  With batched CMA eval, 5-restart
adds < 10 s per inverse → no reason to skip.

---

## 7. Frame weighting empirical

The default `frame_w = [0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0]`
down-weights early release-transient frames (1, 2) where the physics
deviates most from MPM and emphasises late "quasi-static" frames where
the inverse signal is cleanest.  These weights are inherited from
prior work and have not been re-derived for our pipeline.  A
data-driven re-weighting (e.g. inverse-Fisher-information per frame)
could modestly improve inverse accuracy.

---

## 8. Future work summary

1. **Third setup at extreme aspect ratio** to break (n, η) ridge for
   high-σ_y materials.
2. **Surrogate-Hessian-based setup-2 proposer** to replace analytical
   Poiseuille model.
3. **Sim-real gap correction layer** trained on rheometer-truth +
   real-y_obs paired calibration set.
4. **Active-learning infill in n,η-degenerate regions** for high-σ_y
   sub experts.
5. **Resume retraining gids 20-22** for full geometric coverage.
6. **Freshness experiment** (see
   [`FRESHNESS_EXPERIMENT_DESIGN_2026-05-07.md`](FRESHNESS_EXPERIMENT_DESIGN_2026-05-07.md))
   leveraging the ~20 s per-material inverse for high-temporal-resolution
   rheology kinetics, demonstrating a class of measurements impractical
   with conventional rheometry.
