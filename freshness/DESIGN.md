# Freshness Experiment Design — 2026-05-14

σ_y(t) measurement on Class-A handmade materials via dam-break
camera-inverse at 20 s/timepoint, characterising both **hand-made
variability** and **freshness kinetics**. **Microwave + kettle-only**
lab, 6(+1) materials × replicate + technique sweep.

---

## 1. Theoretical foundation

The design rests on six established theoretical pillars, on top of which
we add one methodological combination.

### 1.1 Inverse problem + Hessian-orthogonal experimental design
- **Hamamichi (2023)** — dam-break + MPM-direct CMA-ES inverse for HB
  parameters; defines `propose_next_setup` (Hessian-orthogonal setup-2
  proposer), faithfully ported in `Optimization/libs/mechanism.py`.
- **Atkinson, Donev, Tobias (2007)** *Optimum Experimental Design with
  SAS* (Oxford) — A-/D-optimal design textbook.
- **Tarantola (2005)** *Inverse Problem Theory and Methods for Model
  Parameter Estimation* (SIAM) — Bayesian inverse framework.

### 1.2 Time-resolved rheology + kinetic fitting
- **Bates & Watts (1988)** *Nonlinear Regression Analysis and Its
  Applications* (Wiley) — (T_k, σ_∞) ± Laplace CI from covariance matrix.
- **Larson (1999)** *The Structure and Rheology of Complex Fluids* (OUP)
  — time-dependent rheology textbook.
- **Mewis & Wagner (2009)** "Thixotropy" *Adv Colloid Interface Sci*
  147–148: 214–227.

### 1.3 Foam aging (relevant to #1 meringue, #2 whipped cream)
- **Princen (1979)** "Highly concentrated emulsions: I. Cylindrical
  systems" *J Colloid Interface Sci* 71: 55–66 — foam σ_y ∝ (σ/R) ×
  f(φ) fundamental theory.
- **Princen (1983)** "Rheology of foams and concentrated emulsions: II"
  *J Colloid Interface Sci* 91: 160–175 — yield stress vs bubble
  fraction.
- **Mason, Bibette, Weitz (1995)** "Yielding and flow of monodisperse
  emulsions" *J Colloid Interface Sci* 179: 439–448 — universal scaling.
- **Stang & Schubert (1992)** "Bubble coalescence kinetics" *Chem Eng
  Technol* 15: 372–378 — coalescence timescale τ ∝ R²/D.
- **Dickinson (2010)** "Food emulsions and foams: stabilization by
  particles" *Curr Opin Colloid Interface Sci* 15: 40–49.

### 1.4 Egg/dairy protein thermal gelation (relevant to #4 custard)
- **Mleko, Foegeding (1999)** "Formation of whey protein gels at high
  pH" *J Food Sci* 64: 209–212 — milk protein thermal gel.
- **Croguennec, Nau, Brulé (2002)** "Influence of pH and salts on egg
  white gelation" *J Food Sci* 67: 608–614 — egg white denaturation.
- **Singh (2011)** "Aspects of milk-protein-stabilised emulsions" *Food
  Hydrocoll* 25: 1938–1944.

### 1.5 Starch + fat gel during cooling (relevant to #5 béchamel, #7 katakuriko)
- **Karim, Norziah, Seow (2000)** "Methods for the study of starch
  retrogradation" *Food Chem* 71: 9–36.
- **Miles, Morris, Ring (1985)** "Gelation of amylose" *Carbohydr Polym*
  5: 17–32.
- **Roos (1995)** *Phase Transitions in Foods* (Academic Press).

### 1.6 Gluten relaxation (relevant to #6 udon dough)
- **Bagley & Christianson (1986)** "Response of dough to uniaxial
  compression: stress relaxation properties" *Cereal Chem* 63: 220–223.
- **Dobraszczyk & Morgenstern (2003)** "Rheology and the breadmaking
  process" *J Cereal Sci* 38: 229–245.

### 1.7 Experimental methodology (replicate + dose-response)
- **Box & Draper (1987)** *Empirical Model-Building and Response
  Surfaces* (Wiley) — replicate + factor-sweep design.
- **Roache (1998)** *Verification and Validation in Computational
  Science and Engineering* — multi-anchor V&V.
- **Macosko (1994)** *Rheology: Principles, Measurements, and
  Applications* (Wiley-VCH) — rheometer protocol.

### 1.8 Our contribution (combination + niche claim)

**Method**: σ_y(t) tracking via **dam-break camera + GP-surrogate
inverse (20 s/inverse)**.

**Density correction (core trick)**: Training sim runs at ρ_sim = 1.2
g/cm³, but Class-A handmade materials have ρ_real that varies
batch-to-batch (especially aerated foams ρ ∈ [0.4, 0.7]). We weigh 50
mL per batch → ρ_real, post-hoc:

```
σ_y_real(t) = σ_y_sim(t) × ρ_real / 1.2
T_k_real    = T_k_sim                       (ρ-invariant)
```

T_k is invariant to ρ (numerator and denominator both carry the same
ρ-factor, cancelling), so the freshness kinetics claim is robust to
batch-to-batch ρ scatter.

**Niche claim** — our fast inverse unlocks three measurement classes
that rheometer + Hamamichi cannot do alone:

1. **Same-technique × N replicate**: quantify batch-to-batch scatter —
   rheometer is slow, and hand-made variability invalidates the
   "standardised σ_y_∞" claim (cross-batch scatter 50-200%).
2. **Different-technique × N sweep** (whip time / heat power / stir
   frequency): dose-response — rheometer is too slow for multi-parameter
   sweep.
3. **Same-batch × 6 timepoint trace**: kinetics (T_k) — Hamamichi
   MPM-direct at 30-60 min/inverse can't track T_k ≈ 30-90 min kinetics.

---

## 2. Step plan

### Step 1 — Material selection (Class A, microwave + kettle only)

| # | Material | Per-batch recipe | Volume | Class-A source | Freshness mechanism | T_k range | σ_y_∞ range (Pa) | ρ_real | ρ-rescale needed? |
|---|---------|----------------|--------|--------------|------------------|---------|---------------|--------|-------------------|
| 1 | **Meringue** | 4 egg whites + 160 g sugar + lemon drops | ~600 mL | whip time (2/4/6 min) × speed (M/H) | bubble coalescence (Princen 1983) | 1-3 h | 50-200 | 0.40-0.55 | ✓ |
| 2 | **Whipped cream 35%** | 300 mL heavy cream + 30 g sugar | ~500 mL | peak stiffness (soft/medium/stiff) | fat-foam coalescence | 30 min - 2 h | 100-300 | 0.45-0.65 | ✓ |
| 3 | **Homemade mayonnaise** | 2 yolks + 500 mL oil + 30 mL vinegar + 5 g salt | ~530 mL | oil addition rate (drop 5 min vs trickle 1 min) | emulsion oil-water separation (Mason 1995) | hours | 100-300 | 0.95 | × |
| 4 | **Homemade custard** | 3 eggs + 80 g sugar + 20 g flour + 500 mL milk | ~600 mL | whisk intensity between microwave bursts → scramble vs smooth | egg protein gel + cooling (Mleko 1999) | 30-60 min cooling | 50-300 | 1.05 | × |
| 5 | **Homemade béchamel** | 40 g butter + 40 g flour + 600 mL milk | ~650 mL | milk addition rate + whisk aggressiveness → lumps vs smooth | starch + fat gel + cooling | 30-60 min cooling | 100-500 | 1.05 | × |
| 6 | **Hand-kneaded udon dough (loose)** | 200 g flour + 200 mL water + 4 g salt | ~400 mL | kneading time (5/10/20 min) × force | gluten relaxation (Bagley 1986) | 30 min - 2 h | 200-800 | 1.15 | × |
| (7) | **Homemade katakuriko-kuzu** (option) | 50 g potato starch + 30 g sugar + 500 mL water | ~550 mL | stir timing between microwave bursts | starch retrogradation + cooling (Karim 2000) | 15-60 min cooling | 50-300 | 1.05 | × |

**Coverage axes**:
- foam (1, 2) — sugar-stabilised protein vs fat-stabilised
- emulsion (3) — yolk + oil
- thermal gel + cooling (4, 5, ±7) — egg / starch+fat / starch
- gluten relaxation (6)
- Optional concentration scan: whipped cream 35% vs 47%; udon flour:water
  1:1 vs 1:1.2

### Step 2 — Setup geometry pool (one-time, before any capture)

Per Hamamichi 2023 protocol verbatim: **setup 1 random (no information),
setup 2 derived from setup 1 inverse**.

**(W, H) ∈ [2, 7] × [2, 7] cm box, all 25 gids v2-partitioned and
post-hoc c_k calibrated** (2026-05-10 audit); any (W, H) inside the box
is covered.

**Practical mould pool**:
```
W: 2.5, 4.0, 5.5, 6.5  cm
H: 2.5, 4.0, 5.5, 6.5  cm
```
Prepare ~9 moulds spread across the 4×4 grid (corner + interior points).

**Per material session**:
- **Setup 1**: uniform random draw from the mould pool.
- **Setup 2 per t_i**: `propose_next_setup(θ̂_1(t_i), setup1)` → continuous
  (W₂, H₂) → snap to the closest mould **not** equal to setup 1.

Camera + `camera_params.xml` calibrated daily, fixed during a session
(no recalibration between materials/timepoints).

### Step 3 — Per-material batch preparation (microwave 500 W + kettle only)

Detailed recipes:

```
1. Meringue
   4 egg whites + 160 g sugar + lemon drops
   → no heating; hand mixer at medium speed
   → whip for X min (X = 2/4/6, technique-sweep parameter)
   → transfer to container, hold at 5 °C, t = 0

2. Whipped cream 35%
   300 mL heavy cream (chilled) + 30 g sugar
   → hand mixer at medium speed, no heating
   → whip to X stiffness (X = soft/medium/stiff, visually judged)
   → t = 0, hold at 5 °C

3. Homemade mayonnaise
   2 yolks + 30 mL vinegar + 5 g salt (mix at room T 25 °C)
   → manual whisk
   → add 500 mL salad oil at rate X (X = drop 5 min / trickle 1 min)
   → t = 0, hold at room T

4. Homemade custard
   3 eggs + 80 g sugar + 20 g flour + 500 mL milk → whisk 30 s to mix
   → microwave 500 W × 1 min × 5 cycles, with whisk between of intensity X
     (X = weak 5 s / medium 15 s / strong 30 s)
   → refrigerate at 5 °C, t = 0 = refrigeration start

5. Homemade béchamel
   40 g butter → microwave 30 s → melted → +40 g flour, whisk 30 s →
   microwave 30 s → +100 mL milk, whisk with aggressiveness X → microwave
   1 min → +remaining 500 mL milk, microwave 1 min × 3 with whisk X
   → t = 0 = preparation complete, room T → fridge

6. Hand-kneaded udon dough (loose)
   200 g flour + 4 g salt mixed
   → kettle boiled water 200 mL → cool 5 min to room T → add
   → hand-knead for X min (X = 5/10/20, technique-sweep parameter)
   → t = 0 = kneading complete, hold at room T

7. Homemade katakuriko-kuzu (option)
   50 g potato starch + 30 g sugar + 500 mL water (cold) → stir 30 s
   → microwave 500 W × 30 s × 6 cycles, with stir timing X
     (X = immediate / 5 s after / 10 s after)
   → refrigerate at 5 °C, t = 0 = refrigeration start
```

**Why the microwave-cooked items (4, 5, 7) are stronger Class A**:

| Stovetop | Microwave + manual whisk |
|---------|-----|
| Continuous heat + constant stir → uniform | 30-60 s bursts → centre overheats / edges stay cool |
| Heat power numerically fixed | Wattage fixed but spatial distribution uneven |
| Single-pass cooking | Multi-step; each manual intervention varies in intensity |

Two different people making béchamel/custard in microwave can produce
**σ_y_∞ scatter of 2-3×** (lumpy vs smooth). **This is exactly the
hand-variability we want to quantify.**

**ρ_real measurement**: Right after batch preparation → fill a 50 mL
measuring cup → weigh (g) → ρ_real = mass/50 → record in
`batch_metadata.json`.

**Equipment**: hand mixer, manual whisk, microwave 500 W, kettle, fridge
5 °C, room-T (25 °C) holding containers, electronic scale ±0.1 g,
measuring cups 50/100/200 mL, multiple containers. **No stovetop.**

### Step 4 — Per-material capture session

**Two modes**:

| Mode | Goal | Batch count | Captures per material |
|------|------|------|-----------|
| **A — Replicate** (same technique × 3 batch, full 6-timepoint trace) | Cross-batch repeatability + kinetics | 3 | 6 t × 2 setup × 3 batch = 36 |
| **B — Sweep** (3 techniques × single plateau t=240 min) | Technique → σ_y_∞ dose-response | 3 | 1 t × 2 setup × 3 tech = 6 |

**Timepoints** (Mode A): t = {15, 30, 60, 90, 150, 240} min (6 points,
~4 h session)

**Per-t_i protocol** (Hamamichi-derived, T_pair = 8 min):

```
(setup 1 mould pre-selected from pool before session start)

t_i − 5 min   Take aliquot ~250 mL, fill setup 1 mould
t_i           Setup 1 release (3 s)
t_i + ↓:      ─ Extract video → y_obs_1                       (~5-7 min, auto)
              ─ Run setup-1-alone inverse → θ̂_1(t_i)          (~7 s)
              ─ propose_next_setup(θ̂_1, setup1) → (W₂, H₂)    (~1 s)
              ─ Snap to closest pool mould ≠ setup 1            (~10 s)
              ─ Take fresh aliquot, fill that mould             (~30 s)
t_i + 8 min   Setup 2 release (3 s)
─────────────────────────────────────
T_pair ≈ 8 min between setup 1 and setup 2
```

**Joint-validity gate**: σ_y change within T_pair = 8 min must be < CI
half-width (~15%). For first-order kinetics this requires **T_k ≥ 50 min**.

**T_k estimation + early-abort** (Mode A, after first 3 captures at
t = 15, 30, 60 min):

```
1. Run setup-1-alone inverse on 3 captures → 3 (σ_y, η, n) values
2. Fit σ_y(t) = σ_y_∞ (1 − exp(−t/T_k))     [exp first, §1.2]
3. Check:
   ├─ σ_y_∞ ∈ [10, 500] Pa?      Yes → continue
   │                              No  → adjust recipe ±30-50 %, remake batch
   ├─ T_k ≥ 50 min?               Yes → continue (joint mode)
   │                              No  → shorten to T_pair = 4 min (single-
   │                                    setup only, σ_y prior + identifiability
   │                                    flag); if still fails → drop material
   ├─ sub bbox contains σ_y(60)?  Yes → continue
   │                              No  → drop material
   └─ all pass → continue with t = 90, 150, 240 min
```

**Mode B protocol**: same material × 3 technique parameters → each at
plateau (t = 240 min) capture → compare σ_y_∞(tech1) vs σ_y_∞(tech2)
vs σ_y_∞(tech3).

### Step 5 — Joint inverse offline + ρ-rescale (~10 min/material)

After all captures, run paper-grade joint inverse on every pair:

```bash
python scripts/run_y8q_prior.py --joint-only --restarts 5 \
    --data-root data/freshness_handmade \
    --material <Material>_<batch>_<t>min \
    --out scripts/run_kinetics_<Material>_<batch>.json
```

→ θ̂(t_i) per timepoint with 95% Hessian Laplace CI (sim-space).

**Post-hoc ρ-rescale** (per batch):

```python
import json
meta = json.load(open('batch_metadata.json'))
rho_real = meta[batch_id]['rho_real']     # g/cm^3, measured at Step 3

sigma_y_real_per_t = sigma_y_sim_per_t * (rho_real / 1.2)
eta_real_per_t     = eta_sim_per_t * (rho_real / 1.2) ** (1.0 / (2 - n_sim))  # approx
T_k_real           = T_k_sim                                                  # invariant
```

Fit kinetic model (Bates & Watts 1988):

```python
from scipy.optimize import curve_fit
def model_exp(t, sigma_y_inf, T_k):
    return sigma_y_inf * (1 - np.exp(-t / T_k))

popt, pcov = curve_fit(model_exp, t_array, sigma_y_real_array,
                        sigma=ci_halfwidth_array, absolute_sigma=True)
T_k, sigma_y_inf = popt[1], popt[0]
T_k_err = np.sqrt(pcov[1, 1])
```

If exponential residual > 2× CI band on > 30% of points: try **Avrami
n > 1** (sigmoid induction) or **bi-phasic** (e.g. fast-collapse +
slow-coalesce per Stang-Schubert 1992).

### Step 6 — Validation anchors (per material)

| Anchor | Mode A | Mode B | Theory |
|---|:---:|:---:|---|
| **A. Hamamichi MPM-direct inverse on plateau y_obs (t=240)** | ✓ primary | ✓ | Same y_obs, only inverse algorithm differs; validates absolute σ_y. Workflow: copy `ref_*_240min_*` to Hamamichi sibling repo, run their inverse, compare θ̂. |
| **B. Rheometer plateau (24 h)** | ✓ (only 3, 4, 5, 6, 7 — foams excluded) | ✓ | Macosko 1994; compare σ_y_rheo vs σ_y(240). |
| **C. Replicate consistency (Mode A intra ×3)** | ✓ | n/a | Box & Draper 1987; CV < 25% target. |
| **D. Visual MPM(θ̂) forward vs original video** | ✓ | ✓ | SIGGRAPH-style Snapdiff overlay. |
| **E. ρ_real measurement consistency** | ✓ | ✓ | Quantifies cross-batch ρ scatter; rescale-input stability check. |
| **F. Internal dynamic fit** | ✓ | n/a (single t) | Avrami / exp residual < 2× CI band on > 70% of points. |
| **G. Technique dose-response monotonic** | n/a | ✓ | Mode B primary validation: σ_y_∞ vs technique parameter monotonic. |

**Per-material pass criteria**:
- **Mode A**: A pass (≤ 25% σ_y_∞ deviation vs Hamamichi) AND (B pass OR
  D + F at SIGGRAPH-quality) AND C intra-consistency (CV < 25%) AND F
  residual OK.
- **Mode B**: G monotonic AND A pass at ≥ 2/3 technique points.

### Step 7 — Cross-material trends

**H. Foam physics sanity (Princen 1983)**: σ_y_∞(meringue) /
σ_y_∞(whipped cream) should match bubble fraction × interfacial tension
expectation.

**I. Per-material dose-response** (Mode B):
- Meringue whip time ↑ → bubble size ↓ → σ_y_∞ ↑ (Princen-Mason scaling)
- Whipped cream peak stiffness ↑ → φ_air ↑ → σ_y_∞ ↑
- Mayonnaise drop rate (slower) → smaller oil droplets → σ_y_∞ ↑ + stability ↑
- Custard weaker whisk → more scrambling → σ_y_∞ ↑
- Béchamel weaker whisk → more lumps → σ_y_∞ ↓ (broken structure)
- Udon longer kneading → stronger gluten network → σ_y_∞ ↑

**J. Cross-mechanism σ_y_∞ ranking**:
σ_y_∞(béchamel, udon) > σ_y_∞(custard) > σ_y_∞(mayonnaise) >
σ_y_∞(meringue, whipped cream) (cooked gel > emulsion > aerated foam,
post ρ-rescale)

**Method validation passes** if Step 6 ≥ 4/6 materials pass AND Step 7 H
≥ 4/6 materials monotonic AND J ranking correct.

---

## 3. Time budget

```
Step 1   Material selection                            settled
Step 2   Mould preparation (one-time)                  ½ day
Step 3+4 Per-material capture
         Mode A: 1 day × 3 replicate × 6 mat = 18 lab day (full)
                 OR 1 day × 1 replicate × 6 mat = 6 lab day (minimal)
         Mode B: ½ day × 3 tech × 6 mat = 9 lab day
                 (shareable with Mode A plateau capture, effectively +3 lab day)
         Failure-recovery: each failure +½ day
Step 5   Offline joint inverse + ρ-rescale             ~15 min total (GPU)
Step 6   Validation
         A. Hamamichi MPM (sibling repo)                ~3-6 GPU-h (plateau only)
         B. Rheometer plateau × 5 non-foam              ½ day
         C-G: automated                                  synced with Step 5
Step 7   Cross-material trend analysis                  ½ day

──────────────────────────────────────────
Mode A only (1 replicate):  ~7 lab day + 1 GPU overnight     [minimal]
Mode A only (3 replicate):  ~19 lab day + 1 GPU overnight    [strong replicate claim]
Mode A + B:                 ~9-22 lab day + 1 GPU overnight  [full Class-A claim]
```

**Time-pressure fall-backs**:
1. Mode A: drop to 1 replicate → only kinetics shown, no batch-scatter
2. Drop Mode B → only replicate + kinetics, no technique dose-response
3. Drop material 7 (katakuriko) → 6 materials
4. Drop material 6 (udon) → 5 materials, food focus

---

## 4. Deliverables

```
data/freshness_handmade/
  # Mode A (replicate + kinetics)
  ref_<Material>_<batch>_<t>min_<W1>_<H1>_1/  # × 6 t × 3 batch × 6 mat = 108 dirs
  ref_<Material>_<batch>_<t>min_<W2>_<H2>_2/  # × 6 t × 3 batch × 6 mat = 108 dirs

  # Mode B (technique sweep, plateau only)
  ref_<Material>_<tech>_240min_<W1>_<H1>_1/   # × 3 tech × 1 t × 6 mat = 18 dirs
  ref_<Material>_<tech>_240min_<W2>_<H2>_2/   # × 3 tech × 1 t × 6 mat = 18 dirs

  # Rheometer (non-foam only)
  ref_<Material>_plateau_24h_*.csv            # × 5 (no meringue/whipped cream)

  # Per-batch metadata (ρ_real, technique params)
  batch_metadata.json

scripts/
  run_kinetics_handmade.py    # per-batch + per-t_i joint inverse + ρ-rescale
  fit_T_k.py                  # Bates & Watts CI (existing, reused)
  plot_kinetics_handmade.py   # σ_y(t) trace × replicate + tech sweep panel
  rho_rescale.py              # post-hoc σ_y_real, η_real, T_k_real

docs/figs/
  kinetics_panel_6materials_handmade.png  # MAIN: σ_y(t) × 6 mat × 3 replicate
  tech_sweep_panel.png                    # σ_y_∞ vs technique param × 6 mat (Mode B)
  rho_rescale_validation.png              # ρ_real scatter + rescale consistency
  validation_matrix.png                   # 6 mat × 7 anchor pass/fail
  hamamichi_plateau_xcheck.png            # A. Hamamichi vs ours @ t=240
```

---

## 5. Pre-registration checklist

Lock down before any capture:

- [ ] Materials: **1 meringue, 2 whipped cream, 3 mayonnaise, 4 custard,
      5 béchamel, 6 udon dough** (±7 katakuriko)
- [ ] Setup 1 per batch: uniform random (W, H) from ~9-mould pool
- [ ] Setup 2 per t_i: derived from `propose_next_setup(θ̂_1, setup1)`
      (Hamamichi 2023), snapped to closest pool mould ≠ setup 1
- [ ] T_pair ≈ 8 min (per-time-point Hessian, manual extract ~5-7 min)
- [ ] Timepoints {15, 30, 60, 90, 150, 240} min per material (Mode A)
- [ ] **Mode A**: 3 replicate / same technique / 6 t / 6 materials =
      108 setup 1 + 108 setup 2 = 216 captures
- [ ] **Mode B**: 3 technique params / 1 plateau t / 6 materials =
      18 setup 1 + 18 setup 2 = 36 captures
- [ ] **ρ_real measurement**: each batch right after prep, 50 mL → weigh
      → record ρ_real in batch_metadata.json
- [ ] **Post-hoc rescale**: σ_y_real = σ_y_sim × ρ_real / 1.2; T_k
      unchanged
- [ ] **Early abort** after first 3 captures: T_k ≥ 50 min AND σ_y_∞ ∈
      [10, 500] Pa AND sub bbox contains σ_y
- [ ] **Validation**:
  - A. Hamamichi MPM at plateau (t=240) per material — sibling repo
  - B. Rheometer plateau (non-foam only: 3, 4, 5, 6, 7)
  - C. Replicate CV < 25% (Mode A intra)
  - D. Visual MPM(θ̂) match at SIGGRAPH quality
  - E. ρ_real measurement consistency check
  - F. Dynamic fit residual < 2× CI band on > 70% of points
  - G. Mode B technique dose-response monotonic
- [ ] Pass = A + (B or D + F) + C per material; ≥ 4/6 must pass
- [ ] Cross-material: H (foam sanity), I (per-material dose-response),
      J (mechanism ranking)
- [ ] Kinetic model: exp first; sigmoid / Avrami n > 1 / bi-phasic if
      residual demands
- [ ] Statistical fit: `scipy.optimize.curve_fit` weighted (sigma=CI
      half-widths, absolute_sigma=True); report (T_k, σ_y_∞) ± 1-σ from
      covariance (Bates & Watts 1988)

Post-registration deviations require written amendment.

---

## 6. References

1. Atkinson, A.C., Donev, A.N., Tobias, R.D. (2007) *Optimum Experimental
   Design with SAS*. Oxford University Press.
2. Bagley, E.B., Christianson, D.D. (1986) "Response of dough to uniaxial
   compression: stress relaxation properties" *Cereal Chem* 63: 220–223.
3. Bates, D.M., Watts, D.G. (1988) *Nonlinear Regression Analysis and Its
   Applications*. Wiley.
4. Box, G.E.P., Draper, N.R. (1987) *Empirical Model-Building and
   Response Surfaces*. Wiley.
5. Croguennec, T., Nau, F., Brulé, G. (2002) "Influence of pH and salts
   on egg white gelation" *J Food Sci* 67: 608–614.
6. Dickinson, E. (2010) "Food emulsions and foams: stabilization by
   particles" *Curr Opin Colloid Interface Sci* 15: 40–49.
7. Dobraszczyk, B.J., Morgenstern, M.P. (2003) "Rheology and the
   breadmaking process" *J Cereal Sci* 38: 229–245.
8. Hamamichi, S. et al. (2023) [dam-break + MPM CMA-ES inverse paper —
   replace with actual citation].
9. Karim, A.A., Norziah, M.H., Seow, C.C. (2000) "Methods for the study
   of starch retrogradation" *Food Chem* 71: 9–36.
10. Larson, R.G. (1999) *The Structure and Rheology of Complex Fluids*.
    Oxford University Press.
11. Macosko, C.W. (1994) *Rheology: Principles, Measurements, and
    Applications*. Wiley-VCH.
12. Mason, T.G., Bibette, J., Weitz, D.A. (1995) "Yielding and flow of
    monodisperse emulsions" *J Colloid Interface Sci* 179: 439–448.
13. Mason, T.G., Bibette, J., Weitz, D.A. (1996) "Elasticity of
    compressed emulsions" *Phys Rev Lett* 75: 2051–2054.
14. Mewis, J., Wagner, N.J. (2009) "Thixotropy" *Adv Colloid Interface
    Sci* 147–148: 214–227.
15. Miles, M.J., Morris, V.J., Ring, S.G. (1985) "Gelation of amylose"
    *Carbohydr Polym* 5: 17–32.
16. Mleko, S., Foegeding, E.A. (1999) "Formation of whey protein gels at
    high pH" *J Food Sci* 64: 209–212.
17. Princen, H.M. (1979) "Highly concentrated emulsions: I. Cylindrical
    systems" *J Colloid Interface Sci* 71: 55–66.
18. Princen, H.M. (1983) "Rheology of foams and concentrated emulsions:
    II. Experimental study of the yield stress and wall effects"
    *J Colloid Interface Sci* 91: 160–175.
19. Roache, P.J. (1998) *Verification and Validation in Computational
    Science and Engineering*. Hermosa Pub.
20. Roos, Y.H. (1995) *Phase Transitions in Foods*. Academic Press.
21. Singh, H. (2011) "Aspects of milk-protein-stabilised emulsions"
    *Food Hydrocoll* 25: 1938–1944.
22. Stang, M., Schubert, H. (1992) "Bubble coalescence kinetics in foam
    rheology" *Chem Eng Technol* 15: 372–378.
23. Tarantola, A. (2005) *Inverse Problem Theory and Methods for Model
    Parameter Estimation*. SIAM.
