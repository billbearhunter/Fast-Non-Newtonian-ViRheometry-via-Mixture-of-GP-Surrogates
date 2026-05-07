# Freshness Experiment Design — Meeting Notes 2026-05-07

## Problem statement (老师 meeting)

```
旧 path:  时间序列同一物质的 rheology → rheometer 也能测 → 为什么用 camera?
Hamamichi: 市贩物质同包装 → 参数稳定 → 一致性验证 (但不能展示新鲜度变化)
矛盾:     市贩 = 标准化 → 不展示新鲜度信号
          非市贩 / 现做 → 需要协议保证 reproducibility
```

## Reframing — temporal-resolution advantage

Rheometer ≠ camera in time resolution:

```
rheometer:   sample loading + measure + clean ≈ 30-60 min/data point
            max temporal resolution ≈ 1 pt/hour
camera:     release + film 8 frames + extract ≈ 1-2 min/data point
            max temporal resolution ≈ 1 pt/2 min
            → 30× advantage
```

**Paper framing**: camera is not a replacement for rheometer accuracy, but enables
a class of experiments rheometer cannot execute — high-temporal-resolution
freshness kinetics.

## Three design options (priority A > C > B)

### Option A: Hydrocolloid hydration kinetics (recommended)

```
Material:    powder + water (xanthan / agar / gelatin / carrageenan)
             Commercial powder (standardized) → mixed with water at t=0
             Hydration kinetics start immediately
Protocol:    t=0 mix powder + room-temp water (fixed ratio 0.5-1% w/v)
             t = 1, 3, 5, 10, 20, 30, 60 min — fresh aliquot, camera release-spread
Physics:     polymer chain disentanglement + H-bond formation → σ_y up 1-2 orders
Signal:      dramatic σ_y change within 60 min
Validation:  rheometer at t=5, 30, 60 min (3 sparse points — what rheometer can keep up with)
Comparison:  camera 8 pts in 60 min vs rheometer 1-2 pts → direct 30× resolution demo
```

**Paper claim**: "Camera inverse resolves σ_y kinetics at sub-10-min cadence,
30× faster than rheometer; consistent with rheometer at sparse validation points."

### Option B: Commercial material + controlled intervention

```
Material:    commercial (yogurt / pudding mix / custard)
Intervention: heat 5 min → cool → measure
             Commercial baseline + intervention → freshness-variable state
Protocol:    "intervention t=0" → t=1, 5, 15, 30, 60 min, fresh aliquot per point
Physics:     protein denaturation / starch gelatinization / gel formation → σ_y change
Signal:      controlled by intervention strength
Pros:        commercial = consistent starting point
Cons:        intervention protocol must be tightly defined
```

### Option C: Fresh-cooked food (Chuno-class) aging time series

```
Material:    fresh Chuno (rice paste cooled + retrogradation)
Protocol:    t=0 (just cooled) → t=10, 30, 60, 120 min
             rheometer in parallel
Physics:     starch retrogradation → σ_y rises slowly
Signal:      ~30-min timescale
Pros:        fits existing Chuno work (ref_Chuno_10min_2.7_2.5_1 already created)
Cons:        fresh-cooked = batch-to-batch variation → need strict recipe
             retrogradation slow → temporal resolution advantage less dramatic
```

## Paper triangle (target figure structure)

```
                  Temporal resolution advantage  (Option A data)
                 /                                            \
  vs rheometer consistency                          vs real food relevance
  (sparse validation points)                         (Option C Chuno data)
```

## Action items

1. [ ] Decide on Option A material (xanthan / agar / gelatin)
2. [ ] Define standard protocol (water temp, mix time, container size)
3. [ ] Camera protocol: same (W, H) as Chuno_10min (W=2.5, H=2.7) → routes to gid 0 (calibrated)
4. [ ] Rheometer side: protocol for sparse validation points (t=5, 30, 60 min)
5. [ ] First trial: 1 material × 3 time points to confirm signal magnitude

## Status
- `data/new_real_world_experiments_freshness/ref_Chuno_10min_2.7_2.5_1/settings.xml` exists (gid 0 calibrated)
- Future Option A dirs: `ref_Xanthan_<t>min_2.7_2.5_1/` or similar
