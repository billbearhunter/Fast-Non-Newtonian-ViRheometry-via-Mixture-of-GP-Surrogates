# Round-2 Uniform Infill — Status Snapshot

Last updated: 2026-05-03

---

## 1. Current backgrounds

```
job brwfwshu7    master worker (machine 0)
plan:            cell_plan_q150_2machines.csv
state:           running, healthy
write rate:      ~4.5 s/sim
```

Last 1h check: master CSV grew from ~14k → 22.4k rows. ETA ~15h to finish
its assigned 22,760 sims.

```
worker 1 (remote): ready to start, has WORKER1_NEXT_TASK.md
                   needs to pull cell_plan_q150_2machines.csv first
machine 2:         dropped (poor performance);
                   its 3,003 already-completed rows still count toward pool
```

---

## 2. Data inventory (as of write)

```
Source                              Rows        Notes
─────────────────────────────────────────────────────────────────────
synthetic_splits_v2 (orig pool)    369,715    train+val+test
machine0 worker CSV                 22,441    持续写中 (master)
machine1 worker CSV                  5,166    Round-1 完成,等切换 Q=150
machine2 worker CSV                  3,003    停用,数据保留
─────────────────────────────────────────────────────────────────────
COMBINED                           400,325    (含重复过填)
B=4 non-empty cells                    866    /1024
```

---

## 3. Uniform-extractable size (current data)

Three ways to count "fully uniform" subset (B=4 grid):

```
Method                              Q=100      Q=150 (target)
────────────────────────────────────────────────────────────────
A. 严格 (cells_at_Q × Q,丢稀疏)     67,700      80,250  ★ 最优
B. min(count, Q) per cell          72,195      99,730
C. 全 866-cell 饱和 (理想)         86,600     129,900  ← 远期目标
   要补足的 deficit                 14,405      30,170
```

**Strategy A is the actionable one** — what we can extract right now for
training: **80,250 pts at Q=150**.

---

## 4. Round-2 plan progress

Current plan: `cell_plan_q150_2machines.csv` (B=4, Q=150, 2 machines)

```
machine_id     cells   plan deficit   in-plan done   in-plan remaining
m0 (master)     249       22,760        12,653          12,013
m1 (worker 1)   250       22,565         1,520          21,045
─────────────────────────────────────────────────────────────────────
TOTAL           499       45,325        14,173          33,058 net
```

(out-of-plan rows in master CSV: ~9,788 in cells already-saturated or in
worker 1's slice — preserved, plan deficit already reduced for these.)

---

## 5. Banks built (data_infill outcome banks)

| Bank | Pool used | Subs | Trained | Notes |
|---|---|---|---|---|
| `v10_yshape_v3` | 334k (sparse-sub-补全后) | 496 | ✅ 288/496 | **生产** |
| `v10_yshape_v3p1_partial16k` | 386k (v3 + 16k partial) | 510 | ⏸ 50/510 | training stopped early |
| `v10_yshape_uniform_b4q50` | 21k uniform | 184 | ✅ 全 | yesterday's experiment |
| `v10_yshape_uniform_b4q50_plus16k` | 37k (21k+16k smoke) | 188 | ✅ 187/188 | smoke test bank |
| `v10_yshape_uniform_strict_q150` | 75k strict uniform | 395 | ❌ 未训 | partition only |

`SMOKE_RESULT_PARTIAL16K.md` has the 3-way comparison
(v3 vs uniform_b4q50 vs smoke).

---

## 6. Validated findings so far

1. **16k corner data benefit confirmed (smoke test 21k+16k vs 21k)**:
   - Carbonara 1253% → 489% (massive improvement) ⭐
   - Cobb 40.5% → 24.7%
   - JCabbage 120% → 66%
   - Median: +0.34% (lots of noise on main-diagonal materials)
   - Confirms corner infill is right direction for ridge-prone materials

2. **Strict uniform 21k alone insufficient**:
   - Median fc 35.7% vs v3's 22.9% (9/12 worse)
   - Need both uniformity AND data quantity

3. **v3 production winner remains**: median fc% 22.9% with σY two-stage
   prior + saturation guard

4. **Data uniformity vs sub size trade-off**:
   - v3 (334k full): median sub N=505, max 2660 (大过填)
   - strict uniform Q=150 (75k): median sub N=148, max 778, **完全消除过填**
   - But strict uniform has 14% subs at N<80 (corner sparseness shows up)

---

## 7. Sub N distribution at strict uniform Q=150 (75k bank)

```
N min:        40
N p10:        71
N median:    148
N p75:       250
N max:       778      (vs v3 max 2660 — no overfill)
N mean:      187

Buckets:
  N <  80:     57 subs (14.4%)  ← 偏小,GP 训练 marginal
  N ∈ [80,150):143 subs (36.2%)
  N ∈ [150,300):126 subs (31.9%)
  N ∈ [300,500): 65 subs (16.5%)
  N ≥ 500:        4 subs ( 1.0%)
```

Total subs: 395 (vs v3: 496).

---

## 8. Pending / Blocked items

| Item | Owner | Status |
|---|---|---|
| Worker 1 switch to Q=150 plan | user/worker1 host | waiting |
| Round-2 Q=150 完整 (130k uniform) | running | ~30k more sims |
| Train v3.1_partial16k (510 subs) | next? | partial 50/510 trained, stopped |
| Train strict_uniform_q150 (395 subs) | next? | partition done, not trained |
| Re-run RW inverse + 4-way compare | next | waiting on Round-2 完成 |

---

## 9. Decision log

```
2026-05-02 ~21:00  Round-2 plan generated (Q=100, 39,923 sims, 3 machines 3:1:1)
2026-05-02 ~21:30  Master worker started (job bd1blcjmz)
2026-05-03 ~15:00  Worker 1 + 2 完成 partial Round 1 (8.4k + 5.2k + 3.0k)
2026-05-03 ~16:00  Smoke test (21k + 16k partial) — confirmed corner data 有用
2026-05-03 ~17:00  v3.1_partial16k partition + train (stopped at 50/510)
2026-05-03 ~22:00  Generated strict uniform Q=150 (75k) bank (partition only, not trained)
2026-05-03 ~23:00  Switched master to Q=150 plan (cell_plan_q150.csv)
2026-05-03 ~23:30  Switched master to 2-machine plan (cell_plan_q150_2machines.csv)
                   machine 2 dropped due to poor performance
2026-05-03 ~00:30  Master worker now on cell_plan_q150_2machines.csv (job brwfwshu7)
                   ETA ~25h to finish 22,760 sims
2026-05-03 ~12:30  Status check: master 22,441 rows, in-plan 55.6%,
                   ETA refined to ~15h
```

---

## 10. What to do when Round-2 completes

```
Step 1. Stop master worker
Step 2. Pull worker 1 CSV from remote machine
Step 3. Merge: original pool + machine0 + machine1 + machine2 CSVs
Step 4. Re-partition (k_len=5, k_sh_max=5, min_n_per_sub=80)
        → expect ~530 subs / 25 gids (slightly more than v3 due to extra data)
Step 5. merge_tiny_subs (--threshold 100)
Step 6. Train all GPs on dedicated GPU (~30-45 min, no master worker contention)
Step 7. Run RW inverse 12 mats with CMA + σY prior (production winner config)
Step 8. Build 4-way comparison: v3 / v3.1_partial16k / smoke_21k+16k / final_q150
Step 9. Update Status MD with new fc% medians and per-material results
```

If Round-2 isn't fully completed but worker 1 finishes its slice, can do
the same pipeline with whatever data is collected — still > current 75k
strict uniform.
