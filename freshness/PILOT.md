# Pilot Protocol — 1-hour/material × 4-5 材料

**目的**: 快速验证 freshness experiment 整套 pipeline。比完整 paper-grade
design ([DESIGN.md](DESIGN.md)) 大幅精简, 每材料 1 hour lab work, 用
**intermediate state** + **2-timepoint (t_0, t_0+30)** + 3 setups, 半 lab
day 跑完全 5 材料。

---

## 核心简化思路

| 维度 | 完整 design | Pilot |
|------|------------|-------|
| 材料数 | 6 | **4-5** |
| Timepoint / 材料 | 6 (t = 15, 30, 60, 90, 150, 240 min) | **2 (t_0 ≈ 10 min, t_0+30 ≈ 40 min)** |
| Setups / timepoint | 2 (setup 1 + setup 2 joint) | **3 (+ setup 3 hold-out)** |
| Replicate / 材料 | 3 (Mode A) | **1 (per material per session)** |
| Lab time / 材料 | 4 hour | **1 hour** |
| 总 lab time | 18+ days | **5 hours (= ½-¾ day)** |
| 食材状态 | 完成态 | **intermediate state** (e.g. 半 whipped, 半熟, partial emulsify) |

**Intermediate state insight**: 不需做到完成态。手作 "**做到一半**" 反而**更
A 类** —"啥时候停" 高度主观, σ_y_∞ 跨 batch 散布 3-5× (vs 终态 ~2×)。

---

## 5 材料 set (按失败风险升序, 先易后难)

| 顺序 | 材料 | 配方 (intermediate state) | T_k 期 | σ_y_∞ 期 | A 类源 | ρ_real | ρ-rescale? | 失败风险 |
|:---:|------|---------------------------|--------|--------|------|--------|-----------|---------|
| **1** | **片栗粉糊 partial** | 片栗粉 50 g + 水 400 mL, microwave 30s ×3 (partial gelatinize), 冷却 | 15-45 min | 50-300 Pa | microwave burst 数 + stir timing | 1.05 | × | ⭐ 低 |
| **2** | **半熟カスタード** | 全卵 2 + 牛乳 200 mL + 砂糖 30 g + 薄力粉 10 g, microwave 30s ×2-3 (不完全 set) | 30-60 min cooling | 50-300 Pa | microwave burst 数 + whisk 强度 | 1.05 | × | ⭐⭐ 中 |
| **3** | **半 whipped 生クリーム** | 生クリーム 200 mL + 砂糖 20 g, whip **30 s only** (soft peak) | 15-60 min foam decay | 30-150 Pa | whip 秒数 (15/30/60 s) | 0.7-0.8 | ✓ mild | ⭐⭐⭐ 中高 |
| **4** | **partial 卵黄 emulsion** | 卵黄 2 + 油 50 mL only (不到 full mayo), hand-mix 30 s | 20-60 min separation | 10-100 Pa | mix 秒数 + 油加入量 | 0.98 | × | ⭐⭐⭐ 中高 |
| **5** | **打蛋白 no sugar** | 蛋白 2 個 only, whip 30/60/120 s (无糖, fast decay) | **5-30 min** ⚡ | 30-250 Pa | whip 秒数 | 0.3-0.7 | ✓ 较强 | ⭐⭐⭐⭐ 高 |

**鸡蛋 economy**: 材料 2 用全卵; 材料 4 用 yolk; **材料 5 用 white (= 材料 4 副产)** → 4 個鸡蛋 cover 三个材料。

---

## 1-hour walkthrough — 片栗粉糊 (Day 1, easiest)

### 准备 (5 min)

```
T-5    setup 1 (4.0, 5.5), setup 3 (3.0, 6.0) ← np.random.choice 从模具池
       食材: 片栗粉 50 g, 水 400 mL
T-3    9 模具洗净干燥, camera ready, batch_metadata.json template
```

### Batch prep (5 min)

```
T0     片栗粉 + 水 → ボウル混合 30 s 直到 lump-free
T0+0:30  microwave 500W × 30 s → 取出 stir 5 s
T0+1     microwave 500W × 30 s → stir 5 s
T0+1:30  microwave 500W × 30 s → stir 5 s  (partial gelatinize: 透明→白浊)
T0+2     倒入容器, 量 50 mL → 称 → ρ_real ≈ 1.05
         t_0 written to batch_metadata.json (ISO time)
```

### Capture t_a = T0+10 (15 min)

```
T0+10   aliquot 250 mL → setup 1 mould (4.0, 5.5) → release → 录像
T0+10-15 video extract → y_obs_1_a
T0+15   setup-1-alone inverse → θ̂_1_a = (n, η, σ_y)
        propose_next_setup(θ̂_1_a, setup1) → (W2, H2)
        snap 池中 (≠ setup1, ≠ setup3) → setup 2_a
T0+15   aliquot → setup 2 → release
T0+15-20 extract → y_obs_2_a
T0+20   joint inverse (setup1 + setup2) → θ̂_joint_a + Hessian Laplace CI
T0+20   aliquot → setup 3 mould (3.0, 6.0) → release  [hold-out]
T0+20-25 extract → y_obs_3_a  (不参与 inverse)
```

### 等待 (15 min)

```
T0+25-40  整理 mould (洗净干燥准备 t_b 重用)
```

### Capture t_b = T0+40 (15 min)

```
T0+40   aliquot → setup 1 (4.0, 5.5 again) → record
T0+40-45 extract → y_obs_1_b
T0+45   setup-1-alone inverse → θ̂_1_b
        propose → setup 2_b (可能跟 setup 2_a 不同)
T0+45   aliquot → setup 2 → release
T0+50   joint inverse → θ̂_joint_b
T0+50   aliquot → setup 3 (same 3.0, 6.0) → release  [hold-out]
T0+55   done
```

### Cleanup + offline (5 min)

```
T0+55-60  洗模具, sync 数据, 隔夜 GPU 跑 batch processing:
            python scripts/run_kinetics_handmade.py \
              --batch katakuri_2026-05-XX \
              --bank Models/yshape_mogp_production
```

### 期待 output

```
σ_y(t_a ≈ 10 min) = 60 ± 10 Pa    (温, partial gel)
σ_y(t_b ≈ 40 min) = 150 ± 15 Pa   (冷, retrog 完成)
freshness rate = +3.0 Pa/min        (正方向)

hold-out residual @ setup 3 (3.0, 6.0):
  t_a: 15 % median, 22 % p90  ✓
  t_b: 13 % median, 19 % p90  ✓
```

---

## 半 lab day routine (5 材料)

```
8:30—9:30   片栗粉糊 partial          (pantry, microwave + cool)
9:30—10:30  半熟カスタード             (egg + milk + flour + sugar)
[休憩 30 min, sync 数据]
11:00—12:00 半 whipped 生クリーム      (cream, whip 30 s)
12:00—13:00 partial 卵黄 emulsion      (yolk + 油 mix 30 s) ← 副产 2 個 whites
13:00—14:00 打蛋白 no sugar            (whites from above, whip 30/60/120 s)

14:00—15:00 GPU joint inverse (~30 s 总), ρ-rescale 分析, hold-out forward
```

---

## Dry-run 选项 (推荐先做)

正式 1-hour session 前, 用 1 batch 片栗粉糊 跑 **single-capture pilot**:

```
T0    prepare batch (5 min)
T0+10  setup 1 single release → record → inverse → check σ_y ∈ [10, 500] ✓
       end, ~15 min total.
```

→ 验证 camera 校准 / mould fill / video 提取 / inverse 跑通 / σ_y 数值 OK。
全部 OK 再跑正式 1-hour full session.

---

## Pilot 局限

| 局限 | 原因 | 缓解 |
|------|------|------|
| 只 2 timepoint → 无法独立拟合 σ_y_∞ + T_k 都给 CI | 2 pts ↔ 2 unknowns degenerate | 报 Δσ_y/Δt (freshness rate); 或假设 σ_y(0)=0 求 T_k |
| 1 replicate → 无 batch-to-batch CV | 单 batch | 后续加 2-3 replicates if pilot 通过 |
| Hold-out kinetic anchor 失效 | 2 pts 拟 2 参数 → 无 hold-out leverage | 用 hold-out setup (anchor 5) 替代 (本协议已含) |
| 复杂 kinetic 模型不能区分 | exp / sigmoid / 双相 同样 fit | 后续 6-timepoint 升级 |

→ Pilot **只回答**: "我们方法能否 capture σ_y(t) 单调变化 + cross-material
区分 + hold-out 验证 OK?"
→ Pilot **不回答**: "T_k 精确数值是多少?" — 那要升级到 6-timepoint mode A。

---

## 升级 path (Pilot 跑通后)

1. **Replicate**: 同手技 × 3 batch / 材料 → cross-batch CV / CI validation
2. **Tech sweep**: Mode B 3 手技参数 × 1 plateau → dose-response
3. **Full kinetic trace**: 升 6 timepoint → 完整 σ_y_∞ + T_k 拟合 + hold-out kinetic
4. **Hamamichi cross-check**: 1-2 个材料 plateau t_b 复制到 Hamamichi sibling repo
   → 跑 30-min/inverse 验证 absolute σ_y (anchor A in DESIGN.md)
