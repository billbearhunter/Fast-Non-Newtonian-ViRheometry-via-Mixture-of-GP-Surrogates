# 新鲜度实验设计 — 2026-05-07

证明 camera-based inverse 相对 conventional rheometry 时间分辨优势的新鲜度
/ 动力学实验设计规范。本文档替换早期 meeting-notes 草稿。

伴随 [`METHOD_2026-05-07_CN.md`](METHOD_2026-05-07_CN.md)、
[`RESULTS_2026-05-07_CN.md`](RESULTS_2026-05-07_CN.md)、
[`DISCUSSION_2026-05-07_CN.md`](DISCUSSION_2026-05-07_CN.md)。

---

## 1. 假设与主要 outcome

**H1 (primary)**: 一种经历时变微观结构演化的非牛顿流体表现出单调 σ_y
动力学，camera inverse 能在 ≤ 5-min cadence 解析。

**主要 outcome**: σ_y(t) 曲线 t ∈ [1, 60] min 在 7-9 个时间点采样;
跟稀疏 rheometer reference t ∈ {5, 30, 60} min 比较。

**成功标准** (pre-registered):
1. ≥ 80 % 的 camera 时间点报告 `identifiable[σ_y] = True` (truth-blind
   identifiability flag from Hessian Laplace CI)。
2. Camera σ_y(t) 在 3 个 rheometer 验证时间点位于 ±25 % rheometer
   HB-fit σ_y 内。
3. σ_y(t) 单调 (或匹配已知非单调模式，见 §3.4) 在 camera dispersion
   bands 内。

任何标准未通过触发根因分析 (§9)，**不是** silent retry。

---

## 2. 材料选择 — 决策矩阵

三个候选材料 (Option A in earlier notes; B 与 C 已废弃，见 §10):

| 候选 | Cold-soluble | 动力学 timescale | σ_y range (Pa) at typical conc. | Reproducibility | 推荐 |
|---|:---:|---|---|:---:|:---:|
| **黄原胶 0.5 % w/v** | ✓ | 5–30 min | 5–60 | high | **★ primary** |
| 瓜尔胶 0.7 % w/v | ✓ | 10–60 min | 10–80 | high | secondary |
| ι-卡拉胶 0.5 % w/v + KCl 50 mM | ✓ | 30–60 min | 20–150 | medium | tertiary |
| 琼脂 0.5 % w/v (热溶解 + 冷却) | thermal | 5–20 min cooling-set | 50–300 | medium | excluded — 协议需要加热 |
| 明胶 2 % (热溶解 + 冷却) | thermal | 30–120 min | 20–400 | medium | excluded — 慢 + 温度敏感 |

**主选 rationale (黄原胶)**:
- **冷溶**: 不需要热协议 → 不引入 temperature-as-confound。
- **σ_y range 5–60 Pa**: 舒适地落在校准的 GP coverage 内 (gid 0
  σ_y_floor = 5 Pa; gid 0 覆盖 σ_y 到 ~100 Pa)。
- **动力学 5–30 min**: rheometer 周期 (~30 min) 与 *整个* 实验时长匹配，
  所以单次 rheometer run 只能 capture 一个 "static" 点 — 时间分辨 gap
  的干净 demonstration。
- **食品级、便宜、易得**: 标准家庭原料。单一厂商批次消除 vendor-variability
  confound。

---

## 3. 协议 — 样品制备

### 3.1 试剂
- 黄原胶，食品级，单一批次 (记录 lot 号)。
- 蒸馏水或滤过的自来水，预先 equilibrate 到实验室温度
  (24 ± 1 °C，每次试验开始与结束记录)。

### 3.2 配方 (单时间点)
- 黄原胶质量: **0.500 g** (± 0.005 g, 天平精度)。
- 水质量: **99.500 g** (± 0.05 g) → 0.5 % w/w。
- 容器: 250 mL 烧杯，磁力搅拌器 200 rpm，搅拌子 25 mm。

### 3.3 混合协议
- t = 0: 同时启动搅拌 + 启动计时器 + 在 5 s 内将黄原胶慢慢倒入
  vortex (避免结块)。
- 继续搅拌 60 s。停止搅拌。
- 样品准备好用于 camera release。

### 3.4 时间点
- 线性后转 log spacing: **t ∈ {1, 3, 5, 10, 20, 30, 45, 60} min** (8 点)。
- 每个时间点需要 **新鲜批次** (黄原胶在重新搅拌时会重新分散 → 不能
  reuse)。每次 replicate trial 8 批次。
- 重复: **n = 3 个独立 trials per time point** 总计。
  - 每个 replicate 内，随机化时间点顺序避免 correlated batch effects
    (例如 trial-1 上午 vs trial-3 下午)。

### 3.5 期待动力学形态
黄原胶水化: 单调 σ_y 上升，~30 min 达到 plateau。函数形式:

```
σ_y(t) ≈ σ_y_∞ · (1 − exp(−t/τ))
```

τ ∈ [3, 15] min 取决于温度/剪切历史。从 camera 动力学曲线拟合
`(σ_y_∞, τ)` 作为 secondary outcome。

---

## 4. Camera setup 几何 (W, H)

### 4.1 路由要求
实验前 GP coverage 检查 (§5) 决定几何。Default plan:

| Setup | (W, H) cm | 路由到 gid | 校准? |
|---|---|---|:---:|
| **Setup 1** | (2.5, 2.7) | gid 0  | ✓ (post-hoc c_k) |
| **Setup 2** | (4.5, 2.5) | gid 10 | ✓ (post-hoc c_k) |

两个 gid 都是 post-hoc 校准 (见 DISCUSSION §4)，所以置信区间可信。
Reuse `data/new_real_world_experiments_freshness/
ref_Chuno_10min_2.7_2.5_1/settings.xml` 作为 setup-1 模板; 创建对应的
`ref_Xanthan_<t>min_2.5_4.5_2/settings.xml` 给 setup 2。

### 4.2 单时间点 capture
- 释放底材准备 (干净的 acrylic 板带校准 grid)。
- Setup 1 (W=2.5, H=2.7) 长方体模具 → 释放 → 8 帧 fixed Δt。
- Setup 2 (W=4.5, H=2.5) 在同一 batch 的 *单独 aliquot* 上重复，setup 1
  释放后 ≤ 60 s。
- Camera calibration (`camera_params.xml`) **整天 fixed** — 时间点之间
  不要重新校准。

### 4.3 Joint vs single-setup 选择
- **Joint (推荐)**: paper-grade σ_y 识别 (DISCUSSION §1)。
  代价: 每时间点 2× release，~20 s GPU inverse。
- **Single setup-1 only (fast mode)**: ~10 s inverse，但高 σ_y 晚期
  时间点可能有 (n, η) ridge degeneracy (DISCUSSION §1)。
- **决策规则**: paper-grade 动力学曲线用 joint (主输出)。可选地也报告
  setup-1-alone 曲线作直接 side-by-side cost/quality 比较。

---

## 5. 实验前路由 / coverage 检查

提交 8 时间点 × 3 重复 × 2 setup 协议 (48 camera runs) 之前，用单个
中间时间点试验验证 GP coverage:

```bash
# Pilot trial: t = 15 min (mid-kinetics), n = 1
# 1. 按 §3.3 准备 batch，等 15 min
# 2. 在 setup 1 + setup 2 capture 8 帧
# 3. 跑 paper-grade joint inverse:
python scripts/run_y8q_prior.py \
    --data-root data/new_real_world_experiments_freshness \
    --material Xanthan15min \
    --restarts 5 \
    --joint-only \
    --out scripts/run_y8q_xanthan_pilot.json
```

**Pilot 通过标准**:
1. 两个 setup 的 routed sub bbox 包含 pilot σ_y 估计。
2. y8 z-score < 1.5 给两个 setup (y_obs 在训练 y8 分布内)。
3. Joint σ_y identifiable，dispersion < 10 %。
4. Joint σ_y CI half-width < 50 % point estimate。

如果 pilot 失败，迭代: 试更低/更高浓度让 σ_y range 进入 coverage; 或
改 setup 2 几何到不同 gid。

Pilot 也是验证 rheometer-side 协议 (§6) 的第一次机会; 在 pilot 时间点
同步 rheometer 测量给一个 anchor 点 sanity-check sub bbox。

---

## 6. Rheometer 验证协议

### 6.1 稀疏验证时间点
- t = 5 min, 30 min, 60 min — 每个 replicate trial 三个 rheometer runs。
- 一个 trial × 3 时间点 = 每个 rheometer 日 3 个分开的黄原胶 batches。
- Rheometer 周期 (loading + measurement + cleaning) ≈ 30 min →
  一台 rheometer 专用就能跟上。

### 6.2 Rheometer 设置
- 几何: cone-plate 40 mm Ø, 1° angle, gap 28 µm。
- 温度: 24 °C (匹配 lab/camera 温度)。
- Shear-rate sweep: γ̇ ∈ [0.1, 1000] s⁻¹, log-spaced 21 points,
  pre-shear 30 s at 100 s⁻¹ then 10 s rest。
- 在 γ̇ ∈ [1, 100] s⁻¹ 拟合 HB 模型 (inverse-domain 匹配)。

### 6.3 Camera-rheometer 比较
在 t = 5, 30, 60 min 每个，绘制:
- Camera-joint θ̂ point + 95 % CI。
- Rheometer HB fit point。
- 流曲线 overlay: rheometer 曲线 + 三个验证时间点的 camera-implied HB 曲线。

接受标准: rheometer truth 落在 camera 95 % CI 内 ≥ 2/3 个点。

---

## 7. 统计分析计划

### 7.1 Per-time-point summary
对每个 (t, replicate)，报告:
- θ̂ = (n̂, η̂, σ̂_y) joint inverse point estimate。
- 每参数 95 % Hessian Laplace CI。
- 5-restart z-dispersion 每参数。
- 每参数 identifiability flag。
- 路由 sub_id (setup 1 + setup 2)，bbox，y8 z-score (per setup)。

### 7.2 Across-replicates 聚合
- σ_y(t) primary estimate: 3 个 replicate point estimates 在每 t 的中位数。
- σ_y(t) error bar: max(per-replicate CI half-width, replicate-spread)。
  这保守地报告 inverse uncertainty 与制备变异性中较大者。

### 7.3 动力学参数拟合
拟合 `σ_y(t) = σ_y_∞ · (1 − exp(−t/τ)) + σ_y_0` 用 non-linear least
squares (用 §7.2 误差棒 weighted)。报告 `(σ_y_∞, τ, σ_y_0)` ± 1-σ。

### 7.4 时间分辨 headline statistic
- "Camera 解析 σ_y 动力学在 Δt = 1 min cadence (60 min 内 8 pts) 中位数
  CI half-width X Pa per point"。
- "Rheometer 在同 60 min 产生 1–2 reliable points"。
- → 直接 figure-of-merit: **points-per-hour 比** ≈ 8/2 = 4× 到 8/1 = 8×
  (取决于什么算 "reliable" rheometer point at 高 cadence)。

---

## 8. Pre-registration checklist (pilot 前 commit)

在跑 pilot **前** 把这些决定锁定到文档:

- [ ] 材料: 黄原胶 (或 pilot 失败时备选)。
- [ ] 浓度: 0.5 % w/w。
- [ ] 时间点: {1, 3, 5, 10, 20, 30, 45, 60} min。
- [ ] 重复: n = 3 trials, 随机化时间点顺序。
- [ ] Setups: (W=2.5, H=2.7) + (W=4.5, H=2.5)。
- [ ] Inverse: joint paper-grade (5-restart, floor=0.20, disable_calib)。
- [ ] Rheometer 验证点: t ∈ {5, 30, 60} min。
- [ ] 成功标准 (§1)。

Pre-registration 后，偏离需要书面修正与理由。

---

## 9. 失败模式与 mitigation

| 失败模式 | 检测 | Mitigation |
|---|---|---|
| GP coverage gap (pilot 失败) | y8 z-score > 1.5 或 sub bbox 排除 σ_y | 调整浓度/setup 几何重新 pilot; 考虑第 4 候选材料。 |
| 高 σ_y plateau 的 HB ridge degeneracy | dispersion[η] > 25 % 在晚期时间点 | 已记录 (DISCUSSION §1)。σ_y 仍识别; 报告 (n, η) 为 ridge-degenerate，**不是** bias。 |
| Real-MPM gap dominate inverse residual | snapdiff Row 4 ≈ Row 5 但 Row 6 小 | 已理解 (DISCUSSION §3)。Truth recovery 保留; caption 标注。 |
| Replicate-to-replicate σ_y > 30 % spread | Cross-replicate variance 高 | 检查 batch preparation (水温、搅拌时间、黄原胶粉年龄)。加 4th replicate。 |
| Identifiability 从 t=5 True 翻到 t=30 False | Identifiability 列在 t 上 heterogeneous | 可能 sub-routing 在 σ_y 跨 bin 边界时变化。报告 sub_id per t; 必要时给受影响点跑 K=2 (top-2 sub union) joint inverse。 |
| Rheometer-camera gap > 25 % 在所有 3 验证点 | 验证接受标准失败 | 检查 sim-real gap 校准。可能指示黄原胶专属物理 (charge-screening, 自来水盐含量) 未由 Taichi MPM 建模。 |

---

## 10. Option B 与 C 为什么废弃

**Option B (commercial + intervention)** 废弃，因为 intervention 协议
是被研究的变量本身 — 但 inverse 本身不会从 intervention vs simple
kinetics 得益，rheometer 一侧变难 (intervention 计时要在两边都控制)。

**Option C (Chuno aging)** 废弃为 *主* 设计但 **保留作 secondary
demonstration** 如果 pilot 成功且时间允许。Chuno_10min sample dir 已存在
(`data/new_real_world_experiments_freshness/ref_Chuno_10min_2.7_2.5_1/`,
empty pending camera capture)。两者都跑给 "试剂" + "真实食物" 一对，
强化新鲜度 narrative。

---

## 11. 交付物

实验结束后产出:

```
data/new_real_world_experiments_freshness/
  ref_Xanthan_<t>min_2.7_2.5_1/    × 8 t-values × 3 replicates  ⇒ 24 dirs
  ref_Xanthan_<t>min_2.5_4.5_2/    × 8 t-values × 3 replicates  ⇒ 24 dirs
  ref_Xanthan_<t>min_rheo/...      × 3 t-values × 3 replicates  ⇒ 9 csvs

scripts/
  run_y8q_xanthan_kinetics.py        # batch inverse driver, exports kinetics JSON
  plot_xanthan_kinetics.py           # σ_y(t) curve + rheometer overlay

docs/figs/
  xanthan_kinetics_sigma_y.png       # 主 headline 图
  xanthan_kinetics_flowcurves.png    # rheometer + camera 流曲线在验证点
  xanthan_temporal_resolution.png    # bar chart: camera 8 pts/h vs rheo 1-2 pts/h
```

---

## 12. 状态

- 2026-05-07: 设计 pre-registered (本文档)。
- 待: pilot trial (黄原胶, t = 15 min) 验证 GP coverage 与 rheometer 协议。
- 待: pilot 通过后整个动力学 campaign。

现有 artefacts:
- `data/new_real_world_experiments_freshness/ref_Chuno_10min_2.7_2.5_1/
  settings.xml` (作为 setup-1 模板可重用; ready for camera capture
  on Chuno-aging secondary 实验)。
- gid 0 + gid 10 post-hoc 校准 (适合 paper-grade output)。
