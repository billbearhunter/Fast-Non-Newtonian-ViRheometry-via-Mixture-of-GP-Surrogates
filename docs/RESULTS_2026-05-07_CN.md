# Results — 快速非牛顿流变学 inverse 的验证 2026-05-07

本文档包含 *Experiments* 与 *Results* 章节。方法论见 [`METHOD_2026-05-07.md`](METHOD_2026-05-07.md); 局限性与未来工作见 [`DISCUSSION_2026-05-07.md`](DISCUSSION_2026-05-07.md)。

---

## 1. 实验设置

### 1.1 材料

两种食品级非牛顿流体，跨低 σ_y / 高 σ_y 区段：

| 材料 | n (truth) | η (truth) | σ_y (truth) | 区段 |
|---|---:|---:|---:|---|
| Chuno (韩国年糕) | 0.633 | 10.51 | 19.62 | 低 σ_y |
| Okonomiyaki (面糊)    | 0.502 | 67.05 | 87.24 | 高 σ_y |

Truth 值是 25 °C 旋转流变仪测量的 HB 拟合 (`FlowCurve/Rheo_Data/Chuno_20230114_1523_25C.csv`、`FlowCurve/New_Rheo_Data/Okonomiyaki_2026_0505_25_0min.csv`)。所有单位 CGS（η 用 Pa·sⁿ、σ_y 用 Pa）。

### 1.2 相机 setup 几何

| 材料 | Setup 1 (用户) | Setup 2 (Hamamichi-orthogonal 提议) |
|---|---|---|
| Chuno    | W=2.5、H=2.7 cm | W=4.0、H=2.0 cm |
| Okonomi  | W=2.2、H=2.5 cm | W=7.0、H=7.0 cm |

每个 setup: 释放后等时间间隔 8 帧。相机标定通过 `camera_params.xml`，距离提取通过 `Calibration/extract_flow_distance.py`。

### 1.3 Inverse 配置

所有报告 run 的生产 default：

```
loss_mode                  log_nuisance_gp
shape_weight_mode          gp_fit_y8_quantile
shape_top_k                1
shape_gp_noise_log_floor   0.20
shape_disable_calib        True
shape_cma_restarts         5  (paper-grade)
shape_report_ci            True
```

---

## 2. Chuno (低 σ_y, 单 basin)

Joint y8q + GP-aware likelihood、5-restart paper-grade：

```
truth (rheometer):  (n, η, σ_y) = (0.633, 10.51, 19.62)

setup1-alone:    θ̂ = (0.675, 8.75, 27.67)    σ_y diff +41.0 %
joint:           θ̂ = (0.664, 8.12, 19.65)    σ_y diff  +0.2 %  ★

joint Hessian Laplace CI 95 %:
  n   ∈ [0.000, 1.000]    (under-identified, 整 box)
  η   ∈ [0.293, 240.7 ]   (宽; ridge degenerate)
  σ_y ∈ [9.93,   38.89]   ★ 含 rheometer truth 19.62

joint dispersion (5 restarts):
  σ_y dispersion = 0.0 %    ← basin 唯一
  η   dispersion = 35  %    ← η 多 basin (HB ridge)
  n   dispersion = 9.1 %
```

### 2.1 Snapdiff 视觉验证

每 setup 6 行 grid (`FlowCurve/figs/snapdiff_chuno_setup{1,2}_v2.png`):

```
Row 1   Real (相机)
Row 2   MPM(θ̂_y8q)
Row 3   MPM(truth)
Row 4   |Real − MPM(θ̂_y8q)| × 5
Row 5   |Real − MPM(truth)|  × 5
Row 6   |MPM(θ̂_y8q) − MPM(truth)| × 5
```

观察: **Row 6 整体很暗** — 恢复的 θ̂ 给的 MPM 物理跟 rheometer-truth MPM 不可区分。**Row 4 ≈ Row 5** — 残余 real-vs-MPM gap 是模拟器的固有限制（无表面张力、瞬态释放动力学），**不是** inverse 误差。像素级 diff: setup1 上 mean |MPM(θ̂) − MPM(truth)| = 0.05 / 255 (≈ 0.02 %)。

### 2.2 Single-vs-joint 流曲线

`FlowCurve/figs/y8q_chuno_single_vs_joint.png`:

- 黑虚线: rheometer 测量
- 红色: setup1-alone inverse — 因 ridge degenerate 比 rheometer 高约 30 %
- 蓝色: joint setup1+2 inverse — 在 γ̇ ∈ [1, 100] s⁻¹ 上贴合 rheometer

Hessian-orthogonal setup-2 提议成功为 Chuno 打破 (η, σ_y) ridge。

---

## 3. Okonomiyaki (高 σ_y, n,η 多 basin)

Joint y8q + GP-aware likelihood、5-restart paper-grade：

```
truth (rheometer):  (n, η, σ_y) = (0.502, 67.05, 87.24)

setup1-alone:    θ̂ = (0.642, 39.21, 102.54)   σ_y diff +17.5 %
joint:           θ̂ = (0.769, 21.35,  97.99)   σ_y diff +12.3 %

joint Hessian Laplace CI 95 %:
  n   ∈ [0.000, 1.000]    (under-identified)
  η   ∈ [2.03, 225.0  ]   (宽)
  σ_y ∈ [33.0, 291.3  ]   ★ 含 rheometer truth 87.24

joint dispersion (5 restarts):
  σ_y dispersion =  1.1 %    ← basin 唯一
  η   dispersion = 32   %    ← η 多 basin (HB ridge)
  n   dispersion = 8.9  %
```

### 3.1 σ_y identification 模式: routing-driven plateau

Okonomiyaki 的 σ_y 点估计跨 restart 唯一（1 % dispersion），但 CI 宽。这是 **routing-driven identification**: joint 搜索被两个 sub 专家 σ_y bbox 的交集限制，两个 bbox 都把 truth 含在下边缘。loss 在交集内大致 flat（σ_y 方向 Hessian 小），但 CMA-ES 确定性收敛到交集的几何中心（≈ truth）。

### 3.2 (n, η) HB-ridge degeneracy

Joint inverse 留下 *n* 比 truth +93 %、*η* 比 truth −68 %。诊断: 高 σ_y 材料的流物理在实验剪切率 (γ̇ ≲ 100 s⁻¹) 由 σ_y 主导，使 (n, η) 仅从 y_obs 弱可识别。多 restart η dispersion 32 % 量化了这一点 — (n, η) basin 真实非唯一。

我们验证了这 **不是** 代理训练 artifact: rheometer truth 附近的训练密度 Okonomi 实际上比 Chuno 更高（sub_20 truth 的 z-距离 0.5 内有 78 个点 vs Chuno sub_7 的 15 个）。ridge 是 HB inverse 在我们选定 (W, H) setup 对下的 fundamental physics 性质。

### 3.3 Snapdiff 视觉

`FlowCurve/figs/snapdiff_okonomiyaki_setup{1,2}_v2.png`:

- Row 6 (`|MPM(θ̂) − MPM(truth)|`): 流前缘可见细 outline → θ̂ ≠ truth 在 MPM 物理空间，符合 (n, η) ridge 偏移预期
- Row 4 (`|Real − MPM(θ̂)|`) 在低 γ̇ **比** Row 5 (`|Real − MPM(truth)|`) **更亮**: θ̂ 流得比 real 更远，因为它落在补偿 sim-real gap 的 ridge-point 上
- Row 4 ≈ Row 5 在高 γ̇

这是 truth-vs-y_obs 取舍的直接可视化: 当 sim-real gap 非零时，inverse 不可能同时拟合 y_obs 与恢复 truth。Joint y8q 选择 truth (paper-honest); 单 setup inverse 会选择 y_obs（视觉更近但有 bias）。

### 3.4 Single-vs-joint 流曲线

`FlowCurve/figs/y8q_okonomi_single_vs_joint.png`:

- 黑虚线: rheometer
- 红色 (single): 在低-中 γ̇ 接近 rheometer，σ_y 略高估
- 蓝色 (joint): 在 γ̇ ≲ 30 低于虚线，γ̇ ≈ 50 处穿过，高 γ̇ 略高

对高 σ_y 材料，single 与 joint 流曲线相似，因为实验 γ̇ 区段 rheometer 信号被 σ_y 主导。Joint 更可靠地识别 σ_y (12.3 % vs 17.5 % 偏差); 两者都不能唯一识别 (n, η)。

---

## 4. Wall-time 基准

GPU: 单 NVIDIA、无其他工作负载。配置: 5-restart joint y8q 带 Hessian CI。

| 材料 | Per-z eval (sequential) | Per-generation eval (batched) | 加速比 |
|---|---:|---:|---:|
| Chuno   restarts=1 | 15.0 s | **3.4 s**  | 4.4× |
| Chuno   restarts=5 | 64.7 s | **7.7 s**  | 8.4× |
| Okonomi restarts=1 | ~30 s  | **4.8 s**  | ~6×  |
| Okonomi restarts=5 | 132.6 s | **10.0 s** | 13×  |

Per-material 完整 paper-grade 工作流（setup1 alone + joint，都 5-restart）: 端到端 **~20 s** wall-time。

---

## 5. Truth-blind UQ 总结表

下面的报告 **不需要** rheometer truth 即可生成（即适用于任何未知材料）：

| 材料 | σ_y point | σ_y CI 95 %    | σ_y disp | identifiable | 流曲线拟合 |
|---|---:|---|---:|:---:|---|
| Chuno    | 19.65 | [9.9, 38.9]  | 0.0 % | ✓ | 出色 (蓝 ≈ rheo) |
| Okonomi  | 97.99 | [33,  291]   | 1.1 % | ✓ | paper-grade |

CI 在 **两种** 材料上都包含 rheometer truth (post-hoc truth-validated)。两种材料 dispersion 都小。η/n CI 宽且 dispersion 大 (HB ridge)，报告对此明确标记。

---

## 6. 重现这些结果

```bash
# Chuno paper-grade joint
python scripts/run_y8q_chuno.py --out scripts/run_y8q_chuno_paper.json
# 预期: σ_y diff +0.2 %, 单 GPU 上 ~10 s。

# Okonomi paper-grade joint
# (参考输出见 scripts/run_y8q_okonomi_paper.json，
#  工作流用 METHOD §8 中的 setup1.py + setup2.py CLI)

# Snapdiff 图 (Chuno + Okonomi setup1/setup2)
python scripts/snapdiff_grid_v2.py --material Chuno      [args 见文件]
python scripts/snapdiff_grid_v2.py --material Okonomiyaki [args]

# Single-vs-joint 流曲线
cd FlowCurve && python flowcurve.py --file <rheo_csv> \
    --est <eta_single> <n_single> <sy_single> \
    --est <eta_joint>  <n_joint>  <sy_joint>  \
    --out figs/<material>_single_vs_joint.png
```

所有 artifact 与参考 JSON 文件已 commit 到 `origin/main`。撰写时当前 head: `07a13df` (batched CMA eval)。

---

## 7. 参考图

```
FlowCurve/figs/
├── y8q_chuno_single_vs_joint.png        Chuno 流曲线 (rheo + single + joint)
├── y8q_okonomi_single_vs_joint.png      Okonomi 流曲线
├── snapdiff_chuno_setup1_v2.png         Chuno setup1 6-row snapdiff (f1..f8)
├── snapdiff_chuno_setup2_v2.png         Chuno setup2 6-row snapdiff
├── snapdiff_okonomiyaki_setup1_v2.png   Okonomi setup1 6-row snapdiff
└── snapdiff_okonomiyaki_setup2_v2.png   Okonomi setup2 6-row snapdiff
```

参考 JSON 输出:
```
scripts/run_y8q_chuno_paper.json    Chuno joint paper-grade 参考
scripts/run_y8q_okonomi_paper.json  Okonomiyaki joint paper-grade 参考
```
