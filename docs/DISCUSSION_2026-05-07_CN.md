# Discussion — 局限性与未来工作 2026-05-07

[`METHOD_2026-05-07.md`](METHOD_2026-05-07.md) 与 [`RESULTS_2026-05-07.md`](RESULTS_2026-05-07.md) 的伴随文档。

---

## 1. 高 σ_y 材料的 HB ridge

对实验剪切率下 (γ̇ ≲ 100 s⁻¹、σ_y ≳ 50 Pa) 由屈服应力主导的材料，joint 双 setup inverse 唯一识别 σ_y 但留下 (n, η) 在 HB ridge 上。实证：

```
Chuno    (σ_y =  19.62 Pa) :   joint 把 σ_y、η 识别到 25 % 内
Okonomi  (σ_y =  87.24 Pa) :   joint 识别 σ_y; (n, η) ridge degenerate
                                (MAP 处 n diff +93 %、η diff −68 %;
                                 5-restart η dispersion 32 %)
```

这是我们 (W, H) setup 选择下 HB inverse 的 **fundamental 性质**，不是代理训练限制。我们通过直接密度分析确认: rheometer truth 附近的 GP 训练对 Okonomi 实际上比 Chuno 更**密**。

**缓解方案**（尚未实施）:
- 第三个实验 setup 用很不同的 aspect ratio (e.g. H/W ≫ 2 给剪切主导 profile，或极端 W/H 给快速展布) 直接针对 (n, η) ridge。
- 替代: 把独立 rheometer prior on n 加进来 (e.g. 高剪切率测量) 作为 Bayesian 正则。

---

## 2. Poiseuille proposal 不匹配

`propose_next_setup` (Hamamichi 2023，在 `Optimization/libs/mechanism.py` 忠实 port) 在 **Poiseuille 通道流** 下计算 analytical HB Hessian。我们的实验是带自由面的重力驱动释放-展布。两种物理不同:

```
Poiseuille:    固定通道中压力驱动稳态流
               无自由面、无瞬态、无展布
               (η, n, σ_y) 上闭式 Hessian

释放-展布:     重力驱动、自由面演化
               瞬态 + 展布到有限 Y_8 停止
               后期 yield-stress 主导 quasi-static
```

低 σ_y 材料 (Chuno) 的 Poiseuille analytical proposal 偶然接近最优: 推荐的 (W, H) = (4, 2) 在真实释放-展布 MPM 物理下成功打破 (η, σ_y) ridge。高 σ_y 材料 (Okonomi)，推荐的 (W, H) = (7, 7) 是跟 setup 1 同 aspect ratio 的方块几何 — Poiseuille 线性化模型下 orthogonal 但释放-展布 MPM 物理下不 orthogonal。

**未来工作**: 用基于 MPM 或代理的 Hessian-based proposer 替代 analytical Poiseuille。在候选 (W, H) 处对 GP 计算 ∇²ℓ 是可行的（每候选一次 batched GP forward）; 在多个材料上摊销，固定成本可接受。

---

## 3. Sim-real gap

我们的 forward 模拟器（Taichi MPM）不建模:
- 表面张力（在小 (W, H) 与流前缘相关）
- 瞬态释放-闸门动力学（frame 1 系统偏移）
- 非等温效应（展布期间冷却）

这表现为流距 per frame 上残余 ~5–15 % MPM-vs-real 差异，**与 inverse 精度无关**。我们 likelihood 中的 `σ_obs_log = 0.20` floor 把一部分作为名义观测噪声吸收，但系统偏差未建模。

**我们报告的 θ̂ 是恢复 rheology truth 的，*不是* 最小化 MPM-real 残差的** — 给定 sim-real gap，两者互斥。Snapdiff Row 4 (Real vs MPM(θ̂)) 是 gap 的可视化; 当伴随小 Row 6 (MPM(θ̂) vs MPM(truth))，**它不指示 inverse 误差**。

**缓解方向**:
- 加表面张力到 Taichi MPM（工程量，非概念问题）。
- 训练 sim-to-real 校正层: 小 NN 把 *MPM-y → real-y* 从校准材料集的 (rheometer-truth-θ, real-y_obs) 配对学到。

---

## 4. 生产模型 bank 的校准覆盖

截至 2026-05-07，25 个生产 gid (`Models/v10_yshape_planB/state_gid_*`) 中 20 个 v2-partition 且训练完成:

```
完全训练 + post-hoc c_k 校准:  gid 0、10、24
完全训练 + 默认 c_k = 1.0:     gid 1-9、11-19、23
部分训练 (中断):              gid 20  (23/61 experts)
未触及 (仍 v1 partition):     gid 21、22
```

(W, H) 路由到 gid 20-22 的材料 inverse 时会回退到 v1 partition 或部分 v2; 这些几何区域的 inverse 可能不可靠。继续训练直接：

```
python scripts/retrain_unused_gids.py --gids 20,21,22 \
    --bank Models/v10_yshape_planB
```

---

## 5. 优化 default 取舍

生产 default `σ_obs_log = 0.20` + `disable_calib` 是经验选定，给 Chuno 与 Okonomiyaki 两者一致 paper-grade σ_y 识别。替代 default (`σ_obs_log = 0.05` + `calib_scale enabled`) 给：

- Chuno  : σ_y diff +6.4 % (vs 新 default 的 +0.2 %)
            但 MPM(θ̂) ≈ MPM(truth) 在 sub-pixel 级别
- Okonomi: σ_y diff −4.2 % (vs 新 default 的 +12.1 %)
            但 n = +93 %、η = −63 % → γ̇ = 100 s⁻¹ 处流曲线偏离 3×

新 default 牺牲 Chuno 上 ~5 % σ_y 精度避免高 σ_y 材料上的灾难性流曲线不匹配。向后兼容通过 CLI flag 保留。

---

## 6. Single-restart vs 5-restart

对单 basin loss landscape 的材料，single-restart CMA (default) 已足够。当 (n, η) ridge degenerate 时需要 5-restart: 实证 Okonomi 跨 restart 显示 32 % η dispersion，意味着 global minimum 位置随 seed 改变。

**实用建议**: paper-grade 输出始终用 5-restart。dispersion 度量事后告诉你 single-restart 是否本来就够。批量 CMA eval 下 5-restart 每次 inverse 加 < 10 s → 没理由跳过。

---

## 7. Frame weighting 经验

Default `frame_w = [0.2, 0.2, 0.35, 0.5, 0.8, 1.0, 1.4, 2.0]` 下调早期释放瞬态帧 (1, 2) 的权重（物理偏离 MPM 最严重处），强调晚期"quasi-static"帧（inverse 信号最干净处）。这些权重从前期工作继承，未在我们 pipeline 中重新推导。数据驱动的重新加权（如 inverse-Fisher-information per frame）可适度提升 inverse 精度。

---

## 8. 未来工作总结

1. **极端 aspect ratio 的第三个 setup** 打破高 σ_y 材料的 (n, η) ridge。
2. **基于代理 Hessian 的 setup-2 proposer** 替代 analytical Poiseuille 模型。
3. **Sim-real gap 校正层** 在 rheometer-truth + real-y_obs 配对校准集上训练。
4. **n,η degenerate 区域的 active-learning infill** 给高 σ_y sub 专家。
5. **恢复 gid 20-22 的训练** 给完整几何覆盖。
6. **新鲜度实验** (见 [`FRESHNESS_EXPERIMENT_DESIGN_2026-05-07.md`](FRESHNESS_EXPERIMENT_DESIGN_2026-05-07.md)) 利用 ~20 s per-material inverse 做高时间分辨率流变学动力学，展示 conventional rheometry 不可行的测量类别。
