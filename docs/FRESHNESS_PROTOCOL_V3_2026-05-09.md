# 新鲜度实验完整操作手册 V3.1

> **更新日期**:2026-05-09
> **状态**:锁定协议,可直接进 lab 执行
> **替代关系**:作废 V1/V2 设计 + V3.0(8 mold 池版本)
>
> **V3.1 vs V3.0 主要变化**:
> 1. Setup-1 改为 [2, 7]² cm **连续随机**(对齐 `mechanism.py` 的 1 mm grid)
> 2. Setup-2 由 proposer 直接给连续输出(不再 snap 到离散池)
> 3. Batch 大小 300 g → **400 g**(覆盖 (7, 7) 角的 worst case)
> 4. 新增 **§4 Synthetic time-resolved validation**(sim 差分对照,纯算力)
> 5. 新增 §11 sim diff-ref 后处理脚本说明

---

## 0. 实验目的(写进论文要证明的四件事)

1. **Sim 差分对照**:在 prescribed time-varying $\boldsymbol{\theta}(t)$ 上(ground truth 已知),pipeline 能在 (T_k, σ_Y∞) 误差 < ~20% 内恢复 trajectory ─ 证明反演本身正确。
2. **Within-batch trajectory recovery**:对真实 hand-prepared 淀粉糊,我们的代理反演能在 60 min 内用 3 个 joint 2-setup 测量恢复完整 σ_Y(t) 动力学曲线 ─ Hamamichi 的 8h-per-setup 反演结构上不可能做到。
3. **Cross-batch (T_k, σ_Y∞) 漂移**:不同批次的同配方有显著的 batch-to-batch kinetic 常数差异。这论证 Hamamichi 的 hypothetical "cross-batch staggered" 协议在该类材料上 ill-defined ─ 因为各批次没有共同的 (T_k, σ_Y∞)。
4. **Hoover 浓度缩放**:同 source 不同浓度的 σ_Y∞ 满足 σ_Y∞ ∝ c^α(α ∈ [2, 5],Hoover 2001)的文献缩放。

---

## 1. 关键参数(已锁定)

| 参数 | 值 | 备注 |
|------|-----|------|
| 时间点 | t ∈ {0, 20, 40} min | 连续紧排,session = 60 min calendar |
| **Setup-1 几何** | **每 session 从 [2, 7] × [2, 7] cm 连续 box 均匀随机抽** | 与 `mechanism.py` 的搜索 grid 对齐 |
| **Setup-2 几何** | t≈0 snapshot 反演 → `propose_next_setup` 输出连续 (W_2*, H_2*) | 1 mm grid 步长,无须 snap |
| Pair 协议 | Joint 2-setup (setup-1 + setup-2 各拍一次) | 全条件统一 |
| **Batch 大小** | **400 g/session**(worst case ~392 g) | 覆盖 (7,7)+(7,7) 极端配置 |
| 室温 | 25°C ± 0.5°C | 全程 Peltier 或恒温浴 |
| 相机标定 | ChArUco 一次性标定,所有 session 共用 | `camera_params.xml` |
| 总 sessions | 4 (L1) + 6 (L3 主) + 1 (L3 重复) = 11 lab sessions | 加 1 个 sim panel(无 lab) |
| 总 dambreak(lab 实拍) | 8 (L1) + 36 (L3 主) + 6 (L3 重复) + 7 (snapshot) = **57 lab videos** | |
| 总 dambreak(sim) | 8-12 prescribed × 3 time × 2 setup ≈ **48-72 sim videos** | 算力跑,不占 lab |
| 总 lab 时间 | ~3 lab days | 含 24h rheo follow-up |
| 总算力 | ~3-5 GPU-h | sim panel + 全部反演 |

---

## 2. Day 0:实验前一天准备

### 2.1 Mold 准备

由于 setup-1 是**连续随机**,setup-2 也是 proposer 给的连续输出,你需要:

**选项 A**(推荐):**可调节 mold 装置**
─ 一个有滑动壁/可调侧板的 dam-break 容器,允许 (W, H) 在 [2, 7] cm 任意值实时调到 1 mm 精度。Session 开始前根据 random 抽样调好 setup-1,t≈0 snapshot 后再调到 setup-2。

**选项 B**(回退):**离散 mold 池 + 现场最近 snap**
─ 如果你只有几个固定大小的 mold(比如 (3,3)、(4,4)、(5,5)、(6,6) 之类),把 random 抽样**snap 到最近的 mold**,并在 lab journal 里记录下"random ideal"vs"actual"两个值,后处理时用 actual 跑反演。

**选项 C**(临时):**3D 打印 / 现做 mold per session**
─ 如果你用激光切丙烯酸或 3D 打印,可以 session 前一晚根据 random 抽样现做。约 10 min 制作 + 10 min 晾干 + cleaning。

**记录在 lab journal 的 mold info**:
```
Session # ___:
  Random sampled (W_1, H_1) = (___, ___) cm  ← 抽签结果
  Actual setup-1 mold (W, H) = (___, ___) cm  ← 实际用的 mold,可能是 snap 后的近似
  Snap deviation Δ = sqrt((W_actual - W_random)² + ...) = ___ cm
  
  Proposer output (W_2*, H_2*) = (___, ___) cm
  Actual setup-2 mold (W, H) = (___, ___) cm
  Snap deviation Δ = ___ cm
```

如果 Δ > 0.5 cm,需要在论文里 disclose;如果 Δ < 0.3 cm 可以认为忽略不计。

### 2.2 相机 + 标定

```bash
# 1. 设备:相机 + 三脚架(固定)
# 2. ChArUco 板放镜头前
# 3. 跑 standard calibration:
python -m Calibration.recalibrate_only --board-photo charuco.jpg
# → 生成 camera_params.xml
# 4. 验证:任意 mold 放在工作台上,拍参考帧 I_0
python -m Calibration.test_pipeline --frame I_0.jpg
# 应输出 IoU > 0.99
```

所有 session **共用同一个 `camera_params.xml`**,session 间不重新标定。

### 2.3 Rheometer 预约

| 用途 | 何时 | 多长 |
|------|------|------|
| L1 静态对照 4 mat 流曲线扫描 | Day 1 上午 | ~30 min |
| L3 smooth 子集 24h follow-up(片栗粉×2 + 白玉×1 + repeat 1)= 4 次 | Day 4 上午 | ~25 min |

### 2.4 材料采购清单

| 材料 | 总用量(干粉/原物)| 用途 |
|------|-----------|------|
| 即时燕麦片 | 60 g 干粉 | oats 5% × 1(20 g)+ 9% × 1(36 g),共 2 sessions |
| 片栗粉 | 80 g 干粉 | 3% × 1(12 g)+ 5% × 1(20 g)+ 7% × 1(28 g),共 3 sessions |
| 白玉粉 | 30 g 干粉 | 7% × 1(28 g) |
| パンケーキ粉 | 320 g(混合粉)| 70% × 1(280 g) |
| 矿泉水 | 3 L | 制备用,400g batch 多份 |
| 牛乳(冷藏)| 150 mL | パンケーキ粉用 |
| 甜面酱 | 1 个新包装 | L1 |
| Lotion | 1 个新包装 | L1 |
| Moisturising milk | 1 个新包装 | L1 |
| Tonkatsu | 1 个新包装 | L1 |

(Chuno、Okonomiyaki **不重测**,用 repo 已有数据)

### 2.5 数据目录创建

```bash
cd Fast-Non-Newtonian-ViRheometry-via-Mixture-of-GP-Surrogates
mkdir -p data/freshness_2026-05-09/{L1_static,L3_panel,L3_repeat,L0_sim,rheometer,lab_journals}
```

---

## 3. L1:静态对照 panel(Day 1 上午,~3 h)

### 3.1 协议(每材料 ~30 min)

```
1. 开新包装,搅拌均匀
2. (W_1, H_1) 从 [2, 7]² 连续 random 抽 → 调节 mold 至该尺寸(选项 A)
   或 snap 到最近现成 mold(选项 B)
3. 取 ~50 g 进 setup-1 mold → 释放闸门 + 拍 8 帧 video → 保存 setup-1 video
4. ~30 sec offline single-setup inverse:
   python -m Optimization.estimate_first_setup \
       --state-root Models/yshape_mogp_production \
       -f data/freshness_2026-05-09/L1_static/ref_<mat>_<W1>_<H1>_1 \
       --density <ρ> --shape-cma-restarts 5 --shape-report-ci
   → 得到 θ̂_1
5. 跑 Hessian-orthogonal proposer → 推荐 (W_2*, H_2*),连续值
6. 调节 / 制作 setup-2 mold 至 (W_2*, H_2*)(或 snap)
7. 取另 ~50 g 进 setup-2 mold → 释放 + 拍 → 保存 setup-2 video
8. (Day 4 上午)24h 后,再扫一次 rheometer
```

### 3.2 4 个材料(顺序无所谓,但先做容易溢出的)

| 顺序 | 材料 | (W_1, H_1) random | (W_2*, H_2*) |
|:---:|------|---|---|
| 1 | 甜面酱 | _记录:_ | _记录:_ |
| 2 | Tonkatsu | _记录:_ | _记录:_ |
| 3 | Lotion | _记录:_ | _记录:_ |
| 4 | Moisturising milk | _记录:_ | _记录:_ |

### 3.3 文件命名

```
data/freshness_2026-05-09/L1_static/
├── ref_SweetBeanPaste_<W1>_<H1>_1/       (setup-1)
│   ├── camera_params.xml
│   ├── settings_setup1.xml
│   ├── frame_001.png ... frame_008.png
│   ├── y_obs.npy
│   └── theta_hat.json (后处理生成)
├── ref_SweetBeanPaste_<W2>_<H2>_2/       (setup-2)
└── ... 等等
```

`<W1>_<H1>` 用实际 mold 的 (W, H) 填,精度到 0.1 cm。

### 3.4 验收

每材料完成后即刻 sanity check:
```bash
python -m Optimization.estimate_joint_setup \
    -f data/.../ref_<mat>_<W1>_<H1>_1 \
    -s data/.../ref_<mat>_<W2>_<H2>_2 \
    --shape-warm-start-from-first --shape-cma-restarts 5 --shape-report-ci
```

检查输出 `theta_hat.json` 中 `identifiable` 三个 flag 都为 true。否则需要补一组数据。

---

## 4. L0:Synthetic time-resolved validation(sim 差分对照,Day 0 / 任意算力空闲时跑)

### 4.1 目的

在已知 ground-truth $\boldsymbol{\theta}(t)$ 的 synthetic 视频上验证 pipeline 的 trajectory recovery,作为 §7.6 实验的 ground-truth control。

### 4.2 panel 设计:8-12 个 prescribed kinetic trajectories

每个 trajectory 由 $(T_k^*, \sigma_{Y,\infty}^*, \sigma_{Y,0}^*, n^*, \eta_\infty^*)$ 给定。覆盖范围:

| Trajectory ID | $T_k^*$ (min) | $\sigma_{Y,\infty}^*$ (Pa) | $\sigma_{Y,0}^*$ (Pa) | $n^*$ | $\eta_\infty^*$ (Pa·s^n) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 15 | 20 | 5 | 0.6 | 5 |
| 2 | 15 | 100 | 25 | 0.6 | 5 |
| 3 | 30 | 20 | 5 | 0.7 | 8 |
| 4 | 30 | 100 | 25 | 0.7 | 8 |
| 5 | 30 | 50 | 12 | 0.8 | 12 |
| 6 | 60 | 20 | 5 | 0.6 | 8 |
| 7 | 60 | 100 | 25 | 0.7 | 12 |
| 8 | 60 | 50 | 12 | 0.8 | 5 |
| 9 | 120 | 50 | 12 | 0.7 | 12 |
| 10 | 120 | 200 | 50 | 0.6 | 8 |
| 11 | 240 | 100 | 25 | 0.5 | 15 |
| 12 | 240 | 50 | 12 | 0.7 | 8 |

(覆盖 short / mid / long $T_k$ × low / mid / high $\sigma_{Y,\infty}$ × 不同 n、η_∞ 区域)

### 4.3 协议(全自动)

```
对每个 trajectory ID ∈ {1, ..., 12}:
   对每个 t_i ∈ {0, 20, 40} min:
       computed σ_Y(t_i) = σ_Y,0* + (σ_Y,∞* - σ_Y,0*) × (1 - exp(-t_i / T_k*))
       θ(t_i) = (n*, η_∞*, σ_Y(t_i))   # n 与 η 假设不随时间变,σ_Y 按 Avrami 演化
       
       从 [2, 7]² random 抽 (W_1, H_1)  # 可与 real 实验同一 random seed
       MPM forward(θ(t_i), W_1, H_1) → setup-1 synthetic video
       Single-setup inverse → θ̂_1(t_i, sim)
       Hessian-orthogonal proposer → (W_2*, H_2*)
       MPM forward(θ(t_i), W_2*, H_2*) → setup-2 synthetic video
       Joint inverse → θ̂_joint(t_i, sim)
   
   把 3 个 σ̂_Y_joint(t_i) 拟合 Avrami → 得 (T̂_k, σ̂_Y∞)
   对比:
     |T̂_k - T_k*| / T_k*  ← 期望 < 20%
     |σ̂_Y∞ - σ_Y,∞*| / σ_Y,∞*  ← 期望 < 20%
```

### 4.4 总开销

```
12 trajectory × 3 time × (1 MPM forward × 2 setup) = 72 MPM forward
72 × ~5 min/forward = ~6 GPU-h
+ 36 single-setup inverse × 12s = 7 min
+ 36 joint pair inverse × 24s = 14 min
+ 12 Avrami fit × <1 sec
+ Δ_pair check + 误差报表

总 ~6.5 GPU-h,夜间 batch 跑完,无 lab 占用。
```

### 4.5 输出 deliverables

```
data/freshness_2026-05-09/L0_sim/
├── prescribed_panel.csv    # 12 trajectory 的真值参数
├── trajectory_ID_01/
│   ├── t000_setup1/         # θ(0) MPM video
│   ├── t000_setup2/         # θ(0) setup-2
│   ├── t020_setup1/         # θ(20) MPM video
│   ├── ... (6 video / trajectory)
│   ├── theta_hat_t000.json  # joint inverse 结果
│   ├── theta_hat_t020.json
│   ├── theta_hat_t040.json
│   └── avrami_fit.json      # 拟合的 (T̂_k, σ̂_Y∞) ± 1σ
├── trajectory_ID_02/  ...
└── recovery_summary.csv     # 12 行 (T_k*, σ_Y∞*, T̂_k, σ̂_Y∞, error %)
```

### 4.6 论文角色

放在 §7.6.1 ─ "Synthetic time-resolved validation" ─ 跟现有 §7.3(static synthetic)互补:
- §7.3 验证"pipeline 在 static θ 下恢复正确"
- §7.6.1 验证"pipeline 在 time-varying θ(t) 下恢复正确"
- §7.6.2 onwards 真实材料

---

## 5. L3:新鲜度 panel - 通用 session 协议

### 5.1 时间表(每 session 60-70 min)

```
═══════ 准备阶段 ═══════
t = -10 min  按"第 6 节 各材料配方"准备 400 g batch
             → 倒进 25°C 恒温容器静置
t =  -2 min  从 [2, 7]² 连续 random 抽 (W_1, H_1)(用 numpy.random,seed 记录)
             调节 setup-1 mold 至该尺寸(或 snap 到最近现成 mold)
             取一小份 batch ~ setup-1 容积大小,进 setup-1 mold
             → 释放闸门 + 拍 8 帧 → 保存为 ref_<...>_t000_setup1
             立即 offline 跑 single-setup inverse 得到 θ̂_1(t≈0)
             → 跑 Hessian-orthogonal proposer 得到 (W_2*, H_2*)
             → 调节 setup-2 mold 至该尺寸
             清模 setup-1
             把 setup-2 mold 准备到工作台

═══════ Pair 1(t = 0)═══════
t =   0 min  取 ~setup-1 容积的 aliquot 进 setup-1 mold
             → 释放 + 拍 → 保存为 ref_<...>_t000_setup1_pair1
             清模 setup-1(~5 min)
             换 setup-2 mold 到摄像位
             取 ~setup-2 容积的 aliquot 进 setup-2 mold
             → 释放 + 拍 → 保存为 ref_<...>_t000_setup2_pair1
             清模 setup-2(~5 min)
             把 setup-1 mold 复位回摄像位

═══════ Pair 2(t = 20)═══════
t =  20 min  同 Pair 1,文件名后缀改为 _t020_

═══════ Pair 3(t = 40)═══════
t =  40 min  同上,_t040_

═══════ 收尾 ═══════
t =  60 min  session 结束
             smooth 子集(片栗粉×2 + 白玉×1 + repeat 1):
               → 留 ~50 g batch 残料,密封,4°C 冰箱,等 24h follow-up
             chunky 子集(oats×2 + パンケーキ×1):
               → 残料丢弃(rheo 测不可靠)
             清洁所有 mold
```

### 5.2 每 session 的实验日记本应记录

模板保存到 `data/freshness_2026-05-09/lab_journals/session_<NN>.md`:

```markdown
# Session #__ - <Material> <Conc>%

日期: ____________
材料: ____________
浓度: ____________

## Batch 制备
Batch 总质量: ______ g(秤量记录)
水/牛乳质量: ______ g
干粉质量: ______ g
室温: ______ °C
准备时间(分秒): 起 ______  终 ______  耗时 ______
微波/kettle: ______
搅拌时长: ______ sec

## Setup 选择
Random sampled (W_1, H_1) = (____, ____) cm  [seed=____]
Actual setup-1 mold (W, H) = (____, ____) cm  [snap/调节误差 = ____ cm]
t≈0 snapshot inverse:
  θ̂_1 = (n=____, η=____, σ_Y=____ Pa)
  identifiable: (n=__, η=__, σ_Y=__)
Hessian-orthogonal proposed (W_2*, H_2*) = (____, ____) cm
Actual setup-2 mold (W, H) = (____, ____) cm

## 3 Pair 实测
Pair 1 (t=0):
  Setup-1 释放 ______:______ aliquot=____ g  [filename: ____________]
  Setup-2 释放 ______:______ aliquot=____ g  [filename: ____________]
Pair 2 (t=20):
  ...同上
Pair 3 (t=40):
  ...同上

## 异常 / 注意
_____________________________________________

## 24h Rheo 安排(仅 smooth)
Sample 留量: ____ g, 4°C 冰箱位置: ______
预约 rheometer 时段: Day 4 上午 ______
```

---

## 6. L3:6 条件具体配方(400 g batch)

### 6.1 オートミール 5%(20 g 燕麦 + 380 g 热水)

```
session 编号建议: #5
─────────────────────────────
1. 称 20 g 即时燕麦片 → 玻璃容器 A
2. 称 380 g 矿泉水 → 容器 B,微波到 ~70 °C(~1 min,800W)
3. 把热水倒入容器 A,搅 30 sec(避免颗粒结团)
4. 容器 A 转入 25 °C 恒温浴,t = 0 开始计时
5. 等到 t ≈ 8 min(看 batch 部分稠化)再做 setup-2 选择
6. 颗粒处理:在每个 aliquot 取样前**充分搅拌 10 sec**,避免颗粒沉到容器底
```

**注意**:燕麦片含全粒,aliquot 中颗粒分布不均会引入 measurement noise。**取样前必搅**。

**24h rheo**:✗(颗粒大,rheometer 测不可靠)

**估计 T_k**:30-60 min(直链淀粉,中等)

### 6.2 オートミール 9%(36 g 燕麦 + 364 g 热水)

```
同 6.1,但用 36 g 干粉。批次更稠,搅拌更费力。
```

**估计 T_k**:20-40 min(浓度高,T_k 缩短)

### 6.3 片栗粉 3%(12 g + 388 g 水)

```
session 编号建议: #3
─────────────────────────
1. 称 12 g 片栗粉 → 玻璃容器
2. 加 388 g 室温矿泉水
3. 搅 30 sec(粉先在冷水中分散)
4. 容器入微波,30 sec × 2 加热到 ~70 °C 直到液体变清亮 gel
   (此时片栗粉糊化为透明 gel)
5. 立即从微波拿出,搅 30 sec 防止表面 skin
6. 转入 25 °C 恒温浴,t = 0 开始计时
```

**注意**:片栗粉一旦糊化即开始 retrograde,所以 t=0 取样要快(~5 min 内)。

**24h rheo**:✓(smooth gel,rheometer 可测)

**估计 T_k**:30-40 min

### 6.4 片栗粉 7%(28 g + 372 g 水)

```
同 6.3,但用 28 g 干粉。注意微波加热时间可能要稍长(40 sec × 2)。
batch 比 3% 浓得多,搅拌时容器壁会粘料 → 用刮刀贴壁刮净。
```

**估计 T_k**:20-30 min(浓度高,retrograde 快)

### 6.5 白玉粉 7%(28 g + 372 g 水)

```
session 编号建议: #1
─────────────────────────
1. 称 28 g 白玉粉(糯米粉)
2. 加 372 g 水
3. 搅 60 sec(白玉粉颗粒细但易结小球,需要充分搅匀)
4. 微波 30 sec × 3,直到清亮 gel
5. 从微波取出,搅 30 sec
6. 转入 25°C
```

**注意**:白玉粉的 retrograde 比片栗粉慢,但 σ_Y∞ 通常更高。**这是 panel 中最稳的"长 T_k"代表**。

**24h rheo**:✓

**估计 T_k**:60-120 min

### 6.6 パンケーキ粉 70%(280 g 粉 + 120 g 冷牛乳)

```
session 编号建议: #2
─────────────────────────
1. 称 280 g パンケーキ混合粉
2. 加 120 g 冷牛乳(从冰箱直接拿)
3. 搅 60 sec(注意粉容易飞溅)
4. 转入 25°C 恒温浴(此次不微波,材料是冷拌)
5. t = 0 计时
```

**注意**:
- パンケーキ粉中的膨松剂(baking powder)会缓慢释 CO2,容器留 1/3 空间
- 取样前搅,避免颗粒沉降
- 不要让 batch 在 25°C 放超过 6h(防霉)

**24h rheo**:✗(颗粒 + 膨松剂气泡)

**估计 T_k**:90-180 min

---

## 7. L3 批次重复:片栗粉 5%(独立 2 batch)

### 目的

证明 hand-prepared 材料的 (T_k, σ_Y∞) 跨 batch **存在显著漂移**,从而:
- 论证 Hamamichi 的 hypothetical cross-batch protocol 在该类材料上 ill-defined
- Within-batch trajectory recovery 是该类材料的自然单位

### 协议

```
Batch A:Day 2 上午 / session #4(片栗粉 5%,20 g + 380 g 水)
   按 6.3 协议(同片栗粉)做完整 3 时间点 session
   → 得到 (T_k^A, σ_Y∞^A)

Batch B:Day 3 上午 / session #11(片栗粉 5%,20 g + 380 g 水,完全独立的新 batch)
   严格相同的配方,但:
   - 用新的水(不同批的矿泉水,或者煮开后冷却的水)
   - 用新的粉(从粉罐底部取,或者新开包装)
   - 不同的搅拌时长(允许自然变异,不刻意复制)
   - 25°C 静置,但室温微差
   按 6.3 协议做完整 3 时间点 session
   → 得到 (T_k^B, σ_Y∞^B)

对比:
   Δ_T_k = |T_k^A - T_k^B| / mean(T_k)  ← 期望 30-50%
   Δ_σ_Y∞ = |σ_Y∞^A - σ_Y∞^B| / mean(σ_Y∞)  ← 期望 30-50%
```

**注意**:Batch A 与 Batch B 都做完整 24h rheo 扫描。

### Setup 复现性

Batch B 的 setup-1 / setup-2 几何**重新随机抽**,**不刻意复制 Batch A 的几何选择**(否则就成了"协议复现性测试"而不是"材料 batch 测试")。

---

## 8. 24h Rheometer Follow-up(Day 4 上午)

| Sample | Source | 体积 |
|--------|--------|------|
| 片栗粉 3% — 24h plateau | session #3 残料 | 5 mL |
| 片栗粉 5% Batch A — 24h plateau | session #4 残料 | 5 mL |
| 片栗粉 7% — 24h plateau | session #6 残料 | 5 mL |
| 白玉 7% — 24h plateau | session #1 残料 | 5 mL |
| 片栗粉 5% Batch B — 24h plateau | session #11 残料 | 5 mL |
| L1 4 个材料 — fresh 流曲线 | Day 1 上午直接做 | 5 mL × 4 |

**Rheometer 协议**:Anton-Paar / 平板,gap 1 mm,25°C(Peltier),频率扫描 0.01-100 s⁻¹,记录全 flow curve csv。

**保存到**:`data/freshness_2026-05-09/rheometer/<material>_<conc>_<24h_or_fresh>.csv`

---

## 9. Offline 后处理(Day 4 下午,全自动 batch 跑)

### 9.1 提取 y_obs

```bash
cd Fast-Non-Newtonian-ViRheometry-via-Mixture-of-GP-Surrogates

# Sim panel(L0):
for d in data/freshness_2026-05-09/L0_sim/trajectory_ID_*/t???_setup?; do
    python -m Calibration.extract_flow_distance --dir $d
    python -m Calibration.build_y_obs --dir $d
done

# L1 静态:
for d in data/freshness_2026-05-09/L1_static/ref_*; do
    python -m Calibration.extract_flow_distance --dir $d
    python -m Calibration.build_y_obs --dir $d
done

# L3 panel + repeat 同上 ...
```

### 9.2 跑反演

```bash
# Sim panel(L0):
python scripts/run_synthetic_freshness.py --panel-csv data/freshness_2026-05-09/L0_sim/prescribed_panel.csv

# L1: 4 mat × joint 2-setup
for mat in SweetBeanPaste Lotion MoisturisingMilk Tonkatsu; do
    python -m Optimization.estimate_first_setup -f data/.../L1_static/ref_${mat}_<W1>_<H1>_1
    python -m Optimization.estimate_joint_setup \
        -f data/.../L1_static/ref_${mat}_<W1>_<H1>_1 \
        -s data/.../L1_static/ref_${mat}_<W2>_<H2>_2 \
        --shape-warm-start-from-first --shape-cma-restarts 5 --shape-report-ci
done

# L3 主 panel: 6 cond × 3 time × joint
for cond in oats_5 oats_9 katakuriko_3 katakuriko_7 shiratama_7 pancake_70; do
    for t in 000 020 040; do
        python -m Optimization.estimate_joint_setup \
            -f data/.../L3_panel/ref_${cond}_<W1>_<H1>_t${t}_setup1 \
            -s data/.../L3_panel/ref_${cond}_<W2>_<H2>_t${t}_setup2 \
            --shape-warm-start-from-first --shape-cma-restarts 5 --shape-report-ci
    done
done

# L3 batch repeat: 1 cond × 2 batch × 3 time
for batch in A B; do
    for t in 000 020 040; do
        python -m Optimization.estimate_joint_setup \
            -f data/.../L3_repeat/ref_katakuriko_5_batch${batch}_<W1>_<H1>_t${t}_setup1 \
            -s data/.../L3_repeat/ref_katakuriko_5_batch${batch}_<W2>_<H2>_t${t}_setup2 \
            --shape-warm-start-from-first --shape-cma-restarts 5 --shape-report-ci
    done
done
```

每个调用输出 `theta_hat.json`,包含:
- `theta_hat`: {n, eta, sigma_y}
- `theta_ci_95`: 各参数 95% Laplace CI
- `identifiable`: 各参数 Boolean
- `z_dispersion`: 5-restart 散度

### 9.3 Avrami 拟合(每条件一次)

`scripts/fit_avrami.py`:输入 condition 的 3 个时间点 σ_Y(t_i) + 95% CI 半宽,加上 24h rheo σ_Y(plateau)(若有 smooth),用 weighted nonlinear least squares 拟合 σ_Y(t) = σ_Y∞ (1 - exp(-t / T_k))。

输出:`data/freshness_2026-05-09/avrami_fits.csv` ─ 7 行(6 主 + 1 repeat 的 batch B,batch A 已经在 6 主里),含 (T_k, σ_Y∞) ± 1σ。

### 9.4 Δ_pair joint validity check

`scripts/joint_validity_check.py`:对每个时间点 t_i 计算
```
Δ_pair(t_i) = |σ̂_Y,setup1(t_i) - σ̂_Y,setup2(t_i)| / σ̂_Y,joint(t_i)
```
输出表 ─ 6 cond × 3 time = 18 个 Δ_pair 值。

### 9.5 Hoover 浓度缩放

`scripts/concentration_scaling.py`:对所有 4 source 的 σ_Y∞(c) 散点做 log-log 线性回归,输出 α 估计 + 95% CI,以及画图 fig:hoover。

### 9.6 Sim diff-ref 误差报表

`scripts/run_synthetic_freshness.py` 内置:对 12 个 prescribed trajectory 输出 (T_k*, σ_Y∞*) vs (T̂_k, σ̂_Y∞) 散点 + 误差直方图。
→ tab:sim-recovery 表 + fig:sim-recovery 图。

### 9.7 Batch repeat 对比

对比 (T_k^A, σ_Y∞^A) vs (T_k^B, σ_Y∞^B),输出 `tab:batch-repeat`。

### 9.8 24h rheo plateau 锚点

读 `data/.../rheometer/*.csv`,在 1 s⁻¹ 处或低剪切平台读取 σ_Y_rheo,与 Avrami 拟合的 σ_Y∞ 对比,填入 `tab:fresh-anchors` anchor-B 列。

---

## 10. 数据 deliverable 与论文 figure / table 映射

| 数据 | 进 paper |
|------|----------|
| L0 sim 12 trajectory recovery | **新表** `tab:sim-recovery`(12 行 prescribed vs recovered) + **新 figure** `fig:sim-recovery`(prescribed vs recovered 散点) |
| L1 4 个新材料 θ̂ | `tab:thirteen-materials`(填 4 行)+ `fig:flow-12`(填 4 个 panel) |
| L3 6 条 σ_Y(t) trajectory + Avrami fit | `fig:fresh-traj`(改 6-panel)+ `tab:fresh-drift`(6 行) |
| L3 batch repeat 对比 | **新表** `tab:batch-repeat`(2 行,batch A vs batch B)|
| Δ_pair 18 个值 | **新表** `tab:joint-validity`(6 cond × 3 time) |
| Hoover 缩放 | **新 figure** `fig:hoover`(log-log 散点 + 回归)|
| 24h rheo plateau anchor | `tab:fresh-anchors` 的 anchor-B 列 |
| L3 dambreak 静帧 | `fig:fresh-demos` 代表性 |

---

## 11. 时间表(三天 lab + 一天 rheo + 算力夜跑)

| Day | 上午 | 下午 |
|-----|------|------|
| **Day 0**(夜)| ─ | sim panel L0 算力跑(~6 GPU-h) |
| **Day 1** | L1 4 mat × 1 pair + L1 流变仪扫描 | L3 #1 白玉 7% + #2 パンケーキ 70% |
| **Day 2** | L3 #3 片栗粉 3% + #4 片栗粉 5% Batch A | L3 #5 オートミール 5% + #6 片栗粉 7% |
| **Day 3** | L3 #7 オートミール 9% + #11 片栗粉 5% Batch B(repeat)| 数据整理,提取 y_obs |
| **Day 4** | 24h rheo follow-up(片栗粉×2, 白玉×1, repeat batch B)| Offline 后处理 |

---

## 12. 失败模式与回退

| 失败 | 怎么发现 | 回退 |
|------|---------|------|
| 某 session 的 Δ_pair > 50%(joint validity 严重失败)| 后处理 output | 此 time point 的 joint estimate 不进 trajectory,只用 single-setup 估计 |
| 某 session 的 t≈0 snapshot inverse 给出 sub bbox 之外的 θ̂ | snapshot 阶段 identifiable flag = false | 重抽 setup-1 几何重做 snapshot;或用第二个候选 mold 当 setup-1 |
| 浓度估错(σ_Y∞ 落出 [10, 100] Pa 训练 box)| 24h rheo 看 plateau 偏离 | 重做 batch,浓度调 ±30%,记录原批次为"out-of-box" 不进 main panel |
| 批次重复 batch B 的 (T_k, σ_Y∞) 与 batch A **相差 < 10%**(意外的"高重现性")| 后处理对比 | 仍可写进论文,但论证转向"即便如此可重现,Hamamichi cross-batch 仍要 6× compute + σ_Y(t=0) noise" |
| Sim diff-ref 上 12 个 trajectory 中 > 30% 拟合 (T_k*, σ_Y∞*) 误差 > 30% | sim post-processing 表 | 检查 prescribed 范围是否落出训练 box,重新设计 prescribed panel |
| 24h rheo 平台 σ_Y_rheo 与 Avrami σ_Y∞ 相差 > 50% | 后处理对比 | 写进论文 limitation,可能由 rheometer 0.01-100 s⁻¹ 范围内的 thixotropy / structure-break 引入 |
| パンケーキ batch 中颗粒沉降导致两 setup aliquot 浓度不一致 | 看 σ_Y(t) 出现非单调跳跃 | 取样前必充分搅 30 sec |
| 视频 frame 提取失败(背景污染、闸门反光等)| Calibration extraction 报错 | 手动复查 frame_001-008,必要时重拍该 pair |

---

## 13. Pre-registration checklist(贴到 lab 入口)

进 lab 之前最后核对一次:

- [ ] Sim panel L0(12 trajectory)已经夜跑完,recovery error < 30% 在 ≥ 8/12 个 trajectory 上
- [ ] Mold 装置可以连续调节 (W, H) ∈ [2, 7] cm (或确认 snap 误差 < 0.5 cm)
- [ ] 相机三脚架固定,`camera_params.xml` 当天 ChArUco 复检 IoU > 0.99
- [ ] 25°C 恒温浴/Peltier 开机预热 30 min
- [ ] 干粉、矿泉水、牛乳、4 个市贩材料全部备齐
- [ ] 电子秤 ±0.1 g
- [ ] 微波炉、kettle 测试可用
- [ ] 计时器(手机或秒表)
- [ ] Lab journal markdown template 复制好
- [ ] 数据目录 `data/freshness_2026-05-09/` 已创建
- [ ] 备份硬盘连接(防数据丢失)
- [ ] 第二天 rheometer 时段已确认

---

## 附录 A:一个完整 session 的实际操作示例(片栗粉 5% Batch A)

```
07:55  到 lab,材料从恒温柜取出
08:00  开 25°C 恒温浴
08:05  电子秤 zero
       称 20.0 g 片栗粉 → 玻璃容器(记录:实际称 20.02 g)
08:08  量 380 mL 矿泉水(实际 380 g 因密度 1.00)
08:10  把水倒入粉容器,搅 30 sec
08:12  容器入微波,30 sec × 2,中间搅拌一次
       第二次出来已变清亮 gel
08:15  容器从微波拿出,搅 30 sec(防 skin)
       转入 25°C 恒温浴 → t = 0 开始计时(在 lab journal 记下挂钟时间 08:16)

08:18  numpy.random.uniform(2, 7, 2) → 抽到 (W_1, H_1) = (5.3, 4.8) cm
       调节 dam-break mold 至 (5.3, 4.8)(或用最近 mold (5, 5),记录 Δ=0.36 cm)
       清洁 mold,放摄像位
       取 ~24 g aliquot 进 mold → 释放 + 拍 8 帧
       保存为 data/.../L3_panel/ref_katakuriko_5_5.3_4.8_t000_setup1/
       
08:20  跑 single-setup inverse(在另一台电脑或后台):
       python -m Optimization.estimate_first_setup -f .../t000_setup1 ...
       输出 θ̂_1 ≈ (n=0.65, η=8.2, σ_Y=25 Pa) [示例]
       
       跑 proposer:输出 (W_2*, H_2*) ≈ (3.1, 6.4) cm
       调节 setup-2 mold 至 (3.1, 6.4)

08:23  清模 setup-1
       把 setup-2 mold 准备好

╔═══ Pair 1 (t = 0)═══╗

08:24  取 ~24 g 进 setup-1 mold,等 settle 30 sec
08:25  释放 + 拍 → ref_katakuriko_5_5.3_4.8_t000_setup1_pair1/
       清模 ~ 5 min
08:30  换 setup-2 mold 进摄像位
       取 ~22 g 进 setup-2 mold,等 settle 30 sec
08:31  释放 + 拍 → ref_katakuriko_5_3.1_6.4_t000_setup2_pair1/
       清模 ~ 5 min
08:36  把 setup-1 mold 复位

╔═══ Pair 2 (t = 20)═══╗

08:36  正好 t=20 min  —— 立刻取 24 g 进 setup-1 mold
08:36  释放 + 拍
08:38  清模
08:43  换 setup-2 mold,取 22 g 进
08:43  释放 + 拍
08:45  清模

╔═══ Pair 3 (t = 40)═══╗

08:56  ── (从 08:50 开始 standby) ──
       释放 + 拍 setup-1
       清模
       换 setup-2 mold,释放 + 拍
       清模

09:08  session 结束(t ≈ 50 min)
       留 ~50 g batch 残料密封,4°C 冰箱
       lab journal 填完

总用时:~70 min calendar(从 08:00 开始 batch 制备到 09:08 结束)
       实操约 50 min,等待约 20 min
```

---

## 附录 B:`scripts/run_synthetic_freshness.py` 草案(我会写完整版)

```python
"""
L0 Sim diff-ref panel:在 prescribed time-varying θ(t) 上验证 trajectory recovery。

Usage:
    python scripts/run_synthetic_freshness.py \
        --prescribed-csv data/freshness_2026-05-09/L0_sim/prescribed_panel.csv \
        --out-root       data/freshness_2026-05-09/L0_sim/
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def avrami(t, sigma_inf, T_k, sigma_0=0.0):
    return sigma_0 + (sigma_inf - sigma_0) * (1 - np.exp(-t / T_k))

# 1. Load prescribed panel
panel = pd.read_csv(args.prescribed_csv)  # 12 行,columns: T_k, sigma_inf, sigma_0, n, eta_inf

# 2. For each trajectory, simulate + invert 3 time points
results = []
for _, row in panel.iterrows():
    traj_id = row['id']
    sigma_y_t = [avrami(t, row['sigma_inf'], row['T_k'], row['sigma_0']) 
                 for t in [0, 20, 40]]
    
    # MPM forward + joint inverse at each t_i
    theta_hat_traj = []
    for t, sy in zip([0, 20, 40], sigma_y_t):
        theta_t = (row['n'], row['eta_inf'], sy)
        # random setup-1
        W1, H1 = np.random.uniform(2, 7, 2)
        # MPM forward setup-1 (call out to Taichi MPM driver)
        run_mpm_forward(theta_t, W1, H1, out=f"L0_sim/{traj_id}/t{t:03d}_setup1")
        # single-setup inverse
        theta_1 = run_single_inverse(f"L0_sim/{traj_id}/t{t:03d}_setup1")
        # proposer
        W2, H2 = propose_next_setup(theta_1, W1, H1)
        # MPM forward setup-2
        run_mpm_forward(theta_t, W2, H2, out=f"L0_sim/{traj_id}/t{t:03d}_setup2")
        # joint inverse
        theta_joint = run_joint_inverse(f"L0_sim/{traj_id}/t{t:03d}_setup1", 
                                        f"L0_sim/{traj_id}/t{t:03d}_setup2")
        theta_hat_traj.append(theta_joint)
    
    # Avrami fit on recovered σ_Y(t_i)
    sy_hat = [th[2] for th in theta_hat_traj]
    popt, _ = curve_fit(avrami, [0, 20, 40], sy_hat, p0=[row['sigma_inf'], row['T_k']])
    
    results.append({
        'traj_id': traj_id,
        'T_k_true': row['T_k'], 'T_k_hat': popt[1],
        'sigma_inf_true': row['sigma_inf'], 'sigma_inf_hat': popt[0],
        'T_k_err_pct': abs(popt[1] - row['T_k']) / row['T_k'] * 100,
        'sigma_inf_err_pct': abs(popt[0] - row['sigma_inf']) / row['sigma_inf'] * 100,
    })

pd.DataFrame(results).to_csv(args.out_root + '/recovery_summary.csv')
```

(完整版会处理 MPM driver 的实际调用、文件结构、err handling 等。)

---

## 附录 C:与现有 repo 文档的关系

- 本文件**取代** `FRESHNESS_EXPERIMENT_DESIGN_CN.md`(那个是旧 6 mat × 6 time 设计)
- 本文件 V3.1 取代了 V3.0(8 mold 离散池版本)
- 与 `METHOD_2026-05-07.md` 中描述的 production code path 完全兼容
- 新增 `scripts/run_synthetic_freshness.py`(即将写)+ `scripts/fit_avrami.py` + `scripts/joint_validity_check.py` + `scripts/concentration_scaling.py`

---

## 反馈循环

执行过程中遇到任何**与本文档不一致**的实际情况,在 lab journal 里记下:
- 偏离了哪一步、为什么
- 实际做了什么
- 是否影响后续 step

完成实验后我们一起回顾日志,把 protocol 升级为 V3.2 / V4,并在论文里写明实际偏差。
