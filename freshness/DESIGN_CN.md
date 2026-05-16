# 新鲜度实验设计 — 2026-05-14

A 类手作材料的 σ_y(t) 测量, 用 dam-break camera-inverse 在 20 s/timepoint 内
characterize **hand-made variability** + **freshness 动力学**。**微波 +
熱水壺-only** lab, 6(+1) 材料 × replicate + technique sweep。

---

## 1. 理论基础

设计建立在六个已建立的理论体系上, 我们在上面加一个方法学组合。

### 1.1 Inverse problem + Hessian-orthogonal 实验设计
- **Hamamichi (2023)** — dam-break + MPM-direct CMA-ES inverse for HB
  parameters; 定义 `propose_next_setup` (Hessian-orthogonal setup-2
  proposer), 我们忠实移植到 `Optimization/libs/mechanism.py`
- **Atkinson, Donev, Tobias (2007)** *Optimum Experimental Design with
  SAS* (Oxford) — A-/D-optimal design 教科书
- **Tarantola (2005)** *Inverse Problem Theory and Methods for Model
  Parameter Estimation* (SIAM) — Bayesian inverse 框架

### 1.2 时间分辨流变学 + 动力学拟合
- **Bates & Watts (1988)** *Nonlinear Regression Analysis and Its
  Applications* (Wiley) — 从协方差矩阵 (T_k, σ_∞) ± Laplace CI
- **Larson (1999)** *The Structure and Rheology of Complex Fluids* (OUP)
  — time-dependent rheology 教科书
- **Mewis & Wagner (2009)** "Thixotropy" *Adv Colloid Interface Sci*
  147–148: 214–227

### 1.3 Foam aging (相关 #1 メレンゲ, #2 ホイップ)
- **Princen (1979)** "Highly concentrated emulsions: I. Cylindrical
  systems" *J Colloid Interface Sci* 71: 55–66 — foam σ_y ∝ (σ/R) ×
  f(φ) 基本理论
- **Princen (1983)** "Rheology of foams and concentrated emulsions: II"
  *J Colloid Interface Sci* 91: 160–175 — yield stress vs bubble
  fraction
- **Mason, Bibette, Weitz (1995)** "Yielding and flow of monodisperse
  emulsions" *J Colloid Interface Sci* 179: 439–448 — universal scaling
- **Stang & Schubert (1992)** "Bubble coalescence kinetics" *Chem Eng
  Technol* 15: 372–378 — coalescence timescale τ ∝ R²/D
- **Dickinson (2010)** "Food emulsions and foams: stabilization by
  particles" *Curr Opin Colloid Interface Sci* 15: 40–49

### 1.4 Egg/dairy 蛋白质 thermal gelation (相关 #4 カスタード)
- **Mleko, Foegeding (1999)** "Formation of whey protein gels at high
  pH" *J Food Sci* 64: 209–212 — milk protein thermal gel
- **Croguennec, Nau, Brulé (2002)** "Influence of pH and salts on egg
  white gelation" *J Food Sci* 67: 608–614 — egg white denaturation
  kinetics
- **Singh (2011)** "Aspects of milk-protein-stabilised emulsions" *Food
  Hydrocoll* 25: 1938–1944

### 1.5 Starch + fat gel cooling (相关 #5 ベシャメル, #7 片栗粉)
- **Karim, Norziah, Seow (2000)** "Methods for the study of starch
  retrogradation" *Food Chem* 71: 9–36
- **Miles, Morris, Ring (1985)** "Gelation of amylose" *Carbohydr Polym*
  5: 17–32
- **Roos (1995)** *Phase Transitions in Foods* (Academic Press)

### 1.6 Gluten relaxation (相关 #6 うどん生地)
- **Bagley & Christianson (1986)** "Response of dough to uniaxial
  compression: stress relaxation properties" *Cereal Chem* 63: 220–223
- **Dobraszczyk & Morgenstern (2003)** "Rheology and the breadmaking
  process" *J Cereal Sci* 38: 229–245

### 1.7 实验方法学 (replicate + dose-response)
- **Box & Draper (1987)** *Empirical Model-Building and Response
  Surfaces* (Wiley) — replicate + factor-sweep 设计
- **Roache (1998)** *Verification and Validation in Computational
  Science and Engineering* — multi-anchor V&V
- **Macosko (1994)** *Rheology: Principles, Measurements, and
  Applications* (Wiley-VCH) — rheometer protocol

### 1.8 我们的贡献（组合 + niche claim）

**手法**: σ_y(t) tracking via **dam-break camera + GP-surrogate inverse
(20 s/inverse)**

**密度修正 (核心 trick)**: 训练 sim 在 ρ_sim = 1.2 g/cm³, 但 A 类
手作材料 ρ_real 跨 batch 变化（特别 aerated foam ρ ∈ [0.4, 0.7]）。每
batch 称 50 mL → ρ_real, post-hoc

```
σ_y_real(t) = σ_y_sim(t) × ρ_real / 1.2
T_k_real    = T_k_sim                       (ρ-invariant)
```

T_k 对 ρ 不变（分子分母同 ρ-factor 抵消）, 所以 freshness 动力学 claim
对 batch 间 ρ 散布 robust。

**Niche claim** — 我们的快 inverse 解锁三类 measurement, rheometer +
Hamamichi 单独 都做不到:

1. **同手技 ×N replicate**: batch-to-batch 散布 quantification —
   rheometer 慢, hand-made variability 让标准化 σ_y_∞ 失效 (跨 batch
   散 50-200%)
2. **不同手技参数 (打發时间 / 火力 / 搅频度)** ×N sweep: dose-response —
   rheometer 太慢做 multi-parameter sweep
3. **同 batch ×6 timepoint trace**: 动力学 (T_k) — Hamamichi
   MPM-direct 30-60 min/inverse 跟不上 T_k ≈ 30-90 min 的 kinetics

---

## 2. 步骤计划

### Step 1 — 材料选择 (A 类, 微波 + 熱水壺-only)

| # | 材料 | 单 batch 配方 | 体积 | 手技 A 类源 | freshness 机制 | T_k 期 | σ_y_∞ 期 (Pa) | ρ_real | ρ-rescale 必须? |
|---|------|------------|------|-----------|-------------|-------|-------------|--------|---------------|
| 1 | **メレンゲ** | 蛋白 4 + 砂糖 160 g + lemon 数滴 | ~600 mL | 打發时间 (2/4/6 min) × 速度 (M/H) | bubble coalescence (Princen 1983) | 1-3 h | 50-200 | 0.40-0.55 | ✓ |
| 2 | **ホイップ生クリーム 35%** | 生クリーム 300 mL + 砂糖 30 g | ~500 mL | peak stiffness (soft/medium/stiff) | fat-foam coalescence | 30 min - 2 h | 100-300 | 0.45-0.65 | ✓ |
| 3 | **マヨネーズ自家製** | 卵黄 2 + サラダ油 500 mL + 酢 30 mL + 塩 5 g | ~530 mL | 油加入速度 (滴下 5 min vs 細流 1 min) | emulsion 油水分离 (Mason 1995) | hours | 100-300 | 0.95 | × |
| 4 | **手作りカスタード** | 全卵 3 + 砂糖 80 g + 小麦粉 20 g + 牛乳 500 mL | ~600 mL | 微波 burst 间 whisk 强度 → scramble vs smooth | egg protein gel + 冷却 (Mleko 1999) | 30-60 min 冷却 | 50-300 | 1.05 | × |
| 5 | **手作りベシャメル** | バター 40 g + 小麦粉 40 g + 牛乳 600 mL | ~650 mL | 牛乳加入速度 + whisk 攻撃性 → だま vs smooth | starch + fat gel + 冷却 | 30-60 min 冷却 | 100-500 | 1.05 | × |
| 6 | **手打うどん生地 (稀)** | 中力粉 200 g + 水 200 mL + 塩 4 g | ~400 mL | 捏ね时间 (5/10/20 min) × 力 | gluten 松弛 (Bagley 1986) | 30 min - 2 h | 200-800 | 1.15 | × |
| (7) | **手作り片栗粉葛** (option) | 片栗粉 50 g + 砂糖 30 g + 水 500 mL | ~550 mL | 微波 burst 间 stir timing | starch retrog 冷却 (Karim 2000) | 15-60 min 冷却 | 50-300 | 1.05 | × |

**覆盖 axes**:
- foam (1, 2) — sugar-stabilized 蛋白 vs fat-stabilized 乳脂
- emulsion (3) — 蛋黄 + 油
- thermal gel + 冷却 (4, 5, ±7) — egg / starch+fat / starch
- gluten 松弛 (6)
- 浓度扫描可选: ホイップ 35% vs 47%; うどん 粉:水 1:1 vs 1:1.2

### Step 2 — Setup 几何池 (一次性, 任何 capture 之前)

按 Hamamichi 2023 协议字面: **setup 1 随机 (无信息), setup 2 从 setup 1
inverse 派生**。

**(W, H) ∈ [2, 7] × [2, 7] cm box 内 25 个 gid 都已 v2 partition +
post-hoc c_k calibrated** (2026-05-10 audit), box 内任意 (W, H) 都覆盖。

**实际模具池**:
```
W: 2.5, 4.0, 5.5, 6.5  cm
H: 2.5, 4.0, 5.5, 6.5  cm
```
预备 ~9 个模具均散在 4×4 网格内（角点 + 内部）。

**每材料 session**:
- **Setup 1**: 从池中 uniform random 抽 (W, H)
- **Setup 2 per t_i**: `propose_next_setup(θ̂_1(t_i), setup1)` → 连续
  (W₂, H₂) → snap 到池中**不是 setup 1** 的最近模具

Camera + `camera_params.xml` 当天校准固定, 不在材料/时间点之间重新校准。

### Step 3 — 每材料 batch 制备 (微波 500W + 熱水壺-only)

详细配方:

```
1. メレンゲ
   蛋白 4 个 + 砂糖 160 g + lemon 数滴
   → 不需加熱; ハンドミキサー 中速度
   → 打發 X min (X = 2/4/6, 手技扫描参数)
   → 立即 transfer 容器, 5 °C 保温, t=0

2. ホイップ生クリーム
   生クリーム 300 mL (冷蔵) + 砂糖 30 g
   → ハンドミキサー 中速度, 不需加熱
   → 打發 to X stiffness (X = soft/medium/stiff, 視覚判定)
   → t=0, 5 °C 保温

3. マヨネーズ自家製
   卵黄 2 + 酢 30 mL + 塩 5 g (室温 25 °C 混合)
   → 泡立て器 手動
   → 加 サラダ油 500 mL: 加入速度 = X (X = 滴下 5 min / 細流 1 min)
   → t=0, 室温保温

4. 手作りカスタード
   全卵 3 + 砂糖 80 g + 小麦粉 20 g + 牛乳 500 mL → whisk 30 s 混合
   → 微波 500W × 1 min × 5 回, 间 whisk × X (X = 弱 5 s/中 15 s/強 30 s)
   → 冷蔵 5 °C, t=0 = 冷蔵开始时间

5. 手作りベシャメル
   バター 40 g 微波 30 s 溶 → +小麦粉 40 g whisk 30 s → 微波 30 s
   → +牛乳 100 mL whisk X 攻撃性 (X = 弱/中/強) → 微波 1 min
   → +残り 500 mL whisk → 微波 1 min × 3, 间 whisk X
   → t=0 = 完成時間, 室温→冷蔵

6. 手打うどん生地 (稀)
   中力粉 200 g + 塩 4 g 混合
   → 熱水壺 沸水 200 mL → 5 min 室温冷却 → 加入
   → 手こね X 时间 (X = 5/10/20 min, 手技扫描参数)
   → t=0 = こね完成時間, 室温保温

7. 手作り片栗粉葛 (option)
   片栗粉 50 g + 砂糖 30 g + 水 500 mL (冷水) → stir 30 s 混合
   → 微波 500W × 30 s × 6 回, 间 stir X timing (X = 即/5s後/10s後)
   → 冷蔵 5 °C, t=0 = 冷蔵开始時間
```

**为什么微波-cooked 系 (4, 5, 7) A 类更强**:

| Stovetop | 微波 + manual whisk |
|---------|---------|
| 连续加熱 + 持续搅 → uniform | 30-60 s burst → 中心過熱 / 周边冷点 |
| 火力数值固定 | W 数固定但空间分布不均 |
| 一气呵成 | Multi-step, 每次插入操作强度不同 |

同一配方两个人用微波做 ベシャメル / カスタード, **σ_y_∞ 散布 2-3×**
（有 だま vs smooth）。**这正是 paper 想 quantify 的 hand-variability**。

**ρ_real measurement**: 每 batch 制備完成 → 50 mL 量杯填满 → 称 (g) →
ρ_real = mass/50 → 记入 `batch_metadata.json`

**设备**: ハンドミキサー, 泡立て器, 微波炉 500W, 熱水壺, 冰箱 5 °C, 25 °C
室温保温容器, 電子秤 ±0.1 g, 量杯 50/100/200 mL, 容器多个。**不要
stovetop**。

### Step 4 — 每材料 capture session

**两种 mode**:

| Mode | 目的 | Batch 数 | Capture 数/材料 |
|------|-----|--------|--------------|
| **A — Replicate** (同手技 ×3 batch, 6 t_i 全 trace) | 同手技下 σ_y(t) 跨 batch 重复性 + 动力学 | 3 | 6 t × 2 setup × 3 batch = 36 |
| **B — Sweep** (不同手技 ×3 batch, 单 plateau t=240 min) | 手技参数 → σ_y_∞ dose-response | 3 | 1 t × 2 setup × 3 tech = 6 |

**时间点** (Mode A): t = {15, 30, 60, 90, 150, 240} min (6 点, ~4 h session)

**每 t_i 协议** (Hamamichi-derived, T_pair=8 min):

```
(setup 1 mould 在 session 前从模具池随机选好)

t_i − 5 min   取 aliquot ~250 mL, 装入选定的 setup 1 mold
t_i           setup 1 释放 (3 s)
t_i + ↓:      ─ 提取视频 → y_obs_1                          (~5-7 min, 自动)
              ─ 跑 setup-1-alone inverse → θ̂_1(t_i)         (~7 s)
              ─ propose_next_setup(θ̂_1, setup1) → (W₂, H₂)  (~1 s)
              ─ snap 到池中不是 setup 1 的最近 mould         (~10 s)
              ─ 取新 aliquot, 装入该 mold                    (~30 s)
t_i + 8 min   setup 2 释放 (3 s)
─────────────────────────────────────
T_pair ≈ 8 min 之间 setup 1 - setup 2
```

**Joint 有效性 gate**: T_pair = 8 min 内 σ_y 变化必须 < CI half-width
(~15%), 按一阶动力学这要求 **T_k ≥ 50 min**。

**T_k 估计 + 早 abort** (Mode A, 前 3 capture 后, t = 15, 30, 60 min):

```
1. 跑 setup-1-alone inverse on 3 capture → 3 个 (σ_y, η, n)
2. 拟合 σ_y(t) = σ_y_∞ (1 − exp(−t/T_k))    [exp first, §1.2]
3. 检查:
   ├─ σ_y_∞ ∈ [10, 500] Pa?     Yes → 继续
   │                              No  → 调配方 ±30-50 %, 重做 batch
   ├─ T_k ≥ 50 min?              Yes → 继续 (joint mode)
   │                              No  → 缩短到 T_pair=4 min (single-setup
   │                                    only, σ_y prior + identifiability
   │                                    flag); 若仍不行 → 弃材料
   ├─ sub bbox 含 σ_y(60)?       Yes → 继续
   │                              No  → 弃材料
   └─ 全过 → 继续 t = 90, 150, 240 min
```

**Mode B 协议**: 同材料 × 3 手技参数 → 每个 plateau (t=240 min) capture →
比较 σ_y_∞(技1) vs σ_y_∞(技2) vs σ_y_∞(技3)

### Step 5 — Joint inverse offline + ρ-rescale (每材料 ~10 min)

所有 capture 完, 跑 paper-grade joint inverse on 每对:

```bash
python scripts/run_y8q_prior.py --joint-only --restarts 5 \
    --data-root data/freshness_handmade \
    --material <Material>_<batch>_<t>min \
    --out scripts/run_kinetics_<Material>_<batch>.json
```

→ θ̂(t_i) per 时间点 with 95% Hessian Laplace CI (sim-space)

**Post-hoc ρ-rescale** (per batch):

```python
import json
meta = json.load(open('batch_metadata.json'))
rho_real = meta[batch_id]['rho_real']     # g/cm^3, measured at Step 3

sigma_y_real_per_t = sigma_y_sim_per_t * (rho_real / 1.2)
eta_real_per_t     = eta_sim_per_t * (rho_real / 1.2) ** (1.0 / (2 - n_sim))  # approx
T_k_real           = T_k_sim                                                  # invariant
```

拟合动力学 (Bates & Watts 1988):

```python
from scipy.optimize import curve_fit
def model_exp(t, sigma_y_inf, T_k):
    return sigma_y_inf * (1 - np.exp(-t / T_k))

popt, pcov = curve_fit(model_exp, t_array, sigma_y_real_array,
                        sigma=ci_halfwidth_array, absolute_sigma=True)
T_k, sigma_y_inf = popt[1], popt[0]
T_k_err = np.sqrt(pcov[1, 1])
```

如果指数残差 > 2× CI 带 在 > 30% 点上: 试 **Avrami n>1** (sigmoid 诱导期)
或 **双相** (例如 fast-collapse + slow-coalesce per Stang-Schubert 1992)

### Step 6 — Validation anchors (per material)

| Anchor | Mode A | Mode B | 理论 |
|---|:---:|:---:|---|
| **A. Hamamichi MPM-direct inverse on plateau y_obs (t=240)** | ✓ 主 | ✓ | 同 y_obs, 唯一 inverse 算法不同; 验证 absolute σ_y. Workflow: 复制 `ref_*_240min_*` 到 Hamamichi sibling repo, 跑他 inverse, 比 θ̂ |
| **B. Rheometer plateau (24 h)** | ✓ (3, 4, 5, 6, 7 only — foam 不行) | ✓ | Macosko 1994; 比 σ_y_rheo vs σ_y(240) |
| **C. Replicate consistency (Mode A 内 ×3)** | ✓ | n/a | Box & Draper 1987; CV < 25% target |
| **D. Visual MPM(θ̂) forward vs 原 video** | ✓ | ✓ | SIGGRAPH style; Snapdiff 风格叠加 |
| **E. ρ_real measurement consistency** | ✓ | ✓ | 跨 batch ρ 散布 quantify; rescale input 稳定性 |
| **F. 内部 dyn fit** | ✓ | n/a (single t) | Avrami / exp 残差 < 2× CI 在 > 70% 点 |
| **G. 手技 dose-response monotonic** | n/a | ✓ | Mode B 主验证: σ_y_∞ vs 手技 param 单调 |

**每材料 pass 标准**:
- **Mode A**: A 过 (≤ 25% σ_y_∞ 偏差 vs Hamamichi) AND (B 过 OR D + F 在
  SIGGRAPH-quality) AND C 内部一致性 (CV < 25%) AND F 残差 OK
- **Mode B**: G 单调 AND A 在 ≥ 2/3 手技点过

### Step 7 — 跨材料趋势

**H. Foam 物理 sanity (Princen 1983)**: σ_y_∞(メレンゲ) / σ_y_∞(ホイップ)
应符 bubble fraction × interfacial tension 期望

**I. 手技 dose-response per material** (Mode B):
- メレンゲ 打發时间 ↑ → bubble size ↓ → σ_y_∞ ↑ (Princen-Mason scaling)
- ホイップ peak stiff ↑ → φ_air ↑ → σ_y_∞ ↑
- マヨネーズ 滴下 (慢) → 油滴 size ↓ → σ_y_∞ ↑ + 稳定性 ↑
- カスタード whisk 弱 → scramble 多 → σ_y_∞ ↑
- ベシャメル whisk 弱 → だま 多 → σ_y_∞ ↓
- うどん 捏ね 时间 ↑ → gluten 网络 ↑ → σ_y_∞ ↑

**J. 跨机制 σ_y_∞ ranking**:
σ_y_∞(ベシャメル, うどん) > σ_y_∞(カスタード) > σ_y_∞(マヨネーズ) >
σ_y_∞(メレンゲ, ホイップ)
(cooked gel > emulsion > aerated foam, after ρ-rescale)

**方法验证 pass** if: Step 6 ≥ 4/6 材料过 AND Step 7 H ≥ 4/6 材料 monotonic
AND J 排序对

---

## 3. 时间预算

```
Step 1   材料选择                                  已定
Step 2   预备模具 (一次性)                         ½ day
Step 3+4 每材料 capture
         Mode A: 1 day × 3 replicate × 6 mat = 18 lab day (大型)
                 OR 1 day × 1 replicate × 6 mat = 6 lab day (minimal)
         Mode B: ½ day × 3 tech × 6 mat = 9 lab day
                 (可与 Mode A plateau 共享, 实际 +3 lab day)
         失败-恢复: 每失败 +½ day
Step 5   Offline joint inverse + ρ-rescale         ~15 min total (GPU)
Step 6   Validation
         A. Hamamichi MPM (sibling repo)           ~3-6 GPU-h (plateau only)
         B. Rheometer plateau × 5 non-foam         ½ day
         C-G: 自动                                  Step 5 同步
Step 7   Cross-material 趋势分析                    ½ day

──────────────────────────────────────────
Mode A only (1 replicate): ~7 lab day + 1 GPU 通宵       [minimal]
Mode A only (3 replicate): ~19 lab day + 1 GPU 通宵      [strong replicate claim]
Mode A + B:                ~9-22 lab day + 1 GPU 通宵    [full A 类 claim]
```

**时间紧 fall-back**:
1. Mode A: 1 replicate → 仅展示动力学, 不展示 batch-to-batch 散布
2. Mode B: 弃 → 仅展示 replicate + 动力学, 不展示手技 dose-response
3. 弃材料 7 (片栗粉葛) → 6 材料
4. 弃材料 6 (うどん) → 5 材料, focus 食品

---

## 4. 交付物

```
data/freshness_handmade/
  # Mode A (replicate + 动力学)
  ref_<Material>_<batch>_<t>min_<W1>_<H1>_1/  # × 6 t × 3 batch × 6 mat = 108 dirs
  ref_<Material>_<batch>_<t>min_<W2>_<H2>_2/  # × 6 t × 3 batch × 6 mat = 108 dirs

  # Mode B (手技 sweep, plateau only)
  ref_<Material>_<tech>_240min_<W1>_<H1>_1/   # × 3 tech × 1 t × 6 mat = 18 dirs
  ref_<Material>_<tech>_240min_<W2>_<H2>_2/   # × 3 tech × 1 t × 6 mat = 18 dirs

  # Rheometer (non-foam only)
  ref_<Material>_plateau_24h_*.csv            # × 5 (no メレンゲ/ホイップ)

  # Per-batch metadata (ρ_real, 手技 params)
  batch_metadata.json

scripts/
  run_kinetics_handmade.py    # per-batch + per-t_i joint inverse + ρ-rescale
  fit_T_k.py                  # Bates & Watts CI (existing, reused)
  plot_kinetics_handmade.py   # σ_y(t) trace × replicate + tech sweep panel
  rho_rescale.py              # post-hoc σ_y_real, η_real, T_k_real

docs/figs/
  kinetics_panel_6materials_handmade.png  # MAIN: σ_y(t) × 6 mat × 3 replicate
  tech_sweep_panel.png                    # σ_y_∞ vs 手技 param × 6 mat (Mode B)
  rho_rescale_validation.png              # ρ_real 散布 + rescale 一致性
  validation_matrix.png                   # 6 mat × 7 anchor pass/fail
  hamamichi_plateau_xcheck.png            # A. Hamamichi vs ours @ t=240
```

---

## 5. Pre-registration checklist

任何 capture 之前锁定:

- [ ] 材料: **1 メレンゲ, 2 ホイップ生クリーム, 3 マヨネーズ自家製,
      4 手作りカスタード, 5 手作りベシャメル, 6 手打うどん生地** (±7 片栗粉葛)
- [ ] Setup 1 每 batch 从 ~9 模具池 uniform 随机选 (W, H)
- [ ] Setup 2 每 t_i 从 `propose_next_setup(θ̂_1, setup1)` 派生
      (Hamamichi 2023), snap 到模具池中不是 setup 1 的最近 mould
- [ ] T_pair ≈ 8 min (per-time-point Hessian, manual extract ~5-7 min)
- [ ] 时间点 {15, 30, 60, 90, 150, 240} min 每材料 (Mode A)
- [ ] **Mode A**: 3 replicate / 同手技 / 6 t / 6 材料 = 108 setup 1 +
      108 setup 2 = 216 capture
- [ ] **Mode B**: 3 手技参数 / 1 plateau t / 6 材料 = 18 setup 1 +
      18 setup 2 = 36 capture
- [ ] **ρ_real measurement**: 每 batch 制備直後 50 mL → 称 → ρ_real
      记入 batch_metadata.json
- [ ] **Post-hoc rescale**: σ_y_real = σ_y_sim × ρ_real / 1.2; T_k 不变
- [ ] **Early abort** 前 3 capture: T_k ≥ 50 min AND σ_y_∞ ∈ [10, 500]
      Pa AND sub bbox 含 σ_y
- [ ] **Validation**:
  - A. Hamamichi MPM at plateau (t=240) per material — sibling repo
  - B. Rheometer plateau (non-foam only: 3, 4, 5, 6, 7)
  - C. Replicate CV < 25% (Mode A 内)
  - D. Visual MPM(θ̂) match SIGGRAPH-quality
  - E. ρ_real measurement consistency check
  - F. Dyn fit 残差 < 2× CI 带 在 > 70% 点
  - G. Mode B 手技 dose-response monotonic
- [ ] Pass = A + (B 或 D+F) + C per material; ≥ 4/6 must pass
- [ ] Cross-material: H (foam sanity), I (per-material dose-response),
      J (mechanism ranking)
- [ ] 动力学模型: exp first; sigmoid / Avrami n>1 / 双相 if 残差 demand
- [ ] 统计拟合: `scipy.optimize.curve_fit` weighted (sigma=CI
      half-widths, absolute_sigma=True); 报告 (T_k, σ_y_∞) ± 1-σ from
      covariance (Bates & Watts 1988)

Pre-registration 后偏离需书面修正。

---

## 6. 参考文献

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
