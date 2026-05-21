# Shopping List — Freshness Pilot Experiment

5 材料 × 1 batch × 1 timepoint pair = pilot 范围。如果要做 3 replicates
就乘 3 倍食材。

---

## A. 食材

### Pantry (假设已有)

| 食材 | 单 pilot 用量 | 备注 |
|------|-------------|------|
| 牛乳 | ~200 mL (1 材料) | カスタード 用 |
| 砂糖 (上白糖 or グラニュー糖) | ~50 g (2 材料) | カスタード + 半 whipped cream |
| 薄力粉 | ~10 g (1 材料) | カスタード 用 |
| 片栗粉 | ~50 g (1 材料) | 片栗粉糊 用 |
| **即時オートミール** | **~40 g (备用 1 次)** | **材料 6 备用 fallback (任一材料失败时顶替). 干燕麦保质期 ~6 月, 一袋常备** |

### 需购入 (1 pilot 一次完成所需)

| 食材 | 数量 | 用途 | 概算费 (円) |
|------|------|------|-----------|
| **全卵 (M サイズ)** | **6 個** (3 全卵 给 カスタード, 2 個 拆 yolk/white) | カスタード + 卵黄 emulsion + 打蛋白 | 200 |
| **生クリーム 35%** | **200 mL** (1 パック) | 半 whipped 生クリーム | 400 |
| **サラダ油** (小ボトル) | 100 mL | partial 卵黄 emulsion | 200 |
| **食用色素** (option, visual hook) | 1 セット | SIGGRAPH headline 图 | 300 |

→ **食材总计 ¥1,100** (无 dye 则 ¥800)

### 升级到 3 replicate (后续阶段)

× 3 食材, 但**生クリーム 保质期 5-7 日**, 一定要**分批购入**。

---

## B. 消耗品

| 品目 | 数量 | 用途 | 推荐购入 | 概算费 (円) |
|------|------|------|--------|-----------|
| **使い捨てプラ容器** (500-700 mL, 蓋付) | ~30 個 | batch 保管 + aliquot 取り出し | 100 入り pack | 1,500 |
| **計量カップ 50 mL** | 2 個 | ρ_real 称重 critical | 単品 | 300 |
| **計量カップ 200/300 mL** | 各 2 個 | 配方計量 | 単品 | 600 |
| **泡立て器** (manual whisk, 30 cm) | 1 | マヨ・カスタード・ベシャメル | — | 800 |
| **使い捨て ヘラ・スプーン** | ~20 本 | 攪拌・取り出し | pack | 300 |
| **キッチンペーパー** | 1 ロール | 清掃 | — | 200 |
| **使い捨て手袋** (powder-free) | 1 箱 100 枚 | 衛生 | — | 800 |

→ **消耗品总计 ¥4,500** (一次性, 多次实验共用)

---

## C. 必须 lab-持 hardware (假设已有, 否则一次购入)

| 品目 | 必要 数 | 一次性购入费 (円) |
|------|--------|---------------|
| **ハンドミキサー** (5 段速度) | 1 | 3,000 |
| **微波炉 500W** | 1 | (lab 应该有) |
| **熱水壺** | 1 | 2,000 |
| **電子秤** ±0.1 g | 1 | 3,000 |
| **冷蔵庫** 5 °C | 1 | (lab 应该有) |
| **(W, H) acrylic 模具 9 個** ([2.5, 6.5] × [2.5, 6.5] cm 网格抽点) | 9 | 15,000-30,000 (acrylic 工房) |
| **カメラ + 三脚 + ChArUco 校正板** | 1 set | (already exists per Calibration/) |
| **温度計** デジタル | 1 | 1,500 |
| **大型ステンレスボウル** (2 L+) | 3 | 1,500 |
| **タイマー** | 1 | (智能手机 OK) |

→ **一次性 hardware ~¥30,000-45,000** (大部分 lab 应该已有)

---

## D. 数据 / 软件 (已存在 repo)

| 项目 | 状态 |
|------|------|
| `camera_params.xml` 校正流程 | ✓ exists (`Calibration/prepare_configs.py`) |
| `Optimization/libs/mechanism.py` (Hessian proposer) | ✓ |
| `Models/yshape_mogp_production/` 25 gid all calibrated | ✓ (2026-05-10 audit) |
| `scripts/run_y8q_prior.py` (inverse driver) | ✓ |
| `freshness/scripts/run_kinetics_handmade.py` | ✗ **要写** |
| `freshness/scripts/rho_rescale.py` | ✗ **要写** |
| `freshness/scripts/fit_T_k.py` | ✗ **要写** (或 reuse existing) |

---

## E. 总费用

### 1 次 pilot session (5 材料, 1 batch each)

| カテゴリ | 円 |
|---------|-----|
| 食材 | 1,100 |
| 消耗品 | (一次性 4,500, 分摊) |
| **一次 pilot 仅食材消耗** | **~¥1,100 (~$8 USD)** |

### 完整 paper-grade (5 材料 × 3 replicate + Mode B 3 手技 × 1 plateau)

| カテゴリ | 円 |
|---------|-----|
| 食材 (5 mat × 5 batch × 1.5 buffer) | ~10,000 |
| 消耗品 (一次性 4,500) | 4,500 |
| Hardware (一次性, 假设 lab 部分有) | 15,000-30,000 |
| **总 (initial)** | **~30,000-45,000 (~$200-300 USD)** |

---

## F. 鸡蛋 economy 详细

每 pilot 5 材料用:
- **カスタード**: 全卵 2 個 (whole)
- **卵黄 emulsion**: 蛋黄 2 個 → 用 2 個 卵 (拆 yolk/white)
- **打蛋白**: 蛋白 2 個 ← **来自上面 emulsion 副产 whites**

→ pilot 实际 **6 個鸡蛋** (2 + 2 + 2), 不浪費。

3 replicates × 6 = 18 個 (1.5 dozen, ¥600)

---

## G. 生クリーム 保质期 caveats

生クリーム (heavy cream 35%) 賞味期限 **5-7 日**。如果做 3 replicates 跨
2 周:
- 不要一次买完 (会 spoil)
- 每周 1 次 super run, 买 1-2 パック
- 优先 long-shelf-life 商品 (无菌包装 UHT 处理, 賞味期限 ~30 日)

预算: 2 パック × 4 周 = 8 パック ≈ ¥3,200

---

## H. 失败 batch 处理

大部分 pilot 材料**即使失败也可食用** (custard / 牛乳ベース / 卵黄 emulsion /
ホイップ / 片栗粉糊), **实验后吃光**避免浪費。

打蛋白 (无糖) 不好吃, 直接弃。

---

## I. 一周准备 checklist

实验前一周:
- [ ] 确认 lab 设备清单 (microwave, 冰箱, ハンドミキサー, 電子秤, 模具)
- [ ] 模具 (W, H) 9 個 制作 / 订购
- [ ] camera + 三脚 setup, ChArUco 板准备
- [ ] 一次性消耗品购入 (¥4,500)

实验前 1 天:
- [ ] 食材购入 (生クリーム ← 新鲜, 鸡蛋 M サイズ × 6, サラダ油 100 mL)
- [ ] camera_params.xml 校准跑通
- [ ] `freshness/scripts/` 中的辅助脚本就位 (run_kinetics_handmade, rho_rescale)
- [ ] batch_metadata.json template 准备

实验当天:
- [ ] 8:30 到 lab, camera + mould setup
- [ ] 9:00 dry-run 1 batch 片栗粉糊 (15 min, 验证 pipeline)
- [ ] 9:15-14:00 跑 5 材料 × 1 hour each
- [ ] 14:00-15:00 sync 数据 + 离线分析
