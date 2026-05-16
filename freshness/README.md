# Freshness Experiment Package

实验目的: 用 20 s/inverse 的快 dam-break camera-inverse 在单 batch 内
quantify 手作材料的 **σ_y(t) 动力学** + **手技 batch-to-batch 散布**, 这是
rheometer (需标准化样品) 和 Hamamichi MPM-direct (30 min/inverse 太慢)
都做不到的 niche。

---

## 文档总览

| 文档 | 用途 | 推荐阅读时机 |
|------|------|------------|
| **[PILOT.md](PILOT.md)** | 4-5 材料 × 1 hour/材料 快速验证版 (intermediate states + ρ-rescale) | **新手 / 第一次跑实验先看这个** |
| **[SHOPPING_LIST.md](SHOPPING_LIST.md)** | 食材 + 消耗品 + 器具清单 + 预算 | 实验前 1 天 |
| [DESIGN.md](DESIGN.md) | 完整 6 材料 paper-grade design (EN) | 写 paper 时回查 |
| [DESIGN_CN.md](DESIGN_CN.md) | 完整 design (CN) | 同上 |
| [PROTOCOL_V3.1_starch.md](PROTOCOL_V3.1_starch.md) | 淀粉糊专用 V3.1 锁定协议 (legacy) | 只跑淀粉系时参考 |

---

## 快速 entry

如果只想**今天先开始跑**:
1. 看 [SHOPPING_LIST.md](SHOPPING_LIST.md) → 确认手头有 / 要买
2. 看 [PILOT.md](PILOT.md) §"easiest first" → 先做 **片栗粉糊** 当 pipeline dry-run
3. 1 batch 跑通 → 按 PILOT.md 顺序加其他 4 材料

如果想**完整 paper-grade**:
1. 看 [DESIGN_CN.md](DESIGN_CN.md) Mode A + Mode B 完整 protocol
2. 6 材料 × 3 replicate × 6 timepoint = 完整数据集
3. 时间预算 ~7-22 lab day

---

## 关键 niche claim

我们的快 inverse 解锁三类 measurement，rheometer + Hamamichi 单独 都做不到:

1. **同手技 ×N replicate** → batch-to-batch 散布 quantify
2. **不同手技参数** (打發时间 / 火力 / 搅频度) → dose-response
3. **同 batch ×N timepoint** → 动力学 (T_k)

**核心 trick — ρ-rescale**: aerated 材料 ρ 跨 batch 散布, 每 batch 称
50 mL → ρ_real → post-hoc `σ_y_real = σ_y_sim × ρ_real / 1.2`。
**T_k 对 ρ invariant**, 动力学 claim robust。

---

## 数据 / 脚本 (尚未写, 后续添加)

```
freshness/
├── data/                        # capture 数据 (per-batch + per-timepoint)
│   └── ref_<material>_<batch>_<t>min_<W>_<H>_<setup_idx>/
├── scripts/
│   ├── run_kinetics_handmade.py    # per-batch + per-t_i joint inverse + ρ-rescale
│   ├── rho_rescale.py              # post-hoc σ_y_real, η_real
│   ├── fit_T_k.py                  # Bates & Watts non-linear regression CI
│   └── plot_kinetics_handmade.py   # σ_y(t) trace + replicate + tech sweep figs
└── batch_metadata.json          # per-batch ρ_real, 手技 params, 时间戳
```
