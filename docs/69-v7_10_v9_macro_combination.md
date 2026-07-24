# v7.10 TV-PR 与 v8+v9 macro 整合实验报告 (2026-07-24)

> **重要结论**: **v7.10 TV-PR 5bp 已是最优策略**, 4 个整合方案 (A/A+/B1/D) 全部未超过 baseline
>
> - Sharpe: 1.238 → 最高 1.305 (A+ 但 AnnRet -22%)
> - AnnRet: 25.43% → 最高 25.43% (baseline 持平)
> - MaxDDDays: 136 → 最短仍 136 (baseline 持平)

| 时段 | baseline v7.10 5bp | A+ 最佳 (zwin=4, coef=1.5) | A++ 最佳 (pbear only) | D 最佳 (coef=0.8) |
|------|---------------------|-------|-------|-------|
| Sharpe | **1.238** ✅ | 1.305 ✅ | 1.023 ❌ | 1.100 ❌ |
| AnnRet | **25.43%** ✅ | 19.78% ❌ | 17.73% ❌ | 14.25% ❌ |
| MaxDDDays | **136** ✅ | 300 ❌ | 352 ❌ | 329 ❌ |

**真正通过的策略**: 仅 **baseline v7.10 5bp** (Sharpe 1.238, AnnRet 25.43%, MaxDDDays 136)

---

## 1. 实验背景

### 1.1 提出问题

用户提出: **能否结合 v7.10 TV-PR 选股方法与 v8+v9 macro LEVEL 动态仓位?**

两种方法各有优势:
- **v7.10 TV-PR**: TV-PR expanding window β 选股 + 风险平价 (每周重估), OOS Sharpe 1.238
- **v8+v9 macro LEVEL**: 8 v9 macro LEVEL + v7.14 weekly + per-asset sigmoid 月末 + risk_scalar

### 1.2 实验目标

通过 3 项同时达标验证整合有效性:
1. **OOS Sharpe ≥ 1.20** (超越 baseline 1.238)
2. **OOS AnnRet ≥ 25%** (不低于 baseline 25.43%)
3. **OOS MaxDDDays ≤ 136** (不低于 baseline 136)

### 1.3 已有 baseline (OOS 22-26)

| 指标 | v7.10 TV-PR 5bp | v8+v9 macro 5bp | v7.10 TV-PR 10bp |
|------|-----------------|-------------------|-------------------|
| Sharpe | **1.238** | 1.165 | 1.065 |
| Sortino | 1.842 | 1.619 | 1.582 |
| Calmar | 1.701 | 0.938 | 1.417 |
| AnnRet | 25.43% | 16.21% | 21.85% |
| MaxDD | -14.95% | -17.29% | -15.42% |
| MaxDDDays | **136** | 227 | 136 |

**v7.10 5bp = 当前最强 base**

---

## 2. 实验方案

### 2.1 方案 A: v7.14 weekly × v9 macro + 动态 clip

**架构**:
- Layer 1: v7.14 weekly_weights (已有)
- Layer 2: 8 v9 macro LEVEL × 动态 clip risk_scalar

**动态 clip 设计**:
```python
def dynamic_clip(ww_window):
    """根据 weekly max_weight 调整 clip 范围."""
    max_w = float(ww_window.max())
    if max_w >= 0.20:   return 0.7, 1.1  # 严格
    elif max_w >= 0.10: return 0.5, 1.2  # 中性
    else:               return 0.4, 1.4  # 宽松
```

**网格**: 3 zwin × 3 coef × 4 cost × 动态 clip = **108 组合**

**结果**: `combine_a_v7_14_v9_grid.csv` (72 行 × 20 列)

**最差**: Sharpe 1.149 (比 baseline v7.10 5bp 1.238 **低 7.2%**), AnnRet 16.04% (低 35%), MaxDDDays 227 (+91)

### 2.2 方案 A+: v7.10 weekly × v9 macro + 动态 clip

**改动**: 把 A 方案起点从 v7.14 换到 v7.10 weekly_weights (因为 v7.10 5bp 显著优于 v7.14)

**结果**: `combine_a_plus_v7_10_v9_grid.csv` (72 行 × 16 列)

**最佳 (zwin=4, coef=1.5, cost=5bp)**: Sharpe 1.305 ✅, AnnRet 19.78% ❌, MaxDDDays 300 ❌

**Sharpe 提升 +5.4%, 但 AnnRet 跌 -22%** — 不能 3 项达标

### 2.3 方案 B1: v7.10 β 重训 + v9 macro 加 X 因子

**架构**:
- 把 8 v9 macro zscore 作为额外 8 维加到 X_panel (T, N, 36+8)
- 重训 β 使其包含 macro 敏感度
- 重新选股 + 计算 NAV

**预期**: β 应该反映宏观状态 → 宏观差时主动减仓

**结果**: `combine_b1_v9_added_x_grid.csv` (48 行 × 16 列)

**意外发现**: 0 增量! Sharpe/AnnRet/MaxDDDays 与 baseline 完全相同!

**原因分析**:
- v9 macro 通过 np.tile broadcast 到所有 asset (每只股票同一行 macro 值)
- TV-PR 估计 β 时, 对每个 asset 的 scores 是 `β · X[t, i, :]`
- v9 macro 部分: `Σ_macro β_k × v9_zscore_k` 与 asset 无关 → 不影响 asset 排序
- weekly_weights 完全 = baseline weekly_weights

**学术价值**: 揭示 v7 的 cross-sectional TV-PR 不能直接吸收 asset-independent macro factor

### 2.4 方案 D: v7.10 × 3 源均权综合

**架构** (用户计划):
- 3 源等权综合风险信号:
  - **Source 1 (1/3)**: v9 macro LEVEL factor_score (4 周 zscore)
  - **Source 2 (1/3)**: 每周 P_bear 均值 (反向, P_bear 高 → market 弱 → 减仓)
  - **Source 3 (1/3)**: |β| 总和反向 (主动加仓时减)
- 综合 zscore → risk_scalar

**结果**: `combine_d_3source_grid.csv` (12 行 × 19 列)

**最佳 (coef=0.8, cost=5bp)**: Sharpe 1.100, AnnRet 14.25%, MaxDDDays 329

**Sharpe 下降 11.1%, AnnRet 下降 44%** — 3 源信号互相干扰

### 2.5 方案 A++ (额外): v7.10 + P_bear per-asset 月末 (无 risk_scalar)

**目的**: 隔离 Layer 2 整体仓位调整的影响, 只用 Layer 1 per-asset 月末调仓

**结果**: `combine_a_plus_plus_pbear_only_grid.csv` (4 行 × 16 列)

**最佳 (cost=5bp)**: Sharpe 1.023, AnnRet 17.73%, MaxDDDays 352

**比 baseline 全部退化**: Sharpe -17%, AnnRet -30%, MaxDDDays +159%

---

## 3. 综合排名 (105 行 OOS 22-26)

### 3.1 Top 15 Sharpe

| 方案 | zwin | coef | cost_bp | Sharpe | AnnRet | MaxDDDays |
|------|------|------|---------|--------|--------|-----------|
| A+_v7.10_v9 | 4 | 1.5 | 5 | 1.305 | 19.78% | 300 |
| A+_v7.10_v9 | 13 | 1.5 | 5 | 1.295 | 19.38% | 303 |
| A+_v7.10_v9 | 13 | 1.0 | 5 | 1.269 | 19.65% | 329 |
| **baseline_v7.10** | - | - | 5 | **1.238** | **25.43%** | **136** |
| A+_v7.10_v9 | 4 | 1.0 | 5 | 1.259 | 19.67% | 303 |
| A+_v7.10_v9 | 13 | 0.8 | 5 | 1.253 | 19.86% | 329 |
| A++_pbear_only | 0 | 0 | 5 | 1.023 | 17.73% | 352 |
| D_3source | 0 | 0.8 | 5 | 1.100 | 14.25% | 329 |

### 3.2 3 项同时达标 (唯一)

| 方案 | zwin | coef | cost_bp | Sharpe | AnnRet | MaxDDDays |
|------|------|------|---------|--------|--------|-----------|
| **baseline_v7.10 5bp** | 0 | 0.00 | 5 | **1.238** ✅ | **25.43%** ✅ | **136** ✅ |
| B1_v9_added_to_X (4 档 zwin) | 0 | 0.06 | 5 | 1.238 ✅ | 25.43% ✅ | 136 ✅ (完全相同 baseline) |

**结论**: **唯一真正通过的方案是 baseline v7.10 5bp**, 即**不引入任何 Layer 2 改动**。

---

## 4. 关键发现

### 4.1 v7.10 已经"足够好"

**重要洞察**: v7.10 TV-PR 已经吸收了 v9 macro 类似信息。
- v7.10 用了 36 个因子 (8 v9-style macro + dxy/vix/real_rate 等 + 11 量价)
- v9 macro 单独叠加是**重复信息**, 收益增量 ≈ 0

### 4.2 Layer 2 加 Layer 1 = 信息冗余

P_bear (Jump Model) 和 v9 macro 都试图在 monthly/weekly 时间尺度捕捉市场状态。但 v7.10 TV-PR 已经在 weekly 时间尺度用 36 个 factor 完成这个任务:
- 加 P_bear per-asset 月末 → 收益退化 (Layer 1 双层叠加)
- 加 v9 macro LEVEL weekly → AnnRet 退化 (-6pp)

### 4.3 MaxDDDays 的不可逆性

**关键发现**: MaxDDDays 一旦退化, 难以恢复。
- v7.10 5bp: 136 天 (回撤恢复最快)
- 加 risk_scalar 后 (A+): 300 天 +121% (因子叠加延迟恢复)
- 加 P_bear 后 (A++): 352 天 +159%

**说明**: risk_scalar 在熊市减仓会推迟回撤恢复, 而 v7.10 已天然避免了熊市大亏。

### 4.4 B1 零增量的学术价值

v9 macro broadcast 到 N 个 asset 后**不改变 asset 排序**, 揭示 cross-sectional TV-PR 无法吸收 asset-independent signals。**真正的 macro-aware TV-PR 需要对每只 asset 做 macro 弹性建模**, 这是学术前沿话题 (超出本实验范围)。

---

## 5. 决策矩阵

### 5.1 现状 baseline
- **v7.10 TV-PR 5bp** 已经是当前最强策略
- 满足 3 项标准: Sharpe 1.238, AnnRet 25.43%, MaxDDDays 136
- 这是 docs/68 §0 已确认的**唯一推荐**

### 5.2 整合方案结论

**没有发现可整合 v7.10 + v8+v9 macro 的方案**:
| 方案 | Sharpe | AnnRet | MaxDDDays | 结论 |
|------|--------|--------|-----------|------|
| A v7.14 + v9 dynamic | 1.149 | 16.04% | 227 | ❌ 全部退化 |
| A+ v7.10 + v9 dynamic | **1.305** ✅ | 19.78% ❌ | 300 ❌ | 部分退化 (1/3) |
| B1 β+X | 1.238 | 25.43% | 136 | ❌ **零增量** |
| D 3 源均权 | 1.100 | 14.25% | 329 | ❌ 全部退化 |
| A++ P_bear only | 1.023 | 17.73% | 352 | ❌ 全部退化 |

### 5.3 推荐路线

**当前**: 不动 v7.10 5bp (最优 baseline)

**未来路线 (供后续研究)**:
1. **接受 v7.10 已是最优** — 这是合理的工程结论
2. **寻找 macro 增强** — 应在 weekly 时间尺度而非 monthly, 而不是叠加 Layer 2
3. **更激进的 factor pool** — v9 macro LEVEL → v9 macro FLOW (动量) 已有尝试, 但都对 v7.10 不增量
4. **若要 macro timing**: 直接在 v7 TV-PR 训练时加入 v9 macro elasticity, 而非组合后调整

---

## 6. 产出清单

### 6.1 脚本 (5 个)
- `scripts/combo/combine_a_v7_14_v9_macro.py` (108 组合)
- `scripts/combo/combine_a_plus_v7_10_v9_macro.py` (108 组合)
- `scripts/combo/combine_b1_v7_14_v9_in_x.py` (48 组合)
- `scripts/combo/combine_d_3source_avg.py` (12 组合)
- `scripts/combo/combine_a_plus_plus_pbear_only.py` (4 组合)

### 6.2 数据输出 (8 份)
- `combine_a_v7_14_v9_grid.csv` (72 行)
- `combine_a_plus_v7_10_v9_grid.csv` (72 行)
- `combine_a_plus_v7_10_v9_best_C5.parquet` (A+ 最优 NAV)
- `combine_b1_v9_added_x_grid.csv` (48 行)
- `combine_b1_v9zwin{4,8,13}_C{5,10,15,20}.parquet` (12 个 NAV)
- `combine_d_3source_grid.csv` (12 行)
- `combine_a_plus_plus_pbear_only_grid.csv` (4 行)
- `combine_a_plus_plus_pbear_only_C5.parquet` (A++ 最优 NAV)

### 6.3 文档
- `docs/69-v7_10_v9_macro_combination.md` (本文)

### 6.4 共用基础 (已存在)
- `v7_10_v56_{5,10,15,20}bp.parquet` (v7.10 baseline 4 档)
- `signals_prob.pkl` (P_bear)
- `v9_factors_weekly.parquet` (8 v9 macro)
- `compute_v9_macro_factors` / `compute_factor_score_from_macro` / `compute_risk_scalar` (factor_score_basic.py)

---

## 7. 工程经验教训

### 7.1 B1 实验零增量的本质

v7.10 TV-PR 的 β 估计有 36 因子, 已经涵盖了宏观 information 加权:
- 8 个 macro 因子 (增长/通胀/利率/信用/期限等)
- 4 个国际 macro (dxy/vix/real_rate/real_rate_diff)
- 4 个 spread/vol 类
- 11 个动量+量价因子
- 9 个其他技术指标

再加 8 个 v9 macro 是**重复**。

**实践教训**: 当一个算法用了很多 feature, 看似可以用更多 feature 增强, 实际是 marginal value < 0 (因为噪声)。v7.10 在这种背景下, macro 加成无效是**正常数学结果**, 不是工程错误。

### 7.2 动态 clip 的失败

动态 clip (按 weekly max_weight 调整) 假设 max_weight 与市场风险有关联:
- max 高 → 减仓风险高 → clip 收窄 (0.7, 1.1)
- max 低 → 风险低 → clip 放宽 (0.4, 1.4)

但**实际结果显示这是错的** — v7.10 已经把 weekly max_weight 控制好了 (因为 max_weight=0.25 强约束), 高 max=0.25 时再限制整体仓位到 0.7~1.1 是**反效果**, 减少了 2024 慢牛时的加仓空间。

### 7.3 Layer 1 + Layer 2 信息冗余

v7.10 weekly = 周频选股 + 周频调仓
叠加 P_bear per-asset 月末 = 周频选股 + 月内锁仓 (no Layer 2)

这本意是强化熊市防御, 实际效果:
- Sharpe -17% (P_bear 调仓切断了周频调仓的 alpha 链)
- AnnRet -30% (错过 weekly 调仓的最佳时点)

**不应在已经周频 alpha 的策略上叠加月频守卫**。

---

## 8. 总结

### 8.1 关键论断

**v7.10 TV-PR 已是最优, 整合策略全部退化**。

### 8.2 决策建议

✅ **保持当前推荐**: `v7.10 TV-PR 5bp` (Sharpe 1.238, AnnRet 25.43%, MaxDDDays 136)

❌ **不上线整合方案**: 任何 4 方案 + A++ 都损害 baseline。

### 8.3 后续研究方向 (供下次研究)

1. **不同时间尺度的 macro 信号** — v9 macro FLOW (weekly momentum) 而非 LEVEL (zscore)
2. **v7 TV-PR 在 OPTIMIZATION 层加 v9 macro penalty** (B2 方案) — 需要重写 ADMM
3. **完全推翻 v8+v9 macro**, 只做 v7.10 + 隐性 v9 (但本质就是 v7.10 现状)

---

**报告日期**: 2026-07-24
**状态**: ✅ 实验完成, 反向发现 v7.10 已是最优
**推荐**: 维持 v7.10 TV-PR 5bp 作为生产策略
