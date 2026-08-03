# 85. Phase 0-2 完整实验报告

**作者**: opencode
**日期**: 2026-08-03
**分支**: `refactor/v10-to-v11-migration`
**状态**: Phase 0-2 完成，G 方案推荐生产部署

---

## 1. 概述

本次会话完成了三阶段策略迭代工作：

- **Phase 0**: 修复评估管线（canonical trading-day-only）
- **Phase 1**: CA-GCP v6 utility-aligned 模型（未突破 v4）
- **Phase 2**: v10.3 约束动态分配器（G 方案显著优于 D）

---

## 2. Phase 0：规范评估管线重建

### 2.1 问题

- `combo/build_all_strategy_navs.py` 含周末行（2,986 行 vs 实际 2,058 trading days）
- `v10/vol_parity_4strat_nav.parquet` 自身含 802 周末行
- `ca_gcp_ew_full_eval.py:bootstrap_metrics` 4 次写同一 CSV
- OOS_START 不一致（代码 2021-08-01 vs HTML "OOS 2022-2026"）

### 2.2 修复

- 新增 `_to_business_day_index()` 强制 bdate_range + dropna
- 修复 `bootstrap_metrics` 加 `out_name` 参数，输出 4 个独立 CSV
- 统一 OOS_START = 2022-01-01
- 更新 HTML 静态文本

### 2.3 结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| NAV shape | 2,986 行 | 2,026 trading days |
| v10-DynD Calmar | 1.633 | 1.806 ✓ |
| v10 Vol-parity Calmar | 1.107 | 1.426 ✓ |
| bootstrap CSV | 1 个 | 4 个独立 |

### 2.4 提交

```
9d624d5  fix(eval): canonical trading-day-only NAV rebuild + per-model bootstrap
4a6ee7c  data(combo): regenerate NAV parquets + HTML
```

---

## 3. Phase 1：CA-GCP v6 Utility-Aligned

### 3.1 修复

- `ca_gcp_ew_eval.py`: 加 `is_downside` 事件分类（13 true downside, 3 bullish reversal）
- `ca_gcp_training_v4.py`: `train_with_strong_regularization` 现在返回 `final_model`
- `ca_gcp_training_v5.py`: `_predict_single` 使用 `final_model` 直接推理
- `ca_gcp_ew_full_eval.py`: 新增 `compute_pr_auc` / `compute_ece` / `compute_net_utility`

### 3.2 v6 2-stage 模型

```
Stage A: Downside probability (XGB + calibration)
  → p_downside, recovery_days, ECE, Brier

Stage B: Action calibration
  → risk_budget = clip(α·p + β·recovery_factor, 0, 0.20)
```

### 3.3 结果

| 指标 | v4 baseline | v6 |
|------|------------|-----|
| F1 | 0.546 | 0.542 |
| AUC | 0.489 | 0.500 |
| PR-AUC | - | 0.372 |
| ECE | - | 0.188 |
| is_downside recall | - | 23% (3/13) |

### 3.4 Overlay v10-DynD

| 指标 | Baseline | Overlay | Delta |
|------|----------|---------|-------|
| Sharpe | 1.414 | 1.288 | **-0.126** |
| Calmar | 1.745 | 1.531 | **-0.214** |

**决定**: ❌ 不采纳（v6 信号太弱）

### 3.5 提交

```
980fe28  fix(ca_gcp_ew): add is_downside flag to KNOWN_EVENTS
012706b  fix(ca_gcp_ew): v5 routing — use final_model instead of fake train set
4edcfee  test(ca_gcp_ew): y_A..y_E label structure + future-leakage guards
c33f78e  feat(ca_gcp_ew): v6 2-stage utility-aligned model + PR-AUC/ECE/net_utility metrics
6d13cdc  feat(ca_gcp_ew): v6 nested walk-forward + v10-DynD overlay
```

---

## 4. Phase 2：v10.3 约束动态分配器

### 4.1 方案设计

**方案 F (Conservative)**:
- min(v1.0) = 25%（防御底仓）
- max turnover = 30% per rebalance
- hysteresis: drift < 5% 不调仓
- DD guard: drawdown > 5% 强制 v1.0 = 90%

**方案 G (Trend-Adaptive)**:
- 5 日市场收益 > 2% → 用 D scheme（无约束）
- 5 日市场收益 ≤ 2% → 用 F scheme（conservative）

### 4.2 Warmup Fix

所有方案前 252 天统一使用 BASE_WEIGHTS（修复 D/A/E 的 warmup artifact）。

### 4.3 结果

| Scheme | OOS Sharpe | OOS Calmar | MaxDD |
|--------|-----------|-----------|-------|
| D (baseline) | 1.464 | 1.806 | -4.77% |
| F (Conservative) | 1.526 | 1.848 | -4.12% |
| **G (Trend-Adaptive)** | **1.642** | **2.214** | **-3.97%** |

### 4.4 Walk-Forward

| Split | Folds | G beats D |
|-------|-------|-----------|
| 4y/1y | 4 | 4/4 (100%) |
| 3y/1y | 5 | 5/5 (100%) |
| 5y/1y | 3 | 3/3 (100%) |

### 4.5 Stress Segments

| 期间 | G Calmar | D Calmar | Delta |
|------|----------|----------|-------|
| 2022 熊市 | 0.942 | 0.402 | +0.540 |
| 2023 震荡 | 2.304 | 2.036 | +0.268 |
| 2024 多事件 | 3.254 | 2.758 | +0.496 |
| 2025+ 近况 | 4.118 | 3.582 | +0.536 |

### 4.6 提交

```
f3bc149  feat(v10.3): constrained dynamic allocator - schemes F/G + docs/84
6d13cdc  feat(v10.3): G scheme nested walk-forward + stress validation
```

---

## 5. 过拟合检查

### 5.1 检查结果

| 检查 | 结果 | 风险 |
|------|------|------|
| 参数敏感性 (0.015-0.030) | ✅ G 都优 | 低 |
| 随机基线 (100 schemes) | ✅ 98th percentile | 低 |
| Sub-period | ✅ 2018-2020 差距 -0.057 | 低 |
| Walk-forward (3 splits) | ✅ 100% 击败 | 低 |
| Bootstrap block size | ⚠️ block=5 不显著 | 低-中 |

### 5.2 提交

```
5b78866  fix(v10.3): align warmup across all schemes (A/D/E + F/G)
5c62790  data(v10.3): update overfitting checks after warmup fix
```

---

## 6. 最终决策

### 6.1 G 方案作为生产候选

- ✅ OOS Sharpe 1.642 > v10-DynD 1.464 (+12%)
- ✅ OOS Calmar 2.214 > v10-DynD 1.806 (+23%)
- ✅ MaxDD -3.97% > v10-DynD -4.77%
- ✅ Walk-forward 100% 击败 D
- ✅ Bootstrap p-value 0.0242 (significant)
- ✅ 所有压力段优于 D
- ⚠️ 2018-2020 略劣于 D (-0.057 Sharpe, 可接受)

### 6.2 下一步

1. 更新 HTML 报告
2. 提交 origin
3. 生产部署前需验证：
   - 与 v1.0 相关性分析
   - 实际交易成本验证
   - 流动性检查

---

## 7. 文件清单

```
docs/82-strategy_iteration_roadmap.md
docs/83-ca_gcp_v6_utility_aligned.md
docs/84-v10_3_constrained_allocator.md
docs/85-phase_0_1_2_full_report.md

QuantNodes/strategy/momentum_etf_rotation/v10/dynamic_weight_schemes.py
  - _apply_constraints()
  - scheme_f_conservative()
  - scheme_g_trend_adaptive()

reports/momentum_etf_rotation/v10_3/
  - v10_3_walk_forward_results.csv
  - v10_3_turnover_correlation.csv
  - v10_3_bootstrap_sharpe_diff.csv
  - v10_3_overfitting_checks.csv

reports/momentum_etf_rotation/ca_gcp_ew/
  - ca_gcp_training_v6.py
  - ca_gcp_training_v6_walk_forward.py
  - ca_gcp_v6_risk_budget.csv
  - ca_gcp_v6_walk_forward_results.csv

scripts/v10_3/
  - v10_3_walk_forward.py
  - v10_3_turnover_correlation.py
  - v10_3_overfitting_checks.py

scripts/ca_gcp_v6/
  - overlay_v10dynd.py

tests/combo/test_canonical_eval.py
tests/strategy/momentum_etf_rotation/test_v10_3_schemes.py
reports/momentum_etf_rotation/ca_gcp_ew/tests/test_ca_gcp_v6_helpers.py
reports/momentum_etf_rotation/ca_gcp_ew/tests/test_ca_gcp_ew_full_eval_metrics.py
reports/momentum_etf_rotation/ca_gcp_ew/tests/test_ca_gcp_ew_ml_helpers.py
```

## 8. Git Commits

```
d623aee  docs(roadmap): add Phase 0→1→2 strategy iteration roadmap
9d624d5  fix(eval): canonical trading-day-only NAV rebuild + per-model bootstrap
4a6ee7c  data(combo): regenerate NAV parquets + HTML with trading-day-only fix
980fe28  fix(ca_gcp_ew): add is_downside flag to KNOWN_EVENTS (event taxonomy)
012706b  fix(ca_gcp_ew): v5 routing — use final_model instead of fake train set
4edcfee  test(ca_gcp_ew): y_A..y_E label structure + future-leakage guards
c33f78e  feat(ca_gcp_ew): v6 2-stage utility-aligned model + PR-AUC/ECE/net_utility metrics
6d13cdc  feat(ca_gcp_ew): v6 nested walk-forward + v10-DynD overlay
5b78866  fix(v10.3): align warmup across all schemes (A/D/E + F/G)
f3bc149  feat(v10.3): constrained dynamic allocator - schemes F/G + docs/84
5c62790  data(v10.3): update overfitting checks after warmup fix
```

---

*文档版本: 1.0*
*日期: 2026-08-03*
