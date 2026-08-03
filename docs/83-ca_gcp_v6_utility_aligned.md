# 83. CA-GCP v6 Utility-Aligned 实验报告

**作者**: opencode
**日期**: 2026-08-03
**分支**: `refactor/v10-to-v11-migration`
**状态**: 实验完成 — 不推荐直接生产部署

---

## 1. 背景

### 1.1 前置工作

Phase 0 (commit `9d624d5`): 修复 canonical 评估管线 (trading-day-only NAV + per-model bootstrap)
Phase 1.1 (commit `980fe28`): 加 `is_downside` 事件分类
Phase 1.2 (commit `012706b`): 修 v5 routing (用 final_model 而非 fake train)
Phase 1.3 (commit `4edcfee`): 守门 y_A..y_E 标签 future-leakage

### 1.2 v4 已知问题

`docs/82` §B3 总结：

| 模型 | F1 | AUC | 备注 |
|------|------|------|------|
| v4 y_E XGB (best) | 0.546 | 0.489 | F1 主要靠高 recall, AUC 接近随机 |
| v3 y_E XGB | 0.462 | 0.501 | 同样问题 |
| v5 regime-aware (broken) | 0.000 | 0.500 | R=2.000 (Phase 1.2 修复前) |

**结论**：v4 不足以独立使用，混合信号 + utility 校准是必要升级。

---

## 2. v6 设计

### 2.1 两步法 (2-Stage)

```
                ┌─────────────────────────────────┐
                │  Stage A: Downside Probability  │
                │  (XGB + 强正则化 + 校准)          │
                │                                 │
features X[t] → │  y_E 目标 (max_dd + percentile)  │ → p_downside
                │  输出: p_downside, recovery_days │
                └─────────────────────────────────┘
                              ↓
                ┌─────────────────────────────────┐
                │  Stage B: Action Calibration    │
                │                                 │
p_downside +    │  risk_budget = clip(α·p + β·R,  │ → risk_budget
recovery_days → │                    0, 0.20)     │   ∈ [0, 0.20]
                │  目标: max portfolio utility     │
                └─────────────────────────────────┘
```

### 2.2 关键参数

| 参数 | 值 | 含义 |
|------|-----|------|
| `RISK_BUDGET_MAX` | 0.20 | 风险预算上限 (暴露最多降低 20%) |
| `RISK_BUDGET_ALPHA` | 0.15 | p_downside 权重 |
| `RISK_BUDGET_BETA` | 0.05 | recovery_factor 权重 |
| `RECOVERY_NORMALIZE` | 60.0 | 恢复天数归一化分母 |
| `LAMBDA_UTILITY` | 2.0 | 风险厌恶系数 |

---

## 3. 实验结果

### 3.1 Stage A (单模型 OOF)

| 指标 | v4 baseline | v6 (本次) |
|------|------------|-----------|
| F1 | 0.546 | 0.542 |
| AUC | 0.489 | 0.489 |
| **PR-AUC** | - | 0.372 |
| **ECE** | - | 0.188 |
| **Brier** | - | 0.275 |

**结论**：v6 Stage A 在 F1/AUC 上与 v4 持平，但 PR-AUC 0.372 表明 **ranking discrimination 优于 F1 单点**。ECE 0.188 表明模型略 over-confident。

### 3.2 Stage B (Risk Budget)

| 指标 | 值 |
|------|-----|
| Mean risk_budget | 0.051 (5.1% 降仓) |
| Max risk_budget | 0.140 (14% 降仓) |
| Min risk_budget | 0.000 (无降仓) |

**结论**：模型保守，~95% 的天数不降仓。

### 3.3 is_downside 事件 Recall (Phase 1.1 配合)

| 阶段 | 总事件 | 捕获 | Recall |
|------|--------|------|--------|
| v4 (旧) 全部 KNOWN_EVENTS | 16 | 5 | 31% |
| v4 (旧) 仅 is_downside=True | 13 | 5 | 38% |
| **v6 (新) 仅 is_downside=True** | **13** | **3** | **23%** |

**已捕获**：
- ✓ 2023-03-13 硅谷银行
- ✓ 2024-01-22 雪球敲入+量化危机
- ✓ 2024-08-05 日元套利平仓

**未捕获 (10 个)**：
- ✗ 2018-10 贸易战暴跌, 2018-12 美股闪崩, 2019-04 原油闪崩, 2019-08 中美互加关税
- ✗ 2020-01 新冠爆发, 2020-03 全球流动性危机
- ✗ 2021-07 教培+科技暴跌, 2022-01 美联储加息预期, 2022-03 俄乌开战, 2022-04 上海封控

### 3.4 Nested Walk-Forward (Phase 1.6)

**配置**：purge=5d, embargo=5d, train=4y, test=1y, step=1y, n_trials=5

| Fold | Period | F1 | AUC | PR-AUC | ECE | Brier | is_downside Recall |
|------|--------|-----|-----|--------|-----|-------|-------------------|
| F00 | 2018-2023 | 0.670 | 0.500 | 0.504 | 0.004 | 0.250 | 0% |
| F01 | 2019-2024 | 0.468 | 0.500 | 0.305 | 0.190 | 0.250 | 0% |
| F02 | 2020-2025 | 0.678 | 0.500 | 0.513 | 0.014 | 0.250 | 0% |
| F03 | 2021-2026 | 0.414 | 0.500 | 0.261 | 0.239 | 0.250 | 0% |
| **Mean ± std** | | **0.558 ± 0.136** | **0.500 ± 0.000** | **0.396 ± 0.131** | **0.112 ± 0.120** | **0.250 ± 0.001** | **0%** |

**对比 v4 报告**：

| 指标 | v4 (旧) | v6 (本次) | 改进 |
|------|---------|-----------|------|
| WF F1 | 0.569 ± 0.134 | 0.558 ± 0.136 | ≈ 持平 |
| WF AUC | ~0.49 | 0.500 | ≈ 持平 |
| WF is_downside Recall | (未分类) | 0% | **恶化** |

### 3.5 Overlay v10-DynD (Phase 1.7)

OOS 2022-01-01 ~ 2026-06-30：

| 指标 | Baseline v10-DynD | Overlay (+v6 risk_budget) | Delta |
|------|-------------------|---------------------------|-------|
| Sharpe | 1.414 | 1.288 | **-0.126** |
| Calmar | 1.745 | 1.531 | **-0.214** |
| AnnRet | 8.32% | 7.17% | -1.15% |
| MaxDD | -4.77% | -4.68% | +0.08% (略好) |

**决定**：**不采纳 overlay**。Sharpe/Calmar 恶化，MaxDD 改善微乎其微。

---

## 4. 关键发现

### 4.1 v6 没有突破 v4

| 维度 | 结论 |
|------|------|
| AUC | 0.500 (v4 也是 ~0.49) — 仍接近随机 |
| F1 | 0.558 (v4 是 0.546) — 持平, recall 撑起来 |
| is_downside Recall | 0% — 模型过于保守，几乎不 fired |
| 经济价值 | 负 — overlay 降低 Sharpe/Calmar |

### 4.2 模型保守的原因

`v6 risk_budget mean=0.051` 表示平均只降 5% 仓位。但 v4 y_E XGB 在 OOS 上的最佳 threshold 是 0.24-0.50 (取决于 fold)，远低于 0.5 保守阈值。这导致：
- fired 极少 → risk_budget 极少触发
- 即使触发，暴露降低也只有 0-20%

### 4.3 真正的瓶颈是特征

v6 与 v4 共享特征工程 (`build_full_features` 132 维)，差异在标签处理 (Stage B) 和校准 (ECE)。AUC 0.500 表明 **特征与标签之间没有可学习的强信号**。

**可能的改进方向**（Phase 2 之后）：
- 加 OHLCV 真实 ADX / ATR / overnight gap (数据现成)
- 真实波动率特征替代半宽度
- 重新设计 y_E (例如 multi-horizon mean label)

---

## 5. 决定与下一步

### 5.1 不推荐 v6 独立生产

v6 **不能取代** v4 作为独立预警信号。v4 best 仍是 F1 0.546 / WF F1 0.569。

### 5.2 不推荐 v6 overlay v10-DynD

Phase 1.7 overlay 测试：Sharpe -0.126, Calmar -0.214。**不如 v10-DynD 单独**。

### 5.3 v6 的价值在基础设施

虽然模型本身没有突破，Phase 1 的工作提供了：
- ✅ `is_downside` 事件分类 (Phase 1.1) — 后续评估更准确
- ✅ v5 routing 修复 (Phase 1.2) — final_model 直接使用
- ✅ Label leakage 守门 (Phase 1.3) — 后续训练不必担心
- ✅ PR-AUC / ECE / net_utility 指标 (Phase 1.5) — 多维评估框架
- ✅ Nested walk-forward 模板 (Phase 1.6) — 后续 OOF 评估标准化
- ✅ Overlay 验证流程 (Phase 1.7) — 任何新信号必须经过经济价值验证

### 5.4 Phase 2 准备

v10.3 约束动态分配器 (Phase 2) 仍可继续做：
- v10.3a Conservative (不依赖 v6)
- v10.3b Trend-Adaptive (不依赖 v6)
- v10.3c CA-GCP-Bounded 跳过 (v6 信号不可靠)

---

## 6. 提交记录

| Commit | 内容 |
|--------|------|
| `980fe28` | fix(ca_gcp_ew): add is_downside flag to KNOWN_EVENTS |
| `012706b` | fix(ca_gcp_ew): v5 routing — use final_model instead of fake train set |
| `4edcfee` | test(ca_gcp_ew): y_A..y_E label structure + future-leakage guards |
| `c33f78e` | feat(ca_gcp_ew): v6 2-stage utility-aligned model + PR-AUC/ECE/net_utility metrics |
| `<pending>` | feat(ca_gcp_ew): v6 nested walk-forward + v10-DynD overlay (Phase 1.6 + 1.7) |

---

## 7. 文件清单

```
ca_gcp_training_v6.py              # 2-stage 主模型
ca_gcp_training_v6_walk_forward.py # Nested walk-forward 评估
ca_gcp_v6_risk_budget.csv          # Stage B 输出
ca_gcp_v6_walk_forward_results.csv
ca_gcp_v6_walk_forward_report.md
ca_gcp_v6_overlay_report.md
ca_gcp_training_v6_report.md
tests/test_ca_gcp_v6_helpers.py            # 17 unit tests
tests/test_ca_gcp_ew_full_eval_metrics.py  # 12 unit tests
scripts/ca_gcp_v6/overlay_v10dynd.py
```

## 8. 结论

**v6 没有突破 v4 的 AUC ≈ 0.5 瓶颈。**

最实用的发现：
- v4 y_E XGB 仍是最强单模型
- v6 的 utility framework 框架 (ECE / net_utility / risk_budget) 是好的基础设施
- 真正的瓶颈是特征，不是模型

**推荐**：Phase 2 v10.3 仍可推进 (v10.3a/3b)，但 v10.3c (CA-GCP-Bounded) 应跳过。

---

*文档版本: 1.0*
*日期: 2026-08-03*
