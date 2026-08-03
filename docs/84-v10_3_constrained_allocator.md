# 84. v10.3 约束动态分配器实验报告

**作者**: opencode
**日期**: 2026-08-03
**分支**: `refactor/v10-to-v11-migration`
**状态**: Phase 2 完成 — G (Trend-Adaptive) 显著改善

---

## 1. 背景

### 1.1 v10 基线 (D scheme - signal-weighted)

| 指标 | 值 |
|------|-----|
| OOS Sharpe | 1.464 |
| OOS Calmar | 1.806 |
| MaxDD | -4.77% |
| AnnRet | 8.61% |

### 1.2 Phase 0 审计结果

`docs/77` + `docs/82`: D scheme 是当前 HTML OOS Sharpe 第二名 (1.464)，仅次于 v1.0 locked (1.529)。

---

## 2. v10.3 设计

### 2.1 共享约束 (F + G)

| 约束 | 值 | 含义 |
|------|-----|------|
| `V10_3F_MIN_DEFENSIVE` | 0.25 | v1.0 最低权重 (防御底仓) |
| `V10_3F_MAX_TURNOVER` | 0.30 | 单次调仓换手上限 |
| `V10_3F_HYSTERESIS` | 0.05 | 权重变化 < 5% 不调仓 |
| `V10_3F_DD_GUARD` | 0.05 | NAV drawdown > 5% 强制 v1.0 = 90% |
| `V10_3F_REBAL_FREQ` | 25 | 月频调仓 |

### 2.2 方案 F (Conservative)

在 D 基础上:
1. min(v1.0) = 25% (防御底仓)
2. max turnover = 30% / rebalance
3. hysteresis: drift < 5% 不调仓
4. DD guard: drawdown > 5% 强制 v1.0 90%

### 2.3 方案 G (Trend-Adaptive)

在 F 基础上加趋势判断:
- **趋势向上**: 5 日收益 > 2% → 用 D 方案 (无约束, 顺势)
- **趋势向下**: 5 日收益 ≤ 2% → 用 F 方案 (conservative)

### 2.4 方案 H (CA-GCP-Bounded) — 已跳过

`docs/83`: v6 AUC ≈ 0.5 信号太弱, overlay 降低 Sharpe -0.126。v10.3c 不推荐。

---

## 3. 结果

### 3.1 全期 OOS (2018-01-12 ~ 2026-05-29)

| Scheme | Sharpe | Calmar | AnnRet | MaxDD | 说明 |
|--------|--------|--------|--------|-------|------|
| D (baseline) | 1.464 | 1.806 | 8.61% | -4.77% | 当前最优 v10 |
| **F (Conservative)** | **1.526** | **1.848** | 7.61% | -4.12% | 更稳, 更低 MaxDD |
| **G (Trend-Adaptive)** | **1.642** | **2.214** | **8.80%** | **-3.97%** | **显著改善** |

### 3.2 G vs D 改善

| 指标 | Delta | 改善% |
|------|-------|-------|
| Sharpe | +0.178 | +12.2% |
| Calmar | +0.408 | +22.6% |
| AnnRet | +0.19% | +2.2% |
| MaxDD | +0.80% | 更低回撤 |

### 3.3 vs docs/77 审计

审计 D scheme OOS Sharpe = 1.464, Calmar = 1.806。
G scheme: Sharpe 1.642 (+0.178), Calmar 2.214 (+0.408)。

---

## 4. 关键发现

### 4.1 v10.3 路线图验证

docs/82 §5.5 接受标准:

- [✓] audited OOS Sharpe ≥ v10-DynD + 0.05 (1.642 > 1.464 + 0.05)
- [✓] audited OOS Calmar ≥ v10-DynD + 0.05 (2.214 > 1.806 + 0.05)
- [✓] MaxDD ≥ v10-DynD (-3.97% > -4.77%)
- [ ] 与 v1.0 / v7.10 / DualMom 相关性比 v10-DynD 更低 — 待验证
- [ ] Turnover 满足上限 — 待验证
- [ ] 2022 / 2024 反转 / 2025 段均不崩 — 待验证
- [ ] bootstrap p-value < 0.1 — 待验证

### 4.2 为什么 G 有效

核心原因 (来自项目历史):
1. **动态仓位贡献最大**: Brinson attribution 仓位效应 71%
2. **简单趋势控制更可靠**: v10.2 TrendFilter 是唯一改善 Calmar 的 overlay
3. **独立 sleeve + 动态权重 > 信号叠加**: v11 证明多层反而恶化

G 方案同时利用了:
- D scheme 的独立 sleeve 动态权重 (顺势时)
- F scheme 的防御底仓 + 换手控制 (逆势时)
- 趋势判断本身 (5 日 2% 阈值来自 v10.2 经验)

### 4.3 v10.3c 跳过是正确的

docs/83: v6 overlay 降低 Sharpe -0.126, Calmar -0.214。v6 信号 AUC ≈ 0.5 太弱, 不值得耦合。

---

## 5. 下一步

| 优先级 | 任务 | 状态 |
|--------|------|------|
| 1 | nested walk-forward (purge/embargo) | 待做 |
| 2 | turnover / cost 分析 | 待做 |
| 3 | 2022 / 2024 反转 / 2025 压力段验证 | 待做 |
| 4 | 与 v1.0 / v7.10 / DualMom 相关性分析 | 待做 |
| 5 | bootstrap Sharpe diff p-value | 待做 |

---

## 6. 文件清单

```
QuantNodes/strategy/momentum_etf_rotation/v10/dynamic_weight_schemes.py
  - _apply_constraints(): v10.3 共享约束 (min_defensive / max_turnover / hysteresis)
  - scheme_f_conservative(): Conservative (防御底仓 + 月频)
  - scheme_g_trend_adaptive(): Trend-Adaptive (顺势/逆势切换)
  - V10_3F_MIN_DEFENSIVE = 0.25
  - V10_3F_MAX_TURNOVER = 0.30
  - V10_3F_HYSTERESIS = 0.05
  - V10_3F_DD_GUARD = 0.05
  - V10_3G_TREND_WINDOW = 5
  - V10_3G_TREND_THRESH = 0.02

tests/strategy/momentum_etf_rotation/test_v10_3_schemes.py
  - TestApplyConstraints: 4 unit tests
  - TestSchemeFConservative: 4 unit tests
  - TestSchemeGTrendAdaptive: 3 unit tests
  (共 11 tests)
```

## 7. 结论

**G (Trend-Adaptive) 是当前最显著的收益改善。**

Sharpe 1.642 (vs D 1.464) +12%, Calmar 2.214 (vs 1.806) +23%, MaxDD -3.97% (vs -4.77%)。

这与项目历史经验一致: **简单趋势 + 独立 sleeve 动态权重 > 复杂信号叠加**。

**推荐**: 将 G 方案作为下一个生产候选, 需通过 nested walk-forward + turnover/压力验证。

---

*文档版本: 1.0*
*日期: 2026-08-03*
