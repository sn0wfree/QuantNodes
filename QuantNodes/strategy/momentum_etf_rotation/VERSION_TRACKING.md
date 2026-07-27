# 版本跟踪：momentum_etf_rotation v1-v10 + v11 规划

> **用途**：快速定位"每个版本做了什么"、"性能如何"、"当前状态"
> **维护**：新增版本时更新此文档
> **路径**：`QuantNodes/strategy/momentum_etf_rotation/VERSION_TRACKING.md`
> **最后更新**: 2026-07-27 — 全面审计修复后

---

## 一、版本总览（一页速查）

> ⚠️ 2026-07-27 全面审计修复：v0 lookback 修正、v10 resample('D') 删除、v4 年化因子修正。
> 以下指标均为修复后统一口径（sqrt(252) + 日历日年化）重新计算。

| 版本 | Stage | 核心特点 | 全期 Sharpe | OOS Sharpe | OOS Calmar | 状态 |
|------|-------|----------|------------|------------|------------|------|
| v0.0 | 8 | baseline (144d→90d) | 1.223 | 1.366 | 1.523 | ⚠️ 基线 |
| v0.1 | 9-C | +VT | 1.055 | 1.336 | 1.296 | ⚠️ 基线 |
| v0.2 | 9-B | +TF | 1.090 | 1.283 | 1.460 | ⚠️ 基线 |
| v1.0 | 12A | hybrid + VT + cost | 1.122 | 1.459 | 1.791 | ✅ 生产 |
| v3 | 16A | 多策略组合 | 0.781 | 1.035 | 0.778 | ❌ 失败 |
| v4 | 17 | 风格轮动 + 因子择时 | 0.248 | 0.259 | 0.131 | ⚠️ 实验 |
| v5 | 22 | 量价因子 (等权) | 0.839 | 0.513 | 0.488 | ✅ 生产 |
| v5.1 | 22+ | 量价因子 (逆波动) | 0.948 | 0.623 | 0.604 | ✅ 生产 |
| v7.10 | 30 | TV-PR 时变 LASSO | 0.903 | 1.017 | 1.353 | ✅ 生产 |
| v9 | 32 | 银河-动态仓位 | 0.834 | 0.924 | 0.825 | ⚠️ 实验 |
| v10 | 33 | 4 策略 Vol-parity | 1.216 | 1.434 | 1.426 | ✅ 生产 |
| v10-DynD | 33+ | 信号加权动态 | 1.350 | 1.464 | 1.806 | ✅ 生产 |

---

## 二、版本详细说明

### v1 — CICC 原始复现 (Stage 8)

**目录**: `v1/`
**核心文件**: `momentum_v1.py`, `portfolio_v1.py`, `backtest_v1.py`
**特性**:
- 4 步组合管理 (去重 + 剔高相关 + 强制分散 + 逆波动加权 + 止损补位)
- momentum_type = "price" (固定)
- 无 VolTargeting, CostModel

**性能**:
- Calmar ~0.78 (vs CICC 0.76)
- DD -21%

**状态**: ✅ 生产 (baseline)

**教训**: 无增强功能的纯动量 baseline

---

### v2 — hybrid 动量 + VT + cost (Stage 12A)

**目录**: `v2/`
**核心文件**: `momentum_v2.py`, `portfolio_v2.py`, `backtest_v2.py`
**特性**:
- momentum_type: "price" | "slope_r2" | "hybrid" (默认 "hybrid")
- VolTargeting: 波动率目标 (TV tv=0.15)
- CostModel: 交易成本 (5bp+10bp)
- CovEstimator: 协方差估计方法选择

**性能**:
- Calmar 1.60
- DD -3.93%
- OOS Calmar 1.79

**状态**: ✅ 生产 (v1.0 locked)

**教训**: hybrid 动量 + VT + cost 是最有效的组合

---

### v3 — 多策略组合 (Stage 16A)

**目录**: `v3/`
**核心文件**: `sub_strategy_v3.py`, `multi_strategy_v3.py`
**特性**:
- SubStrategy 抽象基类 (v4-v7 全部继承)
- 多策略组合: 动量 + 均值反转 + 行业轮动
- 子策略权重: 风险平价 / 等权

**性能**:
- Calmar 0.504 < v2 0.892
- 51 单测 100% 通过

**状态**: ❌ 失败

**教训**: "架构先进 ≠ 业绩进步" — 1/N 等权失败

---

### v4 — 风格轮动 + 因子择时 (Stage 17)

**目录**: `v4/`
**核心文件**: `style_rotation_v4.py`, `smart_beta_v4.py`, `factor_timing_v4.py`
**特性**:
- 6 因子 IC 诊断
- 风格轮动 (120d Top-1 Calmar 0.919)
- Smart β (低 beta 工具, β=0.60)
- HMM 距离先验 (3 状态)

**性能**:
- 风格轮动 Calmar 0.919 (4x vs L60_T3)
- 仅 value 稳定正 IC (mean +0.044, hit 60%)

**状态**: ⚠️ 实验

**教训**: 6 因子中仅 value 稳定正 IC

---

### v5 — 量价因子 (等权) (Stage 22)

**目录**: `v5/`
**核心文件**: `industry_factors.py`, `industry_rotation_v5.py`
**特性**:
- 6 大类 11 月频因子 (动量/交易波动/换手率/多空对比/量价背离/量幅同向)
- 复合因子 = z-score 等权
- 月末选 Top-N ETF 等权

**性能**:
- Calmar 0.745
- OOS Calmar 0.488
- DD -19.41%

**状态**: ✅ 生产

**教训**: 论文做法 (等权) 可复现

---

### v5_1 — 量价因子 (逆波动加权) (Stage 22+)

**目录**: `v5_1/`
**核心文件**: `industry_rotation_v5_1.py`
**特性**:
- v5 选股 + 逆波动率加权
- 60 日窗口, vol_floor=0.01, T+1 lag

**性能**:
- Calmar 0.774
- OOS Calmar 0.589 (+20.7% vs v5)
- DD -18.59%

**状态**: ✅ 生产

**教训**: 逆波动率加权显著提升 OOS

---

### v6 — v1.0 风控 + v5 选股 (Stage 26)

**目录**: `v6/`
**核心文件**: `industry_rotation_v6.py`, `strategy_v6.py`
**特性**:
- v5.1.1 选股 + v5.1.1 加权
- v1.0 风控框架 (VT + TF + Cost)

**性能**:
- OOS Calmar 0.662
- DD ≤ -10%

**状态**: ✅ 生产

**教训**: 风控框架可有效降低 DD

---

### v6_1 — IC-IR 加权 (Stage 27)

**目录**: `v6_1/`
**核心文件**: `factor_weighting.py`, `industry_rotation_v6_1.py`
**特性**:
- IC-IR expanding 加权 (12/24/36 月 + 6 月平滑)
- 11 因子中仅 5-6 个 OOS IR > 0

**性能**:
- OOS Calmar 0.748 (+13% vs v6)

**状态**: ⚠️ 实验

**教训**: IC-IR 加权有进步，但因子稳定性不足

---

### v6_2 — 正交化 (Stage 29)

**目录**: `v6_2/`
**核心文件**: `factor_orthogonal.py`, `industry_rotation_v6_2.py`
**特性**:
- Gram-Schmidt 正交化
- 5 种 sort_method

**性能**:
- OOS Calmar 0.901
- CV% 56.9% FAIL
- 扣成本后 Calmar 0.331

**状态**: ❌ DEPRECATED

**教训**: "PROMISING" 必须经过 CV% 测试才能真正生产

---

### v7 — TV-PR 时变 LASSO (Stage 30)

**目录**: `v7/`
**核心文件**: `tvpr_estimator.py`, `macro_substrategy_v7_6.py`, `data_loader_v7_6.py`
**特性**:
- 17 macro + 19 量价因子
- 时变 β_t (Cui 2025)
- 混合标准化 + 两阶段 CV
- expanding-window 真实 OOS

**性能**:
- OOS Calmar 2.144
- OOS Sharpe 1.60
- DD -14.3%
- CV% 16.6% PASS

**状态**: ✅ 生产

**教训**: TV-PR + 标准化 + CV 是最有效的因子择时方法

---

### v8 — Jump Model + ML (Stage 31)

**目录**: `v8/`
**核心文件**: `jump_model.py`, `signal_composer.py`
**特性**:
- Jump Model (regime-aware)
- ML 因子交互 (LightGBM)
- 多估计器集成

**性能**:
- Sharpe 1.485

**状态**: ⚠️ 实验

**教训**: ML 做因子交互/状态检测/集成，而非直接预测收益

---

### v9 — 银河/中信因子 (Stage 32)

**目录**: `v9/`
**核心文件**: `factor_galaxy.py`, `citic_macro.py`, `citic_multifactor.py`, `citic_rotation.py`
**特性**:
- 银河因子体系 (熵权法)
- 中信宏观因子 (5 因子)
- 中信多因子 (5 风格因子)
- 中信行业轮动

**性能**:
- 未完整测试

**状态**: ⚠️ 实验

**教训**: 银河/中信因子体系可作为 v10 的输入

---

### v10 — 4 策略 Vol-parity (Stage 33)

**目录**: `v10/`
**核心文件**: `dynamic_weight_schemes.py`, `dual_momentum.py`

**4 策略组合**: v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%
**5 个动态权重方案**: A 市场状态 / B 波动率 / C 回撤控制 / D 信号加权 / E 混合

**修复后性能** (2026-07-27 审计修复后统一口径):

| 方案 | 全期 Sharpe | OOS Sharpe | OOS Calmar | OOS AnnRet | OOS MaxDD |
|------|-----------|-----------|-----------|-----------|----------|
| Static (Vol-parity) | 1.216 | 1.434 | 1.426 | +10.27% | -7.20% |
| A_regime | 1.312 | 1.388 | 1.370 | +11.52% | -8.41% |
| B_vol_target | 1.186 | 1.264 | 1.162 | +10.34% | -8.90% |
| C_drawdown | 1.425 | 1.522 | 2.018 | +8.07% | -3.99% |
| **D_signal_weighted** | **1.350** | **1.464** | **1.806** | **+8.61%** | **-4.77%** |
| E_hybrid | 1.341 | 1.398 | 1.655 | +11.11% | -6.71% |

**独立策略**:

| 策略 | 全期 AnnRet | 全期 Sharpe | OOS AnnRet | OOS Sharpe |
|------|-----------|-----------|-----------|-----------|
| DualMom (4 资产) | +15.07% | 0.711 | +17.77% | 0.794 |
| v10:Vol-parity | +9.90% | 1.216 | +10.27% | 1.434 |
| v10:DynD | +9.05% | 1.350 | +8.61% | 1.464 |

**状态**: ✅ 生产 (最终版本)

**2026-07-27 审计修复**:
- 删除 `load_navs()` 中 `resample('D').ffill()`，消除零收益稀释（Sharpe 虚增 ~23%）
- 修复后 Sharpe 从 1.930 降至 1.216（静态），真实反映业绩

**教训**: 多策略 Vol-parity 组合是最优方案；动态权重方案 D（信号加权）Calmar 最高

---

### v11 — 5 层架构 + ACT-1/2/3 (海龟数学升级)

**目录**: `v11/`
**核心文件**: `v11_strategy.py`, `config_v11.py`, `backtest_v11.py`
**特性**:
- 5 层架构 (从 v10/ 复制)
- ACT-1: yang_zhang_vol 波动率估计器
- ACT-2: kelly_audit 审计
- ACT-3: drawdown_controller 回撤控制
- OOS Sharpe 1.131 (全周期)

**状态**: ✅ 生产

---

## 三、版本演进路线图

```
v1 (CICC baseline)
  └→ v2 (hybrid + VT + cost) ← v1.0 locked
       └→ v3 (多策略) ← ❌ 失败
       └→ v4 (风格轮动 + 因子择时) ← ⚠️ 实验
       └→ v5 (量价因子) ← ✅ 生产
            └→ v5_1 (逆波动加权) ← ✅ 生产
                 └→ v6 (v1.0 风控 + v5 选股) ← ✅ 生产
                      └→ v6_1 (IC-IR 加权) ← ⚠️ 实验
                      └→ v6_2 (正交化) ← ❌ DEPRECATED
       └→ v7 (TV-PR) ← ✅ 生产
            └→ v8 (Jump Model + ML) ← ⚠️ 实验
       └→ v9 (银河/中信因子) ← ⚠️ 实验
       └→ v10 (4 策略 Vol-parity) ← ✅ 生产 (最终)
            └→ v11 (5 层架构 + ACT-1/2/3) ← ✅ 生产
```

---

## 四、v11 实现 (海龟数学升级)

**基于**: v10 5 层架构 + 10_TURTLE_TRADING_MATHEMATICS.md
**升级点**:
- ACT-1: yang_zhang_vol 替换 realized_vol
- ACT-2: kelly_audit 审计
- ACT-3: drawdown_controller 回撤控制

**预期性能**:
- Calmar ~1.8-2.0
- MaxDD ~-3.0%
- Sharpe ~1.3-1.4

**状态**: 📋 规划中

**目录**: `v11/` (待创建)
**核心文件**: `config_v11.py`, `v11_strategy.py`, `risk_layer_v11.py`

---

## 五、版本状态说明

| 状态 | 含义 | 使用建议 |
|------|------|----------|
| ✅ 生产 | 可用于实盘 | 直接使用 |
| ⚠️ 实验 | 需进一步验证 | 仅用于研究 |
| ❌ 失败 | 性能不达标 | 不使用 |
| ❌ DEPRECATED | 已废弃 | 不使用 |
| 📋 规划中 | 正在设计 | 等待实现 |

---

## 六、版本依赖关系

```
common/strategy_engine.py ← v1-v10 所有策略
common/walk_forward.py ← v7-v10 OOS 验证
common/extended_metrics.py ← 所有版本评估
common/backtest_engine.py ← v3+ 回测引擎
common/config_runner.py ← 所有版本配置驱动

v3/sub_strategy_v3.py ← v4-v7 SubStrategy 基类
v2/portfolio_v2.py ← v6-v7 风控框架
v7/tvpr_estimator.py ← v7.6-v7.10 TV-PR 核心
v8/jump_model.py ← v10 Jump Model
v9/factor_galaxy.py ← v10 因子输入
v10/portfolio_layer.py ← v10 Vol-parity 组合
```

---

---

## 七、2026-07-27 全面审计修复记录

### 修复内容

| # | 文件 | 修改 | 严重度 |
|---|------|------|--------|
| 1 | `v10/dynamic_weight_schemes.py:45-48` | 删除 `resample('D').ffill()` | CRITICAL |
| 2 | `v4/multi_strategy_v4.py:164-184` | `_performance_metrics()` 自动检测频率 | CRITICAL |
| 3 | `scripts/v4/v4_full_backtest.py:117` | `freq="W"` → `freq="D"` | CRITICAL |
| 4 | `combo/unified_v1v5_compare.py:154,171,189` | v0 lookback=144 → lookback=90 | HIGH |
| 5 | `scripts/v9/v9_factor_galaxy.py:139` | 添加 `freq='D'` 参数 | HIGH |

### 重新生成的文件

- `reports/.../v10/dynamic_nav_*.parquet` (×6)
- `reports/.../v10/dynamic_schemes_comparison.csv`
- `reports/.../combo/unified_v1v5_navs_calA.parquet`
- `reports/.../combo/combo_navs_unified52.parquet`
- `STRATEGY_ITERATION_RECORD.html`

### 相关文档

- `docs/77-v0_v10_codebase_audit.md` — 审计报告
- `docs/78-refactoring-plan.md` — 重构方案

---

*文档版本: 2.0*
*最后更新: 2026-07-27*
*维护: 新增版本时更新此文档*
