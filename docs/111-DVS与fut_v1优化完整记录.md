# 111-动态波动率策略DVS与fut_v1优化-完整记录

> **日期**: 2026-08-18
> **状态**: fut_v1优化完成 + DVS开发中

---

## 一、fut_v1策略优化（已完成）

### 1.1 前视偏差修复

**结论**：`shift(-1)`是正确的，无前视偏差。

**验证**：
- purge测试：清洗版 vs 当前版 Sharpe差异<1%（10品种）/ 9%（72品种，属合法信息利用）
- lag测试：Sharpe下降是动量衰减非前视
- shift(-2)测试：Sharpe虚高1.92（确认是假前视）

**代码位置**：`factor_modules.py:426,460`，已添加防御性注释。

---

### 1.2 持仓EMA平滑（核心改进）

**参数**：`position_ema_alpha=0.15`

**效果**：

| 指标 | 无EMA | alpha=0.15 |
|------|-------|-----------|
| Sharpe | 1.65 | **1.91** (+15.8%) |
| 最大回撤 | -10.4% | **-6.1%** (收窄41%) |
| 换手率 | 86x | **19x** (降78%) |

---

### 1.3 L3成交量冲击滑点模型

**模型**：`slippage = daily_vol × sqrt(notional / ADV) × scale`

**文件**：`cta_slippage.py`，集成到`cta_broker.py`和`cta_engine.py`

**推荐参数**：`slippage_bps=0.5`（scale=0.5）

---

### 1.4 逐步换月

**方案**：换月成本在`roll_days=5`天内分摊。对Sharpe影响<0.01。

---

### 1.5 降换手

**机制**：`min_position_change=0.30`（仓位变化<30%不调仓）

---

### 1.6 新增因子

| 因子 | 文件 | 独立效果 | 框架内效果 |
|------|------|---------|-----------|
| CalendarSpreadMR | term_structure.py | Sharpe 0.21 | +0.11 Sharpe |
| BasisTrendComposite | basis_momentum.py | 崩溃(-154%) | 未接入 |
| BasisConvergence | term_structure.py | 崩溃(-157%) | 微弱贡献 |
| SkewReversal | vol_surface.py | — | 见VolSurface模块 |
| IVRegime | vol_surface.py | — | 见VolSurface模块 |
| IVZScore | vol_surface.py | — | 见VolSurface模块 |

---

## 二、VolSurface波动率曲面模块（新增）

### 2.1 模块概述

**文件**：`factors/vol_surface.py` + `VolSurfaceModuleStrategy`（在factor_modules.py中）

**数据来源**：`greeks_cache`（option_greeks.parquet）— 10个商品品种IV数据

**三个因子**：

| 因子 | 逻辑 | 信号 |
|------|------|------|
| SkewReversal | skew极端→标的反转 | +1/-1 |
| IVRegime | IV regime→仓位乘数 | 0.5-1.5 |
| IVZScore | IV z-score极端→均值回归 | +1/-1 |

### 2.2 接入方式

- 作为第8个模块加入`FactorModuleComboStrategy`
- `ComboConfig.module_weights`升级为8元组（1/8等权）
- 自定义`_signal_panel()`加载greeks cache中的IV数据

### 2.3 效果对比

| 版本 | Sharpe | 年化收益 | 最大回撤 | Calmar |
|------|--------|---------|---------|--------|
| 10品种 (无VolSurface) | 2.47 | 21.4% | -7.9% | 2.71 |
| 10品种 (+VolSurface) | 2.34 | 20.2% | -7.5% | 2.68 |
| 72品种 (无VolSurface) | 1.53 | 13.1% | -9.0% | 1.46 |
| **72品种 (+VolSurface)** | **1.93** | **18.7%** | **-7.4%** | **2.54** |
| 组合60/40 (无VolSurface) | 2.60 | 18.5% | -4.7% | 3.92 |
| 组合60/40 (+VolSurface) | 2.57 | 19.6% | -4.9% | 3.97 |

**关键发现**：
- VolSurface对72品种有**巨大提升**（Sharpe +26%）
- 对10品种轻微负面（IV数据仅覆盖3/7.7年）
- 组合效果基本持平

---

## 三、DVS动态波动率策略（开发中）

### 3.1 策略概述

**核心理念**："平时收租，风暴前买保险"

**双引擎**：Theta引擎（卖put spread） + Gamma引擎（买跨式）

### 3.2 系统架构（6层 + 8个模块）

| 层 | 功能 | 文件 |
|---|------|------|
| 1. 市场状态识别 | 5种状态 + IV Crush预警 + 冷静期 | regime_detector.py |
| 2. 品种评分器 | 动态品种选择 | product_scorer.py |
| 3. 曲面套利选股 | detect_outliers找便宜合约 | surface_selector.py |
| 4. 策略路由器 | 状态→策略映射 | strategy_router.py |
| 5. 仓位管理 | 动态止损 + 预算控制 | risk_manager.py |
| 6. 希腊字母对冲 | Delta对冲 | greeks_hedger.py |

### 3.3 市场状态（5种）

| 状态 | 条件 | 操作 |
|------|------|------|
| CALM | IV百分位<30% | 卖put spread |
| TRANSITION | IV百分位30-50% | 卖put+买call翼 |
| HIGH_VOL | IV百分位50-75% | 不交易 |
| VOL_SPIKE | IV动量>10%+期限倒挂 | 买跨式 |
| COMPRESSION | IV百分位>75%但回落 | 卖宽跨 |

### 3.4 回测结果（待优化）

| 指标 | 当前值 | 目标 |
|------|--------|------|
| Sharpe | -0.38 | > 1.6 |
| 年化收益 | -24.6% | > 15% |
| 最大回撤 | -89.9% | < -8% |

**问题**：仓位管理过于激进，止损未触发。需要进一步优化。

---

## 八、GV+fut_v1 组合策略

### 8.1 组件业绩

| 策略 | Sharpe | 年化收益 | 最大回撤 | Calmar |
|------|--------|---------|---------|--------|
| fut_v1 (60/40组合) | 2.17 | 13.9% | -6.9% | 2.00 |
| GV standalone (delta=0.30) | 1.62 | 29.1% | -12.7% | 1.28 |

### 8.2 组合效果

| GV权重 | Sharpe | 年化收益 | 最大回撤 | Calmar |
|--------|--------|---------|---------|--------|
| 0% | 2.17 | 13.9% | -6.9% | 2.00 |
| **10%** | **2.40** | **16.1%** | **-7.2%** | **2.24** |
| 15% | 2.38 | 17.1% | -7.3% | 2.34 |
| 20% | 2.32 | 18.1% | -7.5% | 2.42 |
| 25% | 2.25 | 19.0% | -7.7% | 2.48 |
| 30% | 2.18 | 19.9% | -7.9% | 2.53 |

### 8.3 最优配置

**10% GV + 90% fut_v1**：Sharpe 2.40, AnnRet 16.1%, MaxDD -7.2%

对比fut_v1 alone：Sharpe +11%, 收益+16%, 回撤仅+0.3%。

---

## 四、最终推荐配置

```python
ComboConfig(
    initial_cash=5_000_000,
    target_vol=0.30,
    module_weighting='inv_vol',
    aggregation='tvpr',
    commission_rate=0.0002,  # 2bps

    min_position_change=0.30,
    position_ema_alpha=0.15,

    slippage_bps=0.5,
    roll_cost_bps=2.0,
    roll_days=5,
)
# 组合: 60% 10品种 + 40% 72品种
```

### 最终业绩（含成本）

| 指标 | 值 |
|------|-----|
| **Sharpe** | **2.57** |
| 年化收益 | 19.6% |
| **最大回撤** | **-4.9%** |
| **Calmar** | **3.97** |
| Sortino | 3.71 |
| 换手率 | ~19x |
| 年化交易成本 | ~0.19% |

---

## 五、板块间尾部风险保护（新增）

### 5.1 原理

当各板块（黑色、贵金属、有色、化工、农产品）之间的相关性飙升时，说明市场进入系统性风险状态，全局限仓。

### 5.2 实现

- 计算5个板块的等权平均收益
- 滚动60天计算板块间两两相关性，取均值
- 平均板块相关性 > 阈值时，全局仓位乘以`tail_risk_scale`（默认0.7=降仓30%）

### 5.3 自适应阈值

根据板块内平均品种数调整阈值：
- 平均≤1.5品种/板块：阈值=0.50（极噪声）
- 平均≤2.5品种/板块：阈值=0.40（噪声）
- 平均>2.5品种/板块：阈值=0.30（正常）

### 5.4 效果

| 版本 | Sharpe | 最大回撤 | Calmar |
|------|--------|---------|--------|
| 72品种 (无tail_risk) | 1.93 | -6.1% | 2.54 |
| **72品种 (tail_risk=0.30)** | **1.98** | **-6.1%** | **2.13** |
| 10品种 (无tail_risk) | 2.34 | -7.5% | 2.68 |
| 10品种 (tail_risk=0.40) | 1.86 | -8.5% | 1.59 |

**结论**：72品种有微弱正面效果（Sharpe +2.6%），10品种不适用（板块噪声太大）。

---

## 六、信号一致性衰减（已验证，不启用）

### 6.1 原理

Cluster内（如黑色系RB+I）信号方向冲突时，衰减或取消信号。

### 6.2 测试结果

| 版本 | 无filter | cluster cancel | 变化 |
|------|---------|---------------|------|
| 72品种 | 1.98 | 1.50 | -24% ❌ |
| 10品种 | 2.34 | 1.84 | -21% ❌ |

### 6.3 结论

**信号冲突有alpha**。不同模块在不同市场状态下各有优势（趋势模块在趋势市赚钱，均值回归在震荡市赚钱），冲突信号提供了分散化收益。取消冲突信号=取消分散化。

**不启用 cluster_filter**。

---

## 七、代码变更清单

| Commit | 内容 |
|--------|------|
| `ab74b96` | 信号一致性极端冲突置零 |
| `e880bd4` | 板块间尾部风险保护+cluster filter |
| `a74d59b` | VolSurface模块接入7模块框架 |
| `a7435ea` | DVS与fut_v1优化完整记录 |
| `2ad8489` | 修复numpy导入 |
| `4e56b49` | 跨期价差均值回归+基差趋势复合+波动率曲面因子 |
| `d6e7675` | 基差收敛因子 |
| `9ff1d21` | 持仓EMA平滑（alpha=0.15） |
| `ae81bda` | 逐步换月 |
| `b40e3dd` | L3成交量冲击滑点模型 |
| `d992412` | 降换手+模块级regression/tvpr可选 |
| `99cae5a` | 修复purge测试bug |
| `18897f0` | shift(-1)无前视注释+验证脚本 |
| `bd4858a` | 撤回错误的shift(-2)修改 |
| `c3eb1c6` | fut_v1回测脚本 |
| `9de3ff8` | TVPR前视偏差修复（已撤回） |
| `58d8212` | DVS动态波动率策略初版 |

---

## 六、文件结构

```
QuantNodes/
├── strategy/cta/
│   ├── backtest/
│   │   ├── cta_slippage.py          # L3滑点模型 [NEW]
│   │   ├── cta_broker.py            # 支持slippage [MODIFIED]
│   │   └── cta_engine.py            # 传递slippage [MODIFIED]
│   ├── factors/
│   │   ├── term_structure.py        # +CalendarSpreadMR+BasisConvergence [MODIFIED]
│   │   ├── basis_momentum.py        # +BasisTrendComposite [MODIFIED]
│   │   └── vol_surface.py           # SkewReversal+IVRegime+IVZScore [NEW]
│   ├── options/
│   │   ├── dynamic_vol/             # DVS完整策略 [NEW]
│   │   │   ├── __init__.py
│   │   │   ├── regime_detector.py   # 5种状态+IV Crush+冷静期
│   │   │   ├── product_scorer.py    # 动态品种选择
│   │   │   ├── surface_selector.py  # 曲面套利选股
│   │   │   ├── strategy_router.py   # 状态→策略映射
│   │   │   ├── theta_engine.py      # 卖方引擎
│   │   │   ├── gamma_engine.py      # 买方引擎
│   │   │   ├── risk_manager.py      # 仓位管理
│   │   │   └── greeks_hedger.py     # Delta对冲
│   │   └── dynamic_vol_strategy.py  # DVS主策略类 [NEW]
│   └── strategies/
│       └── factor_modules.py        # EMA+滑点+换月+VolSurface+tail_risk [MODIFIED]
├── scripts/
│   ├── fut_v1_baseline_bt.py        # 72品种基线回测 [NEW]
│   ├── fut_v1_purge_test.py         # 前视偏差purge测试 [NEW]
│   ├── fut_v1_lag_validation.py     # lag敏感性验证 [NEW]
│   ├── fut_v1_cost_analysis.py      # 成本分析 [NEW]
│   ├── fut_v1_yearly_analysis.py    # 分年份分析 [NEW]
│   └── dynamic_vol_bt.py           # DVS回测脚本 [NEW]
├── docs/
│   ├── 109-DVS设计文档.md
│   ├── 110-DVS研发进度记录.md
│   └── 111-本文档
└── output/
    ├── fut_v1_72p_nav.parquet
    ├── fut_v1_72p_fixed_nav.parquet
    └── fut_v1_equity_curve.png
```

---

## 七、待办事项

| 优先级 | 任务 | 状态 |
|--------|------|------|
| P0 | DVS仓位管理优化（止损+激进仓位问题） | 待开发 |
| P0 | DVS回测脚本完善（价格面板构建） | 待开发 |
| P1 | VolSurface IV数据扩展（更多品种+更长历史） | 待数据更新 |
| P1 | 动态组合权重（基于近期表现调整10/72权重） | 待开发 |
| P2 | 参数网格搜索（EMA alpha, min_change等） | 待开发 |
| P2 | BasisTrendComposite接入框架 | 待评估 |
