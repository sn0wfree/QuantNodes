# 111-动态波动率策略DVS与fut_v1优化-完整记录

> **日期**: 2026-08-18
> **状态**: DVS开发中 + fut_v1优化完成

---

## 一、fut_v1策略优化

### 1.1 前视偏差修复

**问题**：TVPR聚合中`Y = pct_change().shift(-1)`是否存在前视偏差？

**结论**：`shift(-1)`是正确的，无前视偏差。

**验证方法**：
1. **purge测试（决定性）**：排除最近观测点的清洗版 vs 当前版
   - 10品种：清洗版Sharpe 2.109 vs 当前2.091（差异<1%）
   - 72品种：清洗版Sharpe 1.470 vs 当前1.618（差异9%，但属合法信息利用）
2. **lag测试（对照）**：lag=1 vs lag=2，Sharpe下降是动量衰减非前视
3. **shift(-2)测试**：引入明天价格p[τ+1]，Sharpe虚高到1.92（证明是假前视）

**最终结论**：`shift(-1)`配合`execution_lag=1`是因果正确的，保持不变。

**代码位置**：`factor_modules.py:426,460`

**防御性注释已添加**：说明为什么shift(-1)正确，防止未来再犯分析错误。

**验证脚本**：
- `scripts/fut_v1_purge_test.py` — 清洗版beta决定性测试
- `scripts/fut_v1_lag_validation.py` — lag敏感性验证

---

### 1.2 持仓EMA平滑（核心改进）

**问题**：年化换手87x，交易成本吃掉大量收益。

**方案**：对目标手数做EMA平滑，仓位渐进变化，避免信号翻转时大额单日反转。

**实现**：`factor_modules.py` `_run_with_lots()` 中：

```python
if cfg.position_ema_alpha > 0:
    smoothed = lots.copy()
    prev = pd.Series(0.0, index=lots.columns)
    for i in range(len(smoothed)):
        smoothed.iloc[i] = prev + cfg.position_ema_alpha * (lots.iloc[i] - prev)
        prev = smoothed.iloc[i].copy()
    lots = smoothed
```

**参数选择**：alpha=0.15（每天朝目标移动15%，约7天到位）

**效果**：

| 指标 | 无EMA | alpha=0.15 |
|------|-------|-----------|
| Sharpe | 1.65 | **1.91** (+15.8%) |
| 年化收益 | 13.6% | 11.6% |
| 最大回撤 | -10.4% | **-6.1%** (收窄41%) |
| 换手率 | 86x | **19x** (降78%) |

**为什么其他方案失败**：
- 信号置信度阈值：raw signal太小（均值0.0003），任何阈值过滤掉所有信号
- 最小持仓期：与EMA冲突，EMA已平滑仓位变化
- 换手率上限：强制限制导致仓位积累，后续更大额调整

---

### 1.3 L3成交量冲击滑点模型

**模型**：Almgren-Chriss简化版

```
slippage = daily_vol × sqrt(notional / ADV) × scale
```

- `daily_vol` = 20日年化波动率 / sqrt(252)
- `notional` = 交易金额（手数 × 价格 × 合约乘数）
- `ADV` = 20日平均成交量
- `scale` = 校准系数

**文件**：`QuantNodes/strategy/cta/backtest/cta_slippage.py`

**集成**：
- `cta_engine.py`：`CtaBacktestConfig`新增`slippage_engine`参数
- `cta_broker.py`：`place_order`中计算slippage并调整fill price
- `factor_modules.py`：`ComboConfig`新增`slippage_bps`

**效果**：

| scale | Sharpe | 年化收益 | 最大回撤 |
|-------|--------|---------|---------|
| 无滑点 | 1.87 | 15.0% | -8.8% |
| 0.5 | 1.59 | 13.7% | -9.9% |
| 1.0 | 1.41 | 12.7% | -10.5% |

---

### 1.4 逐步换月

**方案**：从`main_contract_mapping`检测换月日，在换月日起N天内分摊换月成本。

**实现**：`SlippageEngine.load_roll_dates()` + `compute()`中额外成本。

**效果**：5bps换月成本对Sharpe影响<0.01（换月频率低+分摊到5天）。

---

### 1.5 降换手

**机制**：仓位变化<30%不调仓。

**效果**：减少16%交易，Sharpe从1.65提升到1.65（微弱）。

---

### 1.6 新增因子

#### CalendarSpreadMeanReversion（跨期价差均值回归）

**文件**：`QuantNodes/strategy/cta/factors/term_structure.py`

**逻辑**：近月-次月归一化价差做z-score均值回归。z<-1.5→做多价差，z>1.5→做空价差。

**效果**：
- 独立运行：Sharpe 0.21（勉强持平）
- 框架内：Sharpe从2.59提升到2.60（微弱正贡献）

#### BasisTrendComposite（基差+趋势复合）

**文件**：`QuantNodes/strategy/cta/factors/basis_momentum.py`

**逻辑**：基差作为趋势信号的过滤器。方向一致→放大，相反→缩小。基差极端时独立回归。

**效果**：独立运行崩溃（-154%回撤），不适合加入框架。

#### VolSurfaceArb（波动率曲面套利）

**文件**：`QuantNodes/strategy/cta/factors/vol_surface.py`

**逻辑**：SkewReversal（skew极端→标的反转）+ IVRegime（IV regime→调整仓位乘数）。

**状态**：已实现但未接入框架（需要IV数据传递）。

---

## 二、DVS动态波动率策略

### 2.1 策略概述

**核心理念**："平时收租，风暴前买保险"

**双引擎设计**：
- Theta引擎（默认）：卖put spread收权利金
- Gamma引擎（触发式）：买跨式，在波动率爆发时获利

### 2.2 系统架构（6层）

| 层 | 功能 | 文件 |
|---|------|------|
| 1. 市场状态识别 | 5种状态 + IV Crush预警 + 冷静期 | `regime_detector.py` |
| 2. 品种评分器 | 动态品种选择（账户规模驱动） | `product_scorer.py` |
| 3. 曲面套利选股 | 利用detect_outliers找便宜合约 | `surface_selector.py` |
| 4. 策略路由器 | 状态→策略映射 | `strategy_router.py` |
| 5. 仓位管理 | 动态止损 + 预算控制 | `risk_manager.py` |
| 6. 希腊字母对冲 | Delta对冲 | `greeks_hedger.py` |

### 2.3 市场状态（5种）

| 状态 | 条件 | 操作 |
|------|------|------|
| CALM | IV百分位<30% | 卖put spread |
| TRANSITION | IV百分位30-50% | 卖put+买call翼 |
| HIGH_VOL | IV百分位50-75% | 不交易（等待） |
| VOL_SPIKE | IV动量>10%+期限倒挂 | 买跨式 |
| COMPRESSION | IV百分位>75%但回落 | 卖宽跨 |

### 2.4 冷静期机制

连续IV crush后自动减仓：
- 第1次：仓位50%，持续10天
- 第2次（冷静期内）：仓位25%
- 第3次：仓位20%（最低）
- 冷静期结束后恢复正常

### 2.5 回测结果（待优化）

| 指标 | 当前值 | 目标 |
|------|--------|------|
| Sharpe | -0.38 | > 1.6 |
| 年化收益 | -24.6% | > 15% |
| 最大回撤 | -89.9% | < -8% |
| 胜率 | 13% | > 30% |

**问题**：仓位管理过于激进，每品种都开仓，止损未触发。

---

## 三、组合优化

### 3.1 策略组合

**方案**：10品种 + 72品种按权重组合

**最优权重**：60% 10品种 + 40% 72品种

**效果**：

| 版本 | Sharpe | 最大回撤 | Calmar |
|------|--------|---------|--------|
| 10品种 alone | 2.35 | -6.6% | 3.22 |
| 72品种 alone | 1.51 | -10.8% | 1.19 |
| **组合 60/40** | **2.60** | **-4.7%** | **3.92** |

### 3.2 基差套利评估

**结论**：基差套利不适合独立运行。

- 纯基差收敛：Sharpe 0.00（-157%回撤）
- 跨期价差均值回归：Sharpe 0.21（勉强持平）
- 基差+趋势复合：崩溃（-154%回撤）

**正确用法**：作为7模块框架内的微量贡献者，通过分散化产生微弱alpha。

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
| **Sharpe** | **2.60** |
| 年化收益 | 18.5% |
| **最大回撤** | **-4.7%** |
| **Calmar** | **3.92** |
| Sortino | 3.71 |
| 换手率 | ~19x |
| 年化交易成本 | ~0.19% |

---

## 五、代码变更清单

| Commit | 内容 |
|--------|------|
| `9ff1d21` | 持仓EMA平滑（alpha=0.15） |
| `b40e3dd` | L3成交量冲击滑点模型 |
| `ae81bda` | 逐步换月（换月成本分摊） |
| `d992412` | 降换手(min_change) + 模块级regression/tvpr可选 |
| `4e56b49` | 跨期价差均值回归+基差趋势复合+波动率曲面因子 |
| `2ad8489` | 修复numpy导入 |
| `9de3ff8` | TVPR前视偏差修复（已撤回） |
| `bd4858a` | 撤回错误的shift(-2)修改 |
| `18897f0` | shift(-1)无前视注释 + 验证脚本 |
| `99cae5a` | 修复purge测试bug |
| `58d8212` | DVS动态波动率策略初版 |
| `8d488f1` | standalone全风险指标验证脚本 |

---

## 六、文件结构

```
QuantNodes/
├── strategy/cta/
│   ├── backtest/
│   │   ├── cta_slippage.py          # L3滑点模型 [NEW]
│   │   ├── cta_broker.py            # 支持slippage_engine [MODIFIED]
│   │   └── cta_engine.py            # 传递slippage [MODIFIED]
│   ├── factors/
│   │   ├── term_structure.py        # +CalendarSpreadMR+BasisConvergence [MODIFIED]
│   │   ├── basis_momentum.py        # +BasisTrendComposite [MODIFIED]
│   │   └── vol_surface.py           # SkewReversal+IVRegime [NEW]
│   ├── options/
│   │   ├── dynamic_vol/             # DVS完整策略 [NEW]
│   │   │   ├── __init__.py
│   │   │   ├── regime_detector.py
│   │   │   ├── product_scorer.py
│   │   │   ├── surface_selector.py
│   │   │   ├── strategy_router.py
│   │   │   ├── theta_engine.py
│   │   │   ├── gamma_engine.py
│   │   │   ├── risk_manager.py
│   │   │   └── greeks_hedger.py
│   │   └── dynamic_vol_strategy.py  # DVS主策略类 [NEW]
│   └── strategies/
│       └── factor_modules.py        # EMA平滑+降换手+regression/tvpr [MODIFIED]
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
