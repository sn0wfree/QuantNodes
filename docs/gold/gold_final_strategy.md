# 黄金策略最终方案文档

> 最后更新: 2026-08-28
> 最终策略: `compute_layered_ensemble_signal` + CPPI杠杆 (L=1.5) + 死带平滑
> 核心收益: Walk-Forward CAGR 9.59%, MaxDD -11.48%, Sharpe 1.01, Calmar 0.84
> 换手优化: 死带th=0.05, 变化次数-47%, 换手-5.5%, 收益损失<0.05pp

---

## 一、策略概述

### 1.1 目标

用户原始需求: **"大级别止损，小级别波动容忍"**

核心目标: 在-15% MaxDD约束下最大化风险调整收益。

### 1.2 最终策略架构

```
四路分层集成 + Δtips信号 + CPPI杠杆 + 三级风控

信号层:
  macro_ov  (0.2125)  宏观机制策略 + 独立风控(hc=-0.10)
  rr_ov     (0.2125)  TIPS实际利率择时 + 独立风控
  mom_ov    (0.2125)  52周动量 + 独立风控
  bo_ov     (0.2125)  20日突破 + 独立风控
  delta_bin (0.15)    Δtips10y二值信号(window=10, rank≤0.5)

集成: base = 0.2125×(macro+rr+mom+bo) + 0.15×Δtips

杠杆: L=1.5, clip上限1.8

死带平滑: th=0.05 (忽略<5%的仓位变化, 降低换手47%)
  - 保留清仓/开仓信号, 仅忽略小幅调仓

风控: CPPI三层重铺
  Level 1: 波动率压制 (vol_cap=0.20)
  Level 2: 软减仓 (soft_cap=-0.12)
  Level 3: 硬止损 + 智能再入场 (hard_cap=-0.15, daily_cap=-0.03, cooldown=5)
```

---

## 二、核心性能指标

### 2.1 Walk-Forward验证 (2012-2026, 含5bps成本)

| 指标 | L=1.5 CPPI | 基线(L=1) | Buy&Hold |
|---|---|---|---|
| **CAGR** | **9.59%** | 6.99% | 7.92% |
| **Sharpe** | **1.01** | 0.99 | 0.50 |
| **MaxDD** | **-11.48%** | -9.16% | -41.09% |
| **Calmar** | **0.84** | 0.76 | 0.19 |

### 2.2 双窗口指标

| 指标 | OOS 2010-2019 | IS 2020-2026 |
|---|---|---|
| 年化收益 | 7.95% | 17.54% |
| Sharpe | 1.01 | 1.70 |
| MaxDD | -11.97% | -8.74% |
| Calmar | 0.66 | 2.01 |

### 2.3 关键年份表现

| 年份 | L=1.5 CPPI | 金价 | 策略仓位 |
|---|---|---|---|
| 2013 | -2.50% | -28.93% | 0.08 |
| 2019 | 19.36% | +20.53% | 0.99 |
| 2024 | 19.55% | +28.16% | 0.66 |
| 2025 | 48.19% | +56.63% | 0.74 |
| **2026 YTD** | **22.30%** | +0.93% | 0.11 |

---

## 三、各层详解

### 3.1 Δtips二值信号 (核心alpha来源)

**设计**: tips10y.diff(10) → 504日滚动百分位rank → rank≤0.5=1(看多), >0.5=0(看空)

**逻辑**: 当实际利率下降中(rank≤0.5) → 看多黄金; 当实际利率上升中 → 看空

**IC**: 0.126 (20日前瞻收益)

**权重**: 0.15 (15%仓位贡献)

### 3.2 四路基础信号

| 信号 | 权重 | 数据源 | 信号类型 |
|---|---|---|---|
| macro_regime | 0.2125 | 金价+VIX+GVZ+TIPS | 趋势×宏观 |
| real_rate_tips_ma | 0.2125 | TIPS实际利率 | 均值回归 |
| momentum_long_only | 0.2125 | 金价52周动量 | 动量 |
| breakout | 0.2125 | 金价20日高点突破 | 突破 |

每路独立过 `compute_risk_overlay(hc=-0.10)` 后加权平均。

### 3.3 CPPI三层风控

| 层级 | 触发条件 | 动作 |
|---|---|---|
| Level 1 波动率压制 | realized_vol > 0.20 | 仓位×(0.20/realized_vol) |
| Level 2 软减仓 | rolling_dd < -0.12 | 仓位×0.5 |
| Level 3 硬止损 | rolling_dd < -0.15 OR daily_ret < -0.03 | 清仓+冷却5天 |
| 再入场 | VIX<60%百分位 AND GVZ_z<1.0 | 恢复原仓位 |

### 3.4 智能再入场机制

硬止损触发后:
1. 冷却期5天强制空仓
2. 冷却期结束检查: `vix_pct < 0.6 AND gvz_z < 1.0`
3. 条件不满足 → 继续空仓(延长冷却)
4. 条件满足 → 恢复信号层仓位

---

## 四、换手率与成本

| 策略 | 年均换手 | 5bps成本 | 10bps成本 |
|---|---|---|---|
| L=1.5 CPPI (死带0.05) | **10.1x** | **0.51%** | **1.01%** |
| L=1.5 CPPI (原始) | 14.4x | 0.72% | 1.44% |
| 基线(L=1) | 12.4x | 0.62% | 1.24% |

**成本敏感度**: 5bps下策略仍合规; 15bps仍合规; 20bps越线。

**死带平滑效果** (th=0.05):
- 变化次数: 2240 → 908 (-59.5%)
- MaxDD: -13.32% → -13.08% (+0.24pp改善)
- 年化损失: -0.06pp (几乎无损)
- 净效果: 降低换手成本0.21pp, 净正收益

---

## 五、研究历程

```
期权保护 (Buy Put / Put Spread)
    ↓ 失败: VIX<25触发线,保护未启动
VIX自适应仓位
    ↓ 失败: VIX对黄金暴跌盲区
ATR自适应止损
    ↓ 失效: 止损触发本就稀少,宏观层已主导
多信号集成 + 风控覆盖层
    ↓ 成功: 分层集成翻案
Δtips二值信号
    ↓ 突破: Walk-Forward CAGR+0.81pp, DD+3.43pp
CPPI杠杆
    ↓ 最终: CAGR 9.59%, DD -11.48%
```

### 关键里程碑

1. **分层集成**: 先各自风控→再平均, 解决中间仓位问题
2. **Δtips突破**: 利率变化率(window=10)二值信号, Walk-Forward 10/15折跑赢基线
3. **CPPI杠杆**: L=1.5 + hard_cap=-0.15, 年化从6.99%提升到9.59%

---

## 六、风险与诚实声明

### 6.1 已知局限

1. **2014-2017熊市表现平平**: 年化1-2%, 策略在震荡市缺乏alpha
2. **CPPI触发频率**: 2026年1月30日触发清仓后锁仓至8月, 错过部分反弹
3. **换手率偏高**: 14.4x年均, 成本敏感
4. **Δtips窗口敏感**: window=10最优, 但window=21/42效果递减

### 6.2 与B&H的差距

- Walk-Forward CAGR: 策略9.59% vs B&H 7.92% (**策略领先1.67pp**)
- 但2025年: 策略+48.19% vs B&H +56.63% (牛市跟涨不足)
- 2023年: 策略+19.98% vs B&H +17.14% (策略领先)

**本质**: 风险控制策略在熊市保护远优于B&H(DD -11.48% vs -41.09%), 但牛市跟涨幅度受限。

### 6.3 参数鲁棒性

- Walk-Forward 15折验证, 10/15折跑赢基线
- OOS 2010-2019与IS 2020-2026方向一致
- 子周期(10-14/15-19/20-23/24-26)均有正收益

---

## 七、代码结构

```
QuantNodes/strategy/commodity_fundamental/gold/
├── gold_trend_bnh.py
│   ├── compute_macro_regime_signal()      # 宏观机制策略
│   ├── compute_hysteresis_trend_d()       # 滞回带趋势
│   ├── compute_risk_overlay()             # 三级风控覆盖层
│   ├── compute_adaptive_ensemble()        # 自适应集成(备选)
│   ├── compute_layered_ensemble_signal()  # 分层集成(主用)
│   ├── _zscore_scale()                    # GVZ zscore缩放
│   ├── _rate_gate()                       # Δtips利率门控
│   └── _rolling_drawdown()               # 滚动回撤计算
├── gold_real_rate.py
├── gold_momentum.py
├── gold_breakout.py
└── __init__.py
```

---

## 八、使用方法

```python
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal,
    compute_risk_overlay,
)

# 基础分层集成 (bw=0.2默认)
sig = compute_layered_ensemble_signal()
close, position = sig["close"], sig["position"]

# CPPI杠杆版 (L=1.5)
pos_base = compute_layered_ensemble_signal(breakout_weight=0.2)["position"]
pos_15 = compute_risk_overlay(
    close=close,
    position=(pos_base * 1.5).clip(0, 1.8),
    vix=vix, gvz=gvz,
    hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5,
)
```

---

## 九、文档索引

| 文件 | 内容 |
|---|---|
| `docs/gold/gold_strategy_research_summary.md` | 完整研究总结 |
| `docs/gold/gold_atr_stop_research.md` | ATR止损调研 |
| `docs/gold/risk_overlay_design.md` | 风控覆盖层设计 |
| `docs/gold/substrategies_evaluation.md` | 子策略诊断 |
| `docs/gold/pareto_frontier_results.md` | Pareto前沿结果 |
| `docs/gold/final_evaluation_results.md` | 候选策略评估 |
| `docs/gold/adaptive_ensemble_results.md` | 自适应集成评估 |
| `docs/gold/ensemble_v2_results.md` | 分层集成评估 |
| `docs/gold/walk_forward_ensemble_results.md` | Walk-Forward验证 |
| `reports/gold/gold_nav_curve.png` | 净值曲线 |
| `reports/gold/gold_nav_curve.csv` | 净值数据 |

---

## 十、Git提交历史

```
b7df0b3 docs(gold): 补充黄金策略历史文档
90157a6 feat(gold): Δtips新策略净值曲线对比
8c48ff9 docs(gold): 更新研究总结 - Δtips突破
5b1555a feat(gold): Δtips二值信号突破 - Walk-Forward验证
0b88b47 feat(gold): 默认bw=0.2 + 净值曲线对比
d187b9b feat(gold): 分层集成加入第4路breakout信号
94718ab docs(gold): 黄金策略研究总结文档
3baf005 feat(gold): 分层集成生产函数
e548774 feat(gold): 集成策略Round2 - 分层叠加翻案成功
88cd48e feat(gold): 风控覆盖层 (Risk Overlay) 三级架构
9052172 feat(gold): 8子策略单独诊断
5751d8c feat(gold): 宏观机制策略 + 滞回带 + VIX自适应 + ATR调研
```
