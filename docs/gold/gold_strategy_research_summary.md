# 黄金策略研究总结文档

> 创建日期: 2026-08-27
> 研究周期: 完整一轮(从期权保护→VIX自适应→ATR止损→多信号集成)
> 最终策略: `compute_layered_ensemble_signal()` (分层集成)

---

## 一、研究背景与目标

### 1.1 初始目标

用户原始需求: **"大级别止损，小级别波动容忍"**

核心痛点: 基础策略在OOS 2010-2019的MaxDD过大(-44.81% Buy&Hold), 需要通过风控手段降低回撤。

### 1.2 研究路径

```
期权保护 (Buy Put / Put Spread)
    ↓ 失败: VIX<25触发线,保护未启动
VIX自适应仓位
    ↓ 失败: VIX对黄金暴跌盲区(z=-0.37)
ATR自适应止损
    ↓ 失效: 止损触发本就稀少,宏观层已主导
多信号集成 + 风控覆盖层
    ↓ 成功: 分层集成翻案
最终方案
```

---

## 二、关键诊断发现

### 2.1 止损为什么无效

**数据事实**: 现状trailing stop ts=0.08, 15年仅触发68次, 1.9%时间空仓。

**根因**:
- 止损触发本就稀少 (1.9%时间)
- 宏观因子层(GVZ+利率)已主导仓位 (67%减仓由GVZ驱动)
- ATR自适应对当前组合几乎无效 (G从0.51→0.51无变化)

**结论**: 止损是"兜底保护", 不是"主力减仓"

### 2.2 VIX为什么对黄金失效

**事件研究**: 黄金暴跌日(2010-2019, n=13)

| 指标 | VIX-z 中位 | GVZ-z 均值 |
|---|---|---|
| 暴跌日 | -0.37 (无感) | +1.86 (有感) |
| 含义 | 股票波动率,对黄金盲区 | 黄金隐含波动率,正确反应 |

**结论**: VIX是股票波动率代理, 对黄金特有暴跌(如2013 taper实际利率冲击)完全无感。

### 2.3 为什么集成失败又翻案

**Round 1失败** (先加权→再风控):
- 信号混合成0.4-0.7中间仓位
- 风控层难以精确控制MaxDD

**Round 2翻案** (先各自风控→再平均):
- 每个信号独立过风控,仍是0/1二值
- 平均后得到{0, 0.33, 0.67, 1.0}离散值
- 风控语义清晰

**关键证据**: 三信号低相关
- macro ↔ real_rate: 0.106
- macro ↔ momentum: -0.007
- real_rate ↔ momentum: 0.264

低相关 → 集成有真正分散化价值

---

## 三、最终策略架构

### 3.1 三层架构

```
Level 1: 信号层 (产生原始仓位)
    - compute_macro_regime_signal (趋势×VIX动量+突破×GVZ×利率)
    - compute_real_rate_signal (TIPS MA择时)
    - compute_momentum_signal (52周动量)

Level 2: 各信号独立风控覆盖
    - compute_risk_overlay (三级风控)
    - 每个信号独立应用, 输出仍为0/1

Level 3: 等权平均集成
    - position = (macro_ov + rr_ov + mom_ov) / 3
    - 输出{0, 0.33, 0.67, 1.0}离散仓位
```

### 3.2 风控覆盖层 (`compute_risk_overlay`)

```python
三级风控:
  Level 1 (波动率压制): realized_vol > vol_cap → 仓位×vol_cap/actual_vol
  Level 2 (软减仓): rolling_dd < soft_cap → 仓位×0.5
  Level 3 (硬止损+智能再入场):
    触发: rolling_dd < hard_cap OR daily_ret < daily_cap
    清仓 + 冷却期 (cooldown_days)
    再入场: VIX<60%百分位 AND GVZ-z<1.0
```

默认参数:
- hard_cap = -0.10
- soft_cap = -0.07
- daily_cap = -0.02
- vol_cap = 0.15
- cooldown_days = 5

### 3.3 生产函数

```python
from QuantNodes.strategy.commodity_fundamental.gold import compute_layered_ensemble_signal

sig = compute_layered_ensemble_signal()
close = sig["close"]
position = sig["position"]  # DataFrame [close, position]
```

---

## 四、Walk-Forward验证结果

### 4.1 测试设置

- 训练窗口: 4年
- 测试窗口: 1年
- 滚动步长: 1年
- 总测试期: 15折 (2012-2026)

### 4.2 逐折结果

| 折 | 训练期 | 测试期 | 策略年化 | 策略DD | B&H年化 | B&H DD |
|---|---|---|---|---|---|---|
| 1 | 2008-2011 | 2012 | -4.82% | -6.75% | 4.11% | -13.93% |
| 2 | 2009-2012 | 2013 | -0.84% | -1.25% | -30.13% | -30.90% |
| 3 | 2010-2013 | 2014 | -1.20% | -7.00% | -1.22% | -16.88% |
| 4 | 2011-2014 | 2015 | -4.12% | -5.07% | -6.81% | -16.80% |
| 5 | 2012-2015 | 2016 | 5.95% | -7.49% | 19.59% | -11.26% |
| 6 | 2013-2016 | 2017 | 0.79% | -5.28% | 3.74% | -6.57% |
| 7 | 2014-2017 | 2018 | 1.40% | -2.76% | 3.14% | -6.61% |
| 8 | 2015-2018 | 2019 | 14.11% | -8.11% | 21.36% | -9.51% |
| 9 | 2016-2019 | 2020 | 3.49% | -7.26% | 14.63% | -17.43% |
| 10 | 2017-2020 | 2021 | -2.22% | -5.03% | -6.56% | -11.49% |
| 11 | 2018-2021 | 2022 | 5.85% | -1.55% | 10.94% | -11.23% |
| 12 | 2019-2022 | 2023 | 13.21% | -2.79% | 17.91% | -4.58% |
| 13 | 2020-2023 | 2024 | 16.48% | -4.75% | 29.48% | -7.13% |
| 14 | 2021-2024 | 2025 | 36.43% | -4.64% | 59.25% | -11.02% |
| **15** | **2022-2025** | **2026** | **9.94%** | **-2.11%** | **1.50%** | **-30.37%** |

### 4.3 累计表现

| 指标 | 分层集成 | Buy&Hold | 差异 |
|---|---|---|---|
| **CAGR** | 5.74% | 7.81% | -2.07pp |
| **Sharpe** | 0.82 | 0.49 | +0.33 |
| **MaxDD** | -12.46% | -41.09% | **+28.63pp** |
| **Calmar** | 0.46 | 0.19 | **+0.27** |

### 4.4 关键年份表现

| 年份 | 策略 | B&H | 策略DD | B&H DD |
|---|---|---|---|---|
| 2013 (大崩盘) | -0.84% | -30.13% | -1.25% | -30.90% |
| 2020 (COVID) | 3.49% | 14.63% | -7.26% | -17.43% |
| 2024 (牛市) | 16.48% | 29.48% | -4.75% | -7.13% |
| 2025 (牛市) | 36.43% | 59.25% | -4.64% | -11.02% |
| **2026 YTD** | **9.94%** | **1.50%** | **-2.11%** | **-30.37%** |

---

## 五、与基线对比

### 5.1 双窗口对比 (OOS 2010-2019 / IS 2020-2026)

| 指标 | 旧基线(trend_bnh) | 生产(macro_regime) | **分层集成** |
|---|---|---|---|
| OOS年化 | 3.55% | 6.00% | 4.71% |
| OOS DD | -44.22% | -29.23% | -13.11% |
| IS年化 | 16.97% | 17.57% | 11.51% |
| IS DD | -40.07% | -8.71% | -9.04% |
| G | 0.18 | 0.64 | **0.68** |

### 5.2 Walk-Forward累计对比 (2012-2026)

| 指标 | 分层集成 | Buy&Hold | 差异 |
|---|---|---|---|
| CAGR | 5.74% | 7.81% | -2.07pp |
| Sharpe | 0.82 | 0.49 | +0.33 |
| MaxDD | -12.46% | -41.09% | **+28.63pp** |
| Calmar | 0.46 | 0.19 | **+0.27** |

---

## 六、代码结构

### 6.1 生产代码

```
QuantNodes/strategy/commodity_fundamental/gold/
├── gold_trend_bnh.py          # 主模块
│   ├── compute_macro_regime_signal()    # 宏观机制策略
│   ├── compute_risk_overlay()           # 三级风控覆盖层
│   ├── compute_adaptive_ensemble()      # 自适应集成 (备选)
│   └── compute_layered_ensemble_signal() # 分层集成 (最终)
├── gold_real_rate.py           # 实际利率择时
├── gold_momentum.py            # 长期动量
├── gold_breakout.py            # 突破策略
├── __init__.py                 # 包级导出
└── ...
```

### 6.2 测试脚本

```
scripts/cmf/
├── test_risk_overlay.py         # 风控层单元测试
├── test_pareto_frontier.py      # Pareto前沿搜索
├── test_final_evaluation.py     # 候选策略评估
├── test_adaptive_ensemble.py    # 自适应集成评估
├── test_ensemble_v2.py          # 分层集成评估
├── test_walk_forward_ensemble.py # Walk-Forward验证
├── test_vix_adaptive.py         # VIX自适应扫描
├── test_vix_refine.py           # VIX参数细化
├── test_hysteresis.py           # 滞回带扫描
├── test_oos_diagnosis.py        # OOS失效诊断
├── test_oos_validation.py       # OOS验证
├── test_walk_forward.py         # 基础Walk-Forward
└── ...
```

### 6.3 文档

```
docs/gold/
├── gold_atr_stop_research.md           # ATR止损调研
├── risk_overlay_design.md              # 风控覆盖层设计
├── pareto_frontier_results.md          # Pareto前沿结果
├── final_evaluation_results.md         # 候选策略评估
├── adaptive_ensemble_results.md        # 自适应集成评估
├── ensemble_v2_results.md              # 分层集成评估
├── walk_forward_ensemble_results.md    # Walk-Forward验证
└── substrategies_evaluation.md         # 子策略诊断
```

---

## 七、Git提交历史

```
5de3933 feat(gold): Walk-Forward测试输出累计年化收益(CAGR)+Buy&Hold对比
e2f794d feat(gold): Walk-Forward滚动测试验证分层集成策略稳健性
3baf005 feat(gold): 分层集成生产函数 compute_layered_ensemble_signal
e548774 feat(gold): 集成策略Round2 - 分层叠加翻案成功
1562855 feat(gold): 自适应集成 - 实测集成不优单信号
2b07e03 feat(gold): 候选策略最终评估 - 子周期+滚动+综合评分
67ade8b feat(gold): Pareto前沿搜索 - 162组合评估 + 20个前沿点
88cd48e feat(gold): 风控覆盖层 (Risk Overlay) 三级架构 + 智能再入场
9052172 feat(gold): 8子策略单独诊断 - 全景表+评估卡+串联候选
5751d8c feat(gold): 宏观机制策略 + 滞回带 + VIX自适应 + ATR调研
```

---

## 八、教训与结论

### 8.1 什么有效

1. **分层集成**: 先各自风控→再平均, 解决中间仓位问题
2. **三级风控**: 波动率压制+软减仓+硬止损+智能再入场
3. **低相关信号集成**: macro/real_rate/momentum三信号正交, 有真正分散化价值

### 8.2 什么无效

1. **期权保护 (Buy Put / Put Spread)**: VIX<25触发线,保护未启动
2. **VIX自适应仓位**: VIX对黄金暴跌盲区
3. **ATR自适应止损**: 止损触发本就稀少, 宏观层已主导
4. **先加权后集成**: 信号混合成中间仓位, 风控失效

### 8.3 核心教训

- **止损是兜底保护, 不是主力减仓** (15年仅触发68次)
- **宏观因子(GVZ+利率)才是减仓主力** (67%减仓由GVZ驱动)
- **低相关信号集成有真正价值** (但必须先各自风控)
- **Walk-Forward验证是必须的** (避免过拟合)

---

## 九、待办事项

- [ ] 集成到生产回测框架
- [ ] 实盘部署前的滑点/流动性验证
- [ ] 定期回顾参数有效性 (建议每季度)
- [ ] 考虑加入更多因子 (CFTC持仓/ETF流量)

---

*文档结束*
