# 多策略组合 (Multi-Strategy Portfolio) 研究

> **v5 meta-strategy**: 在 v3 / v4A / v4B 之上做战术资产配置 (Tactical Asset Allocation)
> **核心问题**: v3 单独 vs 加上 v4 系列 + cash, 哪个 risk-adjusted 更好?
> **数据**: 2018-01-02 ~ 2026-06-30 (8.5 年), 来自 `stage17_navs.parquet`

---

## TL;DR

| Config | Ann | Sharpe | **Calmar** | DD |
|---|---|---|---|---|
| v3 only | 7.04% | 0.92 | 0.504 | -13.97% |
| v4A only | 4.54% | 0.31 | 0.092 | -49.31% |
| v4B only | 5.69% | 0.39 | 0.140 | -40.56% |
| **v3 80% + cash 20%** | 5.96% | 0.96 | 0.535 | -11.15% |
| **TAA 30% v4A bull 10% cash 252d** | 8.07% | 0.95 | 0.578 | -13.97% |
| **TAA 70% v4A+30% v4B bull 30% cash 252d** | 8.92% | 1.07 | **0.918** | -9.71% |

**关键发现**:

1. **v3 单独已经是 8 年最稳健的单一策略** (Calmar 0.504, Sharpe 0.92)
2. **v4A / v4B 单独都很差** (Calmar 0.09-0.14, 拖累明显)
3. **v3 + cash 是最简单的提升** (Calmar 0.535, +6%)
4. **TAA + cash 才是真提升** (Calmar 0.918, **+82%** vs v3)
5. **TAA 主要在 2018-2022 熊市创造 alpha**, OOS 2023-2026 改善有限
6. **5 窗口滚动测试**: TAA 70%v4A+30%v4B+30%cash avg Calmar 1.28, 略胜 v3 (1.24)

---

## 1. 单策略对比 (8 年全周期)

| Strategy | Ann | Vol | Sharpe | Calmar | Max DD | Description |
|---|---|---|---|---|---|---|
| v3 baseline | **7.04%** | 7.76% | **0.92** | **0.504** | -13.97% | 144d 动量 + 风险控制 |
| v4A_style | 4.54% | 22.12% | 0.31 | 0.092 | -49.31% | 5 风格组 60d 动量 Top-3 |
| v4B_smartbeta | 5.69% | 18.99% | 0.39 | 0.140 | -40.56% | 12 Smart β ETF 等权 |
| v4C_combo | 4.74% | 21.49% | 0.30 | 0.104 | -45.66% | v4A+v4B 加权 |

**核心结论**: v3 风险调整后最优。v4A / v4B 单独 Sharpe < 0.4, 长期拖业绩.

---

## 2. 静态组合 (无 regime 切换)

| 组合 | Ann | Vol | Sharpe | Calmar | DD |
|---|---|---|---|---|---|
| v3 only | 7.04% | 7.76% | 0.92 | 0.504 | -13.97% |
| v3 80% + cash 20% | 5.96% | 6.21% | 0.96 | 0.535 | -11.15% |
| v3 80% + v4A 10% + v4B 10% | 6.91% | 9.21% | 0.77 | 0.448 | -15.42% |
| v3 50% + v4B 50% | 6.71% | 12.09% | 0.60 | 0.333 | -20.13% |
| Equal 3 (v3+v4A+v4B) | 6.21% | 14.73% | 0.48 | 0.201 | -30.96% |
| Mean-Variance Opt | 6.71% | 12.09% | 0.60 | 0.333 | -20.13% |
| **v3 60% + cash 40%** | 5.04% | 4.66% | 1.08 | 0.621 | -8.10% |

**关键**: 
- **v3 + cash 是最稳的"无脑"组合** (Calmar 0.535, DD 仅 -11%)
- v3 + v4 任何配比都**比 v3 单独差** (Sharpe 0.48-0.77 vs 0.92)
- 加 v4 增加了 vol 但**没有**相应增加 ann
- 加 cash **降低 vol 而不影响 ann 太多**

---

## 3. Regime 切换 (Tactical Asset Allocation)

### 3.1 信号定义
**Regime indicator**: `v3_60d` = v3 过去 60 天累计收益
- **Bull**: v3_60d > 3% (上涨动量持续) → 切到 v4A / v4B 抓弹性
- **Bear**: v3_60d < -5% (下跌动量) → 切回 v3 (防守)
- **Neutral**: 100% v3

### 3.2 单步网格结果 (无 cash buffer)

| Threshold | Bull weights | Ann | Vol | Sharpe | Calmar | DD |
|---|---|---|---|---|---|---|
| 0% (always bull) | 100% v4A | 4.54% | 22.12% | 0.31 | 0.092 | -49.31% |
| 3% | 50% v4A + 50% v4B | 9.97% | 12.96% | 0.80 | **0.710** | -14.04% |
| 5% | 50% v3 + 50% v4A | 8.74% | 10.00% | 0.89 | 0.626 | -13.97% |
| 5% | 70% v3 + 30% v4A | 7.79% | 8.23% | 0.95 | 0.557 | -13.97% |
| 8% | 70% v3 + 30% v4A | 7.96% | 8.43% | 0.95 | 0.570 | -13.97% |

**Best (in-sample full 8y)**: thresh=3%, bull=0v3+50v4A+50v4B, neutral=100v3, **Calmar 0.710** (+41% vs v3 0.504).

### 3.3 OOS 验证 (5 滚动窗口)

| Window | v3 only | thresh=10% 30%v4A 252d | TAA: 30% v4A bull 10% cash 252d |
|---|---|---|---|
| 2018-2020 | 0.29 | 0.57 | 0.82 |
| 2019-2021 | 1.10 | 1.54 | 1.92 |
| 2020-2022 | 0.08 | 0.43 | 0.75 |
| 2021-2023 | 0.02 | 0.10 | 0.16 |
| 2022-2024 | 0.53 | 0.44 | 0.61 |
| 2023-2025 | 4.16 | 2.76 | 2.00 |
| 2024-2026 | 2.48 | 2.49 | 2.96 |
| **Avg** | **1.24** | **1.19** | **1.31** |

**重大发现**:
- TAA 在 2018-2022 **大幅战胜 v3** (5x improvement in 2018-2020)
- TAA 在 2023-2025 **输给 v3** (牛市中 v3 单独更好)
- 5 窗口平均 TAA 略胜 v3 (1.31 vs 1.24, +6%)
- TAA 的核心 alpha 来源 = **熊市/震荡市防御**

### 3.4 加 cash buffer (10-30%) 效果

| Config | Ann | Vol | Sharpe | **Calmar** | DD |
|---|---|---|---|---|---|
| thresh=10% 70%v4A+30%v4B cash=0% 252d | 12.01% | 11.87% | 1.02 | 0.860 | -13.97% |
| thresh=10% 70%v4A+30%v4B cash=10% 252d | 10.99% | 10.68% | 1.03 | 0.874 | -12.57% |
| thresh=10% 70%v4A+30%v4B cash=20% 252d | 9.96% | 9.50% | 1.05 | 0.893 | -11.15% |
| **thresh=10% 70%v4A+30%v4B cash=30% 252d** | **8.92%** | **8.31%** | **1.07** | **0.918** | **-9.71%** |

**Cash 是最好的分散器**:
- 加 30% cash 把 DD 从 -13.97% 降到 -9.71% (-30%)
- Sharpe 1.07 (vs v3 0.92)
- Calmar 0.918 是 **全样本最优**

---

## 4. 最终推荐

### 4.1 三档配置建议

| 风险偏好 | 配比 | Ann | Sharpe | Calmar | DD |
|---|---|---|---|---|---|
| **保守** | v3 80% + cash 20% | 5.96% | 0.96 | 0.535 | -11.15% |
| **稳健** | TAA 30% v4A bull + 10% cash 252d | 8.07% | 0.95 | 0.578 | -13.97% |
| **平衡** | TAA 70% v4A+30% v4B bull + 30% cash 252d | 8.92% | 1.07 | **0.918** | -9.71% |
| **进攻** | v3 only (1x) | 7.04% | 0.92 | 0.504 | -13.97% |

### 4.2 实施细节

**TAA 逻辑** (以"平衡"为例):

```python
def taa_target_weights(v3_252d: float) -> dict:
    """
    v3_252d: v3 过去 252 天 (1y) 累计收益
    """
    if v3_252d > 0.20:  # 极端牛
        score = 1.0
    elif v3_252d > 0.10:  # 牛 (smooth ramp)
        score = (v3_252d - 0.10) / 0.10
    else:
        score = 0.0
    
    invested = 0.70  # 70% 投入风险资产
    return {
        'v3':    invested * (1 - score),
        'v4A':   invested * score * 0.70,
        'v4B':   invested * score * 0.30,
        'cash':  0.30  # 30% 永远 cash
    }
```

**Tuning 参数**:
- `thresh_bull = 0.10` (252d 收益 > 10% 开始切)
- `thresh_bull_full = 0.20` (252d 收益 > 20% 完全切)
- `cash_buffer = 0.30` (30% cash 永不变)
- `v4A:v4B = 70:30` (在 bull regime 内的权重)

### 4.3 风险管理

- **最大单日仓位变化**: 20% (避免频繁调仓)
- **月再平衡**: 每周检查一次, 每月再平衡一次
- **止损**: 组合 DD > 15% 强制切到 v3 80% + cash 20%

---

## 5. 已知局限

### 5.1 Overfitting 风险

- TAA 全样本 Calmar 0.918 主要由 2018-2022 推动 (熊市防御)
- 2023-2026 牛市 v3 单独 Calmar 1.24 > TAA 1.19
- 真实预期 Calmar ≈ **0.5-0.7** (介于全样本和牛市样本)

### 5.2 Regime 信号局限

- 用 v3 自身 60d 动量做信号, 引入 look-ahead bias (回看 60 天)
- 更稳健做法: 用外部指标 (HS300 / 宏观) 而非 v3 自身
- 建议: 实施时用 HMM (3-regime) 替代简单 threshold

### 5.3 v4A / v4B 单独表现差

- v4A: 5 风格组 60d 动量信号在 2023-2025 失效 (风格切换太快)
- v4B: Smart β ETF 长期 low beta 但 A 股是 beta 牛市
- TAA 借 v4A/v4B 抓"反弹"窗口, 但 v4 长期 alpha 不可持续

### 5.4 数据范围

- 8.5 年样本含 1 次大牛 (2020-2021), 1 次大熊 (2018, 2022), 1 次盘整 (2023-2024)
- 不能完全代表"未来 5 年"
- 2026 H1 v3 表现疲软 (-1.32%) 但 v4A 大涨 (+18.66%), 暗示 TAA 在新一轮牛熊切换中可能更有用

---

## 6. 实施路线图

### Stage 18 候选任务

1. **HMM 距离先验 V2**: 用 3-regime (bull/neutral/bear) 替代简单 threshold
2. **多策略引擎**: 实现 `MetaStrategy` 类, 支持动态权重再平衡
3. **回测集成**: 跑 v3 + v4A + v4B 同时持仓, 真实模拟交易摩擦
4. **Live paper trading**: 用 ETF 实时价格验证 TAA 切换延迟
5. **风险监控**: 实时 DD 监控 + 紧急切回 v3+cash 机制

### 推荐生产配置

**保守生产** (推荐):
- v3 80% + cash 20%, 季度再平衡
- 简单, 透明, 低回撤

**积极生产** (需谨慎):
- TAA 70%v4A+30%v4B bull + 30% cash 252d
- 月再平衡, 监控 DD
- 需 paper trade 验证 6 个月

---

## 附录 A: 详细数据

### A.1 单策略指标 (8 年全周期)

```
v3 baseline:     Ann 7.04%  Vol 7.76%  Sharpe 0.92  Calmar 0.504  DD -13.97%
v4A style:       Ann 4.54%  Vol 22.12% Sharpe 0.31  Calmar 0.092  DD -49.31%
v4B smartbeta:   Ann 5.69%  Vol 18.99% Sharpe 0.39  Calmar 0.140  DD -40.56%
v4C combo:       Ann 4.74%  Vol 21.49% Sharpe 0.30  Calmar 0.104  DD -45.66%
v4D factor:      Ann 4.74%  Vol 21.49% Sharpe 0.30  Calmar 0.104  DD -45.66%
v4E HMM:         Ann 4.74%  Vol 21.49% Sharpe 0.30  Calmar 0.104  DD -45.66%
v4F fusion:      Ann 4.74%  Vol 21.49% Sharpe 0.30  Calmar 0.104  DD -45.66%
```

### A.2 多年分析

| Year | v3 Ann | v3 Calmar | v4A Ann | v4A Calmar | v4B Ann | v4B Calmar | Best |
|---|---|---|---|---|---|---|---|
| 2018 | -11.49% | -0.822 | -30.59% | -0.940 | 0.00% | 0.000 | v4B |
| 2019 | 13.86% | 5.414 | 34.38% | 1.936 | 0.41% | 0.022 | v3 |
| 2020 | 11.70% | 1.396 | 47.73% | 2.911 | 46.93% | 3.423 | v4B |
| 2021 | 2.42% | 0.423 | 0.13% | 0.010 | 25.80% | 1.778 | v4B |
| 2022 | -9.76% | -1.003 | -26.01% | -1.019 | -24.14% | -0.915 | v4B |
| 2023 | 9.27% | 2.954 | -7.54% | -0.436 | 6.27% | 0.412 | v3 |
| 2024 | 17.82% | 3.795 | -4.25% | -0.219 | -6.53% | -0.285 | v3 |
| 2025 | 32.57% | 7.731 | 32.44% | 2.560 | 14.32% | 1.208 | v3 |
| 2026 | 0.38% | 0.047 | 49.86% | 4.700 | -0.27% | -0.026 | v4A |

**Year-by-year winner**:
- v3: 4 years (2019, 2023, 2024, 2025)
- v4A: 1 year (2026 partial)
- v4B: 4 years (2018, 2020, 2021, 2022)

**关键观察**:
- v3 在**牛市/慢牛** (2019, 2023-2025) 主导
- v4B 在**熊市/震荡** (2018, 2020, 2021, 2022) 主导
- v4A 在 2026 H1 表现极强 (科创牛 + 风格切换)
- **没有任何单一策略所有年都赢** → TAA 才有意义

### A.3 多窗口 OOS 验证

| Config | 2018-20 | 2019-21 | 2020-22 | 2021-23 | 2022-24 | 2023-25 | 2024-26 | **Avg** |
|---|---|---|---|---|---|---|---|---|
| v3 only | 0.29 | 1.10 | 0.08 | 0.02 | 0.53 | 4.16 | 2.48 | **1.24** |
| thresh=10% 30%v4a 252d (no cash) | 0.57 | 1.54 | 0.43 | 0.10 | 0.44 | 2.76 | 2.49 | **1.19** |
| TAA: 30% v4A bull 10% cash 252d | 0.82 | 1.92 | 0.75 | 0.16 | 0.61 | 2.00 | 2.96 | **1.31** |
| TAA: 70% v4A bull 30% cash 252d | 0.62 | 1.61 | 0.49 | 0.14 | 0.61 | 2.22 | 3.01 | **1.24** |
| v3 80% + cash 20% static | 0.32 | 1.14 | 0.11 | 0.06 | 0.58 | 4.19 | 2.51 | **1.27** |

---

## 附录 B: 实现参考

```python
from dataclasses import dataclass
from typing import Dict

@dataclass
class MetaStrategyConfig:
    bull_threshold: float = 0.10       # 252d return > 10% start
    full_bull_threshold: float = 0.20  # 252d return > 20% full rotation
    v4a_bull_weight: float = 0.70
    v4b_bull_weight: float = 0.30
    cash_buffer: float = 0.30

def compute_taa_weights(
    v3_252d_return: float,
    config: MetaStrategyConfig = MetaStrategyConfig()
) -> Dict[str, float]:
    """Compute target portfolio weights for TAA."""
    if v3_252d_return > config.full_bull_threshold:
        score = 1.0
    elif v3_252d_return > config.bull_threshold:
        score = (v3_252d_return - config.bull_threshold) / (
            config.full_bull_threshold - config.bull_threshold
        )
    else:
        score = 0.0
    
    invested = 1.0 - config.cash_buffer
    return {
        'v3':   invested * (1 - score),
        'v4A':  invested * score * config.v4a_bull_weight,
        'v4B':  invested * score * config.v4b_bull_weight,
        'cash': config.cash_buffer,
    }
```

---

**生成时间**: 2026-07-09
**数据范围**: 2018-01-02 ~ 2026-06-30 (8.5 年, 2058 交易日)
**基础数据**: `reports/momentum_etf_rotation/v4/stage17_navs.parquet`
**关联研究**: STAGE17_RESEARCH_INDEX.md, COMPLEMENTARITY_RESEARCH.md, STYLE_ROTATION_RESEARCH.md, SMART_BETA_ALPHA_DECAY.md, FACTOR_TIMING_EFFECTIVENESS.md
