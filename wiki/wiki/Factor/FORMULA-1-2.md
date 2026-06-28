---
type: Factor
name: FORMULA-1-2
formula: "-rank(ts_std(returns, 20))"
source: auto_research
category: other
tags: [alpha-pipeline, ir=-0.121, logic=volatility]
ic_mean: -0.014420650826284064
ic_std: 0.0
icir: -0.12080949285582535
rank_ic_mean: 0.0
created_at: 2026-06-28T12:00:14.899978
---

# FORMULA-1-2

## 因子公式

```
-rank(ts_std(returns, 20))
```

## 来源逻辑

**逻辑名称**: `volatility`
**逻辑类型**: 量化因子
**方向**: -（IR 负向预测）

## 单因子表现

| 指标 | 值 |
|------|-----|
| IC Mean | -0.014420650826284064 |
| IC IR | -0.12080949285582535 |
| 绝对 IR | 0.1208 |

## 因子描述

20日波动率因子 - 短期波动率信号

## 评估方法

- **数据源**: A 股市场 2023 年全量数据（5380 只股票）
- **前瞻期**: 5 日 / 20 日 forward return
- **评估窗口**: 全市场横截面
- **评估时间**: 2026-06-28

## 相关性

暂无（待后续 MCTS 去重后填充）

## 使用记录

暂无（待策略集成后填充）

## 策略配置 (YAML)

```yaml
factor:
  name: FORMULA-1-2
  formula: "-rank(ts_std(returns, 20))"
  direction: -1
  weight: 0.1
```
