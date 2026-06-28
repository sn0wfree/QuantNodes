---
type: Factor
name: FORMULA-1-1
formula: "rank(sub(close, ts_mean(close, 20)))"
source: auto_research
category: other
tags: [alpha-pipeline, ir=-0.078, logic=momentum]
ic_mean: -0.009515315233963977
ic_std: 0.0
icir: -0.07754819110174828
rank_ic_mean: 0.0
created_at: 2026-06-28T12:00:14.899978
---

# FORMULA-1-1

## 因子公式

```
rank(sub(close, ts_mean(close, 20)))
```

## 来源逻辑

**逻辑名称**: `momentum`
**逻辑类型**: 量化因子
**方向**: -（IR 负向预测）

## 单因子表现

| 指标 | 值 |
|------|-----|
| IC Mean | -0.009515315233963977 |
| IC IR | -0.07754819110174828 |
| 绝对 IR | 0.0775 |

## 因子描述

标准化价格偏离因子

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
  name: FORMULA-1-1
  formula: "rank(sub(close, ts_mean(close, 20)))"
  direction: -1
  weight: 0.1
```
