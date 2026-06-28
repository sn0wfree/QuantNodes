---
type: Factor
name: FORMULA-2-5
formula: "rank(ts_delta(close, 5) - ts_delta(close, 20))"
source: auto_research
category: other
tags: [alpha-pipeline, ir=-0.061, logic=momentum]
ic_mean: -0.007614890523501923
ic_std: 0.0
icir: -0.06099000237713744
rank_ic_mean: 0.0
created_at: 2026-06-28T12:00:14.899978
---

# FORMULA-2-5

## 因子公式

```
rank(ts_delta(close, 5) - ts_delta(close, 20))
```

## 来源逻辑

**逻辑名称**: `momentum`
**逻辑类型**: 量化因子
**方向**: -（IR 负向预测）

## 单因子表现

| 指标 | 值 |
|------|-----|
| IC Mean | -0.007614890523501923 |
| IC IR | -0.06099000237713744 |
| 绝对 IR | 0.0610 |

## 因子描述

价格加速度因子

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
  name: FORMULA-2-5
  formula: "rank(ts_delta(close, 5) - ts_delta(close, 20))"
  direction: -1
  weight: 0.1
```
