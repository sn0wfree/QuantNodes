---
id: L-306
title: 写新策略前先看现行"基础设施"再写
severity: LOW
auto_checkable: manual
category: decision
related_lessons: [L-231, L-232]
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-306: 写新策略前看基础设施

## 一句话总结
所有 V0-V10 期间的 `common/`, `scripts/quant/`, `research/` 模块都是"基础设施"。

## 问题描述
v3 SubStrategy 抽象基类被 v4/v5/v6/v7 全部继承。

## 检测 prompt (给 Agent 的检查清单)

1. **新策略是否重写已有基础设施**:
   - 应复用 BaseStrategy / walk_forward / YAML
   - 不要重写 NAV 循环

## 正确做法

```python
# 错误: 重写 NAV 循环
def my_strategy_backtest():
    weights = []
    for date in dates:
        weights.append(compute_weights(date))
    nav = compute_nav_from_weights(weights)
    return nav

# 正确: 复用基础设施
from base import BaseStrategy
from common.walk_forward import walk_forward

class MyStrategy(BaseStrategy):
    def compute_weights(self, date, price_panel, nav_history):
        return weights

result = walk_forward(MyStrategy(), data)
```

## 历史教训来源
- 首次发现: v3 SubStrategy 抽象基类被 v4/v5/v6/v7 全部继承