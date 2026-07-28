---
id: L-231
title: 统一回测引擎消除 8 个文件重复
severity: MEDIUM
auto_checkable: manual
category: engineering
related_lessons: [L-232]
related_daily: [L-20260720-4]
source: 05_LESSONS_LIBRARY.md
---

# L-231: 统一回测引擎

## 一句话总结
策略只需重写 `compute_weights()` 一个方法, 统一引擎消除 8 个文件重复。

## 问题描述
```python
class BaseStrategy(ABC):
    @abstractmethod
    def compute_weights(self, date, price_panel, nav_history) -> dict[str, float]:
        ...

    def on_risk_check(self, weights, current_nav, regime) -> dict[str, float]:
        """可选回调: 策略内部风险控制"""
        return weights
```

## 检测 prompt (给 Agent 的检查清单)

1. **策略代码是否有重复的 NAV 计算循环**:
   - 应使用统一引擎

2. **策略接口一致性**:
   - 是否继承 BaseStrategy?

## 正确做法

```python
from base import BaseStrategy

class MyStrategy(BaseStrategy):
    def compute_weights(self, date, price_panel, nav_history):
        # 策略核心逻辑
        weights = compute_my_weights(price_panel)
        return weights

    def on_risk_check(self, weights, current_nav, regime):
        # 策略内部风控
        return apply_my_risk(weights, regime)
```

## 历史教训来源
- 首次发现: 统一引擎期 (`53d6e5c`, 2026-07-20)