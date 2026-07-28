---
id: L-213
title: NaN-safe pct_change：避免跨缺口收益与伪零收益
severity: CRITICAL
auto_checkable: agent
category: nan_safe
related_lessons: []
related_daily: [L-20260713-2, L-20260720-6, L-20260728-2]
source: 05_LESSONS_LIBRARY.md
---

# L-213: NaN-safe pct_change

## 一句话总结
pandas `pct_change()` 默认把 NaN 视为 0, 必须显式禁用隐式填充。

## 问题描述
```python
# pandas 默认: pct_change 会把 NaN 视为 0 → 伪零收益或跨缺口收益
returns = nav.pct_change()  # 默认会填 0

# NaN-safe:
returns = nav.pct_change().where(nav.shift(1).notna() & nav.notna())
# 跨缺口收益不应计算
```

## 检测 prompt (给 Agent 的检查清单)

1. **裸 `.pct_change()` 调用**:
   - 是否有后续 `.where(...)` 包装?
   - 若无, 应标记 VIOLATED

2. **NaN-safe 装饰器**:
   ```python
   @nan_safe_pct_change
   def compute_returns(nav):
       return nav.pct_change()
   ```

3. **`.fillna(0)` 误用**:
   - 在收益数据上 `.fillna(0)` 会产生伪零收益
   - 应改用 `.where()`

## 正确做法

```python
# 装饰器形式
@nan_safe_pct_change
def compute_returns(nav):
    return nav.pct_change()

# 显式 NaN-safe
def compute_returns_explicit(nav):
    return nav.pct_change().where(
        nav.shift(1).notna() & nav.notna()
    )

# 长期停牌视为 NaN (不是 0)
def handle_suspension(prices, suspension_threshold=5):
    """连续 NaN > threshold 天视为停牌"""
    is_suspended = prices.isna().rolling(suspension_threshold).sum() >= suspension_threshold
    returns = prices.pct_change().where(~is_suspended.shift(1))
    return returns
```

## 历史教训来源
- 首次发现: v7.14 (`6ad3f88`, 2026-07-20)