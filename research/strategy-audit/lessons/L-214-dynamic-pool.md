---
id: L-214
title: 动态资产池（min_assets=10）作为最简洁工程
severity: MEDIUM
auto_checkable: agent
category: data_quality
related_lessons: []
related_daily: [L-20260720-7]
source: 05_LESSONS_LIBRARY.md
---

# L-214: 动态资产池

## 一句话总结
动态资产池 > 静态 + 复杂缺失处理, min_assets=10。

## 问题描述
```python
def _get_valid_assets(returns, t, window, min_assets=10):
    """返回 t 时有足够历史的资产子集"""
    history = returns.iloc[t-window:t+1]
    valid = history.notna().sum() > window * 0.7
    if valid.sum() < min_assets:
        # 放宽到 0.5
        valid = history.notna().sum() > window * 0.5
    return valid[valid].index.tolist()
```

## 检测 prompt (给 Agent 的检查清单)

1. **miss-data 敏感场景**:
   - 是否使用动态资产池?
   - min_assets 默认 10

2. **动态 vs 静态**:
   - 动态比"复杂 imputation"简洁

## 正确做法

```python
def dynamic_asset_pool(returns, t, window=252, min_assets=10):
    """动态资产池: t 时有足够历史的资产子集"""
    history = returns.iloc[t-window:t+1]
    valid = history.notna().sum() > window * 0.7
    if valid.sum() < min_assets:
        valid = history.notna().sum() > window * 0.5
    return valid[valid].index.tolist()
```

## 历史教训来源
- 首次发现: v7.14 (`40e2d52`, 2026-07-20)