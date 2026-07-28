---
id: L-223
title: 全 sample ADMM 平滑的 β[t] 天然包含未来数据
severity: CRITICAL
auto_checkable: agent
category: lookahead
related_lessons: [L-202]
related_daily: [L-20260720-2]
source: 05_LESSONS_LIBRARY.md
---

# L-223: full_sample TV-PR 不算 OOS

## 一句话总结
`method="admm"` DEPRECATED, 默认 `method="expanding"`。

## 问题描述
```python
# full_sample_tvpr 用 [0, T] 全量估计 β[t]
# β[t] 包含 t 之后数据 → 必然 look-ahead
def full_sample_tvpr(X, Y):
    beta[t] = estimate_admm(X[0:T], Y[0:T])  # ❌

# 正确: expanding
def expanding_tvpr(X, Y):
    for t in range(window, T):
        beta[t] = estimate_admm(X[0:t], Y[0:t])  # ✅
```

## 检测 prompt (给 Agent 的检查清单)

1. **`method="full"` 或 `"admm"`**:
   - DEPRECATED, 应使用 `method="expanding"`

2. **β[t] 是否含 t 之后数据**:
   - 检查 β 估计函数是否用全样本

## 正确做法

```python
# 推荐: expanding (默认)
beta = expanding_window_tvpr(X, Y, window=252)

# DEPRECATED: full_sample (含未来)
beta = full_sample_tvpr(X, Y)  # ❌
```

## 历史教训来源
- 首次发现: v7.6 (`9d56a0b`)