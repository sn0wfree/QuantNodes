---
id: L-205
title: v7.7 Y[t] 和 f18_mom_short[t] 完全重叠，corr=0.96
severity: CRITICAL
auto_checkable: agent
category: lookahead
related_lessons: [L-204, L-108]
related_daily: [L-20260719-1]
source: 05_LESSONS_LIBRARY.md
---

# L-205: Y 与因子重叠检查

## 一句话总结
看起来"高预测力"的因子, 可能跟 Y 是同口径。

## 问题描述
```python
# f18_mom_short[t] = nav[t] / nav[t-1] - 1   ← t-1 → t 收益
# Y[t]              = nav[t] / nav[t-1] - 1   ← t-1 → t 收益
# → 必然 corr=1.0 (or 0.96 with epsilon)
```

## 检测 prompt (给 Agent 的检查清单)

1. **任何"高 IC 因子"先做 Y 重叠检查**:
   - corr=1.0 必有 bug

2. **修复**:
   ```python
   Y[t] = (nav[t+1] / nav[t]) - 1  # t → t+1
   f18_mom_short[t] = (nav[t] / nav[t-1]) - 1  # t-1 → t
   # → corr 应该为 0 (independent tests)
   ```

## 历史教训来源
- 首次发现: v7.7 (`f22c407`, 2026-07-19)