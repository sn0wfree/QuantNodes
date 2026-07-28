---
id: L-222
title: macro 沿时间复制，PV 沿截面填充
severity: HIGH
auto_checkable: agent
category: data_quality
related_lessons: [L-102]
related_daily: [L-20260716-2]
source: 05_LESSONS_LIBRARY.md
---

# L-222: 标准化方向

## 一句话总结
宏观因子和 PV 因子用不同标准化方式, 不要一刀切。

## 问题描述
| 类型 | 标准化维度 | 备注 |
|------|----------|------|
| 宏观因子 | 时间维 Z-score | 沿时间标准化 (同一宏观信号在不同资产含义相同) |
| PV 因子 | 截面 Z-score | 沿资产标准化 (不同资产量级差异大) |

## 检测 prompt (给 Agent 的检查清单)

1. **因子标准化方向**:
   - 宏观因子: `X.expanding().mean()` / `.std()` (时间维)
   - PV 因子: `X.T.mean()` / `.std()` (截面维)

2. **PV 因子用时间维**: 量级差异大, 标准化无效

## 正确做法

```python
# 宏观因子: 沿时间
macro_normalized = (macro - macro.mean()) / macro.std()  # 时间维

# PV 因子: 沿截面
pv_normalized = ((pv.T - pv.T.mean()) / pv.T.std()).T  # 截面维
```

## 历史教训来源
- 首次发现: v7.6 数据加载器 (`09a4e35`)