---
id: L-304
title: 起点依赖测试是 P0 任务（不是 nice-to-have）
severity: HIGH
auto_checkable: manual
category: decision
related_lessons: [L-203]
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-304: 起点依赖测试是 P0

## 一句话总结
CV% 测试应作为任何策略的硬性 P0 任务, 不是 nice-to-have。

## 问题描述
v6.2 CV% 56.9% FAIL (`223ef65`) → v7.10 Stage 32 把 CV% 测试列为硬性 P0。

## 检测 prompt (给 Agent 的检查清单)

1. **CV% 测试是否被列为 P0**:
   - 任何策略上 P0 任务前必须跑 CV% 测试
   - 阈值固定: 25%

## 正确做法

```python
# 任何新策略 P0 任务清单:
# 1. 数据闸门
# 2. IC 闸门
# 3. 因子闸门
# 4. OOS 闸门 (含 CV% < 25%)
# 5. 硬化闸门
```

## 历史教训来源
- 首次发现: v6.2 CV% 56.9% FAIL (`223ef65`) → v7.10 Stage 32