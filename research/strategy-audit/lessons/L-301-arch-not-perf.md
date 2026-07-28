---
id: L-301
title: 架构先进 ≠ 业绩进步
severity: HIGH
auto_checkable: manual
category: decision
related_lessons: [L-123, L-303]
related_daily: [L-20260709-5]
source: 05_LESSONS_LIBRARY.md
---

# L-301: 架构先进 ≠ 业绩进步

## 一句话总结
51 单测 100% 通过, 6 个 v3 模块全实现, 但 Calmar 0.504 < V2 0.892 (-0.39)。

## 问题描述
V3 教训:
- 51 单测 100% 通过
- 6 个 v3 模块全实现 (动量 + 反转 + 行业轮动)
- **Calmar 0.504 < V2 0.892** (-0.39)
- 1/N 等权让动量优势被稀释
- 反转/行业轮动在趋势市反向拖累

## 检测 prompt (给 Agent 的检查清单)

1. **架构改造前是否有"业绩基线对比"**:
   - 没有数字优势不重构

2. **单测通过 ≠ 策略好**:
   - 必须有 OOS Calmar 指标

## 正确做法

```python
# 重构前: 跑 baseline
baseline_calmar = run_oos(V2_baseline)  # 0.892

# 重构后: 对比
new_calmar = run_oos(V3_new)  # 0.504

if new_calmar < baseline_calmar:
    raise ValueError(f"V3 Calmar {new_calmar} < V2 {baseline_calmar}, 不要重构")
```

## 历史教训来源
- 首次发现: V3 验证报告 (`da7f588`, 2026-07-09)