---
id: L-105
title: 反转效应在 A 股不存在
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-101]
related_daily: [L-20260709-3]
source: 05_LESSONS_LIBRARY.md
---

# L-105: 反转效应在 A 股不存在

## 一句话总结
A 股 `reversal` 因子 IC 全为负, 不要做反转策略。

## 问题描述
v4 6 因子 IC 诊断: `reversal` / `dividend` 在 A 股是稳定负 alpha。
A 股没有反转效应的推测原因:
- 散户主导 (与美国相反)
- 政策市特征 (政策驱动单边市)
- 流动性差异 (散户交易集中)

## 检测 prompt (给 Agent 的检查清单)

1. **是否有反转策略**:
   - 是否引入 `reversal` 因子作为正向信号?
   - 是否做"低买高卖"的反转策略?

2. **反转因子的 IC 验证**:
   - 计算近 3 年 IC
   - 若 IC 全为负或接近 0, 应剔除

3. **如必须用反转**: 考虑与其他维度组合 (量价背离作为反转代理)

## 正确做法

```python
# 错误: 反转因子作为正向信号
weights = reversal_score  # A 股 IC 全为负

# 正确: 直接跳过反转因子
FACTORS_TO_USE = [f for f in all_factors if f != 'reversal']

# 或: 反转作为过滤条件 (低反转 = 高动量)
weights = momentum_score * (1 - reversal_score)
```

## 历史教训来源
- 首次发现: v4 6 因子诊断 (`e983294`, 2026-07-09)