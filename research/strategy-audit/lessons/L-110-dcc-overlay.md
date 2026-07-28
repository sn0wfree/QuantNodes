---
id: L-110
title: DCC regime overlay 是危机预警最稳定信号
severity: MEDIUM
auto_checkable: agent
category: methodology
related_lessons: []
related_daily: [L-20260720-3]
source: 05_LESSONS_LIBRARY.md
---

# L-110: DCC regime overlay

## 一句话总结
dcc_zscore_mean > 1.5 触发 crisis 防御 (reduce_factor=0.5, cooldown=4 周)。

## 问题描述
DCC overlay 反映"整个市场相关性结构突变", 比"HS300 < MA200"更有结构性意义。
多次实证 2022 H1 bear_combo 71 天识别。

## 检测 prompt (给 Agent 的检查清单)

1. **regime overlay 是否使用 DCC**:
   - 是否计算动态相关性 (DCC)?
   - 是否用 `dcc_zscore_mean > 1.5` 作为 crisis 触发?

2. **cooldown 机制**:
   - 是否有 4 周冷却期?
   - 避免震荡市来回触发

3. **DCC overlay 是否作为标配**:
   - 应作为所有 TV-PR 系策略的标配

## 正确做法

```python
# DCC overlay 实现
if dcc_zscore_mean(t) > 1.5:
    weights *= 0.5  # reduce_factor
    cooldown_until = t + 4  # 4 周冷却

# 在 TV-PR 主循环中
for t in range(window, T):
    if t < cooldown_until:
        continue  # 冷却期跳过
    if dcc_zscore_mean(t) > 1.5:
        weights[t] *= 0.5
        cooldown_until = t + 4
```

## 历史教训来源
- 首次发现: v7.12 (`2ac6b33`, 2026-07-20)