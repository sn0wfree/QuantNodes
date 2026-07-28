---
id: L-132
title: LW 在低维（5 因子）下不显著优于 IC²
severity: MEDIUM
auto_checkable: manual
category: methodology
related_lessons: []
related_daily: [L-20260709-11]
source: 05_LESSONS_LIBRARY.md
---

# L-132: LW vs IC² 在低维场景

## 一句话总结
5 因子场景 IC² > LW 滚动 λ (0.613 > 0.468)。

## 问题描述
LW (Ledoit-Wolf) 收缩在低维场景无显著优势。
触发 LW 启用条件:
1. 因子数 ≥ 8
2. 多信号输入 ≥ 3
3. 因子间相关性 > 0.5

## 检测 prompt (给 Agent 的检查清单)

1. **LW 是否在低维场景被启用**:
   - 因子数 < 8 时, 优先用 IC²
   - LW 仅在因子数 ≥ 8 时启用

2. **当前生产选择**:
   - v7.10 默认关闭 LW, 使用 IC²
   - v10 可能启用 LW (因子数 ≥ 8 时)

## 历史教训来源
- 首次发现: v4 LW 增强 (`e17098c`, 2026-07-09)