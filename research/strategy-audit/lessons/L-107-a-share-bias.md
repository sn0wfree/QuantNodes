---
id: L-107
title: A 股动量信号偏好海外/商品，A 股低配是结构性
severity: MEDIUM
auto_checkable: manual
category: methodology
related_lessons: []
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-107: A 股动量信号偏好海外/商品

## 一句话总结
A 股动量在熊市天然低配, 这是结构性, 不要试图通过参数调整修复。

## 问题描述
- 2018-2024 期间海外/商品 vs 沪深300 涨幅 168.6% vs 67.9%
- v1.0 9 月底调仓时 A 股宽基仅 2.68%
- 即便 cap 放宽到 6 也只是"少配"而非"配对"

## 检测 prompt (给 Agent 的检查清单)

1. **动量信号是否结构性偏向海外/商品**:
   - 检查动量 top_n 在不同 regime 下的资产分布
   - 在熊市中 A 股是否被低估

2. **不要试图通过参数调整修复**:
   - 提高 max_weight 不解决问题
   - 真正解决: 换信号源 (反转/行业轮动/RSRS/HMM)
   - 或扩展决策上下文 (多策略组合 + 反向因子)

## 历史教训来源
- 首次发现: Stage 14 924 归因 (`c2374f8`, 2026-07-09)