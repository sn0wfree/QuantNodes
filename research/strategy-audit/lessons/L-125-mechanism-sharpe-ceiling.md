---
id: L-125
title: 5 大机制 Sharpe 区间有自然边界
severity: MEDIUM
auto_checkable: manual
category: methodology
related_lessons: []
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-125: 5 大机制 Sharpe 区间

## 一句话总结
每种机制都有 Sharpe 天花板, 不要期望超额。

## 问题描述
| 机制 | Sharpe 区间 | 备注 |
|------|-------------|------|
| A. 风险配置 (RP / 60-40) | 0.20 - 0.35 | 提供底线 |
| B. 宏观择时 (HMM / DCC / 期现利差) | 0.25 - 0.39 | 提供 alpha |
| C. 横截面选股 (多因子 / 11 量价) | **0.62** ⭐ | 中信多因子最高 |
| D. 行业轮动 (动量/反转/相关) | 0.28 | 中游 |
| E. 动态仓位 (Jump Model / 银河方案) | **1.23** ⭐ | 最大 alpha 来源 |

## 检测 prompt (给 Agent 的检查清单)

1. **是否期望 C 类机制做到 Sharpe 1.0**:
   - 不可能, C 类天花板 0.62

2. **E 类 (动态仓位) 才是真正的 alpha 突破口**:
   - 71% alpha 来源

## 历史教训来源
- 首次发现: v9 Brinson 归因 (`51-v9_brinson_attribution.md`)