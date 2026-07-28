---
id: L-221
title: 周频特征 + 日频执行的双频体系
severity: HIGH
auto_checkable: agent
category: frequency
related_lessons: []
related_daily: [L-20260714-4]
source: 05_LESSONS_LIBRARY.md
---

# L-221: 双频体系

## 一句话总结
特征和 TV-PR 训练以周频为主, 绩效和执行使用日频。

## 问题描述
```
日频原始行情
  ↓
周频特征 (因子, 目标)
  ↓
TV-PR 周频权重
  ↓
日频 ETF 收益 (考虑交易成本、周一开盘/周五收盘)
  ↓
日频 NAV
```

## 检测 prompt (给 Agent 的检查清单)

1. **混频检测**:
   - 周频信号误用日频权重? (提前一天成交)
   - 日频信号误用周频权重?

2. **双频都重新计算一遍?**:
   - 不要双频都重新计算

## 正确做法

```python
# 周频信号 → 日频执行
weekly_signal = compute_signal(weekly_panel)  # 周频
daily_weights = expand_to_daily(weekly_signal)  # 转换为日频权重
# 关键: daily_weights[t] 必须使用 weekly_signal[t-1] 的信号
#      (避免同日信号 + 同日成交 = look-ahead)
```

## 历史教训来源
- 首次发现: v7.6 (`fc7fc5a`)