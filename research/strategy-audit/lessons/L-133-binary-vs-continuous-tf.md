---
id: L-133
title: 连续 TF Score 理论优于二值，实测退步
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-101]
related_daily: [L-20260714-1]
source: 05_LESSONS_LIBRARY.md
---

# L-133: 连续 TF Score 退步

## 一句话总结
理论优势不等于实测优势, 二值 MA200 + 50% bear equity 仍是 sweet spot。

## 问题描述
| 配置 | Calmar |
|------|--------|
| 二值 TF | 0.981 |
| 连续 TF Score | **0.317 (-68%)** ❌ |

根因:
- 连续 TF Score 在边界 (TF=0.5 附近) 震荡剧烈, 导致频繁切换
- 二值 TF 有明确冷却期, 减少交易

## 检测 prompt (给 Agent 的检查清单)

1. **TF Score 是二值还是连续**:
   - 警惕"理论更精细"的诱惑
   - 实测前必须有 OOS 回测

## 正确做法

```python
# 二值 TF (推荐)
def trend_filter_binary(prices, ma_window=200):
    ma = prices.rolling(ma_window).mean()
    return (prices > ma).astype(int)  # 1 或 0

# 连续 TF Score (退步, 不推荐)
def trend_filter_continuous(prices, ma_window=200):
    ma = prices.rolling(ma_window).mean()
    ratio = prices / ma
    return (ratio - 1).clip(-0.2, 0.2) / 0.2  # 连续值
```

## 历史教训来源
- 首次发现: v7.5 Step 2 (`f34e4ee`, 2026-07-14)