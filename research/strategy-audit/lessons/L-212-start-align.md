---
id: L-212
title: 起跑日对齐不要全局硬改，改为各策略独立削平
severity: HIGH
auto_checkable: manual
category: data_quality
related_lessons: []
related_daily: [L-20260709-6]
source: 05_LESSONS_LIBRARY.md
---

# L-212: 起跑日对齐

## 一句话总结
各策略独立削平 + 共同有效区间比较, 不要全局硬改起跑日。

## 问题描述
3 次反复:
| commit | 做法 | 问题 |
|--------|------|------|
| `f0cd21b` | 同图内取"最晚首次交易日"为起跑 | 无法跨图比较 |
| `c52b227` | 全局强制 `2019-04-30` | 隐藏 v5 早段收益 |
| `bb5971d` ⭐ | **各策略独立削平 `trim_flat_prefix()`** | 信息保留完整 |

## 检测 prompt (给 Agent 的检查清单)

1. **业务上推荐**: "各策略独立削平"
2. **公平比较**: 另取共同有效区间
3. **不要"为对齐而对齐"**: 隐藏有效信号

## 正确做法

```python
# 错误: 全局强制起跑日
def align_start(navs, start_date='2019-04-30'):
    return {k: v.loc[start_date:] for k, v in navs.items()}

# 正确: 各策略独立削平
def trim_flat_prefix(nav, min_threshold=1e-4):
    """找到 nav 第一次显著 > 1.0 的日期"""
    first_active = (nav > 1.0 + min_threshold).idxmax()
    return nav.loc[first_active:]

# 公平比较: 另取共同有效区间
def common_period(navs_dict):
    starts = [trim_flat_prefix(n).index[0] for n in navs_dict.values()]
    return max(starts)
```

## 历史教训来源
- 首次发现: 5 commit 链 (`f0cd21b` → `c52b227` → `bb5971d`)