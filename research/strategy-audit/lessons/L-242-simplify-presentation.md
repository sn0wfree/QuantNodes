---
id: L-242
title: 业绩呈现精简度直接影响生产决策
severity: LOW
auto_checkable: manual
category: engineering
related_lessons: []
related_daily: [L-20260724-2]
source: 05_LESSONS_LIBRARY.md
---

# L-242: 业绩呈现精简度

## 一句话总结
9 策略 (含 4 策略组合 + 5 单策略) 让"生产首选"一目了然, 不是 24 策略。

## 问题描述
精简过程:
```
8646b25  移除 v3/v4/v6/中信策略 → 24 策略
6342228  移除v0.1/v0.2/EPO/RRG + 等权基准
a67ccb2  移除v8方案B/DynA/DynB/DynC → 16
1952ad5  移除v5量价和银河因子配置 → 14
c032612  按方案B → 9策略 + 加回v10 DualMom
```

## 检测 prompt (给 Agent 的检查清单)

1. **业绩呈现是否包含失败策略**:
   - 应只展示 5-10 个策略
   - 失败策略下架

## 正确做法

```python
# 推荐: 9 策略 (含 4 策略组合 + 5 单策略)
display_strategies = [
    'v10_4strat_volparity',  # 4 策略 Vol-parity
    'v1.0_locked',           # 极致防御
    'v9macro',               # 宏观择时
    'v7.10_tvpr',            # TV-PR
    'DualMom',               # 跨资产防御
    'v6_tf',                 # TF 风控
    'v5.1',                  # 量价因子
    'v3_baseline',           # baseline
    'equal_weight',          # 等权基准
]
```

## 历史教训来源
- 首次发现: 24 → 16 → 14 → 9 策略精简版 (5 commit 链)