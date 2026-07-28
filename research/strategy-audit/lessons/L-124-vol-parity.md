---
id: L-124
title: Vol-parity 是性价比最高的 Sharpe 提升手段
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-123]
related_daily: [L-20260724-1]
source: 05_LESSONS_LIBRARY.md
---

# L-124: Vol-parity 是性价比最高的 Sharpe 提升

## 一句话总结
4 策略 Vol-parity 组合 (v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%) OOS Sharpe 1.991 (1.55x 单策略最佳)。

## 问题描述
| 单策略最佳 | 4 策略 Vol-parity | 倍数 |
|-----------|-----------------|------|
| v1.0 locked Sharpe 1.285 | **Sharpe 1.991** | **1.55x** |
| MaxDD -1.94% | -4.41% | -127% (可接受) |

## 检测 prompt (给 Agent 的检查清单)

1. **是否尝试 Vol-parity 组合**:
   - 单策略达到瓶颈后, 必试 Vol-parity
   - target_vol=0.08 是常用值

2. **权重是否合理**:
   - 低相关策略组合效果最佳
   - V10 权重: v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%

## 正确做法

```python
# V10 4 策略 Vol-parity
weights = {
    'v1.0': 0.74,
    'v9macro': 0.12,
    'v7.10': 0.09,
    'DualMom': 0.05,
}
target_vol = 0.08
```

## 历史教训来源
- 首次发现: v10 Vol-parity 4 策略 (`01b4f3c`, 2026-07-24)