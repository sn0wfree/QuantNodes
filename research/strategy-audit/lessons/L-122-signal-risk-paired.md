---
id: L-122
title: "信号 + 风控" 不可独立可加，必须成对测试
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: []
related_daily: []
source: 05_LESSONS_LIBRARY.md
---

# L-122: 信号 + 风控不可独立可加

## 一句话总结
signal × 风控可能冲突, 必须消融 7 档 (无 / 单层 / 双层 / 全风控)。

## 问题描述
经典失败案例:
- slope_r2 + VT → Calmar 0.92 ❌ (VT 缩放对 slope_r2 高频信号冲突)
- v6.1 + VT → OOS 0.557 ❌ (进攻型信号被 VT 压制)

## 检测 prompt (给 Agent 的检查清单)

1. **是否做了 7 档消融**:
   - 无风控 / 单层 (VT/TF/SL) / 双层组合 / 全风控

2. **每档的 OOS 指标记录**:
   - Calmar / MaxDD / AnnRet
   - 推荐档写在 V6Config.use_* 字段

## 正确做法

```python
# 7 档消融
configs = [
    {'name': 'no_risk', 'vt': False, 'tf': False, 'sl': False},
    {'name': 'vt_only', 'vt': True, 'tf': False, 'sl': False},
    {'name': 'tf_only', 'vt': False, 'tf': True, 'sl': False},
    {'name': 'sl_only', 'vt': False, 'tf': False, 'sl': True},
    {'name': 'vt_tf', 'vt': True, 'tf': True, 'sl': False},
    {'name': 'vt_sl', 'vt': True, 'tf': False, 'sl': True},
    {'name': 'all', 'vt': True, 'tf': True, 'sl': True},
]
for cfg in configs:
    test_oos(cfg)  # 每档 OOS 测试
```

## 历史教训来源
- 首次发现: Stage 12A + VT (`07956ca` + `9e52eb4`)