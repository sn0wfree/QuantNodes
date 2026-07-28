---
id: L-123
title: 粗粒度组合比细粒度多策略更稳
severity: HIGH
auto_checkable: manual
category: methodology
related_lessons: [L-301]
related_daily: [L-20260709-5]
source: 05_LESSONS_LIBRARY.md
---

# L-123: 粗粒度组合比细粒度多策略更稳

## 一句话总结
V3 多策略 1/N 等权失败, 粗粒度组合 (v3 80% + v5 20%) 更稳。

## 问题描述
V3 Stage 16A 多策略主回测:
- 51 单测 100% 通过
- 6 个 v3 模块全实现 (动量 + 反转 + 行业轮动)
- **Calmar 0.504 < V2 0.892** (-0.39)
- 1/N 等权让动量优势被稀释
- 反转/行业轮动在趋势市反向拖累

原理:
- 1/N 等权过度对称, 实际动量应得更多
- 子策略 NAV 跟不全时, signal/risk_parity 权重法失效
- 细粒度多策略需要子策略 NAV 完美对齐

## 检测 prompt (给 Agent 的检查清单)

1. **是否使用 1/N 等权**:
   - 子策略数量 > 3 时, 1/N 等权往往不是最优
   - 检查权重分配是否合理

2. **是否用细粒度组合**:
   - 子策略 NAV 是否完美对齐?
   - 若否, 改用粗粒度组合

## 正确做法

```python
# 错误: 1/N 等权
weights = {sub: 1/n_subs for sub in sub_strategies}

# 正确 1: 粗粒度组合
weights = {'v1.0': 0.80, 'v5.1': 0.20}

# 正确 2: Vol-parity (target_vol)
weights = vol_parity_weights(sub_navs, target_vol=0.08)
```

## 历史教训来源
- 首次发现: V3 1/3 等权失败 (`e12070c`, 2026-07-09)