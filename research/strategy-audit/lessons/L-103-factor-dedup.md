---
id: L-103
title: 因子去重基于实际相关性，不是理论类别
severity: MEDIUM
auto_checkable: manual
category: methodology
related_lessons: [L-101]
related_daily: [L-20260717-2]
source: 05_LESSONS_LIBRARY.md
---

# L-103: 因子去重基于实际相关性

## 一句话总结
跑相关矩阵而非依赖理论类别, r > 0.93 必删其一。

## 问题描述
39 因子去重到 36 因子。剔除示例:
- `f18_mom_short` ↔ `f21_reversal`, r=-1.000 → 删 f21
- `f3_amt_vol` ↔ `f4_vol_vol`, r=0.958 → 删 f4
- `f8_pv_rankcov` ↔ `f9_pv_corr`, r=0.938 → 删 f9

## 检测 prompt (给 Agent 的检查清单)

1. **是否有相关矩阵检查**: 因子合并前是否跑过 correlation matrix?
2. **阈值**: r > 0.93 应删除其一 (信息冗余, 权重"累加")
3. **删除选择**: 保留 IC 更稳定、含义更清晰的因子

## 正确做法

```python
# 1. 跑相关矩阵
corr_matrix = factor_panel.corr()

# 2. 找出高相关对
high_corr_pairs = []
for i in range(len(factors)):
    for j in range(i+1, len(factors)):
        if abs(corr_matrix.iloc[i, j]) > 0.93:
            high_corr_pairs.append((factors[i], factors[j], corr_matrix.iloc[i, j]))

# 3. 删 IC 更弱的
for f1, f2, corr in high_corr_pairs:
    ic1 = compute_ic(f1)
    ic2 = compute_ic(f2)
    if ic1 < ic2:
        factors.remove(f1)
    else:
        factors.remove(f2)
```

## 历史教训来源
- 首次发现: v7.9 (`db4a852`, 2026-07-17)