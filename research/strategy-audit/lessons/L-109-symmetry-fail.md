---
id: L-109
title: Symmetry 正交化破坏因子信息结构
severity: CRITICAL
auto_checkable: agent
category: methodology
related_lessons: [L-104]
related_daily: [L-20260717-1]
source: 05_LESSONS_LIBRARY.md
---

# L-109: Symmetry 正交化破坏因子信息结构

## 一句话总结
等方差变换抹杀相对强度, ADMM 的 L1 罚项依赖原始尺度, 绝不用 Symmetry。

## 问题描述
v7.9 应用 Symmetry 正交化前 Sharpe 1.40, 应用后 Sharpe 0.15 (-91%)。
原理:
- 等方差变换抹杀相对强度
- ADMM 的 L1 罚项依赖原始尺度做因子选择
- 看似"数学干净", 实则让时变 β 估计失效

## 检测 prompt (给 Agent 的检查清单)

1. **正交化是否做"等方差"变换**:
   ```python
   # ❌ 错误 (Symmetry / 等方差)
   X_orth = X / X.std(axis=0)  # 所有因子同方差

   # ❌ 错误 (标准化后再做正交)
   X_orth = StandardScaler().fit_transform(X)  # 仍是 Symmetry 思路
   ```

2. **是否保留原始因子尺度**:
   - 只做因子去重 (删高度相关)
   - log 变换 (处理偏态量纲)
   - 保留原始尺度

## 正确做法

```python
# 错误: Symmetry
X_orth = X / X.std(axis=0)

# 正确: 只做因子去重 + log 变换 + 保留原始尺度
X_dedup = remove_high_corr(X, threshold=0.93)
X_log = np.log1p(X_dedup)  # 处理偏态
# 保留原始尺度, 不再做变换
```

## 历史教训来源
- 首次发现: v7.9 (`db4a852`, 2026-07-17)
- 状态: v7.9 DEPRECATED