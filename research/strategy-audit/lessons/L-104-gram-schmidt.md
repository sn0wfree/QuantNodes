---
id: L-104
title: Gram-Schmidt 残差化有效，QR 对称正交失败
severity: CRITICAL
auto_checkable: agent
category: methodology
related_lessons: [L-109]
related_daily: [L-20260717-1]
source: 05_LESSONS_LIBRARY.md
---

# L-104: Gram-Schmidt 残差化有效，QR 对称正交失败

## 一句话总结
正交化用 Gram-Schmidt 残差化 (金融预定义顺序), 绝不用对称正交。

## 问题描述
| 方法 | OOS Calmar | 备注 |
|------|-----------|------|
| v6.2 Gram-Schmidt (预定顺序) | **0.901** ⭐ | 保留金融意义 |
| v6.2 QR 对称正交 | **0.056** ❌ | 顺序无关但失去金融解释 |
| v7.9 Symmetry 正交化 | Sharpe 1.40→0.15 (-91%) ❌ | 等方差抹杀相对强度 |

## 检测 prompt (给 Agent 的检查清单)

审查正交化代码时, 检查:

1. **正交化方法是否保留金融意义**:
   - 是否有"预定义因子顺序" (动量→反转→多空→量价)?
   - 还是无差别的"对称正交"?

2. **是否使用 Symmetry / 等方差变换**:
   ```python
   # ❌ 错误 (Symmetry)
   X_orth = X / X.std(axis=0)  # 等方差缩放, 抹杀相对强度

   # ✅ 正确 (Gram-Schmidt 残差化)
   X_orth = gram_schmidt(X, order=['momentum', 'reversal', 'lsm', 'pv'])
   ```

3. **ADMM L1 罚项是否依赖原始尺度**:
   - 若因子已等方差, Lasso 退化为 Ridge
   - 因子选择能力丧失

## 正确做法

```python
def gram_schmidt_residualize(X, factor_order):
    """按金融预定义顺序做 Gram-Schmidt 残差化."""
    X_resid = X.copy()
    for i, factor_name in enumerate(factor_order):
        if i == 0:
            continue
        # 对之前所有因子做回归, 取残差
        for j in range(i):
            beta = np.cov(X_resid[:, i], X_resid[:, j])[0, 1] / np.var(X_resid[:, j])
            X_resid[:, i] -= beta * X_resid[:, j]
    return X_resid
```

## 关联代码案例

- v6.2 Gram-Schmidt: Calmar 0.901 → 后续 CV% 56.9% DEPRECATED
- v7.9 Symmetry: Sharpe -91% → 完全放弃

## 历史教训来源
- 首次发现: v6.2 Gram-Schmidt (`d7fac94`) + v7.9 Symmetry (`db4a852`)