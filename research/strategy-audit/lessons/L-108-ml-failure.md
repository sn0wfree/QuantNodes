---
id: L-108
title: 树模型在 ETF 横截面上的预测力约等于 0
severity: CRITICAL
auto_checkable: manual
category: methodology
related_lessons: [L-101]
related_daily: [L-20260718-1]
source: 05_LESSONS_LIBRARY.md
---

# L-108: 树模型在 ETF 横截面上的预测力约等于 0

## 一句话总结
ML 路线失败: 修复 look-ahead 后, 树/线性模型 R² ≈ 0, 核心问题在因子不在模型。

## 问题描述
| 模型 | 修复前 R² | 修复 look-ahead 后 |
|------|---------|-------------------|
| 树模型 (LightGBM/RF/CatBoost) | 0.40 | **≈ 0** |
| 线性 (Lasso/Ridge/Huber/GBR) | 0.30 | **≈ 0** |
| MLP | 0.20 | **< 0** |

## 检测 prompt (给 Agent 的检查清单)

1. **是否有 ML 模型被引入**:
   - 树模型 (LightGBM / XGBoost / RandomForest / CatBoost)
   - 神经网络 (MLP / Transformer)

2. **ML 模型 R² 是否 < 0.05**:
   - 若修复 look-ahead 后 R² ≈ 0, 应 DEPRECATED
   - 核心问题在因子 (IC 噪声), 不在模型

3. **未来路线**:
   - 不要尝试纯 ML 模型
   - 改走 "宏观择时 + 半衰期短因子" 路线

## 正确做法

```python
# 错误: 引入树模型
model = lgb.LGBMRegressor()
model.fit(X, Y)  # Y 含 look-ahead → R² 0.40 (虚假)

# 修复 look-ahead 后:
model.fit(X, Y_real)  # R² ≈ 0, 模型无预测力

# 正确: 宏观择时 + 半衰期短因子
signal = macro_regime * short_term_momentum
weights = compute_weights_from_signal(signal)
```

## 历史教训来源
- 首次发现: v7.7 Phase 1 (`a5de7f3` + `f22c407`, 2026-07-18~19)
- 状态: v7.7 DEPRECATED