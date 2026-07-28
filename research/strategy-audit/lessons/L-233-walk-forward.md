---
id: L-233
title: walk_forward 框架通用化是 OOS 的基石
severity: MEDIUM
auto_checkable: manual
category: oos_validation
related_lessons: []
related_daily: [L-20260714-5]
source: 05_LESSONS_LIBRARY.md
---

# L-233: walk_forward 框架

## 一句话总结
策略无关 + NO LOOKAHEAD 是 walk_forward 的核心。

## 问题描述
```python
# 提供 backtest_fn(Y_train, X_train, **params) → (weights, returns)
# β 估计只用训练数据 (Y_train = Y.iloc[:test_start])
def backtest_fn(Y_train, X_train, **params):
    # 只用 Y_train 训练
    beta = estimate_beta(X_train, Y_train)
    # 测试期用 Y_test 应用权重
    weights = compute_weights(beta, X_test)
    return weights, Y_test
```

## 检测 prompt (给 Agent 的检查清单)

1. **walk_forward 是否 NO LOOKAHEAD**:
   - β 估计只用训练数据
   - 测试期不能泄漏训练信息

2. **warm-start**:
   - 是否从训练期最后状态开始?

## 正确做法

```python
def walk_forward(data, backtest_fn, n_splits=5):
    """Walk-forward OOS validation."""
    splitter = TimeSeriesSplit(n_splits=n_splits)
    oos_results = []

    for train_idx, test_idx in splitter.split(data):
        Y_train = data['Y'].iloc[train_idx]
        X_train = data['X'].iloc[train_idx]
        X_test = data['X'].iloc[test_idx]

        weights, returns = backtest_fn(Y_train, X_train)
        oos_results.append((weights, returns))

    return oos_results
```

## 历史教训来源
- 首次发现: `826db4d` (`common/walk_forward.py`, 990 行)