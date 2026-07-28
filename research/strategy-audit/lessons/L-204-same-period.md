---
id: L-204
title: X[t]→Y[t+1] 信号-执行同期是常见陷阱
severity: CRITICAL
auto_checkable: agent
category: lookahead
related_lessons: [L-201, L-205]
related_daily: [L-20260715-1, L-20260719-1]
source: 05_LESSONS_LIBRARY.md
---

# L-204: X[t]→Y[t+1] 同期陷阱

## 一句话总结
训练时必须有明确的"t 定义", 写代码注释: `X[t] → Y[t+1]`。

## 问题描述
```python
# 错误: 用 t 周收益训练, 预测 t+1 周
Y[t] = (NAV[t] / NAV[t-1]) - 1  # 默认 t-1 到 t
# 但 (X[t], Y[t]) → 这是同期训练, look-ahead!

# 正确: 用 t-1 周收益训练, 预测 t 周
Y[t] = (NAV[t] / NAV[t-1]) - 1  # t-1 → t
X[t-1] → Y[t]  # 严格下一期
```

## 检测 prompt (给 Agent 的检查清单)

审查训练代码时, 检查:

1. **训练时 X 和 Y 的时序**:
   - X[t] 应来自 t 之前的数据
   - Y[t+1] 应该是 t+1 期的目标
   - 若 X[t] 和 Y[t] 同期 → VIOLATED

2. **代码注释**:
   - 是否有 `# X[t] → Y[t+1]` 注释?
   - 若无, 应要求添加

3. **last-out-of-sample 测试**:
   - 是否对每个模型做 last-out-of-sample 测试?

## 正确做法

```python
# 训练数据准备
def prepare_training_data(X, Y):
    """X[t] → Y[t+1] (严格下一期)"""
    X_train = X.iloc[:-1]  # t=0..T-1
    Y_train = Y.iloc[1:]   # t=1..T (下一期目标)
    return X_train, Y_train

# 模型训练
model = Lasso(alpha=0.05)
model.fit(X_train, Y_train)

# 预测 (用 X[T] 预测 Y[T+1])
Y_pred = model.predict(X.iloc[-1:])
```

## 关联代码案例

- v7.7: `f18_mom_short[t]` 和 `Y[t]` 完全重叠, corr=0.96
- v7.6: 6 个未来函数 bug, 详见 L-20260715-1

## 历史教训来源
- 首次发现: v7.6 未来函数 6 Bug (`0c1c6a4`, 2026-07-15)