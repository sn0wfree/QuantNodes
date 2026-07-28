# 教训 005: compute_metrics freq 参数错误

## 日期
2026-07-28

## 严重度
HIGH

## 问题描述

`v7_3_expanded_tf.py` 中对日频 NAV 使用 `freq="W"`：

```python
m = compute_metrics(nav, freq="W")  # 错误: NAV 是日频
```

`compute_metrics` 的 `freq` 参数只影响年化因子 `sqrt(freq)`，不会重采样数据。对日频 NAV 用 `freq="W"` 导致：
- Vol 被低估 `sqrt(252/52) = 2.2x`
- Sharpe 被高估 `2.2x`

## 根因

`freq` 参数的含义是"数据的频率"，不是"想要的输出频率"。日频 NAV 应该用 `freq="D"`。

## 正确做法

```python
# NAV 是日频
m = compute_metrics(nav, freq="D")  # 正确

# NAV 是周频
m = compute_metrics(nav, freq="W")  # 正确
```

## 影响

所有 v7.3 结果的 Sharpe 被高估 2.2x（Calmar/MaxDD/AnnRet 不受影响）。

## 防范流程

1. **检查 NAV 频率**：`nav.index.freq` 或 `pd.infer_freq(nav.index)`
2. **匹配 freq 参数**：日频用 "D"，周频用 "W"
3. **交叉验证**：用两种 freq 计算，确认 Sharpe 比值约等于 `sqrt(252/52)`
