# 教训 004: NAV 用 (1+log_return).cumprod()

## 日期
2026-07-28

## 严重度
HIGH

## 问题描述

`run_v7_3_backtest()` 中 NAV 计算：

```python
nav = (1 + all_ret_series).cumprod()
```

`all_ret_series` 是日对数收益。对 log return 做 `(1+r).cumprod()` 在数学上不等于 `exp(cumsum(r))`，两者差异随时间累积。

## 数学对比

| 方法 | 公式 | 年化误差 |
|---|---|---|
| 错误 | `(1 + log_ret).cumprod()` | ~1.2%/年 |
| 正确 | `np.exp(log_ret.cumsum())` | 0 |
| 最佳 | simple return + `(1+r).cumprod()` | 0 |

## 根因

从 log return 到 NAV 的正确转换是 `exp(cumsum(r))`，不是 `(1+r).cumprod()`。后者适用于 simple return。

## 修复

改用 simple return + `(1+r).cumprod()`，与 `compute_metrics` 的 NAV 计算一致。

## 防范流程

1. **明确收益类型**：log return 用 `exp(cumsum)`，simple return 用 `(1+r).cumprod()`
2. **统一使用 simple return**：与 NAV 计算和 `compute_metrics` 天然匹配
3. **数值验证**：对已知输入验证 NAV 计算正确性
