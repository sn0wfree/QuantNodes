# 教训 002: 对收益数据做 resample.pct_change

## 日期
2026-07-28

## 严重度
CRITICAL

## 问题描述

`run_v7_3_backtest()` 中对收益数据做周频化：

```python
# 错误写法
idx_weekly = index_panel.resample("W").last().pct_change()
```

`index_panel` 是日对数收益，不是价格。`resample("W").last()` 取每周最后一个交易日的日收益值，`.pct_change()` 对这个值做百分比变化 — 完全无意义。

## 根因

源 notebook 的 `main_idx.resample('W').last().pct_change(1)` 操作的是**价格水平**。`load_index_panel()` 被重构为返回对数收益后，`run_v7_3_backtest` 仍沿用原写法，但语义已完全改变。

## 正确做法

```python
# 如果输入是价格
asset_weekly = prices.resample("W").last().pct_change()

# 如果输入是日收益
asset_weekly = daily_returns.resample("W").sum()  # 对数收益求和=周收益
```

## 影响

- Lasso β 几乎全零（拟合噪声）
- 宏观因子信号完全失效
- 策略表现完全依赖等权兜底

## 防范流程

1. **明确数据类型**：每次使用 `resample().pct_change()` 前确认输入是价格还是收益
2. **收益数据的周频化**：用 `sum()`（对数收益）或 `(1+r).prod()-1`（简单收益）
3. **单元测试**：对已知输入/输出的 case 验证转换正确性
