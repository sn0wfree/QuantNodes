# 教训 006: 调仓日当天生效的前视偏差

## 日期
2026-07-28

## 严重度
MODERATE

## 问题描述

`run_v7_3_backtest()` 中权重从调仓日当天开始生效：

```python
mask = (daily_returns.index >= curr_date) & (daily_returns.index < next_date)
```

`>= curr_date` 导致权重在调仓日当天就应用于收益计算。但调仓日当天的价格信息需要收盘后才能观察到，当天无法完成调仓。

## 根因

信号在 `curr_date` 收盘后计算，但权重从 `curr_date` 开始生效 — 等于用了当天的收盘价信息来做当天的交易决策。

## 修复

```python
# 修复前
mask = (daily_returns.index >= curr_date) & ...

# 修复后
mask = (daily_returns.index > curr_date) & ...
```

## 影响

误差约 1 天 / 63 天（季度调仓周期）≈ 1.6%。对 Sharpe 影响不大，但数学上不严谨。

## 防范流程

1. **因果性检查**：信号日期 vs 权重生效日期，确保信号在权重之前
2. **回测框架规范**：调仓日信号 → 下一交易日生效
3. **测试验证**：对比 `>=` 和 `>` 的 NAV 差异
