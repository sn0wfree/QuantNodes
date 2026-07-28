# 教训 003: 混合 simple return 和 log return

## 日期
2026-07-28

## 严重度
HIGH

## 问题描述

`load_expanded_panel()` 返回的 DataFrame 中，51 个 ETF 使用 `pct_change()`（simple return），5 个债券指数使用 `np.log(sub / sub.shift(1))`（log return）。两种收益类型混在同一个 DataFrame 中。

## 根因

- ETF 数据源：parquet NAV → `pct_change()` → simple return
- 债券数据源：`v9_indices_daily.parquet`（已预计算的 log return）
- 两个数据源独立加载，未统一收益类型

## 影响

- FRP 优化器输入不一致
- `(1 + ret).cumprod()` 对 log return 不精确
- 年化波动率计算有偏差

## 正确做法

所有资产统一使用同一种收益类型。推荐 simple return（与 NAV 计算 `(1+r).cumprod()` 天然匹配）。

## 防范流程

1. **数据导入统一返回价格/净值**，收益在策略层计算
2. **新函数签名**：`load_aligned_prices()` 返回价格，不返回收益
3. **代码审查**：检查 DataFrame 中各列的收益类型是否一致
