# 教训 008: 数据管道设计原则

## 日期
2026-07-28

## 严重度
HIGH

## 问题描述

v7.3 的数据管道存在多个问题：
- `load_index_panel()` 返回收益而非价格
- `load_expanded_panel()` 混合收益类型
- `load_factor_returns()` 返回收益而非净值
- 各函数对同一数据的处理方式不一致

## 核心原则

**数据导入只返回价格/净值，不返回收益。收益计算在策略层完成。**

## 正确设计

```python
# 数据层：只返回价格/净值
def load_aligned_prices(pool="index") -> dict:
    return {
        "asset_prices": DataFrame,   # 日频价格
        "factor_nav": DataFrame,     # 周频因子净值
        "benchmark": Series,         # 日频基准价格
    }

# 策略层：从价格计算收益
prices = load_aligned_prices()["asset_prices"]
weekly_returns = prices.resample("W").last().pct_change()
daily_returns = prices.pct_change()
```

## 好处

1. **消除歧义**：调用方明确知道返回的是价格
2. **统一计算**：收益计算在一处完成，不会不一致
3. **易于测试**：价格和收益可以分别验证

## 防范流程

1. **新函数签名审查**：返回类型是否包含"returns"
2. **数据流图**：画出数据从源到使用的完整路径
3. **类型标注**：函数签名中明确标注返回类型
