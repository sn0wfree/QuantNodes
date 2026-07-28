# 教训 001: 数据探索中的指标选择错误

## 日期
2026-07-28

## 严重度
HIGH

## 问题描述

在分析 expanded panel (56 ETF) 的可用性时，使用了错误的判断指标：

```python
# 错误写法
valid_count = (subset.notna().all(axis=0)).sum()
```

`notna().all()` 要求**整列零 NaN**，排除了所有有中国节假日 NaN 的 ETF。结果输出 "2018年: 5/56 资产完全可用"，实际应为 31/56。

## 根因

| 阶段 | 问题 |
|---|---|
| 指标选择 | `notna().all()` 要求零NaN，不是"有数据" |
| 结果解读 | "5/56" 被解读为"只有5只ETF可用" |
| 方案设计 | 基于错误结论设计了不必要的"动态资产池" |

## 正确做法

```python
# 判断"有数据"：至少有1天非NaN
available = subset.notna().any(axis=0).sum()

# 判断"质量"：NaN比例
nan_ratio = subset.isna().mean()
```

## 防范流程

1. **数据探索三板斧**：缺失值统计 → 抽样验证 → 结论自查
2. **禁止**：用 `notna().all()` 判断"可用"
3. **每次结论后**：抽查 3-5 个具体数据点

## 相关代码
- `v7/data_loader.py` - `load_aligned_prices()`
- `v7/macro_substrategy_v7_3.py` - `_filter_beta_with_diversification()`
