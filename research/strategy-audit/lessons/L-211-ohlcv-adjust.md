---
id: L-211
title: OHLCV 各字段必须同口径调整
severity: CRITICAL
auto_checkable: manual
category: data_quality
related_lessons: []
related_daily: [L-20260709-1]
source: 05_LESSONS_LIBRARY.md
---

# L-211: OHLCV 同口径调整

## 一句话总结
修复前 v5 2024 收益 87.36% (虚假), 修复后 35.55% (真实), Calmar +19%。

## 问题描述
拆合股时, 市场只调整价格, 不调整历史价格。
各字段 (open/high/low/close) 必须用同一调整因子, 但 volume 是否乘价格复权因子需要按需选择。

| 修复前 | 修复后 |
|--------|--------|
| v5 2024 收益 87.36% (虚假) | v5 2024 收益 35.55% (真实) |
| v5 Calmar 0.643 | v5 Calmar 0.764 (+19%) |

## 检测 prompt (给 Agent 的检查清单)

1. **OHLCV 是否做了前复权**:
   - 是否运行 `fix_ohlcv_adjust.py`?
   - 复权阈值: 50% 绝对日收益

2. **OHLCV 各字段是否同调整因子**:
   - open/high/low/close 一致

3. **volume 选择**:
   - 价格连续性 vs 真实成交金额

## 正确做法

```python
# 1. 检测拆合股
abs_daily_ret = close.pct_change().abs()
suspect_days = abs_daily_ret > 0.50

# 2. 计算累积调整因子
adj_factor = (close / close.shift(1)) / (1 + ret_no_adj)
cum_adj = adj_factor.cumprod()

# 3. 同字段同调整
for col in ['open', 'high', 'low', 'close']:
    df[col] = df[col] / cum_adj

# 4. volume 可选
df['volume'] = df['volume'] * cum_adj  # 价格连续性优先
```

## 历史教训来源
- 首次发现: v0-V1 时期 9 只 ETF 拆合股检测 (`49d8420`)