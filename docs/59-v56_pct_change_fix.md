# v56 数据修正 + 所有策略重跑 - 最终报告

> **日期**: 2026-07-23
> **改动**: `QuantNodes/strategy/momentum_etf_rotation/v7/data_loader.py:309` 从对数收益改为简单收益

---

## 一、问题根因

`v56_expanded_daily.parquet` 生成代码使用对数收益:
```python
# data_loader.py:309 (原)
etf_rets = np.log(etf_nav / etf_nav.shift(1))  # 对数收益
```

而 `v7_6_daily_etf_returns.parquet` 使用简单收益:
```python
# data_loader_v7_6.py:405
daily_returns = nav.pct_change()  # 简单收益
```

**业界回测标准做法**是 `pct_change()`,但 v56 数据生成时错误地使用了 `np.log()`。

---

## 二、修改内容

### 2.1 修改 `data_loader.py`

```python
# 修正后 (data_loader.py:309)
etf_rets = etf_nav.pct_change().dropna(how="all")  # 简单收益
```

### 2.2 修改 v8 backtest 中国假期处理

`v8/integration.py:_compute_daily_nav_from_weights` 和 `scripts/v8_integrated_comparison.py`:
- 原来: 中国假期 fillna(0) 当 0% 收益
- 现在: 中国假期 ETF 数据全 NaN 时, 跳过该日 (与 v7_6 行为一致)

```python
row = daily_returns.loc[d]
if row[common_codes].isna().all():
    nav.iloc[i] = nav.iloc[i - 1]  # 跳过中国假期
else:
    ret = row.fillna(0.0)
    ...
```

### 2.3 修改 v7.14 backtest (`scripts/combo/regenerate_v7_14_nav_with_v56.py`)

同样添加中国假期跳过逻辑。

---

## 三、验证

### 3.1 数据一致性验证

| ETF | v7_6 均值 | v56 (新) 均值 | 差异 |
|-----|-----------|--------------|------|
| 510300 | 0.000273 | 0.000265 | 0.000008 |
| 510500 | 0.000349 | 0.000309 | 0.000040 |
| 510050 | 0.000187 | 0.000174 | 0.000013 |

均值差异极小 (< 1e-4), 主要来自公共日期的子集差异。

### 3.2 v7.14 公共日期验证

| 指标 | 旧 (v7_6) | 新 (v56) |
|------|----------|----------|
| **公共日期 Sharpe** | **0.9585** | **0.9585** ✅ |
| 公共日期相关系数 | 1.000000 | 1.000000 |
| 公共日期 NAV 差异 | 0 | 0 ✅ |

**完全一致!** 剩余差异仅来自新数据多包含的 93 个中国假期日 (0% 收益)。

---

## 四、最终公平对比 (OOS 2021-08 ~ 2026-05/06)

### 4.1 所有策略排名 (修正后)

| 排名 | 策略 | Sharpe | Calmar | AnnRet | MaxDD |
|------|------|--------|--------|--------|-------|
| 1 | **v8 Jump Model 方案B** | **1.09** | **1.018** | 12.94% | **-12.71%** ⭐ |
| 2 | v7.10 TV-PR (标准化) | 1.03 | 1.109 | 21.16% | -19.08% |
| 3 | v8 Jump Model 优化版 (旧数据) | 1.00 | 1.032 | 13.78% | -13.35% |
| 4 | v7.14 TV-PR (修正) | 0.96 | 0.640 | 16.68% | -28.31% |
| 5 | 银河方案-动态仓位 | 0.88 | 0.633 | 9.97% | -15.74% |
| 6 | 银河因子配置 | 0.59 | 0.337 | 6.93% | -20.55% |
| 7 | 中信大类资产配置 | 0.50 | 0.392 | 7.18% | -18.30% |
| 8 | 中信多因子选股 | 0.46 | 0.367 | 6.42% | -17.50% |
| 9 | 中信里昂全天候 | 0.32 | 0.225 | 4.78% | -21.24% |
| 10 | 基础风险平价 | 0.27 | 0.166 | 4.32% | -26.06% |
| 11 | 等权基准 | 0.21 | 0.128 | 3.39% | -26.51% |
| 12 | 60/40股债 | 0.15 | 0.075 | 2.55% | -33.96% |
| 13 | 中信行业轮动 | 0.07 | 0.063 | 1.87% | -29.55% |

### 4.2 关键修正

| 策略 | 修正前 (虚高) | 修正后 (真实) | 差异 |
|------|---------------|---------------|------|
| v8_method_b | 1.126 | **1.09** | -0.03 (-3%) ✅ |
| v7.14 TV-PR | 1.006 | **0.96** | -0.05 (-5%) ✅ |
| v9 银河方案 | 0.464 | **0.88** | +0.42 (+90%) ✅ |

**关键修正**: v9 银河方案 Sharpe 从 0.464 跳到 0.88, 之前是因为对数收益导致严重低估!

### 4.3 v8 vs v9 最终对比

| 指标 | v8_method_b (v56, 修正) | v9 银河方案 (v56, 修正) |
|------|--------------------------|---------------------------|
| **Sharpe** | **1.09** | 0.88 |
| **Calmar** | **1.018** | 0.633 |
| AnnRet | 12.94% | 9.97% |
| MaxDD | **-12.71%** | -15.74% |

**v8 仍然领先, 但 v9 也不差!**

---

## 五、产出文件

| 操作 | 文件 |
|------|------|
| 修改 | `QuantNodes/strategy/momentum_etf_rotation/v7/data_loader.py` |
| 修改 | `QuantNodes/strategy/momentum_etf_rotation/v8/integration.py` (中国假期处理) |
| 修改 | `scripts/v8_integrated_comparison.py` (中国假期处理) |
| 修改 | `scripts/combo/regenerate_v7_14_nav_with_v56.py` (中国假期处理) |
| 重生成 | `data/high_freq_macro/v56_expanded_daily.parquet` |
| 重跑 | `reports/momentum_etf_rotation/combo/v8_method_b_nav_v56.parquet` |
| 重跑 | `reports/momentum_etf_rotation/combo/v8_all_navs_v56.parquet` |
| 重跑 | `reports/momentum_etf_rotation/combo/v7_14_nav_v56.parquet` |
| 重跑 | `reports/momentum_etf_rotation/combo/v9_navs.parquet` |
| 重跑 | `reports/momentum_etf_rotation/combo/STRATEGY_ITERATION_RECORD.html` |

---

## 六、核心结论

### 6.1 之前的错误结论 (已修正)

| 错误结论 | 修正 |
|---------|------|
| "v8 Sharpe 1.126 是最优" | 真实 1.09 |
| "v9 Sharpe 0.464 比 v8 弱很多" | 真实 0.88,差距缩小 |
| "v9 用对数收益是 bug" | 是的,已修正 |

### 6.2 真正的策略排名 (统一 v56 数据, 简单收益)

| 排名 | 策略 | Sharpe | 特点 |
|------|------|--------|------|
| 🥇 | **v8 Jump Model 方案B** | **1.09** | Sharpe+Calmar+MaxDD 三冠王 |
| 🥈 | v7.10 TV-PR (标准化) | 1.03 | 年化最高 21% |
| 🥉 | v7.14 TV-PR (修正) | 0.96 | 稳定但 MaxDD 较大 |
| 4 | v9 银河方案-动态仓位 | 0.88 | 真正的 alpha 来自仓位 |
| 5 | v8 Jump Model 优化版 | 1.00 | (旧数据, 实际值待重跑) |

### 6.3 给 v11_mega 的建议

**v8 Jump Model 方案B 仍然是 OOS 最佳 (Sharpe 1.09, MaxDD -12.71%)**。

v11_mega 设计应该:
- **Layer 风险控制**: 借鉴 v8 Jump Model (牛熊检测 + 仓位调节)
- **Layer 选股**: 借鉴 v7.14 TV-PR (量价因子 + TV-PR)
- **Layer 因子**: 借鉴 v9 银河方案 (动态仓位)

---

## 七、待处理事项

1. **v8 Jump Model 优化版 (v8_optimized_nav)**: 仍是旧数据,需要重跑
2. **回归测试**: 修改 `data_loader.py` 后, 应该跑 `pytest tests/` 验证
3. **v9 子策略指标更新**: docs/53 中 v9 指标需要根据新数据更新

需要继续处理 v8 优化版吗?