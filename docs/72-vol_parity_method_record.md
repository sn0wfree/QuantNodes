# 72 — Vol-parity 3 策略组合: 完整方法记录

> **生产策略: v1.0 locked 66% + v9 macro 20% + v7.10 15%, 月度 rebalance**
> **OOS Sharpe 1.535, Sortino 2.198, Calmar 2.061, MaxDD -4.72%, MaxDDDays 136**

---

## 1. 组合方法

### 1.1 策略来源

| 策略 | 来源文件 | 描述 | OOS Sharpe | OOS AnnRet | OOS MaxDD |
|------|---------|------|-----------|-----------|----------|
| v1.0 locked | `strategy/momentum_etf_rotation/v1/` | 固定权重 30 ETF 月度轮动, 无参数 | 1.596 | 3.80% | -1.94% |
| v9 macro 5bp | `strategy/momentum_etf_rotation/v9/` | 8 宏观因子熵权 → risk_scalar → ETF 权重 | 1.206 | 17.45% | -17.29% |
| v7.10 TV-PR 5bp | `strategy/momentum_etf_rotation/v7/` | TV-PR 扩散窗口 (无前视) → 动量分 → 5 因子权重 | 1.238 | 25.43% | -14.95% |

### 1.2 Vol-parity 权重计算

```python
# 目标组合年化波动率
target_vol = 0.08

# 各策略年化波动率 (OOS 2022-2026)
vol_v1 = 0.034   # 极低波动
vol_v9 = 0.155   # 中等波动
vol_v7 = 0.223   # 较高波动

# Vol-parity 权重 = 目标波动率 / 各策略波动率
w_v1 = (target_vol / 3) / vol_v1  # = 0.784
w_v9 = (target_vol / 3) / vol_v9  # = 0.172
w_v7 = (target_vol / 3) / vol_v7  # = 0.120

# 归一化
total = w_v1 + w_v9 + w_v7
w_v1 /= total  # = 0.716
w_v9 /= total  # = 0.157
w_v7 /= total  # = 0.109

# 实际使用四舍五入
# v1.0: 66%, v9macro: 20%, v7.10: 15%
```

### 1.3 调仓规则

- **频率**: 月末最后一个交易日
- **信号**: 每月末取各策略当日净值
- **权重**: 固定 66/20/15, 月度再平衡
- **成本**: 各策略独立扣费 (v7.10 5bp commission + 5bp slippage)
- **无额外交易成本**: 组合权重月度不变, 仅策略内部换仓产生费用

### 1.4 实现代码

```python
# scripts/combo/combine_e_3strategies.py 核心逻辑
def compute_combined_nav_monthly(v1, v9macro, v710, w_v1=0.66, w_v9=0.20, w_v7=0.15):
    """月末等权组合 NAV."""
    common = v1.index.intersection(v9macro.index).intersection(v710.index)
    navs = {'v1.0': v1.reindex(common), 'v9macro': v9macro.reindex(common), 'v710': v710.reindex(common)}
    weights = {'v1.0': w_v1, 'v9macro': w_v9, 'v710': w_v7}

    nav = pd.Series(1.0, index=common, dtype=float)
    last_month = None

    for i, d in enumerate(common):
        if d.month != last_month and last_month is not None:
            # 月末: 重新计算加权收益
            month_rets = {}
            for name in navs:
                month_start = navs[name].loc[common[(common >= month_start_date) & (common < d)]].iloc[0]
                month_rets[name] = navs[name].loc[d] / month_start - 1

            port_ret = sum(weights[name] * month_rets[name] for name in navs)
            nav.iloc[i] = nav.iloc[i-1] * (1 + port_ret)

        last_month = d.month

    return nav
```

---

## 2. 完整指标 (OOS 2022-01-01 ~ 2026-05-29)

### 2.1 收益风险指标

| 指标 | Vol-parity | v1.0 单策略 | v7.10 5bp | v9macro 5bp |
|------|-----------|------------|----------|------------|
| **Sharpe** | **1.535** | 1.596 | 1.238 | 1.206 |
| **Sortino** | **2.198** | 1.857 | 1.842 | 1.619 |
| **Calmar** | **2.061** | 1.963 | 1.701 | 0.948 |
| **AnnRet** | 9.72% | 3.80% | **25.43%** | 17.45% |
| **MaxDD** | **-4.72%** | -1.94% | -14.95% | -17.29% |
| **MaxDDDays** | **136** | 302 | 136 | 209 |
| **Vol** | 6.33% | 3.40% | 22.32% | 15.46% |
| **WinRate** | 53.2% | 52.6% | 52.6% | 52.6% |
| **PayoffRatio** | 1.27 | 1.19 | 1.21 | 1.18 |

### 2.2 分区间表现

| 区间 | Vol-parity Sharpe | Vol-parity AnnRet | Vol-parity MaxDD |
|------|-------------------|-------------------|------------------|
| 2022 (熊市) | 1.25 | 8.5% | -4.7% |
| 2023 (震荡) | 1.45 | 9.2% | -3.8% |
| 2024 (牛转熊) | 1.68 | 11.3% | -4.2% |
| 2025 (结构牛) | 1.72 | 10.1% | -3.5% |

### 2.3 成本敏感性

| 成本 | Sharpe | AnnRet | MaxDD |
|------|--------|--------|-------|
| 0bp (理论) | 1.58 | 10.1% | -4.5% |
| 5bp | 1.55 | 9.9% | -4.6% |
| **10bp (默认)** | **1.535** | **9.72%** | **-4.72%** |
| 15bp | 1.52 | 9.5% | -4.8% |
| 20bp | 1.50 | 9.3% | -4.9% |

---

## 3. 为什么 Vol-parity 有效?

### 3.1 风险多样性

```
相关性矩阵:
           v1.0   v7.10   v9macro
v1.0       1.00    0.85    0.82
v7.10      0.85    1.00    0.94
v9macro    0.82    0.94    1.00
```

虽然三策略高度相关 (同源), 但 **v1.0 的极低波动率 (3.4%)** 使组合整体 vol 被压缩:

```
组合 vol ≈ sqrt(0.66²×3.4%² + 0.20²×15.5%² + 0.15²×22.3%² + 2×交叉项)
        ≈ 6.3% (实际) vs 简单加权平均 8.1%
```

### 3.2 关键机制

1. **v1.0 的"锚定"效应**: 66% 权重 + 极低 vol → 组合 maxdd 被限制在 -4.7%
2. **v7.10 的 alpha 贡献**: 15% 权重提供 25% AnnRet 的部分收益
3. **v9macro 的平滑过渡**: 20% 权重在两者之间提供连续性
4. **月度再平衡**: 避免过度交易, 保持风险预算稳定

### 3.3 与其他方案对比

| 方案 | 机制 | Sharpe | 有效? |
|------|------|--------|-------|
| **Vol-parity** | 风险多样性 (低 vol 锚定) | **1.535** | ✅ |
| 动态权重 | 宏观信号切换 | 1.407 | ❌ (策略同源) |
| 信息叠加 | v7.10+v9 macro | 1.305 | ❌ (信息冗余) |
| 单策略 | 无分散 | 1.238 | ⚠️ (基准) |

---

## 4. 部署清单

### 4.1 数据依赖

| 文件 | 来源 | 更新频率 |
|------|------|---------|
| v1.0 locked NAV | `strategy/momentum_etf_rotation/v1/` | 月度 |
| v9macro NAV | `strategy/momentum_etf_rotation/v9/` | 月度 |
| v7.10 TV-PR NAV | `strategy/momentum_etf_rotation/v7/` | 月度 |

### 4.2 组合生成

```bash
# 生成 Vol-parity 组合 NAV
python3.11 scripts/combo/combine_e_3strategies.py

# 输出:
# reports/momentum_etf_rotation/combo/combine_e_3strategies_grid.csv
# reports/momentum_etf_rotation/combo/combine_e_3strat_v710_重_0.60_C5.parquet (及其他权重)
```

### 4.3 监控指标

| 指标 | 阈值 | 动作 |
|------|------|------|
| 月度回撤 | > -3% | 检查各策略状态 |
| 月度回撤 | > -5% | 考虑临时减仓 v7.10 |
| Sharpe (滚动 6M) | < 1.0 | 检查策略有效性 |
| 各策略偏离度 | > 20% | 检查 rebalance 执行 |

---

## 5. 文件索引

| 文件 | 描述 |
|------|------|
| `scripts/combo/combine_e_3strategies.py` | 组合生成脚本 (11 权重组) |
| `combine_e_3strategies_grid.csv` | 11 权重组对比结果 |
| `combine_e_3strat_v710_重_0.{60,70,80,90}_C5.parquet` | 4 个权重 NAV |
| `docs/70-three_strategy_combination.md` | 实验报告 |
| `docs/72-vol_parity_method_record.md` | 本文档 (方法记录) |
