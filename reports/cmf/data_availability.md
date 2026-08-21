# CMF Phase 6 — 数据补齐报告

## 2026-08-20 EIA + FRED + DXY 接入结果

### 新增数据 (3 项)

| 因子 | 数据源 | Series ID / 接口 | 时间范围 | 频率 | 行数 | 用途 |
|---|---|---|---|---|---|---|
| 美国商业原油库存 | EIA Open Data API v2 | `WCRSTUS1` | 1982-08-20 ~ 2026-08-14 | 周 | 2,290 | 原油§3.1库存象限 + §2.2基本面中枢 |
| ICE DXY 美元指数 | akshare 合成 (currency_boc_safe) | EURUSD + USDJPY + GBPUSD + USDCAD + USDSEK + USDCHF | 2016-12-12 ~ 2026-08-20 | 日 | 2,355 | §2.2 宏观对冲 (USD↓利多商品) |
| TIPS 10Y 实际利率 | FRED API (fredapi 库) | `DFII10` | 2003-01-02 ~ 2026-08-18 | 日 | 5,911 | §2.2 实际利率 (实际利率↓利多商品) |

### 数据写入

- 新表 `eia_weekly_inventory` (trade_date, eia_crude_stocks 千桶)
- `fundamentals_daily` 长表新增 value_col:
  - `dxy` (source=`akshare_currency_boc`)
  - `tips10y` (source=`fred_dfii10`)

### Pipeline 改造

**原油 (SC0) 分支**:
- 旧: `_eia_cum` 月度变动量累加 (误差大)
- 新: `_eia_weekly_stocks` EIA WCRSTUS1 周度库存水平 (直接读库存)

**所有品种 CU0/SC0/M0 因子矩阵**:
- 新增 `dxy_neg` (-zscore of DXY, USD↓=利多)
- 新增 `tips_neg` (-zscore of TIPS, 实际利率↓=利多)

### 复现效果对比

| 指标 | 改造前 | 改造后 | Δ | 报告参考 |
|---|---|---|---|---|
| 铜 Sharpe | 0.510 | **0.612** | +20% | 0.942 (65% 实现) |
| 原油 Sharpe | 0.618 | **0.739** | +20% | 0.982 (75% 实现) |
| 豆粕 Sharpe | 0.195 | **0.272** | +39% | 0.754 (36% 实现) |
| **组合 Sharpe** | **0.528** | **0.740** | **+40%** | **1.60 (46% 实现)** |
| 组合年化 | 7.1% | **9.0%** | +27% | 22.8% |
| 组合 MaxDD | -23.7% | -22.2% | 改善 | -15.1% |
| 胜率 | 51.6% | **56.6%** | +5pp | 58.2% |

### 当前仓位 (2026-08-21)

- 铜 (CU0): 0.10
- 原油 (SC0): 0.10
- 豆粕 (M0): 0.70

### 已知仍未补的数据

| 因子 | 原因 | 影响 | 路径 |
|---|---|---|---|
| LME/ICE/CFTC COT | 付费 | §3.3 拥挤度D项缺失 | Wind/iFinD |
| 洋山铜溢价 | akshare无 | 铜Lasso缺1因子 | 钢联/付费 |
| 铜精矿TC | akshare无 | 铜Lasso缺1因子 | SMM/付费 |
| 大豆压榨利润 | akshare无 | 豆粕Lasso缺2因子 | 中国饲料协会 |
| SHFE/DCE库存历史 | akshare仅3月 | 库存象限短期 | 每日累积脚本 |

### 文件改动清单

新增:
- `data/cmf/.keys.json` (gitignored, 真实 API keys)
- `data/cmf/.keys.example.json` (committed, 占位符模板)
- `scripts/cmf/1c_fetch_extras.py` (EIA + FRED + DXY fetcher)

修改:
- `.gitignore` (新增 `data/cmf/.keys.json`)
- `QuantNodes/strategy/commodity_fundamental/pipeline.py` (原油分支 + helpers + 因子矩阵)

### 验收

- ✅ `eia_weekly_inventory` ≥ 2000 行 (实际 2290)
- ✅ `fundamentals_daily` 含 `dxy` / `tips10y`
- ✅ 组合 Sharpe ≥ 0.65 (实际 0.74)
- ✅ SC0 Sharpe ≥ 0.65 (实际 0.74)
- ✅ CU0/M0 未退化
- ✅ `.keys.json` gitignored