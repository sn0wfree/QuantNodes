# 87-CTA 数据准备现状与完整性评估

> 本文档总结 DuckDB 数据仓库（`~/Public/DataCache/futures_options_daily.duckdb`）的当前数据准备情况、完整性评估与已知缺口。

## 一、数据仓库总览

| 维度 | 数值 |
|---|---|
| DuckDB 文件 | `~/Public/DataCache/futures_options_daily.duckdb` (1.13 GB) |
| 总表数 | 255 (92 期货 + 65 期权 + 88 连续主力 + 10 核心/因子/补全表) |
| 期货日线 (`v_all_futures`) | 1,318,312 行 |
| 期权日线 (`v_all_options`) | 17,248,918 行 |
| 覆盖交易所 | CFFEX / SHFE / DCE / CZCE / INE / GFEX 全部 6 家 |
| 期货产品数 | 88 |
| 期权产品数 | 65 |
| 日期范围 | 2018-01-02 ~ 2026-07-21 (8.5 年) |

## 二、数据分层结构

### 1. 原始数据层（92 期货表 + 65 期权表）

按 `{exchange}_{product}_{kind}_daily` 命名（如 `cffex_if_futures_daily`），统一经 `v_all_futures` / `v_all_options` 两个视图 UNION ALL 聚合。

列结构：
- 期货：`open / high / low / close / volume / turnover / open_interest / pre_settlement / pre_close / pre_open_interest`
- 期权：上述 + `underlying / option_type / strike_price`

### 2. 核心派生表

| 表名 | 行数 | 说明 |
|---|---|---|
| `contract_specs` | 92 | 合约乘数/最小变动/计价单位 (人工维护于 `scripts/init_contract_specs.py`) |
| `trading_calendar` | 2,073 | 交易日历 (2018~2026-07) |
| `main_contract_mapping` | 133,497 | 主力合约映射 (每日每产品 OI 最大) |
| `continuous_main_{PRODUCT}_daily` | ×88 | 主力连续面板 (每产品一张) |

### 3. 因子层派生表（`scripts/build_derived_factors.py` 生成）

| 表名 | 行数 | 内容 |
|---|---|---|
| `inter_contract_spread` | 140,472 | 跨期价差: 每产品每日近月/次月/季月 close + spread_1_2/1_3/1_4 |
| `cross_product_spread` | 51,048 | 跨品种套利: 26 主流套利对 (RB-I, Y-M, IF-IC 等) |
| `term_structure_daily` | 1,308,009 | 期限结构: 每产品每日按到期排名 1-15 合约 + days_to_expiry |
| `volatility_hv_daily` | 133,497 | 历史波动率: 主力 20/60/120 日年化 HV (LN return × √252) |
| `options_pcr_daily` | 53,706 | 期权 PCR: 成交量/持仓量双维度 + call/put 汇总 |

### 4. akshare 补全层（`scripts/fetch_akshare_data.py` 生成）

| 表名 | 行数 | 内容 |
|---|---|---|
| `spot_basis_daily` | 83,230 | 现货基差: 55 品种逐日, 近月/主力双基差 + 基差率 |
| `futures_rule_snapshot` | 158 | 交易规则快照: 6 交易所保证金/涨跌停/乘数/tick (as_of 2026-08-11) |

## 三、完整性评估

### 训练窗口（2018-01-02 ~ 2025-12-31）

| 交易所 | 交易日覆盖 | 期货产品 | 期权产品 |
|---|---|---|---|
| CFFEX | 1942/1942 (100%) | 8 | 3 |
| CZCE | 1942/1942 (100%) | 27 | 20 |
| DCE | 1942/1942 (100%) | 23 | 19 |
| SHFE | 1940/1942 (99.9%) | 20 | 17 |
| INE | 1888/1942 (97.2%) | 5 | 1 |
| GFEX | 734/1942 (37.8%, 2022 上市) | 5 | 5 |

- **期货 88 产品全部覆盖训练窗口**，无断档
- **期权 64 产品全部覆盖**
- 连续主力/主力映射/5 张因子表：训练窗口无缺日

### 训练窗口内样本深度不足的产品（回测需注意）

| 产品 | 上市 | 训练内天数 |
|---|---|---|
| PD / PT | 2025-11-27 | 25 |
| OP | 2025-09-10 | 73 |
| PL | 2025-07-22 | 111 |
| BZ | 2025-07-08 | 121 |
| AD | 2025-06-10 | 139 |
| PS / LG / PR | 2024 下半年 | 246-323 |

### 样本外（2026 起，作回测验证用）

| 分层 | 期货产品 | 期权产品 |
|---|---|---|
| 2026-01~07 全覆盖 | 34 | 22 |
| 2026-01~02 | 23 | 19 |
| 仅 <40 天 | 31 | 24 |

> 注：样本外存在 staggered 截断（各产品停更日期不一），滚动验证需按产品各自可用区间切片。

## 四、已知缺口与限制

| 缺口 | 现状 | 影响 |
|---|---|---|
| **现货基差尾部缺失** | akshare 上游 (生意社) 2025-07-22 起失效, 基差仅到 2025-07-21 | 训练窗口基差覆盖 94.2% (1829/1942 天), 基差因子样本止于 2025-07 |
| **SHFE/DCE/INE/GFEX 新鲜度** | FTP 源 (120.136.159.69) 不可达, 各所最新日期不一 | 样本外数据不连续, 训练窗口不受影响 |
| **持仓排名/资金流向** | 仅 SHFE 稳定可用, 其他交易所接口不稳定; 资金流向接口已废弃 | 缺少席位/资金因子维度 |

## 五、生成脚本

```bash
# 合约规格 + v_all 视图
python scripts/init_contract_specs.py

# 核心派生表 (日历/主力映射/连续面板)
python scripts/build_derived_tables.py

# 因子层 (跨期/跨品种/期限结构/HV/PCR)
python scripts/build_derived_factors.py --only inter|cross|term|hv|pcr

# akshare 补全 (现货基差/交易规则)
python scripts/fetch_akshare_data.py --only basis|rule [--start YYYY-MM-DD --end YYYY-MM-DD]
```

## 六、数据访问（CtaDataLoader）

```python
from QuantNodes.strategy.cta.data_loader import CtaDataLoader

dl = CtaDataLoader()
dl.load_futures_daily("IF", start="2024-01-01", end="2024-12-31")   # 单产品期货日线
dl.load_continuous_main("AU")                                        # 主力连续面板
dl.load_spot_basis("AU", start="2025-01-01", end="2025-07-21")       # 现货基差
dl.load_futures_rule()                                               # 交易规则快照
dl.load_v_all_futures(exchanges=["CFFEX"])                           # 跨所聚合
dl.info()                                                            # 元数据摘要
```

## 七、数据资产清单（其他）

- **ETF 日线**：`data/real/per_etf/` 46 个 (510300/510500 等)
- **宏观因子**：`data/high_freq_macro/` — VIX/DXY/US10Y/CN10Y/CPI/M2/PMI
- **指数代理**：`data/ifind_cache/index_proxy/` 18 个 (中证800/全指/创业板/科创/HSTECH/NDX 等)
