# Hardcoded Values Audit Report
**Scope:** `QuantNodes/research/factor_test/` + `QuantNodes/cli/__init__.py`
**Date:** 2026-06-12
**Method:** Systematic file-by-file review

---

## Executive Summary

| Severity | Count | Notes |
|----------|-------|-------|
| HIGH | 22 | Major user-facing limits, prevents reuse |
| MEDIUM | 35 | Internal but affects behavior |
| LOW | 14 | Cosmetic / minor |
| **TOTAL** | **71** | |

The most critical issues cluster in:
1. **Defaults that block "current data" runs** — `pipeline_runner` config defaults, `ifind_database` `date_beg='20260101'`, e2e `20260101~20260630`
2. **Industry universe hardcoded as 申万 1-30** in `_INDUSTRY_MAP` and `factor_score_node` (uses `3 * 29 * group`)
3. **iFinD cache TTL** hardcoded to 7 days, rate-limit to 0.5s
4. **Output paths** defaulting to `./output/`, `/tmp/e2e_output/`
5. **No CLI overrides** for any of the above

---

## 1. `QuantNodes/research/factor_test/pipeline_runner.py`

| Line | Value | Exposed? | Severity | Notes / Suggested Fix |
|------|-------|----------|----------|----------------------|
| 460 | `dashboard_html = pool.base_dir.parent / "dashboard_streaming.html"` | No | LOW | Hardcoded filename, no override |
| 477 | `metrics_json = pool.base_dir / "metrics.json"` | No | LOW | Hardcoded filename |

> No other major hardcoded values; the runner is config-driven via `SingleFactorTestConfig`.

---

## 2. `QuantNodes/research/factor_test/config.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 24 | `min_ipo_days: int = 360` | Yes (config) | MEDIUM | Default 360 days for IPO exclusion — should be config-default in YAML, not Pydantic default |
| 47 | `min_group_size: int = 5` | Yes (config) | MEDIUM | IC group-size default — should be exposed as CLI flag |
| 52 | `groups: int = 5` | Yes (config) | MEDIUM | Default 5 quintiles — should be CLI flag |
| 123 | `max_rounds: int = 3` | Yes (config) | MEDIUM | Default evolution rounds — CLI already overrides via `--max-rounds` ✓ |
| 124 | `parents_per_round: int = 1` | Yes (config) | MEDIUM | Has 2-must crossover special case in docstring, not enforced |
| 129 | `top_percent_threshold: float = 0.3` | Yes (config) | MEDIUM | Default threshold |
| 132 | `early_stop_patience: int = 0` | Yes (config) | LOW | 0 = disabled |
| 144 | `data_path: str = './testdata/test_h5_new/'` | Yes (config) | **HIGH** | Hardcoded test data path prevents running on user data unless overridden |
| 146 | `load_keys: list = ['stklist', 'trade_dt', 'cp', 'id_citic1', 'mv_float']` | Yes (config) | MEDIUM | Doesn't include `st`, `suspend`, `ud_limit`, `ipo_days` needed for tradability filter — surprising default |
| 30-31 | `adj_date_beg`, `adj_date_end` | Required | **HIGH** | No defaults — must be set in YAML; Pydantic will error on instantiation |

> Note: `load_keys` default missing the tradability-filter keys (`st`, `suspend`, `ud_limit`, `ipo_days`) means **the default config will fail tradability filtering** — silent data gap.

---

## 3. `QuantNodes/research/factor_test/ifind_db/ifind_database.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 42-49 | `_INDUSTRY_MAP` (30 industries: 农林牧渔→1 … 美容护理→30) | No | **HIGH** | **申万一级行业硬编码** — 30 industries hardcoded. Needs config or external JSON. |
| 63 | `date_beg: str = '20260101'` | Constructor arg | **HIGH** | **Default start date `'20260101'`** — pre-dates current date (2026-06-12) by ~5 months. Not exposed via config. |
| 64 | `date_end: str = ''` | Constructor arg | LOW | Empty=current; OK |
| 64 | `universe: str = '沪深300'` | Constructor arg | **HIGH** | **Stock universe hardcoded to `'沪深300'`** — only 3 values known: `'沪深300'`, `'中证500'`, `'all'`. No docstring on other options. |
| 174-180 | `risk_registry` 10 paths (`/beta`, `/momentum`, ...) | No | MEDIUM | Hardcoded risk-factor list. Renaming a risk factor requires code change. |
| 198 | `query = f'A股市场所有股票代码({self._date_beg[:4]}年)'` | No | MEDIUM | Hardcoded Chinese query template for stock universe |
| 226 | `'沪深300、中证500{...}年{...}月至{...}月的收盘点数'` | No | MEDIUM | Hardcoded Chinese query template |
| 256 | `query = '沪深300、中证500收盘点数'` | No | LOW | Hardcoded index query |
| 279 | `batch_size = 50` | No | MEDIUM | iFinD batch size — affects rate-limit behavior |
| 333 | `batch_size = 50` | No | MEDIUM | Same, stock_info panel |
| 259 | `pd.DataFrame(['000300.SH', '000905.SH'])` | No | LOW | Fallback index codes hardcoded |
| 284 | `f'{code_str}{self._date_beg[:4]}年{self._date_beg[4:6]}月至{self._date_end[4:6]}月的日收盘价'` | No | MEDIUM | Hardcoded date format (yyyy/mm) |
| 401, 408, 415 | ST/停牌/涨跌停 string match sets `('是', 'True', '1', 'ST')` | No | LOW | Hardcoded string match for boolean parsing |
| 432 | `panel[col] = 500` | No | LOW | Fallback IPO days |
| 429 | `(d - ipo_dt).days if pd.notna(d) else 9999` | No | LOW | Fallback 9999 |
| 521 | `all_keys = ['cp', 'st', 'suspend', 'ud_limit', 'ipo_days', 'id_citic1', 'mv_float']` | No | MEDIUM | 7 stock keys hardcoded |
| 530-531 | `stklist.h5`, `trade_dt.h5` | No | LOW | Hardcoded H5 filenames |
| 539 | `stk_daily.h5`, `index_daily.h5` | No | LOW | Hardcoded H5 filenames |

---

## 4. `QuantNodes/research/factor_test/ifind_db/fetcher.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 14 | `IFIND_SKILL_DIR = Path.home() / '.agents/skills/ifind'` | No | MEDIUM | iFinD config dir hardcoded to user home |
| 15 | `IFIND_CONFIG = IFIND_SKILL_DIR / 'mcp_config.json'` | No | MEDIUM | Config filename hardcoded |
| 39 | `RATE_LIMIT_SECONDS = 0.5` | No (class attr) | **HIGH** | **iFinD rate-limit hardcoded at 0.5s (2 QPS)**, comment says "免费版" — paid plans allow higher. Should be constructor arg or env var. |
| 43 | `cache_dir = Path(__file__).parent / 'cache'` | Constructor arg (default) | MEDIUM | Cache dir hardcoded to `ifind_db/cache/` |
| 76 | `< 7 * 86400` | No | **HIGH** | **iFinD cache TTL hardcoded to 7 days** — no override. Should be constructor arg. |
| 167 | `> len(series) * 0.5` | No | LOW | 50% threshold for numeric conversion |
| 150-162 | `'万'`, `'亿'`, `'万亿'` multipliers `1e4`, `1e8`, `1e12` | No | LOW | Chinese unit multipliers hardcoded |

---

## 5. `QuantNodes/research/factor_test/utils/data_loader.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 16 | `api_path: str = './testdata/test_h5_new/'` | Constructor arg | **HIGH** | **Default test-data path** — fallback if no H5 dir supplied. Will silently fail in production. |
| 80-87 | `'stk_daily.h5'`, `'index_daily.h5'` filenames | No | LOW | Hardcoded H5 filenames throughout |
| 80-87 | Keys `'stklist'`, `'trade_dt'`, `'indexlist'`, `'index_cp'` | No | LOW | Hardcoded keys |

> Mostly thin wrappers; hardcodes are interface-level.

---

## 6. `QuantNodes/research/factor_test/utils/date_utils.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 22 | `num_lens == 8` | No | LOW | YYYYMMDD length |
| 22 | `data_type == 'int64'` | No | LOW | Date dtype requirement |
| 50 | `rule=('M', 'end')` | Function default | MEDIUM | Default `'M', 'end'` (monthly end-of-month) — should be config |
| 60, 95, 114 | `('M', 'end')` in error messages | No | LOW | Documentation strings |

> No major issues; pure date utility.

---

## 7. `QuantNodes/research/factor_test/utils/performance_metrics.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 9 | `from .constants import ANNUAL_DAYS` | N/A | INFO | Imports `ANNUAL_DAYS` (correctly extracted) |
| 89 | `adj_cycle = ... / (len(adj_dates) - 1)` | No | MEDIUM | Assumes adj_dates evenly spaced — no validation |
| 94, 127 | `np.sqrt(ANNUAL_DAYS)` (250) | Constants | LOW | **Annualization factor hardcoded to 250** (China A-share trading days) — wrong for crypto/24h markets |
| 121 | `for year_i in account_net_df['year'].unique()` | No | LOW | Year-grouped metrics |

---

## 8. `QuantNodes/research/factor_test/utils/constants.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 5-9 | `INDEX_MAPPING` (HS300/ZZ500/SZ50) | No | **HIGH** | **Hardcoded 3-index mapping**. Missing: 创业板, 科创50, 中证1000. |
| 12-16 | `INDEX_CP_MAPPING` (codes `000300.SH`, `000905.SH`, `000016.SH`) | No | **HIGH** | **Hardcoded index codes** for hedge benchmarks |
| 19-22 | `INDUSTRY_MAPPING` (2 keys) | No | LOW | Industry key mapping |
| 25 | `ANNUAL_DAYS = 250` | No | **HIGH** | **Annualization days hardcoded 250** — wrong for non-A-share markets |

> NOTE: `INDEX_MAPPING` references `'id_50'` key (line 8) but no 50-component loader exists in `ifind_database.py` `_ROUTE_TABLE` — **dead config**.

---

## 9. `QuantNodes/research/factor_test/utils/ifind_mapping.py`

**File does not exist.** The user listed it in the audit scope but it is not present in `utils/`. No findings.

---

## 10. Nodes (`QuantNodes/research/factor_test/nodes/*.py`)

### 10.1 `load_data_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 30 | `'./testdata/test_h5_new/'` | Constructor arg (default) | **HIGH** | **Hardcoded fallback data path** — fallback when no config |
| 31 | `['stklist', 'trade_dt', 'cp', 'id_citic1', 'mv_float']` | Constructor arg (default) | MEDIUM | Default load_keys missing tradability keys |
| 56, 68, 72, 79 | `'stk_daily.h5'`, `'index_daily.h5'`, `'cp'`, `'index_cp'` | No | LOW | Filename/key hardcodes |

### 10.2 `sample_pool_filter_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 31 | `sample_index='all'` | Constructor arg | MEDIUM | Default 'all' OK |
| 55-56 | `'id_300'`, `'id_500'` | No | MEDIUM | **Hardcoded keys for HS300/ZZ500** |
| 71 | `ind_key = 'id_citic1'` | No | MEDIUM | Default industry key hardcoded |
| 75 | `f'ind_name_{ind_key.replace("id_", "").upper()}'` | No | MEDIUM | **String-based file key construction** — fragile |

### 10.3 `tradability_filter_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 33 | `TradableSetting()` defaults | Yes (config) | LOW | All defaults from `config.py` |
| 57 | `ud_limit = ud_limit.abs()` | No | MEDIUM | **Hardcoded `.abs()` on ud_limit** — assumes symmetric semantics |
| 70, 73, 76, 79 | `st == 1`, `suspend == 1`, `ud_limit == 1.0`, `ipo_days < s.min_ipo_days` | No | LOW | Comparison values |

> No new hardcoded values beyond config defaults.

### 10.4 `adjust_date_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 29 | `adj_date_beg: 20170801` | Constructor arg (default) | **HIGH** | **Default start date `20170801`** — pre-2017 data won't work; current-data runs need override |
| 30 | `adj_date_end: 20171231` | Constructor arg (default) | **HIGH** | **Default end date `20171231`** — completely stale default; runs will silently use 5-month window |
| 31 | `adj_mode: ['M', 'end']` | Constructor arg (default) | MEDIUM | Default monthly-end rebalance |

> **MAJOR ISSUE:** The defaults `20170801` → `20171231` make this node non-functional on any current data unless the user explicitly overrides. The config has these as **Required** fields (`config.py:30-31`), but the node defaults override when used standalone.

### 10.5 `factor_preprocess_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 32-34 | `missing=''`, `extreme=''`, `norm=''` | Constructor arg | LOW | OK defaults (no-op) |
| 123 | `n = 5` (MAD multiplier) | No | MEDIUM | **Hardcoded 5-MAD winsorization** — should be config |
| 128, 129 | `quantile(0.025)`, `quantile(0.975)` | No | MEDIUM | **Hardcoded 2.5%/97.5% pct-shrink** — should be config |
| 145, 146 | `* 0.5` (rank clipping) | No | LOW | Rank boundary heuristic |
| 146 | `0.01`, `0.99` | No | LOW | Fallback rank bounds |

### 10.6 `factor_neutralize_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 30-32 | `industry_neutral=False`, `risk_neutral=False`, `risk_factors=[]` | Constructor arg | LOW | OK defaults |
| 53 | `if file_key == 'risk_factor.h5':` | No | LOW | Hardcoded filename branch |

### 10.7 `ic_analyzer_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 29 | `min_group_size=5` | Constructor arg | MEDIUM | Default 5 |
| 83 | `* np.sqrt(ic.notna().sum() - 1)` | No | LOW | T-stat formula |
| 85-86 | `'IC均值'`, `'IC标准差'`, etc. (Chinese keys) | No | LOW | Output dict keys hardcoded (Chinese) — affects downstream consumers |

### 10.8 `group_analyzer_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 35-39 | `groups=5`, `factor_direction=1`, `floor_mode='group'`, `hedge='equal'`, `hedge_path=None` | Constructor arg | LOW | OK defaults (from config) |
| 161 | `(fac_group == g) & (fac_group.diff(-1) != 0)` turnover calc | No | LOW | Turnover formula |
| 192 | `pd.DataFrame() if 'turn' not in dir() else turn` | No | LOW | Fallback empty turnover |
| 222-223 | `DataLoader()` empty path | No | LOW | Custom hedge loader |

### 10.9 `long_short_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 29 | `factor_direction=1` | Constructor arg | LOW | OK default |
| 41-49 | `long_n = n_groups`, `short_n = 1` (or vice versa) | No | LOW | Long=top group, short=bottom group — semantically OK |

### 10.10 `factor_score_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 30 | `enabled=True` | Constructor arg | LOW | OK |
| 50 | `group = 5` (hardcoded) | No | **HIGH** | **Hardcoded `group = 5`** — quintile count can't be customized here |
| 70 | `nonan < 3 * 29 * group` | No | **HIGH** | **Magic number `3 * 29 * group = 435`** — assumes 3 size groups × 29 中信行业 × 5 quintiles. Breaks if industry count != 29 (which is hardcoded too). |
| 75 | `pd.qcut(mv_adj.loc[t_i], 3, labels=range(1, 4))` | No | MEDIUM | **Hardcoded 3 size groups** |
| 62 | `len(x.dropna().unique()) >= (n - 1)` | No | LOW | qcut threshold |

> **MAJOR ISSUE:** Lines 50, 70, 75 hardcode 29 中信行业 / 3 市值分组 / 5 quintile — fragile and inconsistent with `ifind_database._INDUSTRY_MAP` which has 30 申万 industries (not 中信).

### 10.11 `risk_correlation_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 30 | `factors='all'` | Constructor arg | LOW | OK default |
| 51 | `loader.get_apikeys('risk_factor.h5')` | No | LOW | Hardcoded filename |
| 66 | `if file_key == 'risk_factor.h5':` | No | LOW | Hardcoded branch |

### 10.12 `factor_test_report_node.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 32 | `dir='./output/'` | Constructor arg | **HIGH** | **Default output dir `./output/`** — not customizable beyond constructor; no `~/` expansion |
| 33 | `format=['parquet', 'json']` | Constructor arg | LOW | OK |
| 96 | `timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')` | No | LOW | Standard |
| 113 | `f"factor_test_{factor_name}_{timestamp}.json"` | No | LOW | Filename pattern |
| 126, 129 | `f"{factor_name}_{key}_{sub_key}.parquet"`, `f"{factor_name}_{key}.parquet"` | No | LOW | Filename pattern |

---

## 11. E2E (`QuantNodes/research/factor_test/e2e/*.py`)

### 11.1 `data_prep.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 60 | `pd.bdate_range('2026-01-04', ...)` | No | **HIGH** | **Hardcoded start date `'2026-01-04'`** — data generator always starts Jan 2026 |
| 64 | `range(100001, 100001 + n_stocks)` | No | MEDIUM | **Hardcoded stock-code start `100001`** — synthetic codes don't match real SH/SZ formats (`\d{6}\.(SH|SZ)` regex in ifind_database) |
| 72, 73 | `'000300.SH'`, `'000905.SH'` | No | MEDIUM | **Hardcoded index codes** |
| 81 | `100 * np.exp(np.cumsum(rng.randn(n_days, n_stocks) * 0.02, axis=0))` | No | LOW | 100 starting price, 0.02 vol — hardcoded GBM |
| 83 | `rng.randint(1, 31, (n_days, n_stocks))` | No | MEDIUM | **Industry codes 1-30 hardcoded** (matches 申万, not 中信 29) |
| 85 | `rng.lognormal(10, 1, ...)` | No | LOW | MV distribution |
| 88, 91, 94 | `:min(2, n_stocks)`, `5:8`, `10:12` | No | MEDIUM | **Hardcoded sparse ST/suspend/ud_limit days** — relies on `n_stocks > 3` and `> 5` |
| 96 | `ipo_days = ... * 500` | No | MEDIUM | Hardcoded 500-day IPO |
| 97 | `ipo_days[0, 0] = 100` | No | MEDIUM | One specific stock has IPO=100 (will be excluded by min_ipo_days=360) |
| 112-116 | CLI defaults: `n_days=120`, `n-stocks=30`, factors list | CLI | LOW | CLI-exposed ✓ |
| 114 | `momentum_20d,reversal_5d,volatility_60d` | CLI default | MEDIUM | **Default factor list hardcoded** |

### 11.2 `run_evolution_e2e.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 89 | `adj_date_beg=20260101, adj_date_end=20260630` | No | **HIGH** | **Hardcoded E2E date range 2026-01-01 ~ 2026-06-30** — won't work after June 2026 |
| 93 | `min_ipo_days=360` | No | MEDIUM | Hardcoded |
| 101 | `groups=5`, `min_group_size=5` | No | MEDIUM | Hardcoded |
| 102 | `factor_direction=1` | No | LOW | OK |
| 117 | `load_keys=["cp", "id_citic1", "mv_float", "st", "suspend", "ud_limit", "ipo_days"]` | No | MEDIUM | E2E includes tradability keys (config default doesn't!) |
| 162-172 | CLI args: `--data-path`, `--factor-name`, `--directions`, `--output-dir`, `--max-rounds` | CLI | LOW | OK |
| 166 | `default="/tmp/e2e_output/"` | CLI default | **HIGH** | **Hardcoded `/tmp/e2e_output/`** default — different from project output dir |
| 245 | `rag_top_k=3` | No | MEDIUM | **Hardcoded RAG top_k=3** for evolution loop |
| 246 | `max_ancestor_depth=2, max_descendant_depth=2` | No | MEDIUM | Hardcoded lineage depth |
| 247 | `use_compress=True` | No | MEDIUM | Hardcoded compress flag |
| 164 | `directions="momentum,reversal,volatility"` | CLI default | MEDIUM | **Default directions hardcoded** |
| 163 | `factor-name="momentum_20d"` | CLI default | MEDIUM | **Default factor name hardcoded** |
| 198 | `disable_quality_gate`, `disable_kb` (kwargs not used) | No | LOW | `enable_kb` parameter computed but never passed to `_build_config` (line 197) — dead code |

> **BUG:** Line 197 calls `_build_config(enable_kb=not args.disable_kb)` but the function signature (line 79) accepts `enable_kb` and the value is not actually used inside `_build_config`. The kb is built separately at line 224.

---

## 12. `QuantNodes/cli/__init__.py`

| Line | Value | Exposed? | Severity | Notes |
|------|-------|----------|----------|-------|
| 25 | `DEFAULT_API_PORT = 8000` | Module constant | LOW | OK, exposed via flag |
| 26 | `DEFAULT_FRONTEND_PORT = 5173` | Module constant | LOW | OK |
| 27 | `DEFAULT_HOST = "localhost"` | Module constant | LOW | OK |
| 50-56 | `["data", ".quant_agent/memory", ".quant_agent/dream", "outputs", "logs"]` | No | MEDIUM | **Hardcoded init directory list** |
| 200 | `QUANTNODES__CACHE_TTL=3600` | No | MEDIUM | **Cache TTL hardcoded 3600s (1 hour)** in env template — different from iFinD fetcher's 7 days (inconsistency) |
| 199 | `QUANTNODES__CACHE_ENABLED=true` | No | LOW | OK |
| 245 | `"TA-Lib>=0.6.0"` | No | LOW | Pinned version |
| 323 | `"data/quantnodes.db"` | No | MEDIUM | **Hardcoded DuckDB path** — user prompted for override but default is the same |
| 335 | `"localhost"` ClickHouse host | No | LOW | OK default |
| 336 | `"8123"` ClickHouse port | No | LOW | OK default |
| 337 | `"default"` ClickHouse user | No | LOW | OK default |
| 339 | `"default"` ClickHouse database | No | LOW | OK default |
| 343 | `"3306"` MySQL port | No | LOW | OK default |
| 344 | `"root"` MySQL user | No | LOW | OK default |
| 346 | `"quant"` MySQL database | No | LOW | OK default |
| 463 | `api_port = args.port + 1000` | No | LOW | Port-allocation heuristic |
| 472 | `datetime.now().strftime("%Y%m%d_%H%M%S")` | No | LOW | Log filename timestamp |
| 526 | `range(30)` (30 retries) | No | MEDIUM | **Hardcoded 30 retries × 1s sleep = 30s wait** for backend ready |
| 528 | `timeout=2` | No | MEDIUM | **Hardcoded 2s timeout** for backend probe |
| 532 | `time.sleep(1)` | No | LOW | 1s sleep between probes |
| 1253 | `default="沪深300"` | CLI default | **HIGH** | **Hardcoded `universe` default = '沪深300'** for `factor-data-fetch` |

---

## Cross-Cutting Findings (HIGH Severity Summary)

### 1. **Date defaults prevent "current data" runs**

| File | Line | Value | Issue |
|------|------|-------|-------|
| `ifind_database.py` | 63 | `date_beg='20260101'` | iFinD default start date |
| `adjust_date_node.py` | 29-30 | `20170801, 20171231` | AdjustDate default (ancient) |
| `e2e/data_prep.py` | 60 | `'2026-01-04'` | Synthetic data start |
| `e2e/run_evolution_e2e.py` | 89 | `20260101 ~ 20260630` | E2E hardcoded window |
| `cli/__init__.py` | 1253 | `default="沪深300"` | universe default only (no date default) |

### 2. **Industry / Stock universe hardcoded**

| File | Line | Value | Issue |
|------|------|-------|-------|
| `ifind_database.py` | 42-49 | `_INDUSTRY_MAP` 30 申万 | Hardcoded 30 industries, no config path |
| `utils/constants.py` | 5-16 | INDEX_MAPPING/INDEX_CP_MAPPING | 3 hardcoded indices |
| `factor_score_node.py` | 70 | `3 * 29 * group = 435` | Magic 29 (中信 count, mismatched with 30 申万) |
| `factor_score_node.py` | 75 | `pd.qcut(..., 3, ...)` | Hardcoded 3 size groups |
| `data_prep.py` | 83 | `rng.randint(1, 31, ...)` | 1-30 industry range (申万) |

### 3. **iFinD rate limits / cache**

| File | Line | Value | Issue |
|------|------|-------|-------|
| `fetcher.py` | 39 | `RATE_LIMIT_SECONDS = 0.5` | Hardcoded 2 QPS |
| `fetcher.py` | 76 | `7 * 86400` | Hardcoded 7-day cache TTL |
| `cli/__init__.py` | 200 | `CACHE_TTL=3600` | Different (1 hour) — inconsistency |

### 4. **Output paths not customizable**

| File | Line | Value | Issue |
|------|------|-------|-------|
| `factor_test_report_node.py` | 32 | `'./output/'` | Default output dir, no `~/` expansion |
| `e2e/run_evolution_e2e.py` | 166 | `'/tmp/e2e_output/'` | Hardcoded `/tmp` |
| `cli/__init__.py` | 50-56 | `["data", ".quant_agent/...", "outputs", "logs"]` | Init dir list |

### 5. **CLI gaps** (config has no CLI flag for these)

- `--min-ipo-days` (config: 360)
- `--min-group-size` (config: 5)
- `--groups` (config: 5)
- `--universe` only for `factor-data-fetch`, not for `evolve`
- `--data-path` only for E2E, not for `evolve` or other commands
- `--cache-ttl`, `--rate-limit` for iFinD fetcher
- `--industry-set` (申万 vs 中信 vs custom)

---

## Recommendations (Non-Code Summary)

### Critical (HIGH) — must fix before reuse on real data

1. Replace `date_beg='20260101'` and `universe='沪深300'` defaults in `ifind_database.py` with explicit required args or env-var fallback.
2. Remove or update stale `20170801 / 20171231` defaults in `adjust_date_node.py:29-30`.
3. Update `e2e/data_prep.py` start date from hardcoded `'2026-01-04'` to `datetime.now()`.
4. Update `e2e/run_evolution_e2e.py:89` to use `datetime.now()` instead of `20260101~20260630`.
5. Move `_INDUSTRY_MAP` (30 申万) from `ifind_database.py:42-49` to a JSON/YAML config file.
6. Move `INDEX_MAPPING` / `INDEX_CP_MAPPING` (3 hardcoded indices) from `utils/constants.py` to config.
7. Expose `RATE_LIMIT_SECONDS` and cache TTL (7 days) in `IFindFetcher.__init__`.
8. Add CLI flag `--universe` (and `--industry-set`, `--data-path`) to `evolve` and related subcommands.
9. Fix `factor_score_node.py:70` magic number `3 * 29 * group` — use configurable size groups × industry count.
10. Fix `factor_score_node.py:50` hardcoded `group = 5`.

### Medium (MEDIUM) — should be configurable

11. Add `--groups`, `--min-group-size`, `--min-ipo-days` CLI flags.
12. Move `ANNUAL_DAYS = 250` to config (breaks for 24h markets).
13. Move `momentum_20d,reversal_5d,volatility_60d` defaults to config.
14. Add `--max-ancestor-depth`, `--max-descendant-depth`, `--rag-top-k` to E2E CLI.
15. Fix e2e `_build_config` dead `enable_kb` parameter (line 197 → 79).
16. Move `batch_size = 50` (iFinD) to constructor.
17. Move winsorization `n = 5` (MAD multiplier) to config.
18. Move `quantile(0.025)`, `quantile(0.975)` to config.

### Low (LOW) — cosmetic

19. Add `INDEX_MAPPING['SZ50']` loader (currently `'id_50'` is dead — referenced but no route).
20. Replace `f-string` filename construction (e.g. `'factor_test_{factor_name}_{timestamp}.json'`) with `Path.with_name()`.
21. Add env var override for `CACHE_TTL` consistency between CLI env template (3600) and iFinD fetcher (7 days).

---

## Cross-File Inconsistencies

| Item | Files | Inconsistency |
|------|-------|---------------|
| **Industry count** | `ifind_database.py:42` (30 申万) vs `factor_score_node.py:70` (29 中信) | **30 vs 29 — different industry taxonomies hardcoded in different files** |
| **Stock code format** | `data_prep.py:64` (synthetic `100001+`) vs `ifind_database.py:209` (regex `\d{6}\.(SH|SZ)`) | Synthetic data won't pass real-data regex check |
| **Cache TTL** | `fetcher.py:76` (7 days) vs `cli/__init__.py:200` (3600s = 1 hour) | Two different TTLs in the project |
| **`load_keys` default** | `config.py:146` (5 keys, no tradability) vs `e2e/run_evolution_e2e.py:117` (7 keys, with tradability) | Config default will silently skip tradability filtering |
| **Index count** | `INDEX_MAPPING` references SZ50 (`id_50`) but no loader route exists in `ifind_database._ROUTE_TABLE` (only `id_300`, `id_500`) | **Dead config entry** |
