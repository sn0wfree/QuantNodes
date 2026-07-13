# CICC 动量 ETF 轮动 — 缺失模块重写计划

> **编号**: 33
> **状态**: 📋 计划中（未实施）
> **依赖**: docs/17-固收+动量ETF轮动.md + docs/28-31 (TradingAgents)
> **日期**: 2026-07-08

---

## 一、背景

`QuantNodes/strategy/momentum_etf_rotation/__init__.py` 引用了 5 个不存在的模块：
- `.data` (load_etf_nav_panel, load_bond_etf_nav)
- `.data_tencent` (fetch_one_etf_tencent, write_fetch_log) — 被 `scripts/fetch_real_etf_panel.py` 引用
- `.validation` (ValidationConfig, run_full_validation, etc.) — 被 `agent/tools/validation.py` 引用
- `.extended_metrics` (extended_metrics, format_metrics_table)
- `.contribution` (etf_contribution, category_contribution, etc.)
- `.brinson` (brinson_attribution, CATEGORIES)

**根本原因**: Stage11 commit (702907a) 时这些文件在 working tree 中但 `git add` 时被忽略，只 stage 了 covariance.py + risk_parity.py。`__init__.py` 已损坏地进入 commit。

**可用数据**:
- `data/real/etf_nav_2018-01-01_2025-07-06.parquet` (44 ETFs × 1820 days)
- `data/real/etf_nav_2018-01-01_2026-06-30.parquet`
- `data/real/per_etf/*.parquet` (44 个独立文件)
- `data/real/fetch_log.json` (44 成功 / 0 失败)
- `data/real/validation_report.md` (1/4 通过的完整报告)
- `reports/momentum_etf_rotation/` 完整输出 (extended_metrics.json, brinson.json, *.csv)

**目标**: 从 `data/real/` 和 `reports/momentum_etf_rotation/` 反推并重写 5 个缺失模块，让 `__init__.py` 可以正常 import，所有现有脚本/工具/测试可以运行，并能用真实数据重新生成 `reports/` 输出。

---

## 二、设计原则

1. **从数据反推接口** — `data/real/*.parquet` 已经存在，模块只需要加载即可，不需要重新拉取
2. **从输出反推逻辑** — `reports/*.json` 和 `*.csv` 是已知正确输出，可以反推函数签名和算法
3. **从测试反推契约** — `tests/agent/test_validation_tool.py` 和 `tests/strategy/momentum_etf_rotation/` 已有接口期望
4. **从脚本反推入口** — `scripts/fetch_real_etf_panel.py` 显示了 data_tencent 的输入输出契约
5. **保持向后兼容** — 模块签名必须与 `__init__.py` 的引用、agent/tools/validation.py 的用法、tests 的期望一致

---

## 三、文件结构

```
QuantNodes/strategy/momentum_etf_rotation/
├── __init__.py              (现有, 已修复导入)
├── data.py                  [重写] 加载 data/real/ 下 parquet
├── data_tencent.py          [重写] Tencent API 拉取 (从脚本反推)
├── validation.py            [重写] 4 个抗过拟合检验
├── extended_metrics.py      [重写] 17 个业绩指标
├── contribution.py          [重写] 5 维度归因分析
├── brinson.py               [重写] Brinson 归因
├── universe.py              (现有)
├── momentum.py              (现有)
├── portfolio.py             (现有)
├── covariance.py            (现有)
├── risk_parity.py           (现有)
├── regime_detector.py       (现有)
├── fi_plus.py               (现有)
└── backtest.py              (现有)

tests/strategy/momentum_etf_rotation/
├── test_data_loader.py      [新增]
├── test_validation.py       [新增]
├── test_extended_metrics.py [新增]
├── test_contribution.py     [新增]
├── test_brinson.py          [新增]
└── ... (现有 7 个)

tests/agent/
└── test_validation_tool.py  (现有, 应自动通过)

scripts/
└── fetch_real_etf_panel.py  (现有, 应可运行)
```

---

## 四、模块设计

### 4.1 `data.py` — 加载 data/real/

#### 4.1.1 签名（从 `__init__.py` line 36 反推）

```python
def load_etf_nav_panel(
    start: str = "2018-01-01",
    end: str = "2025-07-06",
    data_dir: Path | None = None,    # 缺省 = <PROJECT_ROOT>/data/real
    codes: list[str] | None = None,  # None = 全部 44 ETFs
    ffill_limit: int = 5,
) -> pd.DataFrame:
    """加载 ETF 净值面板 (44 列 × ~1820 行).
    
    Returns:
        DataFrame: index=DatetimeIndex (工作日), columns=ETF codes
    """

def load_bond_etf_nav(
    code: str = "511260",
    data_dir: Path | None = None,
) -> pd.Series:
    """加载单只国债 ETF 净值 (511260 国泰 10 年期国债 ETF)."""
```

#### 4.1.2 数据源
- `data/real/etf_nav_{start}_{end}.parquet` 主面板 (1820 行 × 44 列)
- `data/real/per_etf/{code}.parquet` per-ETF 缓存

#### 4.1.3 实现要点
- 优先用主面板 parquet，不存在则拼接 per_etf/*.parquet
- `ffill(limit=5)` 与 CICC 实现一致 (见 GAP_ANALYSIS.md 章节 1)
- 自动附加 511260 (国债 ETF) 如果不在面板中

#### 4.1.4 测试用例
```python
def test_load_full_panel() -> None:
    """加载默认范围应返回 44 列 × 1820 行 DataFrame."""

def test_load_specific_codes() -> None:
    """codes=[518880, 518800] 应只返回 2 列."""

def test_load_bond_etf() -> None:
    """load_bond_etf_nav('511260') 应返回 pd.Series."""
```

---

### 4.2 `data_tencent.py` — Tencent API 拉取

#### 4.2.1 签名（从 `scripts/fetch_real_etf_panel.py` lines 30-34 + 93 + 135 反推）

```python
def fetch_one_etf_tencent(
    code: str,
    start: str = "2018-01-01",
    end: str = "2025-07-06",
    sleep_ms: int = 150,
) -> pd.Series:
    """从 Tencent 行情 API 拉取单只 ETF 日线.
    
    API endpoint: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    Returns:
        Series: index=DatetimeIndex, values=close prices
    """

def write_fetch_log(
    fetched: dict[str, int],  # code -> row count
    failed: list[str],
    log_path: Path,
) -> None:
    """写 fetch log JSON (matches data/real/fetch_log.json schema)."""
```

#### 4.2.2 实现要点
- `fetch_one_etf_tencent`: GET `https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market},{code},day,{start},{end},320,qfq`
- `sleep_ms / 1000` 速率限制 (7 req/s 实测)
- 返回 close prices (前复权)
- `write_fetch_log` 写 JSON: `{"fetched": {...}, "failed": [...], "fetched_count": N, "failed_count": M}`

#### 4.2.3 测试用例
```python
@pytest.mark.network
def test_fetch_one_etf() -> None:
    """拉取 510300 应返回 ~1700 行 (2018-2025)."""

def test_write_fetch_log(tmp_path) -> None:
    """write_fetch_log 写出的 JSON 可被 json.load 读取."""
```

---

### 4.3 `validation.py` — 抗过拟合检验

#### 4.3.1 签名（从 `__init__.py` lines 51-60 + `data/real/validation_report.md` + `validation_fix_report.md` 反推）

```python
@dataclass
class ValidationConfig:
    start_points: tuple[str, ...] = (
        "2019-01-01", "2020-06-01", "2022-01-01", "2023-06-01"  # A1 修复后
    )
    perturb_lookbacks: tuple[int, ...] = (80, 100, 120)  # A2 修复后
    perturb_corr_thresholds: tuple[float, ...] = (0.85, 0.90, 0.95)
    perturb_a_share_caps: tuple[int, ...] = (2, 3, 4)
    rebalance_offsets: tuple[int, ...] = (-5, -3, 0, 3, 5)
    calmar_cv_threshold: float = 0.25        # 起点依赖 CV 阈值
    calmar_floor: float = 0.4                 # 参数扰动 Calmar 下限
    ablation_drop_threshold: float = 0.05     # 消融退化阈值
    min_calmar_threshold: float = 0.4

@dataclass
class ValidationResult:
    name: str
    passed: bool
    summary: str
    table: pd.DataFrame  # 详细数据

@dataclass
class ValidationReport:
    actions: list[ValidationResult]
    passed: int
    failed: int

    def to_markdown(self) -> str:
        """生成报告 markdown (匹配 data/real/validation_report.md 格式)."""

def validate_starting_points(
    panel: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    vcfg: ValidationConfig | None = None,
) -> ValidationResult:
    """起点依赖测试: 在多个起点重跑回测, 检查 Calmar 变异系数 ≤ 25%."""

def validate_rebalance_offsets(
    panel: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    vcfg: ValidationConfig | None = None,
) -> ValidationResult:
    """调仓日偏移测试: ±5 个交易日内偏移, Calmar 稳定性."""

def validate_parameter_perturbation(
    panel: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    vcfg: ValidationConfig | None = None,
) -> ValidationResult:
    """参数扰动测试: lookback/corr_threshold/a_share_cap 扰动, Calmar > 0.4."""

def ablation(
    panel: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    vcfg: ValidationConfig | None = None,
) -> ValidationResult:
    """消融实验: 关闭 4 条规则各做一次, 检查每关一项 Calmar 退化 ≥ 5%."""

def run_full_validation(
    panel: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig,
    vcfg: ValidationConfig | None = None,
) -> ValidationReport:
    """跑全 4 项检验, 输出 ValidationReport."""
```

#### 4.3.2 实现要点
- 复用 `backtest.py.run_rotation_backtest`
- 起点依赖: 在每个起点截取 panel 子集跑回测, 计算 Calmar 的变异系数 (CV = std/mean)
- 调仓日偏移: 修改 `BacktestConfig.freq` 或手动调整 rebal_dates
- 参数扰动: 用 `dataclasses.replace(cfg, lookback=..., corr_threshold=..., ...)`
- 消融: 通过修改 `cfg.diversification` / `cfg.cost_model` 等开关

#### 4.3.3 验收 (来自 data/real/validation_report.md)
```
起点依赖: 4 个起点 (2019/2020/2022/2023) Calmar 变异系数应 ≤ 25%
调仓日偏移: 5 个偏移 (-5/-3/0/3/5) Calmar 变异系数应 ≤ 15%
参数扰动: 所有 lookback (80/100/120) Calmar 都 > 0.4
消融: 关闭任一规则 Calmar 退化 ≥ 5%
```

#### 4.3.4 测试用例 (新增 tests/strategy/momentum_etf_rotation/test_validation.py)
```python
def test_validate_starting_points() -> None:
    """4 起点 Calmar CV 应 < 25%."""

def test_validate_rebalance_offsets() -> None:
    """5 偏移 Calmar CV 应 < 15%."""

def test_run_full_validation_returns_markdown() -> None:
    """run_full_validation(...).to_markdown() 应包含 '起点依赖' / '消融' / '总结'."""

def test_ablation_baseline_vs_off() -> None:
    """消融: 关掉逆波动 Calmar 应退化 ≥ 5%."""
```

---

### 4.4 `extended_metrics.py` — 17 个业绩指标

#### 4.4.1 签名（从 `reports/extended_metrics.json` 反推）

```python
def extended_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    rebalance_dates: list[pd.Timestamp] | None = None,
) -> dict:
    """17 个业绩指标.
    
    关键指标:
        ann_return, ann_vol, sharpe, max_drawdown, calmar
        sortino, downside_dev, info_ratio, win_rate
        profit_loss_ratio, max_dd_duration, calmar_avg_dd
        var_95, cvar_95, ann_turnover, max_monthly_loss
        profit_months_ratio, period_returns
    """

def format_metrics_table(
    metrics_inv_vol: dict,
    metrics_equal_weight: dict,
    output_path: Path | None = None,
) -> str:
    """生成 17 指标对比 markdown 表 (matches reports/extended_metrics.md)."""
```

#### 4.4.2 实现要点
- 基础 5 个指标 (ann_return, ann_vol, sharpe, max_dd, calmar) 与 `fi_plus.py:performance_metrics` 一致
- Info Ratio: `(strategy - benchmark).mean() / strategy.std()` annualized
- VaR/CVaR: 历史法 95% 分位数
- Calmar(avg DD): calmar / abs(mean_dd) (用所有回撤段的平均)
- 盈利月比例: `nav.resample('ME').last().pct_change() > 0` 的比例
- `format_metrics_table` 用 `reports/extended_metrics.md` 的格式

#### 4.4.3 测试用例 (新增 tests/.../test_extended_metrics.py)
```python
def test_extended_metrics_keys() -> None:
    """返回 dict 应包含全部 17 个键."""

def test_extended_metrics_synthetic() -> None:
    """合成 5 年单调上涨 nav, sharpe 应 > 1, calmar > 1."""

def test_format_metrics_table() -> None:
    """应生成含 17 行的 markdown 表."""
```

---

### 4.5 `contribution.py` — 5 维度归因

#### 4.5.1 签名（从 `reports/etf_contribution.csv` + `category_contribution.csv` + `risk_contribution.csv` + `marginal_contribution.csv` + `period_contribution.csv` 反推）

```python
DEFAULT_PERIODS: list[tuple[str, str, str]] = [
    # (name, start, end) — 来自 period_contribution.csv
    ("2019 普涨", "2019-01-01", "2019-12-31"),
    ("2020H1 COVID", "2020-01-01", "2020-06-30"),
    ("2020H2-2021 反弹", "2020-07-01", "2021-12-31"),
    ("2022 熊市", "2022-01-01", "2022-12-31"),
    # ... 共 ~10 个 period
]

def reconstruct_daily_weights(
    states: list[PortfolioState],
    rebalance_dates: list[pd.Timestamp],
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """重建每日权重 (前向填充 rebalance 之间)."""

def etf_contribution(
    nav_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    pool: ETFPool,
) -> pd.DataFrame:
    """ETF 维度贡献: code, frequency, avg_weight, total_return, return_contrib
    Returns DataFrame 排序后输出到 etf_contribution.csv."""

def category_contribution(
    weights_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    pool: ETFPool,
) -> pd.DataFrame:
    """类别维度: category, avg_weight, return_contrib, frequency, n_codes."""

def risk_contribution(
    weights_df: pd.DataFrame,
    cov: np.ndarray,  # 用 covariance.estimate_covariance
    codes: list[str],
) -> pd.DataFrame:
    """风险贡献: code, avg_weight, vol_contrib, var_contrib."""

def marginal_contribution(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cov: np.ndarray,
    codes: list[str],
) -> pd.DataFrame:
    """边际贡献: code, correlation, cov_i_p, marginal_sharpe."""

def period_contribution(
    weights_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    periods: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """周期贡献: period, n_days, ann_return, max_drawdown, calmar, period_return, return_pct_of_total."""
```

#### 4.5.2 实现要点
- `reconstruct_daily_weights`: `weights_df = pd.DataFrame(index=trading_dates, columns=codes)`，在 rebalance_dates 填充 weights，其余日期 ffill
- `etf_contribution`: `return_contrib = avg_weight * total_return` (per ETF)，frequency = `weights > 0` 的天数比例
- `category_contribution`: 累加同 category 的 ETF 贡献
- `risk_contribution`: `vol_contrib = w_i * (Σw @ cov)_i / total_vol`，`var_contrib = vol_contrib * corr_i_p`
- `marginal_contribution`: `dSharpe/dw_i`，需要 cov 求逆
- `period_contribution`: 对每个 period 切片, 算 ann_return / max_dd / calmar / 期间收益 / 占总收益比例

#### 4.5.3 测试用例 (新增 tests/.../test_contribution.py)
```python
def test_etf_contribution_columns() -> None:
    """DataFrame 应包含 5 个期望列."""

def test_reconstruct_daily_weights() -> None:
    """Daily weights 应与 rebalance weights 一致 + ffill."""

def test_category_contribution_sums() -> None:
    """各 category 贡献之和应 ≈ 总收益."""
```

---

### 4.6 `brinson.py` — Brinson 归因

#### 4.6.1 签名（从 `__init__.py` line 71 + `reports/brinson.json` 反推）

```python
CATEGORIES: list[str] = [
    "a_broad", "a_sector", "hk", "commodity", "overseas"
]

def brinson_attribution(
    portfolio_weights: pd.DataFrame,  # index=date, columns=codes
    portfolio_returns: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    pool: ETFPool,
) -> dict:
    """Brinson 归因模型.
    
    Returns dict with keys:
        allocation_abs, selection_abs, interaction_abs
        allocation_pct, selection_pct, interaction_pct
        total_active, port_total_return, bench_total_return
    
    公式 (来自 SingleBrinson 类, archive/quantnodes_deprecated/brinson.py):
        allocation = (wp - wb) @ rb
        selection = wb @ (rp - rb)
        interaction = (wp - wb) @ (rp - rb)
        active = allocation + selection + interaction
    """
```

#### 4.6.2 实现要点
- 加权基准权重: `wb = np.ones(N) / N` 或 CSI300 等权
- 单期 attribution 公式 (从 archive 复现)
- 累加多期: `allocation_abs = sum(allocation)`
- Pct: `allocation_pct = allocation_abs / abs(total_active)`

#### 4.6.3 测试用例 (新增 tests/.../test_brinson.py)
```python
def test_brinson_attribution_keys() -> None:
    """返回 dict 应包含全部 9 个键."""

def test_brinson_zero_active() -> None:
    """当 wp == wb 且 rp == rb 时, total_active 应 ≈ 0."""
```

---

## 五、__init__.py 修复

当前 `__init__.py` 已经从 git HEAD 上是损坏的。**不需要修改 imports，只需要补齐 5 个模块即可**。

如果用户希望清理 `__all__` 中已弃用的项，可选：
- 移除 `apply_trend_filter`/`apply_vol_targeting` 等已废弃的导出（保留向后兼容）
- 移除 `data.py`/`data_tencent.py` 的导出从 `__init__.py`（它们是 fetch 工具，不应暴露）

---

## 六、实施顺序

### Phase A: 数据加载层（1d）
1. **`data.py`** — `load_etf_nav_panel` + `load_bond_etf_nav`
2. **`data_tencent.py`** — `fetch_one_etf_tencent` + `write_fetch_log`
3. 测试: `tests/strategy/momentum_etf_rotation/test_data_loader.py`

### Phase B: 抗过拟合检验（1-2d）
4. **`validation.py`** — `ValidationConfig` + 5 个 validate_* + `run_full_validation`
5. 测试: `tests/strategy/momentum_etf_rotation/test_validation.py`
6. 验证: `tests/agent/test_validation_tool.py` 自动通过

### Phase C: 业绩指标（1d）
7. **`extended_metrics.py`** — 17 个指标 + `format_metrics_table`
8. 测试: `tests/strategy/momentum_etf_rotation/test_extended_metrics.py`

### Phase D: 归因分析（1-2d）
9. **`contribution.py`** — 5 个维度函数 + `reconstruct_daily_weights` + `DEFAULT_PERIODS`
10. 测试: `tests/strategy/momentum_etf_rotation/test_contribution.py`

### Phase E: Brinson 归因（0.5d）
11. **`brinson.py`** — `brinson_attribution` + `CATEGORIES`
12. 测试: `tests/strategy/momentum_etf_rotation/test_brinson.py`

### Phase F: 集成验证（1d）
13. **运行** `scripts/fetch_real_etf_panel.py --refresh=false`（应从缓存恢复）
14. **运行** `python -c "from QuantNodes.strategy.momentum_etf_rotation import *"`（应成功）
15. **运行** `python -m pytest tests/strategy/momentum_etf_rotation/ -v`（应全过）
16. **重新生成** `reports/momentum_etf_rotation/` 输出并与历史报告对比（Calmar 偏差 ≤ 1%）

### Phase G: 提交推送（0.5d）
17. **Commit**: `recover(stage11-lost): restore 5 missing modules from data/real + reports`
18. **Push**: `origin/dev/repro-merge-2026-07-04`

**总预估**: 6-9 天

---

## 七、文件变更清单

### 新增 (~12 文件)
```
QuantNodes/strategy/momentum_etf_rotation/data.py        [Phase A]
QuantNodes/strategy/momentum_etf_rotation/data_tencent.py [Phase A]
QuantNodes/strategy/momentum_etf_rotation/validation.py  [Phase B]
QuantNodes/strategy/momentum_etf_rotation/extended_metrics.py  [Phase C]
QuantNodes/strategy/momentum_etf_rotation/contribution.py [Phase D]
QuantNodes/strategy/momentum_etf_rotation/brinson.py     [Phase E]
tests/strategy/momentum_etf_rotation/test_data_loader.py  [Phase A]
tests/strategy/momentum_etf_rotation/test_validation.py   [Phase B]
tests/strategy/momentum_etf_rotation/test_extended_metrics.py  [Phase C]
tests/strategy/momentum_etf_rotation/test_contribution.py [Phase D]
tests/strategy/momentum_etf_rotation/test_brinson.py      [Phase E]
docs/33-CICC动量ETF轮动-缺失模块重写计划.md                [本文件]
```

### 修改 (~2 文件)
```
QuantNodes/strategy/momentum_etf_rotation/__init__.py  (仅调整 __all__ 顺序/可选清理)
tests/strategy/momentum_etf_rotation/test_concentration.py  (可能添加 import 路径测试)
```

---

## 八、与现有代码的关系

| 已有模块 | 关系 |
|---------|------|
| `universe.py` | 20 默认池 stub, 需要扩展到 44 (与 data/real/ 对齐) |
| `momentum.py` | rank_by_momentum / rank_pctl / distance_to_52w_high |
| `portfolio.py` | select_and_weight / apply_stops / inverse_vol_weights |
| `covariance.py` | estimate_covariance (4 种估计器) — contribution.py 复用 |
| `risk_parity.py` | solve_risk_parity |
| `regime_detector.py` | HMMRegimeDetector |
| `fi_plus.py` | FixedIncomePlus 80/20 + performance_metrics (基础指标) |
| `backtest.py` | run_rotation_backtest (validation.py 复用) |

**复用关系**:
- `validation.py` 复用 `backtest.py` 跑多组参数
- `extended_metrics.py` 复用 `fi_plus.py:performance_metrics` 的基础逻辑
- `contribution.py` 复用 `covariance.py:estimate_covariance` 算风险贡献
- `brinson.py` 复用 `universe.py:Category` 分类

---

## 九、测试策略

### 单元测试 (tests/strategy/momentum_etf_rotation/)
- 每个新模块独立测试
- 使用合成数据 (固定 seed)
- 关键边界: 空数据、单点数据、NaN、inf

### 集成测试
- `tests/agent/test_validation_tool.py` 应自动通过（已存在）
- `scripts/fetch_real_etf_panel.py --refresh=false` 应成功（缓存命中）
- `python -c "from QuantNodes.strategy.momentum_etf_rotation import *"` 应成功

### 回归测试 (against reports/)
- 重新跑 `extended_metrics(nav_44_etf)` → 与 `reports/extended_metrics.json` 对比, 偏差 < 1e-6
- 重新跑 `brinson_attribution(...)` → 与 `reports/brinson.json` 对比, 偏差 < 1e-6
- 重新跑 `etf_contribution(...)` → 与 `reports/etf_contribution.csv` 对比, 偏差 < 1e-6

### 端到端验证
- `python -m pytest tests/strategy/momentum_etf_rotation/ -v` 应全过
- 整体 `tests/`: 应保持现有 5682 passed (无回归)
- `tests/agent/test_validation_tool.py`: ERROR → PASS

---

## 十、风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| `reports/*.json` 中字段名与 `__init__.py` 引用不一致 | 反推失败 | 优先用 `__init__.py` 和 tests 的引用为准 |
| 缺失文件的 import 顺序错乱 | 循环引用 | 在模块顶部 lazy import 或重排 |
| 重写后数字与历史 reports 不一致 | 回归 | 用差分测试对比，偏差 < 1% |
| Brinson 模型有多种实现方式 | 公式错误 | 复用 `archive/quantnodes_deprecated/brinson.py` 公式 |
| `run_rotation_backtest` 接口变更 | 测试失败 | 保持现有 8 个测试 (76 passed) 不破坏 |

---

## 十一、与 TradingAgents Phase 1 的关系

按用户的回答 "完成 CICC 修复后做 Phase 1"：

1. **当前任务**: 完成 docs/33 计划的 6 个 Phase (A→F→G), 约 6-9 天
2. **完成后**: 开始 TradingAgents Phase 1 (structured_judge), 约 3-4 天
3. **总计**: ~10-13 天后才到 Phase 1 完成

**建议的代码组织**: 把 CICC 修复工作保留在 `dev/repro-merge-2026-07-04` 分支，TradingAgents Phase 1 在新分支或 master 上做，避免冲突。

---

## 十二、待决策

| # | 问题 | 默认 |
|---|------|------|
| 1 | universe.py 是否扩展到完整 44 ETF (当前 20)? | ✅ 是（与 data/real 一致）|
| 2 | Brinson 公式使用何版本? | `archive/quantnodes_deprecated/brinson.py` (SingleBrinson/MultiBrinson) |
| 3 | DEFAULT_PERIODS 用哪几个 period? | 与 `period_contribution.csv` 实际行对齐 |
| 4 | 测试 tolerance? | 偏差 < 1e-6 (与 reports 完全一致) |
| 5 | 是否需要修 `__init__.py` 的 `__all__`? | ❌ 不需要 (向后兼容) |

---

## 十三、相关文档

| 文档 | 内容 |
|------|------|
| docs/17 | CICC 复现总体说明 + 4 条核心规则 |
| docs/27 | TradingAgents 可视化升级 (暂搁) |
| docs/28-32 | TradingAgents 调研 + 6 工具设计 |
| **docs/33** | **本计划 (CICC 修复)** |
| reports/momentum_etf_rotation/STAGE_SUMMARY.md | Stage 7-13 总结 |
| reports/momentum_etf_rotation/DECISION_LOG.md | go/no-go 决策 |
| reports/momentum_etf_rotation/GAP_ANALYSIS.md | 与 CICC 报告差距 (不可消除) |
| reports/momentum_etf_rotation/validation_fix_report.md | Stage 7 修复报告 |
| data/real/validation_report.md | 当前 1/4 通过的样例输出 |