# StrategyResearch 工具复用设计文档

> **版本**: v1.0
> **日期**: 2026-07-21
> **状态**: 设计完成，待实施

---

## 一、目标

从 `QuantNodes/strategy/momentum_etf_rotation/` 复用通用工具，扩展 `strategy-research` 的能力。

核心原则：
- **复用优先**：直接复制通用工具，不重复造轮子
- **精简依赖**：移除 ETF 特定依赖，保留通用逻辑
- **pandas 实现**：算子使用 pandas 版本，不依赖 polars

---

## 二、可直接复用的工具

### 2.1 性能指标 (`extended_metrics.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/common/extended_metrics.py`

**目标**: `strategy-research/src/strategy_research/core/utils/metrics.py`

**17 个指标**:

| 指标 | 说明 | 公式 |
|------|------|------|
| `ann_return` | 年化收益率 | CAGR from NAV |
| `ann_vol` | 年化波动率 | std(returns) * sqrt(freq) |
| `sharpe` | Sharpe 比率 | ann_return / ann_vol |
| `max_drawdown` | 最大回撤 | min(NAV/cummax - 1) |
| `calmar` | Calmar 比率 | ann_return / abs(max_dd) |
| `sortino` | Sortino 比率 | ann_return / downside_dev |
| `downside_dev` | 下行偏差 | std(negative_returns) * sqrt(freq) |
| `info_ratio` | 信息比率 | mean(excess) / std(excess) * sqrt(freq) |
| `win_rate` | 日胜率 | fraction of positive daily returns |
| `profit_loss_ratio` | 盈亏比 | mean(wins) / abs(mean(losses)) |
| `max_dd_duration` | 最大回撤持续天数 | longest consecutive drawdown run |
| `calmar_avg_dd` | Calmar(avg DD) | calmar / abs(avg_dd) |
| `var_95` | VaR (5%) | 5th percentile of returns |
| `cvar_95` | CVaR (5%) | mean(returns <= VaR) |
| `ann_turnover` | 年化换手率 | placeholder (always 0.0) |
| `max_monthly_loss` | 最大月度亏损 | min(monthly_returns) |
| `profit_months_ratio` | 盈利月占比 | fraction of positive months |

**精简方案**:
- 移除 `format_metrics_table` (ETF 特定格式)
- 保留 `extended_metrics` 函数
- 添加类型注解

### 2.2 回测引擎 (`backtest_engine.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/common/backtest_engine.py`

**目标**: `strategy-research/src/strategy_research/core/utils/backtest_engine.py`

**核心接口**:

```python
class BacktestCallbacks:
    """回调基类。版本特定逻辑通过继承实现。"""

    def compute_signals(self, price_panel, date, state, context) -> dict[str, float]:
        """计算信号分数。REQUIRED."""
        raise NotImplementedError

    def select_assets(self, signals, config) -> list[str]:
        """从信号中选择资产。Default: top_n by score."""

    def compute_weights(self, selected, price_panel, date, config) -> dict[str, float]:
        """计算权重。Default: equal weight."""

    def apply_risk_controls(self, weights, nav_history, date, config) -> dict[str, float]:
        """应用风控。Default: no-op."""

    def post_weights(self, weights, config) -> dict[str, float]:
        """后处理权重。Default: max_weight + normalize."""

@dataclass
class BacktestResult:
    nav_daily: pd.Series
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]]
    rebalance_dates: list[pd.Timestamp]
    metrics: dict

def run_backtest(price_panel, daily_returns=None, config=None, callbacks=None, context=None) -> BacktestResult:
    """运行回测。"""
```

**精简方案**:
- 移除 `from ..fi_plus import performance_metrics` (改用本地 metrics)
- 保留核心回调模式
- 添加类型注解

### 2.3 回测配置 (`backtest_config.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/common/backtest_config.py`

**目标**: `strategy-research/src/strategy_research/core/utils/backtest_config.py`

**配置结构**:

```python
@dataclass
class BacktestConfig:
    rebal_freq: str = "M"               # M/W/Q
    min_history: int = 252               # warm-up periods
    top_n: int = 10                      # select N assets
    max_weight: float = 0.25             # max weight cap
    weight_method: str = "inverse_vol"   # inverse_vol / equal
    vol_window: int = 60                 # vol calculation window
    vol_floor: float = 0.01              # vol floor
    cost: CostConfig = ...               # transaction costs
    vol_targeting: VolTargetingConfig = ...
    trend_filter: TrendFilterConfig = ...
    stop_loss: StopLossConfig = ...
    execution_lag: int = 0               # 0=same-day, 1=T+1
    return_detail: bool = False          # return weight history

@dataclass
class CostConfig:
    enabled: bool = False
    commission_bp: float = 5.0
    slippage_bp: float = 10.0
    impact_factor: float = 0.1
    flat_cost_bps: float | None = None

@dataclass
class VolTargetingConfig:
    enabled: bool = False
    target_vol: float = 0.15
    lookback: int = 60
    min_scale: float = 0.3
    max_scale: float = 2.0

@dataclass
class TrendFilterConfig:
    enabled: bool = False
    benchmark_col: str | None = None
    ma_window: int = 200
    bear_exposure: float = 0.5

@dataclass
class StopLossConfig:
    enabled: bool = False
    threshold: float = -0.10
    cooldown_weeks: int = 5
```

**精简方案**: 直接复制，无需修改

### 2.4 回测工具 (`backtest_utils.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/common/backtest_utils.py`

**目标**: `strategy-research/src/strategy_research/core/utils/backtest_utils.py`

**工具函数**:

| 函数 | 说明 |
|------|------|
| `calculate_turnover(old_weights, new_weights)` | 计算换手率 |
| `calculate_turnover_cost(old_weights, new_weights, cost_cfg)` | 计算换手成本 |
| `generate_rebalance_dates(dates, freq, min_lookback)` | 生成调仓日 |
| `apply_max_weight(weights, max_w, max_iters)` | 应用最大权重约束 |
| `normalize_weights(weights)` | 权重归一化 |
| `compute_daily_nav_from_weights(weights_history, daily_returns, cost_cfg)` | 计算每日 NAV |

**精简方案**: 直接复制，无需修改

### 2.5 协方差估计 (`covariance.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/common/covariance.py`

**目标**: `strategy-research/src/strategy_research/core/utils/covariance.py`

**协方差估计器**:

| 函数 | 说明 |
|------|------|
| `sample_covariance(returns)` | 样本协方差 |
| `ledoit_wolf_shrinkage(returns)` | Ledoit-Wolf 收缩 |
| `ewma_covariance(returns, halflife)` | EWMA 协方差 |
| `diagonal_covariance(returns)` | 对角协方差 |
| `estimate_covariance(returns, method, halflife)` | 统一接口 |
| `is_positive_definite(matrix, tol)` | 正定性检查 |
| `condition_number(matrix)` | 条件数 |

**精简方案**: 直接复制，无需修改

### 2.6 风险平价 (`risk_parity.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/common/risk_parity.py`

**目标**: `strategy-research/src/strategy_research/core/utils/risk_parity.py`

**函数**:

| 函数 | 说明 |
|------|------|
| `risk_contribution(weights, cov)` | 风险贡献 |
| `risk_parity_objective(weights, cov)` | 风险平价目标函数 |
| `solve_risk_parity(cov, max_iter, tol, bounds)` | 求解风险平价权重 |
| `solve_max_diversification(cov, max_iter, tol, bounds)` | 求解最大分散化权重 |

**精简方案**: 直接复制，无需修改

### 2.7 IC 计算工具 (`rd_utils.py`)

**来源**: `QuantNodes/strategy/momentum_etf_rotation/rd_utils.py`

**目标**: `strategy-research/src/strategy_research/core/utils/ic_utils.py`

**函数**:

| 函数 | 说明 |
|------|------|
| `compute_cross_sectional_ic(X_panel, Y, factor_idx, min_obs, start_t)` | 截面 Spearman IC |
| `compute_ic_summary(ic_list)` | IC 汇总统计 (mean/std/ICIR/pct_positive) |
| `compute_time_series_ic(factor_ts, market_ts)` | 时间序列 IC (Pearson + p-value) |

**精简方案**: 移除 ETF 特定依赖，保留通用逻辑

---

## 三、算子扩展计划

### 3.1 当前算子 (16 个)

**文件**: `strategy-research/src/strategy_research/core/compute_factor.py`

**时序算子 (12 个)**:
- `ts_return`, `ts_std`, `ts_corr`, `ts_rank`, `delay`, `delta`
- `ts_max`, `ts_min`, `ts_mean`, `ts_sum`, `ts_skew`, `ts_kurt`

**截面算子 (4 个)**:
- `rank`, `zscore`, `scale`, `winsorize`

### 3.2 新增时序算子 (17 个)

**来源**: `operator_vocab/time_ops.py` (pandas 版本)

| 算子 | 说明 | 实现 |
|------|------|------|
| `ts_median` | N 期中位数 | `series.rolling(window).median()` |
| `ts_var` | N 期方差 | `series.rolling(window).var()` |
| `ts_prod` | N 期乘积 | `series.rolling(window).apply(np.prod, raw=True)` |
| `ts_argmax` | N 期最大值位置 | `series.rolling(window).apply(np.argmax, raw=True)` |
| `ts_argmin` | N 期最小值位置 | `series.rolling(window).apply(np.argmin, raw=True)` |
| `ts_cov` | N 期协方差 | `x.rolling(window).cov(y)` |
| `ts_pct_change` | N 期百分比变化 | `series.pct_change(periods)` |
| `ts_zscore` | N 期滚动 z-score | `(series - rolling_mean) / rolling_std` |
| `ts_decay_linear` | 线性衰减加权 MA | `series.rolling(window).apply(lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True)` |
| `ts_decay_exp` | 指数衰减加权 MA | `series.ewm(halflife=halflife).mean()` |
| `expanding_sum` | 扩展窗口求和 | `series.expanding().sum()` |
| `expanding_mean` | 扩展窗口均值 | `series.expanding().mean()` |
| `expanding_max` | 扩展窗口最大值 | `series.expanding().max()` |
| `expanding_min` | 扩展窗口最小值 | `series.expanding().min()` |
| `ewm_mean` | EWM 均值 | `series.ewm(span=span).mean()` |
| `ewm_std` | EWM 标准差 | `series.ewm(span=span).std()` |
| `ewm_corr` | EWM 相关系数 | `x.ewm(span=span).corr(y)` |

### 3.3 新增截面算子 (11 个)

**来源**: `operator_vocab/section_ops.py` (pandas 版本)

| 算子 | 说明 | 实现 |
|------|------|------|
| `neutralize` | 行业中性化 | `series - series.groupby(group).transform('mean')` |
| `neutralize_market` | 市场中性化 | `series - series.mean()` |
| `group_norm` | 分组标准化 | `series.groupby(group).transform(lambda x: (x - x.mean()) / x.std())` |
| `orthogonalize` | 正交化 | `series - (series.corr(ref) * ref.std() / series.std()) * ref` |
| `mad` | 中位数绝对偏差 | `1.4826 * (series - series.median()).abs().median()` |
| `ic` | Pearson IC | `series.corr(target)` |
| `rank_ic` | Spearman Rank IC | `series.rank().corr(target.rank())` |
| `cross_sectional_rank` | = rank | alias |
| `cross_sectional_zscore` | = zscore | alias |
| `cross_sectional_mean` | 截面均值 | `series.mean()` |
| `cross_sectional_std` | 截面标准差 | `series.std()` |

### 3.4 新增数学算子 (13 个)

**来源**: `operator_vocab/math_ops.py` (pandas 版本)

| 算子 | 说明 | 实现 |
|------|------|------|
| `abs` | 绝对值 | `series.abs()` |
| `log` | 对数 | `np.log(series)` |
| `sign` | 符号函数 | `np.sign(series)` |
| `sqrt` | 平方根 | `np.sqrt(series)` |
| `clip` | 截断 | `series.clip(lower, upper)` |
| `fill_null` | 填充空值 | `series.fillna(value)` |
| `add` | 加法 | `f1 + f2` |
| `sub` | 减法 | `f1 - f2` |
| `mul` | 乘法 | `f1 * f2` |
| `div` | 除法 | `f1 / f2` |
| `where` | 条件选择 | `np.where(condition, true_val, false_val)` |
| `weighted_sum` | 加权求和 | `sum(f * w for f, w in zip(factors, weights))` |
| `combine` | 组合两个因子 | `method(f1, f2)` |

### 3.5 算子总数

| 类别 | 当前 | 新增 | 总计 |
|------|------|------|------|
| 时序算子 | 12 | 17 | 29 |
| 截面算子 | 4 | 11 | 15 |
| 数学算子 | 0 | 13 | 13 |
| **总计** | **16** | **41** | **57** |

---

## 四、文件结构

### 4.1 新增文件

```
strategy-research/src/strategy_research/core/utils/
├── __init__.py              # 工具包入口
├── metrics.py               # 17 指标计算 (精简版)
├── backtest_engine.py       # 回测引擎 (精简版)
├── backtest_config.py       # 回测配置
├── backtest_utils.py        # 回测工具函数
├── covariance.py            # 协方差估计
├── risk_parity.py           # 风险平价
└── ic_utils.py              # IC 计算工具
```

### 4.2 修改文件

```
strategy-research/src/strategy_research/core/
├── compute_factor.py        # 扩展算子数量 (16 → 57)
├── backtest.py              # 使用新工具
└── factor_validate.py       # 使用 ic_utils

strategy-research/pyproject.toml  # 添加 scipy 依赖
```

---

## 五、实施步骤

### 步骤 1: 创建 utils 目录

```bash
mkdir -p research/strategy-research/src/strategy_research/core/utils
```

### 步骤 2: 复制通用工具

| 源文件 | 目标文件 | 精简内容 |
|--------|---------|---------|
| `common/extended_metrics.py` | `utils/metrics.py` | 移除 `format_metrics_table` |
| `common/backtest_engine.py` | `utils/backtest_engine.py` | 移除 `fi_plus` 依赖 |
| `common/backtest_config.py` | `utils/backtest_config.py` | 直接复制 |
| `common/backtest_utils.py` | `utils/backtest_utils.py` | 直接复制 |
| `common/covariance.py` | `utils/covariance.py` | 直接复制 |
| `common/risk_parity.py` | `utils/risk_parity.py` | 直接复制 |
| `rd_utils.py` | `utils/ic_utils.py` | 移除 ETF 特定函数 |

### 步骤 3: 扩展算子

修改 `compute_factor.py`，添加 41 个新算子。

### 步骤 4: 更新 backtest.py

使用新的 `utils/backtest_engine.py` 和 `utils/metrics.py`。

### 步骤 5: 更新 pyproject.toml

```toml
dependencies = [
    "duckdb>=0.9.0",
    "pandas>=2.1.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "scipy>=1.10.0",  # 新增
]
```

### 步骤 6: 测试

```bash
# 安装
pip install -e ~/Public/QuantNodes/research/strategy-research

# 测试算子
python -c "from strategy_research.core.compute_factor import get_available_operators; print(len(get_available_operators()))"

# 测试指标
python -c "from strategy_research.core.utils.metrics import extended_metrics; print('OK')"

# 测试回测
quantnodes-research run /tmp/test -s test
```

---

## 六、依赖关系

```
strategy-research
├── core/
│   ├── db.py
│   ├── factor_validate.py
│   │   ├── compute_factor.py
│   │   └── ic_utils.py (新增)
│   ├── backtest.py
│   │   ├── db.py
│   │   ├── git.py
│   │   ├── utils/metrics.py (新增)
│   │   └── utils/backtest_engine.py (新增)
│   ├── compute_factor.py (扩展)
│   └── utils/ (新增)
│       ├── metrics.py
│       ├── backtest_engine.py
│       ├── backtest_config.py
│       ├── backtest_utils.py
│       ├── covariance.py
│       ├── risk_parity.py
│       └── ic_utils.py
└── cli.py
```

---

## 七、时间估计

| 步骤 | 时间 |
|------|------|
| 创建 utils 目录 | 0.1 小时 |
| 复制通用工具 | 1 小时 |
| 扩展算子 | 2 小时 |
| 更新 backtest.py | 0.5 小时 |
| 更新 pyproject.toml | 0.1 小时 |
| 测试 | 0.5 小时 |
| **总计** | **4.2 小时** |

---

## 八、验收标准

1. ✅ `utils/` 目录包含 7 个工具文件
2. ✅ `compute_factor.py` 包含 57 个算子
3. ✅ `backtest.py` 使用新的工具
4. ✅ 所有测试通过
5. ✅ 无 ETF 特定依赖
6. ✅ 纯 pandas 实现
