# v0-v10 代码重构方案：公共函数提取

> **日期**: 2026-07-27
> **目标**: 提取 v0-v10 中重复实现的公共函数，消除指标计算不统一、回测逻辑散落等问题
> **原则**: 只规划不写代码；遵循 AGENTS.md 精准修改原则

---

## 一、现状问题

### 1.1 指标计算：6 种实现

| 位置 | 年化收益 | 波动率 | 频率检测 |
|------|---------|--------|---------|
| `combo/run_all_strategies.py` | 日历日/365.25 | sqrt(252) | 无 |
| `combo/unified_v1v5_compare.py` | 日历日/365.25 | sqrt(252) | 无 |
| `combo/nav_curves_html.py` | 252/(len-1) | sqrt(252) | 无 |
| `v4/multi_strategy_v4.py` | freq/n | sqrt(freq) | 已修复自动检测 |
| `v9/backtest.py` | len/freq | sqrt(freq) | freq 参数 |
| `v10/dynamic_weight_schemes.py` | 日历日/365.25 | sqrt(252) | 无 |

**问题**: 同一个策略在不同脚本中计算出不同指标。

### 1.2 回测循环：5 种实现

| 位置 | 版本 | 特点 |
|------|------|------|
| `backtest.py::run_rotation_backtest()` | v0-v2 | groupby 月度, 日频复利 |
| `v3/multi_strategy_v3.py` | v3 | 多子策略, 月度调仓 |
| `v4/multi_strategy_v4.py` | v4 | 多子策略, 月度调仓, clip ret |
| `combo/run_all_strategies.py` | v5/v5.1 | 内联回测, resample 月度 |
| `v10/dynamic_weight_schemes.py` | v10 | 权重历史 + compute_nav() |

**问题**: 每个版本重写回测循环，引入微妙差异（clip、cost 计算、调仓日选择）。

### 1.3 数据加载：4 种实现

| 位置 | 加载方式 |
|------|---------|
| `combo/load_unified_data.py` | 52 ETF, OHLCV+NAV 混合 |
| `v5/__init__.py` | 44 ETF OHLCV |
| `v7/data_loader_v7_6.py` | 43 ETF 周频 + 宏观因子 |
| `v10/dynamic_weight_schemes.py` | 4 策略 NAV parquet |

### 1.4 调仓日生成：3 种方式

| 方式 | 代码位置 | 行为差异 |
|------|---------|---------|
| `groupby(period).max()` | backtest.py, v3 | 返回实际交易日 |
| `resample("M").last().index` | v5, combo | 返回日历月末（过滤后等价） |
| `resample('M').last().index` | v10, dual_momentum | 同上 |

---

## 二、重构方案

### 2.1 新建 `common/metrics.py` — 统一指标计算

**职责**: 所有版本共用的业绩指标计算

**接口设计**:

```python
def compute_metrics(
    nav: pd.Series,
    freq: str | int | None = None,  # 'D'=252, 'W'=52, 'M'=12, None=自动检测
    oos_start: str | None = None,    # OOS 起始日
) -> dict:
    """
    统一指标计算:
    - 年化收益: (1+total)^(1/n_years)-1, n_years = 日历日/365.25
    - 波动率: std * sqrt(freq)
    - Sharpe: ann_ret / vol
    - Sortino: ann_ret / downside_vol
    - Calmar: ann_ret / |max_dd|
    - MaxDD, WinRate, PayoffRatio, MaxDDDays
    
    freq 自动检测: 中位数日间隔 > 4天 → 52, 否则 252
    """
```

**关键决策**:
- 年化收益统一用**日历日/365.25**（而非 len/freq），与行业惯例一致
- 波动率用 **sqrt(freq)** 但 freq 必须正确匹配数据
- OOS 指标作为可选输出，避免每次都切片

**受影响文件**（需改为调用此模块）:
- `combo/run_all_strategies.py` → 删除自定义 `metrics()`, `sharpe()`, `ann_return()`
- `combo/unified_v1v5_compare.py` → 同上
- `combo/nav_curves_html.py` → 同上
- `v4/multi_strategy_v4.py` → `_performance_metrics()` 改为调用
- `v9/backtest.py` → `compute_metrics()` 改为调用
- `v10/dynamic_weight_schemes.py` → `metrics()` 改为调用
- `scripts/v5_backtest.py` → `metrics()`, `sharpe()` 改为调用

### 2.2 新建 `common/backtest_engine.py` — 统一回测循环

**职责**: 标准化的 NAV 计算循环

**接口设计**:

```python
def compute_nav_from_weights(
    prices: pd.DataFrame,          # 日频价格面板
    weights_history: pd.DataFrame, # 权重历史 (对齐到调仓日)
    cost_bp: float = 0,            # 交易成本 (单边, basis points)
    rebal_freq: str = 'M',         # 调仓频率
    clip_ret: float | None = None, # 收益裁剪 (如 0.5 = ±50%)
) -> pd.Series:
    """
    标准 NAV 计算:
    1. 日频收益 = prices.pct_change()
    2. 组合收益 = sum(weight * return)
    3. 调仓日计算换手成本
    4. NAV = cumprod(1 + port_ret - cost)
    """
```

**关键决策**:
- 权重只在调仓日变化，中间日 forward-fill
- 成本只在调仓日收取（`turnover * cost_bp / 10000`）
- `clip_ret` 可选，默认不裁剪
- 价格面板必须是日频（调用方负责对齐）

**受影响文件**:
- `v10/dynamic_weight_schemes.py` → `compute_nav()` 改为调用
- `v4/multi_strategy_v4.py` → NAV 循环改为调用
- `combo/run_all_strategies.py` → v5/v5.1 内联回测改为调用
- `backtest.py` → `run_rotation_backtest()` 内部可调用（保留外层接口）

### 2.3 新建 `common/rebalance.py` — 统一调仓日生成

**职责**: 标准化的调仓日计算

**接口设计**:

```python
def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str = 'M',  # 'M'=月末, 'W'=周末, 'Q'=季末
) -> list[pd.Timestamp]:
    """
    返回实际交易日中的调仓日。
    使用 groupby(period).max() 确保返回真实交易日。
    """
```

**受影响文件**:
- `combo/run_all_strategies.py` → `resample("M").last()` 改为调用
- `combo/unified_v1v5_compare.py` → 同上
- `v5_backtest.py` → 同上
- `v10/dual_momentum.py` → 同上

### 2.4 新建 `common/data_loader.py` — 统一数据加载

**职责**: 标准化的数据加载接口

**接口设计**:

```python
def load_etf_panel(
    start: str = '2018-01-01',
    end: str = '2026-06-30',
    pool: str = 'main44',  # 'main44', 'smartbeta12', 'all52'
    field: str = 'close',  # 'close', 'nav', 'ohlcv'
) -> pd.DataFrame:
    """加载 ETF 日频面板"""

def load_strategy_nav(
    name: str,  # 'v1.0', 'v7.10', 'v9macro', 'DualMom'
) -> pd.Series:
    """加载已生成的策略 NAV"""
```

### 2.5 整合到 `common/__init__.py`

```python
from .metrics import compute_metrics
from .backtest_engine import compute_nav_from_weights
from .rebalance import get_rebalance_dates
from .data_loader import load_etf_panel, load_strategy_nav
```

---

## 三、重构步骤（按依赖顺序）

### 阶段 1: 基础模块（无外部依赖）

| 步骤 | 操作 | 验证 |
|------|------|------|
| 1.1 | 创建 `common/metrics.py` | 单测：对比现有 6 种实现输出 |
| 1.2 | 创建 `common/rebalance.py` | 单测：对比 groupby vs resample 输出 |
| 1.3 | 创建 `common/data_loader.py` | 集成测试：加载数据验证形状 |

### 阶段 2: 回测引擎（依赖阶段 1）

| 步骤 | 操作 | 验证 |
|------|------|------|
| 2.1 | 创建 `common/backtest_engine.py` | 用 v10 静态权重验证 NAV 一致 |
| 2.2 | 用 backtest_engine 重新生成 v10 parquet | 对比新旧 NAV 差异 < 1e-10 |

### 阶段 3: 逐版本迁移（依赖阶段 2）

| 步骤 | 操作 | 风险 |
|------|------|------|
| 3.1 | v10: `dynamic_weight_schemes.py` → 调用公共模块 | 低（已修复） |
| 3.2 | v5/v5.1: `combo/run_all_strategies.py` → 调用公共模块 | 低 |
| 3.3 | v0-v3: `combo/unified_v1v5_compare.py` → 调用公共模块 | 中（lookback 已修正） |
| 3.4 | v4: `multi_strategy_v4.py` → 调用公共模块 | 中（warmup 问题） |
| 3.5 | v9: `backtest.py` → 调用公共模块 | 低 |
| 3.6 | `nav_curves_html.py` → 调用公共 metrics | 低 |

### 阶段 4: 验证

| 步骤 | 操作 |
|------|------|
| 4.1 | 全量重算所有 parquet/CSV |
| 4.2 | 对比新旧指标差异（应 < 0.1%） |
| 4.3 | 更新 VERSION_TRACKING.md |
| 4.4 | 更新 77-v0_v10_codebase_audit.md |

---

## 四、目录结构变更

```
strategy/momentum_etf_rotation/
├── common/
│   ├── __init__.py              # 导出公共接口
│   ├── metrics.py               # NEW: 统一指标计算
│   ├── backtest_engine.py       # NEW: 统一回测循环
│   ├── rebalance.py             # NEW: 统一调仓日生成
│   ├── data_loader.py           # NEW: 统一数据加载
│   ├── extended_metrics.py      # 现有（保留，依赖 metrics.py）
│   ├── drawdown_controller.py   # 现有
│   └── universe.py              # 现有
├── v0/ (无目录，配置在 strategy_versions.py)
├── v3/
├── v4/
├── v5/
├── v5_1/
├── v7/
├── v9/
├── v10/
├── backtest.py                  # v0-v2 回测引擎
├── strategy_versions.py         # 版本配置工厂
└── VERSION_TRACKING.md
```

---

## 五、风险评估

| 风险 | 等级 | 缓解措施 |
|------|------|---------|
| 重构引入新 bug | 中 | 每步迁移后对比新旧 NAV 差异 |
| v4 warmup 问题恶化 | 中 | v4 单独处理，不强制统一 |
| 历史 parquet 不一致 | 低 | 阶段 4 全量重算 |
| 性能下降 | 低 | 公共模块纯函数，无额外开销 |

---

## 六、预期收益

1. **消除指标不一致**: 6 种实现 → 1 种
2. **消除频率 bug**: 自动检测替代硬编码
3. **降低维护成本**: 新版本只需调用公共模块
4. **提高可复现性**: 所有版本用相同指标口径

---

*文档版本: 1.0*
*日期: 2026-07-27*
