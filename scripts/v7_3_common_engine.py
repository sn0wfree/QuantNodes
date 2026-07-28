# coding=utf-8
"""v7.3 + TF + 扩展 ETF 池 — 使用 common/backtest_utils 回测.

使用 common/backtest_utils.compute_daily_nav_from_weights() 计算 NAV,
v7.3 逻辑仅负责计算季度权重.
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "QuantNodes").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from QuantNodes.strategy.momentum_etf_rotation.common.backtest_utils import (
    compute_daily_nav_from_weights,
)
from QuantNodes.strategy.momentum_etf_rotation.common.backtest_config import CostConfig
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import (
    load_aligned_prices,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_3 import (
    V7_3Config, V7_4Config, V7_3SubStrategy,
)

log = logging.getLogger("v7_3")


# ============================================================
# 预计算 v7.3 季度权重
# ============================================================
def compute_quarterly_weights(
    price_panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    cfg: V7_3Config,
    benchmark_price: pd.Series | None = None,
) -> list[tuple[pd.Timestamp, dict[str, float]]]:
    """预计算 v7.3 所有季度调仓日的权重.

    返回: [(rebal_date, weights), ...]
    """
    # 构造 sample (与 run_v7_3_backtest 一致): 周频 simple return
    asset_weekly = price_panel[list(cfg.index_pool)].resample("W").last().pct_change()
    factor_weekly = factor_returns[list(cfg.factor_cols)].pct_change()
    how = "all" if isinstance(cfg, V7_4Config) else "any"
    sample = pd.concat([asset_weekly, factor_weekly], axis=1).dropna(how=how)

    # 季度边界
    quarter_idx = pd.DataFrame(index=sample.index).resample(cfg.rebalance_freq).last().index
    quarter_idx = quarter_idx[quarter_idx <= sample.index.max()]

    if len(quarter_idx) <= cfg.quarter_window:
        raise ValueError(f"Insufficient data: need > {cfg.quarter_window} quarters")

    # TF 加载
    if cfg.trend_filter_enabled and benchmark_price is None:
        from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import load_benchmark_price
        benchmark_price = load_benchmark_price(cfg.trend_filter_benchmark)

    sub = V7_3SubStrategy(cfg)
    rebal_dates = list(quarter_idx[cfg.quarter_window:])
    weights_history: list[tuple[pd.Timestamp, dict[str, float]]] = []

    for curr_date in rebal_dates:
        w = sub.select(sample, curr_date)
        if w is None:
            continue
        w_series = pd.Series(w)

        # 趋势过滤
        if cfg.trend_filter_enabled and benchmark_price is not None:
            from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_3 import apply_trend_filter
            w_series = apply_trend_filter(w_series, benchmark_price, curr_date, cfg)

        weights = {col: float(w_series.get(col, 0.0)) for col in cfg.index_pool}
        weights_history.append((curr_date, weights))

    return weights_history


# ============================================================
# 回测函数
# ============================================================
def run_v7_3_backtest_common(
    price_panel: pd.DataFrame,
    factor_returns: pd.DataFrame,
    cfg: V7_3Config,
    benchmark_price: pd.Series | None = None,
    cost_enabled: bool = True,
) -> pd.Series:
    """使用 common/backtest_utils 回测 v7.3, 返回日频 NAV."""
    # 预计算季度权重
    log.info("  预计算季度权重...")
    weights_history = compute_quarterly_weights(
        price_panel, factor_returns, cfg, benchmark_price,
    )
    log.info(f"  {len(weights_history)} 个季度调仓日")

    # 过滤: 只保留在 price_panel 时间范围内的调仓日
    first_date = price_panel.index[0]
    last_date = price_panel.index[-1]
    weights_history = [
        (d, w) for d, w in weights_history
        if first_date <= d <= last_date
    ]
    log.info(f"  过滤后 {len(weights_history)} 个季度调仓日")

    # 日频 simple return (从价格计算)
    daily_returns = price_panel.pct_change()

    # 成本配置
    cost_cfg = CostConfig(
        enabled=cost_enabled,
        flat_cost_bps=(cfg.commission_bp + cfg.slippage_bp),
    )

    # 使用 common 工具计算 NAV
    nav = compute_daily_nav_from_weights(weights_history, daily_returns, cost_cfg)
    return nav


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log.info("=== v7.3 + TF + 扩展 ETF 池 (common/backtest_utils) ===")

    # 加载数据
    log.info("加载数据...")
    data = load_aligned_prices(pool="expanded")
    log.info(f"  asset_prices: {data['asset_prices'].shape}, factor_nav: {data['factor_nav'].shape}")

    # 配置: v7.3 + TF
    cfg = V7_4Config(
        trend_filter_enabled=True,
        trend_filter_ma=200,
        trend_filter_bear=0.5,
        bootstrap_times=100,
    )

    # 回测
    log.info("运行回测...")
    nav = run_v7_3_backtest_common(data["asset_prices"], data["factor_nav"], cfg, data["benchmark"])

    # 计算指标 (common/metrics.py)
    m = compute_metrics(nav, freq="D")
    m_oos = compute_metrics(nav, freq="D", oos_start="2022-01-01")

    print("\n=== Full Period ===")
    print(f"  AnnRet:  {m['AnnRet']*100:+.2f}%")
    print(f"  Vol:     {m['Vol']*100:.2f}%")
    print(f"  Sharpe:  {m['Sharpe']:.3f}")
    print(f"  Sortino: {m['Sortino']:.3f}")
    print(f"  MaxDD:   {m['MaxDD']*100:.2f}%")
    print(f"  Calmar:  {m['Calmar']:.3f}")
    print(f"  WinRate: {m['WinRate']*100:.1f}%")

    print("\n=== OOS 2022-2026 ===")
    for k, v in m_oos["OOS"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
