# coding=utf-8
"""cta_shared — CTA因子研究可复用组件库.

沉淀自 cta_cftc_strategy / cta_inventory_strategy / cta_broker_holding /
cta_zn_inventory 四个研究脚本的公共逻辑.

使用:
    from scripts.cta_shared.factor_research import (
        rolling_zscore, compute_ic, quintile_spread,
        is_oos_direction_consistency, vol_target_backtest,
        period_stats, yearly_returns, load_cost_bps,
    )

方法论门槛 (固化, 见 docs/cta/cta_cftc_factor.md §8):
    - IC显著 ≠ 可交易
    - 五分位分层 IS/OOS方向一致率 < 60% → 止损
    - IS定方向, OOS独立验证, OOS Sharpe < 0.15 → 止损
"""
from .factor_research import (
    DEFAULT_IS_END,
    REPORT_DIR,
    compute_ic,
    is_oos_direction_consistency,
    load_cost_bps,
    period_stats,
    quintile_spread,
    rolling_zscore,
    summarize_factor_test,
    vol_target_backtest,
    yearly_returns,
)
from .portfolio_tools import (
    adaptive_factor,
    adaptive_factors,
    dynamic_leverage,
    levered_portfolio,
    multi_window_factor,
    multi_window_factors,
    turnover_cost,
    yearly_diff,
)

__all__ = [
    "rolling_zscore", "compute_ic", "quintile_spread",
    "is_oos_direction_consistency", "vol_target_backtest",
    "period_stats", "yearly_returns", "load_cost_bps",
    "summarize_factor_test", "DEFAULT_IS_END", "REPORT_DIR",
    "adaptive_factor", "adaptive_factors", "multi_window_factor",
    "multi_window_factors", "dynamic_leverage", "levered_portfolio",
    "turnover_cost", "yearly_diff",
]
