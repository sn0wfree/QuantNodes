# coding=utf-8
"""Commodity Fundamental — 黄金品种子策略包.

双轨策略体系:
    进攻轨: gold_trend_bnh (8%止损+VIX动量, 跑赢B&H Sharpe/Calmar)
    防御轨: gold_all_factor (全因子5层合成, 低回撤)

策略清单:
    gold_real_rate       实际利率择时 (TIPS MA信号)
    gold_usd             美元趋势 (DXY动量信号)
    gold_risk            避险情绪 (VIX+GPRD联合)
    gold_breakout        突破+追踪止损 (20日高点+3%止损)
    gold_momentum        长期动量 (52周动量)
    gold_position_flow   仓位流 (央行+ETF)
    gold_trend_bnh       趋势跟踪B&H (8%止损+VIX动量) - 进攻轨
    gold_all_factor      全因子分层合成 (5层14指标) - 防御轨
    gold_multi_factor    多因子等权组合 (6因子)
    gold_combined        因子择时组合 (v2)

主策略: gold_trend_bnh (进攻) + gold_all_factor (防御)
2018-2025: 趋势B&H+VIX动量 16.59%, Sharpe1.37 MaxDD-14.83%
"""
from __future__ import annotations

from .gold_real_rate import (
    load_gold_price_w,
    load_indicator_w,
    compute_real_rate_signal,
    backtest_real_rate,
    get_weekly_returns,
)
from .gold_usd import (
    load_dxy_w,
    compute_usd_signal,
    backtest_usd,
    get_weekly_returns as get_usd_weekly_returns,
)
from .gold_risk import (
    compute_risk_signal,
    backtest_risk,
    get_weekly_returns as get_risk_weekly_returns,
)
from .gold_breakout import (
    compute_breakout_signal,
    backtest_breakout,
    get_weekly_returns as get_breakout_weekly_returns,
)
from .gold_momentum import (
    compute_momentum_signal,
    backtest_momentum,
    get_weekly_returns as get_momentum_weekly_returns,
)
from .gold_position_flow import (
    compute_position_flow_signal,
    backtest_position_flow,
    get_weekly_returns as get_position_flow_weekly_returns,
)
from .gold_trend_bnh import (
    compute_trend_bnh_signal,
    backtest_trend_bnh,
    get_weekly_returns as get_trend_bnh_weekly_returns,
    compute_hysteresis_trend_d,
    compute_macro_regime_signal,
    compute_risk_overlay,
    compute_adaptive_ensemble,
    compute_layered_ensemble_signal,
    load_gold_price_d_with_open,
)
from .gold_multi_factor import (
    compute_multi_factor_signal,
    backtest_multi_factor,
    get_weekly_returns as get_multi_factor_weekly_returns,
)
from .gold_combined import (
    compute_combined_signal,
    backtest_combined,
)
from .gold_all_factor import (
    compute_composite_score,
    compute_all_factor_signal,
    backtest_all_factor,
    get_weekly_returns as get_all_factor_weekly_returns,
)

__all__ = [
    "compute_real_rate_signal",
    "backtest_real_rate",
    "compute_usd_signal",
    "backtest_usd",
    "compute_risk_signal",
    "backtest_risk",
    "compute_breakout_signal",
    "backtest_breakout",
    "compute_momentum_signal",
    "backtest_momentum",
    "compute_position_flow_signal",
    "backtest_position_flow",
    "compute_trend_bnh_signal",
    "backtest_trend_bnh",
    "compute_hysteresis_trend_d",
    "compute_macro_regime_signal",
    "compute_risk_overlay",
    "compute_adaptive_ensemble",
    "compute_layered_ensemble_signal",
    "load_gold_price_d_with_open",
    "compute_multi_factor_signal",
    "backtest_multi_factor",
    "compute_combined_signal",
    "backtest_combined",
    "compute_composite_score",
    "compute_all_factor_signal",
    "backtest_all_factor",
    "load_gold_price_w",
    "load_indicator_w",
    "load_dxy_w",
]