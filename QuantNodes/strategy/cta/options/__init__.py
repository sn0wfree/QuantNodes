# coding=utf-8
"""CTA 期权工具包.

包含:
    - symbol_parser: 6 种交易所期权 symbol 统一解析
    - greeks:        期权定价/IV/希腊字母 (封装 vendored option_greeks)
    - underlying_map: 期权标的 → 期货价格映射
    - expiry:        期权到期日推断 (数据驱动)
    - iv_panel:      ATM IV 面板 + 分位 + DuckDB 缓存
    - selector:      虚值 Put/Call/PutSpread/Collar/Straddle 合约选择器
    - hedge_layer:   对冲框架 (timing_mode 参数化)
    - hedge_data:    月差 + VRP 面板加载
    - delta_hedge:   Delta 对冲管理器 (portfolio delta → futures hedge)
    - pnl_decompose: P&L 分解 (theta/gamma/vega/delta)
    - roll_manager:  展期管理 (到期前自动换月)
    - slippage:      Bid-Ask 成本模型 (high-low 估计)
    - utils:         工具函数 (cal_pnl, required_share, find_third_friday)
    - contract_marking: c1-c5 主力合约标记 + OTM
    - iv_computer:   IV 批量并行计算 (boost_up)
    - match_engine:  FIFO/LIFO 交易匹配
    - optimizer:     SLSQP 组合优化
    - backtest_bridge: 策略权重 → 交易记录 → P&L
    - strategies:    期权策略池
    - vol_signals:   IVS 斜率 + P/C IV ratio + Regime switching
"""
from __future__ import annotations

from .symbol_parser import OptionSymbol, parse_symbol
from .greeks import (
    CFFEX_UNDERLYING_MAP,
    GreeksResult,
    StraddleGreeks,
    compute_greeks_for_option,
    compute_options_greeks,
    compute_straddle_greeks,
    days_to_expiry,
    map_cffex_underlying,
)
from .underlying_map import UnderlyingMap
from .expiry import OptionExpiry
from .selector import (
    OptionSelection, PutSpread, StraddleSelection,
    select_call, select_collar, select_put, select_put_spread,
    select_straddle, select_option_for_delta,
)
from .hedge_layer import (
    HedgeConfig,
    basis_coefficient,
    compute_hedge_targets,
    iv_coefficient,
    timing_coefficient,
)
from .hedge_data import (
    basis_percentile_panel,
    load_basis_panel,
    load_vrp_detail,
    load_vrp_panel,
    vrp_percentile_panel,
)
from .delta_hedge import DeltaHedgeManager, HedgeRecord
from .pnl_decompose import PnlDecomposer, PnlSnapshot
from .roll_manager import RollManager, RollAction
from .slippage import apply_slippage, estimate_option_spread, estimate_spread

__all__ = [
    "OptionSymbol",
    "parse_symbol",
    "CFFEX_UNDERLYING_MAP",
    "GreeksResult",
    "StraddleGreeks",
    "compute_greeks_for_option",
    "compute_options_greeks",
    "compute_straddle_greeks",
    "days_to_expiry",
    "map_cffex_underlying",
    "UnderlyingMap",
    "OptionExpiry",
    "OptionSelection",
    "PutSpread",
    "StraddleSelection",
    "select_call",
    "select_collar",
    "select_put",
    "select_put_spread",
    "select_straddle",
    "select_option_for_delta",
    "HedgeConfig",
    "compute_hedge_targets",
    "iv_coefficient",
    "basis_coefficient",
    "timing_coefficient",
    "basis_percentile_panel",
    "load_basis_panel",
    "load_vrp_panel",
    "vrp_percentile_panel",
    "load_vrp_detail",
    "DeltaHedgeManager",
    "HedgeRecord",
    "PnlDecomposer",
    "PnlSnapshot",
    "RollManager",
    "RollAction",
    "apply_slippage",
    "estimate_option_spread",
    "estimate_spread",
]
