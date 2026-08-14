# coding=utf-8
"""CTA 期权工具包.

包含:
    - symbol_parser: 6 种交易所期权 symbol 统一解析
    - greeks:        期权定价/IV/希腊字母 (封装 vendored option_greeks)
    - underlying_map: 期权标的 → 期货价格映射
    - expiry:        期权到期日推断 (数据驱动)
    - iv_panel:      ATM IV 面板 + 分位 + DuckDB 缓存
    - selector:      虚值 Put/Call 合约选择器 (目标 delta)
    - hedge_layer:   对冲框架接口 (策略逻辑后延)
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
from .selector import OptionSelection, select_call, select_put
from .hedge_layer import HedgeConfig, compute_hedge_targets, iv_coefficient

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
    "select_call",
    "select_put",
    "HedgeConfig",
    "compute_hedge_targets",
    "iv_coefficient",
]
