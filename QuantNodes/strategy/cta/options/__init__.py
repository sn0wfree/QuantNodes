# coding=utf-8
"""CTA 期权工具包.

包含:
    - symbol_parser: 6 种交易所期权 symbol 统一解析
    - greeks:        期权定价/IV/希腊字母 (封装 vendored option_greeks)
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
]
