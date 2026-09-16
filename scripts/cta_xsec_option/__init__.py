# coding=utf-8
"""CTA 截面动量 × 期权持仓组合策略.

设计文档: docs/cta/xsec_option_design.md
Stage 0: cta_xsec_momentum  — CTA 截面动量信号
Stage 1: option_oi_signals  — 期权 OI 确认信号
Stage 2: cta_xsec_option_strategy — CTA 驱动期权持仓
Stage 3: combined_xsec_portfolio  — 组合层
"""
from .config import (
    CTA_DB,
    COST_BPS,
    IS_END,
    MIN_GROSS,
    MIN_PRODUCTS,
    MOM_WINDOWS,
    TV,
    UNIVERSE,
    load_smoothed_panel,
)
from .cta_xsec_momentum import build_cta_signal, run_cta_xsec_bt

__all__ = [
    "CTA_DB",
    "COST_BPS",
    "IS_END",
    "MIN_GROSS",
    "MIN_PRODUCTS",
    "MOM_WINDOWS",
    "TV",
    "UNIVERSE",
    "load_smoothed_panel",
    "build_cta_signal",
    "run_cta_xsec_bt",
]
