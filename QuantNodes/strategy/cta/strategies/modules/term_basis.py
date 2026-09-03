# coding=utf-8
"""期限结构/基差组 — 近远月价差 + 现货基差类因子.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.term_structure import (
    BasisConvergenceFactor, CalendarSpreadMeanReversion, TermSlopeFactor,
)
from ...factors.basis_momentum import BasisLevelFactor


class TermBasisModuleStrategy(_ModuleStrategy):
    """期限结构/基差组: 近远月价差 + 现货基差类因子.

    数据: load_term_structure (近远月 OI top-4 合约) + spot_basis_daily.
    因子选择 (docs/94 §7.5/§8.9): 9 因子归因仅 term_slope/basis_level 为正
    (basis/roll_yield/term_curvature/oi_concentration/front_month_oi/
    near_basis/basis_momentum 单边信号在国内商品系统性失效, 剔除).
    basis_level 单因子 72 品种归因转负 (-0.44) 但组合层面为正贡献
    (sign 非线性), 保留.
    IF/T 等无现货基差品种无信号 (NaN → 0).
    """

    FACTORS = [
        ("term_slope", TermSlopeFactor, "spread"),
        ("basis_level", BasisLevelFactor, "basis"),
        ("basis_convergence", BasisConvergenceFactor, "basis"),
        ("calendar_spread_mr", CalendarSpreadMeanReversion, "spread"),
    ]
