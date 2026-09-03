# coding=utf-8
"""宏观联动组 — VIX/DXY 与品种联动类因子.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.macro_link import DXYLinkFactor, MacroStateFactor, VIXLinkFactor


class MacroModuleStrategy(_ModuleStrategy):
    """宏观联动组: VIX/DXY 与品种联动类因子."""

    FACTORS = [
        ("vix_link", VIXLinkFactor, "macro_vix"),
        ("dxy_link", DXYLinkFactor, "macro_dxy"),
        ("macro_state", MacroStateFactor, "macro_vix"),
    ]
