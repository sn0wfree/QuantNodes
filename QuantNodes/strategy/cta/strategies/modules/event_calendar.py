# coding=utf-8
"""事件日历组 — 极端波动/跳变/日历效应类因子.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.event import ExtremeMoveFactor, GapStreakFactor, LimitMoveFactor
from ...factors.calendar import MonthTurnFactor, SeasonalityFactor, WeekdayFactor


class EventCalendarModuleStrategy(_ModuleStrategy):
    """事件日历组: 极端波动/跳变/日历效应类因子."""

    FACTORS = [
        ("extreme_move", ExtremeMoveFactor, "panel"),
        ("gap_streak", GapStreakFactor, "panel"),
        ("limit_move", LimitMoveFactor, "panel"),
        ("seasonality", SeasonalityFactor, "panel"),
        ("weekday", WeekdayFactor, "panel"),
        ("month_turn", MonthTurnFactor, "panel"),
    ]
