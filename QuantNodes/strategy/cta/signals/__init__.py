# coding=utf-8
"""CTA 交易信号."""
from __future__ import annotations

from .trend_signal import (
    donchian_signal,
    ma_signal,
    momentum_signal,
    position_from_signal,
)

__all__ = [
    "donchian_signal",
    "ma_signal",
    "momentum_signal",
    "position_from_signal",
]
