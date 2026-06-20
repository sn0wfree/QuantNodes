"""Phase J6 (2026-06-20): Chinese labels for long-short and group analysis output.

Consolidates the hard-coded 中文 column labels used in long_short_node.py
(group_result dict + DataFrame columns) and group_analyzer_node.py's
yearly-eva dict keys.

Reuse from tests via:
    from QuantNodes.research.factor_test.utils.labels import (
        LONG, SHORT, LONG_EXCESS, SHORT_EXCESS, LONGSHORT,
        L_S_COLS, NET_COLS,
    )
"""
from __future__ import annotations


# Column labels for group results
LONG: str = "多头"
SHORT: str = "空头"
LONG_EXCESS: str = "多头超额"
SHORT_EXCESS: str = "空头超额"
LONGSHORT: str = "多空"

# Standard 3-column order for L/S evaluation tables
L_S_COLS: list[str] = [LONG_EXCESS, SHORT_EXCESS, LONGSHORT]

# Standard 5-column order for the net value table
NET_COLS: list[str] = [LONG, SHORT, LONG_EXCESS, SHORT_EXCESS, LONGSHORT]

__all__ = [
    "LONG",
    "SHORT",
    "LONG_EXCESS",
    "SHORT_EXCESS",
    "LONGSHORT",
    "L_S_COLS",
    "NET_COLS",
]
