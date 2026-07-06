"""⚠️ DEPRECATION SHIM — backtest_pkg.strategies.

M3 (PR4): module moved to QuantNodes.research.backtest.strategies.
This shim re-exports everything for backward compatibility.
"""
from __future__ import annotations

import warnings as _warnings

from QuantNodes.research.backtest import strategies as _real_module

_warnings.warn(
    "QuantNodes.research.backtest_pkg.strategies is deprecated; use "
    "QuantNodes.research.backtest.strategies instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    """Forward attribute access to the real module (PEP 562)."""
    return getattr(_real_module, name)


def __dir__():
    return dir(_real_module)
