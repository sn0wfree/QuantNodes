"""⚠️ DEPRECATION SHIM — backtest_pkg.factor_value_store.

M3 (PR4): module moved to QuantNodes.research.backtest.factor_value_store.
This shim re-exports everything for backward compatibility.
"""
from __future__ import annotations

import warnings as _warnings

from QuantNodes.research.backtest import factor_value_store as _real_module

_warnings.warn(
    "QuantNodes.research.backtest_pkg.factor_value_store is deprecated; use "
    "QuantNodes.research.backtest.factor_value_store instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    """Forward attribute access to the real module (PEP 562)."""
    return getattr(_real_module, name)


def __dir__():
    return dir(_real_module)
