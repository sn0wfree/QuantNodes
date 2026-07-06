"""⚠️ DEPRECATION SHIM — backtest_pkg.l5_orchestrator.

M3 (PR4): module moved to QuantNodes.research.backtest.l5_orchestrator.
This shim re-exports everything for backward compatibility.
"""
from __future__ import annotations

import warnings as _warnings

from QuantNodes.research.backtest import l5_orchestrator as _real_module

_warnings.warn(
    "QuantNodes.research.backtest_pkg.l5_orchestrator is deprecated; use "
    "QuantNodes.research.backtest.l5_orchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name: str):
    """Forward attribute access to the real module (PEP 562)."""
    return getattr(_real_module, name)


def __dir__():
    return dir(_real_module)
