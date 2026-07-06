"""⚠️ DEPRECATION SHIM — QuantNodes.research.backtest_pkg.

M3 (PR4): The 8 modules previously in this package have been physically
migrated to ``QuantNodes.research.backtest``. This ``backtest_pkg`` namespace
is kept as a deprecation shim that re-exports the submodules for backward
compatibility.

.. deprecated::
    Import from ``QuantNodes.research.backtest`` instead. This shim will
    be removed in a future release. See ``docs/refactor/REFACTOR_PLAN.md``
    (M3 main) for migration details.

Migration:
    # Old (still works but emits DeprecationWarning):
    from QuantNodes.research.backtest_pkg.l5_validation import run_l5_validation

    # New (preferred):
    from QuantNodes.research.backtest import run_l5_validation

Submodule access:
    ``backtest_pkg.<submodule>.attr`` continues to work because each
    submodule has its own shim file (``backtest_pkg/<submodule>.py``)
    that re-exports everything from ``QuantNodes.research.backtest.<submodule>``
    via PEP 562 ``__getattr__`` forwarding.
"""
from __future__ import annotations

import importlib
import warnings

# Resolve each submodule DIRECTLY via importlib so we bypass the parent
# package's ``__init__`` re-exports (e.g. ``backtest.run_backtest`` is
# the function, not the module). This preserves ``backtest_pkg.run_backtest``
# as a MODULE reference for callers that do
# ``from backtest_pkg import run_backtest as b; b.run_backtest(...)``.
factor_backtest = importlib.import_module("QuantNodes.research.backtest.factor_backtest")
factor_value_store = importlib.import_module("QuantNodes.research.backtest.factor_value_store")
l5_orchestrator = importlib.import_module("QuantNodes.research.backtest.l5_orchestrator")
l5_validation = importlib.import_module("QuantNodes.research.backtest.l5_validation")
metrics = importlib.import_module("QuantNodes.research.backtest.metrics")
quantnodes_repro = importlib.import_module("QuantNodes.research.backtest.quantnodes_repro")
run_backtest = importlib.import_module("QuantNodes.research.backtest.run_backtest")
strategies = importlib.import_module("QuantNodes.research.backtest.strategies")

warnings.warn(
    "QuantNodes.research.backtest_pkg is deprecated; use "
    "QuantNodes.research.backtest instead. This shim will be removed in "
    "a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "factor_backtest",
    "factor_value_store",
    "l5_orchestrator",
    "l5_validation",
    "metrics",
    "quantnodes_repro",
    "run_backtest",
    "strategies",
]