"""Backtest engine abstraction — runs generated code against market data.

Public API:

  New (v3.0.0):
    - FactorResult: per-signal result (code + metrics + status)
    - BacktestEngine: Protocol that runs code and returns metrics dict
    - QuantNodesBacktest: QuantNodes PipelineRunner adapter

  Legacy (from backtest_pkg/, M3 main merge):
    - run_backtest: top-level reproduction entry point
    - run_factor_backtest, run_factor_backtest_universe, generate_adj_dates:
      factor-level backtest compute
    - evaluation, compute_metrics_from_trades, compute_extended_metrics,
      compute_monthly_returns, cal_net_simple: trade-level metrics
    - store_factor_values, query_factor_values, list_stored_factors,
      compute_and_store_factor: DuckDB-backed factor value store
    - analyze_ic/groups/returns/turnover/stability/oos/cost, compute_score,
      run_l5_validation: L5 validation pipeline
    - run_l5_pipeline: L5 orchestrator (LLM-driven hypothesis loop)
    - MACrossStrategyNode, RSIStrategyNode, MomentumStrategyNode,
      VolatilityStrategyNode, FactorRankStrategyNode,
      SignalCompositeStrategyNode, get_strategy_node: prewritten strategy nodes
    - run_paper_backtest, save_report, BacktestOutcome, PaperBacktestReport:
      reproduction flow artifacts

In PR6, `PaperPipeline._process_one_signal` will:
  1. Call LLM codegen (PR2 territory) → code + factor_series
  2. Write H5 + call `backtest_engine.run(code, h5_path, signal)` → metrics
  3. Wrap everything in `FactorResult` and dispatch to sinks (PR4)

Usage:
    from QuantNodes.research.backtest import (
        FactorResult, BacktestEngine, QuantNodesBacktest,
        run_backtest, run_factor_backtest, evaluation,
    )

M3 (PR4): legacy modules from `backtest_pkg/` were physically migrated
to this package. `backtest_pkg.*` paths are kept as deprecation shims
that re-export from here. New code should import from
`QuantNodes.research.backtest.*` directly.
"""
from __future__ import annotations

# New v3.0.0 public API
from .base import BacktestEngine, FactorResult
from .quantnodes import QuantNodesBacktest

# Legacy re-exports (from backtest_pkg/, M3 main merge)
from .run_backtest import run_backtest
from .factor_backtest import (
    generate_adj_dates,
    run_factor_backtest,
    run_factor_backtest_universe,
)
from .metrics import (
    cal_net_simple,
    compute_extended_metrics,
    compute_metrics_from_trades,
    compute_monthly_returns,
    evaluation,
)
from .factor_value_store import (
    compute_and_store_factor,
    list_stored_factors,
    query_factor_values,
    store_factor_values,
)
from .l5_validation import (
    analyze_cost,
    analyze_groups,
    analyze_ic,
    analyze_oos,
    analyze_returns,
    analyze_stability,
    analyze_turnover,
    compute_score,
    run_l5_validation,
)
from .l5_orchestrator import run_l5_pipeline
from .strategies import (
    SIGNAL_NODE_REGISTRY,
    FactorRankStrategyNode,
    MACrossStrategyNode,
    MomentumStrategyNode,
    RSIStrategyNode,
    SignalCompositeStrategyNode,
    VolatilityStrategyNode,
    get_strategy_node,
)
from .quantnodes_repro import (
    BacktestOutcome,
    PaperBacktestReport,
    run_factor_backtest as run_factor_backtest_repro,  # alias to avoid clash
    run_paper_backtest,
    save_report,
)

__all__ = [
    # New v3.0.0
    "FactorResult",
    "BacktestEngine",
    "QuantNodesBacktest",
    # Legacy — run_backtest
    "run_backtest",
    # Legacy — factor_backtest
    "generate_adj_dates",
    "run_factor_backtest",
    "run_factor_backtest_universe",
    # Legacy — metrics
    "cal_net_simple",
    "compute_extended_metrics",
    "compute_metrics_from_trades",
    "compute_monthly_returns",
    "evaluation",
    # Legacy — factor_value_store
    "compute_and_store_factor",
    "list_stored_factors",
    "query_factor_values",
    "store_factor_values",
    # Legacy — l5_validation
    "analyze_cost",
    "analyze_groups",
    "analyze_ic",
    "analyze_oos",
    "analyze_returns",
    "analyze_stability",
    "analyze_turnover",
    "compute_score",
    "run_l5_validation",
    # Legacy — l5_orchestrator
    "run_l5_pipeline",
    # Legacy — strategies
    "SIGNAL_NODE_REGISTRY",
    "FactorRankStrategyNode",
    "MACrossStrategyNode",
    "MomentumStrategyNode",
    "RSIStrategyNode",
    "SignalCompositeStrategyNode",
    "VolatilityStrategyNode",
    "get_strategy_node",
    # Legacy — quantnodes_repro
    "BacktestOutcome",
    "PaperBacktestReport",
    "run_paper_backtest",
    "save_report",
    "run_factor_backtest_repro",
]