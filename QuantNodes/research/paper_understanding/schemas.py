"""Schemas for reproduction module.

Merged schema: our 12 fields + QuantNodes summary/config/security_status/nodes.

WikiFactor V2 (M3 前置, PR6.5):
  原 `WikiFactor` (6 字段) 和 `WikiStrategy` (8 字段) 已被删除, 合并到
  `QuantNodes.research.wiki.WikiFactor` (23 字段)。两个旧 schemas 类
  自 v4.0.0 reproduction merge 以来没有任何生产代码 caller, 仅 4 个
  测试文件使用 — 现在统一到 wiki.py 的 23 字段 WikiFactor。

  需要 `factor_params` / `status` 字段的代码请 import:
    from QuantNodes.research.wiki import WikiFactor

当前保留的 schema 类:
  - BacktestResult       (生产用, run_backtest/factor_backtest)
  - FactorBacktestResult (生产用, factor_backtest/run_factor_backtest_universe)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BacktestResult:
    """Result of a backtest run.

    Unified schema combining:
      - llmwikify fields (status/error/statistics/trades/signal_type/params + final_cash/total_return/sharpe/max_dd/win_rate)
      - QuantNodes fields (summary/config/security_status/nodes)
      - Equity curve and monthly returns (populated by backtest engine)
    """

    # llmwikify fields
    status: str = "success"  # "success" | "error"
    error: str | None = None
    statistics: dict[str, float] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    final_cash: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    signal_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    # QuantNodes fields (optional, only populated when going through QuantNodes path)
    summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    security_status: str = "unknown"
    nodes: dict[str, Any] = field(default_factory=dict)

    # Equity curve and monthly returns (populated by backtest engine)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)  # [{date, value}, ...]
    monthly_returns: dict[str, float] = field(default_factory=dict)   # {"2024-01": 2.3, ...}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error": self.error,
            "statistics": self.statistics,
            "trades": self.trades,
            "final_cash": self.final_cash,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "signal_type": self.signal_type,
            "params": self.params,
            "summary": self.summary,
            "config": self.config,
            "security_status": self.security_status,
            "nodes": self.nodes,
            "equity_curve": self.equity_curve,
            "monthly_returns": self.monthly_returns,
        }


# ─── Three-page architecture (Factor / Strategy / FactorBacktest) ───
#
# Regression note (2026-06-16):
#   The three dataclasses below were accidentally deleted in commit
#   4fd128b ("feat(agent): Agent 服务重构 + Chat 桥接 + 技能系统升级"),
#   which edited schemas.py to remove equity_curve / monthly_returns
#   fields from BacktestResult but unintentionally removed the entire
#   @dataclass blocks for WikiFactor / WikiStrategy / FactorBacktestResult
#   that followed. Restored verbatim from 4fd128b~1 (commit e86bf67).
#   See:
#     - reproduction/__init__.py:4 (public re-exports, ImportError blocked server)
#     - reproduction/factor_backtest.py:25 (12+ instantiation sites)
#     - reproduction/l5_validation.py:266,532
#     - tests/reproduction/test_factor_backtest_cross_section.py:205,375
#     - tests/reproduction/test_quant.py:713


@dataclass
class FactorBacktestResult:
    """Result of a single-factor backtest.

    Supports both single-stock (legacy) and cross-section (universe) modes.
    New fields default to zero / empty so existing callers remain compatible.
    """

    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    t_stat: float = 0.0
    win_rate: float = 0.0         # IC > 0 ratio
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    quantile_returns: dict[str, float] = field(default_factory=dict)  # {group: annual_return}
    ic_series: list[dict[str, Any]] = field(default_factory=list)     # [{date, ic}]
    quantile_curves: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # {group: [{date, value}]}

    # Cross-section (universe) mode fields — populated by
    # run_factor_backtest_universe(). Zero/empty for single-stock mode.
    rank_ic_mean: float = 0.0     # Spearman Rank IC mean
    rank_ic_std: float = 0.0
    rank_icir: float = 0.0        # rank_ic_mean / rank_ic_std
    rank_ic_pos_ratio: float = 0.0  # fraction of rank_ic > 0
    longshort_ann_return: float = 0.0
    longshort_sharpe: float = 0.0
    longshort_mdd: float = 0.0
    longshort_curve: list[dict[str, Any]] = field(default_factory=list)  # [{date, value}]
    universe: str = ""            # e.g. "HS300"
    adj_mode: str = "D"           # "D" / "M-end"
    n_stocks_per_date: list[dict[str, Any]] = field(default_factory=list)  # [{date, n}, ...]
    # Per-group metrics from cross-section quantile analysis.
    # {G1: {sharpe, max_drawdown, win_rate, turnover, n_stocks}, ...}
    group_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    total_rebalances: int = 0     # Total number of rebalance dates
    valid_rebalances: int = 0     # Number of successful IC calculations

    def to_dict(self) -> dict[str, Any]:
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "t_stat": self.t_stat,
            "win_rate": self.win_rate,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "quantile_returns": self.quantile_returns,
            "ic_series": self.ic_series,
            "quantile_curves": self.quantile_curves,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_std": self.rank_ic_std,
            "rank_icir": self.rank_icir,
            "rank_ic_pos_ratio": self.rank_ic_pos_ratio,
            "longshort_ann_return": self.longshort_ann_return,
            "longshort_sharpe": self.longshort_sharpe,
            "longshort_mdd": self.longshort_mdd,
            "longshort_curve": self.longshort_curve,
            "universe": self.universe,
            "adj_mode": self.adj_mode,
            "n_stocks_per_date": self.n_stocks_per_date,
            "group_metrics": self.group_metrics,
            "total_rebalances": self.total_rebalances,
            "valid_rebalances": self.valid_rebalances,
        }
