# coding=utf-8
"""
alpha_backtest.py - Alpha 公式 Trading 回测工具

Alpha-GPT 工作流的【Evaluator】可选工具。
对 top-K 候选公式做完整 Trading 回测，返回年化收益 / Sharpe / 最大回撤。

复用：
- 现有 BacktestTool 思路（CodeSandbox → Strategy → Broker）
- 但更轻量：直接用 top-K 等权组合，不需要完整 Pipeline

Usage::

    tool = AlphaBacktestTool()
    result = await tool.execute(
        formulas=["rank(-ts_mean(returns, 20))"],
        data=df,
        top_k=10,
        initial_cash=1_000_000,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import Tool

logger = logging.getLogger(__name__)


class AlphaBacktestTool(Tool):
    """Alpha 公式 Trading 回测工具（top-K 等权组合）"""

    def __init__(self) -> None:
        pass

    @property
    def name(self) -> str:
        return "alpha_backtest"

    @property
    def description(self) -> str:
        return (
            "对 alpha 公式做 Trading 回测。"
            "对每公式计算 top-K 等权组合的：年化收益 / Sharpe / 最大回撤 / 胜率。"
            "输入 formulas 列表 + 数据 + top_k。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "formulas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "待回测的公式列表（建议 ≤ 20）",
                },
                "data_path": {
                    "type": "string",
                    "description": "数据路径（parquet/csv）",
                },
                "top_k": {
                    "type": "integer",
                    "default": 10,
                    "description": "每期持有 top-K 股票（默认 10）",
                },
                "bottom_k": {
                    "type": "integer",
                    "default": 0,
                    "description": "做空 bottom-K（0=不做空，默认 0）",
                },
                "rebalance_freq": {
                    "type": "integer",
                    "default": 5,
                    "description": "调仓频率（交易日，默认 5 = 周度）",
                },
                "initial_cash": {
                    "type": "number",
                    "default": 1_000_000.0,
                    "description": "初始资金（默认 100 万）",
                },
                "commission": {
                    "type": "number",
                    "default": 0.001,
                    "description": "手续费率（默认 0.001）",
                },
                "date_column": {
                    "type": "string",
                    "default": "date",
                    "description": "日期列名",
                },
                "code_column": {
                    "type": "string",
                    "default": "code",
                    "description": "股票代码列名",
                },
            },
            "required": ["formulas"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(
        self,
        formulas: Sequence[str],
        data_path: Optional[str] = None,
        top_k: int = 10,
        bottom_k: int = 0,
        rebalance_freq: int = 5,
        initial_cash: float = 1_000_000.0,
        commission: float = 0.001,
        date_column: str = "date",
        code_column: str = "code",
        data: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            import polars as pl
            from QuantNodes.research.quant_alpha.adapters import (
                PolarsAlphaCalculator,
            )
            from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool

            df = AlphaEvaluateTool._resolve_data(
                data, data_path, date_column, code_column
            )
            if df is None:
                return {"backtests": [], "summary": {"error": "no data"}}

            forward_returns_dict = AlphaEvaluateTool._build_forward_returns(
                df, [1], date_column, code_column,
            )

            calc = PolarsAlphaCalculator(
                data=df,
                forward_returns=forward_returns_dict,
                date_column=date_column,
                code_column=code_column,
            )

            backtests: List[Dict[str, Any]] = []
            for formula in formulas:
                backtests.append(
                    self._backtest_one(
                        calc,
                        formula,
                        top_k=top_k,
                        bottom_k=bottom_k,
                        rebalance_freq=rebalance_freq,
                        initial_cash=initial_cash,
                        commission=commission,
                        date_column=date_column,
                        code_column=code_column,
                    )
                )

            successful = [b for b in backtests if b["status"] == "success"]
            sharpes = [
                b["backtest"]["sharpe"] for b in successful if b["backtest"]["sharpe"] is not None
            ]
            return {
                "backtests": backtests,
                "summary": {
                    "total": len(backtests),
                    "success": len(successful),
                    "failed": len(backtests) - len(successful),
                    "avg_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
                    "best_sharpe": float(np.max(sharpes)) if sharpes else 0.0,
                },
            }
        except Exception as exc:
            logger.exception("alpha_backtest failed")
            return {"backtests": [], "summary": {"error": str(exc)}}

    def _backtest_one(
        self,
        calc: Any,
        formula: str,
        top_k: int,
        bottom_k: int,
        rebalance_freq: int,
        initial_cash: float,
        commission: float,
        date_column: str,
        code_column: str,
    ) -> Dict[str, Any]:
        try:
            import polars as pl

            from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool

            expr = AlphaEvaluateTool._parse_simple_formula(formula)
            factor_series = calc._evaluate_factor(expr)
            if factor_series is None:
                return {
                    "formula": formula,
                    "status": "failed",
                    "backtest": {},
                    "error_msg": "factor evaluation returned None",
                }

            dates = sorted(calc.data[date_column].unique().to_list())
            if len(dates) < 2:
                return {
                    "formula": formula,
                    "status": "failed",
                    "backtest": {},
                    "error_msg": "not enough dates",
                }

            work = calc.data.select([date_column, code_column, "close"]).with_columns(
                factor_series.alias("_factor")
            )
            fwd = calc.forward_returns[1]
            work = work.with_columns(fwd.alias("_fwd"))

            holdings_returns: List[float] = []
            cash = initial_cash
            equity_curve: List[float] = [initial_cash]
            entry_days: List[int] = list(range(0, len(dates) - 1, rebalance_freq))

            for entry_idx in entry_days:
                if entry_idx + 1 >= len(dates):
                    break
                entry_date = dates[entry_idx]
                exit_date = dates[entry_idx + 1]

                day_slice = work.filter(pl.col(date_column) == entry_date)
                ranked = day_slice.filter(pl.col("_factor").is_not_null()).sort(
                    "_factor", descending=True
                )
                longs = ranked.head(top_k)[code_column].to_list()
                shorts = (
                    ranked.tail(bottom_k)[code_column].to_list() if bottom_k > 0 else []
                )

                if not longs:
                    continue

                next_close = work.filter(
                    (pl.col(date_column) == exit_date)
                    & pl.col(code_column).is_in(longs + shorts)
                )

                if bottom_k > 0 and shorts:
                    long_ret = (
                        next_close.filter(pl.col(code_column).is_in(longs))["_fwd"]
                        .drop_nulls()
                        .mean()
                    )
                    short_ret = (
                        -next_close.filter(pl.col(code_column).is_in(shorts))["_fwd"]
                        .drop_nulls()
                        .mean()
                    )
                    period_ret = float((long_ret or 0) + (short_ret or 0)) / 2.0
                else:
                    long_ret = next_close.filter(pl.col(code_column).is_in(longs))[
                        "_fwd"
                    ].drop_nulls().mean()
                    period_ret = float(long_ret or 0)

                turnover = (len(longs) + len(shorts)) / max(1, len(longs) + len(shorts))
                cost = turnover * commission
                net_ret = period_ret - cost
                cash *= 1.0 + net_ret
                holdings_returns.append(net_ret)
                equity_curve.append(cash)

            if not holdings_returns:
                return {
                    "formula": formula,
                    "status": "failed",
                    "backtest": {},
                    "error_msg": "no valid periods",
                }

            rets = np.array(holdings_returns)
            ann_factor = 252.0 / rebalance_freq
            annual_return = float(np.mean(rets) * ann_factor)
            sharpe = (
                float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(ann_factor))
                if np.std(rets) > 1e-12
                else 0.0
            )
            equity = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity)
            drawdown = (equity - running_max) / running_max
            max_dd = float(np.min(drawdown)) if drawdown.size else 0.0
            wins = int(np.sum(rets > 0))
            win_rate = wins / len(rets) if len(rets) else 0.0

            return {
                "formula": formula,
                "status": "success",
                "backtest": {
                    "annual_return": annual_return,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "win_rate": float(win_rate),
                    "n_periods": len(rets),
                    "equity_curve": equity_curve[:: max(1, len(equity_curve) // 50)],
                },
                "error_msg": None,
            }
        except Exception as exc:
            return {
                "formula": formula,
                "status": "failed",
                "backtest": {},
                "error_msg": str(exc),
            }


__all__ = ["AlphaBacktestTool"]
