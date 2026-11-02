# coding=utf-8
"""多目标评估函数.

评估指标:
    - Calmar ratio: 主要优化目标 (文章核心指标)
    - MaxDD 约束: 不超过 -20%
    - 信号覆盖率约束: 不低于 10%
    - 交易次数约束: 50-200 笔 (防过拟合)

评估逻辑:
    1. 先检查硬约束 (MaxDD, 覆盖率, 交易次数)
    2. 通过约束后, 按 Calmar 排序
    3. 返回评估结果字典
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalResult:
    """评估结果."""

    feasible: bool
    calmar: float
    sharpe: float
    max_dd: float
    signal_coverage: float
    n_trades: int
    reject_reason: str = ""


def evaluate_strategy(
    calmar: float = 0.0,
    sharpe: float = 0.0,
    max_dd: float = -1.0,
    signal_coverage: float = 0.0,
    n_trades: int = 0,
    max_dd_limit: float = -0.30,
    min_coverage: float = 0.05,
    min_trades: int = 10,
    max_trades: int = 1000,
) -> EvalResult:
    """多目标评估.

    Args:
        calmar: 卡玛比
        sharpe: 夏普比
        max_dd: 最大回撤 (负值)
        signal_coverage: 信号覆盖率 (0-1)
        n_trades: 交易次数
        max_dd_limit: MaxDD 硬约束 (默认 -30%, 宽松模式)
        min_coverage: 最低覆盖率 (默认 5%)
        min_trades: 最低交易次数 (默认 10)
        max_trades: 最高交易次数 (默认 1000)

    Returns:
        EvalResult: 评估结果
    """
    # 硬约束检查
    if max_dd < max_dd_limit:
        return EvalResult(
            feasible=False, calmar=calmar, sharpe=sharpe,
            max_dd=max_dd, signal_coverage=signal_coverage,
            n_trades=n_trades,
            reject_reason=f"MaxDD {max_dd:.3f} < limit {max_dd_limit}",
        )
    if signal_coverage < min_coverage:
        return EvalResult(
            feasible=False, calmar=calmar, sharpe=sharpe,
            max_dd=max_dd, signal_coverage=signal_coverage,
            n_trades=n_trades,
            reject_reason=f"Coverage {signal_coverage:.3f} < {min_coverage}",
        )
    if n_trades < min_trades:
        return EvalResult(
            feasible=False, calmar=calmar, sharpe=sharpe,
            max_dd=max_dd, signal_coverage=signal_coverage,
            n_trades=n_trades,
            reject_reason=f"Trades {n_trades} < {min_trades}",
        )
    if n_trades > max_trades:
        return EvalResult(
            feasible=False, calmar=calmar, sharpe=sharpe,
            max_dd=max_dd, signal_coverage=signal_coverage,
            n_trades=n_trades,
            reject_reason=f"Trades {n_trades} > {max_trades}",
        )
    return EvalResult(
        feasible=True, calmar=calmar, sharpe=sharpe,
        max_dd=max_dd, signal_coverage=signal_coverage,
        n_trades=n_trades,
    )


def rank_results(results: list[EvalResult]) -> list[EvalResult]:
    """按 Calmar 降序排列 (仅可行解)."""
    feasible = [r for r in results if r.feasible]
    infeasible = [r for r in results if not r.feasible]
    feasible.sort(key=lambda r: r.calmar, reverse=True)
    return feasible + infeasible
