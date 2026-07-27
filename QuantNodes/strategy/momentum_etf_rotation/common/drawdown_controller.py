# coding=utf-8
"""回撤控制器 (ACT-3): Grossman-Zhou (1993) 连续回撤控制.

参考: 10_TURTLE_TRADING_MATHEMATICS.md ACT-3

公式:
    dd = (peak - equity) / peak
    multiplier = max(0, 1 - dd / max_tolerance)

当 DD → 0 时, multiplier → 1.0 (全仓)
当 DD → max_tolerance 时, multiplier → 0.0 (清仓)

海龟版本的离散化:
    每回撤 10% 减仓 β%
    β = (1/floor) - 1, floor = 1 - max_tolerance

标定表:
    max_tolerance | 每10% DD应削减 | 海龟原值
    40%           | 14%           |
    31%           | 20%           | 20%
    25%           | 26%           |
    20%           | 33%           |
    15%           | 43%           |
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DrawdownConfig:
    """回撤控制器配置."""
    max_tolerance: float = 0.25      # 最大容忍回撤 (默认 25%)
    min_multiplier: float = 0.0      # 最小缩放系数 (0 = 清仓)
    enabled: bool = True             # 是否启用


def drawdown_multiplier(
    equity: float,
    peak: float,
    max_tolerance: float = 0.25,
) -> float:
    """计算当前回撤控制器的缩放系数.

    Args:
        equity: 当前权益
        peak: 历史最高权益
        max_tolerance: 最大容忍回撤 (0-1)

    Returns:
        multiplier ∈ [0, 1], 1.0 = 全仓, 0.0 = 清仓
    """
    if peak <= 0 or equity <= 0:
        return 1.0

    dd = (peak - equity) / peak
    dd = max(0.0, min(1.0, dd))

    if max_tolerance <= 0:
        return 0.0

    multiplier = max(0.0, 1.0 - dd / max_tolerance)
    return float(multiplier)


def drawdown_multiplier_curve(
    equity_range: np.ndarray | list,
    peak: float,
    max_tolerance: float = 0.25,
) -> np.ndarray:
    """计算回撤控制器的缩放曲线 (用于可视化).

    Args:
        equity_range: 权益范围 (如 np.linspace(0.5, 1.0, 100))
        peak: 历史最高权益
        max_tolerance: 最大容忍回撤

    Returns:
        np.ndarray: 缩放系数曲线
    """
    return np.array([
        drawdown_multiplier(eq, peak, max_tolerance)
        for eq in equity_range
    ])


@dataclass
class DrawdownState:
    """回撤控制器状态 (跟踪 peak 和当前 DD)."""
    peak: float = 1.0
    current_equity: float = 1.0
    current_dd: float = 0.0
    current_multiplier: float = 1.0

    def update(self, equity: float, max_tolerance: float = 0.25) -> float:
        """更新状态并返回新的 multiplier.

        Args:
            equity: 当前权益
            max_tolerance: 最大容忍回撤

        Returns:
            multiplier ∈ [0, 1]
        """
        if equity > self.peak:
            self.peak = equity

        self.current_equity = equity
        self.current_dd = (self.peak - equity) / self.peak if self.peak > 0 else 0.0
        self.current_multiplier = drawdown_multiplier(
            equity, self.peak, max_tolerance
        )
        return self.current_multiplier


def apply_drawdown_control(
    weights: dict[str, float],
    equity: float,
    peak: float,
    config: DrawdownConfig | None = None,
) -> dict[str, float]:
    """对权重应用回撤控制.

    Args:
        weights: 当前权重 {code: weight}
        equity: 当前权益
        peak: 历史最高权益
        config: 回撤控制器配置

    Returns:
        调整后的权重
    """
    if config is None:
        config = DrawdownConfig()

    if not config.enabled:
        return weights

    multiplier = drawdown_multiplier(equity, peak, config.max_tolerance)
    multiplier = max(config.min_multiplier, multiplier)

    return {k: v * multiplier for k, v in weights.items()}


# ============================================================
# 海龟版本 (离散化): 每 10% DD 减仓 β%
# ============================================================

def turtle_drawdown_beta(max_tolerance: float = 0.25) -> float:
    """计算海龟版本的每 10% DD 减仓比例.

    公式: β = (1/floor) - 1, floor = 1 - max_tolerance

    Args:
        max_tolerance: 最大容忍回撤

    Returns:
        β: 每 10% DD 应减仓的比例
    """
    floor = 1.0 - max_tolerance
    if floor <= 0:
        return 1.0
    return (1.0 / floor) - 1.0


def turtle_drawdown_multiplier(
    equity: float,
    peak: float,
    max_tolerance: float = 0.25,
) -> float:
    """海龟版本的离散化回撤控制器.

    每 10% DD 减仓 β%, 阶梯式.

    Args:
        equity: 当前权益
        peak: 历史最高权益
        max_tolerance: 最大容忍回撤

    Returns:
        multiplier ∈ [0, 1]
    """
    if peak <= 0 or equity <= 0:
        return 1.0

    dd = (peak - equity) / peak
    dd = max(0.0, min(1.0, dd))

    beta = turtle_drawdown_beta(max_tolerance)
    n_10pct = int(dd / 0.1)  # 完整的 10% 回撤次数

    multiplier = 1.0
    for _ in range(n_10pct):
        multiplier *= (1.0 - beta)

    return max(0.0, float(multiplier))


# ============================================================
# 标定表
# ============================================================

def drawdown_calibration_table() -> pd.DataFrame:
    """生成回撤控制标定表 (海龟 vs Grossman-Zhou).

    Returns:
        pd.DataFrame: 标定表
    """
    tolerances = [0.40, 0.31, 0.25, 0.20, 0.15]
    rows = []
    for t in tolerances:
        beta = turtle_drawdown_beta(t)
        rows.append({
            "max_tolerance": t,
            "max_dd_pct": f"{(1-t)*100:.0f}%",
            "turtle_beta_per_10pct": f"{beta*100:.0f}%",
            "floor": f"{1-t:.2f}",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== 回撤控制标定表 ===")
    print(drawdown_calibration_table())
    print()

    print("=== Grossman-Zhou 缩放曲线 ===")
    equity_range = np.linspace(0.6, 1.0, 9)
    curve = drawdown_multiplier_curve(equity_range, peak=1.0, max_tolerance=0.25)
    for eq, mult in zip(equity_range, curve):
        print(f"  DD={100*(1-eq):.0f}% → multiplier={mult:.2f}")
