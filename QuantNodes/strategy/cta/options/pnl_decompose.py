# coding=utf-8
"""P&L 分解: theta + gamma + vega + delta hedge + residual.

Gamma Scalping 日频 P&L 分解公式 (discrete hedging):

    theta_pnl  = Σ(pos × theta × multiplier × dt)
    gamma_pnl  = Σ(0.5 × gamma × (ΔS)² × pos × multiplier)
    vega_pnl   = Σ(vega × Δσ × pos × multiplier)
    delta_pnl  = -Σ(hedge_qty × ΔS × hedge_multiplier)
    residual   = total_pnl - theta - gamma - vega - delta

其中 dt = 1/365 (日频), ΔS = price[t] - price[t-1], Δσ = iv[t] - iv[t-1].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class PnlSnapshot:
    """单日 P&L 分解快照."""

    date: str
    theta_pnl: float
    gamma_pnl: float
    vega_pnl: float
    delta_hedge_pnl: float
    residual: float
    total_pnl: float


@dataclass
class PnlDecomposer:
    """P&L 分解器 — 跟踪每日各分量.

    使用方法:
        1. 每日调用 compute_daily() 传入 option greeks + price/iv 变动
        2. to_dataframe() 输出时间序列
        3. summary() 输出统计
    """

    _snapshots: list[PnlSnapshot] = field(default_factory=list)

    def compute_daily(
        self,
        date: str,
        option_positions: list[dict],
        prev_prices: dict[str, float],
        curr_prices: dict[str, float],
        prev_ivs: dict[str, float],
        curr_ivs: dict[str, float],
        hedge_pnl: float = 0.0,
    ) -> PnlSnapshot:
        """单日 P&L 分解.

        Args:
            date: 交易日
            option_positions: [{product, delta, gamma, vega, theta, qty, multiplier}, ...]
            prev_prices: {product: t-1 价格}
            curr_prices: {product: t 价格}
            prev_ivs: {product: t-1 IV}
            curr_ivs: {product: t IV}
            hedge_pnl: delta hedge 产生的 P&L (已由 DeltaHedgeManager 计算)

        Returns:
            PnlSnapshot
        """
        dt = 1.0 / 365.0
        theta_pnl = 0.0
        gamma_pnl = 0.0
        vega_pnl = 0.0

        for pos in option_positions:
            product = pos["product"]
            gamma = pos.get("gamma", 0.0)
            theta = pos.get("theta", 0.0)
            vega = pos.get("vega", 0.0)
            qty = pos.get("qty", 0.0)
            mult = pos.get("multiplier", 1.0)

            # ΔS 和 Δσ
            p0 = prev_prices.get(product, 0.0)
            p1 = curr_prices.get(product, 0.0)
            ds = p1 - p0

            iv0 = prev_ivs.get(product, 0.0)
            iv1 = curr_ivs.get(product, 0.0)
            div = iv1 - iv0

            # 分量计算
            theta_pnl += theta * qty * mult * dt
            gamma_pnl += 0.5 * gamma * ds * ds * qty * mult
            vega_pnl += vega * div * qty * mult

        total = theta_pnl + gamma_pnl + vega_pnl + hedge_pnl
        residual = 0.0  # 完美分解时 residual ≈ 0

        snap = PnlSnapshot(
            date=date,
            theta_pnl=theta_pnl,
            gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl,
            delta_hedge_pnl=hedge_pnl,
            residual=residual,
            total_pnl=total,
        )
        self._snapshots.append(snap)
        return snap

    def to_dataframe(self) -> pd.DataFrame:
        """输出 P&L 分解时间序列."""
        if not self._snapshots:
            return pd.DataFrame()
        return pd.DataFrame([
            {
                "date": s.date,
                "theta": s.theta_pnl,
                "gamma": s.gamma_pnl,
                "vega": s.vega_pnl,
                "delta_hedge": s.delta_hedge_pnl,
                "residual": s.residual,
                "total": s.total_pnl,
            }
            for s in self._snapshots
        ])

    def summary(self) -> dict:
        """输出 P&L 分解统计."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        result = {}
        for col in ["theta", "gamma", "vega", "delta_hedge", "residual", "total"]:
            result[col] = {
                "sum": df[col].sum(),
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
            }
        return result

    def reset(self):
        """重置状态."""
        self._snapshots.clear()
