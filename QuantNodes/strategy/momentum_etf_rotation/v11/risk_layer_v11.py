# coding=utf-8
"""v11 风控层 — 海龟数学升级版 (ACT-2 + ACT-3).

功能:
    1. ACT-2: Kelly 审计 (自动输出 sizing 位置)
    2. ACT-3: 回撤控制器 (Grossman-Zhou 1993)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..common.drawdown_controller import (
    DrawdownConfig,
    DrawdownState,
    drawdown_multiplier,
)
from ..common.extended_metrics import kelly_audit


@dataclass
class RiskLayerV11:
    """v11 风控层封装."""

    dd_config: DrawdownConfig
    kelly_audit_enabled: bool = True

    def __post_init__(self):
        self.dd_state = DrawdownState()
        self.kelly_results = []

    def apply_dd_control(
        self,
        weights: pd.DataFrame,
        nav_series: pd.Series,
    ) -> pd.DataFrame:
        """ACT-3: 应用回撤控制器.

        参数:
            weights: (T, N) 权重时序
            nav_series: (T,) NAV 时序

        返回:
            adjusted_weights: (T, N) 调整后权重
        """
        if not self.dd_config.enabled:
            return weights

        adjusted_weights = weights.copy()
        dd_multipliers = []

        for date in weights.index:
            if date in nav_series.index:
                nav_val = nav_series[date]
                peak = nav_series.loc[:date].max()
                mult = drawdown_multiplier(
                    nav_val, peak, self.dd_config.max_tolerance
                )
                mult = max(self.dd_config.min_multiplier, mult)
                dd_multipliers.append(mult)
            else:
                dd_multipliers.append(1.0)

        dd_series = pd.Series(dd_multipliers, index=weights.index)

        # 调整权重
        for date in weights.index:
            if date in dd_series.index:
                adjusted_weights.loc[date] = weights.loc[date] * dd_series[date]

        return adjusted_weights

    def compute_kelly_audit(self, nav_series: pd.Series) -> dict:
        """ACT-2: 计算 Kelly 审计.

        参数:
            nav_series: (T,) NAV 时序

        返回:
            dict: Kelly 审计结果
        """
        if not self.kelly_audit_enabled:
            return {}

        audit = kelly_audit(nav_series)
        self.kelly_results.append(audit)
        return audit

    def get_summary(self) -> dict:
        """获取风控层摘要."""
        return {
            "dd_enabled": self.dd_config.enabled,
            "dd_max_tolerance": self.dd_config.max_tolerance,
            "kelly_audit_enabled": self.kelly_audit_enabled,
            "n_kelly_audits": len(self.kelly_results),
        }


__all__ = ["RiskLayerV11"]
