"""仓位管理器

动态止损 + 仓位管理 + gamma预算
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .regime_detector import Regime

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置"""
    # 止损阈值（按状态）
    stop_loss_calm: float = 0.20
    stop_loss_transition: float = 0.15
    stop_loss_high_vol: float = 0.10
    stop_loss_vol_spike: float = 0.50
    stop_loss_compression: float = 0.15

    # Gamma引擎预算
    gamma_budget_base: float = 0.15
    gamma_budget_max: float = 0.25

    # 最大gamma头寸数
    max_gamma_positions: int = 3

    # 总权利金风险上限
    max_premium_risk_pct: float = 0.05


# 状态→止损映射
REGIME_STOP_LOSS = {
    Regime.CALM: "stop_loss_calm",
    Regime.TRANSITION: "stop_loss_transition",
    Regime.HIGH_VOL: "stop_loss_high_vol",
    Regime.VOL_SPIKE: "stop_loss_vol_spike",
    Regime.COMPRESSION: "stop_loss_compression",
}


class RiskManager:
    """仓位管理器

    - 动态止损（按状态）
    - 仓位大小控制
    - Gamma预算管理
    - 冷静期乘数应用
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()

    def get_stop_loss(self, regime: Regime) -> float:
        """获取状态对应的止损阈值"""
        attr = REGIME_STOP_LOSS.get(regime, "stop_loss_calm")
        return getattr(self.config, attr, 0.20)

    def compute_gamma_budget(
        self, equity: float, iv_percentile: float,
        cooldown_multiplier: float = 1.0,
    ) -> float:
        """计算gamma引擎预算

        base_budget = 15% * equity
        iv_adjustment = max(0, 0.5 - iv_pct) * 0.5 * equity
        gamma_budget = min(25% * equity, base + adjustment)
        """
        cfg = self.config
        base = cfg.gamma_budget_base
        iv_adj = max(0, 0.5 - iv_percentile) * 0.5
        gamma_pct = min(cfg.gamma_budget_max, base + iv_adj)

        return equity * gamma_pct * cooldown_multiplier

    def can_open_gamma_position(self, current_gamma_count: int) -> bool:
        """检查是否可以开新的gamma头寸"""
        return current_gamma_count < self.config.max_gamma_positions

    def check_premium_risk(
        self, total_premium_at_risk: float, equity: float
    ) -> bool:
        """检查总权利金风险是否超限"""
        return total_premium_at_risk / equity <= self.config.max_premium_risk_pct

    def apply_cooldown(
        self, base_qty: int, cooldown_multiplier: float
    ) -> int:
        """应用冷静期乘数"""
        return max(1, int(base_qty * cooldown_multiplier))
