"""策略路由器

根据市场状态将品种路由到具体的Theta或Gamma引擎策略
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .regime_detector import Regime
from .surface_selector import SurfaceSelector

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """策略路由配置"""
    # 止损阈值（按状态）
    stop_loss_calm: float = 0.20
    stop_loss_transition: float = 0.15
    stop_loss_high_vol: float = 0.10
    stop_loss_vol_spike: float = 0.50
    stop_loss_compression: float = 0.15

    # DTE窗口
    min_dte: int = 15
    max_dte: int = 40


# 状态→策略映射
REGIME_STRATEGY_MAP = {
    Regime.CALM: {
        "engine": "theta",
        "strategy": "sell_put_spread",
        "stop_loss_attr": "stop_loss_calm",
        "description": "卖put spread收theta",
    },
    Regime.TRANSITION: {
        "engine": "theta",
        "strategy": "sell_put_buy_call_wing",
        "stop_loss_attr": "stop_loss_transition",
        "description": "卖put+买call翼对冲尾部",
    },
    Regime.HIGH_VOL: {
        "engine": None,  # 高波不卖方，等待
        "strategy": None,
        "stop_loss_attr": None,
        "description": "高波环境不交易，等待明确信号",
    },
    Regime.VOL_SPIKE: {
        "engine": "gamma",
        "strategy": "buy_straddle",
        "stop_loss_attr": "stop_loss_vol_spike",
        "description": "买跨式赚gamma",
    },
    Regime.COMPRESSION: {
        "engine": "theta",
        "strategy": "sell_strangle",
        "stop_loss_attr": "stop_loss_compression",
        "description": "卖宽跨捕捉IV回落",
    },
}


class StrategyRouter:
    """策略路由器

    根据市场状态将品种路由到Theta或Gamma引擎
    """

    def __init__(self, config: Optional[StrategyConfig] = None):
        self.config = config or StrategyConfig()
        self.surface_selector = SurfaceSelector()

    def get_strategy_info(self, regime: Regime) -> Dict[str, Any]:
        """获取状态对应的策略信息"""
        return REGIME_STRATEGY_MAP.get(regime, {})

    def get_stop_loss(self, regime: Regime) -> float:
        """获取状态对应的止损阈值"""
        info = self.get_strategy_info(regime)
        attr = info.get("stop_loss_attr", "stop_loss_calm")
        return getattr(self.config, attr, 0.20)

    def route(
        self,
        regime: Regime,
        product: str,
        trade_date: str,
        underlying_price: float,
        option_data=None,
        loader=None,
        um=None,
        oe=None,
        risk_free_rate: float = 0.02,
    ) -> Optional[Dict[str, Any]]:
        """路由到具体策略

        Returns
        -------
        dict with keys:
            - engine: "theta" or "gamma"
            - strategy: strategy name
            - position: selected contract(s)
            - stop_loss: stop loss threshold
        """
        info = self.get_strategy_info(regime)
        if not info:
            return None

        strategy_type = info["strategy"]
        stop_loss = self.get_stop_loss(regime)

        # 选择合约
        position = self.surface_selector.select_for_strategy(
            strategy_type=strategy_type,
            product=product,
            trade_date=trade_date,
            underlying_price=underlying_price,
            option_data=option_data,
            min_days=self.config.min_dte,
            max_days=self.config.max_dte,
        )

        if position is None:
            logger.debug(
                f"No position found for {product} on {trade_date} "
                f"with strategy {strategy_type}"
            )
            return None

        return {
            "engine": info["engine"],
            "strategy": strategy_type,
            "position": position,
            "stop_loss": stop_loss,
            "regime": regime,
            "description": info["description"],
        }
