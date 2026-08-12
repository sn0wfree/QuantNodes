# coding=utf-8
"""CTA 回测引擎 — 期货/期权专属.

设计:
    - 复用 QuantNodes.backtest 的 Order/OrdersResult 数据结构
    - 复用 momentum_etf_rotation.common.metrics 的指标计算
    - 期货专属: 保证金占用 / 合约乘数 / 换月处理 / 手续费

模块:
    - cta_contract: 期货/期权合约规格
    - cta_broker:   仓位与现金管理 (保证金)
    - cta_engine:   逐日回测主循环
    - cta_risk:     风控节点
"""
from __future__ import annotations

from .cta_contract import CtaContract, CtaContractRegistry, CtaContractSpec
from .cta_broker import CtaBroker, Position, TradeRecord
from .cta_engine import CtaBacktestConfig, CtaBacktestResult, CtaStrategy, run_cta_backtest
from .cta_risk import CtaRisk, CompositeRisk, MarginRisk, PositionLimitRisk

__all__ = [
    "CtaContract",
    "CtaContractRegistry",
    "CtaContractSpec",
    "CtaBroker",
    "Position",
    "TradeRecord",
    "CtaBacktestConfig",
    "CtaBacktestResult",
    "CtaStrategy",
    "run_cta_backtest",
    "CtaRisk",
    "CompositeRisk",
    "MarginRisk",
    "PositionLimitRisk",
]
