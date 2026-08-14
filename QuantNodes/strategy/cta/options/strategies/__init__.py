# coding=utf-8
"""期权策略池 — 基类 + 注册表.

每个策略实现:
    - select(): 选合约 (返回 targets dict)
    - should_exit(): 是否需要平仓
    - describe(): 策略描述

策略通过 register() 注册到 STRATEGY_REGISTRY, 按名称调用.

已注册策略:
    - short_strangle: 卖宽跨式
    - iron_condor: 铁秃鹰
    - calendar_spread: 日历价差
    - covered_call: 备兑看涨
    - bull_call_spread: 牛市看涨价差
    - bear_put_spread: 熊市看跌价差
    - regime_adaptive: Regime 自适应 (根据 IV/HV 路由)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class StrategyConfig:
    """策略通用配置."""

    products: list[str] = field(default_factory=list)
    min_days: int = 30
    max_days: int = 60
    roll_days: int = 7
    max_positions: int = 1        # 每品种最大持仓组数


class OptionStrategy(ABC):
    """期权策略基类."""

    name: str = "base"
    description: str = ""

    def __init__(self, config: StrategyConfig | None = None):
        self._cfg = config or StrategyConfig()

    @abstractmethod
    def select(
        self,
        product: str,
        date: pd.Timestamp,
        underlying_price: float,
        existing_holding: dict | None,
        **kwargs,
    ) -> dict[str, float]:
        """选合约 → {symbol: qty}.

        Args:
            product: 品种代码
            date: 交易日
            underlying_price: 标的价格
            existing_holding: 当前持仓 (None=无持仓)

        Returns:
            {option_symbol: target_qty} (正=买, 负=卖)
        """
        ...

    def should_exit(
        self,
        product: str,
        date: pd.Timestamp,
        holding: dict,
        **kwargs,
    ) -> bool:
        """是否需要平仓 (默认: 到期前 roll_days 天)."""
        expiry = pd.Timestamp(holding.get("expiry", "2099-01-01"))
        return (expiry - date).days <= self._cfg.roll_days

    def describe(self) -> dict:
        """返回策略描述."""
        return {
            "name": self.name,
            "description": self.description,
            "config": {
                "min_days": self._cfg.min_days,
                "max_days": self._cfg.max_days,
                "roll_days": self._cfg.roll_days,
            },
        }


# ── 注册表 ────────────────────────────────────────────────────────────────

STRATEGY_REGISTRY: dict[str, type[OptionStrategy]] = {}


def register(cls: type[OptionStrategy]) -> type[OptionStrategy]:
    """注册策略到全局注册表."""
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str, config: StrategyConfig | None = None) -> OptionStrategy:
    """按名称获取策略实例."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name](config)


def list_strategies() -> list[str]:
    """列出所有已注册策略."""
    return list(STRATEGY_REGISTRY.keys())
