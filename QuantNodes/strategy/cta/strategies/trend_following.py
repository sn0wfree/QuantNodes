# coding=utf-8
"""多品种时序动量趋势跟踪策略 (策略 v1).

流程:
    1. 加载主力连续面板 (continuous_main_{product}_daily)
    2. 构造对齐价格面板 (各产品按日期对齐)
    3. 计算唐奇安通道趋势信号
    4. 波动率目标仓位 (vol targeting)
    5. 组合回测 (run_cta_backtest)

用法:
    from QuantNodes.strategy.cta.strategies.trend_following import TrendFollowingStrategy

    strat = TrendFollowingStrategy(products=["AU","RB","M"], start="2018-01-01")
    result = strat.run()
    print(result.metrics)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_risk import PositionLimitRisk
from ..signals.trend_signal import donchian_signal
from ..factors.volatility import vol_target_positions

# 默认高流动性品种池 (训练窗口覆盖 ≥1900 天)
DEFAULT_PRODUCTS = [
    "IF", "IC", "IH", "T", "TF",          # 股指/国债
    "AU", "AG", "CU", "AL", "ZN", "RB", "HC", "NI",  # 有色/黑色
    "I", "J", "JM", "M", "Y", "P", "OI", "RM",  # 黑色/油粕
    "TA", "MA", "FG", "SF", "SM", "CF", "SR",   # 化工/软商品
    "A", "C", "JD", "PP", "L", "V", "BU",   # 农产品/化工
]


@dataclass
class TrendFollowingConfig:
    """趋势跟踪策略配置."""

    entry_window: int = 20          # 唐奇安入场通道
    exit_window: int = 10           # 唐奇安出场通道
    target_vol: float = 0.15        # 目标年化波动率
    vol_lookback: int = 20          # 波动率回看
    max_leverage: float = 3.0       # 最大杠杆
    initial_cash: float = 2_000_000.0
    commission_rate: float = 0.0001
    max_margin_ratio: float = 0.8   # 保证金占用上限
    min_position_change: float = 0.20  # 仓位变化 < 此比例不调仓 (降换手)


class TrendFollowingStrategy:
    """多品种趋势跟踪策略.

    Args:
        products: 产品池 (默认 DEFAULT_PRODUCTS)
        config: 策略配置
        data_loader: 数据加载器
        start/end: 回测区间
    """

    def __init__(
        self,
        products: Optional[list[str]] = None,
        config: Optional[TrendFollowingConfig] = None,
        data_loader: Optional[CtaDataLoader] = None,
        start: str = "2018-01-01",
        end: str = "2025-12-31",
    ):
        self._products = products or DEFAULT_PRODUCTS
        self._config = config or TrendFollowingConfig()
        self._dl = data_loader or CtaDataLoader()
        self._start = start
        self._end = end
        self._registry = CtaContractRegistry()

        # 预加载面板
        self._price_panel: pd.DataFrame | None = None
        self._dates: list[pd.Timestamp] | None = None

    # ── 数据 ────────────────────────────────────────────────────────────

    def load_panel(self) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
        """加载全部产品主力连续, 对齐为价格面板."""
        panels = {}
        for product in self._products:
            try:
                df = self._dl.load_continuous_main(
                    product, start=self._start, end=self._end,
                )
                if df is None or df.empty:
                    continue
                s = df.set_index("trade_date")["close"]
                s.index = pd.to_datetime(s.index)
                panels[product] = s
            except Exception as e:
                print(f"  [{product}] load failed: {e}")
        price_panel = pd.DataFrame(panels).sort_index()
        # 去除全 NaN 日 (部分产品停更日对齐问题)
        price_panel = price_panel.dropna(how="all")
        self._price_panel = price_panel
        self._dates = list(price_panel.index)
        return price_panel, self._dates

    def _signals(self) -> pd.DataFrame:
        """计算趋势信号."""
        if self._price_panel is None:
            self.load_panel()
        cfg = self._config
        return donchian_signal(
            self._price_panel,
            entry_window=cfg.entry_window,
            exit_window=cfg.exit_window,
        )

    def _positions(self) -> pd.DataFrame:
        """信号 × 波动率目标 → 目标仓位.

        返回每品种的目标手数 (整数), 权重 = vol_target / 每手名义价值占比.
        实现: 权重转手数 = floor(equity × weight × margin_ratio / (price × multiplier)).
        输出与 price_panel 同 index/columns.
        """
        cfg = self._config
        signals = self._signals()
        weights = vol_target_positions(
            signals,
            self._price_panel,
            target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback,
            max_leverage=cfg.max_leverage,
        )

        # 权重 → 目标手数
        # 资金在"当日有信号"的品种间等分 (逐日), 避免各品种都用满杠杆叠加
        equity = cfg.initial_cash
        signal_on = (weights.abs() > 1e-9).astype(float)      # (T,N) 0/1
        n_active = signal_on.sum(axis=1).clip(lower=1)        # (T,) 当日活跃品种数

        lots = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for product in weights.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            # 目标手数 = 当日信号数等分资金 × 波动率权重 / 每手名义价值
            capital_per_product = signal_on[product] * equity / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            lots[product] = target
        return lots

    # ── 回测 ────────────────────────────────────────────────────────────

    def run(self) -> CtaBacktestResult:
        """运行完整回测."""
        price_panel, dates = self.load_panel()
        positions = self._positions()

        # 将面板转成逐 symbol 输入 (symbol = product 简化, 无换月)
        cols = [p for p in self._products if p in price_panel.columns]
        prices_map = {p: price_panel[p].tolist() for p in cols}
        products_map = {p: p for p in prices_map}

        # 目标持仓策略: 趋势策略按信号驱动调仓,
        # 仅当方向变化 (0↔±1) 或翻多翻空时调仓, 避免每日波动率微调造成换手.
        class _PanelStrategy(CtaStrategy):
            def __init__(self, pos_df, min_change=0.0):
                self._pos = pos_df
                self._last = {}
                self._min_change = min_change

            def target_positions(self, date, data):
                if date not in self._pos.index:
                    return {}
                row = self._pos.loc[date]
                targets = {
                    p: float(row[p]) for p in self._pos.columns if abs(row[p]) > 1e-9
                }
                filtered = {}
                for p in set(targets) | set(self._last):
                    new_qty = targets.get(p, 0)
                    old_qty = self._last.get(p, 0)
                    def _sgn(x):
                        return 1 if x > 0 else (-1 if x < 0 else 0)
                    if _sgn(new_qty) == _sgn(old_qty):
                        max_qty = max(abs(old_qty), abs(new_qty), 1)
                        if abs(new_qty - old_qty) / max_qty < self._min_change:
                            filtered[p] = old_qty
                            continue
                    filtered[p] = new_qty
                changed = (filtered != self._last)
                if not changed:
                    return {}
                self._last = filtered
                return filtered

        # 风控: 仓位已由 vol-targeting 限制杠杆, 仅做单品种手数上限
        # (MarginRisk 按资金等比缩放会与手数取整冲突, 造成小数手数微调)
        risk = PositionLimitRisk(products=products_map, registry=self._registry)

        result = run_cta_backtest(
            dates=dates,
            prices=prices_map,
            products=products_map,
            strategy=_PanelStrategy(positions, min_change=self._config.min_position_change),
            config=CtaBacktestConfig(
                initial_cash=self._config.initial_cash,
                commission_rate=self._config.commission_rate,
            ),
            registry=self._registry,
            risk=risk,
        )
        return result

    # ── 便捷 ────────────────────────────────────────────────────────────

    @property
    def products(self) -> list[str]:
        return self._products
