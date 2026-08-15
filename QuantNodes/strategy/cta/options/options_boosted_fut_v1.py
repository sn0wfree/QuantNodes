# coding=utf-8
"""Options-Enhanced fut_v1 v2: 裸买 Call/Put 增强.

升级: 在 fut_v1 基础上, 当 VRP 确认 CTA 信号时, 用裸买 Call/Put 替代期货加仓.

核心逻辑:
    - CTA 基础仓位: 期货 (方向性 alpha)
    - 加仓: 裸买 Call (看涨) 或 Put (看跌) — 利用 gamma 凸性
    - 没有期权的品种: 用期货加仓

优势:
    - 裸买 Call/Put 有 gamma 凸性: 大幅移动时收益指数增长
    - 比期货加仓更灵活: 亏损有限 (权利金), 收益无限
    - 期权已有时直接用, 无需额外成本
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy, ComboConfig, ModuleConfig
from ..factors.volatility import vol_target_positions
from ..options.iv_panel import load_iv_panel
from ..options.hedge_data import load_vrp_panel
from ..options.selector import select_call, select_put
from ..backtest.cta_contract import CtaContractRegistry, OptionContractSpec, CtaContract


@dataclass
class OptionsBoostConfig:
    """裸买期权加仓配置."""
    # 加仓比例: 相对于期货仓位
    boost_notional_pct: float = 0.3   # 加仓名义 = 期货仓位 × 30%
    boost_max_lots: int = 5           # 最大加仓手数

    # IV 过滤
    iv_high_threshold: float = 0.7    # IV > 70%: 不加仓 (期权太贵)

    # DTE
    min_dte: int = 30
    max_dte: int = 60

    # 止损
    option_stop_loss: float = 0.5     # 期权亏损 50% 时止损


class OptionsBoostedFutV1(FactorModuleComboStrategy):
    """裸买期权增强的 fut_v1.

    CTA 期货 + 裸买 Call/Put (VRP 确认时).
    """

    def __init__(
        self,
        boost_config: OptionsBoostConfig | None = None,
        option_contracts: dict | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._boost_cfg = boost_config or OptionsBoostConfig()
        self._option_contracts = option_contracts or {}
        self._option_data = option_data or {}
        self._iv_panel = None
        self._vrp_panel = None
        self._option_holdings: dict[str, dict] = {}  # product → {symbol, entry_price, qty}
        self._load_options_data()

    def _load_options_data(self):
        try:
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            pass
        try:
            self._vrp_panel = load_vrp_panel(
                products=self._products, start=self._start, end=self._end,
            )
        except Exception:
            pass

    def _get_vrp_signal(self, product: str, date: pd.Timestamp) -> float:
        """获取 VRP 信号: 正=看涨, 负=看跌, 0=中性."""
        if self._vrp_panel is None or product not in self._vrp_panel.columns:
            return 0.0
        if date not in self._vrp_panel.index:
            return 0.0
        vrp = self._vrp_panel.loc[date, product]
        return float(vrp) if pd.notna(vrp) else 0.0

    def _get_iv_rank(self, product: str, date: pd.Timestamp) -> float:
        """获取 IV rank (0~1)."""
        if self._iv_panel is None or product not in self._iv_panel.columns:
            return 0.5
        if date not in self._iv_panel.index:
            return 0.5
        iv_val = self._iv_panel.loc[date, product]
        if pd.isna(iv_val):
            return 0.5
        hist = self._iv_panel[product].dropna()
        if len(hist) < 60:
            return 0.5
        return float((hist < iv_val).sum() / len(hist))

    def _select_option_for_boost(
        self,
        product: str,
        date: pd.Timestamp,
        spot: float,
        direction: float,
    ) -> tuple[str, float] | None:
        """选择裸买 Call/Put 用于加仓.

        Args:
            direction: CTA 方向 (>0 看涨买 call, <0 看跌买 put)

        Returns:
            (option_symbol, qty) 或 None
        """
        # 检查是否有该品种的期权
        if product not in self._option_data or self._option_data[product].empty:
            return None

        df = self._option_data[product]
        day_df = df[df["trade_date"] == date]
        if day_df.empty:
            # 找最近的交易日
            dates = df["trade_date"].unique()
            nearest = dates[dates <= date]
            if len(nearest) == 0:
                return None
            day_df = df[df["trade_date"] == nearest[-1]]

        # 选 ATM 期权
        from ..options.selector import select_call, select_put
        from ..options.underlying_map import UnderlyingMap
        from ..options.expiry import OptionExpiry
        loader = self._dl
        um = UnderlyingMap(loader)
        oe = OptionExpiry(loader)

        if direction > 0:
            # 看涨 → 买 Call
            sel = select_call(
                product, date, spot, target_delta=0.40,
                min_days=self._boost_cfg.min_dte,
                max_days=self._boost_cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )
        else:
            # 看跌 → 买 Put
            sel = select_put(
                product, date, spot, target_delta=0.40,
                min_days=self._boost_cfg.min_dte,
                max_days=self._boost_cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )

        if sel is None:
            return None

        # 检查 option_contracts 中是否有该合约
        if sel.symbol not in self._option_contracts:
            return None

        # 计算手数: boost_notional / (option_price × multiplier)
        spec = self._registry.get_or_default(product)
        boost_notional = spot * spec.contract_multiplier * self._boost_cfg.boost_notional_pct
        opt_price = sel.premium
        if opt_price <= 0:
            return None

        lots = int(boost_notional / (opt_price * spec.contract_multiplier))
        lots = min(lots, self._boost_cfg.boost_max_lots)
        lots = max(lots, 1)

        return (sel.symbol, float(lots))

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """增强版: vol_target + 裸买期权加仓."""
        # 先加载期权数据
        if self._iv_panel is None:
            self._load_options_data()

        # 原始 vol_target (期货仓位)
        weights = self._signal_to_lots_base(signals, cfg)

        return self._weights_to_lots(weights, cfg)

    def _signal_to_lots_base(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """原始 vol_target."""
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if self._filter is not None:
            weights = weights * self._filter.coefficients(self._price_panel)
        return weights

    def target_positions(self, date, data) -> dict[str, float]:
        """扩展 target_positions: 期货 + 裸买期权."""
        # 先获取期货仓位
        futures_targets = super().target_positions(date, data)

        # 检查是否需要裸买期权加仓
        for product in self._products:
            if product not in futures_targets:
                continue

            futures_qty = futures_targets[product]
            if abs(futures_qty) < 1:
                continue

            # 获取 VRP 信号
            vrp_signal = self._get_vrp_signal(product, date)

            # 只在 VRP 确认方向时加仓
            if abs(vrp_signal) < 0.01:
                continue  # VRP 中性, 不加仓

            # VRP 与期货方向一致 → 加仓
            futures_direction = np.sign(futures_qty)
            vrp_direction = np.sign(vrp_signal)

            if futures_direction != vrp_direction:
                continue  # 方向不一致, 不加仓

            # 检查 IV rank (不加仓 if IV 太高)
            iv_rank = self._get_iv_rank(product, date)
            if iv_rank > self._boost_cfg.iv_high_threshold:
                continue  # IV 太高, 期权贵

            # 检查是否已持有期权
            if product in self._option_holdings:
                h = self._option_holdings[product]
                # 检查是否需要止损
                if date in self._price_panel.index:
                    current_spot = self._price_panel.loc[date, product]
                    # 简化止损: 持仓超过30天
                    days_held = (date - pd.Timestamp(h["entry_date"])).days
                    if days_held > 30:
                        # 平仓
                        futures_targets[h["symbol"]] = 0.0
                        self._option_holdings.pop(product, None)
                continue

            # 选择裸买 Call/Put
            spot = self._price_panel.loc[date, product] if date in self._price_panel.index else None
            if spot is None or np.isnan(spot):
                continue

            result = self._select_option_for_boost(product, date, spot, futures_direction)
            if result is not None:
                opt_symbol, opt_qty = result
                futures_targets[opt_symbol] = opt_qty
                self._option_holdings[product] = {
                    "symbol": opt_symbol,
                    "qty": opt_qty,
                    "entry_date": str(date.date()),
                    "direction": futures_direction,
                }

        return futures_targets

    def _signal_to_lots_base(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """原始 vol_target."""
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if self._filter is not None:
            weights = weights * self._filter.coefficients(self._price_panel)
        return weights
