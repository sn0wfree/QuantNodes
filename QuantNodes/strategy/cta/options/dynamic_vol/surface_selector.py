"""曲面套利选股器 (快速版)

直接使用预加载的期权数据，避免每次调用都从DuckDB加载
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...common.option_greeks.smile import IVSmileAnalyzer

logger = logging.getLogger(__name__)


FALLBACK_DELTA = 0.25
SELL_BODY_DELTA = 0.25
SELL_WING_DELTA = 0.05


@dataclass
class SurfaceSelectorConfig:
    outlier_threshold: float = 1.5
    min_options_for_surface: int = 5
    fallback_delta: float = FALLBACK_DELTA


@dataclass
class FastOptionSelection:
    """快速期权选择结果"""
    product: str
    symbol: str
    option_type: str
    strike: float
    underlying_price: float
    delta: float
    premium: float
    expiry: str
    days_to_expiry: int


@dataclass
class FastPutSpread:
    """快速Put Spread"""
    long_put: FastOptionSelection
    short_put: FastOptionSelection
    net_premium: float


@dataclass
class FastStraddleSelection:
    """快速Straddle"""
    product: str
    call: FastOptionSelection
    put: FastOptionSelection
    strike: float
    expiry: str
    days_to_expiry: int
    cost: float
    delta: float
    gamma: float
    theta: float
    vega: float


class SurfaceSelector:
    """曲面套利选股器（快速版）"""

    def __init__(self, config: Optional[SurfaceSelectorConfig] = None):
        self.config = config or SurfaceSelectorConfig()
        self.smile_analyzer = IVSmileAnalyzer()

    def select_for_strategy(
        self,
        strategy_type: str,
        product: str,
        trade_date: str | pd.Timestamp,
        underlying_price: float,
        option_data: Optional[pd.DataFrame] = None,
        min_days: int = 15,
        max_days: int = 40,
    ) -> Any:
        """根据策略类型选择合约（快速版，不调用loader）"""
        if option_data is None or option_data.empty:
            return None

        ts = pd.Timestamp(trade_date)
        day_data = option_data[
            (option_data["product"] == product)
            & (option_data["trade_date"] == ts)
        ]
        if day_data.empty:
            return None

        # 过滤DTE窗口
        if "days_to_expiry" in day_data.columns:
            in_window = day_data[
                day_data["days_to_expiry"].between(min_days, max_days)
            ]
        else:
            # 没有DTE列，用全部数据
            in_window = day_data

        if in_window.empty:
            # fallback到最近到期的
            if "days_to_expiry" in day_data.columns:
                min_ttm = day_data["days_to_expiry"].min()
                in_window = day_data[day_data["days_to_expiry"] == min_ttm]
            else:
                in_window = day_data

        if in_window.empty:
            return None

        if strategy_type == "sell_put":
            return self._fast_select_put(in_window, product, underlying_price, ts)
        elif strategy_type == "sell_put_buy_call_wing":
            return self._fast_select_put_spread(in_window, product, underlying_price, ts)
        elif strategy_type == "buy_straddle":
            return self._fast_select_straddle(in_window, product, underlying_price, ts)
        elif strategy_type == "sell_iron_condor":
            return self._fast_select_iron_condor(in_window, product, underlying_price, ts)
        elif strategy_type == "sell_strangle":
            return self._fast_select_strangle(in_window, product, underlying_price, ts)
        elif strategy_type == "sell_put_spread":
            return self._fast_select_put_spread(in_window, product, underlying_price, ts)
        elif strategy_type == "buy_put_wing":
            return self._fast_select_put(in_window, product, underlying_price, ts,
                                          target_delta=SELL_WING_DELTA)
        else:
            return None

    def _fast_select_put(
        self, day_data, product, spot, date,
        target_delta=SELL_BODY_DELTA,
    ) -> Optional[FastOptionSelection]:
        """快速选put：找最接近目标delta的"""
        puts = day_data[day_data["option_type"].str.upper() == "P"]
        if puts.empty:
            return None

        best = None
        best_dist = np.inf

        for _, row in puts.iterrows():
            strike = row["strike_price"]
            premium = row["close"]
            if premium <= 0:
                continue

            # 简化delta估算
            moneyness = np.log(spot / strike) if spot > 0 and strike > 0 else 0
            # OTM put delta ≈ -N(-d1) ≈ -N(-moneyness/(σ√T))
            # 简化：用moneyness近似
            est_delta = -max(0.01, min(0.99, 0.5 - moneyness * 2))

            dist = abs(abs(est_delta) - target_delta)
            if dist < best_dist:
                best_dist = dist
                ttm = row.get("days_to_expiry", 30)
                expiry = str(date.date())  # 简化

                best = FastOptionSelection(
                    product=product,
                    symbol=row["symbol"],
                    option_type="P",
                    strike=strike,
                    underlying_price=spot,
                    delta=est_delta,
                    premium=float(premium),
                    expiry=expiry,
                    days_to_expiry=int(ttm) if pd.notna(ttm) else 30,
                )

        return best

    def _fast_select_straddle(
        self, day_data, product, spot, date,
    ) -> Optional[FastStraddleSelection]:
        """快速选straddle：找ATM的call+put对"""
        calls = day_data[day_data["option_type"].str.upper() == "C"]
        puts = day_data[day_data["option_type"].str.upper() == "P"]

        if calls.empty or puts.empty:
            return None

        # 找最接近ATM的strike
        best_strike = None
        best_dist = np.inf
        for strike in calls["strike_price"].unique():
            dist = abs(strike - spot)
            if dist < best_dist:
                best_dist = dist
                best_strike = strike

        if best_strike is None:
            return None

        c_row = calls[calls["strike_price"] == best_strike].iloc[0]
        p_row = puts[puts["strike_price"] == best_strike].iloc[0]

        call_prem = float(c_row["close"])
        put_prem = float(p_row["close"])

        if call_prem <= 0 or put_prem <= 0:
            return None

        ttm = c_row.get("days_to_expiry", 30)
        expiry = str(date.date())

        call_sel = FastOptionSelection(
            product=product, symbol=c_row["symbol"], option_type="C",
            strike=best_strike, underlying_price=spot,
            delta=0.5, premium=call_prem,
            expiry=expiry, days_to_expiry=int(ttm) if pd.notna(ttm) else 30,
        )
        put_sel = FastOptionSelection(
            product=product, symbol=p_row["symbol"], option_type="P",
            strike=best_strike, underlying_price=spot,
            delta=-0.5, premium=put_prem,
            expiry=expiry, days_to_expiry=int(ttm) if pd.notna(ttm) else 30,
        )

        return FastStraddleSelection(
            product=product, call=call_sel, put=put_sel,
            strike=best_strike, expiry=expiry,
            days_to_expiry=int(ttm) if pd.notna(ttm) else 30,
            cost=call_prem + put_prem, delta=0.0,
            gamma=0.0, theta=0.0, vega=0.0,
        )

    def _fast_select_put_spread(
        self, day_data, product, spot, date,
    ) -> Optional[FastPutSpread]:
        """快速选put spread"""
        long_leg = self._fast_select_put(
            day_data, product, spot, date, target_delta=SELL_BODY_DELTA,
        )
        short_leg = self._fast_select_put(
            day_data, product, spot, date, target_delta=SELL_WING_DELTA,
        )

        if long_leg is None or short_leg is None:
            return None
        if long_leg.strike <= short_leg.strike:
            return None

        net = long_leg.premium - short_leg.premium
        if net < 0:
            return None

        return FastPutSpread(
            long_put=long_leg, short_put=short_leg, net_premium=net,
        )

    def _fast_select_iron_condor(
        self, day_data, product, spot, date,
    ) -> Optional[Dict]:
        """快速选铁鹰"""
        body_put = self._fast_select_put(
            day_data, product, spot, date, target_delta=SELL_BODY_DELTA,
        )
        wing_put = self._fast_select_put(
            day_data, product, spot, date, target_delta=SELL_WING_DELTA,
        )

        calls = day_data[day_data["option_type"].str.upper() == "C"]
        body_call = None
        wing_call = None

        if not calls.empty:
            # body call: 0.25 delta
            best_strike = None
            best_dist = np.inf
            for strike in calls["strike_price"].unique():
                moneyness = np.log(strike / spot) if spot > 0 and strike > 0 else 0
                est_delta = max(0.01, min(0.99, 0.5 + moneyness * 2))
                dist = abs(est_delta - SELL_BODY_DELTA)
                if dist < best_dist:
                    best_dist = dist
                    best_strike = strike

            if best_strike is not None:
                c_row = calls[calls["strike_price"] == best_strike].iloc[0]
                ttm = c_row.get("days_to_expiry", 30)
                body_call = FastOptionSelection(
                    product=product, symbol=c_row["symbol"], option_type="C",
                    strike=best_strike, underlying_price=spot,
                    delta=0.25, premium=float(c_row["close"]),
                    expiry=str(date.date()),
                    days_to_expiry=int(ttm) if pd.notna(ttm) else 30,
                )

            # wing call: 0.05 delta
            best_strike2 = None
            best_dist2 = np.inf
            for strike in calls["strike_price"].unique():
                moneyness = np.log(strike / spot) if spot > 0 and strike > 0 else 0
                est_delta = max(0.01, min(0.99, 0.5 + moneyness * 2))
                dist = abs(est_delta - SELL_WING_DELTA)
                if dist < best_dist2:
                    best_dist2 = dist
                    best_strike2 = strike

            if best_strike2 is not None:
                c_row2 = calls[calls["strike_price"] == best_strike2].iloc[0]
                ttm2 = c_row2.get("days_to_expiry", 30)
                wing_call = FastOptionSelection(
                    product=product, symbol=c_row2["symbol"], option_type="C",
                    strike=best_strike2, underlying_price=spot,
                    delta=0.05, premium=float(c_row2["close"]),
                    expiry=str(date.date()),
                    days_to_expiry=int(ttm2) if pd.notna(ttm2) else 30,
                )

        if body_put and body_call and wing_put and wing_call:
            return {
                "body_put": body_put, "body_call": body_call,
                "wing_put": wing_put, "wing_call": wing_call,
            }
        return None

    def _fast_select_strangle(
        self, day_data, product, spot, date,
    ) -> Optional[Dict]:
        """快速选宽跨"""
        put = self._fast_select_put(
            day_data, product, spot, date, target_delta=SELL_BODY_DELTA,
        )

        calls = day_data[day_data["option_type"].str.upper() == "C"]
        call = None
        if not calls.empty:
            best_strike = None
            best_dist = np.inf
            for strike in calls["strike_price"].unique():
                moneyness = np.log(strike / spot) if spot > 0 and strike > 0 else 0
                est_delta = max(0.01, min(0.99, 0.5 + moneyness * 2))
                dist = abs(est_delta - SELL_BODY_DELTA)
                if dist < best_dist:
                    best_dist = dist
                    best_strike = strike

            if best_strike is not None:
                c_row = calls[calls["strike_price"] == best_strike].iloc[0]
                ttm = c_row.get("days_to_expiry", 30)
                call = FastOptionSelection(
                    product=product, symbol=c_row["symbol"], option_type="C",
                    strike=best_strike, underlying_price=spot,
                    delta=0.25, premium=float(c_row["close"]),
                    expiry=str(date.date()),
                    days_to_expiry=int(ttm) if pd.notna(ttm) else 30,
                )

        if put and call:
            return {"put": put, "call": call}
        return None
