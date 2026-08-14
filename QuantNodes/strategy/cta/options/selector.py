# coding=utf-8
"""期权合约选择器: 给定品种/日期/目标 delta → 虚值 Put 合约.

逻辑:
    1. 加载当日该产品全部期权合约 (v_all_options)
    2. 过滤流动性 (volume > 0 或 OI > 0)
    3. 候选到期窗口 [min_days, max_days] (近月 30-60 天)
    4. 对候选合约反解 delta (compute_greeks_for_option)
    5. 选 |delta - target_delta| 最小 (delta 负值取绝对值)

输出: OptionSelection (symbol / strike / delta / premium / expiry / notional 建议).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from .greeks import compute_greeks_for_option
from .underlying_map import UnderlyingMap
from .expiry import OptionExpiry


@dataclass(slots=True)
class OptionSelection:
    """选中的期权合约."""

    product: str
    symbol: str
    option_type: str          # 'P' / 'C'
    strike: float
    underlying_price: float
    delta: float
    premium: float            # 单位权利金 (点)
    expiry: str               # 'YYYY-MM-DD'
    days_to_expiry: int

    @property
    def multiplier(self) -> float:
        """合约乘数 (由 product 规格决定, 选择器不持有 registry, 外部提供)."""
        return 0.0


def _filter_liquid(day_df: pd.DataFrame) -> pd.DataFrame:
    """过滤低流动性合约 (成交量或持仓量 > 0)."""
    liq = day_df[(day_df["volume"].fillna(0) > 0)
                 | (day_df["open_interest"].fillna(0) > 0)]
    return liq if not liq.empty else day_df


def select_put(
    product: str,
    trade_date: str | pd.Timestamp,
    underlying_price: float,
    target_delta: float = 0.25,
    min_days: int = 30,
    max_days: int = 60,
    loader: CtaDataLoader | None = None,
    um: UnderlyingMap | None = None,
    oe: OptionExpiry | None = None,
    risk_free_rate: float = 0.02,
) -> OptionSelection | None:
    """选择虚值 Put 合约 (|delta - target| 最小).

    Args:
        product: 品种代码 'RB' / 'M' / 'CU'
        trade_date: 交易日
        underlying_price: 标的期货当日收盘价
        target_delta: 目标 delta (负值, 如 -0.25; 内部取绝对值处理)
        min_days / max_days: 到期天数窗口
        risk_free_rate: 无风险利率

    Returns:
        OptionSelection 或 None (无合适合约)
    """
    target_abs = abs(target_delta)
    loader = loader or CtaDataLoader()
    um = um or UnderlyingMap(loader)
    oe = oe or OptionExpiry(loader)
    ts = pd.Timestamp(trade_date)

    df = loader.load_v_all_options(
        products=[product],
        start=str(ts.date()), end=str(ts.date()),
    )
    if df is None or df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    day = df[df["trade_date"] == ts]
    if day.empty:
        return None
    day = _filter_liquid(day)

    # 到期日 → 剩余天数
    syms = day["symbol"].unique().tolist()
    exp_frame = oe.get_frame(syms)
    exp_map = dict(zip(exp_frame["symbol"], exp_frame["expiry_date"]))
    day["_expiry"] = day["symbol"].str.upper().map(exp_map)
    day["_expiry"] = pd.to_datetime(day["_expiry"], errors="coerce")
    day["_ttm_days"] = (day["_expiry"] - ts).dt.days

    # 到期窗口过滤 (窗口空 → 取最近到期月的全部合约)
    in_window = day[day["_ttm_days"].between(min_days, max_days)]
    if in_window.empty and day["_ttm_days"].notna().any():
        min_ttm = day["_ttm_days"].min()
        in_window = day[day["_ttm_days"] == min_ttm]
    if in_window.empty:
        return None

    # 只考虑 Put (option_type P)
    puts = in_window[in_window["option_type"].str.upper() == "P"]
    if puts.empty:
        return None

    # 反解 delta (需要标的价映射)
    best = None
    best_dist = np.inf
    for _, row in puts.iterrows():
        ttm = row["_ttm_days"] / 365.0 if pd.notna(row["_ttm_days"]) else None
        if ttm is None or ttm <= 0:
            continue
        g = compute_greeks_for_option(
            symbol=row["symbol"],
            option_type="P",
            strike_price=row["strike_price"],
            underlying_price=underlying_price,
            market_price=row["close"],
            time_to_maturity=ttm,
            risk_free_rate=risk_free_rate,
        )
        if not g.valid:
            continue
        dist = abs(abs(g.delta) - target_abs)
        if dist < best_dist:
            best_dist = dist
            best = OptionSelection(
                product=product,
                symbol=row["symbol"],
                option_type="P",
                strike=row["strike_price"],
                underlying_price=underlying_price,
                delta=g.delta,
                premium=float(row["close"]),
                expiry=str(row["_expiry"].date()) if pd.notna(row["_expiry"]) else "",
                days_to_expiry=int(row["_ttm_days"]),
            )
    return best


def select_call(
    product: str,
    trade_date: str | pd.Timestamp,
    underlying_price: float,
    target_delta: float = 0.25,
    min_days: int = 30,
    max_days: int = 60,
    loader: CtaDataLoader | None = None,
    um: UnderlyingMap | None = None,
    oe: OptionExpiry | None = None,
    risk_free_rate: float = 0.02,
) -> OptionSelection | None:
    """选择虚值 Call 合约 (与 select_put 同构, 只过滤 Call)."""
    target_abs = abs(target_delta)
    loader = loader or CtaDataLoader()
    um = um or UnderlyingMap(loader)
    oe = oe or OptionExpiry(loader)
    ts = pd.Timestamp(trade_date)

    df = loader.load_v_all_options(
        products=[product],
        start=str(ts.date()), end=str(ts.date()),
    )
    if df is None or df.empty:
        return None
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    day = df[df["trade_date"] == ts]
    if day.empty:
        return None
    day = _filter_liquid(day)

    syms = day["symbol"].unique().tolist()
    exp_frame = oe.get_frame(syms)
    exp_map = dict(zip(exp_frame["symbol"], exp_frame["expiry_date"]))
    day["_expiry"] = day["symbol"].str.upper().map(exp_map)
    day["_expiry"] = pd.to_datetime(day["_expiry"], errors="coerce")
    day["_ttm_days"] = (day["_expiry"] - ts).dt.days

    in_window = day[day["_ttm_days"].between(min_days, max_days)]
    if in_window.empty and day["_ttm_days"].notna().any():
        min_ttm = day["_ttm_days"].min()
        in_window = day[day["_ttm_days"] == min_ttm]
    if in_window.empty:
        return None

    calls = in_window[in_window["option_type"].str.upper() == "C"]
    if calls.empty:
        return None

    best = None
    best_dist = np.inf
    for _, row in calls.iterrows():
        ttm = row["_ttm_days"] / 365.0 if pd.notna(row["_ttm_days"]) else None
        if ttm is None or ttm <= 0:
            continue
        g = compute_greeks_for_option(
            symbol=row["symbol"],
            option_type="C",
            strike_price=row["strike_price"],
            underlying_price=underlying_price,
            market_price=row["close"],
            time_to_maturity=ttm,
            risk_free_rate=risk_free_rate,
        )
        if not g.valid:
            continue
        dist = abs(g.delta - target_abs)
        if dist < best_dist:
            best_dist = dist
            best = OptionSelection(
                product=product,
                symbol=row["symbol"],
                option_type="C",
                strike=row["strike_price"],
                underlying_price=underlying_price,
                delta=g.delta,
                premium=float(row["close"]),
                expiry=str(row["_expiry"].date()) if pd.notna(row["_expiry"]) else "",
                days_to_expiry=int(row["_ttm_days"]),
            )
    return best
