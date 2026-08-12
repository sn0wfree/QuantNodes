# coding=utf-8
"""期权 BS 定价与希腊字母计算 — 封装 vendored option_greeks.

期权计算内核来自 vendored 包 `QuantNodes/strategy/cta/common/option_greeks`
(源码 vendored 自 ~/Public/option_cal/option_greeks, 不依赖 pip 安装):
    - BSPricing      BS 定价
    - calculate_iv   隐含波动率反解 (Newton + Brent)
    - calculate_all_greeks  delta / gamma / vega / theta / rho

本模块提供:
    - 与 CTA 数据层对接的便捷函数 (接受 pandas Series/行)
    - compute_greeks_for_option: 单只期权合约 → 全希腊字母
    - compute_straddle_greeks: 跨式组合希腊字母
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from QuantNodes.strategy.cta.common.option_greeks import (
    BSPricing,
    calculate_all_greeks,
    calculate_iv,
)

_CP_TO_TYPE = {"C": "call", "P": "put", "c": "call", "p": "put"}

# CFFEX 指数期权标的映射: IO→IF, MO→IM, HO→IH (标的为对应期货, 月同步)
CFFEX_UNDERLYING_MAP = {"IO": "IF", "MO": "IM", "HO": "IH"}


def map_cffex_underlying(underlying: str, product: str | None = None) -> str:
    """将 CFFEX 指数期权标的映射为对应期货合约代码.

    例: ('IO2502', 'IO') → 'IF2502'  (IO/MO/HO 的标的期货分别是 IF/IM/IH)
    非 CFFEX 或未匹配时原样返回.
    """
    prod = (product or underlying[:2]).upper()
    if prod in CFFEX_UNDERLYING_MAP:
        # underlying 形如 'IO2502', 替换产品前缀
        suffix = underlying[len(prod):] if underlying.upper().startswith(prod) else underlying[2:]
        return CFFEX_UNDERLYING_MAP[prod] + suffix
    return underlying


def days_to_expiry(trade_date, expiry_date) -> float:
    """距到期天数 → 年 (365 天计)."""
    if trade_date is None or expiry_date is None:
        return 0.0
    d = (pd.Timestamp(expiry_date) - pd.Timestamp(trade_date)).days
    return max(float(d), 1e-9) / 365.0


@dataclass(frozen=True, slots=True)
class GreeksResult:
    """单只期权合约希腊字母快照."""

    symbol: str
    option_type: str            # 'C' / 'P'
    strike_price: float
    underlying_price: float     # 标的期货价格
    time_to_maturity: float     # 剩余期限 (年)
    iv: float                   # 隐含波动率
    delta: float
    gamma: float
    vega: float                 # 每 1% 波动率
    theta: float                # 每 1 交易日
    rho: float
    model_price: float          # 理论价
    market_price: float         # 市价
    valid: bool                 # 计算是否成功 (IV 可解)


def compute_greeks_for_option(
    symbol: str,
    option_type: str,
    strike_price: float,
    underlying_price: float,
    market_price: float,
    time_to_maturity: float,
    risk_free_rate: float = 0.02,
) -> GreeksResult:
    """由期权市价反解 IV 并计算全希腊字母.

    Args:
        symbol: 期权合约代码
        option_type: 'C' / 'P'
        strike_price: 行权价
        underlying_price: 标的期货收盘价
        market_price: 期权市价 (close)
        time_to_maturity: 剩余期限 (年)
        risk_free_rate: 无风险利率 (默认 2%)

    Returns:
        GreeksResult, IV 无法反解时 valid=False
    """
    cp = _CP_TO_TYPE.get(str(option_type).upper(), "call")
    iv = np.nan
    diff = 1.0
    if market_price and market_price > 0 and time_to_maturity > 1e-6:
        try:
            iv, diff = calculate_iv(
                S=underlying_price, K=strike_price, r=risk_free_rate,
                T=time_to_maturity, option_price=float(market_price),
                cp_sign=cp,
            )
        except Exception:
            iv = np.nan

    valid = bool(not np.isnan(iv)) and abs(diff) < 1e-3
    if not valid:
        return GreeksResult(
            symbol=symbol, option_type=cp, strike_price=strike_price,
            underlying_price=underlying_price, time_to_maturity=time_to_maturity,
            iv=iv, delta=np.nan, gamma=np.nan, vega=np.nan,
            theta=np.nan, rho=np.nan, model_price=np.nan,
            market_price=float(market_price) if market_price else np.nan,
            valid=False,
        )

    g = calculate_all_greeks(
        s=underlying_price, k=strike_price, r=risk_free_rate, g=0,
        sigma=iv, t=time_to_maturity, option_type=cp,
    )
    model_price = BSPricing(
        s=underlying_price, k=strike_price, r=risk_free_rate,
        t=time_to_maturity, sigma=iv, cp_sign=cp,
    )
    return GreeksResult(
        symbol=symbol, option_type=cp, strike_price=strike_price,
        underlying_price=underlying_price, time_to_maturity=time_to_maturity,
        iv=float(iv), delta=float(g.delta), gamma=float(g.gamma),
        vega=float(g.vega), theta=float(g.theta), rho=float(g.rho),
        model_price=float(model_price),
        market_price=float(market_price),
        valid=True,
    )


def compute_options_greeks(
    options_df: pd.DataFrame,
    underlying_map: dict,
    expiry_map: dict,
    risk_free_rate: float = 0.02,
) -> list[GreeksResult]:
    """批量计算期权希腊字母.

    Args:
        options_df: 期权行情 DataFrame (含 symbol/option_type/strike_price/close/trade_date)
        underlying_map: {underlying_symbol: 标的价格} (当日)
        expiry_map: {underlying_symbol: expiry_date} (datetime)
        risk_free_rate: 无风险利率

    Returns:
        list[GreeksResult], 仅含计算成功的
    """
    results = []
    for _, row in options_df.iterrows():
        underlying = row.get("underlying")
        if underlying is None:
            continue
        # CFFEX 指数期权标的映射 (IO→IF 等), 若 map 中无该标的则原样
        mapped = map_cffex_underlying(underlying, row.get("product"))
        s_price = underlying_map.get(mapped) or underlying_map.get(underlying)
        if s_price is None or s_price <= 0:
            continue
        expiry = expiry_map.get(underlying)
        if expiry is None:
            continue
        T = days_to_expiry(row.get("trade_date"), expiry)
        r = compute_greeks_for_option(
            symbol=row["symbol"],
            option_type=row["option_type"],
            strike_price=row["strike_price"],
            underlying_price=s_price,
            market_price=row["close"],
            time_to_maturity=T,
            risk_free_rate=risk_free_rate,
        )
        if r.valid:
            results.append(r)
    return results


@dataclass(frozen=True, slots=True)
class StraddleGreeks:
    """跨式组合 (Call + Put) 希腊字母."""

    symbol_base: str          # 标的产品
    call: GreeksResult
    put: GreeksResult
    delta: float              # 组合 delta (通常接近 0)
    gamma: float              # 组合 gamma (long gamma 关键指标)
    vega: float
    theta: float
    cost: float               # 建仓成本 = call + put


def compute_straddle_greeks(
    call: GreeksResult, put: GreeksResult,
) -> Optional[StraddleGreeks]:
    """组合跨式希腊字母.

    要求 call 与 put 同 strike 同到期.
    """
    if not call.valid or not put.valid:
        return None
    return StraddleGreeks(
        symbol_base=call.symbol,
        call=call, put=put,
        delta=call.delta + put.delta,
        gamma=call.gamma + put.gamma,
        vega=call.vega + put.vega,
        theta=call.theta + put.theta,
        cost=call.market_price + put.market_price,
    )
