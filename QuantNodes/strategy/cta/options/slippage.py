# coding=utf-8
"""Bid-Ask 成本模型: 用 high-low 估计 spread, 对交易施加滑点.

方法:
    - Parkinson 估计: spread ≈ 2 × (high - low) / (high + low)
    - 期权 spread 随 delta 偏离 ATM 而扩大
    - 滑点 = spread × 0.5 (买入加, 卖出减)
"""
from __future__ import annotations

import pandas as pd


def build_option_prices(dates, option_data):
    """期权逐日价格序列 {symbol: [price]} + 到期结算价 map.

    关键修复: 期权 symbol 必须进 prices dict, 否则引擎 mark_prices.get(None)
    → place_order 被跳过, 期权层从未真正执行.
    """
    idx = pd.DatetimeIndex(dates)
    prices = {}
    settle = {}
    for prod, odf in option_data.items():
        if odf is None or odf.empty:
            continue
        # 处理两种情况: trade_date 在 index 或 column
        if "trade_date" in odf.columns:
            odf = odf.set_index("trade_date").sort_index()
        for sym, grp in odf.groupby("symbol"):
            if "close" not in grp.columns:
                continue
            ser = grp["close"].sort_index()
            ser = ser[~ser.index.duplicated(keep="last")]
            prices[sym] = ser.reindex(idx, method="ffill").fillna(0.0).tolist()
            settle[sym] = {d.strftime("%Y-%m-%d"): v for d, v in ser.items()}
    return prices, settle


def estimate_spread(high: float, low: float) -> float:
    """标的期货 bid-ask spread 估计 (Parkinson).

    Args:
        high: 最高价
        low:  最低价

    Returns:
        spread 比例 (0~1), 高/低 无效时返回 0.001 (最小 1 tick)
    """
    if high <= 0 or low <= 0 or high <= low:
        return 0.001
    return max(2.0 * (high - low) / (high + low), 0.001)


def estimate_option_spread(
    option_high: float,
    option_low: float,
    underlying_high: float,
    underlying_low: float,
    abs_delta: float = 0.5,
) -> float:
    """期权 bid-ask spread 估计.

    期权 spread 通常 ≥ 标的 spread, 且随 |delta| 偏离 0.5 而扩大.
    公式: max(raw_option_spread, underlying_spread × delta_adj)

    Args:
        option_high/low: 期权合约 high/low
        underlying_high/low: 标的期货 high/low
        abs_delta: |delta| (0~1), 0.5=ATM, 越小越 OTM

    Returns:
        spread 比例
    """
    raw = estimate_spread(option_high, option_low)
    und_spread = estimate_spread(underlying_high, underlying_low)
    # OTM 期权 spread 更宽: delta < 0.3 时乘 2~3 倍
    adj = 1.0 + max(0.5 - abs_delta, 0.0) * 4.0
    return max(raw, und_spread * adj, 0.001)


def apply_slippage(price: float, side: str, spread: float) -> float:
    """对交易价格施加滑点.

    Args:
        price: 成交价 (mid)
        side: 'buy' | 'sell' | 'open_long' | 'close_long' | ...
        spread: bid-ask spread 比例

    Returns:
        调整后价格
    """
    slip = spread * 0.5
    if side in ("buy", "open_long", "close_short", "buy_option"):
        return price * (1.0 + slip)
    return price * (1.0 - slip)
