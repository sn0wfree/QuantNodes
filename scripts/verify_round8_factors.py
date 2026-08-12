# coding=utf-8
"""第八轮 11 因子真实数据验证 (中池 10 品种).

输出:
    1. 每因子 compute 信号非零数 (按品种)
    2. 每因子 raw_value 连续值统计 (NaN 比例 / 值域 / 均值)
"""
from __future__ import annotations

import pandas as pd

from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.factors.mean_reversion import HalfLifeFactor, SqueezeFactor
from QuantNodes.strategy.cta.factors.oscillators import (
    CCIFactor, MACDFactor, StochasticFactor, WilliamsRFactor,
)
from QuantNodes.strategy.cta.factors.trend_advanced import (
    AroonFactor, ChandelierExitFactor, LinearSlopeFactor, TRIXFactor, VortexFactor,
)

PRODUCTS = ["AU", "RB", "M", "CU", "IF", "I", "T", "TA", "MA", "C"]
START, END = "2018-01-01", "2025-12-31"

PANEL_FACTORS = {
    "macd": MACDFactor(),
    "aroon": AroonFactor(),
    "trix": TRIXFactor(),
    "linear_slope": LinearSlopeFactor(),
    "half_life": HalfLifeFactor(),
}
OHLC_FACTORS = {
    "stochastic": StochasticFactor(),
    "cci": CCIFactor(),
    "williams_r": WilliamsRFactor(),
    "vortex": VortexFactor(),
    "chandelier": ChandelierExitFactor(),
    "squeeze": SqueezeFactor(),
}


def load_panels(dl: CtaDataLoader):
    """返回 (close 面板, {product: ohlc_df})."""
    closes, ohlcs = {}, {}
    for p in PRODUCTS:
        try:
            df = dl.load_continuous_main(p, start=START, end=END)
        except Exception as e:
            print(f"  [{p}] skip: {e}")
            continue
        if df is None or df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date").sort_index()
        closes[p] = df["close"]
        ohlcs[p] = df[["open", "high", "low", "close"]]
    return pd.DataFrame(closes).sort_index(), ohlcs


def main():
    dl = CtaDataLoader()
    prices, ohlcs = load_panels(dl)
    print(f"面板: {prices.shape[0]} 日 × {prices.shape[1]} 品种 ({START}~{END})")

    print("\n=== 信号非零数 (compute, 全品种合计) ===")
    print(f"{'因子':<16}{'非零':>8}{'多':>8}{'空':>8}  品种分布")
    for name, f in list(PANEL_FACTORS.items()) + list(OHLC_FACTORS.items()):
        if name in PANEL_FACTORS:
            sig = f.compute(prices)
        else:
            cols = {}
            for p, ohlc in ohlcs.items():
                out = f.compute(ohlc)
                cols[p] = out[out.columns[0]]
            sig = pd.DataFrame(cols).reindex(prices.index)
        nz = (sig != 0).sum()
        long_n = (sig > 0).sum().sum()
        short_n = (sig < 0).sum().sum()
        dist = ", ".join(f"{c}:{int(v)}" for c, v in nz[nz > 0].items())
        print(f"{name:<16}{int(nz.sum()):>8}{int(long_n):>8}{int(short_n):>8}  {dist}")

    print("\n=== raw_value 统计 (全品种合并) ===")
    print(f"{'因子':<16}{'有效值':>10}{'NaN%':>8}{'min':>12}{'max':>12}{'mean':>12}")
    for name, f in list(PANEL_FACTORS.items()) + list(OHLC_FACTORS.items()):
        if name in PANEL_FACTORS:
            raw = f.raw_value(prices)
        else:
            cols = {}
            for p, ohlc in ohlcs.items():
                out = f.raw_value(ohlc)
                cols[p] = out[out.columns[0]]
            raw = pd.DataFrame(cols).reindex(prices.index)
        valid = raw.notna().sum().sum()
        total = raw.size
        vals = raw.to_numpy().ravel()
        vals = vals[~pd.isna(vals)]
        if vals.size:
            print(f"{name:<16}{valid:>10}{100 * (1 - valid / total):>7.1f}%"
                  f"{vals.min():>12.4g}{vals.max():>12.4g}{vals.mean():>12.4g}")
        else:
            print(f"{name:<16}{0:>10}{'100%':>8}  (无有效值)")


if __name__ == "__main__":
    main()
