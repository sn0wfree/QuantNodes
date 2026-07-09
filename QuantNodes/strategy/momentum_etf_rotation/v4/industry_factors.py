# coding=utf-8
"""行业量价因子 — 华西证券《行业有效量价因子与行业轮动策略》实施.

11 个因子 (从 6 大类):
  动量:
    1. 二阶动量 (second_order_momentum)
    2. 动量期限差 (momentum_term_diff)
  交易波动:
    3. 成交金额波动 (amount_volatility)  [反转]
    4. 成交量波动 (volume_volatility)  [反转]
  换手率:
    5. 换手率变化 (turnover_change)
  多空对比:
    6. 多空对比总量 (long_short_total)  [反转]
    7. 多空对比变化 (long_short_change)
  量价背离:
    8. 量价排序协方差 (rank_covariance)  [反转]
    9. 量价相关系数 (price_volume_corr)  [反转]
   10. 一阶量价背离 (first_order_divergence)  [反转]
  量幅同向:
   11. 量幅同向 (volume_range_codirection)

参考:
  - 华西证券《行业有效量价因子与行业轮动策略》 (2022-08-22)
  - reports/momentum_etf_rotation/v4/papers/huaxi_industry_rotation.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


def _ewma(s: pd.Series, window: int) -> pd.Series:
    """指数加权移动平均 (ewm with adjust=False, alpha=2/(window+1))."""
    return s.ewm(span=window, adjust=False, min_periods=max(1, window // 2)).mean()


def _second_order_momentum(
    close: pd.Series,
    window1: int = 20,
    window2: int = 5,
    ewma_window: int = 20,
) -> pd.Series:
    """二阶动量: EWMA(close_dev - delay(close_dev, window2), window).

    close_dev_t = (close_t - mean(close[-window1:t])) / mean(close[-window1:t])

    Returns:
        pd.Series, index=close.index, monthly value
    """
    mean_w1 = close.rolling(window1).mean()
    close_dev = (close / mean_w1 - 1.0)
    delayed = close_dev.shift(window2)
    diff = close_dev - delayed
    return _ewma(diff, ewma_window)


def _momentum_term_diff(
    close: pd.Series,
    window1: int = 120,
    window2: int = 20,
) -> pd.Series:
    """动量期限差: (close_t - close_{t-window1}) / close_{t-window1}
                   - (close_t - close_{t-window2}) / close_{t-window2}.

    window1 > window2.
    """
    ret1 = close.pct_change(window1)
    ret2 = close.pct_change(window2)
    return ret1 - ret2


def _amount_volatility(amount: pd.Series, window: int = 20) -> pd.Series:
    """成交金额波动 (取反): -STD(amount, window)."""
    return -amount.rolling(window).std()


def _volume_volatility(volume: pd.Series, window: int = 20) -> pd.Series:
    """成交量波动 (取反): -STD(volume, window)."""
    return -volume.rolling(window).std()


def _turnover_change(
    turnover: pd.Series,
    window1: int = 60,
    window2: int = 5,
) -> pd.Series:
    """换手率变化: Mean(turnover[-window1:t]) / Mean(turnover[-window2:t]).

    window1 > window2. 因子值越大说明短期换手率相对长期较低 (一致预期).
    """
    mean_long = turnover.rolling(window1).mean()
    mean_short = turnover.rolling(window2).mean()
    return mean_long / mean_short.replace(0, np.nan)


def _long_short_total(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """多空对比总量 (取反): -sum(close-low)/(high-close), t-window+1 to t.

    因子值越大 (反转) 说明过去空头力量较强.
    """
    long_pow = (close - low)
    short_pow = (high - close)
    ratio = long_pow / short_pow.replace(0, np.nan)
    return -ratio.rolling(window).sum()


def _long_short_change(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window1: int = 60,
    window2: int = 5,
) -> pd.Series:
    """多空对比变化: EWMA(Volume × ((close-low)-(high-close))/(high-low), window1)
                    - EWMA(same, window2).

    window1 > window2. 因子值越大说明近期多头减弱.
    """
    diff = (close - low) - (high - close)
    rng = (high - low).replace(0, np.nan)
    score = volume * diff / rng
    return _ewma(score, window1) - _ewma(score, window2)


def _rank_covariance(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """量价排序协方差 (取反): -rank{cov[rank(close), rank(volume), window]}."""
    rank_close = close.rank()
    rank_volume = volume.rank()
    cov = rank_close.rolling(window).cov(rank_volume)
    return -cov.rank(pct=True, na_option="bottom")


def _price_volume_correlation(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """量价相关系数 (取反): -corr(close, volume, window)."""
    return -close.rolling(window).corr(volume)


def _first_order_divergence(
    close: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """一阶量价背离 (取反):
        -corr[rank(Volume_i / Volume_{i-1} - 1),
              rank(Close_i / Open_i - 1), window]
    """
    vol_change = volume / volume.shift(1) - 1.0
    price_change = close / open_ - 1.0
    rank_vol = vol_change.rank()
    rank_price = price_change.rank()
    return -rank_vol.rolling(window).corr(rank_price)


def _volume_range_codirection(
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """量幅同向: corr[rank(Volume_i / Volume_{i-1} - 1),
                         rank(High_i / Low_i - 1), window].
    """
    vol_change = volume / volume.shift(1) - 1.0
    range_change = high / low - 1.0
    rank_vol = vol_change.rank()
    rank_range = range_change.rank()
    return rank_vol.rolling(window).corr(rank_range)


@dataclass
class FactorEngineConfig:
    """11 因子引擎配置.

    默认窗口基于华西论文 (具体值见各因子函数 docstring).
    """
    second_order_window1: int = 20
    second_order_window2: int = 5
    second_order_ewma: int = 20

    momentum_window1: int = 120
    momentum_window2: int = 20

    volatility_window: int = 20

    turnover_window1: int = 60
    turnover_window2: int = 5

    long_short_window: int = 20
    long_short_change_window1: int = 60
    long_short_change_window2: int = 5

    rank_cov_window: int = 20
    pv_corr_window: int = 20
    first_order_window: int = 20
    volume_range_window: int = 20

    name_map: dict[str, str] = None

    def __post_init__(self):
        if self.name_map is None:
            self.name_map = {
                "f1_second_mom": "二阶动量",
                "f2_mom_term": "动量期限差",
                "f3_amt_vol": "成交金额波动",
                "f4_vol_vol": "成交量波动",
                "f5_turnover": "换手率变化",
                "f6_ls_total": "多空对比总量",
                "f7_ls_change": "多空对比变化",
                "f8_pv_rankcov": "量价排序协方差",
                "f9_pv_corr": "量价相关系数",
                "f10_first_div": "一阶量价背离",
                "f11_vol_range": "量幅同向",
            }


def compute_single_factor(
    ohlc: pd.DataFrame,
    factor: str,
    cfg: FactorEngineConfig | None = None,
) -> pd.Series:
    """计算单只 ETF 单个因子.

    Args:
        ohlc: DataFrame, columns=[open, high, low, close, volume]
        factor: 因子名 (e.g. "f1_second_mom")
        cfg: 配置

    Returns:
        pd.Series, monthly (or daily) factor value
    """
    cfg = cfg or FactorEngineConfig()
    close = ohlc["close"]
    high = ohlc["high"]
    low = ohlc["low"]
    open_ = ohlc["open"]
    volume = ohlc["volume"]
    amount = close * volume
    turnover = volume

    if factor == "f1_second_mom":
        return _second_order_momentum(close, cfg.second_order_window1, cfg.second_order_window2, cfg.second_order_ewma)
    elif factor == "f2_mom_term":
        return _momentum_term_diff(close, cfg.momentum_window1, cfg.momentum_window2)
    elif factor == "f3_amt_vol":
        return _amount_volatility(amount, cfg.volatility_window)
    elif factor == "f4_vol_vol":
        return _volume_volatility(volume, cfg.volatility_window)
    elif factor == "f5_turnover":
        return _turnover_change(turnover, cfg.turnover_window1, cfg.turnover_window2)
    elif factor == "f6_ls_total":
        return _long_short_total(close, high, low, cfg.long_short_window)
    elif factor == "f7_ls_change":
        return _long_short_change(close, high, low, volume,
                                  cfg.long_short_change_window1, cfg.long_short_change_window2)
    elif factor == "f8_pv_rankcov":
        return _rank_covariance(close, volume, cfg.rank_cov_window)
    elif factor == "f9_pv_corr":
        return _price_volume_correlation(close, volume, cfg.pv_corr_window)
    elif factor == "f10_first_div":
        return _first_order_divergence(close, open_, volume, cfg.first_order_window)
    elif factor == "f11_vol_range":
        return _volume_range_codirection(high, low, volume, cfg.volume_range_window)
    else:
        raise ValueError(f"未知因子: {factor}")


def compute_all_factors(
    ohlc: pd.DataFrame,
    cfg: FactorEngineConfig | None = None,
) -> pd.DataFrame:
    """计算单只 ETF 全部 11 因子.

    Returns:
        DataFrame, index=date, columns=11 factors
    """
    cfg = cfg or FactorEngineConfig()
    factors = list(cfg.name_map.keys())
    out = pd.DataFrame(index=ohlc.index)
    for fac in factors:
        out[fac] = compute_single_factor(ohlc, fac, cfg)
    return out


def compute_all_factors_panel(
    ohlcv_panel: pd.DataFrame,
    cfg: FactorEngineConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """计算多只 ETF 全部 11 因子.

    Args:
        ohlcv_panel: 多级 columns, (code, field) with field in {open,high,low,close,volume}
        cfg: 配置

    Returns:
        dict, code → DataFrame of 11 factors
    """
    cfg = cfg or FactorEngineConfig()
    codes = sorted(set(ohlcv_panel.columns.get_level_values(0)))
    out = {}
    for code in codes:
        try:
            sub = ohlcv_panel[code].copy()
            sub = sub.dropna(how="all")
            if len(sub) < 252:
                continue
            out[code] = compute_all_factors(sub, cfg)
        except Exception as e:
            print(f"  [{code}] 因子计算失败: {e}")
            continue
    return out


__all__ = [
    "FactorEngineConfig",
    "compute_single_factor",
    "compute_all_factors",
    "compute_all_factors_panel",
    "_second_order_momentum",
    "_momentum_term_diff",
    "_amount_volatility",
    "_volume_volatility",
    "_turnover_change",
    "_long_short_total",
    "_long_short_change",
    "_rank_covariance",
    "_price_volume_correlation",
    "_first_order_divergence",
    "_volume_range_codirection",
]
