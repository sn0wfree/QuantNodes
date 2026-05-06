# coding=utf-8
"""
TA-Lib 技术指标算子

本模块包含 TA-Lib 算子的实际实现。
TA-Lib 算子通过 map_batches 桥接 Polars 与 NumPy。

注意：本模块为实现层，operators/talib.py 为代理层。
"""

from __future__ import annotations

from typing import Union, Tuple

import polars as pl
from polars import Expr

from QuantNodes.factor_node.factor_functions._helpers import (
    OperatorCategory,
    register_operator,
)


def _ensure_expr(f: Union[Expr, str, float]) -> Expr:
    """将输入规范化为 Polars Expr"""
    if isinstance(f, Expr):
        return f
    if isinstance(f, str):
        return pl.col(f)
    return pl.lit(f)


def _to_numpy(expr: Union[Expr, str]) -> Expr:
    """将列转为 numpy 数组的 Expr（用于 map_batches）"""
    e = _ensure_expr(expr)
    return e.cast(pl.Float64)


def sma(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """简单移动平均线 (Simple Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.SMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def ema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """指数移动平均线 (Exponential Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.EMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def wma(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """加权移动平均线 (Weighted Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.WMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def dema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """双重指数移动平均线 (Double Exponential Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.DEMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def tema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """三重指数移动平均线 (Triple Exponential Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.TEMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def trima(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """三角移动平均线 (Triangular Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.TRIMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def kama(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """考夫曼自适应移动平均线 (Kaufman Adaptive Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.KAMA(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def t3(expr: Union[Expr, str], timeperiod: int = 5, vfactor: float = 0.7) -> Expr:
    """T3 移动平均线 (Triple Exponential Moving Average with volume factor)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.T3(s.to_numpy(), timeperiod=timeperiod, vfactor=vfactor)),
        return_dtype=pl.Float64,
    )


def ht_trendline(expr: Union[Expr, str]) -> Expr:
    """希尔伯特变换趋势线 (Hilbert Transform Trendline)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.HT_TRENDLINE(s.to_numpy())),
        return_dtype=pl.Float64,
    )


def mama(expr: Union[Expr, str], fastlimit: float = 0.5, slowlimit: float = 0.05) -> Expr:
    """MAMA 移动平均 (MESA Adaptive Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    def _mama(s: pl.Series) -> pl.Series:
        mama_line, _ = talib.MAMA(s.to_numpy(), fastlimit=fastlimit, slowlimit=slowlimit)
        return pl.Series(mama_line)
    return e.map_batches(_mama, return_dtype=pl.Float64)


def mavp(expr: Union[Expr, str], periods: Union[Expr, str], minperiod: int = 2, maxperiod: int = 30) -> Expr:
    """移动平均变动周期 (MA with Variable Period)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MAVP(s.to_numpy(), s.to_numpy(), minperiod=minperiod, maxperiod=maxperiod)),
        return_dtype=pl.Float64,
    )


def midpoint(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """价格中点 (Midpoint over period)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MIDPOINT(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def midprice(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """中间价格 (Midpoint Price over period)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MIDPRICE(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def sar(expr: Union[Expr, str], acceleration: float = 0.02, maximum: float = 0.2) -> Expr:
    """抛物线 SAR"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.SAR(s.to_numpy(), s.to_numpy(), acceleration=acceleration, maximum=maximum)),
        return_dtype=pl.Float64,
    )


def ma(expr: Union[Expr, str], timeperiod: int = 30, matype: int = 0) -> Expr:
    """通用移动平均线 (Moving Average)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MA(s.to_numpy(), timeperiod=timeperiod, matype=matype)),
        return_dtype=pl.Float64,
    )


def bbands(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
           nbdevdn: float = 2.0, matype: int = 0) -> Tuple[Expr, Expr, Expr]:
    """布林带 — 返回 (upper, middle, lower)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[1]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[2]), return_dtype=pl.Float64),
    )


def bbands_upper(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                 nbdevdn: float = 2.0, matype: int = 0) -> Expr:
    """布林带上轨"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0]),
        return_dtype=pl.Float64,
    )


def bbands_middle(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                  nbdevdn: float = 2.0, matype: int = 0) -> Expr:
    """布林带中轨"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[1]),
        return_dtype=pl.Float64,
    )


def bbands_lower(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                 nbdevdn: float = 2.0, matype: int = 0) -> Expr:
    """布林带下轨"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[2]),
        return_dtype=pl.Float64,
    )


def rsi(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """相对强弱指标 (Relative Strength Index)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.RSI(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def macd(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
         signalperiod: int = 9) -> Tuple[Expr, Expr, Expr]:
    """MACD — 返回 (macd_line, signal_line, histogram)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2]), return_dtype=pl.Float64),
    )


def macd_line(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
              signalperiod: int = 9) -> Expr:
    """MACD 线"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0]),
        return_dtype=pl.Float64,
    )


def macd_signal(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                signalperiod: int = 9) -> Expr:
    """MACD 信号线"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1]),
        return_dtype=pl.Float64,
    )


def macd_hist(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
              signalperiod: int = 9) -> Expr:
    """MACD 柱状图 (histogram)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2]),
        return_dtype=pl.Float64,
    )


def stoch(expr: Union[Expr, str], fastk_period: int = 5, slowk_period: int = 3,
          slowk_matype: int = 0, slowd_period: int = 3, slowd_matype: int = 0) -> Tuple[Expr, Expr]:
    """随机指标 (Stochastic) — 返回 (slowk, slowd)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(), fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(), fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[1]), return_dtype=pl.Float64),
    )


def stoch_k(expr: Union[Expr, str], fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0) -> Expr:
    """随机指标 K 线 (Stochastic %K)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(), fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype)[0]),
        return_dtype=pl.Float64,
    )


def stoch_d(expr: Union[Expr, str], fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0,
            slowd_period: int = 3, slowd_matype: int = 0) -> Expr:
    """随机指标 D 线 (Stochastic %D)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(), fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)[1]),
        return_dtype=pl.Float64,
    )


def cci(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """商品通道指标 (Commodity Channel Index)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CCI(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def willr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """威廉指标 (Williams %R)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.WILLR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def mfi(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """资金流量指标 (Money Flow Index)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MFI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def roc(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
    """变动率 (Rate of Change)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ROC(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def rocp(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
    """变动率百分比 (Rate of Change Percentage)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ROCP(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def rocr(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
    """变动率比率 (Rate of Change Ratio)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ROCR(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def rocr100(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
    """变动率比率*100 (Rate of Change Ratio 100 scale)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ROCR100(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def mom(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
    """动量 (Momentum)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MOM(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def adx(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """平均趋向指标 (Average Directional Movement Index)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ADX(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def adxr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """平均趋向指标评估 (Average Directional Movement Index Rating)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ADXR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def apo(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
    """绝对价格震荡 (Absolute Price Oscillator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.APO(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)),
        return_dtype=pl.Float64,
    )


def ppo(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
    """百分比价格震荡 (Percentage Price Oscillator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.PPO(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)),
        return_dtype=pl.Float64,
    )


def cmo(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """钱德动量摆动指标 (Chande Momentum Oscillator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CMO(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def dx(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """趋向指标 (Directional Movement Index)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.DX(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def minus_di(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """负向指标 (Minus Directional Indicator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MINUS_DI(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def plus_di(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """正向指标 (Plus Directional Indicator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.PLUS_DI(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def minus_dm(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """负向动量 (Minus Directional Movement)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MINUS_DM(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def plus_dm(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """正向动量 (Plus Directional Movement)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.PLUS_DM(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def aroon(expr: Union[Expr, str], timeperiod: int = 14) -> Tuple[Expr, Expr]:
    """阿隆指标 — 返回 (aroondown, aroonup)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[1]), return_dtype=pl.Float64),
    )


def aroondown(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """阿隆下降线"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[0]),
        return_dtype=pl.Float64,
    )


def aroonup(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """阿隆上升线"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[1]),
        return_dtype=pl.Float64,
    )


def aroonosc(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """阿隆振荡指标 (Aroon Oscillator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.AROONOSC(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def bop(expr: Union[Expr, str]) -> Expr:
    """力量指标 (Balance of Power)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.BOP(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def trix(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """三重指数平滑变动率 (1-day Rate of Change of Triple Smooth EMA)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.TRIX(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def ultosc(expr: Union[Expr, str], timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28) -> Expr:
    """终极振荡指标 (Ultimate Oscillator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ULTOSC(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)),
        return_dtype=pl.Float64,
    )


def stochf(expr: Union[Expr, str], fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Tuple[Expr, Expr]:
    """快速随机指标 — 返回 (fastk, fastd)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.STOCHF(s.to_numpy(), s.to_numpy(), s.to_numpy(), fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.STOCHF(s.to_numpy(), s.to_numpy(), s.to_numpy(), fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]), return_dtype=pl.Float64),
    )


def stochrsi(expr: Union[Expr, str], timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Tuple[Expr, Expr]:
    """随机 RSI — 返回 (fastk, fastd)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]), return_dtype=pl.Float64),
    )


def stochrsi_k(expr: Union[Expr, str], timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Expr:
    """随机 RSI %K"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]),
        return_dtype=pl.Float64,
    )


def stochrsi_d(expr: Union[Expr, str], timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Expr:
    """随机 RSI %D"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]),
        return_dtype=pl.Float64,
    )


def macdext(expr: Union[Expr, str], fastperiod: int = 12, fastmatype: int = 0, slowperiod: int = 26, slowmatype: int = 0,
            signalperiod: int = 9, signalmatype: int = 0) -> Tuple[Expr, Expr, Expr]:
    """扩展 MACD — 返回 (macd, signal, hist)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.MACDEXT(s.to_numpy(), fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.MACDEXT(s.to_numpy(), fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)[1]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.MACDEXT(s.to_numpy(), fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)[2]), return_dtype=pl.Float64),
    )


def macdfix(expr: Union[Expr, str], signalperiod: int = 9) -> Tuple[Expr, Expr, Expr]:
    """固定周期 MACD — 返回 (macd, signal, hist)"""
    import talib
    e = _ensure_expr(expr)
    return (
        e.map_batches(lambda s: pl.Series(talib.MACDFIX(s.to_numpy(), signalperiod=signalperiod)[0]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.MACDFIX(s.to_numpy(), signalperiod=signalperiod)[1]), return_dtype=pl.Float64),
        e.map_batches(lambda s: pl.Series(talib.MACDFIX(s.to_numpy(), signalperiod=signalperiod)[2]), return_dtype=pl.Float64),
    )


def atr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """平均真实范围 (Average True Range)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ATR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def natr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """归一化平均真实范围 (Normalized Average True Range)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.NATR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def trange(expr: Union[Expr, str]) -> Expr:
    """真实范围 (True Range)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.TRANGE(s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def ad(expr: Union[Expr, str]) -> Expr:
    """累积/派发线 (Accumulation/Distribution Line)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.AD(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def adosc(expr: Union[Expr, str], fastperiod: int = 3, slowperiod: int = 10) -> Expr:
    """累积/派发震荡 (Chaikin A/D Oscillator)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.ADOSC(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod)),
        return_dtype=pl.Float64,
    )


def obv(expr: Union[Expr, str]) -> Expr:
    """能量潮 (On Balance Volume)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.OBV(s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def cdl_doji(expr: Union[Expr, str]) -> Expr:
    """十字星 (Doji)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLDOJI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_hammer(expr: Union[Expr, str]) -> Expr:
    """锤子线 (Hammer)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLHAMMER(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_engulfing(expr: Union[Expr, str]) -> Expr:
    """吞没形态 (Engulfing Pattern)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLENGULFING(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_morningstar(expr: Union[Expr, str], penetration: float = 0.0) -> Expr:
    """晨星 (Morning Star)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLMORNINGSTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
        return_dtype=pl.Int32,
    )


def cdl_eveningstar(expr: Union[Expr, str], penetration: float = 0.0) -> Expr:
    """暮星 (Evening Star)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLEVENINGSTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
        return_dtype=pl.Int32,
    )


def cdl_hangingman(expr: Union[Expr, str]) -> Expr:
    """上吊线 (Hanging Man)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLHANGINGMAN(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_shootingstar(expr: Union[Expr, str]) -> Expr:
    """射击之星 (Shooting Star)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLSHOOTINGSTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_harami(expr: Union[Expr, str]) -> Expr:
    """孕线形态 (Harami Pattern)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLHARAMI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_piercing(expr: Union[Expr, str]) -> Expr:
    """刺透形态 (Piercing Pattern)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLPIERCING(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_darkcloudcover(expr: Union[Expr, str], penetration: float = 0.0) -> Expr:
    """乌云盖顶 (Dark Cloud Cover)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLDARKCLOUDCOVER(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
        return_dtype=pl.Int32,
    )


def cdl_spinningtop(expr: Union[Expr, str]) -> Expr:
    """纺锤顶 (Spinning Top)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDLSPINNINGTOP(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_3whitesoldiers(expr: Union[Expr, str]) -> Expr:
    """三只白兵 (Three White Soldiers)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDL3WHITESOLDIERS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def cdl_3blackcrows(expr: Union[Expr, str]) -> Expr:
    """三只乌鸦 (Three Black Crows)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CDL3BLACKCROWS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Int32,
    )


def avgprice(expr: Union[Expr, str]) -> Expr:
    """平均价格 (Average Price)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.AVGPRICE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def medprice(expr: Union[Expr, str]) -> Expr:
    """中间价格 (Median Price)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MEDPRICE(s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def typprice(expr: Union[Expr, str]) -> Expr:
    """典型价格 (Typical Price)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.TYPPRICE(s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def wclprice(expr: Union[Expr, str]) -> Expr:
    """加权收盘价 (Weighted Close Price)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.WCLPRICE(s.to_numpy(), s.to_numpy(), s.to_numpy())),
        return_dtype=pl.Float64,
    )


def beta(expr: Union[Expr, str], timeperiod: int = 5) -> Expr:
    """Beta 系数"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.BETA(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def correl(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """相关系数 (Pearson's Correlation Coefficient)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.CORREL(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def linearreg(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """线性回归值 (Linear Regression)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.LINEARREG(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def linearreg_angle(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """线性回归角度 (Linear Regression Angle)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.LINEARREG_ANGLE(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def linearreg_intercept(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """线性回归截距 (Linear Regression Intercept)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.LINEARREG_INTERCEPT(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def linearreg_slope(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """线性回归斜率 (Linear Regression Slope)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.LINEARREG_SLOPE(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def stddev(expr: Union[Expr, str], timeperiod: int = 5, nbdev: int = 1) -> Expr:
    """标准差 (Standard Deviation)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.STDDEV(s.to_numpy(), timeperiod=timeperiod, nbdev=nbdev)),
        return_dtype=pl.Float64,
    )


def tsf(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
    """时间序列预测 (Time Series Forecast)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.TSF(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def var(expr: Union[Expr, str], timeperiod: int = 5, nbdev: int = 1) -> Expr:
    """方差 (Variance)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.VAR(s.to_numpy(), timeperiod=timeperiod, nbdev=nbdev)),
        return_dtype=pl.Float64,
    )


def ht_dcperiod(expr: Union[Expr, str]) -> Expr:
    """希尔伯特变换主导周期 (Hilbert Transform Dominant Cycle Period)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.HT_DCPERIOD(s.to_numpy())),
        return_dtype=pl.Float64,
    )


def ht_dcphase(expr: Union[Expr, str]) -> Expr:
    """希尔伯特变换主导周期相位 (Hilbert Transform Dominant Cycle Phase)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.HT_DCPHASE(s.to_numpy())),
        return_dtype=pl.Float64,
    )


def ht_phasor(expr: Union[Expr, str]) -> Expr:
    """希尔伯特变换相位 (Hilbert Transform Phasor)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.HT_PHASOR(s.to_numpy())[0]),
        return_dtype=pl.Float64,
    )


def ht_sine(expr: Union[Expr, str]) -> Expr:
    """希尔伯特变换正弦 (Hilbert Transform SineWave)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.HT_SINE(s.to_numpy())[0]),
        return_dtype=pl.Float64,
    )


def ht_trendmode(expr: Union[Expr, str]) -> Expr:
    """希尔伯特变换趋势模式 (Hilbert Transform Trend vs Cycle Mode)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.HT_TRENDMODE(s.to_numpy())),
        return_dtype=pl.Float64,
    )


def acos(expr: Union[Expr, str]) -> Expr:
    """反余弦 (Inverse Cosine)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.ACOS(s.to_numpy())), return_dtype=pl.Float64)


def asin(expr: Union[Expr, str]) -> Expr:
    """反正弦 (Inverse Sine)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.ASIN(s.to_numpy())), return_dtype=pl.Float64)


def atan(expr: Union[Expr, str]) -> Expr:
    """反正切 (Inverse Tangent)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.ATAN(s.to_numpy())), return_dtype=pl.Float64)


def cos(expr: Union[Expr, str]) -> Expr:
    """余弦 (Cosine)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.COS(s.to_numpy())), return_dtype=pl.Float64)


def cosh(expr: Union[Expr, str]) -> Expr:
    """双曲余弦 (Hyperbolic Cosine)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.COSH(s.to_numpy())), return_dtype=pl.Float64)


def sin(expr: Union[Expr, str]) -> Expr:
    """正弦 (Sine)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.SIN(s.to_numpy())), return_dtype=pl.Float64)


def sinh(expr: Union[Expr, str]) -> Expr:
    """双曲正弦 (Hyperbolic Sine)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.SINH(s.to_numpy())), return_dtype=pl.Float64)


def sqrt(expr: Union[Expr, str]) -> Expr:
    """平方根 (Square Root)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.SQRT(s.to_numpy())), return_dtype=pl.Float64)


def tan(expr: Union[Expr, str]) -> Expr:
    """正切 (Tangent)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.TAN(s.to_numpy())), return_dtype=pl.Float64)


def tanh(expr: Union[Expr, str]) -> Expr:
    """双曲正切 (Hyperbolic Tangent)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.TANH(s.to_numpy())), return_dtype=pl.Float64)


def exp(expr: Union[Expr, str]) -> Expr:
    """指数函数 (Exponential)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.EXP(s.to_numpy())), return_dtype=pl.Float64)


def ln(expr: Union[Expr, str]) -> Expr:
    """自然对数 (Natural Log)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.LN(s.to_numpy())), return_dtype=pl.Float64)


def log10(expr: Union[Expr, str]) -> Expr:
    """常用对数 (Log10)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.LOG10(s.to_numpy())), return_dtype=pl.Float64)


def ceil(expr: Union[Expr, str]) -> Expr:
    """向上取整 (Ceiling)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.CEIL(s.to_numpy())), return_dtype=pl.Float64)


def floor(expr: Union[Expr, str]) -> Expr:
    """向下取整 (Floor)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(lambda s: pl.Series(talib.FLOOR(s.to_numpy())), return_dtype=pl.Float64)


def add(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
    """加法 (Add)"""
    import talib
    e_a = _ensure_expr(expr_a)
    e_b = _ensure_expr(expr_b)
    return e_a.map_batches(lambda s: pl.Series(talib.ADD(s.to_numpy(), e_b.eval(pl.col(s.name))[0].to_numpy()))) 


def sub(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
    """减法 (Subtract)"""
    return _ensure_expr(expr_a) - _ensure_expr(expr_b)


def mult(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
    """乘法 (Multiply)"""
    return _ensure_expr(expr_a) * _ensure_expr(expr_b)


def div(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
    """除法 (Divide)"""
    return _ensure_expr(expr_a) / _ensure_expr(expr_b)


def max(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """滚动最大值 (Rolling Maximum)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MAX(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def min(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """滚动最小值 (Rolling Minimum)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.MIN(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def sum(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
    """滚动求和 (Rolling Sum)"""
    import talib
    e = _ensure_expr(expr)
    return e.map_batches(
        lambda s: pl.Series(talib.SUM(s.to_numpy(), timeperiod=timeperiod)),
        return_dtype=pl.Float64,
    )


def _wrap_and_register(name: str, func, param_docs: str = ""):
    """包装函数并注册到算子注册表"""
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper.__name__ = f"talib_{name}"
    wrapper.__doc__ = f"TA-Lib {name}\n\n    {param_docs}\n    "
    register_operator(OperatorCategory.TALIB, f"talib_{name}")(wrapper)
    globals()[f"talib_{name}"] = wrapper
    return wrapper


_wrap_and_register("sma", sma, "简单移动平均 (Simple Moving Average)")
_wrap_and_register("ema", ema, "指数移动平均 (Exponential Moving Average)")
_wrap_and_register("wma", wma, "加权移动平均 (Weighted Moving Average)")
_wrap_and_register("dema", dema, "双重指数移动平均 (Double EMA)")
_wrap_and_register("tema", tema, "三重指数移动平均 (Triple EMA)")
_wrap_and_register("trima", trima, "三角移动平均 (Triangular MA)")
_wrap_and_register("kama", kama, "考夫曼自适应移动平均 (KAMA)")
_wrap_and_register("t3", t3, "T3 移动平均")
_wrap_and_register("ht_trendline", ht_trendline, "希尔伯特变换趋势线")
_wrap_and_register("bbands_upper", bbands_upper, "布林带上轨")
_wrap_and_register("bbands_middle", bbands_middle, "布林带中轨")
_wrap_and_register("bbands_lower", bbands_lower, "布林带下轨")
_wrap_and_register("ma", ma, "通用移动平均")

_wrap_and_register("rsi", rsi, "相对强弱指标 (RSI)")
_wrap_and_register("macd_line", macd_line, "MACD 线")
_wrap_and_register("macd_signal", macd_signal, "MACD 信号线")
_wrap_and_register("macd_hist", macd_hist, "MACD 柱状图")
_wrap_and_register("stoch_k", stoch_k, "随机指标 K")
_wrap_and_register("stoch_d", stoch_d, "随机指标 D")
_wrap_and_register("willr", willr, "威廉指标")
_wrap_and_register("cci", cci, "商品通道指标")
_wrap_and_register("adx", adx, "平均趋向指数")
_wrap_and_register("adxr", adxr, "平均趋向指数评级")
_wrap_and_register("mfi", mfi, "资金流量指标")
_wrap_and_register("mom", mom, "动量指标")
_wrap_and_register("roc", roc, "变动率指标")
_wrap_and_register("rocp", rocp, "变动率百分比")
_wrap_and_register("trix", trix, "三重指数移动平均线")
_wrap_and_register("ultosc", ultosc, "终极波动指标")
_wrap_and_register("dx", dx, "方向指标")
_wrap_and_register("minus_di", minus_di, "负方向指标")
_wrap_and_register("plus_di", plus_di, "正方向指标")
_wrap_and_register("ppo", ppo, "价格振荡器")

_wrap_and_register("atr", atr, "平均真实范围")
_wrap_and_register("natr", natr, "标准化平均真实范围")
_wrap_and_register("trange", trange, "真实范围")

_wrap_and_register("ad", ad, "累积/派发线")
_wrap_and_register("adosc", adosc, "蔡金震荡器")
_wrap_and_register("obv", obv, "能量潮")

_wrap_and_register("avgprice", avgprice, "平均价格")
_wrap_and_register("medprice", medprice, "中间价格")
_wrap_and_register("typprice", typprice, "典型价格")
_wrap_and_register("wclprice", wclprice, "加权收盘价")

_wrap_and_register("beta", beta, "贝塔系数")
_wrap_and_register("correl", correl, "相关系数")
_wrap_and_register("linearreg", linearreg, "线性回归")
_wrap_and_register("linearreg_angle", linearreg_angle, "线性回归角度")
_wrap_and_register("linearreg_intercept", linearreg_intercept, "线性回归截距")
_wrap_and_register("linearreg_slope", linearreg_slope, "线性回归斜率")
_wrap_and_register("stddev", stddev, "标准差")
_wrap_and_register("tsf", tsf, "时间序列预测")
_wrap_and_register("var", var, "方差")

_wrap_and_register("ht_dcperiod", ht_dcperiod, "希尔伯特变换主导周期")
_wrap_and_register("ht_dcphase", ht_dcphase, "希尔伯特变换主导周期相位")
_wrap_and_register("ht_phasor", ht_phasor, "希尔伯特变换相位")
_wrap_and_register("ht_sine", ht_sine, "希尔伯特变换正弦")
_wrap_and_register("ht_trendmode", ht_trendmode, "希尔伯特变换趋势模式")

_wrap_and_register("acos", acos, "反余弦")
_wrap_and_register("asin", asin, "反正弦")
_wrap_and_register("atan", atan, "反正切")
_wrap_and_register("cos", cos, "余弦")
_wrap_and_register("cosh", cosh, "双曲余弦")
_wrap_and_register("sin", sin, "正弦")
_wrap_and_register("sinh", sinh, "双曲正弦")
_wrap_and_register("sqrt", sqrt, "平方根")
_wrap_and_register("tan", tan, "正切")
_wrap_and_register("tanh", tanh, "双曲正切")
_wrap_and_register("exp", exp, "指数函数")
_wrap_and_register("ln", ln, "自然对数")
_wrap_and_register("log10", log10, "常用对数")
_wrap_and_register("ceil", ceil, "向上取整")
_wrap_and_register("floor", floor, "向下取整")

_wrap_and_register("add", add, "加法")
_wrap_and_register("sub", sub, "减法")
_wrap_and_register("mult", mult, "乘法")
_wrap_and_register("div", div, "除法")

_wrap_and_register("max", max, "滚动最大值")
_wrap_and_register("min", min, "滚动最小值")
_wrap_and_register("sum", sum, "滚动求和")

_wrap_and_register("macd", macd, "MACD (完整)")
_wrap_and_register("stoch", stoch, "随机指标 (完整)")
_wrap_and_register("minus_dm", minus_dm, "负向动量")
_wrap_and_register("plus_dm", plus_dm, "正向动量")
_wrap_and_register("aroon", aroon, "阿隆指标")
_wrap_and_register("aroondown", aroondown, "阿隆下降线")
_wrap_and_register("aroonup", aroonup, "阿隆上升线")
_wrap_and_register("aroonosc", aroonosc, "阿隆振荡指标")
_wrap_and_register("bop", bop, "力量指标")
_wrap_and_register("stochf", stochf, "快速随机指标")
_wrap_and_register("stochrsi", stochrsi, "随机 RSI (完整)")
_wrap_and_register("stochrsi_k", stochrsi_k, "随机 RSI %K")
_wrap_and_register("stochrsi_d", stochrsi_d, "随机 RSI %D")
_wrap_and_register("macdext", macdext, "扩展 MACD")
_wrap_and_register("macdfix", macdfix, "固定周期 MACD")
_wrap_and_register("cmo", cmo, "钱德动量摆动指标")
_wrap_and_register("apo", apo, "绝对价格震荡")

_wrap_and_register("cdl_doji", cdl_doji, "十字星")
_wrap_and_register("cdl_hammer", cdl_hammer, "锤子线")
_wrap_and_register("cdl_engulfing", cdl_engulfing, "吞没形态")
_wrap_and_register("cdl_morningstar", cdl_morningstar, "晨星")
_wrap_and_register("cdl_eveningstar", cdl_eveningstar, "暮星")
_wrap_and_register("cdl_hangingman", cdl_hangingman, "上吊线")
_wrap_and_register("cdl_shootingstar", cdl_shootingstar, "射击之星")
_wrap_and_register("cdl_harami", cdl_harami, "孕线形态")
_wrap_and_register("cdl_piercing", cdl_piercing, "刺透形态")
_wrap_and_register("cdl_darkcloudcover", cdl_darkcloudcover, "乌云盖顶")
_wrap_and_register("cdl_spinningtop", cdl_spinningtop, "纺锤顶")
_wrap_and_register("cdl_3whitesoldiers", cdl_3whitesoldiers, "三只白兵")
_wrap_and_register("cdl_3blackcrows", cdl_3blackcrows, "三只乌鸦")