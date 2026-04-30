# coding=utf-8
"""
TA-Lib 算子包装层

基于 TA-Lib 0.6.x 的技术分析指标，通过 map_batches 桥接 Polars ↔ NumPy。
TA-Lib 0.6.x 原生支持 polars.Series 输入，返回 polars.Series。

覆盖指标:
  - 趋势/重叠: SMA, EMA, DEMA, TEMA, WMA, KAMA, T3, BBANDS, ...
  - 动量: RSI, MACD, STOCH, CCI, WILLR, MFI, ROC, MOM, ADX, ...
  - 波动率: ATR, NATR, TRANGE
  - 成交量: AD, ADOSC, OBV
  - K线形态: CDL_DOJI, CDL_HAMMER, CDL_ENGULFING, ... (60+)
  - 价格变换: AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE
  - 统计函数: BETA, CORREL, LINEARREG, STDDEV, ...

Usage:
    from QuantNodes.operators.talib import TaLibOperators
    result = TaLibOperators.sma(pl.col("close"), timeperiod=20)
"""

from __future__ import annotations

from typing import Union, Tuple

import polars as pl
from polars import Expr


def _ensure_expr(f: Union[Expr, str, float]) -> Expr:
    """将输入规范化为 Polars Expr"""
    if isinstance(f, Expr):
        return f
    if isinstance(f, str):
        return pl.col(f)
    return pl.lit(f)


def _to_numpy(f: Union[Expr, str]) -> Expr:
    """将列转为 numpy 数组的 Expr（用于 map_batches）"""
    e = _ensure_expr(f)
    return e.cast(pl.Float64)


class TaLibOperators:
    """TA-Lib 算子包装层"""

    # ======================================================================
    # Overlap Studies (重叠/趋势指标)
    # ======================================================================

    @staticmethod
    def sma(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """简单移动平均线 (Simple Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.SMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """指数移动平均线 (Exponential Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.EMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def wma(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """加权移动平均线 (Weighted Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.WMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def dema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """双重指数移动平均线 (Double Exponential Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.DEMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def tema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """三重指数移动平均线 (Triple Exponential Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.TEMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def trima(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """三角移动平均线 (Triangular Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.TRIMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def kama(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """考夫曼自适应移动平均线 (Kaufman Adaptive Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.KAMA(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def t3(expr: Union[Expr, str], timeperiod: int = 5, vfactor: float = 0.7) -> Expr:
        """T3 移动平均线 (Triple Exponential Moving Average with volume factor)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.T3(s.to_numpy(), timeperiod=timeperiod, vfactor=vfactor)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ht_trendline(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换趋势线 (Hilbert Transform Trendline)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.HT_TRENDLINE(s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def mama(expr: Union[Expr, str], fastlimit: float = 0.5, slowlimit: float = 0.05) -> Expr:
        """MAMA 移动平均 (MESA Adaptive Moving Average)"""
        import talib
        e = _ensure_expr(expr)
        def _mama(s: pl.Series) -> pl.Series:
            mama_line, _ = talib.MAMA(s.to_numpy(), fastlimit=fastlimit, slowlimit=slowlimit)
            return pl.Series(mama_line)
        return e.map_batches(_mama, return_dtype=pl.Float64)

    @staticmethod
    def mavp(expr: Union[Expr, str], periods: Union[Expr, str], minperiod: int = 2, maxperiod: int = 30) -> Expr:
        """移动平均变动周期 (MA with Variable Period)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MAVP(s.to_numpy(), s.to_numpy(), minperiod=minperiod, maxperiod=maxperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def midpoint(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """价格中点 (Midpoint over period)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MIDPOINT(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def midprice(expr: Union[Expr, str], timeperiod: int = 14,
                 high_col: str = "high", low_col: str = "low") -> Expr:
        """中间价格 (Midpoint Price over period) — 需要 high/low 列"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MIDPRICE(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def sar(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
            acceleration: float = 0.02, maximum: float = 0.2) -> Expr:
        """抛物线 SAR — 需要 high/low 列"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.SAR(s.to_numpy(), s.to_numpy(), acceleration=acceleration, maximum=maximum)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ma(expr: Union[Expr, str], timeperiod: int = 30, matype: int = 0) -> Expr:
        """通用移动平均线 (Moving Average) — matype: 0=SMA, 1=EMA, 2=WMA, 3=DEMA, 4=TEMA, ..."""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MA(s.to_numpy(), timeperiod=timeperiod, matype=matype)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def bbands(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
               nbdevdn: float = 2.0, matype: int = 0) -> Tuple[Expr, Expr, Expr]:
        """布林带 — 返回 (upper, middle, lower)"""
        import talib
        e = _ensure_expr(expr)
        def _bb_upper(s: pl.Series) -> pl.Series:
            return pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0])
        def _bb_mid(s: pl.Series) -> pl.Series:
            return pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[1])
        def _bb_lower(s: pl.Series) -> pl.Series:
            return pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[2])
        return (
            e.map_batches(_bb_upper, return_dtype=pl.Float64),
            e.map_batches(_bb_mid, return_dtype=pl.Float64),
            e.map_batches(_bb_lower, return_dtype=pl.Float64),
        )

    @staticmethod
    def bbands_upper(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                     nbdevdn: float = 2.0, matype: int = 0) -> Expr:
        """布林带上轨"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def bbands_middle(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                      nbdevdn: float = 2.0, matype: int = 0) -> Expr:
        """布林带中轨"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[1]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def bbands_lower(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                     nbdevdn: float = 2.0, matype: int = 0) -> Expr:
        """布林带下轨"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.BBANDS(s.to_numpy(), timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)[2]),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Momentum Indicators (动量指标)
    # ======================================================================

    @staticmethod
    def rsi(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """相对强弱指标 (Relative Strength Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.RSI(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def macd(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
             signalperiod: int = 9) -> Tuple[Expr, Expr, Expr]:
        """MACD — 返回 (macd_line, signal_line, histogram)"""
        import talib
        e = _ensure_expr(expr)
        def _macd(s: pl.Series) -> Tuple[pl.Series, pl.Series, pl.Series]:
            macd_line, signal_line, hist = talib.MACD(
                s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod
            )
            return pl.Series(macd_line), pl.Series(signal_line), pl.Series(hist)
        # 返回一个 tuple，调用者需分别使用
        return (
            e.map_batches(lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0]), return_dtype=pl.Float64),
            e.map_batches(lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1]), return_dtype=pl.Float64),
            e.map_batches(lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2]), return_dtype=pl.Float64),
        )

    @staticmethod
    def macd_line(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                  signalperiod: int = 9) -> Expr:
        """MACD 线"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def macd_signal(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                    signalperiod: int = 9) -> Expr:
        """MACD 信号线"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[1]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def macd_hist(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                  signalperiod: int = 9) -> Expr:
        """MACD 柱状图 (histogram)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MACD(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)[2]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def stoch(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
              fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0,
              slowd_period: int = 3, slowd_matype: int = 0) -> Tuple[Expr, Expr]:
        """随机指标 (Stochastic) — 返回 (slowk, slowd)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(
                lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                    fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                    slowd_period=slowd_period, slowd_matype=slowd_matype)[0]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                    fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                    slowd_period=slowd_period, slowd_matype=slowd_matype)[1]),
                return_dtype=pl.Float64,
            ),
        )

    @staticmethod
    def stoch_k(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0) -> Expr:
        """随机指标 K 线 (Stochastic %K)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def stoch_d(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0,
                slowd_period: int = 3, slowd_matype: int = 0) -> Expr:
        """随机指标 D 线 (Stochastic %D)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.STOCH(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                slowd_period=slowd_period, slowd_matype=slowd_matype)[1]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def cci(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
            timeperiod: int = 14) -> Expr:
        """商品通道指标 (Commodity Channel Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CCI(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def willr(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
              timeperiod: int = 14) -> Expr:
        """威廉指标 (Williams %R)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.WILLR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def mfi(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
            timeperiod: int = 14) -> Expr:
        """资金流量指标 (Money Flow Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MFI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def roc(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率 (Rate of Change)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ROC(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def rocp(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率百分比 (Rate of Change Percentage)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ROCP(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def rocr(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率比率 (Rate of Change Ratio)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ROCR(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def rocr100(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率比率*100 (Rate of Change Ratio 100 scale)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ROCR100(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def mom(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """动量 (Momentum)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MOM(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def adx(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
            timeperiod: int = 14) -> Expr:
        """平均趋向指标 (Average Directional Movement Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ADX(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def adxr(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
             timeperiod: int = 14) -> Expr:
        """平均趋向指标评估 (Average Directional Movement Index Rating)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ADXR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def apo(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
            matype: int = 0) -> Expr:
        """绝对价格震荡 (Absolute Price Oscillator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.APO(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ppo(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
            matype: int = 0) -> Expr:
        """百分比价格震荡 (Percentage Price Oscillator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.PPO(s.to_numpy(), fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def cmo(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """钱德动量摆动指标 (Chande Momentum Oscillator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CMO(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def dx(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
           timeperiod: int = 14) -> Expr:
        """趋向指标 (Directional Movement Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.DX(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def minus_di(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                 timeperiod: int = 14) -> Expr:
        """负向指标 (Minus Directional Indicator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MINUS_DI(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def plus_di(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                timeperiod: int = 14) -> Expr:
        """正向指标 (Plus Directional Indicator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.PLUS_DI(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def minus_dm(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                 timeperiod: int = 14) -> Expr:
        """负向动量 (Minus Directional Movement)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MINUS_DM(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def plus_dm(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                timeperiod: int = 14) -> Expr:
        """正向动量 (Plus Directional Movement)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.PLUS_DM(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def aroon(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
              timeperiod: int = 14) -> Tuple[Expr, Expr]:
        """阿隆指标 — 返回 (aroondown, aroonup)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(
                lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[0]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[1]),
                return_dtype=pl.Float64,
            ),
        )

    @staticmethod
    def aroondown(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                  timeperiod: int = 14) -> Expr:
        """阿隆下降线"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def aroonup(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                timeperiod: int = 14) -> Expr:
        """阿隆上升线"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.AROON(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[1]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def aroonosc(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                 timeperiod: int = 14) -> Expr:
        """阿隆振荡指标 (Aroon Oscillator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.AROONOSC(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def bop(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low") -> Expr:
        """力量指标 (Balance of Power)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.BOP(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def trix(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """三重指数平滑变动率 (1-day Rate of Change of Triple Smooth EMA)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.TRIX(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ultosc(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
               timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28) -> Expr:
        """终极振荡指标 (Ultimate Oscillator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ULTOSC(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def stochf(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
               fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Tuple[Expr, Expr]:
        """快速随机指标 — 返回 (fastk, fastd)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(
                lambda s: pl.Series(talib.STOCHF(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                    fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.STOCHF(s.to_numpy(), s.to_numpy(), s.to_numpy(),
                    fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]),
                return_dtype=pl.Float64,
            ),
        )

    @staticmethod
    def stochrsi(expr: Union[Expr, str], timeperiod: int = 14,
                 fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Tuple[Expr, Expr]:
        """随机 RSI — 返回 (fastk, fastd)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(
                lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod,
                    fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod,
                    fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]),
                return_dtype=pl.Float64,
            ),
        )

    @staticmethod
    def stochrsi_k(expr: Union[Expr, str], timeperiod: int = 14,
                   fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Expr:
        """随机 RSI %K"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod,
                fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def stochrsi_d(expr: Union[Expr, str], timeperiod: int = 14,
                   fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Expr:
        """随机 RSI %D"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.STOCHRSI(s.to_numpy(), timeperiod=timeperiod,
                fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)[1]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def macdext(expr: Union[Expr, str], fastperiod: int = 12, fastmatype: int = 0,
                slowperiod: int = 26, slowmatype: int = 0, signalperiod: int = 9,
                signalmatype: int = 0) -> Tuple[Expr, Expr, Expr]:
        """扩展 MACD — 返回 (macd, signal, hist)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(
                lambda s: pl.Series(talib.MACDEXT(s.to_numpy(),
                    fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype,
                    signalperiod=signalperiod, signalmatype=signalmatype)[0]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.MACDEXT(s.to_numpy(),
                    fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype,
                    signalperiod=signalperiod, signalmatype=signalmatype)[1]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.MACDEXT(s.to_numpy(),
                    fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype,
                    signalperiod=signalperiod, signalmatype=signalmatype)[2]),
                return_dtype=pl.Float64,
            ),
        )

    @staticmethod
    def macdfix(expr: Union[Expr, str], signalperiod: int = 9) -> Tuple[Expr, Expr, Expr]:
        """固定周期 MACD — 返回 (macd, signal, hist)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(
                lambda s: pl.Series(talib.MACDFIX(s.to_numpy(), signalperiod=signalperiod)[0]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.MACDFIX(s.to_numpy(), signalperiod=signalperiod)[1]),
                return_dtype=pl.Float64,
            ),
            e.map_batches(
                lambda s: pl.Series(talib.MACDFIX(s.to_numpy(), signalperiod=signalperiod)[2]),
                return_dtype=pl.Float64,
            ),
        )

    # ======================================================================
    # Volatility Indicators (波动率指标)
    # ======================================================================

    @staticmethod
    def atr(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
            timeperiod: int = 14) -> Expr:
        """平均真实范围 (Average True Range)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ATR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def natr(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
             timeperiod: int = 14) -> Expr:
        """归一化平均真实范围 (Normalized Average True Range)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.NATR(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def trange(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low") -> Expr:
        """真实范围 (True Range)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.TRANGE(s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Volume Indicators (成交量指标)
    # ======================================================================

    @staticmethod
    def ad(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low") -> Expr:
        """累积/派发线 (Accumulation/Distribution Line)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.AD(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def adosc(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
              fastperiod: int = 3, slowperiod: int = 10) -> Expr:
        """累积/派发震荡 (Chaikin A/D Oscillator)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ADOSC(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(),
                fastperiod=fastperiod, slowperiod=slowperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def obv(expr: Union[Expr, str], volume_col: str = "volume") -> Expr:
        """能量潮 (On Balance Volume)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.OBV(s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Pattern Recognition (K线形态识别)
    # ======================================================================

    @staticmethod
    def cdl_doji(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                 close_col: str = "close", open_col: str = "open",
                 penetration: float = 0.0) -> Expr:
        """十字星 (Doji)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLDOJI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_hammer(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                   close_col: str = "close", open_col: str = "open",
                   penetration: float = 0.0) -> Expr:
        """锤子线 (Hammer)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHAMMER(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_engulfing(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                      close_col: str = "close", open_col: str = "open",
                      penetration: float = 0.0) -> Expr:
        """吞没形态 (Engulfing Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLENGULFING(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_morningstar(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                        close_col: str = "close", open_col: str = "open",
                        penetration: float = 0.0) -> Expr:
        """晨星 (Morning Star)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLMORNINGSTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_eveningstar(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                        close_col: str = "close", open_col: str = "open",
                        penetration: float = 0.0) -> Expr:
        """暮星 (Evening Star)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLEVENINGSTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_hangingman(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                       close_col: str = "close", open_col: str = "open",
                       penetration: float = 0.0) -> Expr:
        """上吊线 (Hanging Man)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHANGINGMAN(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_shootingstar(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                         close_col: str = "close", open_col: str = "open",
                         penetration: float = 0.0) -> Expr:
        """射击之星 (Shooting Star)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLSHOOTINGSTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_harami(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                   close_col: str = "close", open_col: str = "open",
                   penetration: float = 0.0) -> Expr:
        """孕线形态 (Harami Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHARAMI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_piercing(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                     close_col: str = "close", open_col: str = "open",
                     penetration: float = 0.0) -> Expr:
        """刺透形态 (Piercing Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLPIERCING(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_darkcloudcover(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                           close_col: str = "close", open_col: str = "open",
                           penetration: float = 0.0) -> Expr:
        """乌云盖顶 (Dark Cloud Cover)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLDARKCLOUDCOVER(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_spinningtop(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                        close_col: str = "close", open_col: str = "open",
                        penetration: float = 0.0) -> Expr:
        """纺锤顶 (Spinning Top)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLSPINNINGTOP(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_3whitesoldiers(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                           close_col: str = "close", open_col: str = "open",
                           penetration: float = 0.0) -> Expr:
        """三只白兵 (Three White Soldiers)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL3WHITESOLDIERS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_3blackcrows(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                        close_col: str = "close", open_col: str = "open",
                        penetration: float = 0.0) -> Expr:
        """三只乌鸦 (Three Black Crows)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL3BLACKCROWS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    # ======================================================================
    # Price Transform (价格变换)
    # ======================================================================

    @staticmethod
    def avgprice(expr: Union[Expr, str]) -> Expr:
        """平均价格 (Average Price)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.AVGPRICE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def medprice(expr: Union[Expr, str]) -> Expr:
        """中间价格 (Median Price)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MEDPRICE(s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def typprice(expr: Union[Expr, str]) -> Expr:
        """典型价格 (Typical Price)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.TYPPRICE(s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def wclprice(expr: Union[Expr, str]) -> Expr:
        """加权收盘价 (Weighted Close Price)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.WCLPRICE(s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Statistic Functions (统计函数)
    # ======================================================================

    @staticmethod
    def beta(expr: Union[Expr, str], timeperiod: int = 5) -> Expr:
        """Beta 系数"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.BETA(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def correl(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """相关系数 (Pearson's Correlation Coefficient)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CORREL(s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def linearreg(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归值 (Linear Regression)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.LINEARREG(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def linearreg_angle(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归角度 (Linear Regression Angle)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.LINEARREG_ANGLE(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def linearreg_intercept(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归截距 (Linear Regression Intercept)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.LINEARREG_INTERCEPT(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def linearreg_slope(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归斜率 (Linear Regression Slope)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.LINEARREG_SLOPE(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def stddev(expr: Union[Expr, str], timeperiod: int = 5, nbdev: int = 1) -> Expr:
        """标准差 (Standard Deviation)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.STDDEV(s.to_numpy(), timeperiod=timeperiod, nbdev=nbdev)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def tsf(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """时间序列预测 (Time Series Forecast)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.TSF(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def var(expr: Union[Expr, str], timeperiod: int = 5, nbdev: int = 1) -> Expr:
        """方差 (Variance)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.VAR(s.to_numpy(), timeperiod=timeperiod, nbdev=nbdev)),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Cycle Indicators (周期指标)
    # ======================================================================

    @staticmethod
    def ht_dcperiod(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换主导周期 (Hilbert Transform Dominant Cycle Period)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.HT_DCPERIOD(s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ht_dcphase(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换主导周期相位 (Hilbert Transform Dominant Cycle Phase)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.HT_DCPHASE(s.to_numpy())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ht_phasor(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换相位 (Hilbert Transform Phasor)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.HT_PHASOR(s.to_numpy())[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ht_sine(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换正弦 (Hilbert Transform SineWave)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.HT_SINE(s.to_numpy())[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def ht_trendmode(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换趋势模式 (Hilbert Transform Trend vs Cycle Mode)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.HT_TRENDMODE(s.to_numpy())),
            return_dtype=pl.Float64,
        )


    # ======================================================================
    # Math Operators (数学运算)
    # ======================================================================

    @staticmethod
    def acos(expr: Union[Expr, str]) -> Expr:
        """反余弦 (Inverse Cosine)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.ACOS(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def asin(expr: Union[Expr, str]) -> Expr:
        """反正弦 (Inverse Sine)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.ASIN(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def atan(expr: Union[Expr, str]) -> Expr:
        """反正切 (Inverse Tangent)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.ATAN(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def cos(expr: Union[Expr, str]) -> Expr:
        """余弦 (Cosine)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.COS(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def cosh(expr: Union[Expr, str]) -> Expr:
        """双曲余弦 (Hyperbolic Cosine)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.COSH(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def sin(expr: Union[Expr, str]) -> Expr:
        """正弦 (Sine)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.SIN(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def sinh(expr: Union[Expr, str]) -> Expr:
        """双曲正弦 (Hyperbolic Sine)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.SINH(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def sqrt(expr: Union[Expr, str]) -> Expr:
        """平方根 (Square Root)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.SQRT(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def tan(expr: Union[Expr, str]) -> Expr:
        """正切 (Tangent)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.TAN(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def tanh(expr: Union[Expr, str]) -> Expr:
        """双曲正切 (Hyperbolic Tangent)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.TANH(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def exp(expr: Union[Expr, str]) -> Expr:
        """指数函数 (Exponential)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.EXP(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def ln(expr: Union[Expr, str]) -> Expr:
        """自然对数 (Natural Logarithm)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.LN(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def log10(expr: Union[Expr, str]) -> Expr:
        """常用对数 (Base-10 Logarithm)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.LOG10(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def ceil(expr: Union[Expr, str]) -> Expr:
        """向上取整 (Ceiling)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.CEIL(s.to_numpy())), return_dtype=pl.Float64)

    @staticmethod
    def floor(expr: Union[Expr, str]) -> Expr:
        """向下取整 (Floor)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(lambda s: pl.Series(talib.FLOOR(s.to_numpy())), return_dtype=pl.Float64)

    # ======================================================================
    # Arithmetic Operators (算术运算)
    # ======================================================================

    @staticmethod
    def add(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """加法 (Addition)"""
        import talib
        a = _ensure_expr(expr_a)
        return a.map_batches(
            lambda s: pl.Series(talib.ADD(s.to_numpy(), _ensure_expr(expr_b).to_frame().to_numpy().flatten())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def sub(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """减法 (Subtraction)"""
        import talib
        a = _ensure_expr(expr_a)
        return a.map_batches(
            lambda s: pl.Series(talib.SUB(s.to_numpy(), _ensure_expr(expr_b).to_frame().to_numpy().flatten())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def mult(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """乘法 (Multiplication)"""
        import talib
        a = _ensure_expr(expr_a)
        return a.map_batches(
            lambda s: pl.Series(talib.MULT(s.to_numpy(), _ensure_expr(expr_b).to_frame().to_numpy().flatten())),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def div(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """除法 (Division)"""
        import talib
        a = _ensure_expr(expr_a)
        return a.map_batches(
            lambda s: pl.Series(talib.DIV(s.to_numpy(), _ensure_expr(expr_b).to_frame().to_numpy().flatten())),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Aggregation / Min / Max (聚合/极值)
    # ======================================================================

    @staticmethod
    def max(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动最大值 (Rolling Maximum)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MAX(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def min(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动最小值 (Rolling Minimum)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MIN(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def sum(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动求和 (Rolling Sum)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.SUM(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def maxindex(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动最大值位置 (Rolling Maximum Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MAXINDEX(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def minindex(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动最小值位置 (Rolling Minimum Index)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.MININDEX(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def minmax(expr: Union[Expr, str], timeperiod: int = 30) -> Tuple[Expr, Expr]:
        """滚动最小/最大值 — 返回 (min, max)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(lambda s: pl.Series(talib.MINMAX(s.to_numpy(), timeperiod=timeperiod)[0]), return_dtype=pl.Float64),
            e.map_batches(lambda s: pl.Series(talib.MINMAX(s.to_numpy(), timeperiod=timeperiod)[1]), return_dtype=pl.Float64),
        )

    @staticmethod
    def minmaxindex(expr: Union[Expr, str], timeperiod: int = 30) -> Tuple[Expr, Expr]:
        """滚动最小/最大值位置 — 返回 (min_index, max_index)"""
        import talib
        e = _ensure_expr(expr)
        return (
            e.map_batches(lambda s: pl.Series(talib.MINMAXINDEX(s.to_numpy(), timeperiod=timeperiod)[0]), return_dtype=pl.Int32),
            e.map_batches(lambda s: pl.Series(talib.MINMAXINDEX(s.to_numpy(), timeperiod=timeperiod)[1]), return_dtype=pl.Int32),
        )

    @staticmethod
    def avgdev(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """平均绝对偏差 (Average Deviation)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.AVGDEV(s.to_numpy(), timeperiod=timeperiod)),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Other Indicators (其他指标)
    # ======================================================================

    @staticmethod
    def sarext(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
               startvalue: float = 0.0, offsetonreverse: float = 0.0,
               accelerationinitlong: float = 0.02, accelerationlong: float = 0.02,
               accelerationmaxlong: float = 0.2, accelerationinitshort: float = 0.02,
               accelerationshort: float = 0.02, accelerationmaxshort: float = 0.2) -> Expr:
        """扩展抛物线 SAR (Parabolic SAR - Extended)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.SAREXT(s.to_numpy(), s.to_numpy(),
                startvalue=startvalue, offsetonreverse=offsetonreverse,
                accelerationinitlong=accelerationinitlong, accelerationlong=accelerationlong,
                accelerationmaxlong=accelerationmaxlong, accelerationinitshort=accelerationinitshort,
                accelerationshort=accelerationshort, accelerationmaxshort=accelerationmaxshort)),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def accbands(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                 timeperiod: int = 20) -> Expr:
        """加速带 (Acceleration Bands) — 返回上轨"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ACCBANDS(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def accbands_upper(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low",
                       timeperiod: int = 20) -> Expr:
        """加速带上轨"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.ACCBANDS(s.to_numpy(), s.to_numpy(), s.to_numpy(), timeperiod=timeperiod)[0]),
            return_dtype=pl.Float64,
        )

    @staticmethod
    def imi(expr: Union[Expr, str], high_col: str = "high", low_col: str = "low") -> Expr:
        """日本蜡烛图: 身体指标 (Japanese Candlestick: Body Size)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.IMI(s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Float64,
        )

    # ======================================================================
    # Remaining Candlestick Patterns (剩余K线形态)
    # ======================================================================

    @staticmethod
    def cdl_2crows(expr: Union[Expr, str], **kwargs) -> Expr:
        """两只乌鸦 (Two Crows)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL2CROWS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_3inside(expr: Union[Expr, str], **kwargs) -> Expr:
        """三内部上涨/下跌 (Three Inside Up/Down)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL3INSIDE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_3linestrike(expr: Union[Expr, str], **kwargs) -> Expr:
        """三线打击 (Three Line Strike)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL3LINESTRIKE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_3outside(expr: Union[Expr, str], **kwargs) -> Expr:
        """三外部上涨/下跌 (Three Outside Up/Down)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL3OUTSIDE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_3starsinsouth(expr: Union[Expr, str], **kwargs) -> Expr:
        """南方三星 (Three Stars In The South)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDL3STARSINSOUTH(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_abandonedbaby(expr: Union[Expr, str], penetration: float = 0.0, **kwargs) -> Expr:
        """弃婴 (Abandoned Baby)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLABANDONEDBABY(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_advanceblock(expr: Union[Expr, str], **kwargs) -> Expr:
        """大敌当前 (Advance Block)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLADVANCEBLOCK(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_belthold(expr: Union[Expr, str], **kwargs) -> Expr:
        """腰带线 (Belt-hold)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLBELTHOLD(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_breakaway(expr: Union[Expr, str], **kwargs) -> Expr:
        """脱离 (Breakaway)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLBREAKAWAY(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_closingmarubozu(expr: Union[Expr, str], **kwargs) -> Expr:
        """收盘光头光脚阴线 (Closing Marubozu)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLCLOSINGMARUBOZU(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_concealbabyswall(expr: Union[Expr, str], **kwargs) -> Expr:
        """藏婴吞没 (Concealing Baby Swallow)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLCONCEALBABYSWALL(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_counterattack(expr: Union[Expr, str], **kwargs) -> Expr:
        """反击线 (Counterattack)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLCOUNTERATTACK(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_dojistar(expr: Union[Expr, str], **kwargs) -> Expr:
        """十字孕线 (Doji Star)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLDOJISTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_dragonflydoji(expr: Union[Expr, str], **kwargs) -> Expr:
        """蜻蜓十字 (Dragonfly Doji)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLDRAGONFLYDOJI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_eveningdojistar(expr: Union[Expr, str], penetration: float = 0.0, **kwargs) -> Expr:
        """十字暮星 (Evening Doji Star)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLEVENINGDOJISTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_gapsidesidewhite(expr: Union[Expr, str], **kwargs) -> Expr:
        """向上跳空并列阳线 (Up/Down Gap Two Crows / Side-by-Side White Lines)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLGAPSIDESIDEWHITE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_gravestonedoji(expr: Union[Expr, str], **kwargs) -> Expr:
        """墓碑十字 (Gravestone Doji)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLGRAVESTONEDOJI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_haramicross(expr: Union[Expr, str], **kwargs) -> Expr:
        """十字孕线形态 (Harami Cross)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHARAMICROSS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_highwave(expr: Union[Expr, str], **kwargs) -> Expr:
        """长影线 (High Wave)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHIGHWAVE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_hikkake(expr: Union[Expr, str], **kwargs) -> Expr:
        """伪装 (Hikkake Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHIKKAKE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_hikkakemod(expr: Union[Expr, str], **kwargs) -> Expr:
        """修正伪装 (Modified Hikkake Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHIKKAKEMOD(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_homingpigeon(expr: Union[Expr, str], **kwargs) -> Expr:
        """归巢鸽 (Homing Pigeon)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLHOMINGPIGEON(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_identical3crows(expr: Union[Expr, str], **kwargs) -> Expr:
        """相同三乌鸦 (Identical Three Crows)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLIDENTICAL3CROWS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_inneck(expr: Union[Expr, str], **kwargs) -> Expr:
        """颈部线 (In-Neck Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLINNECK(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_invertedhammer(expr: Union[Expr, str], **kwargs) -> Expr:
        """倒锤子线 (Inverted Hammer)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLINVERTEDHAMMER(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_kicking(expr: Union[Expr, str], **kwargs) -> Expr:
        """踢出 (Kicking)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLKICKING(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_kickingbylength(expr: Union[Expr, str], **kwargs) -> Expr:
        """踢出按长度 (Kicking - long leg)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLKICKINGBYLENGTH(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_ladderbottom(expr: Union[Expr, str], **kwargs) -> Expr:
        """梯底 (Ladder Bottom)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLLADDERBOTTOM(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_longleggeddoji(expr: Union[Expr, str], **kwargs) -> Expr:
        """长腿十字 (Long Legged Doji)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLLONGLEGGEDDOJI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_longline(expr: Union[Expr, str], **kwargs) -> Expr:
        """长线 (Long Line)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLLONGLINE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_marubozu(expr: Union[Expr, str], **kwargs) -> Expr:
        """光头光脚阳线/阴线 (Marubozu)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLMARUBOZU(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_matchinglow(expr: Union[Expr, str], **kwargs) -> Expr:
        """相同低价 (Matching Low)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLMATCHINGLOW(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_mathold(expr: Union[Expr, str], penetration: float = 0.5, **kwargs) -> Expr:
        """持有一线 (Mat Hold)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLMATHOLD(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_morningdojistar(expr: Union[Expr, str], penetration: float = 0.0, **kwargs) -> Expr:
        """十字晨星 (Morning Doji Star)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLMORNINGDOJISTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy(), penetration=penetration)),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_onneck(expr: Union[Expr, str], **kwargs) -> Expr:
        """颈部线 (On-Neck Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLONNECK(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_rickshawman(expr: Union[Expr, str], **kwargs) -> Expr:
        """黄包车夫 (Rickshaw Man)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLRICKSHAWMAN(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_risefall3methods(expr: Union[Expr, str], **kwargs) -> Expr:
        """上升/下降三法 (Rising Three Methods / Falling Three Methods)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLRISEFALL3METHODS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_separatinglines(expr: Union[Expr, str], **kwargs) -> Expr:
        """分离线 (Separating Lines)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLSEPARATINGLINES(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_shortline(expr: Union[Expr, str], **kwargs) -> Expr:
        """短线 (Short Line)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLSHORTLINE(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_stalledpattern(expr: Union[Expr, str], **kwargs) -> Expr:
        """受阻形态 (Stalled Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLSTALLEDPATTERN(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_sticksandwich(expr: Union[Expr, str], **kwargs) -> Expr:
        """三明治形态 (Stick Sandwich)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLSTICKSANDWICH(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_takuri(expr: Union[Expr, str], **kwargs) -> Expr:
        """探水竿 (Takuri Line)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLTAKURI(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_tasukigap(expr: Union[Expr, str], **kwargs) -> Expr:
        """跳空并列阴阳线 (Tasuki Gap)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLTASUKIGAP(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_thrusting(expr: Union[Expr, str], **kwargs) -> Expr:
        """插入线 (Thrusting Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLTHRUSTING(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_tristar(expr: Union[Expr, str], **kwargs) -> Expr:
        """三星 (Tristar Pattern)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLTRISTAR(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_uniqueriver(expr: Union[Expr, str], **kwargs) -> Expr:
        """独特三河底 (Unique 3 River)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLUNIQUE3RIVER(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_upsidegap2crows(expr: Union[Expr, str], **kwargs) -> Expr:
        """向上跳空两只乌鸦 (Upside Gap Two Crows)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLUPSIDEGAP2CROWS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )

    @staticmethod
    def cdl_xsidegap3methods(expr: Union[Expr, str], **kwargs) -> Expr:
        """向上/向下跳空三法 (Upside/Downside Gap Three Methods)"""
        import talib
        e = _ensure_expr(expr)
        return e.map_batches(
            lambda s: pl.Series(talib.CDLXSIDEGAP3METHODS(s.to_numpy(), s.to_numpy(), s.to_numpy(), s.to_numpy())),
            return_dtype=pl.Int32,
        )


# 单例导出
talib_ops = TaLibOperators()
