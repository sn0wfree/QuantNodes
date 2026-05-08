# coding=utf-8
"""
TA-Lib 算子（代理层）

本模块将 factor_functions/talib_ops.py 中的注册算子包装为类接口。
TA-Lib 算子通过 map_batches 桥接 Polars ↔ NumPy。

注意：本模块为代理层，实际实现位于 factor_functions/talib_ops.py。

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

from typing import TYPE_CHECKING, Union, Tuple

from polars import Expr


if TYPE_CHECKING:
    pass




class TaLibOperators:
    """TA-Lib 算子代理层"""

    @staticmethod
    def sma(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """简单移动平均线 (Simple Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sma(expr, timeperiod=timeperiod)

    @staticmethod
    def ema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """指数移动平均线 (Exponential Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ema(expr, timeperiod=timeperiod)

    @staticmethod
    def wma(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """加权移动平均线 (Weighted Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_wma(expr, timeperiod=timeperiod)

    @staticmethod
    def dema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """双重指数移动平均线 (Double Exponential Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_dema(expr, timeperiod=timeperiod)

    @staticmethod
    def tema(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """三重指数移动平均线 (Triple Exponential Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_tema(expr, timeperiod=timeperiod)

    @staticmethod
    def trima(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """三角移动平均线 (Triangular Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_trima(expr, timeperiod=timeperiod)

    @staticmethod
    def kama(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """考夫曼自适应移动平均线 (Kaufman Adaptive Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_kama(expr, timeperiod=timeperiod)

    @staticmethod
    def t3(expr: Union[Expr, str], timeperiod: int = 5, vfactor: float = 0.7) -> Expr:
        """T3 移动平均线 (Triple Exponential Moving Average with volume factor)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_t3(expr, timeperiod=timeperiod, vfactor=vfactor)

    @staticmethod
    def ht_trendline(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换趋势线 (Hilbert Transform Trendline)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ht_trendline(expr)

    @staticmethod
    def mama(expr: Union[Expr, str], fastlimit: float = 0.5, slowlimit: float = 0.05) -> Expr:
        """MAMA 移动平均 (MESA Adaptive Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_mama(expr, fastlimit=fastlimit, slowlimit=slowlimit)

    @staticmethod
    def mavp(expr: Union[Expr, str], periods: Union[Expr, str], minperiod: int = 2, maxperiod: int = 30) -> Expr:
        """移动平均变动周期 (MA with Variable Period)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_mavp(expr, periods, minperiod=minperiod, maxperiod=maxperiod)

    @staticmethod
    def midpoint(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """价格中点 (Midpoint over period)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_midpoint(expr, timeperiod=timeperiod)

    @staticmethod
    def midprice(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """中间价格 (Midpoint Price over period)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_midprice(expr, timeperiod=timeperiod)

    @staticmethod
    def sar(expr: Union[Expr, str], acceleration: float = 0.02, maximum: float = 0.2) -> Expr:
        """抛物线 SAR"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sar(expr, acceleration=acceleration, maximum=maximum)

    @staticmethod
    def ma(expr: Union[Expr, str], timeperiod: int = 30, matype: int = 0) -> Expr:
        """通用移动平均线 (Moving Average)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ma(expr, timeperiod=timeperiod, matype=matype)

    @staticmethod
    def bbands(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
               nbdevdn: float = 2.0, matype: int = 0) -> Tuple[Expr, Expr, Expr]:
        """布林带 — 返回 (upper, middle, lower)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_bbands(expr, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    @staticmethod
    def bbands_upper(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                     nbdevdn: float = 2.0, matype: int = 0) -> Expr:
        """布林带上轨"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_bbands_upper(expr, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    @staticmethod
    def bbands_middle(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                      nbdevdn: float = 2.0, matype: int = 0) -> Expr:
        """布林带中轨"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_bbands_middle(expr, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    @staticmethod
    def bbands_lower(expr: Union[Expr, str], timeperiod: int = 5, nbdevup: float = 2.0,
                     nbdevdn: float = 2.0, matype: int = 0) -> Expr:
        """布林带下轨"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_bbands_lower(expr, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)

    @staticmethod
    def rsi(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """相对强弱指标 (Relative Strength Index)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_rsi(expr, timeperiod=timeperiod)

    @staticmethod
    def macd(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
             signalperiod: int = 9) -> Tuple[Expr, Expr, Expr]:
        """MACD — 返回 (macd_line, signal_line, histogram)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_macd(expr, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    @staticmethod
    def macd_line(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                  signalperiod: int = 9) -> Expr:
        """MACD 线"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_macd_line(expr, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    @staticmethod
    def macd_signal(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                    signalperiod: int = 9) -> Expr:
        """MACD 信号线"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_macd_signal(expr, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    @staticmethod
    def macd_hist(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26,
                  signalperiod: int = 9) -> Expr:
        """MACD 柱状图 (histogram)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_macd_hist(expr, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    @staticmethod
    def stoch(expr: Union[Expr, str], fastk_period: int = 5, slowk_period: int = 3,
              slowk_matype: int = 0, slowd_period: int = 3, slowd_matype: int = 0) -> Tuple[Expr, Expr]:
        """随机指标 (Stochastic) — 返回 (slowk, slowd)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stoch(expr, fastk_period=fastk_period, slowk_period=slowk_period, 
                                     slowk_matype=slowk_matype, slowd_period=slowd_period, slowd_matype=slowd_matype)

    @staticmethod
    def stoch_k(expr: Union[Expr, str], fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0) -> Expr:
        """随机指标 K 线 (Stochastic %K)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stoch_k(expr, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype)

    @staticmethod
    def stoch_d(expr: Union[Expr, str], fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0,
                slowd_period: int = 3, slowd_matype: int = 0) -> Expr:
        """随机指标 D 线 (Stochastic %D)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stoch_d(expr, fastk_period=fastk_period, slowk_period=slowk_period, slowk_matype=slowk_matype,
                                       slowd_period=slowd_period, slowd_matype=slowd_matype)

    @staticmethod
    def cci(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """商品通道指标 (Commodity Channel Index)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cci(expr, timeperiod=timeperiod)

    @staticmethod
    def willr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """威廉指标 (Williams %R)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_willr(expr, timeperiod=timeperiod)

    @staticmethod
    def mfi(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """资金流量指标 (Money Flow Index)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_mfi(expr, timeperiod=timeperiod)

    @staticmethod
    def roc(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率 (Rate of Change)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_roc(expr, timeperiod=timeperiod)

    @staticmethod
    def rocp(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率百分比 (Rate of Change Percentage)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_rocp(expr, timeperiod=timeperiod)

    @staticmethod
    def rocr(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率比率 (Rate of Change Ratio)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_rocr(expr, timeperiod=timeperiod)

    @staticmethod
    def rocr100(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """变动率比率*100 (Rate of Change Ratio 100 scale)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_rocr100(expr, timeperiod=timeperiod)

    @staticmethod
    def mom(expr: Union[Expr, str], timeperiod: int = 10) -> Expr:
        """动量 (Momentum)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_mom(expr, timeperiod=timeperiod)

    @staticmethod
    def adx(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """平均趋向指标 (Average Directional Movement Index)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_adx(expr, timeperiod=timeperiod)

    @staticmethod
    def adxr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """平均趋向指标评估 (Average Directional Movement Index Rating)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_adxr(expr, timeperiod=timeperiod)

    @staticmethod
    def apo(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
        """绝对价格震荡 (Absolute Price Oscillator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_apo(expr, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    @staticmethod
    def ppo(expr: Union[Expr, str], fastperiod: int = 12, slowperiod: int = 26, matype: int = 0) -> Expr:
        """百分比价格震荡 (Percentage Price Oscillator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ppo(expr, fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)

    @staticmethod
    def cmo(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """钱德动量摆动指标 (Chande Momentum Oscillator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cmo(expr, timeperiod=timeperiod)

    @staticmethod
    def dx(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """趋向指标 (Directional Movement Index)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_dx(expr, timeperiod=timeperiod)

    @staticmethod
    def minus_di(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """负向指标 (Minus Directional Indicator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_minus_di(expr, timeperiod=timeperiod)

    @staticmethod
    def plus_di(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """正向指标 (Plus Directional Indicator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_plus_di(expr, timeperiod=timeperiod)

    @staticmethod
    def minus_dm(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """负向动量 (Minus Directional Movement)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_minus_dm(expr, timeperiod=timeperiod)

    @staticmethod
    def plus_dm(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """正向动量 (Plus Directional Movement)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_plus_dm(expr, timeperiod=timeperiod)

    @staticmethod
    def aroon(expr: Union[Expr, str], timeperiod: int = 14) -> Tuple[Expr, Expr]:
        """阿隆指标 — 返回 (aroondown, aroonup)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_aroon(expr, timeperiod=timeperiod)

    @staticmethod
    def aroondown(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """阿隆下降线"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_aroondown(expr, timeperiod=timeperiod)

    @staticmethod
    def aroonup(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """阿隆上升线"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_aroonup(expr, timeperiod=timeperiod)

    @staticmethod
    def aroonosc(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """阿隆振荡指标 (Aroon Oscillator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_aroonosc(expr, timeperiod=timeperiod)

    @staticmethod
    def bop(expr: Union[Expr, str]) -> Expr:
        """力量指标 (Balance of Power)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_bop(expr)

    @staticmethod
    def trix(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """三重指数平滑变动率 (1-day Rate of Change of Triple Smooth EMA)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_trix(expr, timeperiod=timeperiod)

    @staticmethod
    def ultosc(expr: Union[Expr, str], timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28) -> Expr:
        """终极振荡指标 (Ultimate Oscillator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ultosc(expr, timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)

    @staticmethod
    def stochf(expr: Union[Expr, str], fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Tuple[Expr, Expr]:
        """快速随机指标 — 返回 (fastk, fastd)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stochf(expr, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)

    @staticmethod
    def stochrsi(expr: Union[Expr, str], timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Tuple[Expr, Expr]:
        """随机 RSI — 返回 (fastk, fastd)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stochrsi(expr, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)

    @staticmethod
    def stochrsi_k(expr: Union[Expr, str], timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Expr:
        """随机 RSI %K"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stochrsi_k(expr, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)

    @staticmethod
    def stochrsi_d(expr: Union[Expr, str], timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0) -> Expr:
        """随机 RSI %D"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stochrsi_d(expr, timeperiod=timeperiod, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=fastd_matype)

    @staticmethod
    def macdext(expr: Union[Expr, str], fastperiod: int = 12, fastmatype: int = 0, slowperiod: int = 26, slowmatype: int = 0,
                signalperiod: int = 9, signalmatype: int = 0) -> Tuple[Expr, Expr, Expr]:
        """扩展 MACD — 返回 (macd, signal, hist)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_macdext(expr, fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype,
                                        signalperiod=signalperiod, signalmatype=signalmatype)

    @staticmethod
    def macdfix(expr: Union[Expr, str], signalperiod: int = 9) -> Tuple[Expr, Expr, Expr]:
        """固定周期 MACD — 返回 (macd, signal, hist)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_macdfix(expr, signalperiod=signalperiod)

    @staticmethod
    def atr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """平均真实范围 (Average True Range)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_atr(expr, timeperiod=timeperiod)

    @staticmethod
    def natr(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """归一化平均真实范围 (Normalized Average True Range)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_natr(expr, timeperiod=timeperiod)

    @staticmethod
    def trange(expr: Union[Expr, str]) -> Expr:
        """真实范围 (True Range)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_trange(expr)

    @staticmethod
    def ad(expr: Union[Expr, str]) -> Expr:
        """累积/派发线 (Accumulation/Distribution Line)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ad(expr)

    @staticmethod
    def adosc(expr: Union[Expr, str], fastperiod: int = 3, slowperiod: int = 10) -> Expr:
        """累积/派发震荡 (Chaikin A/D Oscillator)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_adosc(expr, fastperiod=fastperiod, slowperiod=slowperiod)

    @staticmethod
    def obv(expr: Union[Expr, str]) -> Expr:
        """能量潮 (On Balance Volume)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_obv(expr)

    @staticmethod
    def cdl_doji(expr: Union[Expr, str]) -> Expr:
        """十字星 (Doji)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_doji(expr)

    @staticmethod
    def cdl_hammer(expr: Union[Expr, str]) -> Expr:
        """锤子线 (Hammer)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_hammer(expr)

    @staticmethod
    def cdl_engulfing(expr: Union[Expr, str]) -> Expr:
        """吞没形态 (Engulfing Pattern)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_engulfing(expr)

    @staticmethod
    def cdl_morningstar(expr: Union[Expr, str], penetration: float = 0.0) -> Expr:
        """晨星 (Morning Star)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_morningstar(expr, penetration=penetration)

    @staticmethod
    def cdl_eveningstar(expr: Union[Expr, str], penetration: float = 0.0) -> Expr:
        """暮星 (Evening Star)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_eveningstar(expr, penetration=penetration)

    @staticmethod
    def cdl_hangingman(expr: Union[Expr, str]) -> Expr:
        """上吊线 (Hanging Man)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_hangingman(expr)

    @staticmethod
    def cdl_shootingstar(expr: Union[Expr, str]) -> Expr:
        """射击之星 (Shooting Star)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_shootingstar(expr)

    @staticmethod
    def cdl_harami(expr: Union[Expr, str]) -> Expr:
        """孕线形态 (Harami Pattern)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_harami(expr)

    @staticmethod
    def cdl_piercing(expr: Union[Expr, str]) -> Expr:
        """刺透形态 (Piercing Pattern)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_piercing(expr)

    @staticmethod
    def cdl_darkcloudcover(expr: Union[Expr, str], penetration: float = 0.0) -> Expr:
        """乌云盖顶 (Dark Cloud Cover)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_darkcloudcover(expr, penetration=penetration)

    @staticmethod
    def cdl_spinningtop(expr: Union[Expr, str]) -> Expr:
        """纺锤顶 (Spinning Top)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_spinningtop(expr)

    @staticmethod
    def cdl_3whitesoldiers(expr: Union[Expr, str]) -> Expr:
        """三只白兵 (Three White Soldiers)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_3whitesoldiers(expr)

    @staticmethod
    def cdl_3blackcrows(expr: Union[Expr, str]) -> Expr:
        """三只乌鸦 (Three Black Crows)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cdl_3blackcrows(expr)

    @staticmethod
    def avgprice(expr: Union[Expr, str]) -> Expr:
        """平均价格 (Average Price)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_avgprice(expr)

    @staticmethod
    def medprice(expr: Union[Expr, str]) -> Expr:
        """中间价格 (Median Price)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_medprice(expr)

    @staticmethod
    def typprice(expr: Union[Expr, str]) -> Expr:
        """典型价格 (Typical Price)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_typprice(expr)

    @staticmethod
    def wclprice(expr: Union[Expr, str]) -> Expr:
        """加权收盘价 (Weighted Close Price)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_wclprice(expr)

    @staticmethod
    def beta(expr: Union[Expr, str], timeperiod: int = 5) -> Expr:
        """Beta 系数"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_beta(expr, timeperiod=timeperiod)

    @staticmethod
    def correl(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """相关系数 (Pearson's Correlation Coefficient)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_correl(expr, timeperiod=timeperiod)

    @staticmethod
    def linearreg(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归值 (Linear Regression)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_linearreg(expr, timeperiod=timeperiod)

    @staticmethod
    def linearreg_angle(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归角度 (Linear Regression Angle)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_linearreg_angle(expr, timeperiod=timeperiod)

    @staticmethod
    def linearreg_intercept(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归截距 (Linear Regression Intercept)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_linearreg_intercept(expr, timeperiod=timeperiod)

    @staticmethod
    def linearreg_slope(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """线性回归斜率 (Linear Regression Slope)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_linearreg_slope(expr, timeperiod=timeperiod)

    @staticmethod
    def stddev(expr: Union[Expr, str], timeperiod: int = 5, nbdev: int = 1) -> Expr:
        """标准差 (Standard Deviation)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_stddev(expr, timeperiod=timeperiod, nbdev=nbdev)

    @staticmethod
    def tsf(expr: Union[Expr, str], timeperiod: int = 14) -> Expr:
        """时间序列预测 (Time Series Forecast)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_tsf(expr, timeperiod=timeperiod)

    @staticmethod
    def var(expr: Union[Expr, str], timeperiod: int = 5, nbdev: int = 1) -> Expr:
        """方差 (Variance)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_var(expr, timeperiod=timeperiod, nbdev=nbdev)

    @staticmethod
    def ht_dcperiod(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换主导周期 (Hilbert Transform Dominant Cycle Period)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ht_dcperiod(expr)

    @staticmethod
    def ht_dcphase(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换主导周期相位 (Hilbert Transform Dominant Cycle Phase)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ht_dcphase(expr)

    @staticmethod
    def ht_phasor(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换相位 (Hilbert Transform Phasor)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ht_phasor(expr)

    @staticmethod
    def ht_sine(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换正弦 (Hilbert Transform SineWave)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ht_sine(expr)

    @staticmethod
    def ht_trendmode(expr: Union[Expr, str]) -> Expr:
        """希尔伯特变换趋势模式 (Hilbert Transform Trend vs Cycle Mode)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ht_trendmode(expr)

    @staticmethod
    def acos(expr: Union[Expr, str]) -> Expr:
        """反余弦 (Inverse Cosine)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_acos(expr)

    @staticmethod
    def asin(expr: Union[Expr, str]) -> Expr:
        """反正弦 (Inverse Sine)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_asin(expr)

    @staticmethod
    def atan(expr: Union[Expr, str]) -> Expr:
        """反正切 (Inverse Tangent)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_atan(expr)

    @staticmethod
    def cos(expr: Union[Expr, str]) -> Expr:
        """余弦 (Cosine)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cos(expr)

    @staticmethod
    def cosh(expr: Union[Expr, str]) -> Expr:
        """双曲余弦 (Hyperbolic Cosine)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_cosh(expr)

    @staticmethod
    def sin(expr: Union[Expr, str]) -> Expr:
        """正弦 (Sine)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sin(expr)

    @staticmethod
    def sinh(expr: Union[Expr, str]) -> Expr:
        """双曲正弦 (Hyperbolic Sine)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sinh(expr)

    @staticmethod
    def sqrt(expr: Union[Expr, str]) -> Expr:
        """平方根 (Square Root)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sqrt(expr)

    @staticmethod
    def tan(expr: Union[Expr, str]) -> Expr:
        """正切 (Tangent)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_tan(expr)

    @staticmethod
    def tanh(expr: Union[Expr, str]) -> Expr:
        """双曲正切 (Hyperbolic Tangent)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_tanh(expr)

    @staticmethod
    def exp(expr: Union[Expr, str]) -> Expr:
        """指数函数 (Exponential)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_exp(expr)

    @staticmethod
    def ln(expr: Union[Expr, str]) -> Expr:
        """自然对数 (Natural Log)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ln(expr)

    @staticmethod
    def log10(expr: Union[Expr, str]) -> Expr:
        """常用对数 (Log10)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_log10(expr)

    @staticmethod
    def ceil(expr: Union[Expr, str]) -> Expr:
        """向上取整 (Ceiling)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_ceil(expr)

    @staticmethod
    def floor(expr: Union[Expr, str]) -> Expr:
        """向下取整 (Floor)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_floor(expr)

    @staticmethod
    def add(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """加法 (Add)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_add(expr_a, expr_b)

    @staticmethod
    def sub(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """减法 (Subtract)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sub(expr_a, expr_b)

    @staticmethod
    def mult(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """乘法 (Multiply)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_mult(expr_a, expr_b)

    @staticmethod
    def div(expr_a: Union[Expr, str], expr_b: Union[Expr, str]) -> Expr:
        """除法 (Divide)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_div(expr_a, expr_b)

    @staticmethod
    def max(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动最大值 (Rolling Maximum)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_max(expr, timeperiod=timeperiod)

    @staticmethod
    def min(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动最小值 (Rolling Minimum)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_min(expr, timeperiod=timeperiod)

    @staticmethod
    def sum(expr: Union[Expr, str], timeperiod: int = 30) -> Expr:
        """滚动求和 (Rolling Sum)"""
        from QuantNodes.factor_node.factor_functions import talib_ops
        return talib_ops.talib_sum(expr, timeperiod=timeperiod)


talib_ops = TaLibOperators()