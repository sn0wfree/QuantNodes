# -*- coding: utf-8 -*-
"""QuantNodes.operators.talib 单元测试"""
import pytest
import polars as pl

from QuantNodes.operators import talib_ops


class TestTaLibTrendIndicators:
    """TA-Lib 趋势指标测试"""

    @pytest.fixture
    def price_data(self):
        return pl.Series("close", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0] * 5)

    def test_sma_basic(self, price_data):
        result = talib_ops.sma(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_sma_output_type(self, price_data):
        df = pl.DataFrame({"close": price_data})
        result = df.select(talib_ops.sma(pl.col("close"), timeperiod=5))
        assert result.columns[0] == "close"

    def test_ema_basic(self, price_data):
        result = talib_ops.ema(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_ema_output_type(self, price_data):
        df = pl.DataFrame({"close": price_data})
        result = df.select(talib_ops.ema(pl.col("close"), timeperiod=5))
        assert result.columns[0] == "close"

    def test_wma_basic(self, price_data):
        result = talib_ops.wma(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_dema_basic(self, price_data):
        result = talib_ops.dema(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_tema_basic(self, price_data):
        result = talib_ops.tema(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_trima_basic(self, price_data):
        result = talib_ops.trima(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_kama_basic(self, price_data):
        result = talib_ops.kama(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_t3_basic(self, price_data):
        result = talib_ops.t3(price_data, timeperiod=5)
        assert isinstance(result, pl.Expr)


class TestTaLibMomentumIndicators:
    """TA-Lib 动量指标测试"""

    @pytest.fixture
    def price_data(self):
        return pl.Series("close", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0] * 5)

    def test_rsi_basic(self, price_data):
        result = talib_ops.rsi(price_data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_rsi_output_shape(self, price_data):
        df = pl.DataFrame({"close": price_data})
        result = df.select(talib_ops.rsi(pl.col("close"), timeperiod=14))
        assert len(result) == len(df)

    def test_macd_basic(self, price_data):
        result = talib_ops.macd(price_data, fastperiod=12, slowperiod=26, signalperiod=9)
        assert isinstance(result, tuple)

    def test_macd_returns_tuple(self, price_data):
        df = pl.DataFrame({"close": price_data})
        result = talib_ops.macd(pl.col("close"))
        assert len(result) == 3
        macd_line, signal, hist = result
        assert isinstance(macd_line, pl.Expr)

    def test_stoch_basic(self, price_data):
        result = talib_ops.stoch(price_data, fastk_period=14, slowk_period=3, slowk_matype=0)
        assert isinstance(result, tuple)

    def test_cci_basic(self, price_data):
        result = talib_ops.cci(price_data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_willr_basic(self, price_data):
        result = talib_ops.willr(price_data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_adx_basic(self, price_data):
        result = talib_ops.adx(price_data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_apo_basic(self, price_data):
        result = talib_ops.apo(price_data, fastperiod=12, slowperiod=26, matype=0)
        assert isinstance(result, pl.Expr)

    def test_rocp_basic(self, price_data):
        result = talib_ops.rocp(price_data, timeperiod=10)
        assert isinstance(result, pl.Expr)

    def test_mom_basic(self, price_data):
        result = talib_ops.mom(price_data, timeperiod=10)
        assert isinstance(result, pl.Expr)


class TestTaLibVolatilityIndicators:
    """TA-Lib 波动率指标测试"""

    @pytest.fixture
    def price_data(self):
        return pl.Series("close", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0] * 5)

    def test_atr_basic(self, price_data):
        result = talib_ops.atr(price_data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_natr_basic(self, price_data):
        result = talib_ops.natr(price_data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_trange_basic(self, price_data):
        result = talib_ops.trange(price_data)
        assert isinstance(result, pl.Expr)

    def test_bbands_basic(self, price_data):
        result = talib_ops.bbands_upper(price_data, timeperiod=5, nbdevup=2.0, nbdevdn=2.0, matype=0)
        assert isinstance(result, pl.Expr)

    def test_bbands_returns_three_columns(self, price_data):
        upper = talib_ops.bbands_upper(price_data, timeperiod=5)
        middle = talib_ops.bbands_middle(price_data, timeperiod=5)
        lower = talib_ops.bbands_lower(price_data, timeperiod=5)
        assert isinstance(upper, pl.Expr)
        assert isinstance(middle, pl.Expr)
        assert isinstance(lower, pl.Expr)


class TestTaLibVolumeIndicators:
    """TA-Lib 成交量指标测试"""

    @pytest.fixture
    def price_data(self):
        return pl.Series("close", [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0] * 5)

    def test_obv_basic(self, price_data):
        result = talib_ops.obv(price_data)
        assert isinstance(result, pl.Expr)

    def test_ad_basic(self, price_data):
        result = talib_ops.ad(price_data)
        assert isinstance(result, pl.Expr)

    def test_adosc_basic(self, price_data):
        result = talib_ops.adosc(price_data, fastperiod=3, slowperiod=10)
        assert isinstance(result, pl.Expr)


class TestTaLibEdgeCases:
    """TA-Lib 边界情况测试"""

    @pytest.fixture
    def short_data(self):
        return pl.Series("close", [10.0, 11.0, 12.0])

    def test_sma_with_short_data(self, short_data):
        result = talib_ops.sma(short_data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_sma_with_invalid_timeperiod(self):
        data = pl.Series("close", [10.0, 11.0, 12.0] * 10)
        result = talib_ops.sma(data, timeperiod=0)
        assert isinstance(result, pl.Expr)

    def test_rsi_with_insufficient_data(self):
        data = pl.Series("close", [10.0, 11.0])
        result = talib_ops.rsi(data, timeperiod=14)
        assert isinstance(result, pl.Expr)

    def test_null_input_handling(self):
        data = pl.Series("close", [10.0, None, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        result = talib_ops.sma(data, timeperiod=5)
        assert isinstance(result, pl.Expr)

    def test_negative_price_handling(self):
        data = pl.Series("close", [10.0, -5.0, 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0, 11.0])
        result = talib_ops.sma(data, timeperiod=5)
        assert isinstance(result, pl.Expr)


class TestTaLibPatternRecognition:
    """TA-Lib K线形态识别测试"""

    @pytest.fixture
    def price_data(self):
        return pl.Series("close", [100.0, 101.0, 99.0, 102.0, 98.0] * 10)

    def test_cdl_doji_basic(self, price_data):
        result = talib_ops.cdl_doji(price_data)
        assert isinstance(result, pl.Expr)

    def test_cdl_hammer_basic(self, price_data):
        result = talib_ops.cdl_hammer(price_data)
        assert isinstance(result, pl.Expr)

    def test_cdl_engulfing_basic(self, price_data):
        result = talib_ops.cdl_engulfing(price_data)
        assert isinstance(result, pl.Expr)

    def test_cdl_morningstar_basic(self, price_data):
        result = talib_ops.cdl_morningstar(price_data)
        assert isinstance(result, pl.Expr)


class TestTaLibOperatorsList:
    """TA-Lib 算子列表测试"""

    def test_talib_operators_available(self):
        assert talib_ops is not None
        methods = [m for m in dir(talib_ops) if not m.startswith('_')]
        assert len(methods) > 50

    def test_sma_method_exists(self):
        assert hasattr(talib_ops, 'sma')
        assert callable(talib_ops.sma)

    def test_ema_method_exists(self):
        assert hasattr(talib_ops, 'ema')
        assert callable(talib_ops.ema)

    def test_rsi_method_exists(self):
        assert hasattr(talib_ops, 'rsi')
        assert callable(talib_ops.rsi)

    def test_macd_method_exists(self):
        assert hasattr(talib_ops, 'macd')
        assert callable(talib_ops.macd)

    def test_bbands_method_exists(self):
        assert hasattr(talib_ops, 'bbands')
        assert callable(talib_ops.bbands)

    def test_atr_method_exists(self):
        assert hasattr(talib_ops, 'atr')
        assert callable(talib_ops.atr)


class TestTaLibExpressionIntegration:
    """TA-Lib 表达式集成测试"""

    def test_sma_in_select(self):
        df = pl.DataFrame({"price": [10.0, 11.0, 12.0, 13.0, 14.0] * 5})
        result = df.select([
            pl.col("price"),
            talib_ops.sma(pl.col("price"), timeperiod=3).alias("sma_3")
        ])
        assert "sma_3" in result.columns

    def test_rsi_in_filter(self):
        df = pl.DataFrame({"price": [10.0, 11.0, 12.0, 13.0, 14.0] * 5})
        result = df.filter(talib_ops.rsi(pl.col("price"), timeperiod=5) > 30)
        assert isinstance(result, pl.DataFrame)

    def test_multiple_talib_in_select(self):
        df = pl.DataFrame({"price": [10.0, 11.0, 12.0, 13.0, 14.0] * 5})
        result = df.select([
            pl.col("price"),
            talib_ops.sma(pl.col("price"), timeperiod=3).alias("sma"),
            talib_ops.ema(pl.col("price"), timeperiod=3).alias("ema"),
        ])
        assert "sma" in result.columns
        assert "ema" in result.columns