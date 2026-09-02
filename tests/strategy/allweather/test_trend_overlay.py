# coding=utf-8
"""AllWeather — trend_overlay 单元测试 (Step 3, PLAN §十二)."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.allweather.trend_overlay import (
    combined, momentum_tilt, v1_overlay, v3_overlay, v4_overlay,
)


def _rising_prices(t: int = 200, n: int = 4) -> pd.DataFrame:
    """近确定性上涨 (日均 +0.1%, 极小噪声), 保证动量/夏普为正."""
    r = np.random.default_rng(1)
    return pd.DataFrame(
        np.cumprod(1 + 0.001 + r.normal(0, 1e-6, size=(t, n)), axis=0),
        columns=[f"a{i}" for i in range(n)],
    )


def _falling_prices(t: int = 200, n: int = 4) -> pd.DataFrame:
    """近确定性下跌 (日均 -0.2%)."""
    r = np.random.default_rng(2)
    return pd.DataFrame(
        np.cumprod(1 - 0.002 + r.normal(0, 1e-6, size=(t, n)), axis=0),
        columns=[f"a{i}" for i in range(n)],
    )


class TestV1:
    def test_rising_gets_one_falling_gets_coef(self):
        up = v1_overlay(_rising_prices(), lookback=60, coef=0.0)
        down = v1_overlay(_falling_prices(), lookback=60, coef=0.0)
        assert (up.iloc[60:] == 1.0).all().all()
        assert (down.iloc[60:] == 0.0).all().all()

    def test_half_cut(self):
        down = v1_overlay(_falling_prices(), lookback=60, coef=0.5)
        assert (down.iloc[60:] == 0.5).all().all()

    def test_no_lookahead_truncation_consistent(self):
        prices = _rising_prices(300, 4)
        full = v1_overlay(prices, lookback=60)
        for t in (100, 200, 299):
            trunc = v1_overlay(prices.iloc[: t + 1], lookback=60)
            np.testing.assert_allclose(trunc.iloc[-1], full.iloc[t])

    def test_warmup_period_is_zero(self):
        prices = _rising_prices(200, 4)
        ov = v1_overlay(prices, lookback=60)
        # lookback 窗口未满 → 动量 NaN → overlay 0 (不持仓, 不用不完整信号)
        assert (ov.iloc[:59] == 0.0).all().all()


class TestV4:
    def test_positive_sharpe_normalized(self):
        prices = _rising_prices(400, 5)
        ov = v4_overlay(prices, lookback=252)
        row = ov.iloc[300]
        assert (row > 0).all()
        # 归一化: listed 资产均值 ≈ 1
        assert row.mean() == pytest.approx(1.0, abs=1e-6)

    def test_all_negative_sharpe_gives_floor(self):
        prices = _falling_prices(400, 5)
        ov = v4_overlay(prices, lookback=252, floor=0.0)
        assert (ov.iloc[300:] == 0.0).all().all()

    def test_dispersion_matches_sharpe_ranking(self):
        # 高夏普资产 overlay 应更高 (构造两个夏普差异明显的资产)
        r = np.random.default_rng(5)
        t = 500
        good = np.cumprod(1 + r.normal(0.002, 0.005, size=t), axis=0)
        bad = np.cumprod(1 + r.normal(0.0001, 0.02, size=t), axis=0)
        prices = pd.DataFrame({"good": good, "bad": bad})
        ov = v4_overlay(prices, lookback=252)
        row = ov.iloc[-1]
        assert row["good"] > row["bad"]

    def test_unlisted_asset_is_zero(self):
        prices = _rising_prices(400, 3)
        prices["late"] = np.nan  # 全程未上市
        prices.loc[:350, "a2"] = np.nan  # a2 前 350 日未上市
        ov = v4_overlay(prices, lookback=60)
        assert (ov["late"] == 0.0).all()
        seg = ov.loc[:348, "a2"]
        assert (seg == 0.0).all()
        # 无 NaN 泄漏
        assert not ov.isna().any().any()

    def test_no_lookahead_truncation_consistent(self):
        prices = _rising_prices(400, 4)
        full = v4_overlay(prices, lookback=120)
        for t in (200, 399):
            trunc = v4_overlay(prices.iloc[: t + 1], lookback=120)
            np.testing.assert_allclose(trunc.iloc[-1], full.iloc[t], rtol=1e-8)


class TestCombined:
    def test_none_mode_all_ones(self):
        prices = _rising_prices(100, 3)
        ov = combined(prices, mode="none")
        assert (ov == 1.0).all().all()

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            combined(_rising_prices(50, 2), mode="v9")

    def test_dispatch_v1(self):
        prices = _rising_prices(200, 3)
        assert combined(prices, mode="v1", lookback=60).equals(
            v1_overlay(prices, lookback=60)
        )


class TestV3:
    def test_portfolio_down_gives_coef(self):
        prices = _falling_prices(200, 4)
        ov = v3_overlay(prices, lookback=60, coef=0.0)
        assert (ov.iloc[60:] == 0.0).all().all()

    def test_portfolio_up_gives_one(self):
        prices = _rising_prices(200, 4)
        ov = v3_overlay(prices, lookback=60, coef=0.0)
        assert (ov.iloc[60:] == 1.0).all().all()

    def test_row_uniform(self):
        # 组合级: 同一行所有资产系数相同
        prices = _rising_prices(200, 4)
        ov = v3_overlay(prices, lookback=60)
        assert (ov.nunique(axis=1) == 1).all()

    def test_no_lookahead(self):
        prices = _rising_prices(300, 4)
        full = v3_overlay(prices, lookback=60)
        trunc = v3_overlay(prices.iloc[:201], lookback=60)
        np.testing.assert_allclose(trunc.iloc[-1], full.iloc[200])


class TestMomentumTilt:
    def test_positive_mom_ranked(self):
        r = np.random.default_rng(5)
        t = 300
        good = np.cumprod(1 + r.normal(0.002, 0.005, size=t))
        bad = np.cumprod(1 + r.normal(0.0, 0.005, size=t))
        prices = pd.DataFrame({"good": good, "bad": bad})
        ov = momentum_tilt(prices, lookback=60)
        assert ov["good"].iloc[-1] > ov["bad"].iloc[-1]

    def test_negative_mom_gets_floor(self):
        prices = _falling_prices(200, 3)
        ov = momentum_tilt(prices, lookback=60, floor=0.25)
        assert (ov.iloc[60:] == 0.25).all().all()

    def test_unlisted_zero_no_nan(self):
        prices = _rising_prices(200, 3)
        prices["late"] = np.nan
        ov = momentum_tilt(prices, lookback=60)
        assert (ov["late"] == 0.0).all()
        assert not ov.isna().any().any()

    def test_dispatch(self):
        prices = _rising_prices(200, 3)
        assert combined(prices, mode="v3", lookback=60).equals(
            v3_overlay(prices, lookback=60))
        assert combined(prices, mode="tilt", lookback=60).equals(
            momentum_tilt(prices, lookback=60))
