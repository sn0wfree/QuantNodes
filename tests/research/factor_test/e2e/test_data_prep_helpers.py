# coding: utf-8
"""Unit tests for data_prep synthetic data generators."""

import numpy as np
import pandas as pd

from QuantNodes.research.factor_test.e2e.data_prep import (
    _gen_dates,
    _gen_factor_data,
    _gen_index_cp,
    _gen_stk_daily,
    _gen_stocks,
)


def test_gen_dates_count_and_format():
    dates = _gen_dates(n_days=20)
    assert len(dates) == 20
    assert all(isinstance(d, int) and len(str(d)) == 8 for d in dates)
    assert dates == sorted(dates)


def test_gen_stocks_range():
    stocks = _gen_stocks(5)
    assert stocks == [100001, 100002, 100003, 100004, 100005]


def test_gen_factor_data_momentum_shape():
    rng = np.random.RandomState(0)
    df = _gen_factor_data(rng, 30, 5, "momentum_20d")
    assert df.shape == (30, 5)
    assert df.notna().all().all()


def test_gen_factor_data_reversal_negative_trend():
    rng = np.random.RandomState(0)
    df = _gen_factor_data(rng, 60, 8, "reversal_5d")
    head_mean = df.iloc[:5].mean().mean()
    tail_mean = df.iloc[-5:].mean().mean()
    assert tail_mean < head_mean


def test_gen_factor_data_volatility_low_magnitude():
    rng = np.random.RandomState(0)
    df = _gen_factor_data(rng, 30, 5, "volatility_60d")
    assert df.abs().mean().mean() < 1.5


def test_gen_factor_data_default_random():
    rng = np.random.RandomState(0)
    df = _gen_factor_data(rng, 30, 5, "noise_factor")
    assert df.shape == (30, 5)


def test_gen_factor_data_determinism():
    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(123)
    a = _gen_factor_data(rng1, 20, 3, "momentum")
    b = _gen_factor_data(rng2, 20, 3, "momentum")
    pd.testing.assert_frame_equal(a, b)


def test_gen_index_cp_shape_and_columns():
    rng = np.random.RandomState(1)
    df = _gen_index_cp(rng, 25)
    assert df.shape == (25, 2)
    assert list(df.columns) == ["000300.SH", "000905.SH"]


def test_gen_stk_daily_keys_and_shapes():
    rng = np.random.RandomState(7)
    out = _gen_stk_daily(rng, 30, 6)
    assert set(out.keys()) == {"cp", "st", "suspend", "ud_limit", "ipo_days",
                                "id_citic1", "mv_float"}
    for df in out.values():
        assert df.shape == (30, 6)


def test_gen_stk_daily_st_pattern():
    rng = np.random.RandomState(7)
    out = _gen_stk_daily(rng, 10, 5)
    st = out["st"]
    assert (st.iloc[:, :2] == 1).all().all()
    assert (st.iloc[:, 2:] == 0).all().all()


def test_gen_stk_daily_ipo_days_first_stock_low():
    rng = np.random.RandomState(7)
    out = _gen_stk_daily(rng, 10, 5)
    assert out["ipo_days"].iloc[0, 0] == 100
    assert (out["ipo_days"].iloc[1:] == 500).all().all()


def test_gen_stk_daily_industry_in_range():
    rng = np.random.RandomState(7)
    out = _gen_stk_daily(rng, 20, 4)
    industry = out["id_citic1"].values
    assert industry.min() >= 1 and industry.max() <= 30


def test_gen_stk_daily_price_positive_monotonic_ish():
    rng = np.random.RandomState(7)
    out = _gen_stk_daily(rng, 20, 4)
    assert (out["cp"] > 0).all().all()
