# coding: utf-8
"""K12 (2026-06-21): date_utils 边界与正确性测试.

valid_date / datenum_to_datetime / datetime_to_datenum / chg_idx_to_datestr /
resample_trade_date / offset_date.
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.date_utils import (
    chg_idx_to_datestr,
    datenum_to_datetime,
    datetime_to_datenum,
    offset_date,
    resample_trade_date,
    valid_date,
)


# ── valid_date ──


class TestValidDate:
    def test_valid_int_dataframe(self):
        """yyyymmdd int DataFrame → True."""
        df = pd.DataFrame([20260101, 20260102, 20260103])
        assert valid_date(df) is True

    def test_valid_int_series(self):
        """yyyymmdd int Series → True."""
        s = pd.Series([20260101, 20260102])
        assert valid_date(s) is True

    def test_string_dataframe_rejected(self):
        """str dtype → False."""
        df = pd.DataFrame(["20260101", "20260102"])
        assert valid_date(df) is False

    def test_short_int_rejected(self):
        """5 位数字 → False (不是 yyyymmdd)."""
        df = pd.DataFrame([20210, 20220])
        assert valid_date(df) is False

    def test_non_dataframe_rejected(self):
        """list 等非 DataFrame/Series → False."""
        assert valid_date([20260101]) is False
        assert valid_date(20260101) is False
        assert valid_date(None) is False


# ── datenum_to_datetime / datetime_to_datenum ──


class TestDatenumConversion:
    def test_round_trip(self):
        """datenum → datetime → datenum 应保持不变."""
        df = pd.DataFrame([20260101, 20260615, 20261231])
        dt = datenum_to_datetime(df)
        back = datetime_to_datenum(dt)
        np.testing.assert_array_equal(
            back.iloc[:, 0].values, df.iloc[:, 0].values
        )

    def test_datenum_to_datetime_first(self):
        """20260101 → datetime(2026, 1, 1)."""
        df = pd.DataFrame([20260101])
        result = datenum_to_datetime(df)
        assert result.iloc[0, 0].year == 2026
        assert result.iloc[0, 0].month == 1
        assert result.iloc[0, 0].day == 1

    def test_datetime_to_datenum_first(self):
        """datetime(2026, 12, 31) → 20261231."""
        df = pd.DataFrame([pd.Timestamp("2026-12-31").to_pydatetime()])
        result = datetime_to_datenum(df)
        assert result.iloc[0, 0] == 20261231


# ── chg_idx_to_datestr ──


class TestChgIdxToDatestr:
    def test_basic_conversion(self):
        """20260101 → '2026/01/01'."""
        s = pd.Series([1, 2, 3], index=[20260101, 20260615, 20261231])
        result = chg_idx_to_datestr(s)
        assert list(result.index) == ["2026/01/01", "2026/06/15", "2026/12/31"]

    def test_dataframe_input(self):
        """DataFrame 输入也可工作."""
        df = pd.DataFrame({"a": [1, 2]}, index=[20260101, 20260201])
        result = chg_idx_to_datestr(df)
        assert list(result.index) == ["2026/01/01", "2026/02/01"]

    def test_does_not_mutate_input(self):
        """原 Series 不被修改."""
        s = pd.Series([1, 2], index=[20260101, 20260102])
        chg_idx_to_datestr(s)
        # 原始 index 仍为 int
        assert s.index[0] == 20260101


# ── resample_trade_date ──


@pytest.fixture
def trade_dt_year():
    """2025 整年工作日."""
    dates = [int(d.strftime("%Y%m%d"))
             for d in pd.bdate_range("2025-01-01", "2025-12-31")]
    return pd.DataFrame(dates, columns=[0])


class TestResampleTradeDate:
    def test_monthly_end(self, trade_dt_year):
        """M-end 应输出每月末工作日 (12 月)."""
        result = resample_trade_date(trade_dt_year, ("M", "end"))
        assert isinstance(result, pd.DataFrame)
        assert 10 <= len(result) <= 13

    def test_monthly_begin(self, trade_dt_year):
        """M-begin 应输出每月初工作日."""
        result = resample_trade_date(trade_dt_year, ("M", "begin"))
        assert 10 <= len(result) <= 13

    def test_quarterly_end(self, trade_dt_year):
        """Q-end 应输出每季末: 1 年 ≈ 4 个季度."""
        result = resample_trade_date(trade_dt_year, ("Q", "end"))
        assert 3 <= len(result) <= 5

    def test_invalid_rule_format_raises(self, trade_dt_year):
        """非 tuple 或长度 != 2 应 raise."""
        with pytest.raises(Exception):
            resample_trade_date(trade_dt_year, "M")
        with pytest.raises(Exception):
            resample_trade_date(trade_dt_year, ("M",))

    def test_invalid_mode_raises(self, trade_dt_year):
        """不支持的 mode 应 raise."""
        with pytest.raises(Exception):
            resample_trade_date(trade_dt_year, ("X", "end"))

    def test_invalid_position_raises(self, trade_dt_year):
        """不支持的 position 应 raise."""
        with pytest.raises(Exception):
            resample_trade_date(trade_dt_year, ("M", "middle"))

    def test_invalid_date_format_raises(self):
        """非 yyyymmdd int 应 raise."""
        bad = pd.DataFrame(["2026-01-01", "2026-01-02"])
        with pytest.raises(Exception):
            resample_trade_date(bad, ("M", "end"))


# ── offset_date ──


class TestOffsetDate:
    def test_d_mode_forward_one(self, trade_dt_year):
        """D mode, n=1: 输入日 → 下一交易日."""
        # 抓取 20250108 (周三), 下一日应是 20250109 (周四)
        result = offset_date([20250108], trade_dt_year, n=1, mode="D")
        assert result[0] == 20250109

    def test_d_mode_backward_one(self, trade_dt_year):
        """D mode, n=-1: 输入日 → 上一交易日."""
        result = offset_date([20250108], trade_dt_year, n=-1, mode="D")
        assert result[0] == 20250107

    def test_d_mode_zero_offset(self, trade_dt_year):
        """D mode, n=0: 返回当日."""
        result = offset_date([20250108], trade_dt_year, n=0, mode="D")
        assert result[0] == 20250108

    def test_invalid_mode_raises(self, trade_dt_year):
        """不支持的 mode 应 raise."""
        with pytest.raises(Exception):
            offset_date([20250108], trade_dt_year, n=1, mode="X")

    def test_if_modify_clips_to_bounds(self, trade_dt_year):
        """if_modify=True 时, 超出范围不应 raise (有 fallback).

        Note: 现有实现用 pandas iloc 负索引会 wrap-around 而非 IndexError,
        所以 if_modify 分支只在 iloc[idx+n] 真正越界 (大于 len) 时生效.
        这里只验证不 raise.
        """
        # 取末日 + 进 100 天 (真正越界)
        result = offset_date([20251229], trade_dt_year, n=100, mode="D", if_modify=True)
        assert result[0] is not None
        # 应被裁到末日 (idx + 100 > len)
        last_dt = trade_dt_year.iloc[-1, 0]
        assert result[0] == last_dt

    def test_d_mode_offset_two(self, trade_dt_year):
        """D mode, n=2: 跨 2 个交易日."""
        result = offset_date([20250108], trade_dt_year, n=2, mode="D")
        # 1-08 (Wed) → 1-10 (Fri)
        assert result[0] == 20250110

    def test_batch_input(self, trade_dt_year):
        """批量输入: 5 个日期同时 offset n=1."""
        inputs = [20250106, 20250107, 20250108, 20250109, 20250110]
        result = offset_date(inputs, trade_dt_year, n=1, mode="D")
        # 每个对应 next 工作日
        expected = [20250107, 20250108, 20250109, 20250110, 20250113]
        np.testing.assert_array_equal(result, expected)
