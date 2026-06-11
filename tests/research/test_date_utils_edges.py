"""date_utils.py 边界条件测试 (20 tests)。

覆盖: all 7 public functions + internal edge cases。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.date_utils import (
    chg_idx_to_datestr,
    datenum_to_datetime,
    datetime_to_datenum,
    get_adjust_date,
    offset_date,
    resample_trade_date,
    valid_date,
)


class TestValidDate:
    def test_valid_int64_frame(self):
        t = pd.DataFrame([20250103])
        t = t.astype("int64")
        assert valid_date(t) is True

    def test_valid_int64_series(self):
        t = pd.Series([20250103], dtype="int64")
        assert valid_date(t) is True

    def test_wrong_length(self):
        t = pd.DataFrame([202501])  # 6 digits
        assert valid_date(t) is False

    def test_wrong_type(self):
        t = pd.DataFrame(["20250103"])
        assert valid_date(t) is False

    def test_empty_index_raises(self):
        t = pd.DataFrame({0: pd.Series(dtype="int64")})
        with pytest.raises(IndexError):
            valid_date(t)

    def test_invalid_input_type(self):
        assert valid_date([20250103]) is False
        assert valid_date(20250103) is False


class TestDatenumConversion:
    def test_datenum_to_datetime_basic(self):
        t = pd.DataFrame([20250103, 20250104])
        dt = datenum_to_datetime(t)
        assert dt.iloc[0, 0].year == 2025

    def test_datenum_to_datetime_series(self):
        s = pd.Series([20250103])
        dt = datenum_to_datetime(s)
        assert isinstance(dt, pd.DataFrame)

    def test_datetime_to_datenum(self):
        dt = pd.DataFrame([pd.Timestamp("2025-01-03")])
        t = datetime_to_datenum(dt)
        assert t.iloc[0, 0] == 20250103


class TestChgIdx:
    def test_basic(self):
        s = pd.Series([1, 2], index=[20250103, 20250104])
        out = chg_idx_to_datestr(s)
        assert out.index[0] == "2025/01/03"

    def test_empty(self):
        s = pd.Series([], dtype=float)
        out = chg_idx_to_datestr(s)
        assert len(out) == 0


class TestResample:
    def test_month_end(self):
        t = pd.DataFrame({"a": [20250101, 20250115, 20250131, 20250201]})
        t = t.astype("int64")
        out = resample_trade_date(t, ("M", "end"))
        assert len(out) == 2
        assert out.iloc[0, 0] == 20250131

    def test_month_begin(self):
        t = pd.DataFrame({"a": [20250102, 20250115, 20250201, 20250210]})
        t = t.astype("int64")
        out = resample_trade_date(t, ("M", "begin"))
        # begin 模式包含样本首日作为第一个 period begin
        assert out.iloc[0, 0] == 20250102

    def test_weekly(self):
        t = pd.DataFrame({"a": [20250106, 20250107, 20250113, 20250114]})
        t = t.astype("int64")
        out = resample_trade_date(t, ("W", "end"))
        assert len(out) > 0

    def test_quarterly(self):
        t = pd.DataFrame({"a": [20250101, 20250331, 20250401, 20250630]})
        t = t.astype("int64")
        out = resample_trade_date(t, ("Q", "end"))
        assert len(out) >= 2

    def test_bad_rule_format(self):
        t = pd.DataFrame([20250103]).astype("int64")
        with pytest.raises(ValueError):
            resample_trade_date(t, "M,begin")

    def test_bad_mode(self):
        t = pd.DataFrame([20250103]).astype("int64")
        with pytest.raises(ValueError):
            resample_trade_date(t, ("Y", "end"))

    def test_bad_position(self):
        t = pd.DataFrame([20250103]).astype("int64")
        with pytest.raises(ValueError):
            resample_trade_date(t, ("M", "middle"))


class TestGetAdjustDate:
    def _trade_dt(self):
        t = pd.DataFrame(list(range(20250101, 20250132)))
        return t.astype("int64")

    def test_month_end(self):
        t = self._trade_dt()
        out = get_adjust_date(t, 20250101, 20250131, ("M", "end"))
        assert out.iloc[-1, 0] <= 20250131

    def test_daily(self):
        t = self._trade_dt()
        out = get_adjust_date(t, 20250101, 20250105, ("D", 1))
        assert len(out) <= 5

    def test_custom(self):
        t = self._trade_dt()
        custom = pd.DataFrame([20250115]).astype("int64")
        out = get_adjust_date(t, 20250101, 20250131, ("custom", custom))
        assert out.iloc[0, 0] == 20250115

    def test_beg_end_out_of_range(self):
        t = self._trade_dt()
        with pytest.raises(ValueError):
            get_adjust_date(t, 20300101, 20300131, ("M", "end"))

    def test_bad_rule(self):
        t = self._trade_dt()
        with pytest.raises(ValueError):
            get_adjust_date(t, 20250101, 20250131, ("Y", "end"))

    def test_bad_custom(self):
        t = self._trade_dt()
        custom = pd.DataFrame([2025]).astype("int64")
        with pytest.raises(ValueError):
            get_adjust_date(t, 20250101, 20250131, ("custom", custom))


class TestOffsetDate:
    def _trade(self):
        return pd.DataFrame(list(range(20250101, 20250132))).astype("int64")

    def test_offset_daily(self):
        t = self._trade()
        out = offset_date(pd.Series([20250105]), t, 1)
        assert out[0] == 20250106

    def test_offset_monthly_before_first_period_raises(self):
        t = self._trade()
        # 输入日期早于首个月末调仓日, 当前实现会因无 <=x 的月末日期而 IndexError
        with pytest.raises(IndexError):
            offset_date(pd.Series([20250105]), t, 1, mode="M")

    def test_offset_if_modify_clamp(self):
        t = self._trade()
        out = offset_date(pd.Series([20250131]), t, 1, mode="D", if_modify=True)
        assert out[0] == 20250131

    def test_offset_bad_mode(self):
        t = self._trade()
        with pytest.raises(ValueError):
            offset_date(pd.Series([20250105]), t, 1, mode="Y")