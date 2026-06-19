# coding: utf-8
"""IFinDDatabase constructor + routing unit tests."""

import pandas as pd
import pytest

from QuantNodes.research.factor_test.ifind_db import IFinDDatabase, IFindFetcherStub
from QuantNodes.research.factor_test.ifind_db.ifind_database import (
    _DEFAULT_INDUSTRY_MAP,
    _df_to_hdf_safe,
)


def make_db(**kwargs):
    return IFinDDatabase(
        date_beg=kwargs.pop("date_beg", "20260101"),
        date_end=kwargs.pop("date_end", "20260630"),
        universe=kwargs.pop("universe", "all"),
        fetcher=kwargs.pop("fetcher", IFindFetcherStub()),
        **kwargs,
    )


class TestDateDefaults:
    def test_explicit_dates(self):
        db = make_db(date_beg="20260101", date_end="20260630")
        assert db._date_beg == "20260101"
        assert db._date_end == "20260630"

    def test_empty_date_beg_uses_one_year_ago(self):
        db = IFinDDatabase(date_beg="", date_end="20260101", universe="all",
                          fetcher=IFindFetcherStub())
        assert len(db._date_beg) == 8 and db._date_beg.isdigit()

    def test_empty_date_end_uses_today(self):
        db = IFinDDatabase(date_beg="20250101", date_end="", universe="all",
                          fetcher=IFindFetcherStub())
        assert len(db._date_end) == 8 and db._date_end.isdigit()


class TestIndustryMap:
    def test_default_30_sw_industries(self):
        db = make_db()
        assert db._industry_map == _DEFAULT_INDUSTRY_MAP
        assert len(db._industry_map) == 30

    def test_custom_overrides_default(self):
        custom = {"自定义": 99}
        db = make_db(industry_map=custom)
        assert db._industry_map == custom

    def test_empty_map_accepted(self):
        db = make_db(industry_map={})
        assert db._industry_map == {}


class TestRouteTable:
    def test_route_table_contains_expected_keys(self):
        rt = IFinDDatabase._ROUTE_TABLE
        for k in [
            ("stk_daily.h5", "cp"),
            ("stk_daily.h5", "stklist"),
            ("stk_daily.h5", "trade_dt"),
            ("stk_daily.h5", "id_citic1"),
            ("stk_daily.h5", "mv_float"),
            ("stk_daily.h5", "st"),
            ("stk_daily.h5", "suspend"),
            ("stk_daily.h5", "ud_limit"),
            ("stk_daily.h5", "ipo_days"),
            ("stk_daily.h5", "id_300"),
            ("stk_daily.h5", "id_500"),
            ("index_daily.h5", "index_cp"),
            ("index_daily.h5", "indexlist"),
            ("index_daily.h5", "trade_dt"),
        ]:
            assert k in rt

    def test_load_h5_unknown_key(self):
        db = make_db()
        with pytest.raises(KeyError, match="未映射"):
            db.load_h5("unknown.h5", "x")


class TestGetAxis:
    def test_invalid_axis_type(self):
        db = make_db()
        with pytest.raises(ValueError, match="不支持的 axis_type"):
            db.get_axis("bond")


class TestLoadCustomNotImplemented:
    def test_raises_not_implemented(self):
        db = make_db()
        with pytest.raises(NotImplementedError):
            db.load_custom(("a", "b"))


class TestDfToHdfSafe:
    def test_int64_nullable_with_nan_to_float(self):
        df = pd.DataFrame({"x": pd.array([pd.NA, pd.NA], dtype="Int64")})
        out = _df_to_hdf_safe(df)
        assert out["x"].dtype.kind == "f"

    def test_int64_nullable_with_values_to_int(self):
        df = pd.DataFrame({"x": pd.array([1, pd.NA, 3], dtype="Int64")})
        out = _df_to_hdf_safe(df)
        assert out["x"].dtype == "int64"
        assert out["x"].iloc[1] == 0

    def test_pure_int_unchanged(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        out = _df_to_hdf_safe(df)
        assert out["x"].dtype == "int64"
        assert (out["x"] == [1, 2, 3]).all()

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": pd.array([1, 2], dtype="Int64")})
        before_dtype = df["x"].dtype
        _ = _df_to_hdf_safe(df)
        assert df["x"].dtype == before_dtype


class TestValidShape:
    def test_invalid_shape(self):
        db = make_db()
        db._stklist = pd.DataFrame({0: ["a", "b", "c"]})
        db._trade_dt = pd.DataFrame({0: [20260101, 20260102]})
        df = pd.DataFrame({"a": [1, 2]})
        assert db.valid_shape(df) is False

    def test_valid_shape(self):
        db = make_db()
        db._stklist = pd.DataFrame({0: ["a", "b"]})
        db._trade_dt = pd.DataFrame({0: [20260101, 20260102]})
        import numpy as np
        df = pd.DataFrame(np.zeros((2, 2)))
        assert db.valid_shape(df) is True


class TestAddIndex:
    def test_relabels_when_shape_matches(self):
        db = make_db()
        db._stklist = pd.DataFrame({0: ["X", "Y"]})
        db._trade_dt = pd.DataFrame({0: [20260101, 20260102]})
        import numpy as np
        f = pd.DataFrame(np.zeros((2, 2)))
        out = db.add_index(f)
        assert list(out.index) == [20260101, 20260102]
        assert list(out.columns) == ["X", "Y"]
