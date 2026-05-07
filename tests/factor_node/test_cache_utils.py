# coding=utf-8
"""QuantNodes.core.cache_utils 单元测试"""
import numpy as np

from QuantNodes.core.cache_utils import (
    create_std_data, create_empty_dataframe, _DummyLock
)


class TestCreateStdData:
    def test_creates_float_array_for_double(self):
        result = create_std_data(["dt1", "dt2"], ["id1", "id2", "id3"], "double")
        assert result.shape == (2, 3)
        assert result.dtype == "float"
        assert np.all(np.isnan(result))

    def test_creates_object_array_for_string(self):
        result = create_std_data(["dt1", "dt2"], ["id1", "id2"], "string")
        assert result.shape == (2, 2)
        assert result.dtype == "O"
        assert np.all(np.equal(result, None))

    def test_creates_object_array_for_object(self):
        result = create_std_data(["dt1"], ["id1"], "object")
        assert result.shape == (1, 1)
        assert result.dtype == "O"


class TestCreateEmptyDataframe:
    def test_creates_empty_df_with_index_and_columns(self):
        result = create_empty_dataframe(["dt1", "dt2"], ["id1", "id2"], "double")
        assert result.shape == (2, 2)
        assert list(result.index) == ["dt1", "dt2"]
        assert list(result.columns) == ["id1", "id2"]

    def test_creates_empty_df_without_index(self):
        result = create_empty_dataframe([], ["id1", "id2"], "double", include_index=False)
        assert result.shape == (0, 2)
        assert list(result.columns) == ["id1", "id2"]

    def test_creates_float_df_for_double(self):
        result = create_empty_dataframe(["dt1"], ["id1"], "double")
        assert result.dtypes.iloc[0] == np.float64


class TestDummyLock:
    def test_enter_returns_self(self):
        lock = _DummyLock()
        assert lock.__enter__() is lock

    def test_exit_does_nothing(self):
        lock = _DummyLock()
        lock.__exit__(None, None, None)

    def test_can_use_in_with_statement(self):
        lock = _DummyLock()
        with lock:
            pass
