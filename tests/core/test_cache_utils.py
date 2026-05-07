# -*- coding: utf-8 -*-
"""QuantNodes.core.cache_utils 单元测试"""
import numpy as np
import pandas as pd

from QuantNodes.core.cache_utils import (
    create_std_data,
    create_empty_dataframe,
    _DummyLock,
)


class TestCreateStdData:
    def test_create_double_array(self):
        dts = ['2024-01-01', '2024-01-02', '2024-01-03']
        ids = ['A', 'B']
        result = create_std_data(dts, ids, 'double')
        assert result.shape == (3, 2)
        assert result.dtype == np.float64
        assert np.all(np.isnan(result))

    def test_create_string_array(self):
        dts = ['2024-01-01', '2024-01-02']
        ids = ['X', 'Y', 'Z']
        result = create_std_data(dts, ids, 'string')
        assert result.shape == (2, 3)
        assert result.dtype == np.object_
        assert result[0, 0] is None

    def test_create_object_array(self):
        dts = ['2024-01-01']
        ids = ['obj1']
        result = create_std_data(dts, ids, 'object')
        assert result.shape == (1, 1)
        assert result.dtype == np.object_
        assert result[0, 0] is None

    def test_empty_lists(self):
        result = create_std_data([], [], 'double')
        assert result.shape == (0, 0)

    def test_single_element(self):
        result = create_std_data(['dt1'], ['id1'], 'double')
        assert result.shape == (1, 1)
        assert np.isnan(result[0, 0])


class TestCreateEmptyDataframe:
    def test_create_double_dataframe(self):
        dts = ['2024-01-01', '2024-01-02']
        ids = ['A', 'B', 'C']
        df = create_empty_dataframe(dts, ids, 'double')
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert df.index.tolist() == dts
        assert df.columns.tolist() == ids
        assert df.isnull().all().all()

    def test_create_string_dataframe(self):
        dts = ['2024-01-01']
        ids = ['X']
        df = create_empty_dataframe(dts, ids, 'string')
        assert df.shape == (1, 1)
        import pandas as pd
        assert pd.isna(df['X']['2024-01-01'])

    def test_include_index_false(self):
        dts = ['2024-01-01', '2024-01-02']
        ids = ['A', 'B']
        df = create_empty_dataframe(dts, ids, 'double', include_index=False)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (0, 2)
        assert df.columns.tolist() == ids

    def test_empty_dataframe(self):
        df = create_empty_dataframe([], ['A', 'B'], 'double', include_index=False)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (0, 2)


class TestDummyLock:
    def test_context_manager_enter(self):
        lock = _DummyLock()
        with lock as result:
            assert result is lock

    def test_context_manager_exit(self):
        lock = _DummyLock()
        with lock:
            pass

    def test_can_be_used_in_shelve_context(self):
        lock = _DummyLock()
        with lock:
            x = 1
            assert x == 1


class TestCacheUtilsIntegration:
    def test_create_and_fill_data(self):
        dts = ['2024-01-01', '2024-01-02']
        ids = ['A', 'B']
        arr = create_std_data(dts, ids, 'double')
        arr[0, 0] = 1.5
        assert arr[0, 0] == 1.5
        assert np.isnan(arr[0, 1])

    def test_empty_df_empty_ids(self):
        df = create_empty_dataframe([], [], 'double', include_index=False)
        assert df.shape == (0, 0)
        assert df.empty
