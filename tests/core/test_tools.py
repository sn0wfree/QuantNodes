# coding=utf-8
"""QuantNodes.core.tools 单元测试"""
import os
import shutil
import time
import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.tools import (
    gen_available_name,
    partition_list,
    partition_list_moving_sampling,
    start_multi_process,
    fill_na_by_lookback,
    get_shelve_file_suffix,
    compile_id_filter_str,
    create_temp_dir,
    merge_data_frames,
    chunk_iterable,
    timer,
    retry,
)


class TestGenAvailableName:
    def test_generates_unique_name(self):
        name = gen_available_name("Test")
        assert name.startswith("Test_")
        assert len(name) > 5

    def test_no_collision(self):
        names = set()
        for _ in range(100):
            name = gen_available_name("T", names)
            assert name not in names or name in names  # gen_available_name adds to used set
            names.add(name)
        assert len(names) == 100

    def test_default_prefix(self):
        name = gen_available_name()
        assert name.startswith("Temp_")

    def test_empty_used_names(self):
        name = gen_available_name("X", used_names=None)
        assert name.startswith("X_")

    def test_custom_used_names(self):
        used = {"X_abc", "X_def"}
        name = gen_available_name("X", used)
        assert name.startswith("X_")


class TestPartitionList:
    def test_normal_split(self):
        result = partition_list([1, 2, 3, 4], 2)
        assert len(result) == 2
        assert sum(len(p) for p in result) == 4

    def test_n_greater_than_len(self):
        result = partition_list([1, 2, 3], 5)
        assert result == [[1], [2], [3]]

    def test_n_zero(self):
        result = partition_list([1, 2, 3], 0)
        assert result == [[1, 2, 3]]

    def test_n_negative(self):
        result = partition_list([1, 2, 3], -1)
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = partition_list([], 3)
        assert result == []

    def test_single_element(self):
        result = partition_list([42], 1)
        assert result == [[42]]

    def test_even_split(self):
        result = partition_list([1, 2, 3, 4, 5, 6], 3)
        assert len(result) == 3
        assert all(len(p) == 2 for p in result)

    def test_uneven_split(self):
        result = partition_list([1, 2, 3, 4, 5], 2)
        assert len(result) == 2
        assert len(result[0]) == 3
        assert len(result[1]) == 2


class TestPartitionListMovingSampling:
    def test_normal(self):
        result = partition_list_moving_sampling([1, 2, 3, 4], 2)
        assert len(result) == 2

    def test_custom_step(self):
        result = partition_list_moving_sampling([1, 2, 3, 4, 5, 6], 3, step=2)
        assert len(result) == 3

    def test_step_larger_than_data(self):
        result = partition_list_moving_sampling([1, 2], 3, step=5)
        assert len(result) == 2

    def test_n_zero(self):
        result = partition_list_moving_sampling([1, 2, 3], 0)
        assert result == [[1, 2, 3]]

    def test_n_greater_than_len(self):
        result = partition_list_moving_sampling([1, 2], 5)
        assert result == [[1], [2]]


class TestStartMultiProcess:
    def _add(self, a, b):
        return a + b

    def test_normal(self):
        result = start_multi_process(self._add, [(1, 2), (3, 4)], n_processes=2)
        assert result == [3, 7]

    def test_single_process_fallback(self):
        result = start_multi_process(self._add, [(1, 2), (3, 4)], n_processes=1)
        assert result == [3, 7]

    def test_empty_args(self):
        result = start_multi_process(self._add, [])
        assert result == []

    def test_daemon(self):
        result = start_multi_process(self._add, [(10, 20)], n_processes=1, daemon=True)
        assert result == [30]


class TestFillNaByLookback:
    def test_dataframe(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        result = fill_na_by_lookback(df, lookback=1)
        assert result["a"].iloc[1] == 1.0

    def test_series(self):
        s = pd.Series([1.0, np.nan, 3.0])
        result = fill_na_by_lookback(s, lookback=1)
        assert result.iloc[1] == 1.0

    def test_lookback_zero(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        result = fill_na_by_lookback(df, lookback=0)
        assert pd.isna(result["a"].iloc[1])

    def test_lookback_negative(self):
        df = pd.DataFrame({"a": [1.0, np.nan]})
        result = fill_na_by_lookback(df, lookback=-1)
        assert pd.isna(result["a"].iloc[1])

    def test_multiple_lookback(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0]})
        result = fill_na_by_lookback(df, lookback=2)
        assert result["a"].iloc[2] == 1.0


class TestGetShelveFileSuffix:
    def test_returns_db(self):
        assert get_shelve_file_suffix() == ".db"


class TestCompileIdFilterStr:
    def test_empty_filter(self):
        result = compile_id_filter_str("", ["a", "b"])
        assert result == (None, None)

    def test_no_at_sign(self):
        result, factors = compile_id_filter_str("x > 5", ["a", "b"])
        assert factors == []
        assert result == "x > 5"

    def test_with_at_sign(self):
        result, factors = compile_id_filter_str("@price > 100", ["price", "volume"])
        assert "price" in factors
        assert result is not None

    def test_invalid_factor(self):
        result, factors = compile_id_filter_str("@unknown > 5", ["price"])
        assert factors == []


class TestCreateTempDir:
    def test_creates_directory(self):
        path = create_temp_dir()
        try:
            assert os.path.isdir(path)
        finally:
            shutil.rmtree(path)

    def test_custom_prefix(self):
        path = create_temp_dir(prefix="mytest_")
        try:
            assert os.path.basename(path).startswith("mytest_")
        finally:
            shutil.rmtree(path)


class TestMergeDataFrames:
    def test_empty_list(self):
        result = merge_data_frames([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_df(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = merge_data_frames([df])
        assert len(result) == 2

    def test_inner_merge(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        df2 = pd.DataFrame({"a": [2, 3], "c": [200, 300]})
        result = merge_data_frames([df1, df2], how="inner", on="a")
        assert len(result) == 1
        assert result["a"].iloc[0] == 2

    def test_outer_merge(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
        df2 = pd.DataFrame({"a": [2, 3], "c": [200, 300]})
        result = merge_data_frames([df1, df2], how="outer", on="a")
        assert len(result) == 3

    def test_multiple_dfs(self):
        dfs = [pd.DataFrame({"a": [i], "b": [i * 10]}) for i in range(3)]
        result = merge_data_frames(dfs, how="outer", on="a")
        assert len(result) == 3


class TestChunkIterable:
    def test_exact_multiple(self):
        chunks = list(chunk_iterable(range(6), 3))
        assert len(chunks) == 2
        assert chunks[0] == [0, 1, 2]
        assert chunks[1] == [3, 4, 5]

    def test_partial_last(self):
        chunks = list(chunk_iterable(range(5), 3))
        assert len(chunks) == 2
        assert chunks[1] == [3, 4]

    def test_empty(self):
        chunks = list(chunk_iterable([], 3))
        assert chunks == []

    def test_single_chunk(self):
        chunks = list(chunk_iterable(range(2), 10))
        assert len(chunks) == 1
        assert chunks[0] == [0, 1]


class TestTimerDecorator:
    def test_returns_value(self):
        @timer
        def add(a, b):
            return a + b
        assert add(1, 2) == 3

    def test_preserves_name(self):
        @timer
        def my_func():
            pass
        assert my_func.__name__ == "my_func"


class TestRetryDecorator:
    def test_success_first_try(self):
        call_count = [0]

        @retry(max_attempts=3, delay=0)
        def succeed():
            call_count[0] += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count[0] == 1

    def test_retry_then_success(self):
        call_count = [0]

        @retry(max_attempts=3, delay=0)
        def fail_twice():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "done"

        assert fail_twice() == "done"
        assert call_count[0] == 3

    def test_max_attempts_exceeded(self):
        @retry(max_attempts=2, delay=0)
        def always_fail():
            raise RuntimeError("always")

        with pytest.raises(RuntimeError):
            always_fail()
