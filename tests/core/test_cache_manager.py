# coding=utf-8
"""QuantNodes.core.cache_manager 单元测试"""
import pickle
import shutil
import tempfile
import os

import numpy as np
import pandas as pd

from QuantNodes.core.cache_manager import (
    ErgodicMode,
    OperationMode,
    save_raw_data,
)


class TestErgodicMode:
    def test_defaults(self):
        mode = ErgodicMode()
        assert mode.ForwardPeriod == 600
        assert mode.BackwardPeriod == 1
        assert mode.CacheMode == "因子"
        assert mode.MaxFactorCacheNum == 60
        assert mode.MaxIDCacheNum == 10000
        assert mode.CacheSize == 300
        assert mode.ErgodicDTs == []
        assert mode.ErgodicIDs == []

    def test_custom_values(self):
        mode = ErgodicMode(
            forward_period=100,
            backward_period=5,
            cache_mode="ID",
            max_factor_cache_num=30,
            max_id_cache_num=5000,
            cache_size=100,
            ergodic_dts=["2020-01-01"],
            ergodic_ids=["000001"],
        )
        assert mode.ForwardPeriod == 100
        assert mode.BackwardPeriod == 5
        assert mode.CacheMode == "ID"
        assert mode.ErgodicDTs == ["2020-01-01"]
        assert mode.ErgodicIDs == ["000001"]

    def test_pickle(self):
        mode = ErgodicMode()
        mode._CacheDataProcess = "test"
        data = pickle.dumps(mode)
        restored = pickle.loads(data)
        assert restored._CacheDataProcess is None
        assert restored.ForwardPeriod == 600

    def test_initial_state(self):
        mode = ErgodicMode()
        assert mode._isStarted is False
        assert mode._CurDT is None
        assert mode._CacheData is None


class TestOperationMode:
    def test_defaults(self):
        mode = OperationMode()
        assert mode._FT is None
        assert mode._isStarted is False
        assert mode._Factors == []
        assert mode.SubProcessNum == 0
        assert mode.FactorNames == []

    def test_custom_values(self):
        mode = OperationMode(ft="mock_ft", factor_names=["a", "b"], subprocess_num=4)
        assert mode._FT == "mock_ft"
        assert mode.FactorNames == ["a", "b"]
        assert mode.SubProcessNum == 4

    def test_file_suffix(self):
        mode = OperationMode()
        # get_shelve_file_suffix() returns ".db", then code prepends another "."
        assert mode.FileSuffix == "..db"

    def test_pickle(self):
        mode = OperationMode()
        data = pickle.dumps(mode)
        restored = pickle.loads(data)
        assert restored.SubProcessNum == 0


class TestSaveRawData:
    def test_none_data(self):
        result = save_raw_data(None, [], "/tmp", {}, "test", {})
        assert result == 0

    def test_with_id_column(self):
        tmpdir = tempfile.mkdtemp()
        try:
            pid_dir = os.path.join(tmpdir, "0")
            os.makedirs(pid_dir, exist_ok=True)
            df = pd.DataFrame({"ID": ["a", "b", "c"], "val": [1, 2, 3]})
            pid_ids = {"0": ["a", "b", "c"]}
            result = save_raw_data(df, ["val"], tmpdir, pid_ids, "raw", {"0": None})
            assert result == 0
        finally:
            shutil.rmtree(tmpdir)

    def test_without_id_column(self):
        tmpdir = tempfile.mkdtemp()
        try:
            pid_dir = os.path.join(tmpdir, "0")
            os.makedirs(pid_dir, exist_ok=True)
            data = np.array([1, 2, 3])
            pid_ids = {"0": ["a"]}
            result = save_raw_data(data, [], tmpdir, pid_ids, "raw", {"0": None})
            assert result == 0
        finally:
            shutil.rmtree(tmpdir)
