# coding: utf-8
"""文件格式 Adapter 测试 (Phase 3.3)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.data_source import DataSource
from QuantNodes.research.factor_test.utils.file_loaders import (
    FileFormatLoader,
    CSVLoader,
    NPYLoader,
    ParquetLoader,
    H5Loader,
    build_file_loader,
    register_file_loader,
    available_extensions,
    _FILE_LOADERS,
)


@pytest.fixture
def restore_registry():
    snapshot = dict(_FILE_LOADERS)
    yield
    _FILE_LOADERS.clear()
    _FILE_LOADERS.update(snapshot)


class TestAdapters:
    def test_csv_loader(self, tmp_path):
        p = tmp_path / "f.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(p)
        df = CSVLoader().load(str(p))
        assert df.shape == (3, 1)

    def test_npy_loader(self, tmp_path):
        p = tmp_path / "f.npy"
        np.save(p, np.arange(6).reshape(3, 2))
        df = NPYLoader().load(str(p))
        assert df.shape == (3, 2)

    def test_parquet_loader(self, tmp_path):
        p = tmp_path / "f.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(p)
        df = ParquetLoader().load(str(p))
        assert df.shape == (2, 1)

    def test_h5_loader(self, tmp_path):
        p = tmp_path / "f.h5"
        with pd.HDFStore(str(p), mode="w") as store:
            store.put("data", pd.DataFrame({"a": [1, 2, 3]}), format="table")

        opened = pd.HDFStore(str(p), mode="r")
        try:
            df = H5Loader().load(str(p), key="data", store_getter=lambda _p: opened)
            assert df.shape == (3, 1)
        finally:
            opened.close()

    def test_h5_loader_requires_store_getter(self, tmp_path):
        with pytest.raises(ValueError, match="store_getter"):
            H5Loader().load("x.h5", key="data")

    def test_h5_loader_missing_key_raises(self, tmp_path):
        p = tmp_path / "f.h5"
        with pd.HDFStore(str(p), mode="w") as store:
            store.put("data", pd.DataFrame({"a": [1]}), format="table")

        opened = pd.HDFStore(str(p), mode="r")
        try:
            with pytest.raises(KeyError, match="not found"):
                H5Loader().load(
                    str(p), key="missing", store_getter=lambda _p: opened
                )
        finally:
            opened.close()

    def test_h5_store_getter_reuse(self, tmp_path):
        """store_getter 回调被调用且复用同一 store (Phase H3 缓存语义)。"""
        p = tmp_path / "f.h5"
        with pd.HDFStore(str(p), mode="w") as store:
            store.put("data", pd.DataFrame({"a": [1]}), format="table")
        calls = []
        opened = pd.HDFStore(str(p), mode="r")

        def getter(path):
            calls.append(path)
            return opened

        loader = H5Loader()
        loader.load(str(p), key="data", store_getter=getter)
        loader.load(str(p), key="data", store_getter=getter)
        assert len(calls) == 2
        opened.close()


class TestRegistry:
    def test_build_csv(self):
        assert isinstance(build_file_loader(".csv"), CSVLoader)

    def test_build_h5(self):
        assert isinstance(build_file_loader(".h5"), H5Loader)

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unsupported file extension"):
            build_file_loader(".xlsx")

    def test_available_extensions(self):
        exts = available_extensions()
        assert set(exts) == {".h5", ".csv", ".npy", ".parquet"}

    def test_register_new(self, restore_registry):
        class JSONLoader(FileFormatLoader):
            extensions = (".json",)

            def load(self, path, *, key=None, store_getter=None):
                return pd.read_json(path)

        register_file_loader(JSONLoader())
        assert ".json" in available_extensions()
        assert isinstance(build_file_loader(".json"), JSONLoader)

    def test_register_duplicate_raises(self, restore_registry):
        class Dup(FileFormatLoader):
            extensions = (".csv",)

            def load(self, path, *, key=None, store_getter=None):
                return pd.DataFrame()

        with pytest.raises(ValueError, match="already registered"):
            register_file_loader(Dup())

    def test_register_empty_extensions_raises(self, restore_registry):
        class Empty(FileFormatLoader):
            extensions = ()

            def load(self, path, *, key=None, store_getter=None):
                return pd.DataFrame()

        with pytest.raises(ValueError, match="non-empty"):
            register_file_loader(Empty())


class TestDataSourceRelationship:
    def test_loader_is_datasource(self):
        assert isinstance(CSVLoader(), DataSource)

    def test_fileformatloader_subclass(self):
        assert issubclass(FileFormatLoader, DataSource)

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            FileFormatLoader()
