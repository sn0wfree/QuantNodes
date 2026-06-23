# coding: utf-8
"""文件格式 Adapter (Phase 3.3)

将 ``DataLoader`` 按扩展名硬编码的读取逻辑抽成 Adapter 族 + registry,
新增格式只需加一个 ``FileFormatLoader`` 子类并注册, 无需改 DataLoader。

每个 adapter 暴露统一的 ``load(path, *, key=None, store_getter=None)`` 签名:
  - 扁平文件 (csv/npy/parquet): 只用 ``path``, 忽略 key/store_getter
  - H5: 用 ``store_getter(path)`` 取得 (可缓存的) HDFStore, 按 ``key``
    查找 (与 DataLoader.load_h5 的 key 归一化 fallback 一致)

读取行为与重构前的 ``DataLoader.load_*`` bitwise 等价。
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from QuantNodes.core.data_source import DataSource


class FileFormatLoader(DataSource):
    """单一文件格式的读取适配器。

    Attributes:
        extensions: 该 adapter 负责的扩展名 (含点, 小写), 如 ``('.csv',)``。
    """

    extensions: tuple = ()

    @abstractmethod
    def load(
        self,
        path: str,
        *,
        key: Optional[str] = None,
        store_getter: Optional[Callable[[str], pd.HDFStore]] = None,
    ) -> pd.DataFrame:
        """读取 ``path`` 处的数据为 DataFrame。

        Args:
            path: 文件路径 (H5 为文件路径, 其余为完整路径)。
            key: H5 数据集 key (仅 H5 使用)。
            store_getter: 返回可缓存 HDFStore 的回调 (仅 H5 使用),
                用于复用 DataLoader 的 ``_h5_stores`` 缓存。
        """
        raise NotImplementedError

    def close(self) -> None:
        """adapter 自身无状态; H5Store 生命周期由 DataLoader 管理。"""
        return None


class CSVLoader(FileFormatLoader):
    extensions = ('.csv',)

    def load(self, path, *, key=None, store_getter=None):
        return pd.read_csv(path, index_col=0)


class NPYLoader(FileFormatLoader):
    extensions = ('.npy',)

    def load(self, path, *, key=None, store_getter=None):
        return pd.DataFrame(np.load(path, allow_pickle=True))


class ParquetLoader(FileFormatLoader):
    extensions = ('.parquet',)

    def load(self, path, *, key=None, store_getter=None):
        return pd.read_parquet(path)


class H5Loader(FileFormatLoader):
    extensions = ('.h5',)

    def load(self, path, *, key=None, store_getter=None):
        if store_getter is None:
            raise ValueError("H5Loader.load requires a store_getter callback")
        store = store_getter(path)
        # 标准化 key: HDFStore 自动加 / 前缀
        norm_key = key if key.startswith('/') else '/' + key
        if norm_key in store.keys():
            return store.get(norm_key)
        if key in store.keys():
            return store.get(key)
        raise KeyError(
            f"Key '{key}' not found in {path}. Available: {store.keys()}"
        )


_FILE_LOADERS: Dict[str, FileFormatLoader] = {}


def _register_default_loaders() -> None:
    for cls in (H5Loader, CSVLoader, NPYLoader, ParquetLoader):
        loader = cls()
        for ext in loader.extensions:
            _FILE_LOADERS[ext] = loader


_register_default_loaders()


def build_file_loader(ext: str) -> FileFormatLoader:
    """按扩展名 (含点) 返回对应的 FileFormatLoader 实例。

    Args:
        ext: 文件扩展名, 含点, 如 ``'.csv'``。

    Returns:
        FileFormatLoader 实例。

    Raises:
        ValueError: 扩展名未注册。
    """
    loader = _FILE_LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file extension: {ext}. "
            f"Available: {sorted(_FILE_LOADERS)}"
        )
    return loader


def register_file_loader(loader: FileFormatLoader) -> None:
    """注册一个新的文件格式 adapter (供扩展)。

    Args:
        loader: FileFormatLoader 实例, 其 ``extensions`` 决定负责的扩展名。

    Raises:
        ValueError: extensions 为空或某扩展名已注册。
    """
    if not loader.extensions:
        raise ValueError("loader.extensions must be non-empty")
    for ext in loader.extensions:
        if ext in _FILE_LOADERS:
            raise ValueError(f"extension '{ext}' already registered")
    for ext in loader.extensions:
        _FILE_LOADERS[ext] = loader


def available_extensions() -> list:
    """返回已注册的扩展名列表 (排序)。"""
    return sorted(_FILE_LOADERS)
