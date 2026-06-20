# coding: utf-8
"""统一数据加载器 / Unified Data Loader

Migrated from ~/Public/单因子回测/factor_utils.py (Factor class)
Security: all exec()/eval() eliminated.

Phase H3 (2026-06-20):
  - HDF5 stores are cached per file path in self._h5_stores. Previously
    every load_h5() opened + closed a new HDFStore; now stores stay open
    for the loader lifetime. Read-mode HDFStore is thread-safe.
  - Axis data (stock + index) cached after first get_axis() call.
    add_index() calls get_axis() on every load (10+ per pipeline run),
    previously 2 H5 opens per call.
"""

import pandas as pd
import numpy as np


class DataLoader:
    """统一数据加载接口, 支持 H5/CSV/NPY/Parquet"""

    def __init__(self, api_path: str = './testdata/test_h5_new/'):
        self.api = api_path if api_path.endswith('/') else api_path + '/'
        self._h5_stores: dict[str, pd.HDFStore] = {}
        self._axis_cache: dict[str, tuple] = {}

    def _get_store(self, path: str) -> pd.HDFStore:
        """Get or lazily open a read-mode HDFStore for `path`."""
        store = self._h5_stores.get(path)
        if store is None:
            store = pd.HDFStore(path, mode='r')
            self._h5_stores[path] = store
        return store

    def close(self) -> None:
        """Close all cached HDFStores. Call at end of pipeline run."""
        for store in self._h5_stores.values():
            try:
                store.close()
            except Exception:
                pass
        self._h5_stores.clear()
        self._axis_cache.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def load_h5(self, filename: str, key: str) -> pd.DataFrame:
        """从 H5 文件加载数据 (H3: 复用 cached HDFStore)"""
        path = self.api + filename
        store = self._get_store(path)
        # 标准化 key: HDFStore 自动加 / 前缀
        norm_key = key if key.startswith('/') else '/' + key
        if norm_key in store.keys():
            return store.get(norm_key)
        if key in store.keys():
            return store.get(key)
        raise KeyError(f"Key '{key}' not found in {path}. Available: {store.keys()}")

    def load_csv(self, path: str) -> pd.DataFrame:
        """从 CSV 加载数据"""
        return pd.read_csv(path, index_col=0)

    def load_npy(self, path: str) -> pd.DataFrame:
        """从 NPY 加载数据"""
        return pd.DataFrame(np.load(path, allow_pickle=True))

    def load_parquet(self, path: str) -> pd.DataFrame:
        """从 Parquet 加载数据"""
        return pd.read_parquet(path)

    def load_custom(self, data_dir: tuple) -> pd.DataFrame:
        """从自定义路径加载因子 (H5/CSV/NPY)"""
        dir_path, filename = data_dir
        if filename.endswith('.csv'):
            if dir_path.endswith('/') or dir_path.endswith('\\'):
                return self.load_csv(dir_path + filename)
            else:
                # filename itself is a full path
                return self.load_csv(filename)
        elif filename.endswith('.npy'):
            return self.load_npy(dir_path + filename)
        elif dir_path.endswith('.h5'):
            store = self._get_store(dir_path)
            return store.get(filename)
        else:
            raise ValueError(f"不支持的数据格式: {dir_path}, {filename}")

    def load_factor(self, factor_dir: str, factor_name: str) -> pd.DataFrame:
        """加载因子数据 (统一入口)"""
        if factor_dir.endswith('.h5'):
            return self.load_h5(factor_dir, factor_name)
        elif factor_dir.endswith('.csv'):
            return self.load_csv(factor_dir)
        elif factor_dir.endswith('.npy'):
            return self.load_npy(factor_dir)
        else:
            return self.load_custom((factor_dir, factor_name))

    def get_stock_axis(self) -> tuple:
        """获取股票列表和交易日序列"""
        stklist = self.load_h5('stk_daily.h5', 'stklist')
        trade_dt = self.load_h5('stk_daily.h5', 'trade_dt')
        return stklist, trade_dt

    def get_index_axis(self) -> tuple:
        """获取指数列表和交易日序列"""
        indexlist = self.load_h5('index_daily.h5', 'indexlist')
        trade_dt = self.load_h5('index_daily.h5', 'trade_dt')
        return indexlist, trade_dt

    def get_axis(self, axis_type: str = 'stock') -> tuple:
        """获取轴数据 (H3: cached after first call)"""
        if axis_type in self._axis_cache:
            return self._axis_cache[axis_type]
        if axis_type == 'stock':
            result = self.get_stock_axis()
        elif axis_type == 'index':
            result = self.get_index_axis()
        else:
            raise ValueError(f"不支持的 axis_type: {axis_type}")
        self._axis_cache[axis_type] = result
        return result

    def add_index(self, factor: pd.DataFrame, axis_type: str = 'stock') -> pd.DataFrame:
        """给因子添加标准索引 (行=交易日, 列=股票/指数)"""
        factor = factor.copy()
        assetlist, trade_dt = self.get_axis(axis_type)
        factor.index = trade_dt.iloc[:, 0].values
        factor.columns = assetlist.iloc[:, 0].values
        return factor

    def valid_shape(self, factor: pd.DataFrame, axis_type: str = 'stock') -> bool:
        """检查因子 shape 是否为标准矩阵"""
        assetlist, trade_dt = self.get_axis(axis_type)
        expected = (len(trade_dt), len(assetlist))
        return factor.shape == expected
