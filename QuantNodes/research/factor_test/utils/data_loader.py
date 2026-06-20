# coding: utf-8
"""统一数据加载器 / Unified Data Loader

Migrated from ~/Public/单因子回测/factor_utils.py (Factor class)
Security: all exec()/eval() eliminated.
"""

import pandas as pd
import numpy as np


class DataLoader:
    """统一数据加载接口, 支持 H5/CSV/NPY/Parquet"""

    def __init__(self, api_path: str = './testdata/test_h5_new/'):
        self.api = api_path if api_path.endswith('/') else api_path + '/'

    def load_h5(self, filename: str, key: str) -> pd.DataFrame:
        """从 H5 文件加载数据"""
        path = self.api + filename
        h5_store = pd.HDFStore(path, mode='r')
        try:
            # 标准化 key: HDFStore 自动加 / 前缀
            norm_key = key if key.startswith('/') else '/' + key
            if norm_key in h5_store.keys():
                return h5_store.get(norm_key)
            elif key in h5_store.keys():
                return h5_store.get(key)
            else:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {h5_store.keys()}")
        finally:
            h5_store.close()

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
            h5_store = pd.HDFStore(dir_path, mode='r')
            try:
                return h5_store.get(filename)
            finally:
                h5_store.close()
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
        """获取轴数据"""
        if axis_type == 'stock':
            return self.get_stock_axis()
        elif axis_type == 'index':
            return self.get_index_axis()
        else:
            raise ValueError(f"不支持的 axis_type: {axis_type}")

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
