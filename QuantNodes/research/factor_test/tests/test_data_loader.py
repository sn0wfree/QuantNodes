# coding: utf-8
"""DataLoader 单元测试 - Mock 文件系统"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.research.factor_test.utils.data_loader import DataLoader


class TestDataLoaderInit:
    """测试 DataLoader 初始化"""

    def test_default_path(self):
        loader = DataLoader()
        assert loader.api == './testdata/test_h5_new/'

    def test_trailing_slash_added(self):
        loader = DataLoader('/some/path')
        assert loader.api == '/some/path/'

    def test_trailing_slash_preserved(self):
        loader = DataLoader('/some/path/')
        assert loader.api == '/some/path/'


class TestDataLoaderLoadCSV:
    """测试 CSV 加载"""

    def test_load_csv(self, tmp_path):
        csv_file = tmp_path / 'test.csv'
        csv_file.write_text('idx,col1,col2\n0,1,2\n1,3,4\n')
        loader = DataLoader()
        result = loader.load_csv(str(csv_file))
        assert isinstance(result, pd.DataFrame)
        # index_col=0 把第一列作为 index, 剩余列为 columns
        assert result.shape == (2, 2)
        assert list(result.columns) == ['col1', 'col2']


class TestDataLoaderLoadNPY:
    """测试 NPY 加载"""

    def test_load_npy(self, tmp_path):
        npy_file = tmp_path / 'test.npy'
        arr = np.array([[1, 2], [3, 4]])
        np.save(npy_file, arr)
        loader = DataLoader()
        result = loader.load_npy(str(npy_file))
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 2)


class TestDataLoaderLoadParquet:
    """测试 Parquet 加载"""

    def test_load_parquet(self, tmp_path):
        pq_file = tmp_path / 'test.parquet'
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df.to_parquet(pq_file)
        loader = DataLoader()
        result = loader.load_parquet(str(pq_file))
        pd.testing.assert_frame_equal(result, df)


class TestDataLoaderLoadH5:
    """测试 H5 加载"""

    def test_load_h5_success(self, tmp_path):
        h5_file = tmp_path / 'test.h5'
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        df.to_hdf(h5_file, key='mykey', mode='w')
        loader = DataLoader(str(tmp_path) + '/')
        # HDFStore keys 带 / 前缀
        result = loader.load_h5('test.h5', '/mykey')
        pd.testing.assert_frame_equal(result, df)

    def test_load_h5_key_not_found(self, tmp_path):
        h5_file = tmp_path / 'test.h5'
        df = pd.DataFrame({'A': [1]})
        df.to_hdf(h5_file, key='existing', mode='w')
        loader = DataLoader(str(tmp_path) + '/')
        with pytest.raises(KeyError, match="not found"):
            loader.load_h5('test.h5', '/nonexistent')


class TestDataLoaderLoadCustom:
    """测试自定义路径加载"""

    def test_load_custom_csv_with_slash(self, tmp_path):
        csv_file = tmp_path / 'data.csv'
        csv_file.write_text('idx,X,Y\n0,10,20\n')
        loader = DataLoader()
        result = loader.load_custom((str(tmp_path) + '/', 'data.csv'))
        assert result.shape == (1, 2)

    def test_load_custom_csv_full_path(self, tmp_path):
        csv_file = tmp_path / 'factor.csv'
        csv_file.write_text('idx,A,B\n0,5,6\n')
        loader = DataLoader()
        result = loader.load_custom(('', str(csv_file)))
        assert result.shape == (1, 2)

    def test_load_custom_h5(self, tmp_path):
        h5_path = tmp_path / 'data.h5'
        df = pd.DataFrame({'X': [1, 2]})
        df.to_hdf(h5_path, key='factor_a', mode='w')
        loader = DataLoader()
        result = loader.load_custom((str(h5_path), 'factor_a'))
        pd.testing.assert_frame_equal(result, df)

    def test_load_custom_unsupported(self, tmp_path):
        loader = DataLoader()
        with pytest.raises(ValueError, match="不支持"):
            loader.load_custom(('/tmp/', 'data.xyz'))


class TestDataLoaderLoadFactor:
    """测试因子加载路由"""

    def test_load_factor_h5(self, tmp_path):
        h5_file = tmp_path / 'factor.h5'
        df = pd.DataFrame({'A': [1, 2, 3]})
        df.to_hdf(h5_file, key='momentum', mode='w')
        # load_factor(h5_path, key) -> load_h5(h5_path, key)
        # load_h5 用 self.api + filename, 所以 api 需要是空字符串
        loader = DataLoader(api_path='')
        result = loader.load_factor(str(h5_file), '/momentum')
        pd.testing.assert_frame_equal(result, df)

    def test_load_factor_csv(self, tmp_path):
        csv_file = tmp_path / 'factor.csv'
        csv_file.write_text('idx,C1,C2\n0,7,8\n')
        loader = DataLoader()
        result = loader.load_factor(str(csv_file), 'unused')
        assert result.shape == (1, 2)


class TestDataLoaderAddIndex:
    """测试 add_index"""

    def test_add_index_stock(self, tmp_path):
        h5_file = tmp_path / 'stk_daily.h5'
        stklist = pd.DataFrame(['000001.SZ', '600519.SH'])
        trade_dt = pd.DataFrame([20260104, 20260105])
        stklist.to_hdf(h5_file, key='stklist', mode='w')
        trade_dt.to_hdf(h5_file, key='trade_dt', mode='a')

        loader = DataLoader(str(tmp_path) + '/')
        factor = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        result = loader.add_index(factor, axis_type='stock')

        assert list(result.columns) == ['000001.SZ', '600519.SH']
        assert result.index.tolist() == [20260104, 20260105]

    def test_add_index_index(self, tmp_path):
        h5_file = tmp_path / 'index_daily.h5'
        indexlist = pd.DataFrame(['000300.SH', '000905.SH'])
        trade_dt = pd.DataFrame([20260104])
        indexlist.to_hdf(h5_file, key='indexlist', mode='w')
        trade_dt.to_hdf(h5_file, key='trade_dt', mode='a')

        loader = DataLoader(str(tmp_path) + '/')
        factor = pd.DataFrame([[100.0, 200.0]])
        result = loader.add_index(factor, axis_type='index')
        assert list(result.columns) == ['000300.SH', '000905.SH']


class TestDataLoaderValidShape:
    """测试 valid_shape"""

    def test_valid_shape_true(self, tmp_path):
        h5_file = tmp_path / 'stk_daily.h5'
        stklist = pd.DataFrame(['A', 'B', 'C'])
        trade_dt = pd.DataFrame([1, 2])
        stklist.to_hdf(h5_file, key='stklist', mode='w')
        trade_dt.to_hdf(h5_file, key='trade_dt', mode='a')

        loader = DataLoader(str(tmp_path) + '/')
        df = pd.DataFrame(np.zeros((2, 3)))
        assert loader.valid_shape(df) is True

    def test_valid_shape_false(self, tmp_path):
        h5_file = tmp_path / 'stk_daily.h5'
        stklist = pd.DataFrame(['A', 'B'])
        trade_dt = pd.DataFrame([1])
        stklist.to_hdf(h5_file, key='stklist', mode='w')
        trade_dt.to_hdf(h5_file, key='trade_dt', mode='a')

        loader = DataLoader(str(tmp_path) + '/')
        df = pd.DataFrame(np.zeros((3, 3)))
        assert loader.valid_shape(df) is False


class TestDataLoaderGetAxis:
    """测试 get_axis 路由"""

    def test_get_axis_stock(self, tmp_path):
        h5_file = tmp_path / 'stk_daily.h5'
        stklist = pd.DataFrame(['X'])
        trade_dt = pd.DataFrame([20260104])
        stklist.to_hdf(h5_file, key='stklist', mode='w')
        trade_dt.to_hdf(h5_file, key='trade_dt', mode='a')

        loader = DataLoader(str(tmp_path) + '/')
        result = loader.get_axis('stock')
        assert len(result) == 2

    def test_get_axis_index(self, tmp_path):
        h5_file = tmp_path / 'index_daily.h5'
        indexlist = pd.DataFrame(['000300.SH'])
        trade_dt = pd.DataFrame([20260104])
        indexlist.to_hdf(h5_file, key='indexlist', mode='w')
        trade_dt.to_hdf(h5_file, key='trade_dt', mode='a')

        loader = DataLoader(str(tmp_path) + '/')
        result = loader.get_axis('index')
        assert len(result) == 2

    def test_get_axis_invalid(self, tmp_path):
        loader = DataLoader()
        with pytest.raises(ValueError, match="不支持"):
            loader.get_axis('invalid')
