# coding: utf-8
"""IFinDDatabase 测试 - iFinD API 包装层"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.research.factor_test.ifind_db.fetcher import IFindFetcherStub
from QuantNodes.research.factor_test.ifind_db.ifind_database import IFinDDatabase


class TestIFindFetcherStub:
    """测试 IFindFetcherStub 本身"""

    def test_register_and_query(self, stub_fetcher):
        expected = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        stub_fetcher.register('stock', 'get_stock_info', {'query': 'test'}, expected)
        result = stub_fetcher.query('stock', 'get_stock_info', {'query': 'test'})
        pd.testing.assert_frame_equal(result, expected)

    def test_query_records_calls(self, stub_fetcher):
        stub_fetcher.query('stock', 'get_stock_info', {'query': 'a'})
        stub_fetcher.query('index', 'index_data', {'query': 'b'})
        assert len(stub_fetcher.calls) == 2
        assert stub_fetcher.calls[0][0] == 'stock'
        assert stub_fetcher.calls[1][0] == 'index'

    def test_query_unknown_returns_empty(self, stub_fetcher):
        result = stub_fetcher.query('stock', 'unknown_tool', {'query': 'x'})
        assert result.empty


class TestIFinDDatabaseInit:
    """测试 IFinDDatabase 初始化"""

    def test_init_default(self, stub_fetcher):
        db = IFinDDatabase(fetcher=stub_fetcher)
        assert db._date_beg == '20260101'
        assert db._universe == '沪深300'

    def test_init_custom_dates(self, stub_fetcher):
        db = IFinDDatabase(date_beg='20250101', date_end='20251231',
                           fetcher=stub_fetcher)
        assert db._date_beg == '20250101'
        assert db._date_end == '20251231'

    def test_init_ignores_api_path(self, stub_fetcher):
        db = IFinDDatabase(api_path='/some/path', fetcher=stub_fetcher)
        assert db._date_beg == '20260101'


class TestIFinDDatabaseStockAxis:
    """测试股票轴获取"""

    def test_get_stock_codes_from_index(self, stub_fetcher):
        codes_df = pd.DataFrame({
            '证券代码': ['600519.SH', '300750.SZ', '600036.SH'],
            '证券简称': ['贵州茅台', '宁德时代', '招商银行']
        })
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300成分股列表'}, codes_df)

        db = IFinDDatabase(universe='沪深300', fetcher=stub_fetcher)
        codes = db._get_stock_codes()
        assert len(codes) == 3
        assert '600519.SH' in codes

    def test_get_stock_axis_returns_dataframes(self, stub_fetcher):
        codes_df = pd.DataFrame({
            '证券代码': ['600519.SH', '300750.SZ'],
            '证券简称': ['贵州茅台', '宁德时代']
        })
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300成分股列表'}, codes_df)
        # 需要价格数据来提取交易日
        prices_df = pd.DataFrame({
            '证券代码': ['600519.SH', '300750.SZ'] * 2,
            '日期': ['20260104', '20260104', '20260105', '20260105'],
            '收盘价': [1800.0, 250.0, 1810.0, 255.0]
        })
        stub_fetcher.register('stock', 'get_stock_info',
                              {'query': '600519.SH、300750.SZ2026年01月至06月的日收盘价'},
                              prices_df)

        db = IFinDDatabase(universe='沪深300', fetcher=stub_fetcher)
        stklist, trade_dt = db.get_stock_axis()
        assert isinstance(stklist, pd.DataFrame)
        assert isinstance(trade_dt, pd.DataFrame)
        assert len(stklist) == 2
        assert len(trade_dt) == 2  # 2 个交易日


class TestIFinDDatabaseH5Route:
    """测试 load_h5 路由"""

    def test_known_route(self, stub_fetcher):
        codes_df = pd.DataFrame({'证券代码': ['600519.SH']})
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300成分股列表'}, codes_df)
        prices_df = pd.DataFrame({
            '证券代码': ['600519.SH'],
            '日期': ['20260104'],
            '收盘价': [1800.0]
        })
        stub_fetcher.register('stock', 'get_stock_info',
                              {'query': '600519.SH2026年01月至06月的日收盘价'},
                              prices_df)

        db = IFinDDatabase(universe='沪深300', fetcher=stub_fetcher)
        result = db.load_h5('stk_daily.h5', 'cp')
        assert isinstance(result, pd.DataFrame)

    def test_unknown_route_raises(self, stub_fetcher):
        db = IFinDDatabase(fetcher=stub_fetcher)
        with pytest.raises(KeyError, match="未映射"):
            db.load_h5('unknown.h5', 'unknown_key')


class TestIFinDDatabaseAddIndex:
    """测试 add_index"""

    def test_add_index_labels(self, stub_fetcher):
        codes_df = pd.DataFrame({'证券代码': ['600519.SH', '300750.SZ']})
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300成分股列表'}, codes_df)
        # 注册指数数据 (用于提取交易日)
        index_data_df = pd.DataFrame({
            '证券代码': ['000300.SH', '000905.SH'] * 2,
            '日期': ['20260104', '20260104', '20260105', '20260105'],
            '收盘点数': [3500.0, 6000.0, 3510.0, 6010.0]
        })
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300、中证5002026年01月至06月的收盘点数'},
                              index_data_df)

        db = IFinDDatabase(universe='沪深300', fetcher=stub_fetcher)
        raw = pd.DataFrame([[1800.0, 250.0], [1810.0, 255.0]])
        result = db.add_index(raw, axis_type='stock')
        assert result.index.tolist() == [20260104, 20260105]
        assert list(result.columns) == ['600519.SH', '300750.SZ']


class TestIFinDDatabaseValidShape:
    """测试 valid_shape"""

    def test_valid_shape_true(self, stub_fetcher):
        codes_df = pd.DataFrame({'证券代码': ['600519.SH']})
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300成分股列表'}, codes_df)
        prices_df = pd.DataFrame({
            '证券代码': ['600519.SH'],
            '日期': ['20260104'],
            '收盘价': [1800.0]
        })
        stub_fetcher.register('stock', 'get_stock_info',
                              {'query': '600519.SH2026年01月至06月的日收盘价'},
                              prices_df)

        db = IFinDDatabase(universe='沪深300', fetcher=stub_fetcher)
        df = pd.DataFrame([[1.0]])
        assert db.valid_shape(df, axis_type='stock') is True

    def test_valid_shape_false(self, stub_fetcher):
        codes_df = pd.DataFrame({'证券代码': ['600519.SH']})
        stub_fetcher.register('index', 'index_data',
                              {'query': '沪深300成分股列表'}, codes_df)
        prices_df = pd.DataFrame({
            '证券代码': ['600519.SH'],
            '日期': ['20260104'],
            '收盘价': [1800.0]
        })
        stub_fetcher.register('stock', 'get_stock_info',
                              {'query': '600519.SH2026年01月至06月的日收盘价'},
                              prices_df)

        db = IFinDDatabase(universe='沪深300', fetcher=stub_fetcher)
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
        assert db.valid_shape(df, axis_type='stock') is False


class TestIFinDDatabaseGetApikeys:
    """测试 get_apikeys"""

    def test_returns_risk_factor_list(self, stub_fetcher):
        db = IFinDDatabase(fetcher=stub_fetcher)
        keys = db.get_apikeys('risk_factor.h5')
        assert isinstance(keys, list)
        assert len(keys) == 10
        assert all(k.startswith('/') for k in keys)
        assert '/beta' in keys
        assert '/momentum' in keys
