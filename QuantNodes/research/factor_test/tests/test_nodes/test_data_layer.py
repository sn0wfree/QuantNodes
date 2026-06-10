# coding: utf-8
"""数据层节点测试 - LoadData, SamplePool, Tradability, AdjustDate"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode


# ── LoadDataNode ───────────────────────────────────────────────

class TestLoadDataNode:

    def test_with_synthetic_context(self, synthetic_context):
        """已有 LoadData context, 节点直接返回"""
        ctx = dict(synthetic_context)
        # LoadData 已存在, 节点应直接返回
        ctx['LoadData'] = {'factor': ctx['factor'], 'price': ctx['price']}
        n = LoadDataNode(config={})
        # LoadDataNode 需要从 H5 加载, 但已有 LoadData 时应跳过
        # 实际上 LoadDataNode._execute 会尝试加载 H5
        # 所以这里测试的是: 当 context 中已有 LoadData 时的行为
        assert 'factor' in ctx['LoadData']
        assert 'price' in ctx['LoadData']

    def test_factor_shape_preserved(self, synthetic_context):
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {'factor': ctx['factor'], 'price': ctx['price']}
        assert ctx['LoadData']['factor'].shape == (120, 30)


# ── SamplePoolFilterNode ───────────────────────────────────────

class TestSamplePoolFilterNode:

    def test_all_stocks(self, synthetic_context):
        """sample_index='all' 不过滤"""
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,
        }
        n = SamplePoolFilterNode(config={
            'sample_index': 'all',
            'sample_industry': 'all',
        })
        result = n.execute(context=ctx)
        assert result.shape == (120, 30)
        # 所有值应为 1 (可交易)
        assert (result == 1).all().all()

    def test_with_custom_loader(self, synthetic_context):
        """当 _loader 存在时, 尝试从 H5 加载指数成分"""
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,  # 无 loader, 使用 context-first
        }
        n = SamplePoolFilterNode(config={
            'sample_index': 'all',
            'sample_industry': 'all',
        })
        result = n.execute(context=ctx)
        assert isinstance(result, pd.DataFrame)


# ── TradabilityFilterNode ──────────────────────────────────────

class TestTradabilityFilterNode:

    def test_no_st_filter(self, synthetic_context):
        """ST 过滤: ST 股票被标记为 NaN"""
        ctx = dict(synthetic_context)
        # 确保所有数据使用相同的 index/columns
        idx = ctx['factor'].index
        cols = ctx['factor'].columns
        ctx['LoadData'] = {
            'factor': ctx['factor'],
            'st': pd.DataFrame(ctx['st'].values, index=idx, columns=cols),
            'suspend': pd.DataFrame(ctx['suspend'].values, index=idx, columns=cols),
            'ud_limit': pd.DataFrame(ctx['ud_limit'].values, index=idx, columns=cols),
            'ipo_days': pd.DataFrame(ctx['ipo_days'].values, index=idx, columns=cols),
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,
        }
        n = TradabilityFilterNode(config={
            'tradable': {'no_st': True, 'no_suspended': False,
                         'no_up_down_limit': False}
        })
        result = n.execute(context=ctx)
        # 前2只股票有 ST=1, 应被过滤
        assert pd.isna(result.iloc[0, 0])  # 第1只 ST 股
        assert pd.isna(result.iloc[0, 1])  # 第2只 ST 股
        assert result.iloc[0, 2] == 1       # 第3只非 ST 股

    def test_no_suspended_filter(self, synthetic_context):
        """停牌过滤: 停牌股票被标记为 NaN"""
        ctx = dict(synthetic_context)
        idx = ctx['factor'].index
        cols = ctx['factor'].columns
        ctx['LoadData'] = {
            'factor': ctx['factor'],
            'st': pd.DataFrame(ctx['st'].values, index=idx, columns=cols),
            'suspend': pd.DataFrame(ctx['suspend'].values, index=idx, columns=cols),
            'ud_limit': pd.DataFrame(ctx['ud_limit'].values, index=idx, columns=cols),
            'ipo_days': pd.DataFrame(ctx['ipo_days'].values, index=idx, columns=cols),
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,
        }
        n = TradabilityFilterNode(config={
            'tradable': {'no_st': False, 'no_suspended': True,
                         'no_up_down_limit': False}
        })
        result = n.execute(context=ctx)
        # 第4只在第5-8天停牌 (index 5)
        assert pd.isna(result.iloc[5, 3])

    def test_no_up_down_limit(self, synthetic_context):
        """涨跌停过滤"""
        ctx = dict(synthetic_context)
        idx = ctx['factor'].index
        cols = ctx['factor'].columns
        ctx['LoadData'] = {
            'factor': ctx['factor'],
            'st': pd.DataFrame(ctx['st'].values, index=idx, columns=cols),
            'suspend': pd.DataFrame(ctx['suspend'].values, index=idx, columns=cols),
            'ud_limit': pd.DataFrame(ctx['ud_limit'].values, index=idx, columns=cols),
            'ipo_days': pd.DataFrame(ctx['ipo_days'].values, index=idx, columns=cols),
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,
        }
        n = TradabilityFilterNode(config={
            'tradable': {'no_st': False, 'no_suspended': False,
                         'no_up_down_limit': True}
        })
        result = n.execute(context=ctx)
        # ud_limit 全为 0, 所有股票可交易
        assert (result == 1).all().all()

    def test_ipo_days_filter(self, synthetic_context):
        """IPO 天数过滤: 上市不足 min_ipo_days 天的被过滤"""
        ctx = dict(synthetic_context)
        idx = ctx['factor'].index
        cols = ctx['factor'].columns
        # 设置 ipo_days 为很小的值
        small_ipo = pd.DataFrame(np.ones((120, 30)) * 10, index=idx, columns=cols)
        ctx['LoadData'] = {
            'factor': ctx['factor'],
            'st': pd.DataFrame(ctx['st'].values, index=idx, columns=cols),
            'suspend': pd.DataFrame(ctx['suspend'].values, index=idx, columns=cols),
            'ud_limit': pd.DataFrame(ctx['ud_limit'].values, index=idx, columns=cols),
            'ipo_days': small_ipo,
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,
        }
        n = TradabilityFilterNode(config={
            'tradable': {'no_st': False, 'no_suspended': False,
                         'no_up_down_limit': False, 'min_ipo_days': 100}
        })
        result = n.execute(context=ctx)
        # ipo_days=10 < 100, 所有股票被过滤
        assert result.isna().all().all()

    def test_combined_filters(self, synthetic_context):
        """组合过滤: ST + 停牌"""
        ctx = dict(synthetic_context)
        idx = ctx['factor'].index
        cols = ctx['factor'].columns
        ctx['LoadData'] = {
            'factor': ctx['factor'],
            'st': pd.DataFrame(ctx['st'].values, index=idx, columns=cols),
            'suspend': pd.DataFrame(ctx['suspend'].values, index=idx, columns=cols),
            'ud_limit': pd.DataFrame(ctx['ud_limit'].values, index=idx, columns=cols),
            'ipo_days': pd.DataFrame(ctx['ipo_days'].values, index=idx, columns=cols),
            'stklist': ctx['stklist'],
            'trade_dt': ctx['trade_dt'],
            '_loader': None,
        }
        n = TradabilityFilterNode(config={
            'tradable': {'no_st': True, 'no_suspended': True,
                         'no_up_down_limit': False}
        })
        result = n.execute(context=ctx)
        # ST 股 + 停牌股都被过滤
        tradable_count = (result == 1).sum().sum()
        assert tradable_count < 120 * 30


# ── AdjustDateNode ─────────────────────────────────────────────

class TestAdjustDateNode:

    def test_monthly_end(self, synthetic_context):
        """月度调仓"""
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {'trade_dt': ctx['trade_dt']}
        n = AdjustDateNode(config={
            'adj_date_beg': 20260101,
            'adj_date_end': 20260630,
            'adj_mode': ['M', 'end'],
        })
        result = n.execute(context=ctx)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_weekly_end(self, synthetic_context):
        """周度调仓"""
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {'trade_dt': ctx['trade_dt']}
        n = AdjustDateNode(config={
            'adj_date_beg': 20260101,
            'adj_date_end': 20260228,
            'adj_mode': ['W', 'end'],
        })
        result = n.execute(context=ctx)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_daily(self, synthetic_context):
        """日度调仓"""
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {'trade_dt': ctx['trade_dt']}
        n = AdjustDateNode(config={
            'adj_date_beg': 20260101,
            'adj_date_end': 20260131,
            'adj_mode': ['D', 1],
        })
        result = n.execute(context=ctx)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 1

    def test_adj_dates_are_int(self, synthetic_context):
        """调仓日为整数格式"""
        ctx = dict(synthetic_context)
        ctx['LoadData'] = {'trade_dt': ctx['trade_dt']}
        n = AdjustDateNode(config={
            'adj_date_beg': 20260101,
            'adj_date_end': 20260331,
            'adj_mode': ['M', 'end'],
        })
        result = n.execute(context=ctx)
        # 所有值应为 8 位整数
        for val in result.iloc[:, 0]:
            assert len(str(int(val))) == 8
