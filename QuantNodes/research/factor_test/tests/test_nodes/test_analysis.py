# coding: utf-8
"""分析层节点测试 - ICAnalyzer, GroupAnalyzer, LongShort

GroupAnalyzer/LongShort 需要完整 H5 数据的 index 对齐,
这些节点在 E2E 测试中已验证。此处仅测试:
- ICAnalyzer (可独立测试)
- 错误路径 (None 输入)
- 节点可实例化
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode


def _build_analysis_context(synthetic_data, adj_mode=None):
    """构建分析层 context (通过前 5 个节点)"""
    ctx = dict(synthetic_data)
    ctx['LoadData'] = {
        'factor': ctx['factor'],
        'price': ctx['price'],
        'id_citic1': ctx['id_citic1'],
        'mv_float': ctx['mv_float'],
        'st': ctx['st'],
        'suspend': ctx['suspend'],
        'ud_limit': ctx['ud_limit'],
        'ipo_days': ctx['ipo_days'],
        'index_cp': ctx['index_cp'],
        'stklist': ctx['stklist'],
        'trade_dt': ctx['trade_dt'],
        '_loader': None,
    }
    # Node 3
    n3 = TradabilityFilterNode(config={
        'tradable': {'no_st': True, 'no_suspended': True}
    })
    ctx['TradabilityFilter'] = n3.execute(context=ctx)

    # Node 4
    mode = adj_mode or ['M', 'end']
    n4 = AdjustDateNode(config={
        'adj_date_beg': 20260101, 'adj_date_end': 20260630,
        'adj_mode': mode
    })
    ctx['AdjustDate'] = n4.execute(context=ctx)

    # Node 5
    n5 = FactorPreprocessNode(config={
        'missing': '', 'extreme': 'median', 'norm': 'zscore'
    })
    ctx['FactorPreprocess'] = n5.execute(context=ctx)
    ctx['FactorNeutralize'] = ctx['FactorPreprocess']
    return ctx


# ── ICAnalyzerNode ─────────────────────────────────────────────

class TestICAnalyzerNode:

    def test_basic_ic(self, synthetic_data):
        """基础 IC 计算"""
        ctx = _build_analysis_context(synthetic_data)
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        assert 'ic' in result
        assert 'ic_result' in result
        assert isinstance(result['ic'], pd.Series)

    def test_ic_result_has_metrics(self, synthetic_data):
        """IC 结果包含标准指标"""
        ctx = _build_analysis_context(synthetic_data)
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        ic_result = result['ic_result']
        assert 'IC均值' in ic_result
        assert 'ICIR' in ic_result

    def test_rank_ic_exists(self, synthetic_data):
        """Rank IC 存在"""
        ctx = _build_analysis_context(synthetic_data)
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        assert 'rank_ic' in result
        assert isinstance(result['rank_ic'], pd.Series)

    def test_ic_analyzer_fallback_to_preprocess(self, synthetic_data):
        """当 FactorNeutralize 不存在时回退到 FactorPreprocess"""
        ctx = _build_analysis_context(synthetic_data)
        del ctx['FactorNeutralize']  # 移除, 触发回退
        n = ICAnalyzerNode(config={'min_group_size': 5})
        result = n.execute(context=ctx)
        assert 'ic' in result

    def test_ic_analyzer_no_factor_raises(self, synthetic_data):
        """无因子数据时抛出"""
        ctx = _build_analysis_context(synthetic_data)
        ctx['FactorPreprocess'] = None
        ctx['FactorNeutralize'] = None
        n = ICAnalyzerNode(config={'min_group_size': 5})
        with pytest.raises(Exception):
            n.execute(context=ctx)


# ── GroupAnalyzerNode ──────────────────────────────────────────

class TestGroupAnalyzerNode:

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐, 见 E2E 测试")
    def test_basic_grouping(self, synthetic_data):
        pass

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐")
    def test_group_eva_abs_exists(self, synthetic_data):
        pass

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐")
    def test_excess_returns(self, synthetic_data):
        pass

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐")
    def test_no_benchmark(self, synthetic_data):
        pass

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐")
    def test_floor_mode_last(self, synthetic_data):
        pass

    def test_group_analyzer_no_factor_raises(self, synthetic_data):
        """无因子数据时抛出"""
        ctx = _build_analysis_context(synthetic_data)
        ctx['FactorPreprocess'] = None
        ctx['FactorNeutralize'] = None
        n = GroupAnalyzerNode(config={
            'groups': 5, 'factor_direction': 1,
            'floor_mode': 'group', 'hedge': 'equal'
        })
        with pytest.raises(Exception):
            n.execute(context=ctx)


# ── LongShortNode ──────────────────────────────────────────────

class TestLongShortNode:

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐, 见 E2E 测试")
    def test_basic_longshort(self, synthetic_data):
        pass

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐")
    def test_longshort_direction_minus1(self, synthetic_data):
        pass

    def test_longshort_no_group_raises(self, synthetic_data):
        """无分组数据时抛出"""
        ctx = _build_analysis_context(synthetic_data)
        ctx['GroupAnalyzer'] = None
        n9 = LongShortNode(config={'factor_direction': 1})
        with pytest.raises(Exception):
            n9.execute(context=ctx)

    @pytest.mark.skip(reason="需要 H5 数据的 index 对齐")
    def test_longshort_yearly_breakdown(self, synthetic_data):
        pass
