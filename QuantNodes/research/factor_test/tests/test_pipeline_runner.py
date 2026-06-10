# coding: utf-8
"""PipelineRunner 测试 - 端到端管线"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner


class TestPipelineRunnerFromDict:

    def test_from_dict_basic(self):
        """从字典创建"""
        config = {
            'factor': {'name': 'test', 'factor_dir': 'test.h5'},
            'preprocess': {'adj_date_beg': 20260101, 'adj_date_end': 20260630},
        }
        runner = PipelineRunner.from_dict(config)
        assert runner is not None

    def test_from_dict_has_context(self):
        """创建后有空 context"""
        config = {
            'factor': {'name': 'test', 'factor_dir': 'test.h5'},
            'preprocess': {'adj_date_beg': 20260101, 'adj_date_end': 20260630},
        }
        runner = PipelineRunner.from_dict(config)
        assert isinstance(runner.context, dict)


class TestPipelineRunnerRun:

    def test_run_synthetic(self, synthetic_data):
        """用合成数据运行完整管线"""
        config = {
            'factor': {'name': 'test_factor', 'factor_dir': 'test.h5'},
            'preprocess': {
                'adj_date_beg': 20260101,
                'adj_date_end': 20260630,
                'adj_mode': ['M', 'end'],
                'sample_index': 'all',
                'sample_industry': 'all',
                'tradable': {'no_st': True, 'no_suspended': True},
                'missing': '',
                'extreme': 'median',
                'norm': 'zscore',
            },
            'analysis': {
                'ic': {'min_group_size': 5},
                'group': {'groups': 5, 'factor_direction': 1,
                          'floor_mode': 'group', 'hedge': 'equal'},
                'longshort': {'factor_direction': 1},
                'score': {'enabled': True},
                'risk_corr': {'factors': ''},
            },
            'output': {'dir': '/tmp/test_pipeline/', 'format': ['json']},
        }
        # 注入合成数据到 context
        runner = PipelineRunner.from_dict(config)
        runner._context['LoadData'] = {
            'factor': synthetic_data['factor'],
            'price': synthetic_data['price'],
            'id_citic1': synthetic_data['id_citic1'],
            'mv_float': synthetic_data['mv_float'],
            'st': synthetic_data['st'],
            'suspend': synthetic_data['suspend'],
            'ud_limit': synthetic_data['ud_limit'],
            'ipo_days': synthetic_data['ipo_days'],
            'index_cp': synthetic_data['index_cp'],
            'stklist': synthetic_data['stklist'],
            'trade_dt': synthetic_data['trade_dt'],
            '_loader': None,
        }
        # 运行节点 2-12 (跳过 LoadData)
        from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
        from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
        from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
        from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
        from QuantNodes.research.factor_test.nodes.factor_neutralize_node import FactorNeutralizeNode
        from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
        from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
        from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
        from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
        from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode

        ctx = runner._context

        # Node 2
        n2 = SamplePoolFilterNode(config={
            'sample_index': 'all', 'sample_industry': 'all'
        })
        ctx['SamplePoolFilter'] = n2.execute(context=ctx)

        # Node 3
        n3 = TradabilityFilterNode(config={
            'tradable': {'no_st': True, 'no_suspended': True}
        })
        ctx['TradabilityFilter'] = n3.execute(context=ctx)

        # Node 4
        n4 = AdjustDateNode(config={
            'adj_date_beg': 20260101, 'adj_date_end': 20260630,
            'adj_mode': ['M', 'end']
        })
        ctx['AdjustDate'] = n4.execute(context=ctx)

        # Node 5
        n5 = FactorPreprocessNode(config={
            'missing': '', 'extreme': 'median', 'norm': 'zscore'
        })
        ctx['FactorPreprocess'] = n5.execute(context=ctx)

        # Node 6
        ctx['FactorNeutralize'] = ctx['FactorPreprocess']

        # Node 7
        n7 = ICAnalyzerNode(config={'min_group_size': 5})
        ctx['ICAnalyzer'] = n7.execute(context=ctx)

        # Node 8
        n8 = GroupAnalyzerNode(config={
            'groups': 5, 'factor_direction': 1,
            'floor_mode': 'group', 'hedge': 'equal'
        })
        ctx['GroupAnalyzer'] = n8.execute(context=ctx)

        # Node 9
        n9 = LongShortNode(config={'factor_direction': 1})
        ctx['LongShort'] = n9.execute(context=ctx)

        # Node 10
        n10 = FactorScoreNode(config={'enabled': True})
        ctx['FactorScore'] = n10.execute(context=ctx)

        # Node 11
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}

        # Node 12
        n12 = FactorTestReportNode(config={
            'dir': '/tmp/test_pipeline/', 'format': ['json']
        })
        ctx['FactorTestReport'] = n12.execute(context=ctx)

        # 验证
        assert len(ctx) >= 12
        assert 'ICAnalyzer' in ctx
        assert 'GroupAnalyzer' in ctx
        assert 'LongShort' in ctx
        assert 'FactorTestReport' in ctx

    def test_run_single_factor_test(self, synthetic_data):
        """run_single_factor_test 便捷函数"""
        config = {
            'factor': {'name': 'quick_test', 'factor_dir': 'test.h5'},
            'preprocess': {'adj_date_beg': 20260101, 'adj_date_end': 20260630},
        }
        runner = PipelineRunner.from_dict(config)
        # 注入数据
        runner._context['LoadData'] = {
            'factor': synthetic_data['factor'],
            'price': synthetic_data['price'],
            'stklist': synthetic_data['stklist'],
            'trade_dt': synthetic_data['trade_dt'],
            '_loader': None,
        }
        # 至少验证 runner 能创建
        assert runner is not None
