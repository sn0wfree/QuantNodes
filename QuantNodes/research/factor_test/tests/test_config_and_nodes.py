# coding: utf-8
"""单元测试: Config + Node imports"""

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parents[5])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from QuantNodes.research.factor_test.config import (
    SingleFactorTestConfig, FactorSetting, PreprocessSetting,
    TradableSetting, AnalysisSetting, OutputSetting,
)
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.factor_neutralize_node import FactorNeutralizeNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
from QuantNodes.research.factor_test.nodes.risk_correlation_node import RiskCorrelationNode
from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode
from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner


class TestConfig:
    def test_factor_setting(self):
        fs = FactorSetting(name='ep', factor_dir='./testdata/alpha/ep.h5')
        assert fs.name == 'ep'
        assert fs.format == 'h5'

    def test_tradable_setting(self):
        ts = TradableSetting(no_st=True, min_ipo_days=360)
        assert ts.no_st is True
        assert ts.min_ipo_days == 360

    def test_preprocess_setting(self):
        ps = PreprocessSetting(adj_date_beg=20170801, adj_date_end=20171231)
        assert ps.adj_date_beg == 20170801
        assert ps.missing == ''

    def test_full_config(self):
        config = SingleFactorTestConfig(
            factor={'name': 'ep', 'factor_dir': './testdata/alpha/ep.h5'},
            preprocess={'adj_date_beg': 20170801, 'adj_date_end': 20171231},
        )
        assert config.factor.name == 'ep'
        assert config.preprocess.adj_date_beg == 20170801
        assert config.analysis.group.groups == 5


class TestNodeImports:
    def test_all_nodes_importable(self):
        nodes = [
            LoadDataNode, SamplePoolFilterNode, TradabilityFilterNode,
            AdjustDateNode, FactorPreprocessNode, FactorNeutralizeNode,
            ICAnalyzerNode, GroupAnalyzerNode, LongShortNode,
            FactorScoreNode, RiskCorrelationNode, FactorTestReportNode,
        ]
        assert len(nodes) == 12
        for node_cls in nodes:
            assert hasattr(node_cls, '_execute')

    def test_pipeline_runner_creation(self):
        config = SingleFactorTestConfig(
            factor={'name': 'ep', 'factor_dir': './testdata/alpha/ep.h5'},
            preprocess={'adj_date_beg': 20170801, 'adj_date_end': 20171231},
        )
        runner = PipelineRunner(config)
        assert runner.config.factor.name == 'ep'

    def test_pipeline_runner_from_dict(self):
        data = {
            'factor': {'name': 'mom', 'factor_dir': './testdata/mom.h5'},
            'preprocess': {'adj_date_beg': 20180101, 'adj_date_end': 20181231},
        }
        runner = PipelineRunner.from_dict(data)
        assert runner.config.factor.name == 'mom'
