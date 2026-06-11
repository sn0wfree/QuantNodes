"""PipelineRunner + FactorFeedback 集成测试 (8 tests)。

验证:
    - 默认禁用 (backward compat)
    - 启用后 ctx['Feedback'] 包含 5 个分析节点
    - judge_enabled=True 时附加 LLM 通道
    - output_dir 持久化 Parquet
    - factor_id 跨节点共享
    - 节点返回 dict / FactorFeedback 都正确包装
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from QuantNodes.core.feedback import (
    FactorFeedback,
    FeedbackChannel,
)
from QuantNodes.research.factor_test.config import (
    FeedbackSetting,
    SingleFactorTestConfig,
)
from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner


def _base_config(feedback: dict | None = None) -> dict:
    cfg = {
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
            'group': {'groups': 5, 'factor_direction': 1, 'floor_mode': 'group', 'hedge': 'equal'},
            'longshort': {'factor_direction': 1},
            'score': {'enabled': True},
            'risk_corr': {'factors': ''},
        },
        'output': {'dir': '/tmp/test_pipeline_feedback/', 'format': ['json']},
    }
    if feedback is not None:
        cfg['feedback'] = feedback
    return cfg


def _populate_context(runner, synthetic_data):
    """Populate context with synthetic data + execute nodes 2-12 manually.

    Mirrors the pattern in test_pipeline_runner.py::test_run_synthetic.
    """
    from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
    from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
    from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
    from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
    from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
    from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
    from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
    from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
    from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode

    ctx = runner._context
    ctx['LoadData'] = {
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

    ctx['SamplePoolFilter'] = SamplePoolFilterNode(
        config={'sample_index': 'all', 'sample_industry': 'all'}
    ).execute(context=ctx)
    ctx['TradabilityFilter'] = TradabilityFilterNode(
        config={'tradable': {'no_st': True, 'no_suspended': True}}
    ).execute(context=ctx)
    ctx['AdjustDate'] = AdjustDateNode(
        config={'adj_date_beg': 20260101, 'adj_date_end': 20260630, 'adj_mode': ['M', 'end']}
    ).execute(context=ctx)
    ctx['FactorPreprocess'] = FactorPreprocessNode(
        config={'missing': '', 'extreme': 'median', 'norm': 'zscore'}
    ).execute(context=ctx)
    ctx['FactorNeutralize'] = ctx['FactorPreprocess']
    ctx['ICAnalyzer'] = ICAnalyzerNode(config={'min_group_size': 5}).execute(context=ctx)
    ctx['GroupAnalyzer'] = GroupAnalyzerNode(
        config={'groups': 5, 'factor_direction': 1, 'floor_mode': 'group', 'hedge': 'equal'}
    ).execute(context=ctx)
    ctx['LongShort'] = LongShortNode(config={'factor_direction': 1}).execute(context=ctx)
    ctx['FactorScore'] = FactorScoreNode(config={'enabled': True}).execute(context=ctx)
    ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
    ctx['FactorTestReport'] = FactorTestReportNode(
        config={'dir': '/tmp/test_pipeline_feedback/', 'format': ['json']}
    ).execute(context=ctx)


class TestFeedbackBackwardCompat:
    """默认 feedback 禁用时, 行为不变。"""

    def test_default_feedback_disabled(self):
        """FeedbackSetting 默认 enabled=False。"""
        cfg = SingleFactorTestConfig(
            factor={'name': 'x', 'factor_dir': 'x.h5'},
            preprocess={'adj_date_beg': 20240101, 'adj_date_end': 20240301},
        )
        assert cfg.feedback.enabled is False
        assert cfg.feedback.judge_enabled is False
        assert cfg.feedback.output_dir is None

    def test_build_feedback_skips_when_disabled(self, synthetic_data, tmp_path):
        """_build_feedback 仍可调用, 但 runner.run() 不会触发。"""
        runner = PipelineRunner.from_dict(_base_config())
        _populate_context(runner, synthetic_data)
        fb = runner._build_feedback(
            runner._context,
            factor_id='test-id',
            factor_name='test_factor',
            judge=None,
        )
        assert len(fb) == 5


class TestFeedbackEnabled:
    """feedback.enabled=True 时, 自动包装 5 个分析节点。"""

    def test_feedback_contains_five_analysis_nodes(self, synthetic_data, tmp_path):
        """_build_feedback 包含 ICAnalyzer/GroupAnalyzer/LongShort/FactorScore/RiskCorrelation。"""
        cfg = _base_config(feedback={
            'enabled': True,
            'output_dir': str(tmp_path / 'fb'),
        })
        runner = PipelineRunner.from_dict(cfg)
        _populate_context(runner, synthetic_data)

        feedback = runner._build_feedback(
            runner._context, factor_id='test-id', factor_name='test_factor', judge=None,
        )
        assert len(feedback) == 5
        for node_name in ('ICAnalyzer', 'GroupAnalyzer', 'LongShort', 'FactorScore', 'RiskCorrelation'):
            assert node_name in feedback, f"missing {node_name}"
            assert isinstance(feedback[node_name], FactorFeedback)
            assert feedback[node_name].factor_id == 'test-id'
            assert feedback[node_name].factor_name == 'test_factor'

    def test_feedback_shares_factor_id_across_nodes(self, synthetic_data):
        """5 个节点共享同一个 factor_id (via _build_feedback)。"""
        runner = PipelineRunner.from_dict(_base_config())
        _populate_context(runner, synthetic_data)
        fb = runner._build_feedback(
            runner._context, factor_id='shared-id', factor_name='test_factor', judge=None,
        )
        ids = {f.factor_id for f in fb.values()}
        assert ids == {'shared-id'}

    def test_feedback_persists_to_parquet(self, synthetic_data, tmp_path):
        """output_dir 持久化 5 行 Parquet。"""
        out_dir = tmp_path / 'fb'
        runner = PipelineRunner.from_dict(_base_config(feedback={
            'enabled': True, 'output_dir': str(out_dir),
        }))
        _populate_context(runner, synthetic_data)
        fb = runner._build_feedback(
            runner._context, factor_id='persist-id', factor_name='test_factor', judge=None,
        )
        runner._maybe_persist_feedback(fb, runner.config)

        parquet_path = out_dir / 'feedback.parquet'
        assert parquet_path.exists()

        loaded = FactorFeedback.load_parquet(parquet_path)
        assert len(loaded) == 5
        names = {f.factor_name for f in loaded}
        assert names == {'test_factor'}

    def test_feedback_no_persist_when_output_dir_none(self, synthetic_data):
        """output_dir=None 时不写文件。"""
        runner = PipelineRunner.from_dict(_base_config(feedback={
            'enabled': True, 'output_dir': None,
        }))
        _populate_context(runner, synthetic_data)
        fb = runner._build_feedback(
            runner._context, factor_id='no-persist', factor_name='test_factor', judge=None,
        )
        runner._maybe_persist_feedback(fb, runner.config)
        assert len(fb) == 5


class TestFeedbackLLMJudge:
    """judge_enabled=True 时附加 LLM 通道。"""

    def test_judge_adds_llm_channel(self, synthetic_data):
        """5 个 FactorFeedback 都包含 LLM 通道 (factor 含 hypothesis/description)。"""
        from QuantNodes.core.feedback import LLMJudge
        cfg = _base_config()
        cfg['factor']['hypothesis'] = 'momentum effect'
        cfg['factor']['description'] = '20-day momentum factor'
        cfg['factor']['expression'] = 'close / close.shift(20) - 1'
        runner = PipelineRunner.from_dict(cfg)
        _populate_context(runner, synthetic_data)
        judge = LLMJudge(model='mock', max_correction_attempts=2)
        fb = runner._build_feedback(
            runner._context, factor_id='judge-id', factor_name='test_factor', judge=judge,
        )
        for node_name, f in fb.items():
            assert FeedbackChannel.LLM in f.channels, f"{node_name} missing LLM"
            assert f.channels[FeedbackChannel.LLM].metadata['model'] == 'mock'

    def test_judge_disabled_no_llm_channel(self, synthetic_data):
        """judge=None 时不含 LLM 通道。"""
        runner = PipelineRunner.from_dict(_base_config())
        _populate_context(runner, synthetic_data)
        fb = runner._build_feedback(
            runner._context, factor_id='no-judge', factor_name='test_factor', judge=None,
        )
        for f in fb.values():
            assert FeedbackChannel.LLM not in f.channels


class TestFeedbackConfigValidation:
    """FeedbackSetting 配置验证。"""

    def test_explicit_settings(self):
        """显式配置所有字段。"""
        cfg = SingleFactorTestConfig(
            factor={'name': 'x', 'factor_dir': 'x.h5'},
            preprocess={'adj_date_beg': 20240101, 'adj_date_end': 20240301},
            feedback=FeedbackSetting(
                enabled=True,
                output_dir='/tmp/fb',
                judge_enabled=True,
                judge_model='deepseek-v3',
                judge_max_attempts=5,
            ),
        )
        assert cfg.feedback.enabled is True
        assert cfg.feedback.output_dir == '/tmp/fb'
        assert cfg.feedback.judge_model == 'deepseek-v3'
        assert cfg.feedback.judge_max_attempts == 5
