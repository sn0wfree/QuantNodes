# coding: utf-8
"""Score / Report-layer node tests: FactorScore, RiskCorrelation, FactorTestReport.

历史来源: 迁移自 ``QuantNodes/research/factor_test/tests/test_nodes/test_score_report.py`` (C2 收敛).
部分节点需要 H5 数据的 index 对齐, 在 E2E 测试中已验证.
此处测试: 错误路径、节点实例化、FactorTestReport 输出.
"""

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
from QuantNodes.research.factor_test.nodes.risk_correlation_node import RiskCorrelationNode
from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode


def _build_score_context(synthetic_data):
    """构建评分层 context."""
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
    n3 = TradabilityFilterNode(config={
        'tradable': {'no_st': True, 'no_suspended': True},
    })
    ctx['TradabilityFilter'] = n3.execute(context=ctx)
    n4 = AdjustDateNode(config={
        'adj_date_beg': 20260101, 'adj_date_end': 20260630,
        'adj_mode': ['M', 'end'],
    })
    ctx['AdjustDate'] = n4.execute(context=ctx)
    n5 = FactorPreprocessNode(config={
        'missing': '', 'extreme': 'median', 'norm': 'zscore',
    })
    ctx['FactorPreprocess'] = n5.execute(context=ctx)
    ctx['FactorNeutralize'] = ctx['FactorPreprocess']
    n7 = ICAnalyzerNode(config={'min_group_size': 5})
    ctx['ICAnalyzer'] = n7.execute(context=ctx)
    return ctx


# ── FactorScoreNode ────────────────────────────────────────────

class TestFactorScoreNode:

    def test_score_disabled(self, synthetic_data):
        """评分禁用返回空 dict."""
        ctx = _build_score_context(synthetic_data)
        n = FactorScoreNode(config={'enabled': False})
        result = n.execute(context=ctx)
        assert result == {}


# ── RiskCorrelationNode ────────────────────────────────────────

class TestRiskCorrelationNode:

    def test_no_loader_raises(self, synthetic_data):
        """无 loader 时抛出错误.

        注: RiskCorrelationNodeConfig.factors 现为 str (默认 'all'),
        不能传 []. 这里只验证无 _loader 的错误路径.
        """
        ctx = _build_score_context(synthetic_data)
        ctx['LoadData']['_loader'] = None
        ctx['FactorNeutralize'] = ctx['FactorPreprocess']
        n = RiskCorrelationNode(config={'factors': 'all'})
        with pytest.raises(Exception):
            n.execute(context=ctx)

    def test_risk_correlation_no_factor_raises(self, synthetic_data):
        """无因子数据时抛出."""
        ctx = _build_score_context(synthetic_data)
        ctx['FactorPreprocess'] = None
        ctx['FactorNeutralize'] = None
        n = RiskCorrelationNode(config={'factors': 'all'})
        with pytest.raises(Exception):
            n.execute(context=ctx)


# ── FactorTestReportNode ───────────────────────────────────────

class TestFactorTestReportNode:

    def test_report_json(self, synthetic_data, tmp_path):
        """JSON 报告生成."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame(np.cumsum(np.random.randn(5, 5) * 0.01, axis=0)),
            'group_eva_abs': pd.DataFrame({'SR': [0.5, 0.3, 0.1, -0.1, -0.3]}),
        }
        ctx['LongShort'] = {
            'eva_total': pd.DataFrame({'多空': [0.2, 0.1]}),
        }
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        n = FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': ['json'],
        })
        result = n.execute(context=ctx)
        assert 'factor_name' in result
        assert 'timestamp' in result

    def test_report_has_ic_section(self, synthetic_data, tmp_path):
        """报告包含 IC 部分."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame(),
            'group_eva_abs': pd.DataFrame(),
        }
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        n = FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': ['json'],
        })
        result = n.execute(context=ctx)
        assert 'ic' in result

    def test_report_has_group_section(self, synthetic_data, tmp_path):
        """报告包含分组部分."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame(),
            'group_eva_abs': pd.DataFrame(),
        }
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        n = FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': ['json'],
        })
        result = n.execute(context=ctx)
        assert 'group' in result

    def test_report_has_longshort_section(self, synthetic_data, tmp_path):
        """报告包含多空部分."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame(),
            'group_eva_abs': pd.DataFrame(),
        }
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        n = FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': ['json'],
        })
        result = n.execute(context=ctx)
        assert 'longshort' in result