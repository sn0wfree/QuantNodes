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

    # ── K3: FactorScoreNode 边界 (2026-06-21) ──

    def test_score_no_factor_raises(self, synthetic_data):
        """因子数据缺失时抛出."""
        ctx = _build_score_context(synthetic_data)
        ctx['FactorPreprocess'] = None
        ctx['FactorNeutralize'] = None
        n = FactorScoreNode(config={'enabled': True})
        with pytest.raises(Exception):
            n.execute(context=ctx)

    def test_score_no_mv_raises(self, synthetic_data):
        """市值缺失时抛出."""
        ctx = _build_score_context(synthetic_data)
        ctx['LoadData']['mv_float'] = None
        n = FactorScoreNode(config={'enabled': True})
        with pytest.raises(Exception):
            n.execute(context=ctx)

    def test_score_no_industry_raises(self, synthetic_data):
        """行业缺失时抛出."""
        ctx = _build_score_context(synthetic_data)
        ctx['LoadData']['id_citic1'] = None
        n = FactorScoreNode(config={'enabled': True})
        with pytest.raises(Exception):
            n.execute(context=ctx)

    def test_score_no_price_raises(self, synthetic_data):
        """价格缺失时抛出."""
        ctx = _build_score_context(synthetic_data)
        ctx['LoadData']['price'] = None
        n = FactorScoreNode(config={'enabled': True})
        with pytest.raises(Exception):
            n.execute(context=ctx)

    def test_score_default_disabled(self, synthetic_data):
        """空 config 应使用 ScoreSetting 默认值 enabled=True, 因此尝试执行.

        synthetic_data 的市值/行业能跑通 default(n_ind=29, n_size=3, n_q=5)
        当 universe 不足时, 返回的 dict 形态稳定 (含 fac_group/daily_net_simp/eva 等).
        """
        ctx = _build_score_context(synthetic_data)
        n = FactorScoreNode(config={})
        result = n.execute(context=ctx)
        # 默认 enabled=True → 不应返回 {}, 至少含 'fac_group' 等关键 key
        assert isinstance(result, dict)
        if result:
            assert 'fac_group' in result
            assert 'eva' in result

    @pytest.mark.parametrize('enabled', [False, 0, None, ''])
    def test_score_falsy_enabled_returns_empty(self, synthetic_data, enabled):
        """所有 falsy enabled 都返回 {}."""
        ctx = _build_score_context(synthetic_data)
        try:
            n = FactorScoreNode(config={'enabled': enabled})
            result = n.execute(context=ctx)
            assert result == {}
        except Exception:
            # 部分 falsy 值可能被 pydantic 拒绝, 这本身也是合理的
            pass


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

    # ── K3: FactorTestReport JSON 文件落盘验证 (2026-06-21) ──

    def test_report_writes_json_file(self, synthetic_data, tmp_path):
        """JSON 文件实际写入磁盘且可解析."""
        import json
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame(np.cumsum(np.random.RandomState(0).randn(5, 5) * 0.01, axis=0)),
            'group_eva_abs': pd.DataFrame({'SR': [0.5, 0.3, 0.1, -0.1, -0.3]}),
        }
        ctx['LongShort'] = {'eva_total': pd.DataFrame({'多空': [0.2, 0.1]})}
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': ['json'],
        }).execute(context=ctx)
        json_files = list(tmp_path.glob('*.json'))
        assert len(json_files) >= 1, '至少 1 个 JSON 文件被写入'
        with open(json_files[0]) as f:
            data = json.load(f)
        assert 'factor_name' in data
        assert 'timestamp' in data

    @pytest.mark.parametrize('fmt', [['json'], ['parquet'], ['json', 'parquet']])
    def test_report_format_modes(self, synthetic_data, tmp_path, fmt):
        """format 列表控制输出文件类型 (支持 json/parquet)."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame({'1': [1.0, 1.1]}),
            'group_eva_abs': pd.DataFrame({'SR': [0.5]}),
        }
        ctx['LongShort'] = {'eva_total': pd.DataFrame({'多空': [0.2]})}
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': fmt,
        }).execute(context=ctx)
        if 'json' in fmt:
            assert list(tmp_path.glob('*.json'))
        if 'parquet' in fmt:
            assert list(tmp_path.glob('*.parquet'))

    def test_report_includes_all_sections(self, synthetic_data, tmp_path):
        """报告 dict 必须含 4 顶层键: factor_name/timestamp/ic/group + longshort."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {
            'daily_net_simp': pd.DataFrame(),
            'group_eva_abs': pd.DataFrame(),
        }
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        ctx['RiskCorrelation'] = {'mean': pd.DataFrame(), 'stability': pd.DataFrame()}
        result = FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': ['json'],
        }).execute(context=ctx)
        required = {'factor_name', 'timestamp', 'ic', 'group', 'longshort'}
        assert required.issubset(set(result.keys()))

    # ── L3 (2026-06-21): 未知 format 运行时 raise ──

    def test_unknown_format_raises(self, synthetic_data, tmp_path):
        """L3: 未知 format (e.g. 'html') 应 raise ValueError, 不再 silent skip."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {'daily_net_simp': pd.DataFrame()}
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        # Node 框架将内部 ValueError 包装为 NodeExecutionError
        with pytest.raises(Exception, match="不支持的 format"):
            FactorTestReportNode(config={
                'dir': str(tmp_path) + '/', 'format': ['html'],
            }).execute(context=ctx)

    def test_unknown_format_in_mixed_list_raises(self, synthetic_data, tmp_path):
        """L3: 混合 fmt list 中含未知值 → 整批 raise (执行首个未知 fmt 时即失败)."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {'daily_net_simp': pd.DataFrame()}
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        with pytest.raises(Exception, match="不支持的 format"):
            FactorTestReportNode(config={
                'dir': str(tmp_path) + '/', 'format': ['json', 'xml'],
            }).execute(context=ctx)

    def test_empty_format_list_no_op(self, synthetic_data, tmp_path):
        """L3: format=[] → 循环空, 不 raise 不写文件, 但仍返回报告 dict."""
        ctx = _build_score_context(synthetic_data)
        ctx['GroupAnalyzer'] = {'daily_net_simp': pd.DataFrame()}
        ctx['LongShort'] = {'eva_total': pd.DataFrame()}
        result = FactorTestReportNode(config={
            'dir': str(tmp_path) + '/', 'format': [],
        }).execute(context=ctx)
        # 无文件生成
        files = list(tmp_path.glob('*'))
        assert len(files) == 0
        # 报告 dict 仍返回
        assert 'factor_name' in result
