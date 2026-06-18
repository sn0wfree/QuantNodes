# coding: utf-8
"""Phase 3.1 T0: 12 节点 Pydantic 化基础设施测试

覆盖:
- T0-1: nodes/configs.py 12 *Config + NODE_CONFIG_SCHEMAS 路由表
- T0-2: 6 隐式默认 (mad_n/pct_low/pct_high/n_industries/n_size_groups/n_quantile_groups) 补 Pydantic 字段
- T0-3: H10 PreprocessSetting.adj_date_beg/end Optional + 启动校验
- T0-4: 12 节点 __init__ Union 化 (dict/Config/None)
- 向后兼容: self._xxx 实例属性保留 (5 处测试零改动)
- 跨进程: model_dump() → dict → 子进程路径稳定
- extra="forbid" 拼写错立即 ValidationError (新防线)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from QuantNodes.research.factor_test.config import PreprocessSetting, ScoreSetting
from QuantNodes.research.factor_test.nodes.configs import (
    LoadDataNodeConfig, SamplePoolNodeConfig, TradabilityNodeConfig,
    AdjustDateNodeConfig, PreprocessNodeConfig, NeutralizeNodeConfig,
    ICAnalyzerNodeConfig, GroupAnalyzerNodeConfig, LongShortNodeConfig,
    ScoreNodeConfig, RiskCorrelationNodeConfig, ReportNodeConfig,
    NODE_CONFIG_SCHEMAS,
)
from QuantNodes.research.factor_test.nodes.adjust_date_node import AdjustDateNode
from QuantNodes.research.factor_test.nodes.factor_neutralize_node import FactorNeutralizeNode
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import FactorPreprocessNode
from QuantNodes.research.factor_test.nodes.factor_score_node import FactorScoreNode
from QuantNodes.research.factor_test.nodes.factor_test_report_node import FactorTestReportNode
from QuantNodes.research.factor_test.nodes.group_analyzer_node import GroupAnalyzerNode
from QuantNodes.research.factor_test.nodes.ic_analyzer_node import ICAnalyzerNode
from QuantNodes.research.factor_test.nodes.load_data_node import LoadDataNode
from QuantNodes.research.factor_test.nodes.long_short_node import LongShortNode
from QuantNodes.research.factor_test.nodes.risk_correlation_node import RiskCorrelationNode
from QuantNodes.research.factor_test.nodes.sample_pool_filter_node import SamplePoolFilterNode
from QuantNodes.research.factor_test.nodes.tradability_filter_node import TradabilityFilterNode


# ============================================================================
# T0-1: 路由表完整性
# ============================================================================

class TestNodeConfigSchemas:
    """T0-1: NODE_CONFIG_SCHEMAS 12 个节点全覆盖"""

    def test_schema_count(self):
        assert len(NODE_CONFIG_SCHEMAS) == 12

    @pytest.mark.parametrize("node_name,expected_cls", [
        ("LoadData", LoadDataNodeConfig),
        ("SamplePoolFilter", SamplePoolNodeConfig),
        ("TradabilityFilter", TradabilityNodeConfig),
        ("AdjustDate", AdjustDateNodeConfig),
        ("FactorPreprocess", PreprocessNodeConfig),
        ("FactorNeutralize", NeutralizeNodeConfig),
        ("ICAnalyzer", ICAnalyzerNodeConfig),
        ("GroupAnalyzer", GroupAnalyzerNodeConfig),
        ("LongShort", LongShortNodeConfig),
        ("FactorScore", ScoreNodeConfig),
        ("RiskCorrelation", RiskCorrelationNodeConfig),
        ("FactorTestReport", ReportNodeConfig),
    ])
    def test_schema_mapping(self, node_name, expected_cls):
        assert NODE_CONFIG_SCHEMAS[node_name] is expected_cls


# ============================================================================
# T0-2: 6 隐式默认补 Pydantic 字段
# ============================================================================

class TestImplicitDefaultsPydantic:
    """T0-2: 6 隐式默认从节点 __init__ 提升到 Pydantic 字段"""

    def test_preprocess_setting_default_mad_n(self):
        s = PreprocessSetting(adj_date_beg=20240101, adj_date_end=20241231)
        assert s.mad_n == 5.0

    def test_preprocess_setting_default_pct_low(self):
        s = PreprocessSetting(adj_date_beg=20240101, adj_date_end=20241231)
        assert s.pct_low == 0.025

    def test_preprocess_setting_default_pct_high(self):
        s = PreprocessSetting(adj_date_beg=20240101, adj_date_end=20241231)
        assert s.pct_high == 0.975

    def test_score_setting_default_n_industries(self):
        s = ScoreSetting()
        assert s.n_industries == 29

    def test_score_setting_default_n_size_groups(self):
        s = ScoreSetting()
        assert s.n_size_groups == 3

    def test_score_setting_default_n_quantile_groups(self):
        s = ScoreSetting()
        assert s.n_quantile_groups == 5

    def test_preprocess_custom_mad_n(self):
        s = PreprocessSetting(
            adj_date_beg=20240101, adj_date_end=20241231, mad_n=3.0,
        )
        assert s.mad_n == 3.0

    def test_score_custom_n_industries(self):
        s = ScoreSetting(n_industries=30)
        assert s.n_industries == 30


# ============================================================================
# T0-3: H10 Optional[int] + 启动校验
# ============================================================================

class TestH10OptionalDate:
    """T0-3: PreprocessSetting.adj_date_beg/end 改 Optional + 启动校验"""

    def test_none_default_beg(self):
        s = PreprocessSetting()
        assert s.adj_date_beg is None

    def test_none_default_end(self):
        s = PreprocessSetting()
        assert s.adj_date_end is None

    def test_explicit_values_accepted(self):
        s = PreprocessSetting(adj_date_beg=20240101, adj_date_end=20241231)
        assert s.adj_date_beg == 20240101
        assert s.adj_date_end == 20241231


# ============================================================================
# T0-4: 12 节点 Union 化 (dict / *Config / None)
# ============================================================================

class TestUnionAcceptance:
    """T0-4: 节点 __init__ 接受 dict / *Config / None"""

    @pytest.mark.parametrize("node_cls,config_cls", [
        (LoadDataNode, LoadDataNodeConfig),
        (SamplePoolFilterNode, SamplePoolNodeConfig),
        (TradabilityFilterNode, TradabilityNodeConfig),
        (AdjustDateNode, AdjustDateNodeConfig),
        (FactorPreprocessNode, PreprocessNodeConfig),
        (FactorNeutralizeNode, NeutralizeNodeConfig),
        (ICAnalyzerNode, ICAnalyzerNodeConfig),
        (GroupAnalyzerNode, GroupAnalyzerNodeConfig),
        (LongShortNode, LongShortNodeConfig),
        (FactorScoreNode, ScoreNodeConfig),
        (RiskCorrelationNode, RiskCorrelationNodeConfig),
        (FactorTestReportNode, ReportNodeConfig),
    ])
    def test_dict_input_accepted(self, node_cls, config_cls):
        """(a) dict 输入 (LoadDataNode P-2 后需提供 data_path)"""
        if node_cls is LoadDataNode:
            node = node_cls(config={"data_path": "/tmp/test/"})
        else:
            node = node_cls(config={})
        assert node is not None

    @pytest.mark.parametrize("node_cls,config_cls", [
        (LoadDataNode, LoadDataNodeConfig),
        (SamplePoolFilterNode, SamplePoolNodeConfig),
        (TradabilityFilterNode, TradabilityNodeConfig),
        (AdjustDateNode, AdjustDateNodeConfig),
        (FactorPreprocessNode, PreprocessNodeConfig),
        (FactorNeutralizeNode, NeutralizeNodeConfig),
        (ICAnalyzerNode, ICAnalyzerNodeConfig),
        (GroupAnalyzerNode, GroupAnalyzerNodeConfig),
        (LongShortNode, LongShortNodeConfig),
        (FactorScoreNode, ScoreNodeConfig),
        (RiskCorrelationNode, RiskCorrelationNodeConfig),
        (FactorTestReportNode, ReportNodeConfig),
    ])
    def test_config_instance_input_accepted(self, node_cls, config_cls):
        """(b) Config 实例输入 (LoadDataNode P-2 后需 data_path)"""
        if node_cls is LoadDataNode:
            cfg = config_cls(data_path="/tmp/test/")
        else:
            cfg = config_cls()
        node = node_cls(config=cfg)
        assert node is not None

    @pytest.mark.parametrize("node_cls", [
        LoadDataNode, SamplePoolFilterNode, TradabilityFilterNode,
        AdjustDateNode, FactorPreprocessNode, FactorNeutralizeNode,
        ICAnalyzerNode, GroupAnalyzerNode, LongShortNode,
        FactorScoreNode, RiskCorrelationNode, FactorTestReportNode,
    ])
    def test_none_input_accepted(self, node_cls):
        """(c) None 输入 (LoadDataNode P-2 后 None 抛 ValidationError)"""
        if node_cls is LoadDataNode:
            # P-2: data_path 必填, None config 抛错
            with pytest.raises(ValidationError, match="data_path"):
                node_cls(config=None)
        else:
            node = node_cls(config=None)
            assert node is not None

    @pytest.mark.parametrize("node_cls", [
        LoadDataNode, SamplePoolFilterNode, TradabilityFilterNode,
        AdjustDateNode, FactorPreprocessNode, FactorNeutralizeNode,
        ICAnalyzerNode, GroupAnalyzerNode, LongShortNode,
        FactorScoreNode, RiskCorrelationNode, FactorTestReportNode,
    ])
    def test_invalid_type_raises_typeerror(self, node_cls):
        """(d) 非 dict/Config 类型抛 TypeError"""
        with pytest.raises(TypeError, match="config must be"):
            node_cls(config=42)


# ============================================================================
# T0-4 + T0-3: AdjustDateNode H10 启动校验
# ============================================================================

class TestAdjustDateStartupValidation:
    """T0-3: AdjustDateNode 默认 config 启动时抛 ValueError"""

    def test_default_config_raises(self, tmp_path):
        """H10 兼容: AdjustDateNode() 默认 config → 启动校验 ValueError"""
        import pandas as pd
        import numpy as np
        # 构造最小 context (LoadData 含 trade_dt)
        from QuantNodes.research.factor_test.utils.data_loader import DataLoader
        d = tmp_path
        n_days, n_stocks = 5, 2
        dates = [20250101 + i for i in range(n_days)]
        stks = [f"00000{i}.SZ" for i in range(n_stocks)]
        with pd.HDFStore(d / "stk_daily.h5", mode="w") as store:
            store.put("stklist", pd.DataFrame({0: stks}), format="table")
            store.put("trade_dt", pd.DataFrame({0: dates}), format="table")
            store.put("cp", pd.DataFrame(np.ones((n_days, n_stocks)), index=dates, columns=stks), format="table")
        loader = DataLoader(str(d) + "/")
        stklist, trade_dt = loader.get_stock_axis()

        node = AdjustDateNode()  # 默认 config
        ctx = {"LoadData": {
            "trade_dt": trade_dt, "stklist": stklist, "_loader": loader,
        }}
        with pytest.raises(ValueError, match="adj_date_beg.*adj_date_end"):
            node._execute(context=ctx)

    def test_valid_config_executes(self, tmp_path):
        """显式 config → 正常执行"""
        import pandas as pd
        import numpy as np
        from QuantNodes.research.factor_test.utils.data_loader import DataLoader
        d = tmp_path
        n_days, n_stks = 28, 5
        dates = [20250101 + i for i in range(n_days)]
        stks = [f"00000{i}.SZ" for i in range(n_stks)]
        with pd.HDFStore(d / "stk_daily.h5", mode="w") as store:
            store.put("stklist", pd.DataFrame({0: stks}), format="table")
            store.put("trade_dt", pd.DataFrame({0: dates}), format="table")
            store.put("cp", pd.DataFrame(np.ones((n_days, n_stks)), index=dates, columns=stks), format="table")
        loader = DataLoader(str(d) + "/")
        stklist, trade_dt = loader.get_stock_axis()

        node = AdjustDateNode(config={"adj_date_beg": 20250101, "adj_date_end": 20250131})
        ctx = {"LoadData": {"trade_dt": trade_dt, "stklist": stklist, "_loader": loader}}
        out = node._execute(context=ctx)
        assert out is not None


# ============================================================================
# T0-1: extra="forbid" 拼写错立即 ValidationError
# ============================================================================

class TestExtraForbidValidation:
    """T0-1: extra='forbid' 拼写错立即失败 (新防线)"""

    def test_preprocess_typo_madn(self):
        with pytest.raises(ValidationError):
            FactorPreprocessNode(config={"madn": 5.0})  # 错拼 (正确: mad_n)

    def test_preprocess_typo_pctlow(self):
        with pytest.raises(ValidationError):
            FactorPreprocessNode(config={"pctlow": 0.01})

    def test_score_typo_nindust(self):
        with pytest.raises(ValidationError):
            FactorScoreNode(config={"n_indust": 29})

    def test_score_typo_nsize(self):
        with pytest.raises(ValidationError):
            FactorScoreNode(config={"n_size": 3})

    def test_load_data_typo_loadkeys(self):
        with pytest.raises(ValidationError):
            LoadDataNode(config={"loadkeys": []})

    def test_adjust_date_typo_adjdate(self):
        with pytest.raises(ValidationError):
            AdjustDateNode(config={"adjdate_beg": 20240101})


# ============================================================================
# 向后兼容: self._xxx 实例属性保留
# ============================================================================

class TestSelfXxxBackwardsCompat:
    """向后兼容: 5 处现有测试使用的 self._xxx 仍存在"""

    def test_factor_score_self_attrs(self):
        node = FactorScoreNode(config={})
        assert node._n_industries == 29
        assert node._n_size_groups == 3
        assert node._n_quantile_groups == 5
        assert node._enabled is True

    def test_factor_score_custom_attrs(self):
        node = FactorScoreNode(config={
            "n_industries": 30, "n_size_groups": 2, "n_quantile_groups": 10,
        })
        assert (node._n_industries, node._n_size_groups, node._n_quantile_groups) == (30, 2, 10)

    def test_preprocess_self_attrs(self):
        node = FactorPreprocessNode(config={"extreme": "median"})
        assert node._mad_n == 5.0
        assert node._pct_low == 0.025
        assert node._pct_high == 0.975
        assert node._missing == ""
        assert node._extreme == "median"
        assert node._norm == ""

    def test_preprocess_custom_pct(self):
        node = FactorPreprocessNode(config={"pct_low": 0.01, "pct_high": 0.99})
        assert (node._pct_low, node._pct_high) == (0.01, 0.99)

    def test_load_data_self_attrs(self):
        # P-2: data_path 必填
        node = LoadDataNode(config={
            "data_path": "/tmp/test/",
            "load_keys": [
                "stklist", "trade_dt", "cp", "id_citic1", "mv_float",
                "st", "suspend", "ud_limit", "ipo_days",
            ],
        })
        for key in ("st", "suspend", "ud_limit", "ipo_days"):
            assert key in node._load_keys

    def test_adjust_date_self_attrs(self):
        node = AdjustDateNode(config={"adj_date_beg": 20240101, "adj_date_end": 20241231})
        assert node._adj_date_beg == 20240101
        assert node._adj_date_end == 20241231
        assert node._adj_mode == ["M", "end"]


# ============================================================================
# 跨进程: model_dump() → dict → 子进程路径稳定
# ============================================================================

class TestCrossProcessDictStability:
    """跨进程: Pydantic model_dump() → dict, 子进程 dict.get 链稳定"""

    @pytest.mark.parametrize("config_cls,kwargs", [
        (LoadDataNodeConfig, {"data_path": "/tmp/"}),
        (SamplePoolNodeConfig, {"sample_index": "HS300"}),
        (AdjustDateNodeConfig, {"adj_date_beg": 20240101, "adj_date_end": 20241231}),
        (PreprocessNodeConfig, {}),
        (NeutralizeNodeConfig, {"industry_neutral": True}),
        (ICAnalyzerNodeConfig, {"min_group_size": 10}),
        (GroupAnalyzerNodeConfig, {"groups": 10}),
        (LongShortNodeConfig, {"factor_direction": -1}),
        (ScoreNodeConfig, {"n_industries": 30}),
        (RiskCorrelationNodeConfig, {"factors": "all"}),
        (ReportNodeConfig, {"dir": "/tmp/output/"}),
    ])
    def test_model_dump_roundtrip(self, config_cls, kwargs):
        """model_dump() 产生 dict, 重建后字段一致"""
        cfg = config_cls(**kwargs)
        d = cfg.model_dump()
        assert isinstance(d, dict)
        cfg2 = config_cls.model_validate(d)
        # 比较字段
        for k, v in kwargs.items():
            assert getattr(cfg2, k) == v

    def test_report_node_path_is_path(self):
        """P-1: self._output_dir 是 Path 对象 (expanduser 已应用)"""
        node = FactorTestReportNode(config={})
        assert isinstance(node._output_dir, Path)


# ============================================================================
# P-1 (顺带): FactorTestReportNode 路径优先级
# ============================================================================

class TestReportNodePath:
    """P-1: env QUANTNODES_OUTPUT_DIR > expanduser > default"""

    def test_default_path(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUANTNODES_OUTPUT_DIR", raising=False)
        node = FactorTestReportNode(config={})
        # 默认 './output/' 经 expanduser 规范化, 是 Path 对象
        assert isinstance(node._output_dir, Path)
        assert node._output_dir.name == "output"

    def test_expanduser(self, tmp_path, monkeypatch):
        """~ 展开 (用 monkeypatch HOME 隔离测试环境)"""
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.delenv("QUANTNODES_OUTPUT_DIR", raising=False)
        node = FactorTestReportNode(config={"dir": "~/my_reports/"})
        assert node._output_dir == fake_home / "my_reports"

    def test_env_priority(self, tmp_path, monkeypatch):
        env_dir = tmp_path / "env_output"
        env_dir.mkdir()
        monkeypatch.setenv("QUANTNODES_OUTPUT_DIR", str(env_dir))
        node = FactorTestReportNode(config={"dir": "./output/"})
        # env 覆盖 config
        assert node._output_dir == env_dir

    def test_env_with_tilde(self, tmp_path, monkeypatch):
        """env 也支持 ~ 展开"""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("QUANTNODES_OUTPUT_DIR", "~/env_output/")
        node = FactorTestReportNode(config={})
        assert node._output_dir == home / "env_output"

    def test_env_priority_over_expanduser(self, tmp_path, monkeypatch):
        """env 优先于 config 的 expanduser"""
        env_dir = tmp_path / "env"
        config_dir = tmp_path / "config"
        env_dir.mkdir()
        config_dir.mkdir()
        monkeypatch.setenv("QUANTNODES_OUTPUT_DIR", str(env_dir))
        node = FactorTestReportNode(config={"dir": str(config_dir)})
        assert node._output_dir == env_dir


# ============================================================================
# P-2: LoadDataNode data_path 必填校验
# ============================================================================

class TestLoadDataPathRequired:
    """P-2: data_path 必填, 启动时报错 (None/缺字段/空串)"""

    def test_missing_data_path_raises_validation_error(self):
        """缺字段 → __init__ 抛 ValidationError"""
        with pytest.raises(ValidationError, match="data_path"):
            LoadDataNode(config={})

    def test_none_data_path_raises_validation_error(self):
        """None 显式 → __init__ 抛 ValidationError"""
        with pytest.raises(ValidationError, match="data_path"):
            LoadDataNode(config={"data_path": None})

    def test_empty_data_path_raises_value_error_on_execute(self, tmp_path):
        """空字符串 → __init__ 不挡 (Pydantic 限制), _execute 抛 ValueError"""
        node = LoadDataNode(config={"data_path": ""})
        with pytest.raises(ValueError, match="data_path required"):
            node._execute()

    def test_valid_data_path_executes(self, tmp_path):
        """有效路径 → _execute 进入加载流程 (允许失败但不应 data_path 校验错)"""
        d = tmp_path
        # 构造最小 H5
        import pandas as pd
        import numpy as np
        n_days, n_stks = 5, 2
        dates = [20250101 + i for i in range(n_days)]
        stks = [f"00000{i}.SZ" for i in range(n_stks)]
        with pd.HDFStore(d / "stk_daily.h5", mode="w") as store:
            store.put("stklist", pd.DataFrame({0: stks}), format="table")
            store.put("trade_dt", pd.DataFrame({0: dates}), format="table")
            store.put("cp", pd.DataFrame(np.ones((n_days, n_stks)), index=dates, columns=stks), format="table")
        node = LoadDataNode(config={"data_path": str(d) + "/"})
        # _execute 应通过 data_path 校验, 进入加载流程
        out = node._execute()
        assert out is not None
        assert "stklist" in out
        assert "trade_dt" in out
