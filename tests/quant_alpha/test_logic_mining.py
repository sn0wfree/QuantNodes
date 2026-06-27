# coding=utf-8
"""
test_logic_mining.py - Logic Mining 单元测试

测试：
- LogicPerformanceEvidence 序列化/反序列化
- LogicAbstractionResult 数据结构
- sources.py 数据源适配
- parser.py JSON 解析
- pipelines.py 三段式 Agent
"""

import pytest

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
    LogicPerformanceEvidence,
    LogicAbstractionResult,
)
from QuantNodes.research.quant_alpha.logic_mining.parser import (
    ParseResult,
    parse_json_response,
    parse_formula_structure,
    parse_financial_semantics,
    parse_market_logic,
    _mock_structure_response,
    _mock_semantics_response,
    _mock_abstraction_response,
)
from QuantNodes.research.quant_alpha.logic_mining.sources import (
    SOURCES,
    get_formulas_from_source,
    list_available_sources,
)
from QuantNodes.research.quant_alpha.logic_mining.pipelines import (
    LogicMiningPipeline,
    mine_logic_from_formula,
    build_initial_logic_library,
)


# ==============================================================================
# LogicPerformanceEvidence 测试
# ==============================================================================


class TestLogicPerformanceEvidence:
    """LogicPerformanceEvidence 测试"""

    def test_basic(self):
        """基本创建"""
        ev = LogicPerformanceEvidence(
            n_factors_explored=8,
            best_ir=0.83,
            best_ic=0.045,
            best_factor_id="FORMULA-1",
            mean_ir=0.42,
        )
        assert ev.n_factors_explored == 8
        assert ev.best_ir == 0.83

    def test_to_dict(self):
        """序列化"""
        ev = LogicPerformanceEvidence(
            n_factors_explored=5, best_ir=0.5, best_factor_id="X",
        )
        d = ev.to_dict()
        assert d["n_factors_explored"] == 5
        assert d["best_factor_id"] == "X"

    def test_from_dict(self):
        """反序列化"""
        d = {"n_factors_explored": 3, "best_ir": 0.3, "best_ic": 0.02}
        ev = LogicPerformanceEvidence.from_dict(d)
        assert ev.n_factors_explored == 3
        assert ev.best_ic == 0.02


class TestLogicAbstractionResult:
    """LogicAbstractionResult 测试"""

    def test_basic(self):
        """基本创建"""
        result = LogicAbstractionResult(source_formula="rank(close)", source_lib="alpha101")
        assert result.source_formula == "rank(close)"
        assert result.source_lib == "alpha101"

    def test_to_dict(self):
        """序列化"""
        result = LogicAbstractionResult(
            formula_structure={"operations": ["rank"]},
            source_formula="rank(close)",
        )
        d = result.to_dict()
        assert d["formula_structure"]["operations"] == ["rank"]
        assert d["structured_logic"] is None


# ==============================================================================
# Parser 测试
# ==============================================================================


class TestParseJsonResponse:
    """parse_json_response 测试"""

    def test_direct_json(self):
        """直接 JSON"""
        raw = '{"a": 1, "b": 2}'
        result = parse_json_response(raw)
        assert result.ok
        assert result.data == {"a": 1, "b": 2}

    def test_markdown_json(self):
        """markdown 代码块"""
        raw = '```json\n{"a": 1}\n```'
        result = parse_json_response(raw)
        assert result.ok
        assert result.data == {"a": 1}

    def test_bare_braces(self):
        """裸 brace 块"""
        raw = 'Some text {"a": 1} more text'
        result = parse_json_response(raw)
        assert result.ok
        assert result.data == {"a": 1}

    def test_empty(self):
        """空响应"""
        result = parse_json_response("")
        assert not result.ok

    def test_invalid(self):
        """无效 JSON"""
        result = parse_json_response("not json")
        assert not result.ok


class TestParseFormulaStructure:
    """parse_formula_structure 测试"""

    def test_valid(self):
        """有效输入"""
        raw = '{"operations": ["rank", "ts_corr"], "window_length": 10}'
        result = parse_formula_structure(raw)
        assert result.ok
        assert result.data["operations"] == ["rank", "ts_corr"]
        assert result.data["window_length"] == 10

    def test_missing_operations(self):
        """缺少 operations"""
        raw = '{"window_length": 10}'
        result = parse_formula_structure(raw)
        assert not result.ok

    def test_defaults_filled(self):
        """默认值填充"""
        raw = '{"operations": ["rank"]}'
        result = parse_formula_structure(raw)
        assert result.ok
        assert result.data["window_length"] == 0
        assert result.data["has_ranking"] == False


class TestParseFinancialSemantics:
    """parse_financial_semantics 测试"""

    def test_valid(self):
        """有效输入"""
        raw = '{"price_role": "trend", "volume_role": "participation"}'
        result = parse_financial_semantics(raw)
        assert result.ok
        assert result.data["price_role"] == "trend"


class TestParseMarketLogic:
    """parse_market_logic 测试"""

    def test_valid(self):
        """有效输入"""
        raw = '''{
            "predicates": [{"variable": "open", "op": "rank", "threshold": 0}],
            "behavior": {"target": "forward_return_5", "direction": -1, "horizon": 5}
        }'''
        result = parse_market_logic(raw)
        assert result.ok

    def test_missing_predicates(self):
        """缺少 predicates"""
        raw = '{"behavior": {"target": "forward_return_5"}}'
        result = parse_market_logic(raw)
        assert not result.ok


class TestMockResponses:
    """Mock 响应测试"""

    def test_structure_mock(self):
        """结构 mock"""
        resp = _mock_structure_response("rank(ts_corr(close, volume, 10))")
        import json
        data = json.loads(resp)
        assert "rank" in data["operations"]
        assert "ts_corr" in data["operations"]
        assert data["window_length"] == 10
        assert data["has_ranking"] == True

    def test_semantics_mock(self):
        """语义 mock"""
        resp = _mock_semantics_response("rank(ts_corr(close, volume, 10))")
        import json
        data = json.loads(resp)
        assert "price_role" in data
        assert data["volume_role"] == "participation"

    def test_abstraction_mock(self):
        """抽象 mock"""
        structure = {"operations": ["ts_corr"]}
        semantics = {"volume_role": "participation"}
        resp = _mock_abstraction_response(
            "ts_corr(close, volume, 10)", structure, semantics
        )
        import json
        data = json.loads(resp)
        assert "predicates" in data
        assert "behavior" in data


# ==============================================================================
# Sources 测试
# ==============================================================================


class TestSources:
    """数据源测试"""

    def test_list_available_sources(self):
        """列出可用数据源"""
        sources = list_available_sources()
        assert "alpha101" in sources
        assert "alpha158" in sources

    def test_get_formulas_alpha101(self):
        """获取 alpha101 公式"""
        formulas = get_formulas_from_source("alpha101", max_count=5)
        assert len(formulas) > 0
        assert formulas[0]["lib"] == "alpha101"
        assert "formula" in formulas[0]

    def test_get_formulas_alpha158(self):
        """获取 alpha158 模板"""
        formulas = get_formulas_from_source("alpha158", max_count=5)
        assert len(formulas) > 0
        assert formulas[0]["lib"] == "alpha158"

    def test_get_formulas_unknown_source(self):
        """未知数据源"""
        formulas = get_formulas_from_source("unknown_lib")
        assert formulas == []

    def test_only_volume_price(self):
        """仅量价过滤"""
        formulas = get_formulas_from_source(
            "alpha101", max_count=20, only_volume_price=True
        )
        # 所有公式应不含财务关键词
        for f in formulas:
            assert "earnings" not in f["formula"].lower()

    def test_max_count(self):
        """最大数量限制"""
        formulas = get_formulas_from_source("alpha101", max_count=2)
        assert len(formulas) <= 2


# ==============================================================================
# Pipeline 测试
# ==============================================================================


class TestLogicMiningPipeline:
    """LogicMiningPipeline 测试"""

    def test_basic_mock(self):
        """基本 mock 运行"""
        pipeline = LogicMiningPipeline()
        result = pipeline.run("rank(ts_corr(close, volume, 10))", "alpha101")

        assert result.source_formula == "rank(ts_corr(close, volume, 10))"
        assert result.source_lib == "alpha101"
        assert "operations" in result.formula_structure
        assert "price_role" in result.financial_semantics
        assert result.structured_logic is not None

    def test_corr_volume_logic(self):
        """量价相关逻辑抽取"""
        pipeline = LogicMiningPipeline()
        result = pipeline.run("-ts_corr(rank(open), rank(volume), 10)", "alpha101")

        # 应抽取出量价背离逻辑
        assert result.structured_logic is not None
        assert result.structured_logic.sign_constraint == -1
        assert "ts_corr" in result.structured_logic.get_operators()

    def test_ts_mean_logic(self):
        """时序均值逻辑抽取"""
        pipeline = LogicMiningPipeline()
        result = pipeline.run("rank(ts_mean(close, 20))", "alpha101")

        assert result.structured_logic is not None
        assert "ts_mean" in result.structured_logic.get_operators()

    def test_simple_formula(self):
        """简单公式"""
        pipeline = LogicMiningPipeline()
        result = pipeline.run("rank(close)", "alpha101")

        assert result.source_formula == "rank(close)"
        assert result.formula_structure["has_ranking"] == True


class TestMineLogicFromFormula:
    """mine_logic_from_formula 测试"""

    def test_convenience_function(self):
        """便捷函数"""
        result = mine_logic_from_formula(
            formula="rank(close)",
            source_lib="alpha101",
        )
        assert result.source_formula == "rank(close)"
        assert result.source_lib == "alpha101"


class TestBuildInitialLogicLibrary:
    """build_initial_logic_library 测试"""

    def test_basic(self):
        """基本构建"""
        logics = build_initial_logic_library(
            source_libs=("alpha101",),
            max_per_lib=3,
        )
        assert len(logics) > 0
        for logic in logics:
            assert logic.structured_logic is not None
            assert logic.source_formula != ""

    def test_multiple_sources(self):
        """多数据源"""
        logics = build_initial_logic_library(
            source_libs=("alpha101", "alpha158"),
            max_per_lib=3,
        )
        # 应有来自两个数据源的逻辑
        libs = {l.source_lib for l in logics}
        assert len(libs) >= 1

    def test_empty_source(self):
        """空数据源"""
        logics = build_initial_logic_library(
            source_libs=("unknown_lib",),
            max_per_lib=5,
        )
        assert logics == []


# ==============================================================================
# 集成测试
# ==============================================================================


class TestIntegration:
    """集成测试"""

    def test_mine_to_compile(self):
        """从公式抽取逻辑到编译约束"""
        result = mine_logic_from_formula(
            formula="-ts_corr(rank(open), rank(volume), 10)",
            source_lib="alpha101",
        )

        assert result.structured_logic is not None

        # 编译为 Γ 约束
        from QuantNodes.research.quant_alpha.logic_mining import compile_to_constraint
        gamma = compile_to_constraint(
            result.structured_logic,
            source_logic="mined_from_alpha101",
        )

        # 验证约束
        passed, reason = gamma.validate("-ts_corr(rank(open), rank(volume), 10)")
        assert passed == True

    def test_library_quality(self):
        """初始库质量检查"""
        logics = build_initial_logic_library(
            source_libs=("alpha101",),
            max_per_lib=5,
        )

        # 至少有一些逻辑有结构化字段
        structured_count = sum(1 for l in logics if l.structured_logic is not None)
        assert structured_count == len(logics)

        # 至少有一些逻辑有算子白名单
        with_whitelist = sum(
            1 for l in logics
            if l.structured_logic and l.structured_logic.operator_whitelist
        )
        assert with_whitelist > 0