# coding=utf-8
"""
test_alpha_logics.py - 外层循环单元测试

测试：
- MarketLogicGenerator 生成逻辑
- MarketLogicRefinementDirection 反馈
- AlphaLogicsWorkflow 外层循环
- WikiLogic 结构化字段序列化
"""

import pytest

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
    LogicPerformanceEvidence,
)
from QuantNodes.research.quant_alpha.logic_mining.generator import (
    MarketLogicGenerator,
    MarketLogicRefinementDirection,
    generate_logic_name,
)
from QuantNodes.research.quant_alpha.workflow.alpha_logics import (
    AlphaLogicsConfig,
    AlphaLogicsWorkflow,
    AlphaLogicsResult,
    InnerLoopResult,
    _build_inner_evidence,
)
from QuantNodes.research.wiki import WikiLogic, LogicSource


# ==============================================================================
# WikiLogic 结构化字段测试
# ==============================================================================


class TestWikiLogicStructured:
    """WikiLogic 结构化字段测试"""

    def test_to_structured_dict(self):
        """序列化"""
        structured = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="ts_mean", threshold=0, window=20)],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        )
        logic = WikiLogic(
            name="test_logic",
            content="test content",
            source=LogicSource.RESEARCH_REPORT,
            structured=structured,
            refinement_round=1,
            parent_logic="parent_logic",
        )
        d = logic.to_structured_dict()
        assert d["name"] == "test_logic"
        assert d["structured"]["predicates"][0]["variable"] == "close"
        assert d["refinement_round"] == 1
        assert d["parent_logic"] == "parent_logic"

    def test_from_structured_dict(self):
        """反序列化"""
        d = {
            "name": "test",
            "content": "test",
            "source": "research_report",
            "structured": {
                "predicates": [{"variable": "open", "op": "rank", "threshold": 0}],
                "behavior": {"target": "forward_return_5", "direction": -1, "horizon": 5},
            },
            "refinement_round": 2,
            "parent_logic": "parent",
        }
        logic = WikiLogic.from_structured_dict(d)
        assert logic.name == "test"
        assert logic.structured is not None
        assert logic.structured.predicates[0].variable == "open"
        assert logic.refinement_round == 2
        assert logic.parent_logic == "parent"

    def test_roundtrip(self):
        """往返序列化"""
        original = WikiLogic(
            name="roundtrip_test",
            content="content",
            source=LogicSource.RESEARCH_REPORT,
            structured=WikiLogicStructured(
                predicates=[LogicCondition(variable="close", op="ts_mean", threshold=0, window=20)],
                behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            ),
            refinement_round=3,
        )
        d = original.to_structured_dict()
        restored = WikiLogic.from_structured_dict(d)
        assert restored.name == original.name
        assert restored.structured.predicates[0].variable == "close"
        assert restored.refinement_round == 3


# ==============================================================================
# MarketLogicGenerator 测试
# ==============================================================================


class TestMarketLogicGenerator:
    """MarketLogicGenerator 测试"""

    def test_generate_initial(self):
        """初始轮生成"""
        gen = MarketLogicGenerator(llm_client=None)
        logic = gen.generate(
            library=[],
            current_logic=None,
            history=[],
            evidence=[],
            round_idx=1,
        )
        assert logic.refinement_round == 1
        assert logic.parent_logic is None
        assert logic.structured is not None

    def test_generate_with_current(self):
        """基于当前逻辑生成"""
        gen = MarketLogicGenerator(llm_client=None)
        current = WikiLogic(
            name="current",
            content="test",
            source=LogicSource.RESEARCH_REPORT,
            structured=WikiLogicStructured(
                predicates=[LogicCondition(variable="close", op="ts_mean", threshold=0, window=20)],
                behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            ),
        )
        new = gen.generate(
            library=[current],
            current_logic=current,
            history=[current],
            evidence=[],
            round_idx=2,
        )
        assert new.refinement_round == 2
        assert new.parent_logic == "current"

    def test_generate_with_evidence(self):
        """基于 evidence 趋势生成"""
        gen = MarketLogicGenerator(llm_client=None)
        ev1 = LogicPerformanceEvidence(
            n_factors_explored=5, best_ir=0.5, refinement_round=1
        )
        ev2 = LogicPerformanceEvidence(
            n_factors_explored=5, best_ir=0.3, refinement_round=2
        )
        new = gen.generate(
            library=[],
            current_logic=None,
            history=[],
            evidence=[ev1, ev2],
            round_idx=3,
        )
        # IR 下降，应反转方向
        assert new.structured is not None
        assert new.structured.sign_constraint == 1

    def test_generate_name(self):
        """名称生成"""
        gen = MarketLogicGenerator(llm_client=None, base_name="test")
        logic = gen.generate(library=[], round_idx=5)
        assert "test" in logic.name.lower()


class TestGenerateLogicName:
    """generate_logic_name 测试"""

    def test_basic(self):
        """基本生成"""
        assert generate_logic_name("alpha", 1) == "alpha_v1"
        assert generate_logic_name("logic", 10) == "logic_v10"


# ==============================================================================
# MarketLogicRefinementDirection 测试
# ==============================================================================


class TestMarketLogicRefinementDirection:
    """MarketLogicRefinementDirection 测试"""

    def test_refine_empty_evidence(self):
        """无证据反馈"""
        refiner = MarketLogicRefinementDirection(llm_client=None)
        logic = WikiLogic(
            name="test",
            content="test",
            source=LogicSource.RESEARCH_REPORT,
        )
        fb = refiner.refine(logic, [], [])
        assert "diagnosis" in fb
        assert "direction" in fb
        assert "suggested_changes" in fb

    def test_refine_declining_ir(self):
        """IR 下降反馈"""
        refiner = MarketLogicRefinementDirection(llm_client=None)
        logic = WikiLogic(
            name="test",
            content="test",
            source=LogicSource.RESEARCH_REPORT,
        )
        ev1 = LogicPerformanceEvidence(best_ir=0.5, refinement_round=1)
        ev2 = LogicPerformanceEvidence(best_ir=0.3, refinement_round=2)
        fb = refiner.refine(logic, [], [ev1, ev2])
        # IR 下降，诊断应为 logic_too_broad
        assert fb["diagnosis"] == "logic_too_broad"

    def test_refine_saturated(self):
        """饱和反馈"""
        refiner = MarketLogicRefinementDirection(llm_client=None)
        logic = WikiLogic(
            name="test",
            content="test",
            source=LogicSource.RESEARCH_REPORT,
        )
        ev1 = LogicPerformanceEvidence(best_ir=0.5, refinement_round=1)
        ev2 = LogicPerformanceEvidence(best_ir=0.5, refinement_round=2)
        fb = refiner.refine(logic, [], [ev1, ev2])
        assert fb["diagnosis"] == "saturated"


# ==============================================================================
# AlphaLogicsWorkflow 测试
# ==============================================================================


class TestAlphaLogicsWorkflow:
    """AlphaLogicsWorkflow 测试"""

    def test_config_default(self):
        """默认配置"""
        config = AlphaLogicsConfig()
        assert config.max_outer_rounds == 4
        assert config.inner_iterations == 5
        assert config.persist_best_logic is True

    def test_workflow_init(self):
        """Workflow 初始化"""
        config = AlphaLogicsConfig(wiki_path="/tmp/test_wiki")
        workflow = AlphaLogicsWorkflow(config=config)
        assert workflow.config == config
        assert workflow.mining_pipeline is not None
        assert workflow.generator is not None
        assert workflow.refiner is not None


class TestBuildInnerEvidence:
    """_build_inner_evidence 测试"""

    def test_build_with_no_result(self):
        """无结果"""
        ev = _build_inner_evidence("test_logic", None, 1)
        assert ev.n_factors_explored == 0
        assert ev.best_ir == 0.0
        assert ev.refinement_round == 1

    def test_build_with_result(self):
        """有结果"""
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class FakeRecord:
            formula_id: str
            ir: float

        @dataclass
        class FakeSummary:
            total_evaluated: int

        @dataclass
        class FakeResult:
            final_pool: list
            summary: dict = dc_field(default_factory=dict)

        records = [
            FakeRecord(formula_id="F1", ir=0.5),
            FakeRecord(formula_id="F2", ir=0.8),
            FakeRecord(formula_id="F3", ir=0.3),
        ]
        result = FakeResult(final_pool=records)

        ev = _build_inner_evidence("test", result, 2)
        assert ev.n_factors_explored == 3
        assert ev.best_ir == 0.8
        assert ev.best_factor_id == "F2"
        assert ev.mean_ir == pytest.approx(0.533, rel=0.01)
        assert ev.refinement_round == 2


# ==============================================================================
# 集成测试
# ==============================================================================


class TestIntegration:
    """集成测试"""

    def test_generate_to_structure(self):
        """生成到结构化"""
        gen = MarketLogicGenerator(llm_client=None)
        logic = gen.generate(library=[], round_idx=1)

        # 应有结构化字段
        assert logic.structured is not None
        assert len(logic.structured.predicates) > 0
        assert logic.structured.behavior is not None

    def test_serialize_with_evidence(self):
        """含证据的序列化"""
        from QuantNodes.research.wiki import WikiLogic

        structured = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="ts_mean", threshold=0, window=20)],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        )
        logic = WikiLogic(
            name="integration_test",
            content="test",
            source=LogicSource.RESEARCH_REPORT,
            structured=structured,
            performance_evidence=LogicPerformanceEvidence(
                n_factors_explored=10,
                best_ir=0.65,
                best_factor_id="F1",
                refinement_round=1,
            ),
            refinement_round=1,
        )
        d = logic.to_structured_dict()
        assert d["performance_evidence"]["best_ir"] == 0.65

        # 反序列化
        restored = WikiLogic.from_structured_dict(d)
        assert restored.performance_evidence.best_ir == 0.65
        assert restored.performance_evidence.best_factor_id == "F1"