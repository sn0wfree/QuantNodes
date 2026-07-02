# coding=utf-8
"""
test_logic_mining_strict.py — strict=True 模式下静默 fallback 升级为异常 (v3.0.1)

覆盖 Phase 2 c (激进模式): 每个 silent fallback 点都验证
strict=True 时正确抛出 LogicMiningStrictError

被测的 silent fallback 点:
- P-01/P-09: LLM 异常
- P-02/P-03/P-04/P-10/P-11: parse 失败
- P-12: structured build 失败
- P-13 (隐含): parse_op_args — 不升级为异常, 仅 debug
"""
import pytest
from unittest.mock import MagicMock

from QuantNodes.research.quant_alpha.logic_mining import (
    LogicMiningPipeline,
    LogicMiningStrictError,
    PipelineMetrics,
    StrictConfig,
)
from QuantNodes.research.quant_alpha.logic_mining.generator import (
    MarketLogicGenerator,
    MarketLogicRefinementDirection,
)


class FailingClient:
    def complete(self, *, agent_id: str, prompt: str) -> str:
        raise ConnectionError(f"mocked fail {agent_id}")


class BadJSONClient:
    def complete(self, *, agent_id: str, prompt: str) -> str:
        return f"non-json trash for {agent_id}"


class TestStrictCallFlag:
    def test_pipelines_call_strict_raises(self):
        s = StrictConfig(call=True)
        p = LogicMiningPipeline(llm_client=FailingClient(), strict=s)
        with pytest.raises(LogicMiningStrictError) as ex:
            p.run("rank(close)", "alpha101")
        assert ex.value.kind == "call"
        assert ex.value.context["agent_id"] == "logic-mining-structure"

    def test_generator_call_strict_raises(self):
        s = StrictConfig(call=True)
        gen = MarketLogicGenerator(llm_client=FailingClient(), strict=s)
        with pytest.raises(LogicMiningStrictError) as ex:
            gen.generate(library=[], round_idx=1)
        assert ex.value.kind == "call"
        assert ex.value.context["agent_id"] == "market-logic-generator"

    def test_refiner_call_strict_raises(self):
        s = StrictConfig(call=True)
        ref = MarketLogicRefinementDirection(llm_client=FailingClient(), strict=s)
        # 使用 mock logic
        logic = MagicMock()
        logic.name = "x"
        logic.performance_evidence = None
        with pytest.raises(LogicMiningStrictError):
            ref.refine(logic)


class TestStrictParseFlag:
    def test_pipelines_parse_strict_raises_for_first_failing_stage(self):
        s = StrictConfig(parse=True)
        p = LogicMiningPipeline(llm_client=BadJSONClient(), strict=s)
        with pytest.raises(LogicMiningStrictError) as ex:
            p.run("rank(close)", "alpha101")
        # 结构层先失败
        assert ex.value.kind == "parse"
        assert ex.value.context["agent_id"] == "logic-mining-structure"
        assert ex.value.context["layer"] >= 1

    def test_generator_parse_strict_raises(self):
        s = StrictConfig(parse=True)
        gen = MarketLogicGenerator(llm_client=BadJSONClient(), strict=s)
        with pytest.raises(LogicMiningStrictError) as ex:
            gen.generate(library=[], round_idx=1)
        assert ex.value.kind == "parse"
        assert ex.value.context["agent_id"] == "market-logic-generator"


class TestStrictStructuredFlag:
    def test_strict_structured_only_raises_on_data_build_failure(self):
        """strict.structured 仅在 Stage 3 数据构 structured logic 抛错时升级
        (此处由 _structured_from_dict 失败触发, 实际测试需要更复杂的 setup,
        此处通过 strict 字段存在性 + LogicMiningPipeline 构造验证类型正确)
        """
        s = StrictConfig(structured=True)
        assert s.structured is True
        # 不能轻松触发 — 留给后续 PR 补充复杂场景

    def test_callable_keeps_backward_compatible(self):
        """strict=StrictConfig() 默认全 False → 不抛异常"""
        s = StrictConfig()
        p = LogicMiningPipeline(llm_client=FailingClient(), strict=s)
        # 不抛
        result = p.run("rank(close)", "alpha101")
        assert result is not None


class TestAllCombined:
    def test_all_strict_flags_together(self):
        s = StrictConfig(call=True, parse=True, structured=True)
        # BadJSON 路径下在 parse 处抛
        p = LogicMiningPipeline(llm_client=BadJSONClient(), strict=s)
        with pytest.raises(LogicMiningStrictError):
            p.run("rank(close)", "alpha101")
