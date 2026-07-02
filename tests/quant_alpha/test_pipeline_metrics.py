# coding=utf-8
"""
test_pipeline_metrics.py — PipelineMetrics / StrictConfig / LogicMiningStrictError (v3.0.1)

覆盖 Phase 2 第 4-7 步的 silent fallback 接入:
- LogicMiningPipeline._call_llm (P-01)
- LogicMiningPipeline.run 三个 stage parse failure (P-02/P-03/P-04)
- MarketLogicGenerator._call_llm (P-09)
- MarketLogicGenerator._structured_from_dict (P-12)
"""
import pytest

from QuantNodes.research.quant_alpha.logic_mining import (
    LogicMiningPipeline,
    LogicMiningStrictError,
    PipelineMetrics,
    StrictConfig,
)
from QuantNodes.research.quant_alpha.logic_mining.models import LogicAbstractionResult
from QuantNodes.research.quant_alpha.logic_mining.generator import (
    MarketLogicGenerator,
)


class FailingClient:
    """LLM client that always raises ConnectionError"""

    def complete(self, *, agent_id: str, prompt: str) -> str:
        raise ConnectionError(f"mocked fail for {agent_id}")


class BadJSONClient:
    """LLM client that returns non-JSON each call (3 layers fail)"""

    def complete(self, *, agent_id: str, prompt: str) -> str:
        return f"non-json trash for {agent_id} (no braces)"


class TestPipelineMetricsBasics:
    def test_default_empty(self):
        m = PipelineMetrics()
        assert m.total_failures() == 0
        d = m.to_dict()
        assert d["wiki_failures"] == 0
        assert d["inner_loop_failures"] == 0
        assert d["call_failures"] == {}

    def test_record_call_failure_increments(self):
        m = PipelineMetrics()
        m.record_call_failure("x")
        m.record_call_failure("x")
        m.record_call_failure("y")
        assert m.call_failures["x"] == 2
        assert m.call_failures["y"] == 1

    def test_record_parse_failure_tracks_max_layer(self):
        m = PipelineMetrics()
        m.record_parse_failure("a", 1)
        m.record_parse_failure("a", 2)
        m.record_parse_failure("a", 3)
        assert m.parse_failures["a"] == 3
        assert m.parse_layer_reached["a"] == 3

    def test_total_failures_sums_all_categories(self):
        m = PipelineMetrics()
        m.record_call_failure("a")
        m.record_call_failure("a")
        m.record_parse_failure("a", 1)
        m.record_structured_failure("b")
        m.record_wiki_failure()
        m.record_inner_loop_failure()
        assert m.total_failures() == 2 + 1 + 1 + 1 + 1


class TestStrictConfig:
    def test_default_all_off(self):
        s = StrictConfig()
        assert s.call is False
        assert s.parse is False
        assert s.structured is False

    def test_selective(self):
        s = StrictConfig(parse=True)
        assert s.call is False
        assert s.parse is True
        assert s.structured is False


class TestLogicMiningStrictError:
    def test_kind_and_context_preserved(self):
        err = LogicMiningStrictError("boom", kind="parse", agent_id="x", layer=2)
        assert err.kind == "parse"
        assert err.context["agent_id"] == "x"
        assert err.context["layer"] == 2
        assert "boom" in str(err)


class TestCallLLMFailure:
    """P-01: pipelines._call_llm 失败时 metrics.record_call_failure & strict.call 抛"""

    def test_no_client_returns_mock(self):
        m = PipelineMetrics()
        p = LogicMiningPipeline(llm_client=None, metrics=m)
        r = p.run("rank(close)", "alpha101")
        # mock 路径不应记录失败
        assert m.total_failures() == 0
        assert r.structured_logic is not None

    def test_failing_client_records_metric(self):
        m = PipelineMetrics()
        p = LogicMiningPipeline(llm_client=FailingClient(), metrics=m, strict=StrictConfig())
        p.run("rank(close)", "alpha101")
        # 3 个 stage 都失败 → 每个 agent_id 加 1
        assert m.call_failures["logic-mining-structure"] == 1
        assert m.call_failures["logic-mining-semantics"] == 1
        assert m.call_failures["logic-mining-abstraction"] == 1

    def test_strict_call_raises(self):
        m = PipelineMetrics()
        s = StrictConfig(call=True)
        p = LogicMiningPipeline(llm_client=FailingClient(), metrics=m, strict=s)
        with pytest.raises(LogicMiningStrictError) as exc_info:
            p.run("rank(close)", "alpha101")
        err = exc_info.value
        assert err.kind == "call"
        assert err.context["agent_id"] == "logic-mining-structure"


class TestParseFailureStages:
    """P-02 / P-03 / P-04: 三个 stage parse 失败时 metrics.parse_failures 计数 + result.parse_error"""

    def test_structure_parse_failure_metric(self):
        m = PipelineMetrics()
        p = LogicMiningPipeline(llm_client=BadJSONClient(), metrics=m)
        r = p.run("rank(close)", "alpha101")
        # 第 1 stage parse 失败 → 计数
        assert m.parse_failures["logic-mining-structure"] >= 1
        # result 携带 parse_error
        assert r.parse_error is not None
        assert "Cannot parse JSON" in r.parse_error
        # 因 strict=False, structured_logic 仍由 fallback 写出 (取决于 mock)
        # 但本测试只关注 failure 路径
        assert isinstance(r, LogicAbstractionResult)

    def test_strict_parse_raises_with_layer(self):
        m = PipelineMetrics()
        s = StrictConfig(parse=True)
        p = LogicMiningPipeline(llm_client=BadJSONClient(), metrics=m, strict=s)
        with pytest.raises(LogicMiningStrictError) as exc_info:
            p.run("rank(close)", "alpha101")
        err = exc_info.value
        assert err.kind == "parse"
        assert err.context["agent_id"] == "logic-mining-structure"
        assert err.context["layer"] >= 1


class TestMarketLogicGeneratorFailure:
    """P-09 / P-12: generator._call_llm + _structured_from_dict 接入"""

    def test_generator_call_failure_records(self):
        m = PipelineMetrics()
        gen = MarketLogicGenerator(
            llm_client=FailingClient(),
            metrics=m,
            strict=StrictConfig(),
            base_name="test_logic",
        )
        gen.generate(library=[], round_idx=1)
        assert m.call_failures["market-logic-generator"] == 1

    def test_generator_strict_call_raises(self):
        m = PipelineMetrics()
        gen = MarketLogicGenerator(
            llm_client=FailingClient(),
            metrics=m,
            strict=StrictConfig(call=True),
            base_name="test_logic",
        )
        with pytest.raises(LogicMiningStrictError) as exc_info:
            gen.generate(library=[], round_idx=1)
        assert exc_info.value.kind == "call"
        assert exc_info.value.context["agent_id"] == "market-logic-generator"
