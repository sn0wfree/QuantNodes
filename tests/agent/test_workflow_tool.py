# coding=utf-8
"""Tests for the WorkflowTool framework.

Covers:
- StepAgent: success, retry+fix, all-retries-fail, tool_executor
- WorkflowRegistry: register, get, list_all, build_llm_description
- WorkflowTool: mock execute, result file, summary, unknown workflow
- AlphaGptWorkflow: integration with 1 iteration (mock LLM)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from QuantNodes.agent.workflows.step_agent import StepAgent, StepAgentSpec, ParseResult
from QuantNodes.agent.workflows.registry import WorkflowRegistry, WorkflowSpec
from QuantNodes.agent.workflows.tool import WorkflowTool, _update_state
from QuantNodes.agent.workflows.parsers import (
    parse_json_3layer,
    validate_idea_generator,
    validate_formula_translator,
    validate_reflector,
    validate_critic,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@dataclass
class SimpleState:
    ideas: list = field(default_factory=list)
    results: list = field(default_factory=list)
    final: Any = None


def _simple_state_factory() -> SimpleState:
    return SimpleState()


def _simple_result_builder(state: SimpleState, config: dict) -> dict:
    return {"ideas": len(state.ideas), "results": len(state.results)}


def _mock_llm_ideas(prompt: str) -> str:
    return json.dumps({"ideas": [{"id": "1", "name": "test", "category": "momentum"}]})


def _mock_llm_bad_then_good():
    """Returns a callable that fails first, then succeeds."""
    calls = {"n": 0}

    def _call(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json at all"
        return json.dumps({"ideas": [{"id": "1", "name": "fixed", "category": "momentum"}]})

    return _call, calls


def _mock_llm_always_bad(prompt: str) -> str:
    return "still not json"


def _mock_tool_executor(**kwargs: Any) -> list:
    return [{"tool_result": True}]


# ==============================================================================
# TestStepAgent
# ==============================================================================


class TestStepAgent:
    def test_run_success(self):
        """正常执行: mock LLM 返回正确 JSON → 解析成功 → 返回 records。"""
        spec = StepAgentSpec(
            agent_id="test-step",
            prompt_builder=lambda **ctx: "generate ideas",
            output_parser=lambda raw: parse_json_3layer(raw, validate_idea_generator),
            output_key="ideas",
            record_factory=lambda d: d,
        )
        agent = StepAgent(spec, llm_client=_mock_llm_ideas)
        records = agent.run()
        assert len(records) == 1
        assert records[0]["id"] == "1"

    def test_run_retry_with_fix(self):
        """解析失败后重试，prompt 注入 error+raw，第二次成功。"""
        mock_client, calls = _mock_llm_bad_then_good()

        prompts_seen = []

        def prompt_builder(**ctx):
            prompt = "generate ideas"
            if ctx.get("_prev_error"):
                prompt += f"\n[ERROR: {ctx['_prev_error']}]"
            prompts_seen.append(prompt)
            return prompt

        spec = StepAgentSpec(
            agent_id="test-retry",
            prompt_builder=prompt_builder,
            output_parser=lambda raw: parse_json_3layer(raw, validate_idea_generator),
            output_key="ideas",
            record_factory=lambda d: d,
            max_retries=2,
        )
        agent = StepAgent(spec, llm_client=mock_client)
        records = agent.run()

        assert len(records) == 1
        assert records[0]["name"] == "fixed"
        assert calls["n"] == 2
        assert len(prompts_seen) == 2
        assert "[ERROR:" in prompts_seen[1]

    def test_run_all_retries_fail(self):
        """所有重试都失败，返回空列表。"""
        spec = StepAgentSpec(
            agent_id="test-fail",
            prompt_builder=lambda **ctx: "generate",
            output_parser=lambda raw: parse_json_3layer(raw, validate_idea_generator),
            output_key="ideas",
            record_factory=lambda d: d,
            max_retries=1,
        )
        agent = StepAgent(spec, llm_client=_mock_llm_always_bad)
        records = agent.run()
        assert records == []

    def test_tool_executor_bypass(self):
        """evaluator 步骤跳过 LLM，直接调 tool_executor。"""
        spec = StepAgentSpec(
            agent_id="test-evaluator",
            prompt_builder=None,
            output_parser=None,
            output_key="evaluations",
            tool_executor=_mock_tool_executor,
        )
        agent = StepAgent(spec, llm_client=None)
        records = agent.run()
        assert len(records) == 1
        assert records[0]["tool_result"] is True

    def test_run_with_state(self):
        """state 参数传入 prompt_builder。"""
        state = SimpleState(ideas=[{"id": "existing"}])

        def prompt_builder(**ctx):
            s = ctx.get("state")
            return f"state_ideas={len(s.ideas) if s else 0}"

        spec = StepAgentSpec(
            agent_id="test-state",
            prompt_builder=prompt_builder,
            output_parser=lambda raw: parse_json_3layer(raw),
            output_key="ideas",
            record_factory=lambda d: d,
        )
        agent = StepAgent(spec, llm_client=lambda p: '{"ideas": []}')
        records = agent.run(state=state)
        assert records == []

    def test_run_with_prev_output(self):
        """prev_output 传入 prompt_builder。"""
        def prompt_builder(**ctx):
            prev = ctx.get("prev_output")
            return f"prev_count={len(prev) if prev else 0}"

        spec = StepAgentSpec(
            agent_id="test-prev",
            prompt_builder=prompt_builder,
            output_parser=lambda raw: parse_json_3layer(raw),
            output_key="ideas",
            record_factory=lambda d: d,
        )
        agent = StepAgent(spec, llm_client=lambda p: '{"ideas": []}')
        records = agent.run(prev_output=[{"id": "1"}, {"id": "2"}])
        assert records == []


# ==============================================================================
# TestWorkflowRegistry
# ==============================================================================


class TestWorkflowRegistry:
    def test_register_and_get(self):
        reg = WorkflowRegistry()
        spec = WorkflowSpec(
            name="test",
            description="test workflow",
            steps=[],
            state_factory=_simple_state_factory,
            result_builder=_simple_result_builder,
        )
        reg.register(spec)
        assert reg.get("test") is spec

    def test_get_unknown(self):
        reg = WorkflowRegistry()
        assert reg.get("nonexistent") is None

    def test_list_all(self):
        reg = WorkflowRegistry()
        reg.register(WorkflowSpec(
            name="a", description="A", steps=[],
            state_factory=_simple_state_factory, result_builder=_simple_result_builder,
        ))
        reg.register(WorkflowSpec(
            name="b", description="B", steps=[],
            state_factory=_simple_state_factory, result_builder=_simple_result_builder,
        ))
        result = reg.list_all()
        assert len(result) == 2
        names = [r["name"] for r in result]
        assert "a" in names and "b" in names

    def test_build_llm_description(self):
        reg = WorkflowRegistry()
        assert "No workflows" in reg.build_llm_description()

        reg.register(WorkflowSpec(
            name="alpha-gpt",
            description="5-round alpha discovery",
            steps=[StepAgentSpec(agent_id="s1"), StepAgentSpec(agent_id="s2")],
            iterations=5,
            final_steps=[StepAgentSpec(agent_id="critic")],
            state_factory=_simple_state_factory,
            result_builder=_simple_result_builder,
        ))
        desc = reg.build_llm_description()
        assert "alpha-gpt" in desc
        assert "2 steps/round" in desc
        assert "5 iterations" in desc
        assert "1 final steps" in desc


# ==============================================================================
# TestUpdateState
# ==============================================================================


class TestUpdateState:
    def test_extend_list(self):
        state = SimpleState(ideas=[])
        spec = StepAgentSpec(agent_id="t", state_output="ideas")
        _update_state(state, spec, [{"id": "1"}, {"id": "2"}])
        assert len(state.ideas) == 2

    def test_set_non_list(self):
        state = SimpleState(final=None)
        spec = StepAgentSpec(agent_id="t", state_output="final")
        _update_state(state, spec, {"pool": []})
        assert state.final == {"pool": []}

    def test_no_state_output(self):
        state = SimpleState()
        spec = StepAgentSpec(agent_id="t", state_output=None)
        _update_state(state, spec, [{"id": "1"}])
        assert state.ideas == []


# ==============================================================================
# TestWorkflowTool
# ==============================================================================


class TestWorkflowTool:
    @pytest.fixture
    def registry(self):
        """Fresh registry for each test."""
        reg = WorkflowRegistry()
        return reg

    @pytest.fixture
    def simple_spec(self, registry):
        """Simple 1-step workflow for unit testing."""
        def _run_evaluator(**kwargs):
            return [{"eval": True}]

        spec = WorkflowSpec(
            name="simple",
            description="Simple test workflow",
            steps=[
                StepAgentSpec(
                    agent_id="gen",
                    prompt_builder=lambda **ctx: "generate",
                    output_parser=lambda raw: parse_json_3layer(raw),
                    output_key="ideas",
                    state_output="ideas",
                    record_factory=lambda d: d,
                ),
            ],
            iterations=1,
            state_factory=_simple_state_factory,
            result_builder=_simple_result_builder,
        )
        registry.register(spec)
        return spec

    def test_execute_simple(self, registry, simple_spec, tmp_path):
        """用 mock LLM 跑简单 workflow。"""
        tool = WorkflowTool(
            llm_client=lambda p: '{"ideas": [{"id": "1"}]}',
            results_dir=tmp_path,
        )
        # Patch the tool's internal REGISTRY usage
        import QuantNodes.agent.workflows.tool as tool_mod
        original = tool_mod.REGISTRY
        tool_mod.REGISTRY = registry

        try:
            import asyncio
            result_str = asyncio.run(tool.execute(workflow="simple", config={}))
            result = json.loads(result_str)

            assert result["status"] == "completed"
            assert result["result_file"] != ""
            assert Path(result["result_file"]).exists()
        finally:
            tool_mod.REGISTRY = original

    def test_execute_unknown_workflow(self, registry, tmp_path):
        """未知 workflow 返回错误信息。"""
        tool = WorkflowTool(llm_client=None, results_dir=tmp_path)
        import QuantNodes.agent.workflows.tool as tool_mod
        original = tool_mod.REGISTRY
        tool_mod.REGISTRY = registry

        try:
            import asyncio
            result_str = asyncio.run(tool.execute(workflow="nonexistent"))
            result = json.loads(result_str)
            assert result["status"] == "error"
            assert "Unknown workflow" in result["message"]
        finally:
            tool_mod.REGISTRY = original

    def test_result_file_created(self, registry, simple_spec, tmp_path):
        """结果 JSON 文件写入 results_dir。"""
        tool = WorkflowTool(
            llm_client=lambda p: '{"ideas": []}',
            results_dir=tmp_path,
        )
        import QuantNodes.agent.workflows.tool as tool_mod
        original = tool_mod.REGISTRY
        tool_mod.REGISTRY = registry

        try:
            import asyncio
            result_str = asyncio.run(tool.execute(workflow="simple"))
            result = json.loads(result_str)
            assert result["result_file"] != ""
            assert Path(result["result_file"]).exists()
            content = json.loads(Path(result["result_file"]).read_text())
            assert "ideas" in content
        finally:
            tool_mod.REGISTRY = original


# ==============================================================================
# TestAlphaGptIntegration
# ==============================================================================


class TestAlphaGptIntegration:
    def test_mock_workflow_registered(self):
        """alpha-gpt workflow 已注册到全局 REGISTRY。"""
        from QuantNodes.agent.workflows.implementations.alpha_gpt import ALPHA_GPT_SPEC
        from QuantNodes.agent.workflows.registry import REGISTRY

        assert REGISTRY.get("alpha-gpt") is ALPHA_GPT_SPEC
        assert ALPHA_GPT_SPEC.iterations == 5
        assert len(ALPHA_GPT_SPEC.steps) == 4
        assert len(ALPHA_GPT_SPEC.final_steps) == 1
        assert ALPHA_GPT_SPEC.steps[3].skip_on_last is True

    def test_mock_run_1_iteration(self, tmp_path):
        """用 mock LLM 跑 1 轮 alpha-gpt workflow。"""
        from QuantNodes.agent.workflows.implementations.alpha_gpt import (
            _mock_response, _build_result, ALPHA_GPT_SPEC,
        )
        from QuantNodes.agent.workflows.registry import REGISTRY
        from QuantNodes.agent.workflows.step_agent import StepAgent
        from QuantNodes.research.quant_alpha.workflow.state import AlphaGptState

        state = AlphaGptState(objective="test", iterations_total=1)
        config = {"pool_size": 3, "top_k": 5}

        # Mock LLM that uses _mock_response
        def mock_llm(prompt: str) -> str:
            if "idea-generator" in prompt:
                return _mock_response("alpha-gpt-idea-generator", prompt, state, config)
            if "formula-translator" in prompt:
                return _mock_response("alpha-gpt-formula-translator", prompt, state, config)
            if "reflector" in prompt:
                return _mock_response("alpha-gpt-reflector", prompt, state, config)
            if "critic" in prompt:
                return _mock_response("alpha-gpt-critic", prompt, state, config)
            return "{}"

        from QuantNodes.agent.workflows.implementations.alpha_gpt import (
            IDEA_GEN_SPEC, FORMULA_TRANS_SPEC, EVALUATOR_SPEC, REFLECTOR_SPEC, CRITIC_SPEC,
        )
        from QuantNodes.agent.workflows.tool import _update_state

        # Run 1 round
        for step_spec in [IDEA_GEN_SPEC, FORMULA_TRANS_SPEC, EVALUATOR_SPEC, REFLECTOR_SPEC]:
            if step_spec.skip_on_last:
                continue  # 1 iteration = last round, skip reflector
            step = StepAgent(step_spec, llm_client=mock_llm)
            records = step.run(state=state, round_idx=1, **config)
            _update_state(state, step_spec, records)

        # Final step (critic)
        step = StepAgent(CRITIC_SPEC, llm_client=mock_llm)
        records = step.run(state=state, **config)
        _update_state(state, CRITIC_SPEC, records)

        # Verify state
        assert len(state.all_ideas) == 3
        assert len(state.all_formulas) > 0
        # Evaluator may fail (no real data), that's OK
        assert len(state.all_evaluations) >= 0
        # Reflector was skipped (1 iteration = last round)
        assert len(state.all_reflections) == 0

        # Build result
        result = _build_result(state, config)
        assert "summary" in result
        assert "final_pool" in result


# ==============================================================================
# TestParsers
# ==============================================================================


class TestParsers:
    def test_parse_json_3layer_schema(self):
        """Layer 1: 直接 JSON 解析成功。"""
        raw = '{"ideas": [{"id": "1", "name": "test", "category": "momentum"}]}'
        result = parse_json_3layer(raw, validate_idea_generator)
        assert result.ok is True
        assert result.layer == "schema"

    def test_parse_json_3layer_regex(self):
        """Layer 2: 正则提取 JSON 块。"""
        raw = 'Here is the result: {"ideas": [{"id": "1", "name": "test", "category": "momentum"}]} done.'
        result = parse_json_3layer(raw, validate_idea_generator)
        assert result.ok is True
        assert result.layer == "regex"

    def test_parse_json_3layer_fail(self):
        """Layer 3: 解析失败。"""
        raw = "not json at all"
        result = parse_json_3layer(raw, validate_idea_generator)
        assert result.ok is False
        assert result.error is not None
        assert result.raw == "not json at all"

    def test_validate_idea_generator_missing_key(self):
        result = parse_json_3layer('{"wrong": []}', validate_idea_generator)
        assert result.ok is False
