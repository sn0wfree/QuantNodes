# coding=utf-8
"""Tests for MCTS WorkflowSpec.

覆盖：
- MCTSState：状态初始化、字段
- MCTS prompt builders：种子生成、反思
- MCTS validators：种子验证、反思验证
- MCTS record factories：种子工厂、反思工厂
- MCTS tool_executor：MCTS 搜索执行
- MCTS result builder：结果构建
- MCTS WorkflowSpec：注册、描述、端到端
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest

from QuantNodes.agent.workflows.implementations.mcts import (
    MCTSState,
    MCTS_SPEC,
    MCTS_SEED_GEN_SPEC,
    MCTS_SEARCH_SPEC,
    MCTS_REFLECT_SPEC,
    _build_seed_prompt,
    _build_reflect_prompt,
    _validate_seeds,
    _validate_mcts_reflection,
    _seed_factory,
    _reflection_factory,
    _run_mcts_search,
    _build_mcts_result,
    _infer_category,
    _mock_mcts_response,
)
from QuantNodes.agent.workflows.registry import REGISTRY
from QuantNodes.research.quant_alpha.workflow.state import ReflectionRecord


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def mcts_state() -> MCTSState:
    """MCTSState 实例"""
    return MCTSState(objective="test objective")


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成测试数据"""
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            rows.append({
                "date": date,
                "code": code,
                "close": float(np.random.randn() * 5 + 100),
                "open": float(np.random.randn() * 5 + 100),
                "high": float(np.random.randn() * 5 + 102),
                "low": float(np.random.randn() * 5 + 98),
                "vol": float(np.random.randint(1000, 5000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


# ==============================================================================
# TestMCTSState
# ==============================================================================


class TestMCTSState:
    """MCTSState 测试"""

    def test_initial_state(self):
        """初始状态正确"""
        state = MCTSState(objective="test")
        assert state.objective == "test"
        assert state.iterations_total == 3
        assert state.round_idx_hint == 1
        assert state.seed_formulas == []
        assert state.best_k_nodes == []
        assert state.search_stats == {}
        assert state.all_reflections == []
        assert state.all_best_nodes == []

    def test_field_mutation(self):
        """字段可变"""
        state = MCTSState(objective="test")
        state.seed_formulas = ["rank(close)", "ts_mean(close, 20)"]
        state.best_k_nodes = [{"formula": "rank(close)", "score": 0.8}]
        state.search_stats = {"total_nodes": 10, "valid_nodes": 5}

        assert len(state.seed_formulas) == 2
        assert len(state.best_k_nodes) == 1
        assert state.search_stats["total_nodes"] == 10


# ==============================================================================
# TestMCTSPromptBuilders
# ==============================================================================


class TestMCTSPromptBuilders:
    """Prompt builders 测试"""

    def test_build_seed_prompt_basic(self):
        """基本种子提示"""
        prompt = _build_seed_prompt(
            state=None,
            round_idx=1,
            objective="捕捉反转效应",
            data_columns=["close", "volume"],
        )
        assert "mcts-seed-generator.md" in prompt
        assert "捕捉反转效应" in prompt
        assert "close" in prompt
        assert "volume" in prompt
        assert "STRICT JSON" in prompt

    def test_build_seed_prompt_with_reflection(self):
        """带反思的种子提示"""
        state = MCTSState(objective="test")
        state.all_reflections.append(
            ReflectionRecord(
                round_idx=1,
                verdicts=[],
                suggestions={"preferred_operators": ["ts_rank"]},
            )
        )
        prompt = _build_seed_prompt(state=state, round_idx=2, objective="test")
        assert "previous_reflection" in prompt

    def test_build_seed_prompt_with_retry(self):
        """带重试的种子提示"""
        prompt = _build_seed_prompt(
            state=None,
            round_idx=1,
            objective="test",
            _prev_error="Invalid JSON",
            _prev_raw="not json",
        )
        assert "previous response was not valid JSON" in prompt
        assert "Invalid JSON" in prompt

    def test_build_reflect_prompt_basic(self):
        """基本反思提示"""
        state = MCTSState(objective="test")
        state.search_stats = {"total_nodes": 10, "valid_nodes": 5}
        state.best_k_nodes = [{"formula": "rank(close)", "overall_score": 0.8}]

        prompt = _build_reflect_prompt(state=state, round_idx=1)
        assert "mcts-reflector.md" in prompt
        assert "round 1" in prompt

    def test_build_reflect_prompt_with_retry(self):
        """带重试的反思提示"""
        prompt = _build_reflect_prompt(
            state=None,
            round_idx=1,
            _prev_error="Invalid JSON",
            _prev_raw="not json",
        )
        assert "previous response was not valid JSON" in prompt


# ==============================================================================
# TestMCTSValidators
# ==============================================================================


class TestMCTSValidators:
    """Validators 测试"""

    def test_validate_seeds_valid(self):
        """有效种子"""
        data = {
            "seed_formulas": [
                {"formula": "rank(close)", "category": "wrap"},
                {"formula": "ts_mean(close, 20)", "category": "window"},
            ]
        }
        assert _validate_seeds(data) is True

    def test_validate_seeds_empty(self):
        """空种子列表"""
        data = {"seed_formulas": []}
        assert _validate_seeds(data) is True

    def test_validate_seeds_missing_formula(self):
        """缺少 formula 字段"""
        data = {
            "seed_formulas": [
                {"category": "wrap"},  # 缺少 formula
            ]
        }
        assert _validate_seeds(data) is False

    def test_validate_seeds_not_dict(self):
        """非字典输入"""
        assert _validate_seeds("not a dict") is False
        assert _validate_seeds(123) is False

    def test_validate_seeds_missing_key(self):
        """缺少 seed_formulas 键"""
        data = {"other_key": []}
        assert _validate_seeds(data) is False

    def test_validate_mcts_reflection_valid(self):
        """有效反思"""
        data = {
            "formula_feedback": [{"formula": "rank(close)", "verdict": "keep"}],
            "next_round_suggestions": {"preferred_operators": ["ts_rank"]},
        }
        assert _validate_mcts_reflection(data) is True

    def test_validate_mcts_reflection_minimal(self):
        """最小反思"""
        data = {"formula_feedback": []}
        assert _validate_mcts_reflection(data) is True

    def test_validate_mcts_reflection_only_suggestions(self):
        """只有建议"""
        data = {"next_round_suggestions": {}}
        assert _validate_mcts_reflection(data) is True

    def test_validate_mcts_reflection_invalid(self):
        """无效反思"""
        assert _validate_mcts_reflection({}) is False
        assert _validate_mcts_reflection("not a dict") is False


# ==============================================================================
# TestMCTSRecordFactories
# ==============================================================================


class TestMCTSRecordFactories:
    """Record factories 测试"""

    def test_seed_factory(self):
        """种子工厂"""
        assert _seed_factory({"formula": "rank(close)"}) == "rank(close)"
        assert _seed_factory({"other": "value"}) == ""

    def test_reflection_factory(self):
        """反思工厂"""
        d = {
            "formula_feedback": [{"formula": "rank(close)", "verdict": "keep"}],
            "next_round_suggestions": {"preferred_operators": ["ts_rank"]},
        }
        record = _reflection_factory(d, round_idx=2)
        assert isinstance(record, ReflectionRecord)
        assert record.round_idx == 2
        assert len(record.verdicts) == 1
        assert record.suggestions["preferred_operators"] == ["ts_rank"]


# ==============================================================================
# TestMCTSMockResponse
# ==============================================================================


class TestMCTSMockResponse:
    """Mock LLM 响应测试"""

    def test_mock_seed_generator(self):
        """种子生成器 mock"""
        state = MCTSState(objective="test")
        response = _mock_mcts_response(
            "mcts-seed-generator", "test prompt", state=state
        )
        data = json.loads(response)
        assert "seed_formulas" in data
        assert len(data["seed_formulas"]) > 0
        assert "formula" in data["seed_formulas"][0]

    def test_mock_reflector(self):
        """反思器 mock"""
        state = MCTSState(objective="test")
        response = _mock_mcts_response(
            "mcts-reflector", "test prompt", state=state
        )
        data = json.loads(response)
        assert "formula_feedback" in data
        assert "next_round_suggestions" in data

    def test_mock_unknown(self):
        """未知 agent mock"""
        response = _mock_mcts_response("unknown-agent", "test prompt")
        data = json.loads(response)
        assert data == {}


# ==============================================================================
# TestMCTSResultBuilder
# ==============================================================================


class TestMCTSResultBuilder:
    """Result builder 测试"""

    def test_build_result_empty(self):
        """空结果"""
        state = MCTSState(objective="test")
        config = {"top_k": 10}
        result = _build_mcts_result(state, config)

        assert result["objective"] == "test"
        assert result["total_formulas"] == 0
        assert result["final_pool"] == []
        assert result["summary"]["selected"] == 0

    def test_build_result_with_nodes(self):
        """有节点的结果"""
        state = MCTSState(objective="test")
        state.all_best_nodes = [
            {
                "entry_id": "node-1",
                "formula": "rank(close)",
                "overall_score": 0.8,
                "dimension_scores": {"execution": 1.0, "shape": 1.0},
                "depth": 1,
                "metadata": {"ic_mean": 0.05, "ir": 0.8},
            },
            {
                "entry_id": "node-2",
                "formula": "ts_mean(close, 20)",
                "overall_score": 0.6,
                "dimension_scores": {"execution": 1.0, "shape": 0.5},
                "depth": 2,
                "metadata": {"ic_mean": 0.03, "ir": 0.5},
            },
        ]
        state.search_stats = {"total_nodes": 10, "valid_nodes": 5}

        config = {"top_k": 10}
        result = _build_mcts_result(state, config)

        assert result["objective"] == "test"
        assert result["total_formulas"] == 10
        assert len(result["final_pool"]) == 2
        assert result["final_pool"][0]["rank"] == 1
        assert result["final_pool"][0]["formula"] == "rank(close)"
        assert result["final_pool"][0]["ir"] == 0.8

    def test_build_result_dedup(self):
        """去重"""
        state = MCTSState(objective="test")
        state.all_best_nodes = [
            {
                "entry_id": "node-1",
                "formula": "rank(close)",
                "overall_score": 0.8,
                "dimension_scores": {},
                "depth": 1,
                "metadata": {"ic_mean": 0.05, "ir": 0.8},
            },
            {
                "entry_id": "node-2",
                "formula": "rank(close)",  # 重复
                "overall_score": 0.7,
                "dimension_scores": {},
                "depth": 2,
                "metadata": {"ic_mean": 0.03, "ir": 0.5},
            },
        ]
        state.search_stats = {"total_nodes": 10, "valid_nodes": 5}

        config = {"top_k": 10}
        result = _build_mcts_result(state, config)

        assert len(result["final_pool"]) == 1
        assert result["final_pool"][0]["ir"] == 0.8

    def test_build_result_top_k(self):
        """top_k 限制"""
        state = MCTSState(objective="test")
        for i in range(20):
            state.all_best_nodes.append({
                "entry_id": f"node-{i}",
                "formula": f"formula_{i}",
                "overall_score": 0.5 + i * 0.01,
                "dimension_scores": {},
                "depth": 1,
                "metadata": {"ic_mean": 0.01 * i, "ir": 0.5 + i * 0.01},
            })
        state.search_stats = {"total_nodes": 20, "valid_nodes": 20}

        config = {"top_k": 5}
        result = _build_mcts_result(state, config)

        assert len(result["final_pool"]) == 5
        # 按 overall_score 降序（对应 ir）
        assert result["final_pool"][0]["ir"] > result["final_pool"][4]["ir"]


# ==============================================================================
# TestMCTSInferCategory
# ==============================================================================


class TestMCTSInferCategory:
    """_infer_category 测试"""

    def test_wrap(self):
        assert _infer_category({"formula": "rank(close)"}) == "wrap"
        assert _infer_category({"formula": "zscore(volume)"}) == "wrap"

    def test_window(self):
        assert _infer_category({"formula": "ts_mean(close, 20)"}) == "window"
        assert _infer_category({"formula": "ts_std(volume, 10)"}) == "window"

    def test_unary(self):
        assert _infer_category({"formula": "abs(returns)"}) == "unary"
        assert _infer_category({"formula": "log(volume)"}) == "unary"

    def test_diff(self):
        assert _infer_category({"formula": "close - ts_mean(close, 20)"}) == "diff"

    def test_ratio(self):
        assert _infer_category({"formula": "close / ts_lag(close, 20) - 1"}) == "ratio"

    def test_unknown(self):
        assert _infer_category({"formula": "custom_op(close)"}) == "unknown"


# ==============================================================================
# TestMCTSWorkflowSpecRegistration
# ==============================================================================


class TestMCTSWorkflowSpecRegistration:
    """WorkflowSpec 注册测试"""

    def test_mcts_spec_registered(self):
        """MCTS_SPEC 已注册"""
        spec = REGISTRY.get("alpha-mcts")
        assert spec is not None
        assert spec.name == "alpha-mcts"

    def test_mcts_spec_description(self):
        """MCTS_SPEC 描述"""
        spec = REGISTRY.get("alpha-mcts")
        assert "MCTS" in spec.description
        assert "seed generation" in spec.description

    def test_mcts_spec_steps(self):
        """MCTS_SPEC 步骤"""
        spec = REGISTRY.get("alpha-mcts")
        assert len(spec.steps) == 3
        assert spec.steps[0].agent_id == "mcts-seed-generator"
        assert spec.steps[1].agent_id == "mcts-search"
        assert spec.steps[2].agent_id == "mcts-reflector"

    def test_mcts_spec_iterations(self):
        """MCTS_SPEC 迭代次数"""
        spec = REGISTRY.get("alpha-mcts")
        assert spec.iterations == 3

    def test_mcts_spec_in_list_all(self):
        """MCTS_SPEC 在 list_all 中"""
        all_workflows = REGISTRY.list_all()
        names = [w["name"] for w in all_workflows]
        assert "alpha-mcts" in names

    def test_alpha_gpt_also_registered(self):
        """Alpha-GPT 也已注册"""
        spec = REGISTRY.get("alpha-gpt")
        assert spec is not None
        assert spec.name == "alpha-gpt"
