# coding=utf-8
"""端到端自动化因子挖掘测试。

覆盖：
- Mock LLM 模式（无 API 成本）
- 真实 LLM 模式（需要 API key）
- Alpha-GPT 工作流
- MCTS 工作流
- 完整流水线
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import polars as pl
import pytest

from QuantNodes.agent.workflows.registry import REGISTRY
from QuantNodes.agent.workflows.tool import WorkflowTool
from QuantNodes.agent.workflows.implementations.mcts import MCTSState


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成测试数据"""
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            close = float(np.random.randn() * 5 + 100)
            rows.append({
                "date": date,
                "code": code,
                "close": close,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "vol": 1000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def data_path(sample_data: pl.DataFrame, tmp_path: Path) -> str:
    """保存测试数据到 parquet 文件"""
    path = tmp_path / "test_data.parquet"
    sample_data.write_parquet(path)
    return str(path)


class MockLLMClient:
    """Mock LLM 客户端，返回预设的 JSON 响应。"""

    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.calls = []

    def complete(self, agent_id: str, prompt: str) -> str:
        self.call_count += 1
        self.calls.append({"agent_id": agent_id, "prompt": prompt[:200]})

        # 检查是否有预设响应
        for key, response in self.responses.items():
            if key in agent_id:
                return response

        # 默认响应
        return self._default_response(agent_id)

    def _default_response(self, agent_id: str) -> str:
        if "idea-generator" in agent_id:
            return json.dumps({
                "ideas": [
                    {"id": "IDEA-1", "name": "reversal", "category": "reversal",
                     "description": "20-day reversal", "expected_direction": "long",
                     "suggested_lookback": 20, "a_share_compatible": True}
                ]
            })
        if "formula-translator" in agent_id:
            return json.dumps({
                "formulas": [
                    {"formula": "sub(close, ts_mean(close, 20))", "idea_id": "IDEA-1",
                     "complexity": 2, "a_share_compatible": True}
                ]
            })
        if "reflector" in agent_id:
            return json.dumps({
                "formula_feedback": [],
                "next_round_suggestions": {"preferred_operators": ["ts_rank"]}
            })
        if "critic" in agent_id:
            return json.dumps({"final_pool": []})
        if "seed-generator" in agent_id:
            return json.dumps({
                "seed_formulas": [
                    {"formula": "rank(close)", "category": "wrap", "rationale": "cross-sectional ranking"},
                    {"formula": "ts_mean(close, 20)", "category": "window", "rationale": "20-day mean"},
                ]
            })
        if "mcts-reflector" in agent_id:
            return json.dumps({
                "formula_feedback": [],
                "next_round_suggestions": {
                    "preferred_operators": ["ts_rank", "ts_decay_linear"],
                    "preferred_windows": [10, 20],
                }
            })
        return "{}"


# ==============================================================================
# TestAlphaGptMock
# ==============================================================================


class TestAlphaGptMock:
    """Alpha-GPT Mock LLM 测试（无 API 成本）"""

    def test_workflow_registered(self):
        """Alpha-GPT workflow 已注册"""
        spec = REGISTRY.get("alpha-gpt")
        assert spec is not None
        assert spec.name == "alpha-gpt"

    def test_mock_run_1_iteration(self, sample_data, data_path):
        """Mock 运行 1 轮"""
        mock_client = MockLLMClient()
        tool = WorkflowTool(llm_client=mock_client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="alpha-gpt",
            config={
                "data_path": data_path,
                "iterations": 1,
                "pool_size": 2,
                "top_k": 5,
                "objective": "test reversal",
            },
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        assert "summary" in result_data
        assert mock_client.call_count > 0

    def test_mock_run_3_iterations(self, sample_data, data_path):
        """Mock 运行 3 轮"""
        mock_client = MockLLMClient()
        tool = WorkflowTool(llm_client=mock_client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="alpha-gpt",
            config={
                "data_path": data_path,
                "iterations": 3,
                "pool_size": 2,
                "top_k": 5,
                "objective": "test momentum",
            },
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        # 3 轮 × 3 步 (idea + formula + reflector, evaluator 跳过 LLM) + critic = 10
        # 但 reflector 在最后一轮跳过，所以是 3*3 - 1 + 1 = 9
        assert mock_client.call_count >= 9


# ==============================================================================
# TestMCTSMock
# ==============================================================================


class TestMCTSMock:
    """MCTS Mock LLM 测试（无 API 成本）"""

    def test_workflow_registered(self):
        """MCTS workflow 已注册"""
        spec = REGISTRY.get("alpha-mcts")
        assert spec is not None
        assert spec.name == "alpha-mcts"

    def test_mock_run_1_iteration(self, sample_data, data_path):
        """Mock 运行 1 轮"""
        mock_client = MockLLMClient()
        tool = WorkflowTool(llm_client=mock_client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="alpha-mcts",
            config={
                "data_path": data_path,
                "iterations": 1,
                "top_k": 5,
                "objective": "test reversal",
                "compute_ic_ir": False,  # 跳过 IC/IR 以加速
            },
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        assert "summary" in result_data

    def test_mock_run_3_iterations(self, sample_data, data_path):
        """Mock 运行 3 轮"""
        mock_client = MockLLMClient()
        tool = WorkflowTool(llm_client=mock_client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="alpha-mcts",
            config={
                "data_path": data_path,
                "iterations": 3,
                "top_k": 5,
                "objective": "test momentum",
                "compute_ic_ir": False,
            },
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "completed"


# ==============================================================================
# TestMCTSState
# ==============================================================================


class TestMCTSState:
    """MCTSState 测试"""

    def test_state_initialization(self):
        """状态初始化"""
        state = MCTSState(objective="test")
        assert state.objective == "test"
        assert state.iterations_total == 3
        assert state.seed_formulas == []
        assert state.best_k_nodes == []
        assert state.all_reflections == []
        assert state.all_best_nodes == []

    def test_state_mutation(self):
        """状态可变"""
        state = MCTSState(objective="test")
        state.seed_formulas = ["rank(close)", "ts_mean(close, 20)"]
        state.best_k_nodes = [{"formula": "rank(close)", "score": 0.8}]
        state.search_stats = {"total_nodes": 10, "valid_nodes": 5}

        assert len(state.seed_formulas) == 2
        assert len(state.best_k_nodes) == 1
        assert state.search_stats["total_nodes"] == 10


# ==============================================================================
# TestFullPipelineMock
# ==============================================================================


class TestFullPipelineMock:
    """完整流水线 Mock 测试（无 API 成本）"""

    def test_alphagpt_then_mcts(self, sample_data, data_path):
        """Alpha-GPT + MCTS 联合测试"""
        mock_client = MockLLMClient()
        tool = WorkflowTool(llm_client=mock_client)

        import asyncio

        # 1. 运行 Alpha-GPT
        result1 = asyncio.run(tool.execute(
            workflow="alpha-gpt",
            config={
                "data_path": data_path,
                "iterations": 1,
                "pool_size": 2,
                "top_k": 5,
                "objective": "test reversal",
            },
        ))
        result1_data = json.loads(result1)
        assert result1_data["status"] == "completed"

        # 2. 运行 MCTS
        result2 = asyncio.run(tool.execute(
            workflow="alpha-mcts",
            config={
                "data_path": data_path,
                "iterations": 1,
                "top_k": 5,
                "objective": "test reversal",
                "compute_ic_ir": False,
            },
        ))
        result2_data = json.loads(result2)
        assert result2_data["status"] == "completed"


# ==============================================================================
# TestRealLLM (需要 API key)
# ==============================================================================


@pytest.mark.skipif(
    not os.environ.get("QUANTNODES__LLM__API_KEY"),
    reason="需要 QUANTNODES__LLM__API_KEY 环境变量"
)
class TestRealLLM:
    """真实 LLM 测试（需要 API key）"""

    def _get_real_client(self):
        """获取真实 LLM 客户端"""
        from QuantNodes.ai.llm.gateway import LLMGateway
        return LLMGateway()

    def test_alphagpt_real_1_iteration(self, sample_data, data_path):
        """真实 LLM 运行 Alpha-GPT 1 轮"""
        client = self._get_real_client()
        tool = WorkflowTool(llm_client=client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="alpha-gpt",
            config={
                "data_path": data_path,
                "iterations": 1,
                "pool_size": 2,
                "top_k": 5,
                "objective": "capture A-share reversal effect",
            },
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        assert "summary" in result_data

    def test_mcts_real_1_iteration(self, sample_data, data_path):
        """真实 LLM 运行 MCTS 1 轮"""
        client = self._get_real_client()
        tool = WorkflowTool(llm_client=client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="alpha-mcts",
            config={
                "data_path": data_path,
                "iterations": 1,
                "top_k": 5,
                "objective": "capture A-share momentum effect",
                "compute_ic_ir": False,  # 跳过 IC/IR 以加速
            },
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "completed"


# ==============================================================================
# TestWorkflowToolIntegration
# ==============================================================================


class TestWorkflowToolIntegration:
    """WorkflowTool 集成测试"""

    def test_list_workflows(self):
        """列出所有 workflow"""
        all_workflows = REGISTRY.list_all()
        names = [w["name"] for w in all_workflows]
        assert "alpha-gpt" in names
        assert "alpha-mcts" in names

    def test_workflow_description(self):
        """workflow 描述"""
        spec = REGISTRY.get("alpha-mcts")
        assert "MCTS" in spec.description
        assert "seed generation" in spec.description

    def test_workflow_steps(self):
        """workflow 步骤"""
        spec = REGISTRY.get("alpha-mcts")
        assert len(spec.steps) == 3
        assert spec.steps[0].agent_id == "mcts-seed-generator"
        assert spec.steps[1].agent_id == "mcts-search"
        assert spec.steps[2].agent_id == "mcts-reflector"

    def test_unknown_workflow(self, data_path):
        """未知 workflow 返回错误"""
        mock_client = MockLLMClient()
        tool = WorkflowTool(llm_client=mock_client)

        import asyncio
        result = asyncio.run(tool.execute(
            workflow="unknown-workflow",
            config={"data_path": data_path},
        ))

        result_data = json.loads(result)
        assert result_data["status"] == "error"
        assert "Unknown workflow" in result_data["message"]
