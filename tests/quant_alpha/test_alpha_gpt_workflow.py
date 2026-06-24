# coding=utf-8
"""
test_alpha_gpt_workflow.py - AlphaGptWorkflow 协调器测试
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.workflow import (
    AlphaGptConfig,
    AlphaGptWorkflow,
    AlphaGptResult,
    AlphaGptState,
    IdeaRecord,
    FormulaRecord,
    EvaluationRecord,
    FinalFormulaRecord,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 31)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            close = float(np.random.randn() * 5 + 100)
            rows.append({
                "date": date, "code": code, "close": close,
                "open": close + np.random.randn() * 0.5,
                "high": close + abs(np.random.randn()),
                "low": close - abs(np.random.randn()),
                "vol": float(np.random.randint(1000, 5000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


class MockLLMClient:
    """Mock LLM：可定制返回"""

    def __init__(self, responses: Dict[str, str]):
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, agent_id: str, prompt: str) -> str:
        self.calls.append((agent_id, prompt))
        return self.responses.get(
            agent_id,
            json.dumps({"empty": True}),
        )


# ==============================================================================
# Config + State
# ==============================================================================


class TestAlphaGptConfig:
    def test_defaults(self):
        c = AlphaGptConfig(objective="test")
        assert c.iterations == 5
        assert c.pool_size == 10
        assert c.top_k == 10
        assert c.llm_provider == "deepseek"

    def test_custom(self):
        c = AlphaGptConfig(
            objective="X",
            iterations=3,
            pool_size=5,
            top_k=20,
            enable_backtest=True,
        )
        assert c.iterations == 3
        assert c.top_k == 20
        assert c.enable_backtest is True


class TestStateRecords:
    def test_idea_record_from_dict(self):
        d = {"id": "I1", "name": "test", "category": "reversal"}
        r = IdeaRecord.from_dict(d, round_idx=1)
        assert r.id == "I1"
        assert r.round_idx == 1

    def test_evaluation_record_to_dict(self):
        r = EvaluationRecord(
            formula_id="F1",
            formula="x",
            status="success",
            ic_mean=0.05,
            ir=1.5,
            ic_decay={1: 0.05, 5: 0.03},
        )
        d = r.to_dict()
        assert d["status"] == "success"
        assert d["ic_decay"]["1"] == 0.05

    def test_final_formula_record_from_dict(self):
        d = {
            "formula": "rank(x)",
            "formula_id": "F1",
            "metrics": {"ir": 2.0, "ic_mean": 0.05},
            "selection_reason": "good",
        }
        r = FinalFormulaRecord.from_dict(d, rank=1)
        assert r.rank == 1
        assert r.ir == 2.0


# ==============================================================================
# Workflow E2E（mock LLM）
# ==============================================================================


class TestWorkflowE2E:
    def test_basic_run_with_mock_llm(self, sample_data):
        config = AlphaGptConfig(
            objective="test reversal",
            iterations=2,
            pool_size=3,
            forward_returns=[1, 5],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        result = workflow.run()

        assert isinstance(result, AlphaGptResult)
        assert result.iterations_completed == 2
        assert result.total_formulas >= 1
        assert "total_evaluated" in result.summary

    def test_final_pool_uses_fallback_when_critic_empty(self, sample_data):
        """Mock critic 返回空 → fallback 从 evaluations 排序"""
        config = AlphaGptConfig(
            objective="test",
            iterations=2,
            pool_size=3,
            forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        result = workflow.run()

        assert len(result.final_pool) >= 1
        for f in result.final_pool:
            assert "fallback" in f.selection_reason.lower() or "IR=" in f.selection_reason

    def test_custom_llm_client_called(self, sample_data):
        """注入的 llm_client 应该被调用"""
        client = MockLLMClient(responses={
            "alpha-gpt-idea-generator": json.dumps({
                "ideas": [{"id": "I1", "name": "test1", "category": "reversal"}]
            }),
            "alpha-gpt-formula-translator": json.dumps({
                "formulas": [{"formula": "ts_mean(close, 5)", "idea_id": "I1"}]
            }),
        })
        config = AlphaGptConfig(objective="test", iterations=1, pool_size=1, forward_returns=[1])
        workflow = AlphaGptWorkflow(config=config, data=sample_data, llm_client=client)
        workflow.run()

        agent_ids_called = [c[0] for c in client.calls]
        assert "alpha-gpt-idea-generator" in agent_ids_called
        assert "alpha-gpt-formula-translator" in agent_ids_called

    def test_unknown_operator_in_formula_skipped(self, sample_data):
        """含未知算子的公式应该被跳过（不阻塞其他公式）"""
        client = MockLLMClient(responses={
            "alpha-gpt-idea-generator": json.dumps({
                "ideas": [
                    {"id": "I1", "name": "a", "category": "reversal"},
                    {"id": "I2", "name": "b", "category": "momentum"},
                ]
            }),
            "alpha-gpt-formula-translator": json.dumps({
                "formulas": [
                    {"formula": "ts_mean(close, 5)", "idea_id": "I1"},
                    {"formula": "ts_macd(close, 12)", "idea_id": "I2"},
                ]
            }),
        })
        config = AlphaGptConfig(objective="test", iterations=1, pool_size=2, forward_returns=[1])
        workflow = AlphaGptWorkflow(config=config, data=sample_data, llm_client=client)
        result = workflow.run()

        formulas_in_pool = [f.formula for f in result.final_pool]
        assert any("ts_mean" in f for f in formulas_in_pool)
        assert not any("ts_macd" in f for f in formulas_in_pool)

    def test_multiple_rounds(self, sample_data):
        """3 轮 iteration，每轮 2 个公式"""
        config = AlphaGptConfig(
            objective="test",
            iterations=3,
            pool_size=2,
            forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        result = workflow.run()

        assert result.iterations_completed == 3
        assert result.total_formulas == 6  # 3 rounds × 2 formulas

    def test_custom_few_shot_param(self, sample_data):
        """custom_few_shot 字段应可注入"""
        config = AlphaGptConfig(
            objective="test",
            iterations=1,
            pool_size=1,
            forward_returns=[1],
            custom_few_shot=[{"formula": "ts_mean(close, 5)"}],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        result = workflow.run()
        assert result.iterations_completed == 1


# ==============================================================================
# Prompt 构建
# ==============================================================================


class TestPromptBuilding:
    def test_idea_prompt_contains_objective(self, sample_data):
        config = AlphaGptConfig(objective="reversal effect", iterations=1, pool_size=1)
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        prompt = workflow._build_idea_prompt(1, None)
        assert "reversal effect" in prompt
        assert "round=1" in prompt

    def test_formula_prompt_contains_operators(self, sample_data):
        config = AlphaGptConfig(objective="X", iterations=1, pool_size=1)
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        prompt = workflow._build_formula_prompt(
            1, [{"id": "I1", "name": "test"}], ["ts_mean"], ["close"],
        )
        assert "ts_mean" in prompt
        assert "close" in prompt


# ==============================================================================
# Operator 注入
# ==============================================================================


class TestOperatorInjection:
    def test_get_available_operators(self, sample_data):
        config = AlphaGptConfig(objective="test", iterations=1, pool_size=1)
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        ops = workflow._get_available_operators()
        assert isinstance(ops, list)
        assert "ts_mean" in ops

    def test_get_data_columns_from_df(self, sample_data):
        config = AlphaGptConfig(objective="test", iterations=1, pool_size=1)
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        cols = workflow._get_data_columns()
        assert "close" in cols
        assert "code" in cols

    def test_get_data_columns_no_data(self):
        config = AlphaGptConfig(objective="test", iterations=1, pool_size=1)
        workflow = AlphaGptWorkflow(config=config, data=None)
        cols = workflow._get_data_columns()
        assert "close" in cols


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
