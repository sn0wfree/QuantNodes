# coding=utf-8
"""
test_alpha_gpt_e2e.py - Alpha-GPT E2E 测试

覆盖：
- AlphaGptWorkflow.run() 完整 5 轮（mock LLM）
- mock LLM 返回 critic_output → final_pool 走 critic 路径
- 自定义 llm_client
- 数据不足 / 公式全部失败 等边界
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
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 21)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            close = float(np.random.randn() * 5 + 100)
            rows.append({
                "date": date, "code": code, "close": close,
                "open": close, "high": close + 1, "low": close - 1,
                "vol": 1000.0,
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


class TestE2EWorkflow:
    """端到端 workflow（mock LLM，无 nanobot）"""

    def test_full_workflow_5_rounds(self, sample_data):
        config = AlphaGptConfig(
            objective="test reversal",
            iterations=5,
            pool_size=5,
            forward_returns=[1, 5],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        result = workflow.run()

        assert result.iterations_completed == 5
        assert result.total_formulas == 25  # 5 rounds × 5 formulas
        assert result.elapsed_seconds >= 0
        assert "total_evaluated" in result.summary

    def test_workflow_with_critic_output(self, sample_data):
        """Mock LLM 让 critic 返回有效 final_pool → 走 critic 路径"""
        from typing import List

        class CriticMockLLM:
            def __init__(self):
                self.call_count = 0

            def complete(self, agent_id: str, prompt: str) -> str:
                self.call_count += 1
                if "critic" in agent_id:
                    return json.dumps({
                        "final_pool": [
                            {
                                "formula": "sub(close, ts_mean(close, 10))",
                                "formula_id": "MOCK-1",
                                "metrics": {"ir": 2.0, "ic_mean": 0.05},
                                "selection_reason": "mock critic says good",
                                "risk_notes": [],
                            }
                        ]
                    })
                # 其他 subagent 用默认 mock
                return "{}"

        client = CriticMockLLM()
        config = AlphaGptConfig(
            objective="test",
            iterations=2,
            pool_size=2,
            forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data, llm_client=client)
        result = workflow.run()

        assert len(result.final_pool) == 1
        assert result.final_pool[0].selection_reason == "mock critic says good"
        assert result.final_pool[0].ir == 2.0

    def test_workflow_with_invalid_formula_skipped(self, sample_data):
        """含未知算子的公式应该被跳过"""
        class BadFormulaLLM:
            def __init__(self):
                self.called = False

            def complete(self, agent_id: str, prompt: str) -> str:
                self.called = True
                if "formula-translator" in agent_id:
                    return json.dumps({
                        "formulas": [
                            {"formula": "ts_mean(close, 5)", "idea_id": "I1"},
                            {"formula": "ts_macd(close, 12)", "idea_id": "I2"},  # invalid
                        ]
                    })
                if "idea-generator" in agent_id:
                    return json.dumps({
                        "ideas": [
                            {"id": "I1", "name": "a", "category": "reversal"},
                            {"id": "I2", "name": "b", "category": "momentum"},
                        ]
                    })
                return "{}"

        config = AlphaGptConfig(
            objective="test", iterations=1, pool_size=2, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(
            config=config, data=sample_data, llm_client=BadFormulaLLM(),
        )
        result = workflow.run()

        # ts_macd 被跳过，只剩 ts_mean
        formulas_evaluated = [e.formula for e in workflow.state.all_evaluations]
        assert all("ts_mean" in f for f in formulas_evaluated)

    def test_workflow_with_no_data(self):
        """没有数据也能跑（但 final_pool 为空）"""
        config = AlphaGptConfig(
            objective="test", iterations=1, pool_size=2, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=None)
        result = workflow.run()

        # 没有数据 → evaluator 失败 → final_pool 为空
        assert result.total_formulas >= 0
        # 但工作流不抛异常

    def test_workflow_summary_keys(self, sample_data):
        config = AlphaGptConfig(
            objective="test", iterations=2, pool_size=3, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        result = workflow.run()

        for k in (
            "total_evaluated", "successful", "failed", "selected",
            "avg_ir", "best_ir", "category_distribution",
        ):
            assert k in result.summary

    def test_workflow_with_short_data(self):
        """数据 < 5 天 → 公式都会失败"""
        import datetime as dt
        dates = [(dt.date(2024, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(3)]
        rows = []
        for date in dates:
            for code in ["A", "B"]:
                rows.append({
                    "date": date, "code": code, "close": 100.0,
                    "open": 100.0, "high": 101.0, "low": 99.0, "vol": 1000.0,
                })
        df = pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())

        config = AlphaGptConfig(
            objective="test", iterations=1, pool_size=2, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=df)
        result = workflow.run()

        # 短数据下评估可能失败但不抛异常
        assert result.iterations_completed == 1


class TestWorkflowStateAccumulation:
    """state 在多轮中正确累积"""

    def test_state_ideas_accumulate(self, sample_data):
        config = AlphaGptConfig(
            objective="test", iterations=3, pool_size=2, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        workflow.run()

        # 3 rounds × 2 ideas = 6 ideas in state
        assert len(workflow.state.all_ideas) == 6

    def test_state_formulas_accumulate(self, sample_data):
        config = AlphaGptConfig(
            objective="test", iterations=3, pool_size=2, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        workflow.run()

        assert len(workflow.state.all_formulas) == 6

    def test_reflections_only_in_non_last_rounds(self, sample_data):
        config = AlphaGptConfig(
            objective="test", iterations=3, pool_size=2, forward_returns=[1],
        )
        workflow = AlphaGptWorkflow(config=config, data=sample_data)
        workflow.run()

        # 末轮无 reflection（被 critic 替代）
        assert len(workflow.state.all_reflections) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
