# coding=utf-8
"""
test_table4_evaluator.py - PolarsAlphaCalculatorEvaluator + G1/G2/G3 baseline 测试
"""

from __future__ import annotations

import pytest

from QuantNodes.research.quant_alpha.evaluation import (
    FactorSpec,
    MockDataLoader,
    PolarsAlphaCalculatorEvaluator,
)
from QuantNodes.research.quant_alpha.evaluation.baselines import (
    G1Handcrafted,
    G2LlmOnly,
    G3AlphaGpt,
)


@pytest.fixture(scope="module")
def sample_data():
    loader = MockDataLoader(n_stocks=20, n_days=50, seed=42)
    return loader.load()


class TestPolarsAlphaCalculatorEvaluator:
    def test_empty_factors(self, sample_data):
        evaluator = PolarsAlphaCalculatorEvaluator()
        result = evaluator.evaluate([], sample_data)
        assert result == []

    def test_valid_formula(self, sample_data):
        evaluator = PolarsAlphaCalculatorEvaluator()
        factors = [
            FactorSpec(formula_id="f1", formula="ts_mean(close, 5)", source="test"),
        ]
        metrics = evaluator.evaluate(factors, sample_data)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.formula_id == "f1"
        assert m.status == "success"
        assert isinstance(m.ir, float)
        assert isinstance(m.ic_mean, float)

    def test_invalid_formula_marked_failed(self, sample_data):
        evaluator = PolarsAlphaCalculatorEvaluator()
        factors = [
            FactorSpec(formula_id="f1", formula="rank(close)", source="test"),
        ]
        metrics = evaluator.evaluate(factors, sample_data)
        m = metrics[0]
        assert m.status == "failed"
        assert m.error_msg is not None

    def test_mixed_valid_and_invalid(self, sample_data):
        evaluator = PolarsAlphaCalculatorEvaluator()
        factors = [
            FactorSpec(formula_id="f1", formula="ts_mean(close, 5)", source="test"),
            FactorSpec(formula_id="f2", formula="rank(close)", source="test"),
            FactorSpec(formula_id="f3", formula="ts_std(vol, 3)", source="test"),
        ]
        metrics = evaluator.evaluate(factors, sample_data)
        assert len(metrics) == 3
        statuses = [m.status for m in metrics]
        assert "success" in statuses
        assert "failed" in statuses


class TestG1Handcrafted:
    def test_default_n(self):
        g1 = G1Handcrafted(n=5)
        assert g1.group_name == "G1_Handcrafted"
        factors = g1.generate_factors()
        assert len(factors) == 5

    def test_formula_id_unique(self):
        g1 = G1Handcrafted(n=20)
        factors = g1.generate_factors()
        ids = [f.formula_id for f in factors]
        assert len(ids) == len(set(ids))

    def test_source_attribute(self):
        g1 = G1Handcrafted(n=3)
        factors = g1.generate_factors()
        assert all(f.source == "g1_handcrafted" for f in factors)


class TestG2LlmOnly:
    def test_includes_invalid(self):
        """G2 至少要包含一些 invalid 公式（模拟 LLM 错误）"""
        g2 = G2LlmOnly(n=20)
        factors = g2.generate_factors()
        assert len(factors) == 20
        # ~15% 应该是 invalid
        invalid_count = sum(
            1 for f in factors if "rank(" in f.formula or "IndNeutralize" in f.formula
        )
        assert invalid_count >= 1

    def test_source_attribute(self):
        g2 = G2LlmOnly(n=5)
        factors = g2.generate_factors()
        assert all(f.source == "g2_llm_only" for f in factors)


class TestG3AlphaGpt:
    def test_default_n(self):
        g3 = G3AlphaGpt(n=5, iterations=1, pool_size=3)
        factors = g3.generate_factors()
        assert len(factors) == 5

    def test_source_attribute(self):
        g3 = G3AlphaGpt(n=5, iterations=1, pool_size=3)
        factors = g3.generate_factors()
        assert all(f.source == "g3_alpha_gpt" for f in factors)

    def test_fallback_when_workflow_empty(self):
        """当 AlphaGptWorkflow mock 返回空时，fallback 仍能凑齐 n 个因子"""
        g3 = G3AlphaGpt(n=10, iterations=1, pool_size=3)
        factors = g3.generate_factors()
        assert len(factors) == 10