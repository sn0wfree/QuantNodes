# coding=utf-8
"""
test_consistency_hook.py - PR-5 一致性 hook 单元测试

测试：
- collect_llm_channel 三种模式（mock_keyword / structured_logic_match / llm_judge）
- LogicDrivenPipeline 端到端串联
- MCTSConfig structured_logic 接入
"""

import pytest
import json

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicCondition,
    LogicBehavior,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.mcts.feedback import (
    collect_llm_channel,
    _mock_keyword_match,
    _structured_logic_match,
    _llm_judge_consistency,
)
from QuantNodes.research.quant_alpha.mcts.search import MCTSSearchConfig


# ==============================================================================
# collect_llm_channel 测试
# ==============================================================================


class TestCollectLLMChannel:
    """collect_llm_channel 三种模式测试"""

    def test_no_hypothesis_pass(self):
        """无 hypothesis 时 pass"""
        fb = collect_llm_channel(formula="rank(close)")
        assert fb.passed == True
        assert "no hypothesis" in fb.detail

    def test_with_hypothesis(self):
        """有 hypothesis"""
        fb = collect_llm_channel(
            formula="rank(close)",
            hypothesis="Use rank to normalize",
            description="rank is the cross-sectional rank",
        )
        # 应使用 mock_keyword 模式
        assert fb.metadata.get("mode") == "mock_keyword"

    def test_with_structured_logic(self):
        """有 structured_logic"""
        logic = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="ts_mean", threshold=0, window=20)],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        )
        fb = collect_llm_channel(
            formula="rank(ts_mean(close, 20))",
            structured_logic=logic,
        )
        # 应使用 structured_logic_match 模式
        assert fb.metadata.get("mode") == "structured_logic_match"
        assert "operator_overlap" in fb.metadata


class TestMockKeywordMatch:
    """_mock_keyword_match 测试"""

    def test_match(self):
        """关键字匹配"""
        fb = _mock_keyword_match(
            formula="rank(close)",
            hypothesis="Use rank to normalize close price",
            description=None,
        )
        assert fb.metadata.get("mode") == "mock_keyword"
        assert fb.metadata["matches"] >= 1

    def test_no_match(self):
        """无关键字匹配"""
        fb = _mock_keyword_match(
            formula="rank(close)",
            hypothesis="something unrelated here",
            description=None,
        )
        # 关键字匹配率低
        assert fb.metadata["match_ratio"] < 0.3


class TestStructuredLogicMatch:
    """_structured_logic_match 测试"""

    def test_full_match(self):
        """完全匹配"""
        logic = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="rank", threshold=0)],
            behavior=LogicBehavior(target="forward_return_1", direction=1, horizon=1),
            operator_whitelist=["rank"],
            sign_constraint=1,
        )
        fb = _structured_logic_match("rank(close)", logic)
        assert fb.score >= 0.7
        assert fb.metadata["mode"] == "structured_logic_match"

    def test_partial_match(self):
        """部分匹配"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="close", op="ts_mean", threshold=0, window=20),
                LogicCondition(variable="volume", op="ts_mean", threshold=0, window=20),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            operator_whitelist=["ts_mean", "rank", "sub", "div"],
            sign_constraint=-1,
        )
        fb = _structured_logic_match("rank(ts_mean(close, 20))", logic)
        # 应有部分重叠
        assert fb.score > 0
        assert fb.score < 1.0

    def test_direction_mismatch(self):
        """方向不匹配"""
        logic = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="rank", threshold=0)],
            behavior=LogicBehavior(target="forward_return_1", direction=-1, horizon=1),
            sign_constraint=-1,  # 要求反向
        )
        fb = _structured_logic_match("rank(close)", logic)
        # rank(close) 不是负向，应有方向不匹配
        assert fb.metadata.get("direction_match", 1.0) == 0.0


class TestLLMJudgeConsistency:
    """_llm_judge_consistency 测试"""

    def test_json_response(self):
        """JSON 响应"""
        class MockLLM:
            def complete(self, agent_id, prompt):
                return json.dumps({"score": 0.8, "reason": "matches well"})

        fb = _llm_judge_consistency(
            formula="rank(close)",
            hypothesis="use rank",
            description=None,
            structured_logic=None,
            llm_client=MockLLM(),
        )
        assert fb.score == 0.8
        assert fb.passed == True
        assert fb.metadata.get("mode") == "llm_judge"

    def test_low_score(self):
        """低分"""
        class MockLLM:
            def complete(self, agent_id, prompt):
                return json.dumps({"score": 0.3, "reason": "mismatch"})

        fb = _llm_judge_consistency(
            formula="rank(close)",
            hypothesis="volatility indicator",
            description=None,
            structured_logic=None,
            llm_client=MockLLM(),
            score_threshold=0.5,
        )
        assert fb.score == 0.3
        assert fb.passed == False

    def test_invalid_json(self):
        """无效 JSON"""
        class MockLLM:
            def complete(self, agent_id, prompt):
                return "not json"

        fb = _llm_judge_consistency(
            formula="rank(close)",
            hypothesis="test",
            description=None,
            structured_logic=None,
            llm_client=MockLLM(),
        )
        # 应使用 fallback 评分
        assert fb.score >= 0
        assert fb.score <= 1

    def test_llm_failure_fallback(self):
        """LLM 失败时回退"""
        class FailingLLM:
            def complete(self, agent_id, prompt):
                raise Exception("API error")

        fb = _llm_judge_consistency(
            formula="rank(close)",
            hypothesis="rank normalization",
            description=None,
            structured_logic=None,
            llm_client=FailingLLM(),
        )
        # 应回退到 mock
        assert fb.metadata.get("mode") == "mock_keyword"


# ==============================================================================
# MCTSConfig 接入测试
# ==============================================================================


class TestMCTSConfigStructuredLogic:
    """MCTSSearchConfig structured_logic 测试"""

    def test_config_with_structured_logic(self):
        """带 structured_logic 配置"""
        logic = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="rank", threshold=0)],
            behavior=LogicBehavior(target="forward_return_1", direction=1, horizon=1),
        )
        config = MCTSSearchConfig(
            iterations=10,
            structured_logic=logic,
            llm_client=None,
        )
        assert config.structured_logic == logic
        assert config.llm_client is None

    def test_config_default(self):
        """默认配置"""
        config = MCTSSearchConfig(iterations=10)
        assert config.structured_logic is None
        assert config.llm_client is None


# ==============================================================================
# 集成测试
# ==============================================================================


class TestIntegration:
    """集成测试"""

    def test_pipeline_config_structured_logic(self):
        """PipelineConfig 支持 structured_logic"""
        from QuantNodes.research.quant_alpha.pipeline import PipelineConfig

        logic = WikiLogicStructured(
            predicates=[LogicCondition(variable="close", op="ts_mean", threshold=0, window=20)],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
        )
        config = PipelineConfig(
            objective="test",
            structured_logic=logic,
        )
        assert config.structured_logic == logic

    def test_logic_driven_pipeline_import(self):
        """LogicDrivenPipeline 可导入"""
        from QuantNodes.research.quant_alpha.logic_driven_pipeline import (
            LogicDrivenPipeline,
            LogicDrivenPipelineConfig,
            LogicDrivenPipelineResult,
        )

        config = LogicDrivenPipelineConfig(
            objective="test",
            logic_driven=False,  # 不实际运行
        )
        assert config.logic_driven == False

    def test_collect_llm_channel_with_logic(self):
        """collect_llm_channel 集成"""
        logic = WikiLogicStructured(
            predicates=[
                LogicCondition(variable="close", op="ts_corr", threshold=0, window=10),
            ],
            behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
            operator_whitelist=["ts_corr", "rank", "sign"],
            sign_constraint=-1,
        )
        # 通过的公式
        fb1 = collect_llm_channel(
            formula="sign(-ts_corr(close, close, 10))",
            structured_logic=logic,
            score_threshold=0.5,
        )
        assert fb1.score > 0

        # 完全不相关的公式
        fb2 = collect_llm_channel(
            formula="rank(ts_argmax(close, 5))",
            structured_logic=logic,
            score_threshold=0.5,
        )
        assert fb2.score < fb1.score