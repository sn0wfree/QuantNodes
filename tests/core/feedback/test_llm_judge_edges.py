"""llm_judge.py 错误/降级/边界测试 (12 tests)。

聚焦:
    - mock 模式: 关键词匹配、hypothesis+description 为空、expression 为空
    - 自定义 llm_callable: 一直返 JSON / 一直返非法
    - max_correction_attempts: 重试上限 + 最终 fallback
    - 真实 model 未实现抛 NotImplementedError
    - prompt 包含三段 (hypothesis/description/expression)
"""
from __future__ import annotations

import json

import pytest

from QuantNodes.core.feedback import LLMJudge, FeedbackChannel
from QuantNodes.core.feedback.llm_judge import _extract_field


# ============================================================================
# 1. Mock 模式 (5 tests)
# ============================================================================

class TestMockMode:
    def test_momentum_keyword_passes(self):
        j = LLMJudge(model="mock")
        r = j.judge("momentum alpha", "20-day momentum", "close - close.shift(20)")
        assert r.channel == FeedbackChannel.LLM
        assert r.passed is True
        # score >= 0.8 (关键词命中)
        assert r.score >= 0.8

    def test_volume_keyword_passes(self):
        j = LLMJudge(model="mock")
        r = j.judge("volume oscillator", "volume spike", "volume.diff(5)")
        assert r.passed is True

    def test_empty_hypothesis_and_description_fails(self):
        j = LLMJudge(model="mock")
        r = j.judge("", "", "close")
        assert r.passed is False
        assert "hypothesis" in r.detail.lower() or "为空" in r.detail

    def test_empty_expression_fails(self):
        j = LLMJudge(model="mock")
        r = j.judge("h", "d", "")
        assert r.passed is False
        assert "表达式" in r.detail or "表达式" in r.detail

    def test_default_passes_when_no_keywords(self):
        """无关键词匹配, 但 hypothesis+description 都有内容 → mock 默认 pass。"""
        j = LLMJudge(model="mock")
        r = j.judge("xyz alpha", "xyz description", "xyz_expression")
        # mock fallback 0.7 score
        assert r.passed is True
        assert r.score >= 0.5

    def test_metadata_model_and_attempt(self):
        j = LLMJudge(model="mock")
        r = j.judge("h", "d", "close")
        assert r.metadata["model"] == "mock"
        assert r.metadata["attempt"] == 1


# ============================================================================
# 2. 自定义 llm_callable (3 tests)
# ============================================================================

class TestCustomLLMCallable:
    def test_callable_returning_valid_json(self):
        def fake_llm(prompt):
            return json.dumps({
                "consistent": True, "reason": "manually passed", "score": 0.95,
            })
        j = LLMJudge(llm_callable=fake_llm)
        r = j.judge("h", "d", "close")
        assert r.passed is True
        assert r.score == 0.95
        assert "manually" in r.detail

    def test_callable_returning_invalid_json(self):
        """llm_callable 一直返非法 JSON → 最终 fallback failed。"""
        def bad_llm(prompt):
            return "not json at all"
        j = LLMJudge(llm_callable=bad_llm, max_correction_attempts=2)
        r = j.judge("h", "d", "close")
        assert r.passed is False
        assert r.score == 0.0
        # attempt = max + 1
        assert r.metadata["attempt"] == 3

    def test_callable_intermittent_succeeds(self):
        """前 2 次失败, 第 3 次成功 → 第一次成功后停止。"""
        attempt_count = [0]

        def intermittent_llm(prompt):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                return "bad json"
            return json.dumps({
                "consistent": True, "reason": "ok", "score": 0.9,
            })

        j = LLMJudge(llm_callable=intermittent_llm, max_correction_attempts=5)
        r = j.judge("h", "d", "close")
        assert r.passed is True
        assert r.metadata["attempt"] == 3

    def test_callable_raises_exception_propagates(self):
        """llm_callable 抛 RuntimeError → 当前实现未捕获, 异常上抛 (已知限制)。"""
        def bad_llm(prompt):
            raise RuntimeError("network error")
        j = LLMJudge(llm_callable=bad_llm, max_correction_attempts=1)
        with pytest.raises(RuntimeError, match="network error"):
            j.judge("h", "d", "close")

    def test_callable_raises_type_error_caught(self):
        """llm_callable 抛 TypeError → judge 捕获 (在 except 列表中)。"""
        def bad_llm(prompt):
            raise TypeError("type error")
        j = LLMJudge(llm_callable=bad_llm, max_correction_attempts=0)
        r = j.judge("h", "d", "close")
        # TypeError 被 except 捕获, 走 fallback
        assert r.passed is False
        assert r.metadata["attempt"] == 1


# ============================================================================
# 3. 真实 model 未实现 (2 tests)
# ============================================================================

class TestRealModel:
    def test_real_model_no_callable_raises(self):
        """model='deepseek' 无 llm_callable → NotImplementedError。"""
        j = LLMJudge(model="deepseek-v3")
        with pytest.raises(NotImplementedError, match="真实 LLM"):
            j.judge("h", "d", "close")


# ============================================================================
# 4. 内部辅助 _extract_field (2 tests)
# ============================================================================

class TestExtractField:
    def test_extract_existing_field(self):
        prompt = "Hypothesis: alpha momentum\nDescription: 20-day\nExpression: close"
        assert _extract_field(prompt, "Hypothesis") == "alpha momentum"
        assert _extract_field(prompt, "Description") == "20-day"
        assert _extract_field(prompt, "Expression") == "close"

    def test_extract_missing_field_returns_empty(self):
        prompt = "Hypothesis: x\nDescription: y"
        assert _extract_field(prompt, "Expression") == ""
        assert _extract_field(prompt, "Hypothesis") == "x"

    def test_field_without_colon_returns_empty(self):
        prompt = "Hypothesis alpha\nDescription: d"
        assert _extract_field(prompt, "Hypothesis") == ""
