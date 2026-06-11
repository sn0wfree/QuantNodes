"""FactorFeedback 模块测试 (20 tests)。

覆盖:
    - Dataclass 创建/序列化 (5)
    - 4 通道采集器 (13)
    - FeedbackCollector (4)
    - LLMJudge (4) — 实际覆盖 mock 主路径
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
    FeedbackCollector,
    LLMJudge,
    collect_code,
    collect_execution,
    collect_shape,
    collect_value,
    ensure_feedback,
)


# ============================================================================
# 1. Dataclass tests (5)
# ============================================================================

def test_channel_enum_values():
    """5 个通道枚举值正确。"""
    assert FeedbackChannel.EXECUTION.value == "execution"
    assert FeedbackChannel.SHAPE.value == "shape"
    assert FeedbackChannel.CODE.value == "code"
    assert FeedbackChannel.VALUE.value == "value"
    assert FeedbackChannel.LLM.value == "llm"
    assert len(FeedbackChannel) == 5


def test_channel_feedback_creation():
    """ChannelFeedback 默认值正确。"""
    fb = ChannelFeedback(
        channel=FeedbackChannel.CODE,
        passed=True,
        detail="OK",
    )
    assert fb.channel == FeedbackChannel.CODE
    assert fb.passed is True
    assert fb.detail == "OK"
    assert fb.score == 1.0
    assert fb.metadata == {}

    p, d, s = fb.values()
    assert (p, d, s) == (True, "OK", 1.0)


def test_factor_feedback_creation():
    """FactorFeedback 默认值 + UUID。"""
    fb = FactorFeedback(factor_name="momentum_20d")
    assert fb.factor_name == "momentum_20d"
    assert len(fb.factor_id) > 0
    assert fb.decision is False
    assert fb.channels == {}
    assert fb.duration_ms == 0.0


def test_feedback_to_dict():
    """to_dict 包含所有通道且可 JSON 化。"""
    fb = FactorFeedback(
        factor_name="test",
        channels={
            FeedbackChannel.CODE: ChannelFeedback(
                FeedbackChannel.CODE, True, "OK", 1.0,
            ),
        },
        decision=True,
        summary="ok",
    )
    d = fb.to_dict()
    json.dumps(d)  # 验证可序列化
    assert d["factor_name"] == "test"
    assert "code" in d["channels"]
    assert d["channels"]["code"]["passed"] is True


def test_feedback_from_dict():
    """from_dict 正确还原。"""
    original = FactorFeedback(
        factor_name="test",
        channels={
            FeedbackChannel.SHAPE: ChannelFeedback(
                FeedbackChannel.SHAPE, True, "match", 1.0,
            ),
        },
        decision=True,
        summary="ok",
    )
    d = original.to_dict()
    restored = FactorFeedback.from_dict(d)
    assert restored.factor_id == original.factor_id
    assert restored.factor_name == "test"
    assert FeedbackChannel.SHAPE in restored.channels
    assert restored.channels[FeedbackChannel.SHAPE].detail == "match"


# ============================================================================
# 2. Channel collectors (13)
# ============================================================================

def test_collect_execution_success():
    """exit=0 通过。"""
    fb = collect_execution("hello", "", 0)
    assert fb.passed is True
    assert fb.score == 1.0
    assert "exit=0" in fb.detail
    assert fb.metadata["exit_code"] == 0


def test_collect_execution_failure():
    """exit!=0 失败。"""
    fb = collect_execution("", "NameError: x is not defined", 1)
    assert fb.passed is False
    assert fb.score == 0.0
    assert "NameError" in fb.detail
    assert fb.metadata["exit_code"] == 1


def test_collect_shape_match():
    """形状一致通过。"""
    fb = collect_shape((20, 30), (20, 30))
    assert fb.passed is True
    assert "actual=(20, 30)" in fb.detail


def test_collect_shape_mismatch():
    """形状不一致失败。"""
    fb = collect_shape((20, 30), (20, 40))
    assert fb.passed is False
    assert "expected=(20, 40)" in fb.detail


def test_collect_code_simple():
    """简单表达式通过。"""
    fb = collect_code("close - open")
    assert fb.passed is True
    assert "OK" in fb.detail


def test_collect_code_long():
    """长表达式失败。"""
    long_expr = "close + open + " + " + ".join([f"x_{i}" for i in range(100)])
    fb = collect_code(long_expr)
    assert fb.passed is False
    assert "length=" in fb.detail


def test_collect_code_many_features():
    """多特征失败 (>5 base features)。"""
    expr = "open + high + low + close + volume + vwap + turnover + mv_float"
    fb = collect_code(expr)
    assert fb.passed is False
    assert "features=" in fb.detail


def test_collect_code_syntax_error():
    """语法错误失败。"""
    fb = collect_code("close - )")
    assert fb.passed is False
    assert "语法错误" in fb.detail


def test_collect_value_normal():
    """正常分布通过。"""
    s = pd.Series(np.random.default_rng(42).normal(0, 1, 100))
    fb = collect_value(s)
    assert fb.passed is True
    assert "OK" in fb.detail


def test_collect_value_nan_heavy():
    """NaN 过多失败。"""
    s = pd.Series([1.0] * 5 + [float("nan")] * 95)
    fb = collect_value(s)
    assert fb.passed is False
    assert "NaN=" in fb.detail


def test_collect_value_inf():
    """Inf 检测失败。"""
    s = pd.Series([1.0, 2.0, float("inf"), 4.0])
    fb = collect_value(s)
    assert fb.passed is False
    assert "Inf=" in fb.detail


def test_collect_value_constant():
    """常量检测失败 (std=0)。"""
    s = pd.Series([5.0] * 10)
    fb = collect_value(s)
    assert fb.passed is False
    assert "std=" in fb.detail


def test_collect_value_all_nan():
    """全 NaN 失败。"""
    s = pd.Series([float("nan")] * 10)
    fb = collect_value(s)
    assert fb.passed is False


# ============================================================================
# 3. FeedbackCollector (4)
# ============================================================================

def test_collector_add_channels():
    """添加多个通道。"""
    fc = FeedbackCollector("id1", "name1")
    fc.add(FeedbackChannel.CODE, True, "ok")
    fc.add(FeedbackChannel.SHAPE, True, "match")
    assert fc.has(FeedbackChannel.CODE)
    assert fc.has(FeedbackChannel.SHAPE)
    assert not fc.has(FeedbackChannel.VALUE)


def test_collector_finalize_default_decision():
    """默认决策 = 全部通道通过。"""
    fc = FeedbackCollector("id1", "name1")
    fc.add(FeedbackChannel.CODE, True, "ok")
    fc.add(FeedbackChannel.VALUE, True, "ok")
    fb = fc.finalize()
    assert fb.decision is True
    assert fb.summary == "全部通过"


def test_collector_finalize_explicit_decision():
    """显式决策。"""
    fc = FeedbackCollector("id1", "name1")
    fc.add(FeedbackChannel.CODE, True, "ok")
    fc.add(FeedbackChannel.VALUE, False, "NaN too high")
    fb = fc.finalize(decision=False, summary="手动拒绝")
    assert fb.decision is False
    assert fb.summary == "手动拒绝"


def test_collector_summary_generation():
    """自动生成总结 (失败通道列表)。"""
    fc = FeedbackCollector("id1", "name1")
    fc.add(FeedbackChannel.CODE, True, "ok")
    fc.add(FeedbackChannel.VALUE, False, "NaN too high")
    fb = fc.finalize()
    assert "value" in fb.summary
    assert fb.decision is False


# ============================================================================
# 4. LLMJudge (4)
# ============================================================================

def test_llm_judge_passes():
    """关键词匹配 + 表达式含 returns/close → 一致。"""
    judge = LLMJudge(model="mock")
    fb = judge.judge("momentum effect", "20-day momentum", "close / close.shift(20) - 1")
    assert fb.passed is True
    assert "关键词" in fb.detail or "默认" in fb.detail


def test_llm_judge_fails():
    """hypothesis 和 description 都为空 → 不一致。"""
    judge = LLMJudge(model="mock")
    fb = judge.judge("", "", "close - open")
    assert fb.passed is False
    assert "为空" in fb.detail


def test_llm_judge_custom_callable():
    """支持自定义 llm_callable。"""
    def fake_llm(prompt):
        return json.dumps({"consistent": True, "reason": "fake ok", "score": 0.95})
    judge = LLMJudge(llm_callable=fake_llm)
    fb = judge.judge("h", "d", "e")
    assert fb.passed is True
    assert "fake ok" in fb.detail
    assert fb.metadata["model"] == "mock"


def test_llm_judge_parse_failure():
    """解析失败达到最大尝试次数后返回失败。"""
    def bad_llm(prompt):
        return "not a json"
    judge = LLMJudge(llm_callable=bad_llm, max_correction_attempts=2)
    fb = judge.judge("h", "d", "e")
    assert fb.passed is False
    assert "解析失败" in fb.detail
    assert fb.metadata["attempt"] == 3  # 1 initial + 2 retries


# ============================================================================
# 5. Serialization round-trip (补充, 不计入 20)
# ============================================================================

def test_parquet_round_trip():
    """Parquet 写入+读取。"""
    fc = FeedbackCollector("id1", "name1")
    fc.add(FeedbackChannel.CODE, True, "ok", score=0.9)
    fc.add(FeedbackChannel.VALUE, True, "ok", score=0.95)
    fb = fc.finalize(metadata={"ic": 0.05})

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fb.parquet"
        fb.save_parquet(p)
        loaded = FactorFeedback.load_parquet(p)
        assert len(loaded) == 1
        assert loaded[0].factor_name == "name1"
        assert FeedbackChannel.CODE in loaded[0].channels


def test_json_round_trip():
    """JSON 写入+读取。"""
    fb = FactorFeedback(
        factor_name="test",
        channels={FeedbackChannel.CODE: ChannelFeedback(
            FeedbackChannel.CODE, True, "ok", 1.0,
        )},
        decision=True,
    )
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fb.json"
        fb.save_json(p)
        loaded = FactorFeedback.load_json(p)
        assert loaded.factor_name == "test"
        assert loaded.channels[FeedbackChannel.CODE].detail == "ok"


def test_ensure_feedback_dict():
    """dict 包装。"""
    fb = ensure_feedback({"ic": 0.05, "sharpe": 1.2, "unknown": 999}, "id1", "factor1")
    assert fb.factor_id == "id1"
    assert fb.factor_name == "factor1"
    assert fb.metadata["ic"] == 0.05
    assert "sharpe" in fb.metadata
    assert "unknown" not in fb.metadata  # 不在白名单内


def test_ensure_feedback_already_feedback():
    """FactorFeedback 透传。"""
    original = FactorFeedback(factor_name="orig")
    result = ensure_feedback(original, "id2", "factor2")
    assert result is original


def test_ensure_feedback_invalid_type():
    """非 dict / FactorFeedback 抛错。"""
    with pytest.raises(TypeError, match="不支持"):
        ensure_feedback(42, "id", "name")
