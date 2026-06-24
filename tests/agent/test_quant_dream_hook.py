# coding=utf-8
"""Tests for QuantNodes.agent.core.quant_dream.

Covers:
- QuantDreamHook.should_analyze() keyword detection
- QuantDreamHook.analyze_session() insight extraction
- QuantDreamHook.append() / get_recent_dreams() round-trip via topic-quant-dream.md
- DreamEngine shim (backward compat): both ``workspace=`` and ``dream_store=`` kwargs
- QuantDreamInsight: id auto-generated, to_markdown format
"""

import json
from pathlib import Path

import pytest

from QuantNodes.agent.core.quant_dream import (
    BACKTEST_KEYWORDS,
    FACTOR_KEYWORDS,
    QuantDreamHook,
    QuantDreamInsight,
    STRATEGY_KEYWORDS,
    DreamEngine,
)


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    return tmp_path


def test_should_analyze_factor_keyword(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    assert hook.should_analyze("研究 alpha 因子 IC", "IC mean = 0.05")
    assert hook.should_analyze("", "momentum factor with rank IC")


def test_should_analyze_backtest_keyword(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    assert hook.should_analyze("", "annualized return 18%, sharpe 1.5, drawdown -8%")


def test_should_analyze_strategy_keyword(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    assert hook.should_analyze("", "rebalance every 5 days, portfolio weight 0.5")


def test_should_not_analyze_unrelated(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    assert not hook.should_analyze("hello", "how are you?")
    assert not hook.should_analyze("", "weather is nice today")


def test_analyze_session_returns_insight(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    insight = hook.analyze_session(
        "default",
        "momentum factor IC analysis",
        "IC mean is 0.05, ICIR is 1.2 over 2020-2024.",
    )
    assert insight is not None
    assert insight.type == "factor_insight"
    assert insight.confidence == 0.7
    assert "IC mean" in insight.content or len(insight.insights) >= 1


def test_analyze_session_none_when_no_keyword(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    insight = hook.analyze_session("default", "hello", "general chat")
    assert insight is None


def test_append_creates_topic_file(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace, min_confidence=0.0)
    dream = QuantDreamInsight(
        type="factor_insight",
        content="momentum_20d has ICIR=1.2",
        insights=["ICIR > 1 is good", "20-day lookback"],
        confidence=0.8,
    )
    hook.append(dream)
    assert hook.topic_file.exists()
    text = hook.topic_file.read_text(encoding="utf-8")
    assert "momentum_20d" in text
    assert "factor_insight" in text


def test_append_skipped_below_min_confidence(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace, min_confidence=0.9)
    dream = QuantDreamInsight(
        type="factor_insight",
        content="low confidence insight",
        confidence=0.3,
    )
    hook.append(dream)
    assert not hook.topic_file.exists()


def test_get_recent_dreams_round_trip(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace, min_confidence=0.0)
    dreams = [
        QuantDreamInsight(
            type="factor_insight",
            content=f"insight #{i}",
            insights=[f"point {i}"],
            confidence=0.8,
            timestamp=f"2026-06-{20+i:02d}T00:00:00Z",
            source="test",
        )
        for i in range(3)
    ]
    for d in dreams:
        hook.append(d)

    recent = hook.get_recent_dreams(limit=10)
    assert len(recent) == 3
    assert recent[0].content == "insight #2"
    assert recent[-1].content == "insight #0"
    for r in recent:
        assert r.type == "factor_insight"
        assert r.source == "quant_dream"


def test_get_recent_dreams_empty_file(tmp_workspace: Path):
    hook = QuantDreamHook(tmp_workspace)
    assert hook.get_recent_dreams(limit=10) == []


def test_dream_engine_workspace_kwarg(tmp_workspace: Path):
    engine = DreamEngine(workspace=tmp_workspace)
    assert engine.workspace == tmp_workspace
    assert isinstance(engine.hook, QuantDreamHook)


def test_dream_engine_dream_store_kwarg_backcompat(tmp_workspace: Path):
    """Legacy callers pass ``dream_store=``; engine should accept and use workspace."""

    class FakeStore:
        workspace = tmp_workspace

    engine = DreamEngine(dream_store=FakeStore())
    assert engine.workspace == tmp_workspace


def test_dream_engine_generate_dream(tmp_workspace: Path):
    engine = DreamEngine(workspace=tmp_workspace)
    dream = engine.generate_dream(
        dream_type="backtest_pattern",
        content="sample backtest completed",
        source="test",
        confidence=0.9,
        tags=["momentum", "sharpe"],
    )
    assert dream.id.startswith("dream-")
    assert dream.type == "backtest_pattern"
    assert dream.confidence == 0.9
    assert engine.hook.topic_file.exists()


def test_quant_dream_insight_auto_id():
    """Two insights with no explicit id should get distinct auto-generated ids."""
    a = QuantDreamInsight(type="x", content="a")
    b = QuantDreamInsight(type="x", content="b")
    assert a.id != b.id
    assert a.id.startswith("dream-")


def test_to_markdown_format():
    dream = QuantDreamInsight(
        type="factor_insight",
        content="content here",
        insights=["i1", "i2"],
        tags=["t1"],
        timestamp="2026-06-23T00:00:00Z",
    )
    md = dream.to_markdown()
    assert "2026-06-23 - factor_insight" in md
    assert "content here" in md
    assert "- i1" in md
    assert "- i2" in md
    assert "tags: t1" in md


def test_keyword_constants_nonempty():
    """Sanity: keyword sets should be non-empty for production keyword detection."""
    assert len(FACTOR_KEYWORDS) > 0
    assert len(BACKTEST_KEYWORDS) > 0
    assert len(STRATEGY_KEYWORDS) > 0
    assert "alpha" in FACTOR_KEYWORDS
    assert "sharpe" in BACKTEST_KEYWORDS
    assert "strategy" in STRATEGY_KEYWORDS
