# coding=utf-8
"""Tests for core/feedback/collector.py — FeedbackCollector aggregator.

Covers: FeedbackCollector.add(), add_feedback(), has(), get(), finalize()
with all 3 agg_mode values, dataclass serialization (to_dict, save_json,
save_parquet), and ensure_feedback() helper.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.core.feedback.collector import FeedbackCollector
from QuantNodes.core.feedback.dataclass import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
    ensure_feedback,
    _safe_scalar,
)


# ============================================================================
# FeedbackCollector Creation
# ============================================================================

class TestFeedbackCollectorCreation:
    def test_creation(self):
        c = FeedbackCollector(factor_id="f-1", factor_name="my_factor")
        assert c.factor_id == "f-1"
        assert c.factor_name == "my_factor"
        assert c._channels == {}

    def test_creation_starts_timer(self):
        c = FeedbackCollector("f-1", "my_factor")
        assert c._t0 > 0


# ============================================================================
# FeedbackCollector.add() — Chained API
# ============================================================================

class TestFeedbackCollectorAdd:
    def test_add_single_channel(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        assert c.has(FeedbackChannel.CODE)

    def test_add_returns_self_for_chaining(self):
        c = FeedbackCollector("f-1", "my_factor")
        result = c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        assert result is c

    def test_chain_multiple_channels(self):
        c = (FeedbackCollector("f-1", "my_factor")
             .add(FeedbackChannel.CODE, passed=True, detail="ok")
             .add(FeedbackChannel.SHAPE, passed=False, detail="bad shape")
             .add(FeedbackChannel.VALUE, passed=True, detail="value ok"))
        assert c.has(FeedbackChannel.CODE)
        assert c.has(FeedbackChannel.SHAPE)
        assert c.has(FeedbackChannel.VALUE)

    def test_add_with_score(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok", score=0.95)
        fb = c.get(FeedbackChannel.CODE)
        assert fb.score == 0.95

    def test_add_with_metadata(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.EXECUTION, passed=True, detail="ok", exit_code=0)
        fb = c.get(FeedbackChannel.EXECUTION)
        assert fb.metadata.get("exit_code") == 0

    def test_add_overwrites_same_channel(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="first")
        c.add(FeedbackChannel.CODE, passed=False, detail="second")
        fb = c.get(FeedbackChannel.CODE)
        assert fb.passed is False
        assert fb.detail == "second"


# ============================================================================
# FeedbackCollector.add_feedback()
# ============================================================================

class TestFeedbackCollectorAddFeedback:
    def test_add_feedback_object(self):
        c = FeedbackCollector("f-1", "my_factor")
        fb = ChannelFeedback(
            channel=FeedbackChannel.LLM,
            passed=True,
            detail="llm ok",
            score=0.8,
        )
        c.add_feedback(fb)
        assert c.has(FeedbackChannel.LLM)
        assert c.get(FeedbackChannel.LLM) is fb

    def test_add_feedback_returns_self(self):
        c = FeedbackCollector("f-1", "my_factor")
        fb = ChannelFeedback(channel=FeedbackChannel.CODE, passed=True, detail="ok")
        result = c.add_feedback(fb)
        assert result is c


# ============================================================================
# FeedbackCollector.has() / get()
# ============================================================================

class TestFeedbackCollectorHasGet:
    def test_has_returns_false_for_missing(self):
        c = FeedbackCollector("f-1", "my_factor")
        assert c.has(FeedbackChannel.LLM) is False

    def test_get_returns_none_for_missing(self):
        c = FeedbackCollector("f-1", "my_factor")
        assert c.get(FeedbackChannel.LLM) is None

    def test_get_returns_channel_feedback(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        fb = c.get(FeedbackChannel.CODE)
        assert fb is not None
        assert fb.passed is True


# ============================================================================
# FeedbackCollector.finalize() — agg_mode
# ============================================================================

class TestFeedbackCollectorFinalize:
    def test_finalize_empty_channels_decision_true(self):
        c = FeedbackCollector("f-1", "my_factor")
        fb = c.finalize()
        assert fb.decision is True

    def test_finalize_all_agg_all_pass(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        c.add(FeedbackChannel.SHAPE, passed=True, detail="ok")
        fb = c.finalize(agg_mode="all")
        assert fb.decision is True

    def test_finalize_all_agg_any_fail(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        c.add(FeedbackChannel.SHAPE, passed=False, detail="bad")
        fb = c.finalize(agg_mode="all")
        assert fb.decision is False

    def test_finalize_any_agg_one_pass(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        c.add(FeedbackChannel.SHAPE, passed=False, detail="bad")
        fb = c.finalize(agg_mode="any")
        assert fb.decision is True

    def test_finalize_any_agg_all_fail(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=False, detail="bad")
        c.add(FeedbackChannel.SHAPE, passed=False, detail="bad")
        fb = c.finalize(agg_mode="any")
        assert fb.decision is False

    def test_finalize_majority_agg_pass(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        c.add(FeedbackChannel.SHAPE, passed=True, detail="ok")
        c.add(FeedbackChannel.VALUE, passed=False, detail="bad")
        fb = c.finalize(agg_mode="majority")
        assert fb.decision is True  # 2/3 pass

    def test_finalize_majority_agg_fail(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=False, detail="bad")
        c.add(FeedbackChannel.SHAPE, passed=False, detail="bad")
        c.add(FeedbackChannel.VALUE, passed=True, detail="ok")
        fb = c.finalize(agg_mode="majority")
        assert fb.decision is False  # 1/3 pass

    def test_finalize_unknown_agg_raises(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        with pytest.raises(ValueError, match="未知 agg_mode"):
            c.finalize(agg_mode="invalid_mode")

    def test_finalize_explicit_decision(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=False, detail="bad")
        fb = c.finalize(decision=True)
        assert fb.decision is True

    def test_finalize_with_summary(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        fb = c.finalize(summary="manual summary")
        assert fb.summary == "manual summary"

    def test_finalize_auto_summary_all_pass(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        fb = c.finalize()
        assert "全部通过" in fb.summary

    def test_finalize_auto_summary_with_failures(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=False, detail="bad")
        fb = c.finalize()
        assert "失败通道" in fb.summary

    def test_finalize_preserves_factor_id_name(self):
        c = FeedbackCollector("f-123", "my_alpha")
        fb = c.finalize()
        assert fb.factor_id == "f-123"
        assert fb.factor_name == "my_alpha"

    def test_finalize_records_duration(self):
        c = FeedbackCollector("f-1", "my_factor")
        fb = c.finalize()
        assert fb.duration_ms >= 0

    def test_finalize_with_metadata(self):
        c = FeedbackCollector("f-1", "my_factor")
        fb = c.finalize(source="test", round=1)
        assert fb.metadata.get("source") == "test"
        assert fb.metadata.get("round") == 1

    def test_finalize_channels_dict_preserved(self):
        c = FeedbackCollector("f-1", "my_factor")
        c.add(FeedbackChannel.CODE, passed=True, detail="ok")
        c.add(FeedbackChannel.SHAPE, passed=True, detail="ok")
        fb = c.finalize()
        assert len(fb.channels) == 2
        assert FeedbackChannel.CODE in fb.channels
        assert FeedbackChannel.SHAPE in fb.channels


# ============================================================================
# FactorFeedback.to_dict() / from_dict()
# ============================================================================

class TestFactorFeedbackSerialization:
    def test_to_dict_basic(self):
        fb = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            decision=True,
            summary="ok",
        )
        d = fb.to_dict()
        assert d["factor_id"] == "f-1"
        assert d["factor_name"] == "test"
        assert d["decision"] is True

    def test_to_dict_with_channels(self):
        fb = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            channels={
                FeedbackChannel.CODE: ChannelFeedback(
                    channel=FeedbackChannel.CODE,
                    passed=True,
                    detail="ok",
                ),
            },
        )
        d = fb.to_dict()
        assert "code" in d["channels"]
        assert d["channels"]["code"]["passed"] is True

    def test_roundtrip_dict(self):
        original = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            decision=True,
            summary="ok",
        )
        d = original.to_dict()
        restored = FactorFeedback.from_dict(d)
        assert restored.factor_id == original.factor_id
        assert restored.factor_name == original.factor_name
        assert restored.decision == original.decision

    def test_roundtrip_with_channels(self):
        original = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            channels={
                FeedbackChannel.CODE: ChannelFeedback(
                    channel=FeedbackChannel.CODE,
                    passed=True,
                    detail="ok",
                    score=0.9,
                ),
            },
        )
        d = original.to_dict()
        restored = FactorFeedback.from_dict(d)
        assert FeedbackChannel.CODE in restored.channels
        assert restored.channels[FeedbackChannel.CODE].score == 0.9


# ============================================================================
# FactorFeedback JSON Persistence
# ============================================================================

class TestFactorFeedbackJSON:
    def test_save_and_load_json(self, tmp_path):
        original = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            decision=True,
            summary="ok",
        )
        path = tmp_path / "fb.json"
        original.save_json(path)
        assert path.exists()

        loaded = FactorFeedback.load_json(path)
        assert loaded.factor_id == "f-1"
        assert loaded.factor_name == "test"

    def test_save_json_creates_parent_dir(self, tmp_path):
        original = FactorFeedback(factor_id="f-1", factor_name="test")
        path = tmp_path / "subdir" / "fb.json"
        original.save_json(path)
        assert path.exists()


# ============================================================================
# FactorFeedback Parquet Persistence
# ============================================================================

class TestFactorFeedbackParquet:
    def test_to_parquet_row_basic(self):
        fb = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            decision=True,
            summary="ok",
        )
        row = fb.to_parquet_row()
        assert row["factor_id"] == "f-1"
        assert row["decision"] is True

    def test_to_parquet_row_all_channels(self):
        fb = FactorFeedback(
            factor_id="f-1",
            factor_name="test",
            channels={
                FeedbackChannel.CODE: ChannelFeedback(
                    channel=FeedbackChannel.CODE,
                    passed=True,
                    detail="ok",
                ),
            },
        )
        row = fb.to_parquet_row()
        assert row["code_passed"] is True
        assert row["code_detail"] == "ok"
        # Other channels are None
        assert row["shape_passed"] is None

    def test_save_and_load_parquet_single(self, tmp_path):
        fb = FactorFeedback(factor_id="f-1", factor_name="test", decision=True)
        path = tmp_path / "fb.parquet"
        fb.save_parquet(path)
        assert path.exists()

        loaded_list = FactorFeedback.load_parquet(path)
        assert len(loaded_list) == 1
        assert loaded_list[0].factor_id == "f-1"

    def test_save_parquet_appends(self, tmp_path):
        """save_parquet appends to existing file."""
        fb1 = FactorFeedback(factor_id="f-1", factor_name="test1")
        fb2 = FactorFeedback(factor_id="f-2", factor_name="test2")

        path = tmp_path / "fb.parquet"
        fb1.save_parquet(path)
        fb2.save_parquet(path)

        loaded = FactorFeedback.load_parquet(path)
        assert len(loaded) == 2
        assert loaded[0].factor_id == "f-1"
        assert loaded[1].factor_id == "f-2"


# ============================================================================
# ChannelFeedback.values()
# ============================================================================

class TestChannelFeedbackValues:
    def test_values_returns_tuple(self):
        fb = ChannelFeedback(
            channel=FeedbackChannel.CODE,
            passed=True,
            detail="ok",
            score=0.5,
        )
        passed, detail, score = fb.values()
        assert passed is True
        assert detail == "ok"
        assert score == 0.5


# ============================================================================
# ensure_feedback() helper
# ============================================================================

class TestEnsureFeedback:
    def test_from_factor_feedback(self):
        original = FactorFeedback(factor_id="orig", factor_name="orig_name")
        result = ensure_feedback(original, factor_id="new_id", factor_name="new_name")
        assert result is original

    def test_from_factor_feedback_preserves_id(self):
        original = FactorFeedback(factor_id="orig", factor_name="orig_name")
        result = ensure_feedback(original, factor_id="new_id", factor_name="new_name")
        assert result.factor_id == "orig"  # Already set, not overwritten
        assert result.factor_name == "orig_name"

    def test_from_dict(self):
        result = ensure_feedback(
            {"ic": 0.05, "ir": 0.5},
            factor_id="f-1",
            factor_name="my_factor",
        )
        assert result.decision is True
        assert result.factor_id == "f-1"
        assert "my_factor" == result.factor_name

    def test_from_invalid_type_raises(self):
        with pytest.raises(TypeError, match="不支持"):
            ensure_feedback("not a dict or feedback", factor_id="f-1", factor_name="x")

    def test_from_list_raises(self):
        with pytest.raises(TypeError):
            ensure_feedback([1, 2, 3], factor_id="f-1", factor_name="x")


# ============================================================================
# _safe_scalar helper
# ============================================================================

class TestSafeScalar:
    def test_passes_through_int(self):
        assert _safe_scalar(42) == 42

    def test_passes_through_float(self):
        assert _safe_scalar(3.14) == 3.14

    def test_passes_through_str(self):
        assert _safe_scalar("hello") == "hello"

    def test_numpy_int(self):
        import numpy as np
        assert _safe_scalar(np.int64(42)) == 42

    def test_numpy_float(self):
        import numpy as np
        assert _safe_scalar(np.float64(3.14)) == 3.14

    def test_numpy_bool(self):
        import numpy as np
        assert _safe_scalar(np.bool_(True)) is True

    def test_empty_series_returns_none(self):
        s = pd.Series([], dtype=float)
        assert _safe_scalar(s) is None

    def test_series_to_scalar(self):
        s = pd.Series([42])
        assert _safe_scalar(s) == 42


# ============================================================================
# All 8 Channels
# ============================================================================

class TestAllChannels:
    def test_all_8_channels_can_be_added(self):
        c = FeedbackCollector("f-1", "test")
        for ch in FeedbackChannel:
            c.add(ch, passed=True, detail=f"ok {ch.value}")
        for ch in FeedbackChannel:
            assert c.has(ch)

    def test_finalize_with_all_8_channels(self):
        c = FeedbackCollector("f-1", "test")
        for ch in FeedbackChannel:
            c.add(ch, passed=True, detail="ok")
        fb = c.finalize()
        assert len(fb.channels) == len(FeedbackChannel)