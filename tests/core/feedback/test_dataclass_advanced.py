"""FactorFeedback dataclass/collector 进阶边界测试 (20 tests)。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
    FeedbackCollector,
    ensure_feedback,
)
from QuantNodes.core.feedback.dataclass import _safe_scalar


class TestSafeScalar:
    def test_series_empty(self):
        assert _safe_scalar(pd.Series([], dtype=float)) is None

    def test_series_first_value(self):
        assert _safe_scalar(pd.Series([1.5, 2.5])) == 1.5

    def test_ndarray_empty(self):
        assert _safe_scalar(np.array([])) is None

    def test_ndarray_first_value(self):
        assert _safe_scalar(np.array([[7, 8], [9, 10]])) == 7

    def test_numpy_scalar_types(self):
        assert _safe_scalar(np.int64(3)) == 3
        assert _safe_scalar(np.float64(1.2)) == 1.2
        assert _safe_scalar(np.bool_(True)) is True

    def test_numpy_nan_inf_to_none(self):
        assert _safe_scalar(np.float64(np.nan)) is None
        assert _safe_scalar(np.float64(np.inf)) is None


class TestEnsureFeedbackAdvanced:
    def test_existing_feedback_fills_missing_id_name(self):
        fb = FactorFeedback(factor_id="", factor_name="", decision=True)
        out = ensure_feedback(fb, "fid", "fname")
        assert out is fb
        assert out.factor_id == "fid"
        assert out.factor_name == "fname"

    def test_existing_feedback_preserves_existing_id_name(self):
        fb = FactorFeedback(factor_id="old", factor_name="oldname", decision=True)
        out = ensure_feedback(fb, "new", "newname")
        assert out.factor_id == "old"
        assert out.factor_name == "oldname"

    def test_dict_only_known_metrics_in_metadata(self):
        out = ensure_feedback({"sharpe": 1.2, "unknown": 9}, "fid", "fname")
        assert out.metadata == {"sharpe": 1.2}
        assert "unknown" not in out.metadata

    def test_dict_safe_scalar_conversion(self):
        out = ensure_feedback({
            "sharpe": pd.Series([1.5]),
            "arr": np.array([0.2]),
            "mdd": np.float64(np.nan),
        }, "fid", "fname")
        assert out.metadata["sharpe"] == 1.5
        assert out.metadata["arr"] == 0.2
        assert out.metadata["mdd"] is None

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="节点返回类型不支持"):
            ensure_feedback([1, 2, 3], "fid", "fname")


class TestParquetAdvanced:
    def _feedback(self, factor_id="f1"):
        return FactorFeedback(
            factor_id=factor_id,
            factor_name="alpha",
            channels={
                FeedbackChannel.CODE: ChannelFeedback(
                    FeedbackChannel.CODE, True, "ok", 0.9,
                    metadata={"nested": {"a": 1}},
                ),
                FeedbackChannel.VALUE: ChannelFeedback(
                    FeedbackChannel.VALUE, False, "bad", 0.1,
                ),
            },
            decision=False,
            summary="mixed",
            metadata={"sharpe": 1.2, "tags": ["a", "b"]},
        )

    def test_save_parquet_appends(self, tmp_path: Path):
        p = tmp_path / "feedback.parquet"
        self._feedback("f1").save_parquet(p)
        self._feedback("f2").save_parquet(p)
        df = pd.read_parquet(p)
        assert len(df) == 2
        assert set(df["factor_id"]) == {"f1", "f2"}

    def test_load_parquet_roundtrip_basic(self, tmp_path: Path):
        p = tmp_path / "feedback.parquet"
        self._feedback("f1").save_parquet(p)
        loaded = FactorFeedback.load_parquet(p)
        assert len(loaded) == 1
        fb = loaded[0]
        assert fb.factor_id == "f1"
        assert fb.factor_name == "alpha"
        assert fb.decision is False
        assert FeedbackChannel.CODE in fb.channels
        assert FeedbackChannel.VALUE in fb.channels

    def test_load_parquet_bad_metadata_json(self, tmp_path: Path):
        p = tmp_path / "feedback.parquet"
        df = pd.DataFrame([{
            "factor_id": "f1", "factor_name": "a", "decision": True,
            "summary": "s", "duration_ms": 1.0,
            "timestamp": datetime.now().isoformat(),
            "metadata": "{bad json",
            "code_passed": True, "code_score": 1.0, "code_detail": "ok",
        }])
        df.to_parquet(p)
        loaded = FactorFeedback.load_parquet(p)[0]
        assert loaded.metadata == {}

    def test_to_parquet_row_all_channels_present(self):
        row = self._feedback().to_parquet_row()
        for ch in FeedbackChannel:
            assert f"{ch.value}_passed" in row
            assert f"{ch.value}_score" in row
            assert f"{ch.value}_detail" in row


class TestFeedbackCollectorAdvanced:
    def test_empty_collector_decision_true(self):
        c = FeedbackCollector("fid", "fname")
        fb = c.finalize()
        assert fb.decision is True
        assert fb.summary == "全部通过"

    def test_chain_add_and_override(self):
        c = FeedbackCollector("fid", "fname")
        c.add(FeedbackChannel.CODE, True, "ok").add(FeedbackChannel.CODE, False, "bad")
        fb = c.finalize()
        assert fb.decision is False
        assert fb.channels[FeedbackChannel.CODE].detail == "bad"

    def test_has_get_missing(self):
        c = FeedbackCollector("fid", "fname")
        assert not c.has(FeedbackChannel.CODE)
        assert c.get(FeedbackChannel.CODE) is None

    def test_finalize_explicit_decision_overrides(self):
        c = FeedbackCollector("fid", "fname")
        c.add(FeedbackChannel.CODE, False, "bad")
        fb = c.finalize(decision=True, summary="forced")
        assert fb.decision is True
        assert fb.summary == "forced"

    def test_finalize_metadata(self):
        c = FeedbackCollector("fid", "fname")
        fb = c.finalize(source="unit", n=3)
        assert fb.metadata == {"source": "unit", "n": 3}
        assert fb.duration_ms >= 0
