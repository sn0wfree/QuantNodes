"""Tests for QuantNodes.research.sink — base/single_json/batch_summary/yaml_duckdb."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl
import pytest


# ── Helpers ────────────────────────────────────────────────────


def _make_signal(signal_id: str = "alpha-001") -> "Signal":
    from QuantNodes.research.signal_source.base import Signal
    return Signal(
        id=signal_id,
        name="Test Signal",
        formula_brief="ts_mean(close, 5)",
    )


def _make_result(
    status: str = "success",
    signal_id: str = "alpha-001",
    ic_mean: float | None = 0.05,
    code: str | None = "ts_mean(close, 5)",
) -> "FactorResult":
    from QuantNodes.research.backtest.base import FactorResult
    return FactorResult(
        signal=_make_signal(signal_id),
        status=status,
        code=code,
        code_chars=len(code) if code else 0,
        factor_series=pl.Series("f", [1.0, 2.0, 3.0]) if status == "success" else None,
        backtest={"ic_mean": ic_mean, "icir": 0.5, "ic_winrate": 0.51},
        stage="done" if status == "success" else "compile",
        error=None if status == "success" else "boom",
        elapsed_sec=1.5,
    )


# ── Sink Protocol ──────────────────────────────────────────────


class TestSinkProtocol:
    """Verify Sink Protocol methods exist with right signatures."""

    def test_sink_protocol_has_three_methods(self):
        from QuantNodes.research.sink.base import Sink

        assert hasattr(Sink, "write_one")
        assert hasattr(Sink, "write_batch")
        assert hasattr(Sink, "flush")


# ── SingleJsonSink ─────────────────────────────────────────────


class TestSingleJsonSink:
    """Tests for SingleJsonSink (per-signal JSON writer)."""

    def test_creates_output_dir(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path / "nested" / "subdir")
        sink.write_one(_make_result())
        assert (tmp_path / "nested" / "subdir").exists()

    def test_write_one_returns_path(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        result = sink.write_one(_make_result())
        assert result.exists()
        assert result.name == "single_factor_alpha-001.json"

    def test_filename_sanitizes_slashes(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        result = sink.write_one(_make_result(signal_id="1601/00991"))
        assert result.name == "single_factor_1601_00991.json"

    def test_filename_sanitizes_backslashes(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        result = sink.write_one(_make_result(signal_id="a\\b"))
        assert result.name == "single_factor_a_b.json"

    def test_json_has_expected_fields(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        path = sink.write_one(_make_result())
        data = json.loads(path.read_text())
        assert data["status"] == "success"
        assert data["code"] == "ts_mean(close, 5)"
        assert data["ic_mean"] == 0.05

    def test_failed_result_written(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        path = sink.write_one(_make_result(status="failed"))
        data = json.loads(path.read_text())
        assert data["status"] == "failed"
        assert data["error"] == "boom"

    def test_indent_custom(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path, indent=4)
        path = sink.write_one(_make_result())
        content = path.read_text()
        assert "\n    " in content  # 4-space indent

    def test_write_batch_returns_empty(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        assert sink.write_batch([]) == []

    def test_flush_is_noop(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        assert sink.flush() is None

    def test_output_dir_property(self, tmp_path: Path):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        assert sink.output_dir == tmp_path

    def test_debug_log(self, tmp_path: Path, caplog):
        from QuantNodes.research.sink.single_json import SingleJsonSink

        sink = SingleJsonSink(tmp_path)
        with caplog.at_level(logging.DEBUG, logger="QuantNodes.research.sink.single_json"):
            sink.write_one(_make_result())
        assert any("[sink] wrote" in r.message for r in caplog.records)


# ── BatchSummarySink ───────────────────────────────────────────


class TestBatchSummarySink:
    """Tests for BatchSummarySink (end-of-batch JSON + MD writer)."""

    def test_write_batch_writes_json_and_md(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="test")
        results = [_make_result(signal_id=f"alpha-{i:03d}") for i in range(3)]
        paths = sink.write_batch(results)
        assert len(paths) == 2
        assert sink.json_path.exists()
        assert sink.md_path.exists()

    def test_default_filenames_use_paper_id(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="my_paper")
        assert sink.json_path.name == "multi_alpha_my_paper.json"
        assert sink.md_path.name == "multi_alpha_my_paper.md"

    def test_custom_filenames(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(
            tmp_path,
            paper_id="ignored",
            json_filename="custom.json",
            md_filename="custom.md",
        )
        assert sink.json_path.name == "custom.json"
        assert sink.md_path.name == "custom.md"

    def test_json_content_has_alphas(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="test")
        results = [_make_result(signal_id=f"alpha-{i:03d}") for i in range(2)]
        sink.write_batch(results)
        data = json.loads(sink.json_path.read_text())
        assert data["total"] == 2
        assert len(data["alphas"]) == 2

    def test_md_content_has_table(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="test")
        results = [_make_result()]
        sink.write_batch(results)
        content = sink.md_path.read_text()
        assert "| Alpha |" in content

    def test_write_one_returns_dev_null(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path)
        result = sink.write_one(_make_result())
        assert result == Path("/dev/null")

    def test_creates_output_dir(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path / "nested")
        sink.write_batch([_make_result()])
        assert (tmp_path / "nested").exists()

    def test_log_summary_disabled(self, tmp_path: Path, caplog):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="test", log_summary=False)
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            sink.write_batch([_make_result()])
        # log_summary should NOT have been called → no "Summary" log
        assert not any("Summary" in r.message for r in caplog.records)

    def test_log_summary_enabled(self, tmp_path: Path, caplog):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="test", log_summary=True)
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            sink.write_batch([_make_result()])
        assert any("Summary" in r.message for r in caplog.records)

    def test_flush_is_noop(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path)
        assert sink.flush() is None

    def test_failed_results_included(self, tmp_path: Path):
        from QuantNodes.research.sink.batch_summary import BatchSummarySink

        sink = BatchSummarySink(tmp_path, paper_id="test")
        results = [
            _make_result(status="success", signal_id="alpha-001"),
            _make_result(status="failed", signal_id="alpha-002"),
        ]
        paths = sink.write_batch(results)
        assert len(paths) == 2
        data = json.loads(sink.json_path.read_text())
        assert data["success_count"] == 1
        assert data["failed_count"] == 1