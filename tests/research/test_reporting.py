"""Tests for QuantNodes.research.reporting — aggregator/serializer/reporter/adapters."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pytest


# ── BatchAggregator ────────────────────────────────────────────


class TestBatchAggregatorAggregate:
    """Tests for BatchAggregator.aggregate (NaN-safe metrics)."""

    def test_empty_list(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        agg = BatchAggregator.aggregate([])
        assert agg["total"] == 0
        assert agg["success_count"] == 0
        assert agg["failed_count"] == 0
        assert agg["ic_mean"] is None
        assert agg["icir"] is None
        assert agg["winrate"] is None

    def test_all_success(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        results = [
            {"status": "success", "ic_mean": 0.05, "icir": 0.5, "ic_winrate": 0.51},
            {"status": "success", "ic_mean": 0.03, "icir": 0.3, "ic_winrate": 0.49},
        ]
        agg = BatchAggregator.aggregate(results)
        assert agg["total"] == 2
        assert agg["success_count"] == 2
        assert agg["failed_count"] == 0
        assert agg["ic_mean"] == 0.04
        assert agg["icir"] == 0.4
        assert agg["winrate"] == 0.5

    def test_mixed_success_failed(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        results = [
            {"status": "success", "ic_mean": 0.05, "icir": 0.5, "ic_winrate": 0.5},
            {"status": "failed"},
            {"status": "success", "ic_mean": 0.03, "icir": 0.3, "ic_winrate": 0.5},
        ]
        agg = BatchAggregator.aggregate(results)
        assert agg["total"] == 3
        assert agg["success_count"] == 2
        assert agg["failed_count"] == 1

    def test_nan_filtered_out(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        results = [
            {"status": "success", "ic_mean": 0.05, "icir": float("nan"), "ic_winrate": 0.5},
            {"status": "success", "ic_mean": 0.03, "icir": 0.3, "ic_winrate": 0.5},
        ]
        agg = BatchAggregator.aggregate(results)
        # icir nan filtered; mean over [0.3] = 0.3
        assert agg["icir"] == 0.3
        # ic_mean: both valid → 0.04
        assert agg["ic_mean"] == 0.04

    def test_all_nan_returns_none(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        results = [
            {"status": "success", "ic_mean": float("nan"), "icir": float("nan")},
        ]
        agg = BatchAggregator.aggregate(results)
        assert agg["ic_mean"] is None
        assert agg["icir"] is None

    def test_missing_keys_default_to_nan_filtered(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        results = [{"status": "success"}]  # no ic fields
        agg = BatchAggregator.aggregate(results)
        assert agg["success_count"] == 1
        assert agg["ic_mean"] is None

    def test_rounded_to_4_decimals(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        results = [
            {"status": "success", "ic_mean": 0.123456, "icir": 1.234567, "ic_winrate": 0.5123},
        ]
        agg = BatchAggregator.aggregate(results)
        assert agg["ic_mean"] == 0.1235
        assert agg["icir"] == 1.2346
        assert agg["winrate"] == 0.5123


class TestBatchAggregatorFormatMetric:
    """Tests for BatchAggregator.format_metric (NaN-safe string formatter)."""

    def test_none_returns_na(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        assert BatchAggregator.format_metric(None) == "  NaN"

    def test_nan_returns_na(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        assert BatchAggregator.format_metric(float("nan")) == "  NaN"

    def test_format_with_sign(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        assert BatchAggregator.format_metric(0.05) == "+0.0500"
        assert BatchAggregator.format_metric(-0.03) == "-0.0300"

    def test_format_zero(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        assert BatchAggregator.format_metric(0.0) == "+0.0000"

    def test_custom_format_spec(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        assert BatchAggregator.format_metric(0.123, fmt=".2f") == "0.12"

    def test_custom_na_string(self):
        from QuantNodes.research.reporting.aggregator import BatchAggregator

        assert BatchAggregator.format_metric(None, na="MISSING") == "MISSING"
        assert BatchAggregator.format_metric(float("nan"), na="N/A") == "N/A"


# ── BatchSerializer ────────────────────────────────────────────


class TestBatchSerializerWriteJson:
    """Tests for BatchSerializer.write_json."""

    def _results(self):
        return [
            {
                "alpha_index": 1,
                "status": "success",
                "ic_mean": 0.05,
                "icir": 0.5,
                "ic_winrate": 0.51,
                "code_chars": 100,
                "elapsed_sec": 1.5,
                "stage": "done",
            },
            {
                "alpha_index": 2,
                "status": "failed",
                "error": "compile error",
                "stage": "compile",
            },
        ]

    def test_write_and_read(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        path = tmp_path / "out.json"
        BatchSerializer.write_json(self._results(), path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["total"] == 2
        assert data["success_count"] == 1
        assert data["failed_count"] == 1
        assert len(data["alphas"]) == 2

    def test_error_truncated_to_200(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"status": "failed", "error": "x" * 500}]
        path = tmp_path / "out.json"
        BatchSerializer.write_json(results, path)
        data = json.loads(path.read_text())
        assert len(data["alphas"][0]["error"]) == 200

    def test_aggregate_section(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [
            {"status": "success", "ic_mean": 0.05, "icir": 0.5, "ic_winrate": 0.5},
            {"status": "success", "ic_mean": 0.03, "icir": 0.3, "ic_winrate": 0.5},
        ]
        path = tmp_path / "out.json"
        BatchSerializer.write_json(results, path)
        data = json.loads(path.read_text())
        assert data["aggregate"]["ic_mean_avg"] == 0.04
        assert data["aggregate"]["icir_avg"] == 0.4

    def test_none_stage_coerced_to_empty(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"status": "failed", "stage": None}]
        path = tmp_path / "out.json"
        BatchSerializer.write_json(results, path)
        data = json.loads(path.read_text())
        assert data["alphas"][0]["stage"] == ""

    def test_ensure_ascii_false(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"status": "success", "stage": "完成"}]
        path = tmp_path / "out.json"
        BatchSerializer.write_json(results, path)
        raw = path.read_text(encoding="utf-8")
        assert "完成" in raw


class TestBatchSerializerWriteMarkdown:
    """Tests for BatchSerializer.write_markdown."""

    def test_write_minimal(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"alpha_index": 1, "status": "success", "ic_mean": 0.05}]
        path = tmp_path / "out.md"
        BatchSerializer.write_markdown(results, path)
        content = path.read_text()
        assert "# 101-Alpha Batch Results (v2)" in content
        assert "| Alpha |" in content

    def test_with_metrics(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [
            {"alpha_index": 1, "status": "success",
             "ic_mean": 0.05, "icir": 0.5, "ic_winrate": 0.51,
             "code_chars": 100, "elapsed_sec": 1.5},
        ]
        path = tmp_path / "out.md"
        BatchSerializer.write_markdown(results, path)
        content = path.read_text()
        assert "Avg IC:" in content
        assert "Avg ICIR:" in content
        assert "Avg Winrate:" in content

    def test_with_failed_section(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [
            {"alpha_index": 1, "status": "failed",
             "stage": "compile", "error": "syntax error"},
        ]
        path = tmp_path / "out.md"
        BatchSerializer.write_markdown(results, path)
        content = path.read_text()
        assert "## Failed Alphas" in content
        assert "alpha-001" in content
        assert "compile" in content

    def test_index_format_three_digits(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"alpha_index": 5, "status": "success"}]
        path = tmp_path / "out.md"
        BatchSerializer.write_markdown(results, path)
        content = path.read_text()
        assert "alpha-005" in content

    def test_index_format_non_int(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"alpha_index": "x", "status": "success"}]
        path = tmp_path / "out.md"
        BatchSerializer.write_markdown(results, path)
        content = path.read_text()
        assert "alpha-x" in content

    def test_no_metrics_no_avg_line(self, tmp_path: Path):
        from QuantNodes.research.reporting.serializer import BatchSerializer

        results = [{"alpha_index": 1, "status": "failed"}]
        path = tmp_path / "out.md"
        BatchSerializer.write_markdown(results, path)
        content = path.read_text()
        assert "Avg IC:" not in content


# ── BatchReporter ──────────────────────────────────────────────


class TestBatchReporterLogBanner:
    """Tests for BatchReporter.log_banner."""

    def test_logs_banner(self, caplog):
        from QuantNodes.research.reporting.reporter import BatchReporter

        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            BatchReporter.log_banner()
        assert any("101-Alpha Batch Runner (v2)" in r.message for r in caplog.records)


class TestBatchReporterLogRow:
    """Tests for BatchReporter.log_row."""

    def test_logs_success_row(self, caplog):
        from QuantNodes.research.reporting.reporter import BatchReporter

        result = {
            "status": "success",
            "ic_mean": 0.05,
            "icir": 0.5,
            "ic_winrate": 0.51,
            "elapsed_sec": 1.5,
        }
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            BatchReporter.log_row(1, result, elapsed_cum=1.5)
        assert any("success" in r.message for r in caplog.records)

    def test_logs_failed_row_includes_stage(self, caplog):
        from QuantNodes.research.reporting.reporter import BatchReporter

        result = {
            "status": "failed",
            "stage": "compile",
            "elapsed_sec": 0.5,
        }
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            BatchReporter.log_row(2, result, elapsed_cum=0.5)
        assert any("compile" in r.message for r in caplog.records)

    def test_winrate_nan_safe(self, caplog):
        from QuantNodes.research.reporting.reporter import BatchReporter

        result = {
            "status": "success",
            "ic_mean": 0.05,
            "icir": 0.5,
            "ic_winrate": None,
            "elapsed_sec": 1.0,
        }
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            BatchReporter.log_row(1, result, elapsed_cum=1.0)
        # Should not crash, should log "NaN"
        assert any("NaN" in r.message for r in caplog.records)


class TestBatchReporterLogSummary:
    """Tests for BatchReporter.log_summary."""

    def test_summary_with_mixed_results(self, caplog):
        from QuantNodes.research.reporting.reporter import BatchReporter

        results = [
            {"status": "success", "ic_mean": 0.05, "icir": 0.5, "ic_winrate": 0.5},
            {"status": "failed", "stage": "compile", "error": "boom"},
        ]
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            BatchReporter.log_summary(results)
        msgs = [r.message for r in caplog.records]
        assert any("Total:  2" in m for m in msgs)
        assert any("Success: 1" in m for m in msgs)
        assert any("Failed: 1" in m for m in msgs)

    def test_summary_with_avg_metrics(self, caplog):
        from QuantNodes.research.reporting.reporter import BatchReporter

        results = [
            {"status": "success", "ic_mean": 0.04, "icir": 0.4, "ic_winrate": 0.5},
        ]
        with caplog.at_level(logging.INFO, logger="QuantNodes.research.reporting.reporter"):
            BatchReporter.log_summary(results)
        assert any("Avg IC:" in r.message for r in caplog.records)


# ── adapters ───────────────────────────────────────────────────


class TestFactorResultsToDicts:
    """Tests for factor_results_to_dicts (bridge FactorResult → dict)."""

    def test_empty_list(self):
        from QuantNodes.research.reporting.adapters import factor_results_to_dicts

        assert factor_results_to_dicts([]) == []

    def test_calls_to_dict(self):
        """factor_results_to_dicts delegates to each result's to_dict()."""

        class _Stub:
            def to_dict(self):
                return {"alpha_index": 1, "status": "success"}

        from QuantNodes.research.reporting.adapters import factor_results_to_dicts

        out = factor_results_to_dicts([_Stub(), _Stub()])
        assert out == [
            {"alpha_index": 1, "status": "success"},
            {"alpha_index": 1, "status": "success"},
        ]