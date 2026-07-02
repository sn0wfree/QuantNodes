# coding=utf-8
"""
test_metrics_report.py - MetricsReportBuilder 单元测试 (v3.0.2 Step 3)

覆盖:
- MetricsReportBuilder 从 LogicMiningBatchResult 构建
- to_dict shape
- to_json 写文件
- to_markdown 表格格式
- 空 batch 构建
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from QuantNodes.research.quant_alpha.logic_mining.batch import (
    LogicMiningBatchResult,
    ThreadSafeMetrics,
)
from QuantNodes.research.quant_alpha.logic_mining.report import MetricsReportBuilder


# ======================================================================
# Fixtures
# ======================================================================
def _make_batch(
    *,
    mined: int = 3,
    skipped: int = 2,
    failed: int = 1,
    warnings: List[str] | None = None,
    wall_clock: float = 1.23,
    with_metrics: bool = True,
) -> LogicMiningBatchResult:
    """构造 LogicMiningBatchResult 用于测试"""
    results = [MagicMock() for _ in range(mined)]
    skipped_ids = {f"skip{i}" for i in range(skipped)}
    failed_ids = [(f"fail{i}", f"error{i}") for i in range(failed)]
    metrics = None
    if with_metrics:
        m = ThreadSafeMetrics()
        m.record_call_failure("logic-mining-structure")
        m.record_parse_failure("logic-mining-semantics", layer_reached=2)
        m.record_structured_failure("logic-mining-abstraction")
        metrics = m
    pool = MagicMock()
    pool.summary.return_value = {"n_total": mined, "by_source_lib": {"alpha101": mined}}
    pool.values.return_value = [
        MagicMock(source_lib="alpha101") for _ in range(mined)
    ]
    batch = LogicMiningBatchResult(
        results=results,
        pool=pool,
        metrics=metrics,
        skipped_ids=skipped_ids,
        attempted_ids=[f"alpha101-f{i}" for i in range(mined + failed)],
        failed_ids=failed_ids,
        wall_clock_s=wall_clock,
        warnings=warnings or [],
    )
    return batch


# ======================================================================
# from_batch 构建
# ======================================================================
class TestMetricsReportBuilderConstruction:
    def test_from_batch_populates_fields(self):
        batch = _make_batch(mined=3, skipped=2, failed=1, warnings=["w1"])
        report = MetricsReportBuilder.from_batch(batch)
        assert report.total_attempted == 4
        assert report.total_mined == 3
        assert report.total_skipped == 2
        assert report.total_failed == 1
        assert report.warnings == ["w1"]

    def test_from_batch_with_metrics(self):
        batch = _make_batch(with_metrics=True)
        report = MetricsReportBuilder.from_batch(batch)
        assert "logic-mining-structure" in report.agent_stats
        assert report.agent_stats["logic-mining-structure"]["call_failures"] == 1
        assert report.agent_stats["logic-mining-semantics"]["parse_failures"] == 1

    def test_from_batch_without_metrics(self):
        batch = _make_batch(with_metrics=False)
        report = MetricsReportBuilder.from_batch(batch)
        assert report.agent_stats == {}

    def test_source_lib_breakdown(self):
        batch = _make_batch(mined=2, failed=1)
        report = MetricsReportBuilder.from_batch(batch)
        assert "alpha101" in report.source_lib_breakdown
        assert report.source_lib_breakdown["alpha101"]["mined"] == 2


# ======================================================================
# to_dict
# ======================================================================
class TestMetricsReportBuilderToDict:
    def test_to_dict_shape(self):
        batch = _make_batch(mined=3, skipped=2, failed=1, wall_clock=2.5)
        report = MetricsReportBuilder.from_batch(batch)
        d = report.to_dict()
        assert "report" in d
        assert "summary" in d
        assert "source_lib_breakdown" in d
        assert "agent_stats" in d
        assert "failed_ids" in d
        assert "warnings" in d

    def test_success_rate_calculation(self):
        batch = _make_batch(mined=8, skipped=0, failed=2)
        report = MetricsReportBuilder.from_batch(batch)
        d = report.to_dict()
        assert d["summary"]["success_rate"] == 0.8

    def test_success_rate_zero_attempted(self):
        batch = LogicMiningBatchResult(
            results=[],
            skipped_ids=set(),
            failed_ids=[],
            wall_clock_s=0.1,
        )
        report = MetricsReportBuilder.from_batch(batch)
        d = report.to_dict()
        assert d["summary"]["success_rate"] == 0.0


# ======================================================================
# to_json
# ======================================================================
class TestMetricsReportBuilderJson:
    def test_to_json_creates_file(self, tmp_path: Path):
        batch = _make_batch(mined=2)
        report = MetricsReportBuilder.from_batch(batch)
        json_path = tmp_path / "report.json"
        report.to_json(json_path)
        assert json_path.exists()

    def test_to_json_roundtrip(self, tmp_path: Path):
        import json
        batch = _make_batch(mined=2, failed=1)
        report = MetricsReportBuilder.from_batch(batch)
        json_path = tmp_path / "report.json"
        report.to_json(json_path)
        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["summary"]["total_mined"] == 2
        assert loaded["summary"]["total_failed"] == 1

    def test_to_json_creates_parent_dirs(self, tmp_path: Path):
        batch = _make_batch()
        report = MetricsReportBuilder.from_batch(batch)
        json_path = tmp_path / "nested" / "deep" / "report.json"
        report.to_json(json_path)
        assert json_path.exists()


# ======================================================================
# to_markdown
# ======================================================================
class TestMetricsReportBuilderMarkdown:
    def test_to_markdown_basic_shape(self):
        batch = _make_batch(mined=3, skipped=2, failed=1, warnings=["w1"])
        report = MetricsReportBuilder.from_batch(batch)
        md = report.to_markdown()
        assert "# Logic Mining Run Report" in md
        assert "**Generated**:" in md
        assert "**Wall clock**:" in md
        assert "## Summary" in md
        assert "| Metric | Value |" in md
        assert "## Source Library Breakdown" in md
        assert "## Agent Statistics" in md
        assert "## Failed Formulas" in md
        assert "## Warnings" in md
        assert "w1" in md

    def test_to_markdown_includes_failed_ids(self):
        batch = _make_batch(mined=1, failed=2)
        report = MetricsReportBuilder.from_batch(batch)
        md = report.to_markdown()
        assert "fail0" in md
        assert "fail1" in md

    def test_to_markdown_empty_batch(self):
        batch = LogicMiningBatchResult()
        report = MetricsReportBuilder.from_batch(batch)
        md = report.to_markdown()
        assert "# Logic Mining Run Report" in md
        assert "## Summary" in md

    def test_to_markdown_rate_percentage(self):
        batch = _make_batch(mined=8, skipped=0, failed=2)
        report = MetricsReportBuilder.from_batch(batch)
        md = report.to_markdown()
        assert "80.0%" in md

    def test_to_markdown_footer(self):
        batch = _make_batch()
        report = MetricsReportBuilder.from_batch(batch)
        md = report.to_markdown()
        assert "QuantNodes Logic Mining v3.0.2" in md