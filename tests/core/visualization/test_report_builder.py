# coding=utf-8
"""Tests for ReportBuilder (Phase 1.3, Builder pattern).

Covers:
  - Fluent API (with_title/with_overview/add_section/build)
  - Section rendering (plotly Figure + non-plotly fallback)
  - with_evolve_preset (backward compat with generate_report)
  - build_to_html (file write + return string)
  - Backward compat: generate_report/generate_html still work
"""
from typing import List
from unittest.mock import MagicMock

import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.trajectory import TrajectoryEntry
from QuantNodes.core.visualization import (
    Report,
    ReportBuilder,
    Section,
    generate_html,
    generate_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_entry(round_idx: int, sharpe: float = 0.5, decision: bool = True) -> TrajectoryEntry:
    fb = FactorFeedback(decision=decision)
    return TrajectoryEntry(
        round_idx=round_idx,
        metrics={"sharpe": sharpe},
        feedback=fb,
    )


@pytest.fixture
def small_entries() -> List[TrajectoryEntry]:
    return [
        make_entry(0, 0.3, decision=True),
        make_entry(1, 0.5, decision=True),
        make_entry(1, 0.2, decision=False),
        make_entry(2, 0.8, decision=True),
    ]


def make_fake_figure(title: str = "fake fig") -> MagicMock:
    """Mock 一个有 .to_html() 方法的对象 (模拟 plotly Figure)。"""
    fig = MagicMock()
    fig.to_html.return_value = f"<div id='{title}'></div>"
    return fig


# ---------------------------------------------------------------------------
# Section dataclass
# ---------------------------------------------------------------------------

class TestSection:
    def test_render_with_to_html(self):
        fig = make_fake_figure("test")
        sec = Section(title="My Section", payload=fig)
        out = sec.render(div_id="test")
        assert "test" in out
        fig.to_html.assert_called_once()

    def test_render_non_to_html_falls_back_to_pre(self):
        class Plain:
            def __repr__(self): return "PLAIN_OBJ"
        sec = Section(title="Plain", payload=Plain())
        out = sec.render()
        assert "<pre>" in out
        assert "PLAIN_OBJ" in out


# ---------------------------------------------------------------------------
# ReportBuilder fluent API
# ---------------------------------------------------------------------------

class TestReportBuilderFluent:
    def test_empty_build(self):
        report = ReportBuilder().build()
        assert isinstance(report, Report)
        assert report.title == "Report"
        assert report.overview == {}
        assert report.sections == []

    def test_with_title_returns_self(self):
        b = ReportBuilder()
        assert b.with_title("X") is b

    def test_with_overview_returns_self(self):
        b = ReportBuilder()
        assert b.with_overview({"a": 1}) is b

    def test_add_section_returns_self(self):
        b = ReportBuilder()
        assert b.add_section("S", make_fake_figure()) is b

    def test_full_fluent_chain(self):
        report = (
            ReportBuilder()
            .with_title("My Report")
            .with_overview({"size": 10})
            .add_section("First", make_fake_figure("a"))
            .add_section("Second", make_fake_figure("b"))
            .build()
        )
        assert report.title == "My Report"
        assert report.overview == {"size": 10}
        assert len(report.sections) == 2
        assert report.sections[0].title == "First"
        assert report.sections[1].title == "Second"

    def test_to_dict_uses_lowercased_title_as_key(self):
        report = (
            ReportBuilder()
            .add_section("Lineage DAG", make_fake_figure())
            .add_section("Metrics Distribution", make_fake_figure())
            .build()
        )
        d = report.to_dict()
        assert "lineage_dag" in d
        assert "metrics_distribution" in d


class TestReportBuilderWithEvolvePreset:
    def test_loads_overview_and_5_sections(self, small_entries):
        report = (
            ReportBuilder()
            .with_title("Test")
            .with_evolve_preset(small_entries, metric="sharpe")
            .build()
        )
        # overview
        assert report.overview["size"] == 4
        assert report.overview["passed"] == 3
        assert report.overview["rejected"] == 1
        assert report.overview["metric"] == "sharpe"
        assert report.overview["best_metric"] == 0.8
        # 5 sections
        assert len(report.sections) == 5
        titles = [s.title for s in report.sections]
        assert "lineage_dag" in titles
        assert "metric_distribution" in titles
        assert "metric_per_round" in titles
        assert "gate_breakdown" in titles
        assert "operation_breakdown" in titles

    def test_accepts_mapping(self):
        m = {f"e{i}": make_entry(i) for i in range(3)}
        report = (
            ReportBuilder()
            .with_evolve_preset(m, metric="sharpe")
            .build()
        )
        assert report.overview["size"] == 3

    def test_empty_entries(self):
        report = (
            ReportBuilder()
            .with_evolve_preset([], metric="sharpe")
            .build()
        )
        assert report.overview["size"] == 0
        assert report.overview["passed_pct"] == 0.0
        assert report.overview["best_metric"] == 0.0

    def test_custom_figure_factories(self, small_entries):
        custom_lineage = make_fake_figure("custom_lineage")
        report = (
            ReportBuilder()
            .with_evolve_preset(
                small_entries,
                metric="sharpe",
                figure_factories={"lineage_dag": lambda: custom_lineage},
            )
            .build()
        )
        # lineage_dag section uses custom factory
        lineage_sec = next(s for s in report.sections if s.title == "lineage_dag")
        assert lineage_sec.payload is custom_lineage


# ---------------------------------------------------------------------------
# build_to_html
# ---------------------------------------------------------------------------

class TestBuildToHtml:
    def test_returns_html_string(self, small_entries):
        html = (
            ReportBuilder()
            .with_title("Test")
            .with_evolve_preset(small_entries, metric="sharpe")
            .build_to_html()
        )
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "Test" in html
        assert "plotly" in html.lower()

    def test_writes_to_file(self, small_entries, tmp_path):
        out = tmp_path / "report.html"
        html = (
            ReportBuilder()
            .with_title("File Test")
            .with_evolve_preset(small_entries, metric="sharpe")
            .build_to_html(output_path=out)
        )
        assert out.exists()
        assert out.read_text(encoding="utf-8") == html
        content = out.read_text(encoding="utf-8")
        assert "File Test" in content

    def test_creates_parent_dir(self, small_entries, tmp_path):
        out = tmp_path / "subdir" / "report.html"
        (
            ReportBuilder()
            .with_evolve_preset(small_entries, metric="sharpe")
            .build_to_html(output_path=out)
        )
        assert out.exists()

    def test_custom_plotly_cdn(self, small_entries):
        custom = '<script src="https://my.cdn/plotly.js"></script>'
        html = (
            ReportBuilder()
            .with_evolve_preset(small_entries, metric="sharpe")
            .build_to_html(plotly_cdn=custom)
        )
        assert "my.cdn/plotly" in html

    def test_no_overview_omits_table(self):
        html = (
            ReportBuilder()
            .with_title("No Overview")
            .add_section("Only", make_fake_figure())
            .build_to_html()
        )
        assert "No Overview" in html
        assert "<h2>概览</h2>" not in html


# ---------------------------------------------------------------------------
# Backward compat: generate_report / generate_html
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_generate_report_returns_dict_with_overview_and_figures(self, small_entries):
        result = generate_report(small_entries, metric="sharpe", title="Compat Test")
        assert "overview" in result
        assert result["overview"]["size"] == 4
        # 5 figure keys
        for key in ["lineage_dag", "metric_distribution", "metric_per_round",
                    "gate_breakdown", "operation_breakdown"]:
            assert key in result

    def test_generate_html_returns_string(self, small_entries):
        html = generate_html(small_entries, metric="sharpe", title="Compat HTML")
        assert "<!DOCTYPE html>" in html
        assert "Compat HTML" in html

    def test_generate_html_writes_file(self, small_entries, tmp_path):
        out = tmp_path / "compat.html"
        generate_html(small_entries, metric="sharpe", title="X", output_path=out)
        assert out.exists()
        assert "X" in out.read_text(encoding="utf-8")

    def test_generate_html_accepts_mapping(self, small_entries):
        m = {f"e{i}": e for i, e in enumerate(small_entries)}
        html = generate_html(m, metric="sharpe", title="Map")
        assert "Map" in html
