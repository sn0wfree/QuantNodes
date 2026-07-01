# coding=utf-8
"""Tests for core/visualization/builder.py — ReportBuilder (Phase 1.3 fluent API).

Covers: Section, Report, ReportBuilder chained API, build, build_to_html,
with_evolve_preset, _render_html.
"""

from pathlib import Path

import pytest

from QuantNodes.core.visualization.builder import (
    Section,
    Report,
    ReportBuilder,
    _render_html,
    _OVERVIEW_TEMPLATE,
)
from QuantNodes.core.trajectory.entry import TrajectoryEntry
from QuantNodes.core.feedback.dataclass import FactorFeedback


# ============================================================================
# Mock payload (has to_html method)
# ============================================================================

class MockPayload:
    """Simple payload that mimics plotly Figure's to_html()."""
    def __init__(self, html_content: str = "<div>mock</div>"):
        self.html_content = html_content
        self.to_html_called_with = None

    def to_html(self, full_html=None, include_plotlyjs=None, div_id=None):
        self.to_html_called_with = {
            "full_html": full_html,
            "include_plotlyjs": include_plotlyjs,
            "div_id": div_id,
        }
        return self.html_content


# ============================================================================
# Section
# ============================================================================

class TestSection:
    def test_creation(self):
        s = Section(title="My Section", payload=MockPayload())
        assert s.title == "My Section"
        assert s.payload is not None

    def test_render_payload_none(self):
        """None payload renders graceful degradation hint."""
        s = Section(title="Chart", payload=None)
        result = s.render()
        assert "plotly not installed" in result

    def test_render_with_to_html_payload(self):
        payload = MockPayload("<div>chart</div>")
        s = Section(title="My Chart", payload=payload)
        result = s.render()
        assert result == "<div>chart</div>"
        assert payload.to_html_called_with["full_html"] is False

    def test_render_includes_plotlyjs(self):
        payload = MockPayload("<div>chart</div>")
        s = Section(title="My Chart", payload=payload)
        s.render(include_plotlyjs=True)
        assert payload.to_html_called_with["include_plotlyjs"] is True

    def test_render_uses_default_div_id(self):
        payload = MockPayload("<div>chart</div>")
        s = Section(title="My Chart", payload=payload)
        s.render()
        # Default div_id derived from title
        assert payload.to_html_called_with["div_id"] == "fig_my_chart"

    def test_render_uses_custom_div_id(self):
        payload = MockPayload("<div>chart</div>")
        s = Section(title="My Chart", payload=payload)
        s.render(div_id="custom_div")
        assert payload.to_html_called_with["div_id"] == "custom_div"

    def test_render_payload_without_to_html(self):
        """Object without to_html renders as repr."""
        s = Section(title="My Section", payload=42)
        result = s.render()
        assert "<pre>" in result
        assert "42" in result


# ============================================================================
# Report
# ============================================================================

class TestReport:
    def test_creation(self):
        r = Report(title="My Report")
        assert r.title == "My Report"
        assert r.overview == {}
        assert r.sections == []

    def test_to_dict_basic(self):
        r = Report(title="Report 1", overview={"size": 5})
        d = r.to_dict()
        assert d["overview"] == {"size": 5}

    def test_to_dict_with_sections(self):
        payload = MockPayload()
        r = Report(
            title="Report",
            overview={"size": 1},
            sections=[Section(title="Lineage DAG", payload=payload)],
        )
        d = r.to_dict()
        # Section titles become keys with underscores
        assert "lineage_dag" in d
        assert d["lineage_dag"] is payload

    def test_to_dict_section_keys_lowercase_underscore(self):
        r = Report(
            title="Report",
            sections=[Section(title="My Section", payload=MockPayload())],
        )
        d = r.to_dict()
        assert "my_section" in d

    def test_to_dict_multiple_sections(self):
        r = Report(
            title="Report",
            sections=[
                Section(title="A", payload=MockPayload()),
                Section(title="B", payload=MockPayload()),
                Section(title="C", payload=MockPayload()),
            ],
        )
        d = r.to_dict()
        assert "a" in d
        assert "b" in d
        assert "c" in d


# ============================================================================
# ReportBuilder Creation
# ============================================================================

class TestReportBuilderCreation:
    def test_creation(self):
        b = ReportBuilder()
        assert b._title == "Report"
        assert b._overview == {}
        assert b._sections == []

    def test_preset_not_loaded_by_default(self):
        b = ReportBuilder()
        assert b._preset_loaded is False


# ============================================================================
# Fluent API
# ============================================================================

class TestReportBuilderFluentAPI:
    def test_with_title(self):
        b = ReportBuilder()
        result = b.with_title("My Report")
        assert result is b
        assert b._title == "My Report"

    def test_with_overview(self):
        b = ReportBuilder()
        b.with_overview({"size": 10, "passed": 8})
        assert b._overview["size"] == 10
        assert b._overview["passed"] == 8

    def test_with_overview_copies_dict(self):
        b = ReportBuilder()
        overview = {"size": 10}
        b.with_overview(overview)
        overview["size"] = 20
        # Should be isolated
        assert b._overview["size"] == 10

    def test_add_section(self):
        b = ReportBuilder()
        payload = MockPayload()
        result = b.add_section("Chart 1", payload)
        assert result is b
        assert len(b._sections) == 1

    def test_add_multiple_sections(self):
        b = ReportBuilder()
        b.add_section("A", MockPayload())
        b.add_section("B", MockPayload())
        b.add_section("C", MockPayload())
        assert len(b._sections) == 3

    def test_chained_api(self):
        report = (ReportBuilder()
                  .with_title("Chained Report")
                  .with_overview({"size": 1})
                  .add_section("A", MockPayload())
                  .build())
        assert report.title == "Chained Report"
        assert len(report.sections) == 1


# ============================================================================
# build()
# ============================================================================

class TestReportBuilderBuild:
    def test_build_basic(self):
        b = ReportBuilder().with_title("Test").with_overview({"x": 1})
        r = b.build()
        assert isinstance(r, Report)
        assert r.title == "Test"
        assert r.overview == {"x": 1}

    def test_build_with_sections(self):
        b = (ReportBuilder()
             .with_title("Test")
             .add_section("A", MockPayload())
             .add_section("B", MockPayload()))
        r = b.build()
        assert len(r.sections) == 2

    def test_build_returns_new_list(self):
        """sections in Report should be a copy of internal list."""
        b = ReportBuilder().add_section("A", MockPayload())
        r = b.build()
        b.add_section("B", MockPayload())
        # Report r should not see B
        assert len(r.sections) == 1


# ============================================================================
# build_to_html()
# ============================================================================

class TestBuildToHtml:
    def test_build_to_html_returns_string(self):
        b = ReportBuilder().with_title("Test")
        html = b.build_to_html()
        assert isinstance(html, str)
        assert "<html" in html
        assert "Test" in html

    def test_build_to_html_writes_file(self, tmp_path):
        b = ReportBuilder().with_title("Test")
        path = tmp_path / "report.html"
        b.build_to_html(output_path=path)
        assert path.exists()
        assert "<html" in path.read_text(encoding="utf-8")

    def test_build_to_html_creates_parent_dir(self, tmp_path):
        b = ReportBuilder().with_title("Test")
        path = tmp_path / "subdir" / "report.html"
        b.build_to_html(output_path=path)
        assert path.exists()

    def test_build_to_html_with_overview(self):
        b = ReportBuilder().with_title("Test").with_overview({
            "size": 5, "rounds": 1, "passed": 3, "passed_pct": 0.6,
            "rejected": 2, "best_metric": 0.5, "metric": "sharpe",
        })
        html = b.build_to_html()
        assert "概览" in html
        assert "5" in html

    def test_build_to_html_with_sections(self):
        b = (ReportBuilder()
             .with_title("Test")
             .add_section("My Chart", MockPayload("<div>chart</div>")))
        html = b.build_to_html()
        assert "<div>chart</div>" in html
        assert "My Chart" in html

    def test_build_to_html_with_custom_plotly_cdn(self):
        b = ReportBuilder()
        cdn = '<script src="https://custom-cdn.com/plotly.js"></script>'
        html = b.build_to_html(plotly_cdn=cdn)
        assert cdn in html


# ============================================================================
# _render_html internal
# ============================================================================

class TestRenderHtml:
    def test_renders_title(self):
        report = Report(title="My Title")
        html = _render_html(report, plotly_cdn="")
        assert "My Title" in html
        assert "<h1>My Title</h1>" in html

    def test_renders_sections(self):
        payload = MockPayload("<div>section_content</div>")
        report = Report(title="T", sections=[Section(title="S1", payload=payload)])
        html = _render_html(report, plotly_cdn="")
        assert "<h2>S1</h2>" in html
        assert "<div>section_content</div>" in html

    def test_renders_overview(self):
        report = Report(
            title="T",
            overview={"size": 5, "rounds": 2, "passed": 3, "passed_pct": 0.6,
                       "rejected": 2, "best_metric": 0.95, "metric": "sharpe"},
        )
        html = _render_html(report, plotly_cdn="")
        assert "概览" in html

    def test_overview_template_format(self):
        # Test that template can format with all expected keys
        result = _OVERVIEW_TEMPLATE.format(
            size=10, rounds=3, passed=8, passed_pct=0.8,
            rejected=2, best_metric=0.5, metric="sharpe",
        )
        assert "10" in result
        assert "8" in result
        assert "0.5000" in result


# ============================================================================
# with_evolve_preset()
# ============================================================================

class TestWithEvolvePreset:
    def test_preset_empty_entries(self):
        b = ReportBuilder().with_evolve_preset(entries=[])
        assert b._overview["size"] == 0
        assert b._overview["passed"] == 0

    def test_preset_with_entries(self):
        entries = [
            TrajectoryEntry(
                entry_id="e-1",
                feedback=FactorFeedback(factor_id="f-1", factor_name="f1", decision=True),
                metrics={"sharpe": 1.5},
                round_idx=0,
            ),
            TrajectoryEntry(
                entry_id="e-2",
                feedback=FactorFeedback(factor_id="f-2", factor_name="f2", decision=False),
                metrics={"sharpe": 0.5},
                round_idx=1,
            ),
        ]
        b = ReportBuilder().with_evolve_preset(entries=entries)
        assert b._overview["size"] == 2
        assert b._overview["passed"] == 1
        assert b._overview["rejected"] == 1
        assert b._overview["best_metric"] == 1.5
        assert b._preset_loaded is True

    def test_preset_rounds_counted(self):
        entries = [
            TrajectoryEntry(entry_id=f"e-{i}", round_idx=i % 2)
            for i in range(5)
        ]
        b = ReportBuilder().with_evolve_preset(entries=entries)
        assert b._overview["rounds"] == 2

    def test_preset_with_custom_metric(self):
        entries = [
            TrajectoryEntry(
                entry_id="e-1",
                feedback=FactorFeedback(factor_id="f-1", decision=True),
                metrics={"ic": 0.08},
            ),
        ]
        b = ReportBuilder().with_evolve_preset(entries=entries, metric="ic")
        assert b._overview["metric"] == "ic"
        assert b._overview["best_metric"] == 0.08

    def test_preset_accepts_mapping(self):
        """Mapping input (like dict of entry_id -> entry)."""
        e1 = TrajectoryEntry(entry_id="e-1", feedback=FactorFeedback(decision=True))
        e2 = TrajectoryEntry(entry_id="e-2", feedback=FactorFeedback(decision=False))
        mapping = {"e-1": e1, "e-2": e2}
        b = ReportBuilder().with_evolve_preset(entries=mapping)
        assert b._overview["size"] == 2

    def test_preset_with_figure_factories(self):
        """Custom figure factories override defaults."""
        e = TrajectoryEntry(
            entry_id="e-1",
            feedback=FactorFeedback(decision=True),
            metrics={"sharpe": 1.0},
        )
        custom_payload = MockPayload("<div>custom</div>")
        b = ReportBuilder().with_evolve_preset(
            entries=[e],
            figure_factories={"lineage_dag": lambda: custom_payload},
        )
        assert b._preset_loaded is True
        # First section should be the custom one
        assert b._sections[0].payload is custom_payload

    def test_preset_skipped_metrics_handled(self):
        """Entries without the requested metric are skipped."""
        entries = [
            TrajectoryEntry(entry_id="e-1", metrics={"ic": 0.05}),
            TrajectoryEntry(entry_id="e-2", metrics={"other": 0.1}),  # no ic
        ]
        b = ReportBuilder().with_evolve_preset(entries=entries, metric="ic")
        # Only one entry has ic
        assert b._overview["best_metric"] == 0.05


# ============================================================================
# Edge Cases
# ============================================================================

class TestReportBuilderEdgeCases:
    def test_empty_report_html(self):
        b = ReportBuilder()
        html = b.build_to_html()
        assert "<html" in html
        assert "Report" in html  # default title

    def test_section_with_empty_title(self):
        s = Section(title="", payload=MockPayload())
        html = s.render()
        assert html is not None

    def test_report_with_no_overview(self):
        r = Report(title="T")
        d = r.to_dict()
        assert d == {"overview": {}}

    def test_overview_isolation(self):
        b = ReportBuilder()
        original = {"x": 1}
        b.with_overview(original)
        original["x"] = 99
        assert b._overview["x"] == 1