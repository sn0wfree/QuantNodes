# coding=utf-8
"""Tests for core/visualization/{gate_breakdown,metric_distribution,lineage_dag}.py.

Covers: figure factory functions with graceful degradation (plotly missing),
build_lineage_layout BFS depth assignment, color/size heuristics.
"""

from pathlib import Path

import pytest

from QuantNodes.core.trajectory.entry import TrajectoryEntry
from QuantNodes.core.feedback.dataclass import FactorFeedback, FeedbackChannel
from QuantNodes.core.visualization.gate_breakdown import (
    gate_breakdown_figure,
    operation_breakdown_figure,
)
from QuantNodes.core.visualization.metric_distribution import (
    metric_distribution_figure,
    metric_per_round_figure,
)
from QuantNodes.core.visualization.lineage_dag import (
    build_lineage_layout,
    lineage_dag_figure,
    _OPERATION_COLORS,
)


# ============================================================================
# Fixtures
# ============================================================================

def make_entry(
    entry_id: str = "e1",
    operation: str = "original",
    round_idx: int = 0,
    decision: bool = True,
    parent_ids=None,
    metric_value=None,
) -> TrajectoryEntry:
    fb_channels = {
        FeedbackChannel.CODE: type("FB", (), {
            "passed": True, "score": 1.0,
        })(),
        FeedbackChannel.VALUE: type("FB", (), {
            "passed": decision, "score": 1.0 if decision else 0.0,
        })(),
    }
    feedback = FactorFeedback(
        factor_id=entry_id,
        factor_name=entry_id,
        decision=decision,
        summary="mock",
        channels=fb_channels,
    )
    metrics = {"sharpe": metric_value} if metric_value is not None else {}
    return TrajectoryEntry(
        entry_id=entry_id,
        operation=operation,
        round_idx=round_idx,
        parent_ids=parent_ids or [],
        feedback=feedback,
        metrics=metrics,
        config_snapshot={"factor": {"name": entry_id}},
    )


@pytest.fixture
def entries():
    return [
        make_entry("root", operation="original", round_idx=0, decision=True, metric_value=1.5),
        make_entry("c1", operation="mutation", round_idx=1, parent_ids=["root"], decision=True, metric_value=1.2),
        make_entry("c2", operation="crossover", round_idx=1, parent_ids=["root"], decision=False, metric_value=0.5),
    ]


# ============================================================================
# gate_breakdown_figure
# ============================================================================

class TestGateBreakdownFigure:
    def test_basic(self, entries):
        fig = gate_breakdown_figure(entries)
        # May be None (no plotly) or a Figure
        # If plotly is available, it should be a figure
        assert fig is None or hasattr(fig, "to_html")

    def test_empty_entries(self):
        fig = gate_breakdown_figure([])
        # Empty figure still created (or None for graceful degradation)
        assert fig is None or hasattr(fig, "to_html")

    def test_with_title(self, entries):
        fig = gate_breakdown_figure(entries, title="Test Title")
        assert fig is None or hasattr(fig, "to_html")


# ============================================================================
# operation_breakdown_figure
# ============================================================================

class TestOperationBreakdownFigure:
    def test_basic(self, entries):
        fig = operation_breakdown_figure(entries)
        assert fig is None or hasattr(fig, "to_html")

    def test_empty(self):
        fig = operation_breakdown_figure([])
        assert fig is None or hasattr(fig, "to_html")


# ============================================================================
# metric_distribution_figure
# ============================================================================

class TestMetricDistributionFigure:
    def test_basic(self, entries):
        fig = metric_distribution_figure(entries)
        assert fig is None or hasattr(fig, "to_html")

    def test_custom_metric(self, entries):
        fig = metric_distribution_figure(entries, metric="ir")
        assert fig is None or hasattr(fig, "to_html")

    def test_custom_bins(self, entries):
        fig = metric_distribution_figure(entries, n_bins=10)
        assert fig is None or hasattr(fig, "to_html")

    def test_empty(self):
        fig = metric_distribution_figure([])
        assert fig is None or hasattr(fig, "to_html")


# ============================================================================
# metric_per_round_figure
# ============================================================================

class TestMetricPerRoundFigure:
    def test_basic(self, entries):
        fig = metric_per_round_figure(entries)
        assert fig is None or hasattr(fig, "to_html")

    def test_custom_metric(self, entries):
        fig = metric_per_round_figure(entries, metric="ir")
        assert fig is None or hasattr(fig, "to_html")

    def test_empty(self):
        fig = metric_per_round_figure([])
        assert fig is None or hasattr(fig, "to_html")


# ============================================================================
# build_lineage_layout
# ============================================================================

class TestBuildLineageLayout:
    def test_basic(self, entries):
        layout = build_lineage_layout(entries)
        assert "nodes" in layout
        assert "edges" in layout

    def test_node_count_matches_entries(self, entries):
        layout = build_lineage_layout(entries)
        assert len(layout["nodes"]) == len(entries)

    def test_root_depth_zero(self, entries):
        layout = build_lineage_layout(entries)
        root = next(n for n in layout["nodes"] if n["id"] == "root")
        assert root["y"] == 0  # depth 0

    def test_child_depth_one(self, entries):
        layout = build_lineage_layout(entries)
        c1 = next(n for n in layout["nodes"] if n["id"] == "c1")
        assert c1["y"] == 1

    def test_node_fields(self, entries):
        layout = build_lineage_layout(entries)
        for node in layout["nodes"]:
            assert "id" in node
            assert "x" in node
            assert "y" in node
            assert "label" in node
            assert "color" in node
            assert "size" in node

    def test_with_dict_input(self):
        e_dict = {
            "root": make_entry("root", operation="original"),
            "c1": make_entry("c1", operation="mutation", parent_ids=["root"]),
        }
        layout = build_lineage_layout(e_dict)
        assert len(layout["nodes"]) == 2

    def test_node_size_scales_with_metric(self, entries):
        layout = build_lineage_layout(entries, metric="sharpe")
        root = next(n for n in layout["nodes"] if n["id"] == "root")
        c2 = next(n for n in layout["nodes"] if n["id"] == "c2")
        # root has higher metric_value (1.5), c2 has lower (0.5)
        assert root["size"] > c2["size"]

    def test_color_assigned_per_operation(self, entries):
        layout = build_lineage_layout(entries)
        # root = original (blue), c1 = mutation (orange), c2 = crossover (red)
        for node in layout["nodes"]:
            if node["id"] == "root":
                assert node["color"] == _OPERATION_COLORS["original"]
            elif node["id"] == "c1":
                assert node["color"] == _OPERATION_COLORS["mutation"]
            elif node["id"] == "c2":
                assert node["color"] == _OPERATION_COLORS["crossover"]

    def test_edges_between_parent_child(self, entries):
        layout = build_lineage_layout(entries)
        # root → c1 and root → c2
        edge_pairs = {(e["source"], e["target"]) for e in layout["edges"]}
        assert ("root", "c1") in edge_pairs
        assert ("root", "c2") in edge_pairs

    def test_decision_in_metadata(self, entries):
        layout = build_lineage_layout(entries)
        for node in layout["nodes"]:
            assert "decision" in node


# ============================================================================
# lineage_dag_figure
# ============================================================================

class TestLineageDagFigure:
    def test_basic(self, entries):
        fig = lineage_dag_figure(entries)
        assert fig is None or hasattr(fig, "to_html")

    def test_custom_metric(self, entries):
        fig = lineage_dag_figure(entries, metric="ir")
        assert fig is None or hasattr(fig, "to_html")

    def test_with_title(self, entries):
        fig = lineage_dag_figure(entries, title="My Lineage")
        assert fig is None or hasattr(fig, "to_html")

    def test_empty_entries(self):
        fig = lineage_dag_figure([])
        assert fig is None or hasattr(fig, "to_html")


# ============================================================================
# _OPERATION_COLORS
# ============================================================================

class TestOperationColors:
    def test_known_operations(self):
        # Keys are lowercase string of operation names
        assert "original" in _OPERATION_COLORS
        assert "mutation" in _OPERATION_COLORS
        assert "crossover" in _OPERATION_COLORS

    def test_values_are_hex_colors(self):
        for color in _OPERATION_COLORS.values():
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_entry(self):
        e = make_entry("solo", operation="original")
        layout = build_lineage_layout([e])
        assert len(layout["nodes"]) == 1
        assert len(layout["edges"]) == 0

    def test_no_metric_in_entry(self):
        """Entry without metric should still be plotted with default size."""
        e = make_entry("e1", metric_value=None)
        layout = build_lineage_layout([e])
        node = layout["nodes"][0]
        assert node["size"] > 0  # Some default size

    def test_multiple_parents(self):
        """Entry with multiple parents should create multiple edges."""
        root1 = make_entry("r1", operation="original")
        root2 = make_entry("r2", operation="original")
        child = make_entry("c", operation="crossover", parent_ids=["r1", "r2"])
        layout = build_lineage_layout([root1, root2, child])
        edge_pairs = {(e["source"], e["target"]) for e in layout["edges"]}
        assert ("r1", "c") in edge_pairs
        assert ("r2", "c") in edge_pairs

    def test_deep_lineage(self):
        """Multi-level deep lineage."""
        entries = [
            make_entry(f"e{i}", round_idx=i, parent_ids=[f"e{i-1}"] if i > 0 else [])
            for i in range(5)
        ]
        layout = build_lineage_layout(entries)
        # Max depth should be 4
        max_depth = max(n["y"] for n in layout["nodes"])
        assert max_depth == 4

    def test_metrics_zero_value(self):
        """Entry with metric value 0 should still be plotted."""
        e = make_entry("e1", metric_value=0.0)
        layout = build_lineage_layout([e], metric="sharpe")
        assert layout["nodes"][0]["size"] > 0

    def test_metrics_negative_value(self):
        """Negative metric value should still produce valid node."""
        e = make_entry("e1", metric_value=-0.5)
        layout = build_lineage_layout([e], metric="sharpe")
        assert layout["nodes"][0]["size"] > 0