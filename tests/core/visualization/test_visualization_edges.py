"""visualization + monitoring/dashboard 边界条件测试 (15 tests)。

聚焦:
    - build_lineage_layout: 空 entries、单 entry、parent 不在 entries
    - generate_report: 5 个 figure + overview
    - generate_html: 含 CDN、概览表、output 写入
    - generate_dashboard_html: streaming=True 加 LIVE badge + JS, streaming=False 无
    - dashboard 6 个 figure 函数单独不崩
"""
from __future__ import annotations

from pathlib import Path


from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.monitoring import (
    EvolutionMetrics,
    MetricCollector,
    QualityMetrics,
    RagMetrics,
    generate_dashboard_html,
)
from QuantNodes.core.trajectory import TrajectoryEntry
from QuantNodes.core.visualization import (
    build_lineage_layout,
    generate_html,
    generate_report,
    gate_breakdown_figure,
    lineage_dag_figure,
    metric_distribution_figure,
    metric_per_round_figure,
    operation_breakdown_figure,
)


def _entry(
    entry_id: str, round_idx: int = 0,
    parent_ids: list[str] | None = None,
    op: str = "original", sharpe: float = 0.5,
    decision: bool = True, name: str = "f",
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id, round_idx=round_idx,
        operation=op, parent_ids=parent_ids or [],
        feedback=FactorFeedback(
            factor_id=entry_id, factor_name=name,
            decision=decision, summary=f"sharpe={sharpe}",
        ),
        metrics={"sharpe": sharpe},
    )


# ============================================================================
# 1. build_lineage_layout (4 tests)
# ============================================================================

class TestLineageLayout:
    def test_empty_entries(self):
        layout = build_lineage_layout([])
        assert layout == {"nodes": [], "edges": []}

    def test_single_entry_no_parent(self):
        e = _entry("e1", round_idx=0)
        layout = build_lineage_layout([e])
        assert len(layout["nodes"]) == 1
        assert layout["nodes"][0]["id"] == "e1"
        assert layout["nodes"][0]["y"] == 0
        assert layout["edges"] == []

    def test_chain_depth(self):
        e1 = _entry("e1", round_idx=0)
        e2 = _entry("e2", round_idx=1, parent_ids=["e1"], op="mutation")
        e3 = _entry("e3", round_idx=2, parent_ids=["e2"], op="crossover")
        layout = build_lineage_layout([e1, e2, e3])
        nodes = {n["id"]: n for n in layout["nodes"]}
        assert nodes["e1"]["y"] == 0
        assert nodes["e2"]["y"] == 1
        assert nodes["e3"]["y"] == 2
        # edges
        assert len(layout["edges"]) == 2
        edge_pairs = {(e["source"], e["target"]) for e in layout["edges"]}
        assert ("e1", "e2") in edge_pairs
        assert ("e2", "e3") in edge_pairs

    def test_parent_not_in_entries(self):
        """parent 不在 entries 中 → 不建边, depth 仍可算。"""
        e1 = _entry("e1", round_idx=1, parent_ids=["missing"])
        layout = build_lineage_layout([e1])
        # parent_ids 不全在 depths → max_parent_depth = -1 → depth 0
        assert layout["nodes"][0]["y"] == 0
        # missing parent 不创建边
        assert layout["edges"] == []

    def test_node_size_positive(self):
        """节点大小必为正数。"""
        e1 = _entry("e1", round_idx=0, sharpe=2.0)
        e2 = _entry("e2", round_idx=0, sharpe=-1.0)  # 负值
        layout = build_lineage_layout([e1, e2])
        for n in layout["nodes"]:
            assert n["size"] >= 10  # min 10


# ============================================================================
# 2. 5 个 figure 函数 (5 tests)
# ============================================================================

class TestFigureFunctions:
    def test_lineage_dag_figure_empty(self):
        fig = lineage_dag_figure([])
        assert fig is not None
        # title 应包含 "空"
        title = fig.layout.title.text
        assert "空" in title or "empty" in title.lower()

    def test_lineage_dag_figure_with_entries(self):
        e1 = _entry("e1", round_idx=0)
        e2 = _entry("e2", round_idx=1, parent_ids=["e1"], op="mutation")
        fig = lineage_dag_figure([e1, e2])
        assert fig is not None

    def test_metric_distribution_figure(self):
        e1 = _entry("e1", round_idx=0, sharpe=1.0)
        e2 = _entry("e2", round_idx=1, sharpe=2.0)
        fig = metric_distribution_figure([e1, e2], metric="sharpe")
        assert fig is not None

    def test_metric_per_round_figure(self):
        e1 = _entry("e1", round_idx=0, sharpe=1.0)
        e2 = _entry("e2", round_idx=1, sharpe=2.0)
        fig = metric_per_round_figure([e1, e2], metric="sharpe")
        assert fig is not None

    def test_gate_breakdown_figure(self):
        e1 = _entry("e1", round_idx=0, decision=True)
        e2 = _entry("e2", round_idx=0, decision=False)
        fig = gate_breakdown_figure([e1, e2])
        assert fig is not None

    def test_operation_breakdown_figure(self):
        e1 = _entry("e1", round_idx=0, op="original")
        e2 = _entry("e2", round_idx=1, op="mutation")
        fig = operation_breakdown_figure([e1, e2])
        assert fig is not None


# ============================================================================
# 3. generate_report / generate_html (3 tests)
# ============================================================================

class TestGenerateReport:
    def test_generate_report_5_figures(self):
        e1 = _entry("e1", round_idx=0, sharpe=1.0)
        report = generate_report([e1], metric="sharpe")
        assert "overview" in report
        assert "lineage_dag" in report
        assert "metric_distribution" in report
        assert "metric_per_round" in report
        assert "gate_breakdown" in report
        assert "operation_breakdown" in report
        # overview 数据
        assert report["overview"]["size"] == 1
        assert report["overview"]["passed"] == 1

    def test_generate_html_contains_cdn(self):
        e1 = _entry("e1", round_idx=0, sharpe=1.0)
        html = generate_html([e1], metric="sharpe")
        # CDN
        assert "cdn.plot.ly" in html
        # overview table
        assert "总 entry 数" in html
        assert "Best sharpe" in html
        # 5 figures (5 个 h2)
        assert html.count("<h2>") >= 5

    def test_generate_html_writes_file(self, tmp_path: Path):
        e1 = _entry("e1", round_idx=0, sharpe=1.0)
        output = tmp_path / "report.html"
        html = generate_html([e1], metric="sharpe", output_path=output)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "QuantNodes" in content


# ============================================================================
# 4. generate_dashboard_html (3 tests)
# ============================================================================

class TestDashboardHtml:
    def test_no_streaming_no_live(self):
        c = MetricCollector()
        c.add_evolution(EvolutionMetrics(round=0, pool_size=5))
        c.add_quality(QualityMetrics(round=0, code_pass=3))
        c.add_rag(RagMetrics(round=0, n_queries=2, hit_at_5=0.8))
        html = generate_dashboard_html(c, streaming=False)
        # 不含 LIVE badge
        assert "LIVE" not in html
        # 不含 setInterval
        assert "setInterval" not in html

    def test_streaming_includes_live_and_js(self):
        c = MetricCollector()
        c.add_evolution(EvolutionMetrics(round=0, pool_size=5))
        html = generate_dashboard_html(c, streaming=True, refresh_interval_sec=5)
        # LIVE badge
        assert "LIVE" in html
        # JS 轮询
        assert "setInterval" in html
        # 5 秒刷新
        assert "5 * 1000" in html or "5000" in html
        # metricsPath
        assert "metrics.json" in html
        # 5s 注释
        assert "5s" in html or "5 秒" in html

    def test_streaming_writes_file(self, tmp_path: Path):
        c = MetricCollector()
        c.add_evolution(EvolutionMetrics(round=0, pool_size=5))
        output = tmp_path / "dash.html"
        html = generate_dashboard_html(c, output_path=output, streaming=True)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "LIVE" in content

    def test_empty_collector_doesnt_crash(self):
        c = MetricCollector()
        html = generate_dashboard_html(c)
        assert "n_rag" in html or "概览" in html
