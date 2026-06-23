"""Visualization 模块测试 (15 tests)。

覆盖:
    - build_lineage_layout (4)
    - 4 个 Figure (4)
    - generate_report (2)
    - generate_html (5)
"""
from __future__ import annotations


import pytest

# v3.0.0 graceful degradation: all figure-based tests require plotly
# to be installed. When plotly is missing, the QuantNodes visualization
# module returns None for figure functions (see dashboard.py etc.),
# so these tests are not meaningful and are skipped.
pytest.importorskip("plotly", reason="plotly not installed; figure tests skipped")

from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool
from QuantNodes.core.visualization import (
    build_lineage_layout,
    gate_breakdown_figure,
    generate_html,
    generate_report,
    lineage_dag_figure,
    metric_distribution_figure,
    metric_per_round_figure,
)


# ============================================================================
# Fixtures
# ============================================================================

def _make_entry(
    eid: str,
    round_idx: int = 0,
    operation: str = "original",
    parent_ids: list[str] | None = None,
    sharpe: float = 1.0,
    decision: bool = True,
    factor_name: str | None = None,
    channels: dict | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid,
        round_idx=round_idx,
        operation=operation,
        parent_ids=parent_ids or [],
        feedback=FactorFeedback(
            factor_name=factor_name or eid,
            decision=decision,
            summary="ok" if decision else "rejected",
            channels=channels or {},
        ),
        metrics={"sharpe": sharpe, "arr": sharpe * 0.1, "ic_mean": 0.04},
    )


@pytest.fixture
def small_pool() -> list[TrajectoryEntry]:
    return [
        _make_entry("a", 0, "original", sharpe=1.0, factor_name="A"),
        _make_entry("b", 0, "original", sharpe=0.5, factor_name="B"),
        _make_entry("c", 1, "mutation", parent_ids=["a"], sharpe=1.5, factor_name="C"),
        _make_entry("d", 2, "crossover", parent_ids=["a", "c"], sharpe=2.0, factor_name="D"),
    ]


# ============================================================================
# 1. build_lineage_layout (4)
# ============================================================================

def test_lineage_layout_empty():
    """空输入返回空 nodes/edges。"""
    layout = build_lineage_layout([])
    assert layout == {"nodes": [], "edges": []}


def test_lineage_layout_bfs_depth():
    """BFS 深度正确 (original=0, mutation=1, crossover=2)。"""
    layout = build_lineage_layout([
        _make_entry("a", 0, "original"),
        _make_entry("c", 1, "mutation", parent_ids=["a"]),
        _make_entry("d", 2, "crossover", parent_ids=["a", "c"]),
    ])
    depths = {n["id"]: n["y"] for n in layout["nodes"]}
    assert depths["a"] == 0
    assert depths["c"] == 1
    assert depths["d"] == 2


def test_lineage_layout_edges_correct():
    """边 source/target 正确。"""
    layout = build_lineage_layout([
        _make_entry("a", 0, "original"),
        _make_entry("c", 1, "mutation", parent_ids=["a"]),
    ])
    edges = layout["edges"]
    assert len(edges) == 1
    assert edges[0]["source"] == "a"
    assert edges[0]["target"] == "c"


def test_lineage_layout_node_attributes():
    """节点含 color/size/label 等属性。"""
    layout = build_lineage_layout([_make_entry("a", 0, "original", sharpe=2.0)])
    node = layout["nodes"][0]
    assert node["operation"] == "original"
    assert node["round_idx"] == 0
    assert node["color"]  # 非空颜色
    assert node["size"] >= 10  # 最小尺寸
    assert "sharpe=2.00" in node["label"]


# ============================================================================
# 2. Figure functions (4)
# ============================================================================

def test_lineage_dag_figure_returns_figure(small_pool):
    """lineage_dag_figure 返回 plotly Figure。"""
    fig = lineage_dag_figure(small_pool)
    assert fig is not None
    assert "演化谱系" in fig.layout.title.text
    # 应有边 + 节点 + 图例 traces
    assert len(fig.data) >= 2  # edges + nodes + 3 legend


def test_metric_distribution_figure(small_pool):
    """metric_distribution_figure 按 operation 分桶。"""
    fig = metric_distribution_figure(small_pool, metric="sharpe")
    # 3 个 operation = 3 个 histogram
    histogram_traces = [t for t in fig.data if t.type == "histogram"]
    assert len(histogram_traces) == 3  # original, mutation, crossover
    titles = {t.name for t in histogram_traces}
    assert titles == {"original", "mutation", "crossover"}


def test_metric_per_round_figure(small_pool):
    """metric_per_round_figure 含 best + mean 两条线。"""
    fig = metric_per_round_figure(small_pool)
    scatter_traces = [t for t in fig.data if t.type == "scatter"]
    assert len(scatter_traces) == 2  # best + mean
    names = {t.name for t in scatter_traces}
    assert "best sharpe" in names
    assert "mean sharpe" in names


def test_gate_breakdown_figure():
    """gate_breakdown_figure 含 pass + fail 2 个 bar。"""
    entries = [
        _make_entry("a", 0, "original", decision=True, channels={
            FeedbackChannel.CODE: ChannelFeedback(FeedbackChannel.CODE, True, "ok"),
        }),
        _make_entry("b", 0, "original", decision=False, channels={
            FeedbackChannel.CODE: ChannelFeedback(FeedbackChannel.CODE, False, "fail"),
        }),
    ]
    fig = gate_breakdown_figure(entries)
    bar_traces = [t for t in fig.data if t.type == "bar"]
    assert len(bar_traces) == 2  # pass + fail
    assert "pass" in {t.name for t in bar_traces}


# ============================================================================
# 3. generate_report (2)
# ============================================================================

def test_generate_report_contains_all_figures(small_pool):
    """generate_report 返回 6 个 key (overview + 5 figures)。"""
    report = generate_report(small_pool, metric="sharpe")
    assert set(report.keys()) == {
        "overview", "lineage_dag", "metric_distribution",
        "metric_per_round", "gate_breakdown", "operation_breakdown",
    }
    assert report["overview"]["size"] == 4
    assert report["overview"]["passed"] == 4
    assert report["overview"]["best_metric"] == 2.0


def test_generate_report_empty_pool():
    """空 pool 也返回 report (含空 figures)。"""
    report = generate_report([], metric="sharpe")
    assert report["overview"]["size"] == 0
    assert report["overview"]["passed"] == 0
    assert report["overview"]["best_metric"] == 0.0


# ============================================================================
# 4. generate_html (5)
# ============================================================================

def test_generate_html_returns_string(small_pool):
    """generate_html 返回 HTML 字符串。"""
    html = generate_html(small_pool, metric="sharpe")
    assert isinstance(html, str)
    assert "<html" in html
    assert "演化" in html or "演化" in html


def test_generate_html_contains_all_figures(small_pool):
    """HTML 含 5 个 figure div。"""
    html = generate_html(small_pool, metric="sharpe")
    for fig_key in ("lineage_dag", "metric_distribution", "metric_per_round",
                    "gate_breakdown", "operation_breakdown"):
        assert f"id=\"fig_{fig_key}\"" in html, f"missing {fig_key}"


def test_generate_html_contains_plotly_cdn(small_pool):
    """HTML 含 plotly CDN script。"""
    html = generate_html(small_pool, metric="sharpe")
    assert "plotly" in html.lower()
    assert "cdn" in html.lower()


def test_generate_html_writes_file(small_pool, tmp_path):
    """output_path 写文件。"""
    out = tmp_path / "report.html"
    html = generate_html(small_pool, metric="sharpe", output_path=out)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html
    assert out.stat().st_size > 1000  # 至少 1KB


def test_generate_html_contains_overview_table(small_pool):
    """HTML 含概览表 (size / passed / best metric)。"""
    html = generate_html(small_pool, metric="sharpe")
    assert "总 entry 数" in html
    assert "通过数" in html
    assert "Best sharpe" in html


# ============================================================================
# 5. CLI 集成 (1)
# ============================================================================

def test_cli_visual_subcommand(tmp_path, small_pool):
    """CLI factor-visual 生成 HTML 文件。"""
    # 先把 entries 写入 pool
    pool = TrajectoryPool(tmp_path / "pool")
    for e in small_pool:
        pool.add(e)

    # 通过 generate_html 直接验证 (CLI 子命令验证在 test_cli_evolution)
    out = tmp_path / "report.html"
    generate_html(pool, metric="sharpe", output_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000
