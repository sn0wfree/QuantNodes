"""Monitoring 模块测试 (Week 13) — 8 tests。

覆盖:
    - 数据类 (2)
    - MetricCollector 增删 + 集成 (3)
    - Dashboard HTML (2)
    - CLI factor-dashboard (1)
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from QuantNodes.cli import cmd_factor_dashboard
from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
)
from QuantNodes.core.monitoring import (
    EvolutionMetrics,
    MetricCollector,
    QualityMetrics,
    RagMetrics,
    generate_dashboard_html,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# Fixtures
# ============================================================================

def _make_entry(
    eid: str, name: str, round_idx: int = 0, operation: str = "original",
    decision: bool = True, sharpe: float = 1.0, parent_ids: list | None = None,
    channels: dict | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid, round_idx=round_idx, operation=operation,
        parent_ids=parent_ids or [],
        config_snapshot={"factor": {"name": name, "expression": "x", "hypothesis": "h", "description": "d"}},
        feedback=FactorFeedback(
            factor_name=name, decision=decision,
            summary="ok" if decision else "fail",
            channels=channels or {},
        ),
        metrics={"sharpe": sharpe},
    )


@pytest.fixture
def pool_with_metrics() -> TrajectoryPool:
    """3 round × 多 entry 的 pool, 含 channels。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    # round 0
    pool.add(_make_entry("a", "A", round_idx=0, sharpe=1.0, channels={
        FeedbackChannel.CODE: ChannelFeedback(FeedbackChannel.CODE, True, "ok"),
        FeedbackChannel.VALUE: ChannelFeedback(FeedbackChannel.VALUE, True, "ok"),
    }))
    pool.add(_make_entry("b", "B", round_idx=0, decision=False, channels={
        FeedbackChannel.CODE: ChannelFeedback(FeedbackChannel.CODE, False, "fail"),
        FeedbackChannel.VALUE: ChannelFeedback(FeedbackChannel.VALUE, True, "ok"),
    }))
    # round 1
    pool.add(_make_entry("c", "C", round_idx=1, operation="mutation", parent_ids=["a"],
                         sharpe=1.5, channels={
                             FeedbackChannel.CODE: ChannelFeedback(FeedbackChannel.CODE, True, "ok"),
                             FeedbackChannel.VALUE: ChannelFeedback(FeedbackChannel.VALUE, False, "fail"),
                         }))
    return pool


# ============================================================================
# 1. 数据类 (2)
# ============================================================================

def test_rag_metrics_defaults():
    """RagMetrics 默认值 + 自动 timestamp。"""
    m = RagMetrics(round=1, n_queries=3)
    assert m.round == 1
    assert m.hit_at_5 == 0.0
    assert m.timestamp != ""


def test_evolution_metrics_defaults():
    """EvolutionMetrics 默认值。"""
    m = EvolutionMetrics(round=2)
    assert m.round == 2
    assert m.pool_size == 0
    assert m.best_factor_name == ""


# ============================================================================
# 2. MetricCollector (3)
# ============================================================================

def test_collector_add_methods():
    """add_rag / add_evolution / add_quality 基本操作。"""
    c = MetricCollector()
    c.add_rag(RagMetrics(round=1, n_queries=3))
    c.add_evolution(EvolutionMetrics(round=1, pool_size=3))
    c.add_quality(QualityMetrics(round=1, code_pass=3, code_fail=0))
    assert len(c.rag_history) == 1
    assert len(c.evolution_history) == 1
    assert len(c.quality_history) == 1
    assert len(c) == 3


def test_collector_update_quality_from_pool(pool_with_metrics):
    """update_quality_from_pool 统计 3 通道。"""
    c = MetricCollector()
    c.update_quality_from_pool(pool_with_metrics, round_idx=0)
    m = c.quality_history[0]
    assert m.round == 0
    assert m.code_pass == 1
    assert m.code_fail == 1
    assert m.value_pass == 2
    assert m.value_fail == 0


def test_collector_update_evolution_from_pool(pool_with_metrics):
    """update_evolution_from_pool 累积统计。"""
    c = MetricCollector()
    c.update_evolution_from_pool(pool_with_metrics, round_idx=1)
    m = c.evolution_history[0]
    assert m.round == 1
    # round 0 (2) + round 1 (1) = 3 pool size
    assert m.pool_size == 3
    # 2 passed (a, c), 1 rejected (b)
    assert m.total_count == 2
    assert m.rejected_count == 1
    # best sharpe in c (1.5)
    assert m.best_metric == 1.5
    assert m.best_factor_name == "C"


# ============================================================================
# 3. Dashboard HTML (2)
# ============================================================================

def test_dashboard_html_basic():
    """generate_dashboard_html 返回 HTML 字符串。"""
    c = MetricCollector()
    c.add_rag(RagMetrics(round=1, n_queries=3, hit_at_5=0.5))
    c.add_evolution(EvolutionMetrics(round=1, pool_size=2, best_metric=1.0))
    c.add_quality(QualityMetrics(round=1, code_pass=2))
    html = generate_dashboard_html(c, title="Test")
    assert isinstance(html, str)
    assert "<html" in html
    assert "Test" in html
    # 6 figures
    for k in ("rag_lines", "rag_scatter", "evo_bar", "evo_line", "qg_stacked", "qg_rejection"):
        assert f'id="fig_{k}"' in html


def test_dashboard_html_empty():
    """空 collector 也生成 dashboard (无数据 figure)。"""
    c = MetricCollector()
    html = generate_dashboard_html(c, title="Empty")
    assert "0 metrics" in html or "无数据" in html or len(html) > 1000


# ============================================================================
# 4. save/load + CLI (1)
# ============================================================================

def test_collector_save_load_json(tmp_path):
    """save → load 往返一致。"""
    c = MetricCollector()
    c.add_rag(RagMetrics(round=1, n_queries=3, hit_at_5=0.5))
    c.add_evolution(EvolutionMetrics(round=1, pool_size=3))
    c.add_quality(QualityMetrics(round=1, code_pass=2))
    out = tmp_path / "metrics.json"
    c.save(out)
    assert out.exists()
    c2 = MetricCollector.load(out)
    assert len(c2) == 3
    assert c2.rag_history[0].hit_at_5 == 0.5


def test_cli_factor_dashboard(pool_with_metrics, tmp_path):
    """CLI factor-dashboard 写 HTML + JSON。"""
    # pool_with_metrics fixture 用 tempfile.mkdtemp, 需要先复制到 tmp_path
    import shutil
    real_pool_dir = tmp_path / "pool"
    shutil.copytree(pool_with_metrics.base_dir, real_pool_dir)
    new_pool = TrajectoryPool(real_pool_dir)

    class Args:
        pool_dir = str(real_pool_dir)
        output = str(tmp_path / "dash.html")
        title = "Test Dashboard"
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_dashboard(Args())
    assert rc == 0
    out = buf.getvalue()
    assert "RAG" in out
    assert "Evo" in out
    assert "Quality" in out
    # 写文件
    assert (tmp_path / "dash.html").exists()
    assert (tmp_path / "dash_metrics.json").exists()
    html = (tmp_path / "dash.html").read_text()
    assert len(html) > 5000
