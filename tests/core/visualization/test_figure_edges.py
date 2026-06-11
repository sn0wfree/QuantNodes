"""figure 边界条件测试 (10 tests)。

聚焦:
    - gate_breakdown: 空/单/多/无 channels/无 feedback
    - operation_breakdown: 空/单/多/无 feedback
    - metric_distribution: 空/单/多 metric/无 metrics/常数值
    - metric_per_round: 空/单/多 round
    - 单 entry 不崩
"""
from __future__ import annotations

import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback, FactorFeedback, FeedbackChannel,
)
from QuantNodes.core.trajectory import TrajectoryEntry
from QuantNodes.core.visualization import (
    gate_breakdown_figure,
    metric_distribution_figure,
    metric_per_round_figure,
    operation_breakdown_figure,
)


def _entry(
    entry_id: str, round_idx: int = 0, op: str = "original",
    decision: bool = True, sharpe: float = 0.5,
    channels: dict | None = None, name: str = "f",
) -> TrajectoryEntry:
    fb = FactorFeedback(
        factor_id=entry_id, factor_name=name, decision=decision,
        summary="ok", channels=channels or {},
    ) if decision or channels else None
    return TrajectoryEntry(
        entry_id=entry_id, round_idx=round_idx, operation=op,
        feedback=fb, metrics={"sharpe": sharpe},
    )


# ============================================================================
# 1. gate_breakdown_figure (4 tests)
# ============================================================================

class TestGateBreakdown:
    def test_empty_returns_fig(self):
        fig = gate_breakdown_figure([])
        assert fig is not None
        assert "空" in fig.layout.title.text

    def test_single_entry_no_channels(self):
        """单 entry 无 channels → 0 pass/fail。"""
        e = _entry("e1")
        fig = gate_breakdown_figure([e])
        assert fig is not None

    def test_single_entry_with_channels(self):
        e = _entry("e1", channels={
            FeedbackChannel.CODE: ChannelFeedback(
                channel=FeedbackChannel.CODE, passed=True, detail="ok",
            ),
        })
        fig = gate_breakdown_figure([e])
        assert fig is not None

    def test_no_feedback_skipped(self):
        """feedback=None entry → 跳过。"""
        e = TrajectoryEntry(entry_id="e1", feedback=None)
        fig = gate_breakdown_figure([e])
        assert fig is not None

    def test_all_pass_no_fail(self):
        """全 pass → fail=0。"""
        entries = [
            _entry(f"e{i}", channels={
                FeedbackChannel.CODE: ChannelFeedback(
                    channel=FeedbackChannel.CODE, passed=True, detail="ok",
                ),
            })
            for i in range(5)
        ]
        fig = gate_breakdown_figure(entries)
        assert fig is not None

    def test_dict_input(self):
        """接受 dict[str, TrajectoryEntry]。"""
        e = _entry("e1")
        fig = gate_breakdown_figure({"e1": e})
        assert fig is not None


# ============================================================================
# 2. operation_breakdown_figure (3 tests)
# ============================================================================

class TestOperationBreakdown:
    def test_empty(self):
        fig = operation_breakdown_figure([])
        assert fig is not None

    def test_single_op(self):
        e = _entry("e1", op="original")
        fig = operation_breakdown_figure([e])
        assert fig is not None

    def test_multiple_ops(self):
        entries = [
            _entry("e1", op="original"),
            _entry("e2", op="mutation"),
            _entry("e3", op="crossover", decision=False),
        ]
        fig = operation_breakdown_figure(entries)
        assert fig is not None

    def test_no_feedback_counted_as_fail(self):
        """feedback=None entry → fail。"""
        e = TrajectoryEntry(entry_id="e1", feedback=None, operation="original")
        fig = operation_breakdown_figure([e])
        assert fig is not None


# ============================================================================
# 3. metric_distribution_figure (2 tests)
# ============================================================================

class TestMetricDistribution:
    def test_empty(self):
        fig = metric_distribution_figure([])
        assert fig is not None
        assert "无" in fig.layout.title.text

    def test_no_metrics(self):
        """所有 entry 无 metrics → 空 fig。"""
        e1 = TrajectoryEntry(entry_id="e1", metrics={})
        e2 = TrajectoryEntry(entry_id="e2", metrics={})
        fig = metric_distribution_figure([e1, e2], metric="sharpe")
        assert "无" in fig.layout.title.text

    def test_constant_values_single_bin(self):
        """所有 metric 相同 → 1 个 bin。"""
        entries = [_entry(f"e{i}", sharpe=1.0) for i in range(5)]
        fig = metric_distribution_figure(entries, metric="sharpe")
        assert fig is not None

    def test_multiple_operations(self):
        entries = [
            _entry("e1", op="original", sharpe=0.5),
            _entry("e2", op="mutation", sharpe=1.0),
            _entry("e3", op="crossover", sharpe=0.8),
        ]
        fig = metric_distribution_figure(entries, metric="sharpe")
        assert fig is not None


# ============================================================================
# 4. metric_per_round_figure (2 tests)
# ============================================================================

class TestMetricPerRound:
    def test_empty(self):
        fig = metric_per_round_figure([])
        assert fig is not None

    def test_single_round(self):
        entries = [_entry(f"e{i}", round_idx=0, sharpe=0.5 + i*0.1) for i in range(3)]
        fig = metric_per_round_figure(entries, metric="sharpe")
        assert fig is not None

    def test_multiple_rounds(self):
        entries = []
        for r in range(3):
            for i in range(3):
                entries.append(_entry(f"e{r}_{i}", round_idx=r, sharpe=r + i*0.1))
        fig = metric_per_round_figure(entries, metric="sharpe")
        assert fig is not None

    def test_no_metrics(self):
        """无 metrics → 不崩, fig 仍生成 (但无数据线)。"""
        e = TrajectoryEntry(entry_id="e1", metrics={})
        fig = metric_per_round_figure([e])
        # 不会崩, fig 仍生成
        assert fig is not None
        # 但不应有数据 trace (或为空)
        assert len(fig.data) == 0 or all(len(t.x) == 0 for t in fig.data)
