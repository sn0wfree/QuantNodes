"""Quality Gate 拦截率柱状图 — Plotly bar chart."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

# v3.0.0 graceful degradation: if plotly is not installed, the figure
# functions return None. The HTML generators downstream skip None
# figures with a friendly install hint.
try:
    import plotly.graph_objects as _PLOTLY_GO
except ImportError:
    _PLOTLY_GO = None

    import warnings
    warnings.warn(
        "plotly not installed; visualization figures will be skipped. "
        "Install plotly for full rendering: pip install plotly",
        ImportWarning,
        stacklevel=1,
    )



def gate_breakdown_figure(
    entries,
    title: str | None = None,
) -> Any:
    """统计每种 quality gate 通道的拦截率, 生成柱状图。

    拦截定义: feedback.decision = False (任一通道失败)
    按 channel 细分: CODE / VALUE / LLM (来自 FactorFeedback.channels)

    Args:
        entries: TrajectoryEntry 列表
        title: 标题

    Returns:
        plotly.graph_objects.Figure, or None if plotly is not installed.
    """
    if _PLOTLY_GO is None:
        return None
    go = _PLOTLY_GO

    items = list(entries.values() if isinstance(entries, Mapping) else entries)
    if not items:
        fig = go.Figure()
        fig.update_layout(title="空 pool")
        return fig

    # 统计每通道 pass / fail
    from ..feedback import FeedbackChannel
    channel_stats: dict[str, dict[str, int]] = {
        ch.value: {"pass": 0, "fail": 0} for ch in FeedbackChannel
    }
    for e in items:
        if not e.feedback or not e.feedback.channels:
            continue
        for ch, fb in e.feedback.channels.items():
            key = ch.value
            if key not in channel_stats:
                continue
            if fb.passed:
                channel_stats[key]["pass"] += 1
            else:
                channel_stats[key]["fail"] += 1

    channels = sorted(channel_stats.keys())
    pass_counts = [channel_stats[ch]["pass"] for ch in channels]
    fail_counts = [channel_stats[ch]["fail"] for ch in channels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=channels, y=pass_counts,
        name="pass", marker_color="#54A24B",
        hovertemplate="%{x}<br>pass=%{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=channels, y=fail_counts,
        name="fail", marker_color="#E45756",
        hovertemplate="%{x}<br>fail=%{y}<extra></extra>",
    ))

    fig.update_layout(
        title=title or "Quality Gate 拦截率 (按通道)",
        barmode="stack",
        xaxis_title="channel",
        yaxis_title="count",
        height=400,
        plot_bgcolor="white",
    )
    return fig


def operation_breakdown_figure(
    entries,
    title: str | None = None,
) -> Any:
    """按 operation 分类的成功率柱状图。"""
    if _PLOTLY_GO is None:
        return None
    go = _PLOTLY_GO

    items = list(entries.values() if isinstance(entries, Mapping) else entries)
    if not items:
        fig = go.Figure()
        fig.update_layout(title="空 pool")
        return fig

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0})
    for e in items:
        decision = bool(e.feedback.decision) if e.feedback else False
        key = "pass" if decision else "fail"
        stats[e.operation][key] += 1

    ops = sorted(stats.keys())
    pass_counts = [stats[o]["pass"] for o in ops]
    fail_counts = [stats[o]["fail"] for o in ops]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ops, y=pass_counts,
        name="pass", marker_color="#54A24B",
        text=[str(c) for c in pass_counts], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=ops, y=fail_counts,
        name="fail", marker_color="#E45756",
        text=[str(c) for c in fail_counts], textposition="inside",
    ))

    fig.update_layout(
        title=title or "Operation 通过率",
        barmode="stack",
        xaxis_title="operation",
        yaxis_title="count",
        height=400,
        plot_bgcolor="white",
    )
    return fig
