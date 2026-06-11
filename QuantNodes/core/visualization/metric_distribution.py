"""指标分布直方图 — Plotly histogram。"""
from __future__ import annotations

from typing import Any, Mapping

from ..trajectory import TrajectoryEntry


def metric_distribution_figure(
    entries,
    metric: str = "sharpe",
    title: str | None = None,
    n_bins: int = 20,
) -> Any:
    """生成 Plotly histogram (按 metric 分桶, 颜色按 operation)。

    Args:
        entries: TrajectoryEntry 列表
        metric: 指标名
        title: 标题
        n_bins: 直方图桶数

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    items = list(entries.values() if isinstance(entries, Mapping) else entries)
    # 按 operation 分组
    by_op: dict[str, list[float]] = {}
    for e in items:
        val = (e.metrics or {}).get(metric)
        if val is None:
            continue
        op = e.operation
        by_op.setdefault(op, []).append(float(val))

    if not by_op:
        fig = go.Figure()
        fig.update_layout(title=f"无 {metric} 指标数据")
        return fig

    fig = go.Figure()
    for op, vals in sorted(by_op.items()):
        fig.add_trace(go.Histogram(
            x=vals,
            name=op,
            opacity=0.7,
            xbins=dict(size=(max(vals) - min(vals)) / n_bins if max(vals) > min(vals) else 1),
            hovertemplate=f"{op}<br>{metric}=%{{x:.4f}}<br>count=%{{y}}<extra></extra>",
        ))

    fig.update_layout(
        title=title or f"{metric} 分布 (按 operation 着色)",
        barmode="overlay",
        xaxis_title=metric,
        yaxis_title="count",
        height=450,
        plot_bgcolor="white",
    )
    return fig


def metric_per_round_figure(
    entries,
    metric: str = "sharpe",
    title: str | None = None,
) -> Any:
    """每轮 best metric 趋势线 (line chart)。"""
    import plotly.graph_objects as go

    items = list(entries.values() if isinstance(entries, Mapping) else entries)
    if not items:
        fig = go.Figure()
        fig.update_layout(title="空 pool")
        return fig

    # 按 round 分组, 取每轮 max
    by_round: dict[int, list[float]] = {}
    for e in items:
        val = (e.metrics or {}).get(metric)
        if val is None:
            continue
        by_round.setdefault(e.round_idx, []).append(float(val))

    rounds = sorted(by_round.keys())
    bests = [max(by_round[r]) for r in rounds]
    means = [sum(by_round[r]) / len(by_round[r]) for r in rounds]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=bests, mode="lines+markers",
        name=f"best {metric}", line=dict(color="green", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=rounds, y=means, mode="lines+markers",
        name=f"mean {metric}", line=dict(color="blue", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=title or f"{metric} per round 趋势",
        xaxis_title="round",
        yaxis_title=metric,
        height=400,
        plot_bgcolor="white",
    )
    return fig
