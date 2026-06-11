"""谱系 DAG 布局 — BFS 分层 + 节点定位。

返回:
    - nodes: list of (entry_id, x, y, label, color, metadata)
    - edges: list of (parent_id, child_id)
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping

from ..trajectory import TrajectoryEntry


# 节点颜色 (按 operation)
_OPERATION_COLORS = {
    "original": "#4C78A8",
    "mutation": "#F58518",
    "crossover": "#E45756",
}


def build_lineage_layout(
    entries: Mapping[str, TrajectoryEntry] | list[TrajectoryEntry],
    metric: str = "sharpe",
) -> dict[str, Any]:
    """构造谱系 DAG 布局。

    Args:
        entries: TrajectoryEntry 列表或 dict
        metric: 用于节点大小/颜色的指标 (默认 sharpe)

    Returns:
        dict: {
            'nodes': [{'id', 'x', 'y', 'label', 'color', 'size', 'metric', 'operation', 'round_idx'}, ...],
            'edges': [{'source', 'target', 'color'}, ...],
        }
    """
    items = list(entries.values() if isinstance(entries, Mapping) else entries)
    if not items:
        return {"nodes": [], "edges": []}

    # 1. 分层: BFS 找每个节点的 round_idx (depth)
    depths: dict[str, int] = {}
    for e in items:
        if not e.parent_ids:
            depths[e.entry_id] = 0
        else:
            # 找最深 parent + 1
            max_parent_depth = -1
            for pid in e.parent_ids:
                if pid in depths:
                    max_parent_depth = max(max_parent_depth, depths[pid])
            depths[e.entry_id] = max_parent_depth + 1

    # 2. 按 depth 分组
    by_depth: dict[int, list[TrajectoryEntry]] = defaultdict(list)
    for e in items:
        by_depth[depths[e.entry_id]].append(e)

    # 3. 节点布局: x = index in layer, y = depth
    nodes = []
    for depth in sorted(by_depth.keys()):
        layer = by_depth[depth]
        for x_idx, e in enumerate(layer):
            metric_val = float((e.metrics or {}).get(metric, 0) or 0)
            op = e.operation
            color = _OPERATION_COLORS.get(op, "#999999")
            # 节点大小: 按 metric (min 10, max 30)
            size = 10 + min(20, max(0, metric_val) * 4)
            name = e.feedback.factor_name if e.feedback else e.entry_id[:8]
            decision = e.feedback.decision if e.feedback else False
            nodes.append({
                "id": e.entry_id,
                "x": x_idx,
                "y": depth,
                "label": f"{name}<br>r{depth} {op}<br>{metric}={metric_val:.2f}",
                "color": color,
                "size": size,
                "metric": metric_val,
                "operation": op,
                "round_idx": depth,
                "decision": decision,
            })

    # 4. 边: parent_id → child_id
    edges = []
    by_id = {e.entry_id: e for e in items}
    for e in items:
        for pid in e.parent_ids:
            if pid in by_id:
                parent_op = by_id[pid].operation
                edges.append({
                    "source": pid,
                    "target": e.entry_id,
                    "color": _OPERATION_COLORS.get(parent_op, "#999999"),
                })

    return {"nodes": nodes, "edges": edges}


def lineage_dag_figure(entries, metric: str = "sharpe", title: str | None = None) -> Any:
    """生成 Plotly Figure (交互式谱系 DAG)。

    Args:
        entries: TrajectoryEntry 列表
        metric: 用于节点大小/颜色的指标
        title: 图标题

    Returns:
        plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    layout = build_lineage_layout(entries, metric=metric)
    if not layout["nodes"]:
        fig = go.Figure()
        fig.update_layout(title="空 pool: 无 entry 可视化")
        return fig

    # 节点: scatter (x, y)
    node_x = [n["x"] for n in layout["nodes"]]
    node_y = [n["y"] for n in layout["nodes"]]
    node_text = [n["label"] for n in layout["nodes"]]
    node_color = [n["color"] for n in layout["nodes"]]
    node_size = [n["size"] for n in layout["nodes"]]
    node_custom = [n for n in layout["nodes"]]

    # 边: 通过 (source_x, source_y) → (target_x, target_y) 划线
    by_id = {n["id"]: n for n in layout["nodes"]}
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for e in layout["edges"]:
        src = by_id.get(e["source"])
        tgt = by_id.get(e["target"])
        if src is None or tgt is None:
            continue
        edge_x.extend([src["x"], tgt["x"], None])
        edge_y.extend([src["y"], tgt["y"], None])

    fig = go.Figure()

    # 边先画 (在底层)
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.5, color="#888"),
            hoverinfo="skip",
            showlegend=False,
        ))

    # 节点
    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color="#222"),
            opacity=0.85,
        ),
        text=[n["label"] for n in layout["nodes"]],
        textposition="top center",
        hovertemplate="%{text}<extra></extra>",
        customdata=node_custom,
        showlegend=False,
    ))

    # 图例 (operation 颜色)
    for op, color in _OPERATION_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=color),
            name=op,
            showlegend=True,
        ))

    fig.update_layout(
        title=title or f"演化谱系 DAG (按 {metric})",
        xaxis=dict(title="layer index", showgrid=False, zeroline=False),
        yaxis=dict(title="round (depth)", autorange="reversed", showgrid=False, zeroline=False),
        hovermode="closest",
        height=600,
        plot_bgcolor="white",
    )
    return fig
