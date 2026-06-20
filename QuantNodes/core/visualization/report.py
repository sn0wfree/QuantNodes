"""Visualization report — generate_html() 主入口, 拼接 3 图 + 概览表。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..trajectory import TrajectoryEntry
from .gate_breakdown import gate_breakdown_figure, operation_breakdown_figure
from .lineage_dag import lineage_dag_figure
from .metric_distribution import metric_distribution_figure, metric_per_round_figure
from QuantNodes.core.path_utils import ensure_parent


_OVERVIEW_TEMPLATE = """
<h2>概览</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
  <tr><th>指标</th><th>值</th></tr>
  <tr><td>总 entry 数</td><td>{size}</td></tr>
  <tr><td>演化轮数</td><td>{rounds}</td></tr>
  <tr><td>通过数</td><td>{passed} ({passed_pct:.1%})</td></tr>
  <tr><td>拒绝数</td><td>{rejected}</td></tr>
  <tr><td>Best {metric}</td><td>{best_metric:.4f}</td></tr>
</table>
"""


def generate_report(
    entries: list[TrajectoryEntry] | Mapping[str, TrajectoryEntry],
    metric: str = "sharpe",
    title: str = "QuantNodes 演化实验报告",
) -> dict[str, Any]:
    """生成 4 个 Plotly Figure (不输出 HTML)。

    Returns:
        dict: {
            'overview': {...},
            'lineage_dag': go.Figure,
            'metric_distribution': go.Figure,
            'metric_per_round': go.Figure,
            'gate_breakdown': go.Figure,
            'operation_breakdown': go.Figure,
        }
    """
    items = list(entries.values() if isinstance(entries, Mapping) else entries)
    n = len(items)
    n_passed = sum(1 for e in items if e.feedback and e.feedback.decision)
    n_rejected = n - n_passed
    rounds = sorted({e.round_idx for e in items}) if items else []
    metrics_vals = [
        float((e.metrics or {}).get(metric, 0) or 0)
        for e in items
        if (e.metrics or {}).get(metric) is not None
    ]
    best_metric = max(metrics_vals) if metrics_vals else 0.0

    overview = {
        "size": n,
        "rounds": len(rounds),
        "passed": n_passed,
        "passed_pct": (n_passed / n) if n > 0 else 0.0,
        "rejected": n_rejected,
        "best_metric": best_metric,
        "metric": metric,
    }

    return {
        "overview": overview,
        "lineage_dag": lineage_dag_figure(
            items, metric=metric, title=f"演化谱系 DAG (按 {metric})"
        ),
        "metric_distribution": metric_distribution_figure(items, metric=metric),
        "metric_per_round": metric_per_round_figure(items, metric=metric),
        "gate_breakdown": gate_breakdown_figure(items),
        "operation_breakdown": operation_breakdown_figure(items),
    }


def generate_html(
    entries,
    metric: str = "sharpe",
    title: str = "QuantNodes 演化实验报告",
    output_path: str | Path | None = None,
) -> str:
    """生成完整 HTML 报告, 含 4 个交互图 + 概览表。

    Args:
        entries: TrajectoryEntry 列表
        metric: 用于可视化的指标
        title: 报告标题
        output_path: 若指定, 写入文件; 否则返回字符串

    Returns:
        str: HTML 内容 (若 output_path=None)
    """
    report = generate_report(entries, metric=metric, title=title)
    overview = report["overview"]

    figures = [
        ("lineage_dag", "演化谱系 DAG"),
        ("metric_distribution", "指标分布"),
        ("metric_per_round", "每轮指标趋势"),
        ("gate_breakdown", "Quality Gate 拦截率"),
        ("operation_breakdown", "Operation 通过率"),
    ]

    fig_html_parts = []
    for key, name in figures:
        fig = report[key]
        # include_plotlyjs='cdn' 让 5 个图共享 1 个 plotly.js 引用
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig_{key}")
        fig_html_parts.append(f"<h2>{name}</h2>\n{fig_html}")

    overview_html = _OVERVIEW_TEMPLATE.format(**overview)
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; max-width: 1200px; }}
h1 {{ border-bottom: 2px solid #4C78A8; padding-bottom: 8px; }}
table {{ background: #fafafa; }}
th {{ background: #4C78A8; color: white; }}
</style>
{plotly_cdn}
</head>
<body>
<h1>{title}</h1>
{overview_html}
{"".join(fig_html_parts)}
<hr>
<p style="color: #888; font-size: 12px;">
生成自 QuantNodes 演化框架 (Week 6 Visualization)
</p>
</body>
</html>
"""

    if output_path is not None:
        output_path = Path(output_path)
        ensure_parent(output_path)
        output_path.write_text(html, encoding="utf-8")
    return html
