"""Visualization report — generate_html() 主入口, 拼接 3 图 + 概览表。

Phase 1.3: 内部委托给 ReportBuilder, 保留向后兼容 API。
新代码推荐使用 ReportBuilder 流式 API。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from QuantNodes.core.trajectory.entry import TrajectoryEntry

from .builder import ReportBuilder


def generate_report(
    entries: list[TrajectoryEntry] | Mapping[str, TrajectoryEntry],
    metric: str = "sharpe",
    title: str = "QuantNodes 演化实验报告",
) -> dict[str, Any]:
    """生成 4 个 Plotly Figure (不输出 HTML)。

    Phase 1.3: 内部委托 ReportBuilder.with_evolve_preset()。
    返回格式保持向后兼容 (dict with overview + figure keys)。

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
    builder = (
        ReportBuilder()
        .with_title(title)
        .with_evolve_preset(entries, metric=metric)
    )
    return builder.build().to_dict()


def generate_html(
    entries,
    metric: str = "sharpe",
    title: str = "QuantNodes 演化实验报告",
    output_path: str | Path | None = None,
) -> str:
    """生成完整 HTML 报告, 含 4 个交互图 + 概览表。

    Phase 1.3: 内部委托 ReportBuilder.build_to_html()。

    Args:
        entries: TrajectoryEntry 列表
        metric: 用于可视化的指标
        title: 报告标题
        output_path: 若指定, 写入文件; 否则返回字符串

    Returns:
        str: HTML 内容 (若 output_path=None)
    """
    return (
        ReportBuilder()
        .with_title(title)
        .with_evolve_preset(entries, metric=metric)
        .build_to_html(output_path=output_path)
    )
