"""Visualization — 演化实验交互式 HTML 报告。

公开 API:
    - generate_report(entries, metric) -> dict[Figure]
    - generate_html(entries, metric, output_path) -> str / 写文件
    - ReportBuilder / Section / Report (Phase 1.3 fluent API)
    - build_lineage_layout(entries, metric) -> {nodes, edges}
    - lineage_dag_figure / metric_distribution_figure / metric_per_round_figure
    - gate_breakdown_figure / operation_breakdown_figure
"""
from .lineage_dag import build_lineage_layout, lineage_dag_figure
from .metric_distribution import (
    metric_distribution_figure,
    metric_per_round_figure,
)
from .gate_breakdown import gate_breakdown_figure, operation_breakdown_figure
from .report import generate_html, generate_report
from .builder import ReportBuilder, Section, Report

__all__ = [
    "build_lineage_layout",
    "lineage_dag_figure",
    "metric_distribution_figure",
    "metric_per_round_figure",
    "gate_breakdown_figure",
    "operation_breakdown_figure",
    "generate_report",
    "generate_html",
    # Phase 1.3
    "ReportBuilder",
    "Section",
    "Report",
]
