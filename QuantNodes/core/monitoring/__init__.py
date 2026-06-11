"""Monitoring — 3 类指标中央收集 + Plotly dashboard。

公开 API:
    - RagMetrics / EvolutionMetrics / QualityMetrics (数据类)
    - MetricCollector (中央收集器)
    - generate_dashboard_html (HTML 报告)
"""
from .collector import (
    EvolutionMetrics,
    MetricCollector,
    QualityMetrics,
    RagMetrics,
)
from .dashboard import generate_dashboard_html

__all__ = [
    "RagMetrics",
    "EvolutionMetrics",
    "QualityMetrics",
    "MetricCollector",
    "generate_dashboard_html",
]
