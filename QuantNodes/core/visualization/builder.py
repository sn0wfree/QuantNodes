# coding=utf-8
"""ReportBuilder — 流式构造演化报告 (Phase 1.3, Builder pattern).

替代原来 generate_report() 内部硬编码的 dict 拼接逻辑。
提供 .with_title().with_overview().add_section(...).build() 流式 API,
并保证向后兼容 (旧 generate_report() 内部调用 ReportBuilder)。

设计要点:
  - Section 封装一个 fig + 可选标题
  - 报告结构: title + overview table + 多个 section, 顺序可定制
  - 与 plotly 解耦, 任何有 .to_html() 方法的对象都可作 section
  - build() 返回 Report dataclass (含 sections 列表 + overview dict),
    渲染时由 ReportRenderer 转换为 HTML
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from QuantNodes.core.path_utils import ensure_parent
from QuantNodes.core.trajectory.entry import TrajectoryEntry


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class Section:
    """报告中的一个 section, 含可选标题 + 可渲染对象 (有 .to_html() 方法)。"""
    title: str
    payload: Any  # plotly Figure 或其他有 to_html 的对象

    def render(self, include_plotlyjs: bool = False, div_id: Optional[str] = None) -> str:
        """渲染为 HTML fragment, 不含 <h2> 标题。

        div_id: 传给 payload.to_html() 的 div id; 传 None 时用默认
                 f"fig_{title.lower().replace(' ', '_')}" (与旧实现兼容)。

        v3.0.0 graceful degradation: when ``payload`` is None (e.g. plotly
        not installed), we emit a friendly install hint instead of
        crashing. ``to_dict`` still records the None value so callers
        can introspect the report structure.
        """
        if self.payload is None:
            return (
                "<p><em>(plotly not installed — install with "
                "<code>pip install plotly</code> to render this chart)</em></p>"
            )
        if hasattr(self.payload, "to_html"):
            kwargs = {"full_html": False, "include_plotlyjs": include_plotlyjs}
            if div_id is None:
                div_id = f"fig_{self.title.lower().replace(' ', '_')}"
            kwargs["div_id"] = div_id
            return self.payload.to_html(**kwargs)
        return f"<pre>{self.payload!r}</pre>"


@dataclass
class Report:
    """构建完成的报告。"""
    title: str
    overview: Dict[str, Any] = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """与旧 generate_report() 返回格式兼容: {overview, *section_keys}。"""
        out: Dict[str, Any] = {"overview": self.overview}
        for i, s in enumerate(self.sections):
            key = s.title.lower().replace(" ", "_")
            out[key] = s.payload
        return out


# ============================================================================
# ReportBuilder
# ============================================================================

class ReportBuilder:
    """流式构造 Report (Builder pattern)。

    用法:
        >>> from QuantNodes.core.visualization import ReportBuilder
        >>> report = (
        ...     ReportBuilder()
        ...     .with_title("演化报告")
        ...     .with_overview({"size": 10, "passed": 8})
        ...     .add_section("Lineage DAG", lineage_fig)
        ...     .add_section("Metrics", metric_fig)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._title: str = "Report"
        self._overview: Dict[str, Any] = {}
        self._sections: List[Section] = []
        self._preset_loaded = False

    def with_title(self, title: str) -> "ReportBuilder":
        """设置报告标题。"""
        self._title = title
        return self

    def with_overview(self, overview: Dict[str, Any]) -> "ReportBuilder":
        """设置概览 dict (会被原样放入 Report.overview)。"""
        self._overview = dict(overview)
        return self

    def add_section(self, title: str, payload: Any) -> "ReportBuilder":
        """追加一个 section。payload 通常是 plotly Figure, 任何有 to_html() 的对象都接受。"""
        self._sections.append(Section(title=title, payload=payload))
        return self

    def with_evolve_preset(
        self,
        entries: Sequence[TrajectoryEntry],
        metric: str = "sharpe",
        figure_factories: Optional[Dict[str, Callable]] = None,
    ) -> "ReportBuilder":
        """预置: 加载 4-5 个演化实验常用图 + 概览 (Phase 1.3 兼容 generate_report)。

        Args:
            entries: TrajectoryEntry 列表或 Mapping
            metric: 用于可视化的指标名
            figure_factories: 覆盖默认 5 个图 builder 的 dict
                e.g. {"lineage_dag": my_lineage_factory}
        """
        from .gate_breakdown import gate_breakdown_figure, operation_breakdown_figure
        from .lineage_dag import lineage_dag_figure
        from .metric_distribution import metric_distribution_figure, metric_per_round_figure

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

        self._overview = {
            "size": n,
            "rounds": len(rounds),
            "passed": n_passed,
            "passed_pct": (n_passed / n) if n > 0 else 0.0,
            "rejected": n_rejected,
            "best_metric": best_metric,
            "metric": metric,
        }
        self.with_title(self._title or "QuantNodes 演化实验报告")

        factories = figure_factories or {
            "lineage_dag": lambda: lineage_dag_figure(
                items, metric=metric, title=f"演化谱系 DAG (按 {metric})"
            ),
            "metric_distribution": lambda: metric_distribution_figure(items, metric=metric),
            "metric_per_round": lambda: metric_per_round_figure(items, metric=metric),
            "gate_breakdown": lambda: gate_breakdown_figure(items),
            "operation_breakdown": lambda: operation_breakdown_figure(items),
        }
        for title, factory in factories.items():
            self._sections.append(Section(title=title, payload=factory()))
        self._preset_loaded = True
        return self

    def build(self) -> Report:
        """构建最终 Report 对象。"""
        return Report(title=self._title, overview=self._overview, sections=list(self._sections))

    def build_to_html(
        self,
        output_path: Optional[Union[str, Path]] = None,
        plotly_cdn: str = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>',
    ) -> str:
        """直接构建并渲染为 HTML 字符串 (可选写入文件)。"""
        report = self.build()
        html = _render_html(report, plotly_cdn=plotly_cdn)
        if output_path is not None:
            output_path = Path(output_path)
            ensure_parent(output_path)
            output_path.write_text(html, encoding="utf-8")
        return html


# ============================================================================
# HTML rendering
# ============================================================================

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


def _render_html(report: Report, plotly_cdn: str) -> str:
    overview_html = _OVERVIEW_TEMPLATE.format(**report.overview) if report.overview else ""
    # 使用 title 派生 div_id (与旧 generate_html 行为兼容, div_id 格式 fig_<key>)
    parts = [f"<h2>{s.title}</h2>\n{s.render(include_plotlyjs=False)}"
             for s in report.sections]
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{report.title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; max-width: 1200px; }}
h1 {{ border-bottom: 2px solid #4C78A8; padding-bottom: 8px; }}
table {{ background: #fafafa; }}
th {{ background: #4C78A8; color: white; }}
</style>
{plotly_cdn}
</head>
<body>
<h1>{report.title}</h1>
{overview_html}
{"".join(parts)}
<hr>
<p style="color: #888; font-size: 12px;">Generated by QuantNodes ReportBuilder</p>
</body>
</html>
"""
