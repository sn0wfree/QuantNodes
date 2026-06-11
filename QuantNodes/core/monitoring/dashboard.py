"""Dashboard — 6 个 Plotly 图的 HTML 仪表板。

3 类指标 × 2 个图 (汇总 + per-round) = 6 个 figure + 1 个概览表。

图列表:
    1. RAG 折线 (HR@5/NDCG@5/MRR/Diversity 随 round 变化)
    2. RAG 散点 (NDCG@5 vs HR@5 关联性)
    3. 演化柱状 (pool_size / total / rejected per round)
    4. 演化折线 (best_metric 趋势)
    5. Quality Gate 堆叠柱 (3 通道 pass/fail)
    6. Quality Gate 拦截率折线 (per-round rejection rate)
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .collector import MetricCollector


def _fig_rag_lines(c: MetricCollector) -> Any:
    """RAG 指标折线: HR/NDCG/MRR/Diversity 随 round。"""
    import plotly.graph_objects as go
    if not c.rag_history:
        fig = go.Figure()
        fig.update_layout(title="RAG 指标 (无数据)")
        return fig
    df = pd.DataFrame([m.__dict__ for m in c.rag_history])
    fig = go.Figure()
    for col, name, color in [
        ("hit_at_5", "HitRate@5", "#2E86AB"),
        ("ndcg_at_5", "NDCG@5", "#A23B72"),
        ("mrr", "MRR", "#F18F01"),
        ("diversity", "Diversity", "#6A994E"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["round"], y=df[col], mode="lines+markers",
                name=name, line=dict(color=color, width=2),
            ))
    fig.update_layout(
        title="RAG 指标趋势 (per round)",
        xaxis_title="round", yaxis_title="score",
        height=400, plot_bgcolor="white",
    )
    return fig


def _fig_rag_scatter(c: MetricCollector) -> Any:
    """RAG NDCG vs HitRate 散点。"""
    import plotly.graph_objects as go
    if not c.rag_history:
        fig = go.Figure()
        fig.update_layout(title="RAG 散点 (无数据)")
        return fig
    df = pd.DataFrame([m.__dict__ for m in c.rag_history])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["hit_at_5"], y=df["ndcg_at_5"],
        mode="markers+text", text=df["round"],
        marker=dict(size=12, color="#4C78A8"),
        hovertemplate="round %{text}<br>HR@5=%{x:.3f}<br>NDCG@5=%{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="RAG: HitRate@5 vs NDCG@5",
        xaxis_title="HitRate@5", yaxis_title="NDCG@5",
        height=350, plot_bgcolor="white",
    )
    return fig


def _fig_evolution_bar(c: MetricCollector) -> Any:
    """演化统计柱状: total / rejected per round。"""
    import plotly.graph_objects as go
    if not c.evolution_history:
        fig = go.Figure()
        fig.update_layout(title="演化统计 (无数据)")
        return fig
    df = pd.DataFrame([m.__dict__ for m in c.evolution_history])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["round"], y=df["total_count"],
                         name="passed", marker_color="#54A24B"))
    fig.add_trace(go.Bar(x=df["round"], y=df["rejected_count"],
                         name="rejected", marker_color="#E45756"))
    fig.update_layout(
        title="演化统计 (per round)",
        barmode="stack", xaxis_title="round", yaxis_title="count",
        height=400, plot_bgcolor="white",
    )
    return fig


def _fig_evolution_line(c: MetricCollector) -> Any:
    """best_metric 折线趋势。"""
    import plotly.graph_objects as go
    if not c.evolution_history:
        fig = go.Figure()
        fig.update_layout(title="Best metric 趋势 (无数据)")
        return fig
    df = pd.DataFrame([m.__dict__ for m in c.evolution_history])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["round"], y=df["best_metric"],
        mode="lines+markers+text", text=df["best_factor_name"],
        textposition="top center",
        line=dict(color="#06A77D", width=3),
        marker=dict(size=12),
    ))
    fig.update_layout(
        title="Best Metric 趋势 (per round)",
        xaxis_title="round", yaxis_title="best metric",
        height=400, plot_bgcolor="white",
    )
    return fig


def _fig_quality_stacked(c: MetricCollector) -> Any:
    """Quality Gate 3 通道 pass/fail 堆叠柱。"""
    import plotly.graph_objects as go
    if not c.quality_history:
        fig = go.Figure()
        fig.update_layout(title="Quality Gate (无数据)")
        return fig
    df = pd.DataFrame([m.__dict__ for m in c.quality_history])
    channels = ["code", "value", "llm"]
    pass_by_ch = {ch: [getattr(m, f"{ch}_pass") for m in c.quality_history] for ch in channels}
    fail_by_ch = {ch: [getattr(m, f"{ch}_fail") for m in c.quality_history] for ch in channels}
    fig = go.Figure()
    for ch in channels:
        fig.add_trace(go.Bar(name=f"{ch} pass", x=df["round"], y=pass_by_ch[ch],
                             marker_color="#54A24B",
                             legendgroup=ch, offsetgroup=ch))
        fig.add_trace(go.Bar(name=f"{ch} fail", x=df["round"], y=fail_by_ch[ch],
                             marker_color="#E45756",
                             legendgroup=ch, offsetgroup=ch))
    fig.update_layout(
        title="Quality Gate 3 通道 (per round)",
        barmode="stack", xaxis_title="round", yaxis_title="count",
        height=400, plot_bgcolor="white",
    )
    return fig


def _fig_quality_rejection(c: MetricCollector) -> Any:
    """Quality Gate 拦截率折线 (pass / total per channel)。"""
    import plotly.graph_objects as go
    if not c.quality_history:
        fig = go.Figure()
        fig.update_layout(title="Quality 拦截率 (无数据)")
        return fig
    df_dicts = []
    for m in c.quality_history:
        for ch in ("code", "value", "llm"):
            pass_n = getattr(m, f"{ch}_pass")
            fail_n = getattr(m, f"{ch}_fail")
            total = pass_n + fail_n
            rate = fail_n / total if total > 0 else 0.0
            df_dicts.append({"round": m.round, "channel": ch, "rejection_rate": rate, "total": total})
    df = pd.DataFrame(df_dicts)
    fig = go.Figure()
    colors = {"code": "#4C78A8", "value": "#F58518", "llm": "#E45756"}
    for ch in ("code", "value", "llm"):
        sub = df[df["channel"] == ch]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["round"], y=sub["rejection_rate"],
            mode="lines+markers", name=ch,
            line=dict(color=colors[ch], width=2),
        ))
    fig.update_layout(
        title="Quality Gate 拦截率 (per channel, per round)",
        xaxis_title="round", yaxis_title="rejection rate",
        yaxis=dict(range=[0, 1]),
        height=400, plot_bgcolor="white",
    )
    return fig


_OVERVIEW_TEMPLATE = """
<h2>概览</h2>
<table border="1" cellpadding="6" style="border-collapse:collapse;">
  <tr><th>指标</th><th>值</th></tr>
  <tr><td>RAG 数据点</td><td>{n_rag}</td></tr>
  <tr><td>演化数据点</td><td>{n_evo}</td></tr>
  <tr><td>质量门数据点</td><td>{n_qg}</td></tr>
  <tr><td>最终 best metric</td><td>{best_metric:.4f}</td></tr>
  <tr><td>最终 best factor</td><td>{best_factor}</td></tr>
  <tr><td>生成时间</td><td>{timestamp}</td></tr>
</table>
"""


def generate_dashboard_html(
    collector: MetricCollector,
    title: str = "QuantNodes 演化 Dashboard",
    output_path: Path | str | None = None,
    streaming: bool = False,
    refresh_interval_sec: int = 10,
) -> str:
    """生成完整 HTML dashboard, 6 figure + 概览表。

    Args:
        streaming: True → 加 JS 定时刷新 (读取同目录 metrics.json)
        refresh_interval_sec: 刷新间隔秒数 (默认 10s)
    """
    import plotly.graph_objects as go

    figures = [
        ("rag_lines", "RAG 指标趋势", _fig_rag_lines(collector)),
        ("rag_scatter", "RAG HR@5 vs NDCG@5", _fig_rag_scatter(collector)),
        ("evo_bar", "演化统计 (per round)", _fig_evolution_bar(collector)),
        ("evo_line", "Best Metric 趋势", _fig_evolution_line(collector)),
        ("qg_stacked", "Quality Gate 通道", _fig_quality_stacked(collector)),
        ("qg_rejection", "Quality 拦截率", _fig_quality_rejection(collector)),
    ]

    overview = {
        "n_rag": len(collector.rag_history),
        "n_evo": len(collector.evolution_history),
        "n_qg": len(collector.quality_history),
        "best_metric": (
            collector.evolution_history[-1].best_metric
            if collector.evolution_history else 0.0
        ),
        "best_factor": (
            collector.evolution_history[-1].best_factor_name
            if collector.evolution_history else "(none)"
        ),
        "timestamp": datetime.now().isoformat(),
    }

    fig_html_parts = []
    for key, name, fig in figures:
        fig_html = fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"fig_{key}")
        fig_html_parts.append(f"<h2>{name}</h2>\n{fig_html}")

    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

    # Streaming JS (可选)
    streaming_js = ""
    if streaming:
        # JS: 定时刷新, 检测 metrics.json 变化后重新加载
        streaming_js = f"""
<script>
(function() {{
  var INTERVAL = {refresh_interval_sec} * 1000;
  var lastModified = 0;
  var metricsPath = "{str(Path(output_path).parent / (Path(output_path).stem + '_metrics.json')) if output_path else 'metrics.json'}";

  function checkUpdate() {{
    fetch(metricsPath + '?t=' + Date.now(), {{method: 'HEAD'}})
      .then(r => {{
        var m = new Date(r.headers.get('Last-Modified') || 0).getTime();
        if (m > lastModified) {{
          lastModified = m;
          location.reload();
        }}
      }})
      .catch(() => {{}});
  }}

  setInterval(checkUpdate, INTERVAL);
  console.log("[Streaming] 每 {refresh_interval_sec}s 检测更新...");
</script>
"""
    streaming_badge = (
        '<span style="background:#06A77D;color:white;padding:2px 8px;'
        'border-radius:4px;font-size:12px;margin-left:10px;">LIVE</span>'
        if streaming else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 20px; max-width: 1400px; }}
h1 {{ border-bottom: 2px solid #4C78A8; padding-bottom: 8px; }}
h2 {{ border-top: 1px solid #ddd; padding-top: 12px; color: #2E86AB; }}
table {{ background: #fafafa; }}
th {{ background: #4C78A8; color: white; }}
.live-badge {{ animation: pulse 2s infinite; }}
@keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
</style>
{plotly_cdn}
</head>
<body>
<h1>{title} {streaming_badge}</h1>
{_OVERVIEW_TEMPLATE.format(**overview)}
{"".join(fig_html_parts)}
<hr>
<p style="color: #888; font-size: 12px;">
生成自 QuantNodes 演化框架 (Week 16 Streaming Dashboard)
{'  · 自动刷新: ' + str(refresh_interval_sec) + 's' if streaming else ''}
</p>
{streaming_js}
</body>
</html>
"""

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    return html
