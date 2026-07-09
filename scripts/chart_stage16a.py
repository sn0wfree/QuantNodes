# coding=utf-8
"""Stage 16A: 生成多策略对比图表 (v3 vs v2)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = Path("reports/momentum_etf_rotation/charts/v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    """加载 NAV 序列和 summary."""
    nav_df = pd.read_parquet("reports/momentum_etf_rotation/v3/stage16a_navs.parquet")
    with open("reports/momentum_etf_rotation/v3/stage16a_summary.json") as f:
        summary = json.load(f)
    return nav_df, summary


def make_nav_comparison_chart(nav_df: pd.DataFrame, summary: dict) -> str:
    """NAV 对比图: v2 vs v3_equal vs v3_signal."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("累计净值曲线", "924 专项 (2024-09-23 ~ 2024-10-31)"),
    )

    colors = {"v2": "#0d6efd", "v3_equal": "#28a745", "v3_signal": "#dc3545"}

    # 上图: 全周期 NAV
    for col in nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_df.index, y=nav_df[col],
                mode="lines", name=col,
                line=dict(color=colors.get(col, "#666"), width=2),
                legendgroup=col, showlegend=True,
            ),
            row=1, col=1,
        )

    # 下图: 924 专项
    for col in nav_df.columns:
        win = nav_df[col].loc["2024-09-23":"2024-10-31"]
        if len(win) > 0:
            win_norm = win / win.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=win_norm.index, y=win_norm.values,
                    mode="lines+markers", name=f"{col} (924)",
                    line=dict(color=colors.get(col, "#666"), width=2, dash="dot"),
                    marker=dict(size=4),
                    legendgroup=col, showlegend=False,
                ),
                row=2, col=1,
            )

    fig.update_layout(
        title="Stage 16A: 多策略组合 (v3) vs 单策略 (v2) — 净值对比",
        height=700, hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="累计净值", row=1, col=1)
    fig.update_yaxes(title_text="924 归一化净值", row=2, col=1)

    out = OUT_DIR / "stage16a_nav_comparison.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def make_metrics_comparison_chart(summary: dict) -> str:
    """指标对比图: 柱状图."""
    metrics_keys = ["ann_return", "ann_vol", "sharpe", "calmar"]
    metric_labels = {
        "ann_return": "年化收益",
        "ann_vol": "年化波动",
        "sharpe": "Sharpe",
        "calmar": "Calmar",
    }

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[metric_labels[k] for k in metrics_keys],
    )

    configs = list(summary.keys())
    colors = ["#0d6efd", "#28a745", "#dc3545"]

    for i, k in enumerate(metrics_keys, start=1):
        values = [summary[c]["metrics"].get(k, 0) for c in configs]
        # ann_vol 转为百分比
        if k in ("ann_return", "ann_vol"):
            values = [v * 100 for v in values]
        # max_drawdown 取绝对值百分比
        if k == "max_drawdown":
            values = [abs(summary[c]["metrics"].get(k, 0)) * 100 for c in configs]

        fig.add_trace(
            go.Bar(
                x=configs, y=values,
                marker_color=colors[:len(configs)],
                text=[f"{v:.2f}" for v in values],
                textposition="outside",
                showlegend=False,
            ),
            row=1, col=i,
        )

    fig.update_layout(
        title="Stage 16A: 全周期指标对比",
        height=400, template="plotly_white",
    )

    out = OUT_DIR / "stage16a_metrics_comparison.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def make_924_focus_chart(nav_df: pd.DataFrame, summary: dict) -> str:
    """924 专项深度图."""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("924 期间详细走势 (2024-09-23 ~ 2024-10-31)",),
    )

    colors = {"v2": "#0d6efd", "v3_equal": "#28a745", "v3_signal": "#dc3545"}

    for col in nav_df.columns:
        win = nav_df[col].loc["2024-09-23":"2024-10-31"]
        if len(win) > 0:
            win_norm = win / win.iloc[0]
            ret = (win_norm.iloc[-1] - 1) * 100
            fig.add_trace(
                go.Scatter(
                    x=win_norm.index, y=(win_norm.values - 1) * 100,
                    mode="lines+markers", name=f"{col} (累计 +{ret:.2f}%)",
                    line=dict(color=colors.get(col, "#666"), width=2),
                    marker=dict(size=6),
                ),
            )

    # 标记 9/24 政策发布日
    fig.add_vline(
        x="2024-09-24", line_dash="dash", line_color="red",
        annotation_text="9/24 政策发布", annotation_position="top",
    )

    fig.update_layout(
        title="Stage 16A: 924 政策窗口累计收益 (%)",
        height=500, hovermode="x unified",
        template="plotly_white",
        yaxis_title="累计收益 (%)",
        xaxis_title="日期",
    )

    out = OUT_DIR / "stage16a_924_focus.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def main():
    print("=" * 60)
    print("生成 v3 多策略对比图表")
    print("=" * 60)

    nav_df, summary = load_data()
    print(f"加载数据: {nav_df.shape[0]} 天, {nav_df.shape[1]} 配置")

    print("\n[1/3] NAV 对比图...")
    p1 = make_nav_comparison_chart(nav_df, summary)
    print(f"  -> {p1}")

    print("\n[2/3] 指标对比柱状图...")
    p2 = make_metrics_comparison_chart(summary)
    print(f"  -> {p2}")

    print("\n[3/3] 924 专项深度图...")
    p3 = make_924_focus_chart(nav_df, summary)
    print(f"  -> {p3}")

    print("\n所有图表生成完成!")


if __name__ == "__main__":
    main()
