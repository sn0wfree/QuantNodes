# coding=utf-8
"""Stage 17 v4 图表生成 (6 模式 + IC + HMM).

输出:
- charts/v4/mode_comparison.html: 6 模式 NAV 对比
- charts/v4/factor_ic_heatmap.html: 6 因子 IC 时序
- charts/v4/factor_weights.html: 因子权重时序
- charts/v4/hmm_regime.html: HMM 状态时序
- charts/v4/distance_transmat.html: 距离先验矩阵热力图
- charts/v4/period_returns.html: 关键区间柱状图
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPORTS_DIR = PROJECT_ROOT / "reports" / "momentum_etf_rotation" / "v4"
CHARTS_DIR = PROJECT_ROOT / "charts" / "v4"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

REGIME_LABELS = {0: "牛 (bull)", 1: "熊 (bear)", 2: "转换 (transition)"}


def chart_mode_comparison() -> str:
    """6 模式 NAV 对比 + 关键区间填充."""
    nav_all = pd.read_parquet(REPORTS_DIR / "stage17_navs.parquet")

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=("Stage 17 — 6 模式 NAV 对比 (2018-2026)", "关键区间放大"),
    )

    colors = {
        "v3_baseline":   "#0d6efd",
        "v4A_style":     "#dc3545",
        "v4B_smartbeta": "#ffc107",
        "v4C_combo":     "#198754",
        "v4D_ic":        "#6f42c1",
        "v4E_hmm":       "#fd7e14",
        "v4F_fusion":    "#20c997",
    }
    widths = {
        "v3_baseline":   3,
        "v4A_style":     1,
        "v4B_smartbeta": 1,
        "v4C_combo":     1.5,
        "v4D_ic":        1,
        "v4E_hmm":       1,
        "v4F_fusion":    1,
    }

    for col in nav_all.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_all.index, y=nav_all[col],
                mode="lines", name=col,
                line=dict(color=colors.get(col, "#999"), width=widths.get(col, 1)),
            ),
            row=1, col=1,
        )

    # 2026 H1 放大
    win = nav_all.loc["2025-12-31":"2026-06-30"]
    for col in nav_all.columns:
        win_norm = (win[col] / win[col].iloc[0] - 1) * 100
        fig.add_trace(
            go.Scatter(
                x=win_norm.index, y=win_norm.values,
                mode="lines+markers", name=f"{col} (2026 H1 %)",
                line=dict(color=colors.get(col, "#999"), width=1.5),
                marker=dict(size=3),
                showlegend=False,
            ),
            row=2, col=1,
        )

    fig.update_layout(
        title="Stage 17 v4 — 6 模式 NAV 对比",
        height=800, hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="2026 H1 累计收益 (%)", row=2, col=1)

    out = CHARTS_DIR / "mode_comparison.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def chart_factor_ic_heatmap() -> str:
    """6 因子 IC 时序热力图 (rolling mean)."""
    ic = pd.read_parquet(REPORTS_DIR / "ic_history.parquet")

    fig = go.Figure()

    for factor in ic.columns:
        fig.add_trace(
            go.Scatter(
                x=ic.index, y=ic[factor],
                mode="lines", name=factor,
            ),
        )

    # 添加 0 轴线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title="6 因子 IC 时序 (rolling mean, 12 step smooth)",
        xaxis_title="日期",
        yaxis_title="IC (Spearman)",
        height=500, hovermode="x unified",
        template="plotly_white",
    )
    out = CHARTS_DIR / "factor_ic_heatmap.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def chart_factor_weights() -> str:
    """因子权重时序."""
    weights = pd.read_parquet(REPORTS_DIR / "factor_weights.parquet")

    fig = go.Figure()

    colors = ["#0d6efd", "#dc3545", "#198754", "#ffc107", "#6f42c1", "#fd7e14"]
    for i, factor in enumerate(weights.columns):
        fig.add_trace(
            go.Scatter(
                x=weights.index, y=weights[factor],
                mode="lines", name=factor,
                stackgroup="one",  # 堆叠
                line=dict(color=colors[i % len(colors)], width=1),
            ),
        )

    fig.update_layout(
        title="因子权重时序 (IC 加权, 堆叠)",
        xaxis_title="日期",
        yaxis_title="权重",
        height=500, hovermode="x unified",
        template="plotly_white",
    )
    out = CHARTS_DIR / "factor_weights.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def chart_hmm_regime() -> str:
    """HMM 状态时序."""
    try:
        hmm = pd.read_csv(REPORTS_DIR / "hmm_regime_history.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return "(no HMM data)"

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.3, 0.7],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("HMM Regime State", "NAV with Regime Overlay"),
    )

    # 上图: regime 状态
    colors = {0: "#198754", 1: "#dc3545", 2: "#ffc107"}
    hmm_arr = hmm.iloc[:, 0].values  # 第一列
    for i, state in enumerate(np.unique(hmm_arr)):
        mask = hmm_arr == state
        fig.add_trace(
            go.Scatter(
                x=hmm.index[mask], y=[state] * mask.sum(),
                mode="markers",
                marker=dict(size=8, color=colors.get(int(state), "#999")),
                name=REGIME_LABELS.get(int(state), f"state_{int(state)}"),
            ),
            row=1, col=1,
        )

    # 下图: NAV + regime 背景
    nav_all = pd.read_parquet(REPORTS_DIR / "stage17_navs.parquet")
    fig.add_trace(
        go.Scatter(
            x=nav_all.index, y=nav_all["v4C_combo"],
            mode="lines", name="v4C_combo",
            line=dict(color="#0d6efd", width=2),
        ),
        row=2, col=1,
    )

    # 给 regime 区域加背景色
    for i, state in enumerate(np.unique(hmm_arr)):
        mask = hmm_arr == state
        if mask.sum() > 0:
            fig.add_vrect(
                x0=hmm.index[mask][0], x1=hmm.index[mask][-1],
                fillcolor=colors.get(int(state), "#999"),
                opacity=0.15, line_width=0,
                row=2, col=1,
            )

    fig.update_layout(
        title="HMM Regime 时序 + NAV 叠加",
        height=700, hovermode="x unified",
        template="plotly_white",
    )
    fig.update_yaxes(title_text="Regime", row=1, col=1, tickmode="array", tickvals=[0, 1, 2],
                     ticktext=["bull", "bear", "transition"])
    fig.update_yaxes(title_text="NAV", row=2, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)

    out = CHARTS_DIR / "hmm_regime.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def chart_distance_transmat() -> str:
    """距离先验矩阵热力图."""
    from QuantNodes.strategy.momentum_etf_rotation.v4 import build_distance_transmat

    m = build_distance_transmat(alpha=1.5, gamma=0.3)

    text = [[f"{m[i, j]:.3f}" for j in range(3)] for i in range(3)]
    fig = go.Figure(data=go.Heatmap(
        z=m, x=["bear", "transition", "bull"], y=["bear", "transition", "bull"],
        text=text, texttemplate="%{text}",
        colorscale="YlOrRd",
        colorbar=dict(title="P(to|from)"),
    ))
    fig.update_layout(
        title="距离先验转移矩阵 (alpha=1.5, gamma=0.3)",
        xaxis_title="目标状态", yaxis_title="源状态",
        height=500,
    )
    out = CHARTS_DIR / "distance_transmat.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def chart_period_returns() -> str:
    """关键区间收益柱状图."""
    with open(REPORTS_DIR / "stage17_summary.json") as f:
        summary = json.load(f)

    period_data = summary["period_returns"]
    modes = list(summary["metrics"].keys())
    periods = list(period_data.keys())

    fig = go.Figure()
    colors = {
        "v3_baseline":   "#0d6efd",
        "v4A_style":     "#dc3545",
        "v4B_smartbeta": "#ffc107",
        "v4C_combo":     "#198754",
        "v4D_ic":        "#6f42c1",
        "v4E_hmm":       "#fd7e14",
        "v4F_fusion":    "#20c997",
    }

    for mode in modes:
        vals = [period_data[p].get(mode, 0) for p in periods]
        fig.add_trace(
            go.Bar(
                x=periods, y=[v * 100 for v in vals],
                name=mode,
                marker_color=colors.get(mode, "#999"),
            ),
        )

    fig.update_layout(
        title="关键区间收益对比 (3 区间 × 6 模式)",
        xaxis_title="区间",
        yaxis_title="区间收益 (%)",
        barmode="group", height=500,
        template="plotly_white",
    )
    out = CHARTS_DIR / "period_returns.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def main():
    print("=" * 60)
    print("Stage 17 v4 图表生成")
    print("=" * 60)

    print("\n[1/6] mode_comparison.html...")
    print(f"  -> {chart_mode_comparison()}")

    print("\n[2/6] factor_ic_heatmap.html...")
    print(f"  -> {chart_factor_ic_heatmap()}")

    print("\n[3/6] factor_weights.html...")
    print(f"  -> {chart_factor_weights()}")

    print("\n[4/6] hmm_regime.html...")
    print(f"  -> {chart_hmm_regime()}")

    print("\n[5/6] distance_transmat.html...")
    print(f"  -> {chart_distance_transmat()}")

    print("\n[6/6] period_returns.html...")
    print(f"  -> {chart_period_returns()}")

    print("\n所有图表生成完成!")


if __name__ == "__main__":
    main()
