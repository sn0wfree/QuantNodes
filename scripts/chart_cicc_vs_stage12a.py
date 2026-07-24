# coding=utf-8
"""CICC vs Stage 12A 对比图表 (2 个).

1. cicc_vs_stage12a.html: 4 个配置 NAV 对比 (2018-2026 完整 + 2026 H1 放大)
2. data_version_impact.html: 新旧数据对 NAV 的影响 (假设同一策略)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = Path("reports/momentum_etf_rotation/charts/common")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_main_comparison(nav_df: pd.DataFrame) -> str:
    """主对比图: 4 个配置 NAV + 2026 H1 放大."""
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=("全周期 NAV 对比 (2018-2026)", "2026 H1 专项"),
    )

    colors = {
        "v1_CICC_baseline": "#6c757d",   # 灰 - CICC 原始
        "v1_CICC_vt": "#fd7e14",         # 橙 - CICC + VT
        "v2_hybrid": "#0d6efd",          # 蓝 - Stage 12A hybrid
        "v2_hybrid_vt": "#28a745",       # 绿 - Stage 12A hybrid + VT (推荐)
    }
    labels = {
        "v1_CICC_baseline": "v1 CICC baseline (price, 无 VT)",
        "v1_CICC_vt":       "v1 CICC + VT (price + tv=0.15)",
        "v2_hybrid":        "v2 hybrid (price + slope_r²)",
        "v2_hybrid_vt":     "v2 hybrid + VT + Cost ⭐推荐",
    }
    dashes = {
        "v1_CICC_baseline": "dot",
        "v1_CICC_vt":       "dot",
        "v2_hybrid":        "solid",
        "v2_hybrid_vt":     "solid",
    }

    # 上图: 全周期 NAV
    for col in nav_df.columns:
        fig.add_trace(
            go.Scatter(
                x=nav_df.index, y=nav_df[col],
                mode="lines", name=labels[col],
                line=dict(color=colors[col], width=2, dash=dashes[col]),
                legendgroup=col, showlegend=True,
            ),
            row=1, col=1,
        )

    # 下图: 2026 H1 专项 (2025-12-31 ~ 2026-06-30)
    win_2026 = nav_df.loc["2025-12-31":"2026-06-30"]
    for col in nav_df.columns:
        win_norm = win_2026[col] / win_2026[col].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=win_norm.index, y=(win_norm.values - 1) * 100,
                mode="lines+markers", name=labels[col] + " (2026 H1)",
                line=dict(color=colors[col], width=2, dash=dashes[col]),
                marker=dict(size=4),
                legendgroup=col, showlegend=False,
            ),
            row=2, col=1,
        )

    # 标记 2026 月度调仓日
    month_ends = pd.Series(win_2026.index).groupby(win_2026.index.to_period("M")).max()
    for d in month_ends:
        if pd.Timestamp(d) <= win_2026.index[-1]:
            fig.add_vline(
                x=str(d.date()), line_dash="dot", line_color="gray",
                line_width=0.5, row=2, col=1,
            )

    fig.update_layout(
        title="CICC (v1) vs Stage 12A (v2) 业绩对比 · 2018-2026",
        height=800, hovermode="x unified",
        template="plotly_white",
    )
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="累计净值", row=1, col=1)
    fig.update_yaxes(title_text="2026 H1 累计收益 (%)", row=2, col=1)

    out = OUT_DIR / "cicc_vs_stage12a.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def make_data_version_impact() -> str:
    """数据版本影响: 同一策略在新旧数据上的 NAV 差异.

    注意: 由于 v1 旧 HTML 图表使用了真实回测结果 (旧数据), 我们重新跑相同参数
    在新数据上,对比 2025-07-04 时点的 NAV 差异.
    """
    # 加载新旧数据
    panel_old = pd.read_parquet("data/real/etf_nav_2018-01-01_2025-07-06.parquet")
    panel_new = pd.read_parquet("data/real/etf_nav_2018-01-01_2026-06-30.parquet")

    from QuantNodes.strategy.momentum_etf_rotation import (
        DEFAULT_POOL, BacktestConfig, RotationConfig, run_rotation_backtest,
    )

    # 同一配置: v1 CICC baseline (price, 无 VT)
    rot = RotationConfig(lookback=144, top_n=10, momentum_type="price")

    print("  跑旧数据版本...")
    r_old = run_rotation_backtest(panel_old, DEFAULT_POOL, BacktestConfig(rotation=rot, freq="ME"))

    print("  跑新数据版本...")
    r_new = run_rotation_backtest(panel_new, DEFAULT_POOL, BacktestConfig(rotation=rot, freq="ME"))

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("同一策略 (v1 CICC baseline) 在新旧数据上的 NAV 对比",),
    )

    fig.add_trace(
        go.Scatter(
            x=r_old.nav.index, y=r_old.nav.values,
            mode="lines", name="旧数据 (2025-07-04 截止)",
            line=dict(color="#6c757d", width=2),
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=r_new.nav.index, y=r_new.nav.values,
            mode="lines", name="新数据 (2026-06-30 截止)",
            line=dict(color="#0d6efd", width=2),
        ),
    )

    # 标记 2025-07-04 (旧数据终点)
    fig.add_vline(
        x="2025-07-04", line_dash="dash", line_color="red",
        annotation_text="旧数据终点 2025-07-04",
        annotation_position="top",
    )

    fig.update_layout(
        title="数据版本影响: 同一策略在新旧数据上的 NAV",
        height=500, hovermode="x unified",
        template="plotly_white",
        yaxis_title="累计净值",
        xaxis_title="日期",
    )

    out = OUT_DIR / "data_version_impact.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)
    return str(out)


def main():
    print("=" * 60)
    print("生成 CICC vs Stage 12A 对比图表")
    print("=" * 60)

    nav_df = pd.read_parquet("reports/momentum_etf_rotation/docs/cicc_vs_stage12a_navs.parquet")

    print("\n[1/2] 主对比图...")
    p1 = make_main_comparison(nav_df)
    print(f"  -> {p1}")

    print("\n[2/2] 数据版本影响图...")
    p2 = make_data_version_impact()
    print(f"  -> {p2}")

    print("\n所有图表生成完成!")


if __name__ == "__main__":
    main()
