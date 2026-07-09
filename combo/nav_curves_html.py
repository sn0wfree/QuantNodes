# coding=utf-8
"""v1-v5 业绩曲线对比 HTML (纯 NAV 曲线, 轻量级).

聚焦: 8 个策略的 NAV 曲线对比
特点:
- 单一图表 8 策略叠加 (主曲线)
- 拆分面板: v1.0 演进路径 / 进攻型 v3+v5 / 风险型 v0.x
- OOS 区间高亮
- 关键事件标注 (2022 熊市, 2024 9月行情)
- 全期 + OOS 指标表
- 内联 plotly.js (脱机可用)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

REPO = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "reports/momentum_etf_rotation/combo"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTLY_JS = Path("/home/ll/.local/lib/python3.11/site-packages/plotly/package_data/plotly.min.js")

OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-06-30")

COLORS = {
    "v0.0 baseline":    "#888888",
    "v0.1 +VT":         "#FF7F0E",
    "v0.2 +TF":         "#D62728",
    "v1.0 locked":      "#2CA02C",
    "v3 (52 池)":       "#1F77B4",
    "v4 style":         "#9467BD",
    "v4 factor":        "#8C564B",
    "v5 量价":          "#E377C2",
}

STAGE_MAP = {
    "v0.0 baseline": "Stage 8 (原始 CICC 复现)",
    "v0.1 +VT":      "Stage 9-C (波动率目标)",
    "v0.2 +TF":      "Stage 9-B (趋势过滤)",
    "v1.0 locked":   "Stage 12A (斜率×R² 混合, v1.0 锁定)",
    "v3 (52 池)":    "Stage 16A (多策略组合)",
    "v4 style":      "Stage 18 (风格轮动)",
    "v4 factor":     "Stage 18 (IC 因子择时)",
    "v5 量价":       "Stage 22 (11 量价因子)",
}

# 分组
GROUPS = {
    "v1.0 演进路径": ["v0.0 baseline", "v0.1 +VT", "v0.2 +TF", "v1.0 locked"],
    "进攻型":       ["v3 (52 池)", "v5 量价", "v1.0 locked"],
    "风险型 (VT)":  ["v1.0 locked", "v0.1 +VT"],
    "全部 8 策略":  None,  # 全部
}


# ============================================================
# 工具
# ============================================================
def ann_return(nav):
    valid = nav.dropna()
    if len(valid) < 2:
        return 0.0
    r = valid.iloc[-1] / valid.iloc[0]
    n = (valid.index[-1] - valid.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav):
    valid = nav.dropna()
    if len(valid) < 2:
        return 0.0
    return float((valid / valid.cummax() - 1.0).min())


def sharpe(nav):
    rets = nav.pct_change().dropna()
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def metrics(nav):
    valid = nav.dropna()
    if len(valid) < 2:
        return {"ann_return": 0, "ann_vol": 0, "sharpe": 0,
                "max_dd": 0, "calmar": 0, "final": 0}
    rets = valid.pct_change().dropna()
    ar = ann_return(valid)
    dd = max_dd(valid)
    return {
        "ann_return": ar,
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "sharpe": sharpe(valid),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
        "final": float(valid.iloc[-1]),
    }


# ============================================================
# 业绩曲线 (主图: 全部 8 策略叠加)
# ============================================================
def chart_all_curves(navs):
    fig = go.Figure()

    # HS300 基准 (深灰虚线, 第一个画)
    if "HS300 基准" in navs.columns:
        valid = navs["HS300 基准"].dropna()
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines", name="HS300 基准",
            line=dict(color="#333333", width=2, dash="dashdot"),
            hovertemplate="<b>HS300 基准</b><br>%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ))

    # 先画非重点 (淡色, 在背景)
    background = ["v4 style", "v4 factor", "v0.0 baseline", "v0.2 +TF"]
    for col in background:
        if col not in navs.columns or col == "HS300 基准":
            continue
        valid = navs[col].dropna()
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines", name=col,
            line=dict(color=COLORS.get(col, "#aaa"), width=1.2, dash="dot"),
            opacity=0.45,
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>NAV=%{{y:.3f}}<extra></extra>",
        ))

    # 重点策略 (实线)
    foreground = ["v1.0 locked", "v3 (52 池)", "v5 量价", "v0.1 +VT"]
    for col in foreground:
        if col not in navs.columns or col == "HS300 基准":
            continue
        valid = navs[col].dropna()
        is_best = (col == "v1.0 locked")
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines", name=f"⭐ {col}" if is_best else col,
            line=dict(color=COLORS.get(col, "#333"), width=2.2 if is_best else 1.6,
                      shape="spline", smoothing=0.6),
            hovertemplate=f"<b>{col}</b> ({STAGE_MAP.get(col, '')})<br>"
                          "%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ))

    # OOS 区间高亮
    fig.add_vrect(
        x0=OOS_START, x1=OOS_END,
        fillcolor="rgba(100, 100, 200, 0.06)",
        line_width=0, annotation_text="OOS 2022-2026",
        annotation_position="top left",
        annotation_font_size=10,
    )

    # 关键事件标注
    fig.add_vline(x=pd.Timestamp("2024-09-23"),
                  line_dash="dot", line_color="#FF6B6B", line_width=1.2,
                  annotation_text="2024-09 政策", annotation_position="top right",
                  annotation_font_size=9, annotation_font_color="#FF6B6B")
    fig.add_vline(x=pd.Timestamp("2022-04-26"),
                  line_dash="dot", line_color="#888", line_width=1,
                  annotation_text="2022 熊市", annotation_position="bottom right",
                  annotation_font_size=9, annotation_font_color="#888")

    fig.update_layout(
        title=dict(
            text="<b>v1-v5 业绩曲线对比 (2018-2026)</b>"
                 "<br><sub>实线=重点, 虚线=参考 | ⭐=v1.0 locked (OOS Calmar 1.791) | "
                 "深灰虚线=HS300 基准 | 高亮=OOS 区间</sub>",
            x=0.02, xanchor="left", font=dict(size=15),
        ),
        xaxis=dict(
            title="日期", showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            gridwidth=1, zeroline=False,
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="NAV (起点=1.0)", showgrid=True,
            gridcolor="rgba(0,0,0,0.06)", gridwidth=1, zeroline=False,
        ),
        template="plotly_white",
        height=520,
        margin=dict(l=70, r=40, t=90, b=70),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
    )
    return fig


# ============================================================
# 分组对比 (subplot grid)
# ============================================================
def chart_grouped_curves(navs):
    """2x2 subplot: v1.0 演进 / 进攻型 / 风险型 / 全部对比."""
    panels = [
        ("v1.0 演进 (Stage 8 → v1.0)",
         ["v0.0 baseline", "v0.1 +VT", "v0.2 +TF", "v1.0 locked"]),
        ("进攻型 (v3 / v5 / v1.0 / HS300)",
         ["v3 (52 池)", "v5 量价", "v1.0 locked", "HS300 基准"]),
        ("风险型 (VT 类, 含基准)",
         ["v0.1 +VT", "v1.0 locked", "HS300 基准"]),
        ("Top-3 策略 vs 基准",
         ["v1.0 locked", "v3 (52 池)", "v5 量价", "HS300 基准"]),
    ]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[p[0] for p in panels],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    for i, (title, cols) in enumerate(panels, 1):
        row = (i - 1) // 2 + 1
        col_idx = (i - 1) % 2 + 1
        for col in cols:
            if col not in navs.columns:
                continue
            valid = navs[col].dropna()
            is_best = (col == "v1.0 locked")
            is_bench = (col == "HS300 基准")
            line_dash = "dashdot" if is_bench else "solid"
            line_width = 1.8 if is_bench else (2.5 if is_best else 1.6)
            line_color = "#333333" if is_bench else COLORS.get(col, "#333")
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid.values,
                mode="lines", name=col, showlegend=(i == 1),
                line=dict(color=line_color, width=line_width, dash=line_dash,
                          shape="spline", smoothing=0.5),
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>NAV=%{{y:.3f}}<extra></extra>",
            ), row=row, col=col_idx)
        # OOS 区间
        fig.add_vrect(x0=OOS_START, x1=OOS_END,
                      fillcolor="rgba(100,100,200,0.06)", line_width=0,
                      row=row, col=col_idx)

    fig.update_layout(
        title=dict(text="<b>分组业绩曲线 (2×2 网格)</b>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        template="plotly_white",
        height=720,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
    )
    for r in range(1, 3):
        for c in range(1, 3):
            fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                             zeroline=False, row=r, col=c)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.05)",
                             zeroline=False, row=r, col=c)
    return fig


# ============================================================
# 基准超额收益 (Alpha 曲线)
# ============================================================
def chart_alpha_curves(navs):
    """策略 NAV / HS300 基准 NAV, 计算超额收益."""
    if "HS300 基准" not in navs.columns:
        return None
    bench = navs["HS300 基准"]

    fig = go.Figure()
    for col in ["v1.0 locked", "v3 (52 池)", "v5 量价", "v0.1 +VT"]:
        if col not in navs.columns:
            continue
        s = navs[col].dropna()
        b = bench.reindex(s.index).dropna()
        common = s.index.intersection(b.index)
        if len(common) < 10:
            continue
        alpha = (s.loc[common] / b.loc[common] - 1.0) * 100
        is_best = (col == "v1.0 locked")
        fig.add_trace(go.Scatter(
            x=common, y=alpha.values,
            mode="lines", name=col,
            line=dict(color=COLORS.get(col, "#333"),
                      width=2.2 if is_best else 1.6,
                      shape="spline", smoothing=0.5),
            fill="tozeroy" if is_best else None,
            fillcolor="rgba(44,160,44,0.08)" if is_best else None,
            hovertemplate=f"<b>{col} 超额 HS300</b><br>"
                          "%{x|%Y-%m-%d}<br>α=%{y:+.2f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
    fig.add_vrect(x0=OOS_START, x1=OOS_END,
                  fillcolor="rgba(100,100,200,0.06)", line_width=0,
                  annotation_text="OOS", annotation_position="top left",
                  annotation_font_size=9)
    fig.update_layout(
        title=dict(text="<b>超额收益 (Alpha) vs HS300</b>"
                        "<br><sub>α = 策略 NAV / HS300 NAV − 1 (%)</sub>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        xaxis=dict(title="日期", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", zeroline=False),
        yaxis=dict(title="超额收益 (%)", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", zeroline=True,
                   zerolinecolor="rgba(0,0,0,0.3)"),
        template="plotly_white", height=460,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
    )
    return fig


# ============================================================
# 回撤对比
# ============================================================
def chart_drawdown_compare(navs):
    fig = go.Figure()
    # 基准先画 (在最底)
    if "HS300 基准" in navs.columns:
        valid = navs["HS300 基准"].dropna()
        dd = (valid / valid.cummax() - 1.0) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name="HS300 基准",
            line=dict(color="#333333", width=1.8, dash="dashdot"),
            hovertemplate="<b>HS300 基准</b><br>%{x|%Y-%m-%d}<br>DD=%{y:.2f}%<extra></extra>",
        ))
    for col in [c for c in navs.columns if c != "HS300 基准"]:
        valid = navs[col].dropna()
        if len(valid) < 2:
            continue
        dd = (valid / valid.cummax() - 1.0) * 100
        is_best = (col == "v1.0 locked")
        hex_c = COLORS.get(col, "#333").lstrip("#")
        rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name=col,
            line=dict(color=COLORS.get(col, "#333"),
                      width=1.6 if not is_best else 2,
                      shape="spline", smoothing=0.5),
            fill="tozeroy" if is_best else None,
            fillcolor=f"rgba{rgb + (0.10,)}" if is_best else None,
            opacity=0.85,
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>DD=%{{y:.2f}}%<extra></extra>",
        ))
    fig.add_vrect(x0=OOS_START, x1=OOS_END,
                  fillcolor="rgba(100,100,200,0.04)", line_width=0)
    fig.update_layout(
        title=dict(text="<b>回撤对比 (Drawdown Over Time)</b>"
                        "<br><sub>DD 越接近 0 越好 | v1.0 locked 用绿色填充 | HS300 基准用深灰虚线</sub>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        xaxis=dict(title="日期", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", zeroline=False),
        yaxis=dict(title="回撤 (%)", showgrid=True,
                   gridcolor="rgba(0,0,0,0.05)", zeroline=True,
                   zerolinecolor="rgba(0,0,0,0.3)"),
        template="plotly_white", height=480,
        margin=dict(l=60, r=30, t=80, b=60),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
    )
    return fig


# ============================================================
# 全期 vs OOS 收益对比
# ============================================================
def chart_period_compare(navs):
    full_metrics = {col: metrics(navs[col]) for col in navs.columns}
    oos = navs.loc[OOS_START:]
    oos_metrics = {col: metrics(oos[col]) for col in navs.columns}

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("全期 (2018-2026) 年化收益", "OOS (2022-2026) 年化收益"),
        horizontal_spacing=0.10,
    )
    for col in navs.columns:
        full_ret = full_metrics[col]["ann_return"] * 100
        oos_ret = oos_metrics[col]["ann_return"] * 100
        is_best = (col == "v1.0 locked")
        is_bench = (col == "HS300 基准")
        line_color = "#888888" if is_bench else COLORS.get(col, "#333")
        line_w = 1.5 if is_bench else (2 if is_best else 1)
        opacity = 0.6 if is_bench else 1
        fig.add_trace(go.Bar(
            x=[col], y=[full_ret], name=col, showlegend=False,
            marker=dict(
                color=line_color,
                line=dict(color="gold" if is_best else "rgba(0,0,0,0.3)",
                          width=2 if is_best else 0.5),
            ),
            opacity=opacity,
            text=[f"{full_ret:+.1f}%"], textposition="outside",
            textfont=dict(size=10),
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=[col], y=[oos_ret], name=col, showlegend=False,
            marker=dict(
                color=line_color,
                line=dict(color="gold" if is_best else "rgba(0,0,0,0.3)",
                          width=2 if is_best else 0.5),
            ),
            opacity=opacity,
            text=[f"{oos_ret:+.1f}%"], textposition="outside",
            textfont=dict(size=10),
        ), row=1, col=2)
    fig.update_yaxes(title_text="年化收益 (%)", row=1, col=1, gridcolor="rgba(0,0,0,0.05)")
    fig.update_yaxes(title_text="年化收益 (%)", row=1, col=2, gridcolor="rgba(0,0,0,0.05)")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>全期 vs OOS 年化收益</b>"
                        "<br><sub>金边=最佳 (v1.0 locked), 灰色=HS300 基准 | 柱顶=年化收益</sub>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        template="plotly_white", height=480,
        margin=dict(l=60, r=30, t=80, b=80), barmode="group",
    )
    return fig


# ============================================================
# 指标雷达图
# ============================================================
def chart_radar(navs):
    """8 维雷达图: 8 个策略的 6 维指标."""
    full_metrics = {col: metrics(navs[col]) for col in navs.columns}
    oos_metrics = {col: metrics(navs.loc[OOS_START:][col]) for col in navs.columns}

    categories = ["年化收益", "Sharpe", "Calmar", "1/|DD|", "低波动", "稳定性"]
    fig = go.Figure()

    for col in navs.columns:
        m = oos_metrics[col]
        # 归一化到 0-1
        ann_ret_norm = min(max(m["ann_return"] * 100, -10) / 20, 1.0)
        sharpe_norm = min(max(m["sharpe"], -1) / 3, 1.0)
        calmar_norm = min(max(m["calmar"], -0.5) / 2, 1.0)
        inv_dd_norm = min(10 / max(abs(m["max_dd"]) * 100, 1), 1.0)
        vol_norm = min(5 / max(m["ann_vol"] * 100, 1), 1.0)  # 越低越好
        stability_norm = inv_dd_norm  # 同 1/|DD|

        is_best = (col == "v1.0 locked")
        is_bench = (col == "HS300 基准")
        opacity = 0.35 if is_bench else (1.0 if is_best else 0.6)
        line_color = "#666666" if is_bench else COLORS.get(col, "#333")
        vals = [ann_ret_norm, sharpe_norm, calmar_norm, inv_dd_norm,
                vol_norm, stability_norm, ann_ret_norm]  # 闭合
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=categories + [categories[0]],
            mode="lines+markers", name=col,
            line=dict(color=line_color,
                      width=2.5 if is_best else 1.2,
                      dash="dashdot" if is_bench else "solid"),
            marker=dict(size=5 if is_best else 3),
            opacity=opacity,
            fill="toself" if is_best else None,
            fillcolor="rgba(44,160,44,0.10)" if is_best else None,
        ))
    fig.update_layout(
        title=dict(text="<b>OOS 性能雷达图 (2022-2026)</b>"
                        "<br><sub>6 维归一化指标, 越靠外越好 | ⭐=v1.0 locked (绿色填充)</sub>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        template="plotly_white", height=520,
        margin=dict(l=60, r=60, t=80, b=60),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(0,0,0,0.1)",
                            tickfont=dict(size=9)),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
        ),
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
    )
    return fig


# ============================================================
# 月度收益热图 (策略 × 月份)
# ============================================================
def chart_monthly_heatmap(navs):
    """2D 热图: 行=策略, 列=月份."""
    monthly_rets = []
    valid_cols = []
    for col in navs.columns:
        m = navs[col].resample("ME").last().pct_change().dropna()
        if len(m) < 12:
            continue
        monthly_rets.append(m)
        valid_cols.append(col)

    # 对齐到统一时间轴
    all_dates = sorted(set().union(*[set(m.index) for m in monthly_rets]))
    matrix = pd.DataFrame(index=valid_cols, columns=all_dates, dtype=float)
    for col, m in zip(valid_cols, monthly_rets):
        matrix.loc[col, m.index] = m.values * 100

    # 转成年-月标签
    ym_labels = [d.strftime("%Y-%m") for d in all_dates]
    fig = go.Figure(data=go.Heatmap(
        z=matrix.values, x=ym_labels, y=valid_cols,
        colorscale="RdYlGn", zmid=0, zmin=-15, zmax=15,
        colorbar=dict(title="月收益 %", thickness=15, len=0.85),
        hovertemplate="%{y}<br>%{x}<br>%{z:.2f}%<extra></extra>",
        xgap=1, ygap=2,
    ))
    fig.update_layout(
        title=dict(text="<b>月度收益热图 (2018-2026)</b>"
                        "<br><sub>红=亏损, 绿=盈利 | 列=月份, 行=策略</sub>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        template="plotly_white", height=480,
        margin=dict(l=120, r=80, t=80, b=80),
        xaxis=dict(title="月份", tickangle=-45, tickfont=dict(size=9),
                   showgrid=False),
        yaxis=dict(title="策略", tickfont=dict(size=11),
                   showgrid=False, autorange="reversed"),
    )
    return fig


# ============================================================
# Main: 生成完整 HTML
# ============================================================
def main():
    print("[curve] 加载数据...")
    navs_A = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")
    oos = navs_A.loc[OOS_START:]

    # 加载 HS300 基准
    print("[curve] 加载 HS300 基准...")
    nav_main = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    hs300_raw = nav_main["510300"].dropna()
    # 归一化到 1.0 起点
    hs300_nav = (hs300_raw / hs300_raw.iloc[0]).rename("HS300 基准")
    # 截取到与策略相同的日期范围
    hs300_nav = hs300_nav.loc[:OOS_END]
    navs_A_with_bench = navs_A.copy()
    navs_A_with_bench["HS300 基准"] = hs300_nav.reindex(navs_A.index)

    # 基准色 (灰色虚线)
    COLORS_BENCH = "#333333"

    print("[curve] 生成 7 个业绩图表...")
    figs = {
        "all_curves": chart_all_curves(navs_A_with_bench),
        "grouped": chart_grouped_curves(navs_A_with_bench),
        "alpha": chart_alpha_curves(navs_A_with_bench),
        "drawdown": chart_drawdown_compare(navs_A_with_bench),
        "period_compare": chart_period_compare(navs_A_with_bench),
        "radar": chart_radar(navs_A_with_bench),
        "monthly_heatmap": chart_monthly_heatmap(navs_A_with_bench),
    }

    # 嵌入 plotly.js
    if PLOTLY_JS.exists():
        plotly_src = PLOTLY_JS.read_text(encoding="utf-8")
    else:
        print("  [WARN] plotly.min.js not found, 使用 CDN")
        plotly_src = ""

    # 计算指标
    full_metrics = {col: metrics(navs_A_with_bench[col]) for col in navs_A_with_bench.columns}
    oos_metrics = {col: metrics(oos[col]) if col in oos.columns else full_metrics[col]
                   for col in navs_A_with_bench.columns}
    # 排序只对策略列 (排除 HS300 基准)
    strategy_cols = [c for c in navs_A_with_bench.columns if c != "HS300 基准"]
    oos_sorted = sorted(
        [(c, oos_metrics[c]) for c in strategy_cols],
        key=lambda x: x[1]["calmar"], reverse=True
    )
    # HS300 单独放最后
    hs300_oos = oos_metrics.get("HS300 基准", full_metrics.get("HS300 基准"))

    # 关键事件标签
    events = [
        (pd.Timestamp("2018-01-29"), "2018-01 春节前"),
        (pd.Timestamp("2020-03-23"), "2020-03 疫情底"),
        (pd.Timestamp("2022-04-26"), "2022-04 熊市底"),
        (pd.Timestamp("2022-10-31"), "2022-10 反弹"),
        (pd.Timestamp("2024-09-23"), "2024-09 政策"),
        (pd.Timestamp("2025-09-30"), "2025 Q3 末"),
    ]

    # 表格 HTML
    table_rows = []
    for rank, (col, om) in enumerate(oos_sorted, 1):
        fm = full_metrics[col]
        cls = "best" if rank == 1 else ""
        star = " ⭐" if rank == 1 else ""
        table_rows.append(f"""
        <tr class="{cls}">
          <td>{rank}</td>
          <td>{col}{star}</td>
          <td>{fm['ann_return']*100:+.2f}%</td>
          <td>{fm['ann_vol']*100:.2f}%</td>
          <td>{fm['calmar']:.3f}</td>
          <td>{om['ann_return']*100:+.2f}%</td>
          <td>{om['ann_vol']*100:.2f}%</td>
          <td>{om['sharpe']:.2f}</td>
          <td>{om['max_dd']*100:.2f}%</td>
          <td><b>{om['calmar']:.3f}</b></td>
        </tr>""")

    # HS300 基准行 (单独样式)
    if hs300_oos:
        fm = full_metrics.get("HS300 基准", hs300_oos)
        table_rows.append(f"""
        <tr class="benchmark">
          <td>—</td>
          <td>HS300 基准 📊</td>
          <td>{fm['ann_return']*100:+.2f}%</td>
          <td>{fm['ann_vol']*100:.2f}%</td>
          <td>{fm['calmar']:.3f}</td>
          <td>{hs300_oos['ann_return']*100:+.2f}%</td>
          <td>{hs300_oos['ann_vol']*100:.2f}%</td>
          <td>{hs300_oos['sharpe']:.2f}</td>
          <td>{hs300_oos['max_dd']*100:.2f}%</td>
          <td>{hs300_oos['calmar']:.3f}</td>
        </tr>""")

    # 子图
    sections = []
    for key, fig in figs.items():
        sections.append(f"""
        <section id="{key}">
          <h2>{fig.layout.title.text.split('<')[0].strip() if fig.layout.title else key}</h2>
          {fig.to_html(full_html=False, include_plotlyjs=False, div_id=key)}
        </section>""")

    # 全期 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>v1-v5 业绩曲线对比</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1280px; margin: 20px auto; padding: 0 20px; background: #fafafa; color: #2C3E50; }}
h1 {{ color: #1F77B4; border-bottom: 3px solid #1F77B4; padding-bottom: 8px; margin-top: 0; }}
h2 {{ color: #2C3E50; margin-top: 40px; border-left: 5px solid #1F77B4; padding-left: 10px; }}
section {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 14px; }}
th {{ background: #1F77B4; color: white; padding: 10px 8px; text-align: center; font-weight: 600; }}
td {{ padding: 8px; text-align: center; border-bottom: 1px solid #e0e0e0; }}
tr.best {{ background: #FFF9E6; font-weight: bold; }}
tr.best td {{ background: #FFF9E6; }}
tr.benchmark {{ background: #F0F0F0; color: #555; font-style: italic; border-top: 2px solid #999; }}
tr.benchmark td {{ background: #F0F0F0 !important; }}
tr:hover:not(.best):not(.benchmark) {{ background: #F5F5F5; }}
.events {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin: 12px 0; }}
.event {{ background: #F0F4F8; padding: 6px 10px; border-radius: 4px; font-size: 12px; border-left: 3px solid #1F77B4; }}
.event-date {{ font-weight: bold; color: #1F77B4; }}
.key-finding {{ background: linear-gradient(135deg, #E8F5E9 0%, #FFF9E6 100%); padding: 16px; border-radius: 8px; margin: 12px 0; border-left: 5px solid #2CA02C; }}
.navbar {{ position: sticky; top: 0; background: rgba(255,255,255,0.95); padding: 12px; border-bottom: 1px solid #ddd; z-index: 100; backdrop-filter: blur(10px); }}
.navbar a {{ margin-right: 14px; color: #1F77B4; text-decoration: none; font-size: 14px; padding: 4px 8px; border-radius: 4px; }}
.navbar a:hover {{ background: #E8F5E9; text-decoration: none; }}
.navbar a.active {{ background: #1F77B4; color: white; }}
.color-tag {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
</style>
</head>
<body>

<h1>v1-v5 业绩曲线对比</h1>

<div class="navbar">
  <a href="#all_curves">主曲线</a>
  <a href="#grouped">分组曲线</a>
  <a href="#alpha">α 超额</a>
  <a href="#drawdown">回撤对比</a>
  <a href="#period_compare">全期 vs OOS</a>
  <a href="#radar">雷达图</a>
  <a href="#monthly_heatmap">月度热图</a>
  <a href="#metrics_table">指标表</a>
  <a href="#events">关键事件</a>
</div>

<div class="key-finding">
  <strong>核心发现 (口径 A OOS 2022-2026, 含 HS300 基准):</strong><br>
  • <b>风险调整冠军</b>: v1.0 locked — OOS Calmar <b>1.791</b>, Sharpe <b>1.51</b>, DD -1.94%<br>
  • <b>绝对收益冠军</b>: v5 量价 — OOS 年化 <b>9.47%</b> (HS300 同期 ~ -8%, α 显著)<br>
  • <b>最均衡</b>: v3 (52 池) — OOS 年化 7.69%, Sharpe 1.08, DD -9.89%<br>
  • <b>HS300 基准</b>: OOS 年化 {hs300_oos['ann_return']*100:+.2f}%, DD {hs300_oos['max_dd']*100:.2f}%, Calmar {hs300_oos['calmar']:.3f}<br>
  • <b>推荐组合</b>: v1.0 80% + v5 20% — 全期 Calmar 1.079, OOS 0.886
</div>

<div class="events">
"""

    for date, label in events:
        html += f'  <div class="event"><span class="event-date">{date.strftime("%Y-%m-%d")}</span> {label}</div>\n'

    html += f"""</div>

<section id="metrics_table">
  <h2>8 策略 OOS 业绩表 (按 Calmar 排序)</h2>
  <table>
    <tr>
      <th rowspan="2">排名</th>
      <th rowspan="2">策略</th>
      <th colspan="3">全期 (2018-2026)</th>
      <th colspan="5">OOS (2022-2026)</th>
    </tr>
    <tr>
      <th>年化</th><th>年化波动</th><th>Calmar</th>
      <th>年化</th><th>年化波动</th><th>Sharpe</th><th>最大回撤</th><th>Calmar</th>
    </tr>
    {"".join(table_rows)}
  </table>
</section>

{"".join(sections)}

<section id="events">
  <h2>关键事件时间线</h2>
  <table>
    <tr><th>日期</th><th>事件</th><th>对策略的影响</th></tr>
    <tr><td>2018-01-29</td><td>2018 春节前</td><td>v3 / v4 启动期, v1.0 早期动量信号</td></tr>
    <tr><td>2020-03-23</td><td>疫情底</td><td>v5 量价 反弹后 +40% 收益</td></tr>
    <tr><td>2022-04-26</td><td>2022 熊市底</td><td>v4 因子 -23.59%, v1.0 VT 保护 -0.71%</td></tr>
    <tr><td>2024-09-23</td><td>政策利好</td><td>v5 量价 +35% / 2024, v3 +12%</td></tr>
    <tr><td>2025-09-30</td><td>Q3 末</td><td>v1.0 +5.26%, v5 +32.29% (2025 最佳)</td></tr>
  </table>
  <p style="font-size:12px;color:#666;margin-top:20px;">
  <b>注</b>: 表格中"全期"和"OOS"分别指 2018-2026 和 2022-2026 的回测区间。
  数据源: <code>combo/unified_v1v5_compare.py</code> (口径 A: 52 ETF 池 + 5bp 单边成本 + CICC cap=3)。
  </p>
</section>

</body>
</html>"""

    # 嵌入 plotly.js
    if plotly_src:
        html = html.replace("</head>", f'<script>{plotly_src}</script></head>')

    out_html = OUT_DIR / "V1V5_NAV_CURVES.html"
    out_html.write_text(html, encoding="utf-8")
    size_mb = out_html.stat().st_size / 1024 / 1024
    print(f"\n[save] {out_html} ({size_mb:.2f} MB)")
    print(f"      7 个图表: 主曲线 / 分组 / α超额 / 回撤 / 全期对比 / 雷达 / 月度热图")
    print(f"      指标表 (含 HS300 基准行) + 关键事件时间线")


if __name__ == "__main__":
    main()
