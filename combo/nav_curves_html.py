# coding=utf-8
"""v1-v7.10 业绩曲线对比 HTML (纯 NAV 曲线, 轻量级).

聚焦: 13 个策略 + HS300 基准的 NAV 曲线对比 (v0.0 → v7.10 TV-PR)
特点:
- 单一图表多策略叠加 (主曲线)
- 拆分面板: v1.0 演进路径 / 进攻型 v3+v5 / 风险型 v0.x
- OOS 区间高亮
- 关键事件标注 (2022 熊市, 2024 9月行情)
- 全期 + OOS 指标表
- 13 个策略简述 + HS300 基准 (v0 - v7.10 演进轨迹, 主文件含, 报告版省略)
- 内联 plotly.js (脱机可用)
- 双输出: STRATEGY_ITERATION_RECORD.html (主文件, 内部跟踪) + STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html (报告版, 对外汇报)
"""
from __future__ import annotations

import os
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

# plotly 兼容 (避免 pandas2.x Timestamp+int 报错): 用毫秒整数
OOS_START_MS = OOS_START.value // 10**6
OOS_END_MS = OOS_END.value // 10**6
OOS_2024_MS = pd.Timestamp("2024-09-23").value // 10**6
OOS_2022_MS = pd.Timestamp("2022-04-26").value // 10**6

COLORS = {
    "v0.0 baseline":    "#888888",
    "v0.1 +VT":         "#FF7F0E",
    "v0.2 +TF":         "#D62728",
    "v1.0 locked":      "#2CA02C",
    "v3 (52 池)":       "#1F77B4",
    "v4 style":         "#9467BD",
    "v4 factor":        "#8C564B",
    "v5 量价":          "#E377C2",
    "v5.1 量价 (逆波动)": "#FF6B9D",
    "v6 (TF 趋势过滤)": "#17BECF",  # 青色 — Stage 26 v6 单策略风控
    "v6 TF+Cost":      "#17BECF",
    "v6.1 IC12":       "#D62728",  # 深红 — Stage 27 v6.1 IC 加权 (RECOMMENDED)
    "v6.2 ir_expanding": "#9467BD",  # 紫 — Stage 29 v6.2 因子正交化 (PROMISING)
    "v7.10 TV-PR (标准化+CV)":      "#FF4500",  # 橙红 — v7.10 TV-PR (标准化+CV) (Cui 2025, 时变 β_t)
}

STAGE_MAP = {
    "v0.0 baseline":    "Stage 8 (原始 CICC 复现)",
    "v0.1 +VT":         "Stage 9-C (波动率目标)",
    "v0.2 +TF":         "Stage 9-B (趋势过滤)",
    "v1.0 locked":      "Stage 12A (斜率×R² 混合, v1.0 锁定)",
    "v3 (52 池)":       "Stage 16A (多策略组合)",
    "v4 style":         "Stage 18 (风格轮动)",
    "v4 factor":        "Stage 18 (IC 因子择时)",
    "v5 量价":          "Stage 22 (11 量价因子, 等权)",
    "v5.1 量价 (逆波动)": "Stage 25.1 (v5.1 升级, S1+S3+S4 消融选中)",
    "v6 (TF 趋势过滤)": "Stage 26 (v6 单策略: TF 风控 + v5 因子排序 + v5.1 逆波动加权)",
    "v6 TF+Cost":       "Stage 26 (v6 单策略: TF 风控 + 调仓成本 + v5.1.1)",
    "v6.1 IC12":       "Stage 27 (v6.1: IC-IR 加权, IC12 expanding, RECOMMENDED)",
    "v6.2 ir_expanding": "Stage 29 (v6.2: 因子正交化 + ir_expanding + IC12, ⭐ OOS Calmar 0.821, 5-fold 4/5 胜 v6.1)",
    "v7.10 TV-PR (标准化+CV)":       "v7.10 (TV-PR 标准化+CV: 17 macro + 19 量价, 时变 β_t, 🚀 Stage 31 NEW, ⭐ OOS Calmar 2.144)",
}

# 分组
GROUPS = {
    "v1.0 演进路径": ["v0.0 baseline", "v0.1 +VT", "v0.2 +TF", "v1.0 locked"],
    "进攻型":       ["v3 (52 池)", "v5 量价", "v5.1 量价 (逆波动)", "v6 (TF 趋势过滤)", "v6.1 IC12", "v6.2 ir_expanding", "v7.10 TV-PR (标准化+CV)", "v1.0 locked"],
    "量价族演进 (Stage 22→29)":  ["v5 量价", "v5.1 量价 (逆波动)", "v6 (TF 趋势过滤)", "v6.1 IC12", "v6.2 ir_expanding"],
    "v7.10 TV-PR (标准化+CV) 对比":  ["v1.0 locked", "v6.2 ir_expanding", "v7.10 TV-PR (标准化+CV)"],
    "风险型 (VT)":  ["v1.0 locked", "v0.1 +VT"],
    "全部 13 策略":  None,  # 全部
}


# ============================================================
# 工具
# ============================================================
def ann_return(nav):
    """年化收益 = (终值/初值) ^ (252 / 实际交易日数) - 1 (trading days 口径).

    与 Stage 27 v6_2 ablation 的 (prod(1+r))^(252/n) - 1 一致.
    实际交易日数取 len(valid) - 1 (即两个端点之间的 trading day 数).
    """
    valid = nav.dropna()
    if len(valid) < 2:
        return 0.0
    r = valid.iloc[-1] / valid.iloc[0]
    n = len(valid) - 1  # 实际 trading days 区间长度
    return float(r ** (252 / n) - 1) if n > 0 else 0.0


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


def _extract_chart_title(figure) -> str:
    """从 plotly figure 提取主标题 (处理 <b>...</b><br><sub>...</sub> 格式).

    Args:
        figure: plotly figure 对象.

    Returns:
        主标题字符串 (HTML 已剥离), 无标题时返回 fig key 占位.
    """
    if not figure.layout.title or not figure.layout.title.text:
        return ""
    text = figure.layout.title.text
    import re
    m = re.search(r"<b>(.*?)</b>", text)
    if m:
        return m.group(1).strip()
    if "<" in text:
        return text.split("<")[0].strip()
    return text.strip()


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
    print(f"  [chart 1/7] all_curves...", flush=True)
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
    foreground = [
        "v1.0 locked", "v3 (52 池)", "v5 量价", "v5.1 量价 (逆波动)",
        "v6 (TF 趋势过滤)", "v6.1 IC12", "v6.2 ir_expanding", "v7.10 TV-PR (标准化+CV)",
    ]
    for col in foreground:
        if col not in navs.columns or col == "HS300 基准":
            continue
        valid = navs[col].dropna()
        is_best = (col == "v1.0 locked")
        is_v51 = (col == "v5.1 量价 (逆波动)")
        is_v6 = (col == "v6 (TF 趋势过滤)")
        is_v62 = (col == "v6.2 ir_expanding")
        is_v710 = (col == "v7.10 TV-PR (标准化+CV)")
        prefix = "⭐ " if is_best else ("🆕 " if is_v51 or is_v6 else ("🚀 " if is_v710 else ""))
        display_name = f"{prefix}{col}"
        width = 2.6 if is_v710 else (2.4 if is_best else (2.0 if is_v6 or is_v62 else (1.9 if is_v51 else 1.7)))
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines", name=display_name,
            line=dict(color=COLORS.get(col, "#333"),
                      width=width,
                      shape="spline", smoothing=0.6),
            hovertemplate=f"<b>{col}</b> ({STAGE_MAP.get(col, '')})<br>"
                          "%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ))

    # OOS 区间高亮 (plotly 兼容: 用 ms 整数代替 Timestamp, 避免 pandas2.x Timestamp+int 报错)
    fig.add_vrect(
        x0=OOS_START_MS, x1=OOS_END_MS,
        fillcolor="rgba(100, 100, 200, 0.06)",
        line_width=0, annotation_text="OOS 2022-2026",
        annotation_position="top left",
        annotation_font_size=10,
    )

    # 关键事件标注
    fig.add_vline(x=OOS_2024_MS,
                  line_dash="dot", line_color="#FF6B6B", line_width=1.2,
                  annotation_text="2024-09 政策", annotation_position="top right",
                  annotation_font_size=9, annotation_font_color="#FF6B6B")
    fig.add_vline(x=OOS_2022_MS,
                  line_dash="dot", line_color="#888", line_width=1,
                  annotation_text="2022 熊市", annotation_position="bottom right",
                  annotation_font_size=9, annotation_font_color="#888")

    fig.update_layout(
        title=dict(
            text="<b>v0 - v7.10 业绩曲线对比 (2018-2026, 13 策略 + HS300)</b>"
                 "<br><sub>实线=重点, 虚线=参考 | ⭐=v1.0 locked (历史 OOS 最佳 1.791) | "
                 "🚀=v7.10 TV-PR (Stage 31 NEW, OOS Calmar 2.144) | 深灰虚线=HS300 基准 | 高亮=OOS 区间</sub>",
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
    """2x2 subplot: v1.0 演进 / 进攻型 / 风险型 / 量价族演进."""
    print(f"  [chart 2/7] grouped...", flush=True)
    panels = [
        ("v1.0 演进 (Stage 8 → v1.0)",
         ["v0.0 baseline", "v0.1 +VT", "v0.2 +TF", "v1.0 locked"]),
        ("进攻型 (v3 / v5 / v5.1 / v6.2 / v1.0 / HS300)",
         ["v3 (52 池)", "v5 量价", "v5.1 量价 (逆波动)", "v6.2 ir_expanding", "v7.10 TV-PR (标准化+CV)", "v1.0 locked", "HS300 基准"]),
        ("v5 → v5.1 → v6.1 → v6.2 量价族演进",
         ["v5 量价", "v5.1 量价 (逆波动)", "v6.1 IC12", "v6.2 ir_expanding"]),
        ("v5.1 vs v6 vs v6.1 vs v6.2 加权演进 (Stage 23→29)",
         ["v5 量价", "v5.1 量价 (逆波动)", "v6 (TF 趋势过滤)", "v6.1 IC12", "v6.2 ir_expanding", "v7.10 TV-PR (标准化+CV)", "HS300 基准"]),
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
        fig.add_vrect(x0=OOS_START_MS, x1=OOS_END_MS,
                      fillcolor="rgba(100,100,200,0.06)", line_width=0,
                      row=row, col=col_idx)

    fig.update_layout(
        title=dict(text="<b>分组业绩曲线 (2×2 网格, 13 策略分组对比)</b>",
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
    print(f"  [chart 3/7] alpha...", flush=True)
    if "HS300 基准" not in navs.columns:
        return None
    bench = navs["HS300 基准"]

    fig = go.Figure()
    for col in [
        "v1.0 locked", "v3 (52 池)", "v5 量价", "v5.1 量价 (逆波动)", "v6.2 ir_expanding", "v7.10 TV-PR (标准化+CV)",
    ]:
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
    fig.add_vrect(x0=OOS_START_MS, x1=OOS_END_MS,
                  fillcolor="rgba(100,100,200,0.06)", line_width=0,
                  annotation_text="OOS", annotation_position="top left",
                  annotation_font_size=9)
    fig.update_layout(
        title=dict(text="<b>超额收益 (Alpha) vs HS300 (v0 - v7.10)</b>"
                        "<br><sub>α = 策略 NAV / HS300 NAV − 1 (%). 重点策略在 OOS 期间相对 HS300 跑赢幅度</sub>",
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
    print(f"  [chart 4/7] drawdown...", flush=True)
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
    fig.add_vrect(x0=OOS_START_MS, x1=OOS_END_MS,
                  fillcolor="rgba(100,100,200,0.04)", line_width=0)
    fig.update_layout(
        title=dict(text="<b>回撤对比 (Drawdown Over Time, v0 - v7.10)</b>"
                        "<br><sub>DD 越接近 0 越好 | v1.0 locked 几乎无回撤 | v6.2 ir_expanding DD -16.7% | v7.10 TV-PR (标准化+CV) DD -14.3%</sub>",
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
    print(f"  [chart 5/7] period_compare...", flush=True)
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
        title=dict(text="<b>全期 vs OOS 年化收益 (v0 - v7.10)</b>"
                        "<br><sub>金边=最佳 (v1.0 OOS Calmar 1.791), 灰色=HS300 基准 | 柱顶=年化收益</sub>",
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
    print(f"  [chart 6/7] radar...", flush=True)
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
        title=dict(text="<b>OOS 性能雷达图 (2022-2026, v0 - v7.10)</b>"
                        "<br><sub>6 维归一化指标, 越靠外越好 | ⭐=v1.0 locked (绿色填充, Calmar 1.791) | v6.2 ir_expanding 紫色 (0.821) | 🚀 v7.10 TV-PR (标准化+CV) 橙红 (2.144)</sub>",
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
    print(f"  [chart 7/7] monthly_heatmap...", flush=True)
    monthly_rets = []
    valid_cols = []
    for col in navs.columns:
        m = navs[col].resample("M").last().pct_change().dropna()
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
        title=dict(text="<b>月度收益热图 (2018-2026, v0 - v7.10 共 13 策略)</b>"
                        "<br><sub>红=亏损, 绿=盈利 | 列=月份, 行=策略 | v1.0 全期基本无红格</sub>",
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
def main(include_strategies: bool = True):
    """生成 STRATEGY_ITERATION_RECORD.html 主文件 (内部跟踪, 完整版) 和 STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html 报告版.

    Args:
        include_strategies: True = 完整版 (含策略简述, 内部跟踪用)
                            False = 简化版 (无策略简述, 对外汇报用)
    """
    print(f"[curve] include_strategies={include_strategies}", flush=True)
    print("[curve] 加载数据...", flush=True)
    navs_A = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")
    print(f"  [data] parquet: {navs_A.shape}, cols={len(navs_A.columns)}, "
          f"{navs_A.index[0].date()}~{navs_A.index[-1].date()}", flush=True)
    oos = navs_A.loc[OOS_START:]  # DataFrame (用于表格)
    oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}  # dict (用于显示)

    # 加载 v6 NAV (Stage 26: v6 单策略, 推荐 TF 档)
    v6_path = OUT_DIR / "v6_navs.parquet"
    if v6_path.exists():
        v6_navs = pd.read_parquet(v6_path)
        # 推荐档: TF (OOS Calmar 0.662)
        if "v6 只 TF" in v6_navs.columns:
            navs_A["v6 (TF 趋势过滤)"] = v6_navs["v6 只 TF"]
            print(f"[curve] v6 TF 已加入 (Stage 26): OOS Calmar="
                  f"{metrics(v6_navs['v6 只 TF'].loc[OOS_START:])['calmar']:.3f}")
        # TF+Cost 档
        if "v6 TF+Cost" in v6_navs.columns:
            navs_A["v6 TF+Cost"] = v6_navs["v6 TF+Cost"]
        # 重建 oos_metrics 含 v6 (后续 main() 还会再算, 这里先设保险)
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v6.1 / v6.2 combined NAV (Stage 27 + Stage 29 更新)
    v6_combined_path = OUT_DIR / "v6_1_v6_2_combined_navs.parquet"
    if v6_combined_path.exists():
        v6_combined = pd.read_parquet(v6_combined_path)
        if "v6.1 IC12" in v6_combined.columns:
            navs_A["v6.1 IC12"] = v6_combined["v6.1 IC12"]
            print(f"[curve] v6.1 IC12 已加入 (Stage 27 RECOMMENDED): OOS Calmar="
                  f"{metrics(v6_combined['v6.1 IC12'].loc[OOS_START:])['calmar']:.3f}")
        if "v6.2 ir_expanding" in v6_combined.columns:
            navs_A["v6.2 ir_expanding"] = v6_combined["v6.2 ir_expanding"]
            print(f"[curve] v6.2 ir_expanding 已加入 (Stage 29 PROMISING): OOS Calmar="
                  f"{metrics(v6_combined['v6.2 ir_expanding'].loc[OOS_START:])['calmar']:.3f}")
        elif "v6.2 (正交+IC36)" in v6_combined.columns:
            # 兼容旧列名
            navs_A["v6.2 ir_expanding"] = v6_combined["v6.2 (正交+IC36)"]
            print(f"[curve] v6.2 (旧列名兼容) 已加入: OOS Calmar="
                  f"{metrics(v6_combined['v6.2 (正交+IC36)'].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 构造 v6 Key Finding HTML 片段
    v6_tf_oos = oos_metrics.get('v6 (TF 趋势过滤)', {})
    v51_oos = oos_metrics.get('v5.1 量价 (逆波动)', {})
    if v6_tf_oos:
        v6_delta_pct = (v6_tf_oos['calmar'] - v51_oos['calmar']) / v51_oos['calmar'] * 100
        v6_dd_v51 = abs(v51_oos['max_dd'])*100
        v6_dd_v6 = abs(v6_tf_oos['max_dd'])*100
        v6_key_finding = (
            f"• <b>v6 (TF 趋势过滤) Stage 26</b>: v5 因子排序 + v5.1 逆波动加权 + TF 风控 → "
            f"OOS Calmar <b>{v6_tf_oos['calmar']:.3f}</b> (+{v6_delta_pct:.1f}%), "
            f"DD -<b>{v6_dd_v6:.1f}%</b> "
            f"(vs v5.1.1 -{v6_dd_v51:.1f}%)<br>"
        )
        v6_aggressive_row = (
            f"<tr><td>🚀 进取 (TF 风控)</td><td>v6 (TF 趋势过滤) 🆕🆕</td>"
            f"<td>v5.1.1 + TF 风控, DD 仅 <b>-{v6_dd_v6:.1f}%</b> "
            f"(vs v5.1.1 -{v6_dd_v51:.1f}%)</td>"
            f"<td><b>{v6_tf_oos['calmar']:.3f}</b></td></tr>"
        )
        v6_dd_improve = v6_dd_v51 - v6_dd_v6
        v6_strategy_card = f"""
  <div class="strategy-card" style="background: #E0F7FA; border-color: #17BECF;">
    <h4>v6 (TF 趋势过滤) <span class="legend-box legend-best">⭐ Stage 26 单策略</span></h4>
    <p><b>类型</b>: <b>v5 因子排序</b> + v5.1 逆波动加权 + <b>TF 风控</b> (HS300 MA200, bear=0.7)</p>
    <p><b>核心</b>: v6 单策略版, 排序逻辑沿用 v5 (截面 z-score + 复合因子 Top-N), 加权沿用 v5.1 逆波动, 仅加 1 层风控 (TF). TF 在 HS300 < MA200 时缩仓至 70%, 留 30% 给 511260 国债 ETF.</p>
    <p><b>OOS</b>: {v6_tf_oos['ann_return']*100:.2f}% / Sharpe {v6_tf_oos['sharpe']:.2f} / DD {v6_dd_v6:.2f}% / <b>Calmar {v6_tf_oos['calmar']:.3f}</b> ⭐</p>
    <p><b>v5.1.1 → v6 TF 改善</b>: OOS Calmar +{v6_delta_pct:.1f}%, DD 改善 {v6_dd_improve:.2f}pp</p>
    <p><b>设计动机</b>: v5.1.1 无风控高 beta 收益, 单月最大回撤 -18%. 加 TF 让熊市只亏 70%, 大幅降 DD. 不开 VT 是因为 v5.1.1 信号源已经偏进攻, VT 会严重拖累收益 (消融实验证实).</p>
  </div>
"""
    else:
        v6_key_finding = ""
        v6_aggressive_row = ""
        v6_strategy_card = ""
    # 删除冗余: oos_metrics 和 full_metrics_v6 在 main 后面重算

    # v6.1 / v6.2 Key Finding + 策略卡 (Stage 27 + Stage 29)
    v62_oos = oos_metrics.get('v6.2 ir_expanding', oos_metrics.get('v6.2 (正交+IC36)', {}))
    v710_oos = oos_metrics.get('v7.10 TV-PR (标准化+CV)', {})
    v61_oos = oos_metrics.get('v6.1 IC12', {})
    if v62_oos and v61_oos:
        v62_cal = v62_oos['calmar']
        v61_cal = v61_oos['calmar']
        v51_cal = oos_metrics.get('v5.1 量价 (逆波动)', {}).get('calmar', 0)
        v62_dd_pct = abs(v62_oos['max_dd']) * 100
        v61_dd_pct = abs(v61_oos['max_dd']) * 100
        v62_key_finding = (
            f"• <b>v6.2 ir_expanding Stage 29 ⭐ PROMISING</b>: v5 因子排序 + IC 加权 + 因子正交化 (Gram-Schmidt, expanding IR 排序) → "
            f"OOS Calmar <b>{v62_oos.get('calmar', 0):.3f}</b> "
            f"<br>• <b>v7.10 TV-PR (标准化+CV) 🚀 Stage 31 NEW</b>: 17 macro + 19 量价, 时变 β_t (Cui 2025), 混合标准化 + 两阶段 CV + expanding window → "
            f"OOS Calmar <b>{v710_oos.get('calmar', 0):.3f}</b> "
            f"<br>• <b>v7.10 vs v6.2</b>: v7.10 OOS Calmar {v710_oos.get('calmar', 0):.3f} vs v6.2 {v62_oos.get('calmar', 0):.3f} "
            f"({'✅ 优于' if v710_oos.get('calmar', 0) > v62_oos.get('calmar', 0) else '⚠️ 待优化'})"
            f"OOS Calmar <b>{v62_cal:.3f}</b> (vs v6.1 IC12 {v61_cal:.3f}, +{(v62_cal/v61_cal-1)*100:+.1f}%) "
            f"5-fold walk-forward <b>4/5 胜 v6.1</b>, DD -{v62_dd_pct:.1f}%<br>"
        )
        v62_strategy_card = f"""
  <div class="strategy-card" style="background: #F3E5F5; border-color: #9467BD;">
    <h4>v6.2 ir_expanding <span class="legend-box legend-best">⭐ Stage 29 PROMISING</span></h4>
    <p><b>类型</b>: 11 量价因子 + IC 加权 + 因子正交化 + 逆波动率加权</p>
    <p><b>核心</b>: Gram-Schmidt 残差化, expanding IR 排序, 最大化因子纯度</p>
    <p><b>类型</b>: <b>v5 因子排序</b> + <b>IC 加权 (12 月 expanding)</b> + <b>Gram-Schmidt 因子正交化</b> (按 expanding IR 排序) + v5.1 逆波动加权</p>
    <p><b>核心</b>: 用截至 d<sub>i-1</sub> 的 expanding IR 给 11 因子排序, Gram-Schmidt 残差化去除 f8-f9, f3-f4 冗余. 5-fold 验证 4/5 胜 v6.1 IC12, 真实提升非运气.</p>
    <p><b>OOS</b>: {v62_oos['ann_return']*100:+.2f}% / Sharpe {v62_oos['sharpe']:.2f} / DD {v62_dd_pct:.2f}% / <b>Calmar {v62_cal:.3f}</b> ⭐</p>
    <p><b>5-fold 跨期</b>: mean=1.512, min=-0.016 (vs v6.1 mean=0.867, min=-0.604). 4/5 fold 胜 v6.1.</p>
    <p><b>状态</b>: Stage 29 升为 v6.2 默认 (PROMISING). v6.1 IC12 仍为 RECOMMENDED (稳健 baseline).</p>
  </div>
"""
        # v7.10 详细策略卡 (内部版独有, 外发版不显示)
        v710_dd_pct = abs(v710_oos.get('max_dd', 0)) * 100
        v710_strategy_card = f"""
  <div class="strategy-card" style="background: #FFE4B5; border-color: #FF4500;">
    <h4>v7.10 TV-PR (标准化+CV) <span class="legend-box legend-best">🚀 Stage 31 NEW</span></h4>
    <p><b>类型</b>: 17 macro + 19 量价因子, 时变 β_t (TV-PR, Cui 2025)</p>
    <p><b>核心</b>: 混合标准化 (宏观=时间序列 Z-score, PV=截面 Z-score + Winsorize) + 两阶段 CV (粗搜 10 → 细搜 25) + expanding window + β[t-1] (滞后一期) + step=4 (月度调仓) + 逆波动加权</p>
    <p><b>诊断</b>: 条件数 2.17e+10 → 1.32e+02 (改善 8 个数量级); Beta TV Norm 166.78 → 56.44</p>
    <p><b>OOS</b>: ret {v710_oos.get('ann_return', 0)*100:+.2f}% / Sharpe {v710_oos.get('sharpe', 0):.2f} / DD -{v710_dd_pct:.1f}% / <b>Calmar {v710_oos.get('calmar', 0):.3f}</b> 🚀</p>
    <p><b>消融</b>: β[t] 劣于 β[t-1] (Sharpe -4.8%); step=1 劣于 step=4 (Sharpe -28%, DD_duration 374天); EMA/MA 平滑均无改善</p>
    <p><b>状态</b>: Stage 31 OOS Calmar <b>{v710_oos.get('calmar', 0):.3f}</b> (vs v6.2 {v62_oos.get('calmar', 0):.3f}, +{(v710_oos.get('calmar', 0)/v62_oos.get('calmar', 0)-1)*100:+.1f}%), DD -{v710_dd_pct:.1f}%, 跨期稳健</p>
  </div>
"""
        v6_1_row = (
            f"<tr><td>🎯 聪明 (IC 加权, baseline)</td><td>v6.1 IC12</td>"
            f"<td>IC 加权, 自动剔除失效因子, DD -<b>{v61_dd_pct:.1f}%</b> (v7.10 已超越)</td>"
            f"<td><b>{v61_cal:.3f}</b></td></tr>"
        )
    else:
        v62_key_finding = ""
        v62_strategy_card = ""
        v710_strategy_card = ""
        v6_1_row = ""

    # ===== 8 张基础策略卡 (v0 - v5) + v5.1 + v6.1 IC12 + HS300 基准 =====
    # 从 Stage 27 归档版恢复, 供主文件 (内部跟踪) 完整版使用
    v00_oos = oos_metrics.get('v0.0 baseline', {})
    v01_oos = oos_metrics.get('v0.1 +VT', {})
    v02_oos = oos_metrics.get('v0.2 +TF', {})
    v10_oos = oos_metrics.get('v1.0 locked', {})
    v3_oos  = oos_metrics.get('v3 (52 池)', {})
    v4s_oos = oos_metrics.get('v4 style', {})
    v4f_oos = oos_metrics.get('v4 factor', {})
    v5_oos  = oos_metrics.get('v5 量价', {})
    v51_oos_full = oos_metrics.get('v5.1 量价 (逆波动)', {})

    def _fmt_m(oos: dict) -> str:
        """OOS dict → 渲染串. 缺失时返回 'N/A'."""
        if not oos:
            return "N/A"
        return (f"ret {oos['ann_return']*100:+.2f}% / "
                f"Sharpe {oos['sharpe']:+.2f} / "
                f"DD {oos['max_dd']*100:.2f}% / "
                f"<b>Calmar {oos['calmar']:+.3f}</b>")

    v0_strategy_card = f"""
  <div class="strategy-card">
    <h4>v0.0 baseline <span class="legend-box legend-good">Stage 8</span></h4>
    <p><b>类型</b>: CICC 原始复现 | <b>信号</b>: 144 日纯价格动量 | <b>选股</b>: 池中 Top-10 | <b>加权</b>: 逆波动</p>
    <p><b>核心</b>: CICC 报告 (2026-07-03) 的 4 步组合管理, 仅价格动量, 无任何增强</p>
    <p><b>OOS</b>: {_fmt_m(v00_oos)}</p>
  </div>"""

    v01_strategy_card = f"""
  <div class="strategy-card">
    <h4>v0.1 +VT <span class="legend-box legend-good">Stage 9-C</span></h4>
    <p><b>类型</b>: 波动率目标 | <b>增强</b>: 在 v0.0 基础上启用 VT (target=0.15, lookback=60, scale∈[0.3, 1.5])</p>
    <p><b>核心</b>: 高波动期降仓, 目标年化波动 15%, 极端时只保留 30% 仓位</p>
    <p><b>OOS</b>: {_fmt_m(v01_oos)} — 收益被压低但 DD 大幅改善</p>
  </div>"""

    v02_strategy_card = f"""
  <div class="strategy-card">
    <h4>v0.2 +TF <span class="legend-box legend-good">Stage 9-B</span></h4>
    <p><b>类型</b>: 趋势过滤 | <b>增强</b>: 在 v0.0 基础上启用 TF (HS300 MA200, bear=0.7)</p>
    <p><b>核心</b>: 沪深 300 跌破 200 日均线时, 整体仓位缩至 70% (剩余 30% 转 511260 国债 ETF)</p>
    <p><b>OOS</b>: {_fmt_m(v02_oos)} — TF 收益略高于 VT</p>
  </div>"""

    v1_strategy_card = f"""
  <div class="strategy-card">
    <h4>v1.0 locked <span class="legend-box legend-best">⭐ OOS 最佳 (历史)</span></h4>
    <p><b>类型</b>: v1.0 锁定配置 | <b>增强</b>: 斜率×R² 混合动量 + VT + 成本 (5bp+10bp)</p>
    <p><b>核心</b>: 用 (价格动量 × 0.5 + 斜率×R² × 0.5) 替代纯价格动量, 信号更稳定; VT 缩放降低 DD; 含完整交易成本</p>
    <p><b>OOS</b>: {_fmt_m(v10_oos)} — 风险调整冠军 (12 SmartBeta 子集, 口径 B)</p>
  </div>"""

    v3_strategy_card = f"""
  <div class="strategy-card">
    <h4>v3 (52 池) <span class="legend-box legend-good">Stage 16A</span></h4>
    <p><b>类型</b>: 多策略组合 | <b>子策略</b>: 动量(144d) + 均值反转(60d 反向+MA 金叉) + 行业轮动(60d 动量+加速度, 周度)</p>
    <p><b>核心</b>: 三子策略互补, 动量抓趋势/反转抓修复/行业轮动抓短期切换, 子策略等权</p>
    <p><b>OOS</b>: {_fmt_m(v3_oos)} — 最均衡</p>
  </div>"""

    v4_style_card = f"""
  <div class="strategy-card">
    <h4>v4 style <span class="legend-box legend-bad">Stage 18 ⚠️</span></h4>
    <p><b>类型</b>: 风格轮动 | <b>池</b>: 5 风格组 (HS300/CSI500/ChiNext/STAR50/红利) | <b>信号</b>: 多窗口 (5/20/120/180) 动量</p>
    <p><b>核心</b>: 多窗口动量排名 + 20% 红利底仓 + Top-2 精选 + Sideways 缩仓 50%</p>
    <p><b>问题</b>: 5 风格组高度相关 (0.86-0.90), 选股 ≈ 噪声; 70% 时间是 sideways, 年化 -2.5%</p>
    <p><b>OOS</b>: {_fmt_m(v4s_oos)} — 表现差</p>
  </div>"""

    v4_factor_card = f"""
  <div class="strategy-card">
    <h4>v4 factor <span class="legend-box legend-bad">Stage 18 ❌ 失效</span></h4>
    <p><b>类型</b>: IC 因子择时 | <b>因子</b>: 5 个 (momentum/reversal/value/dividend/quality) | <b>权重</b>: max(0, IC+0.05)²</p>
    <p><b>核心</b>: 滚动 IC 估计 + 因子特异性窗口 + Regime 条件化 (bull/bear/sideways)</p>
    <p><b>问题</b>: IC 信噪比极差 (84-94% 低于 0.05 阈值); 70% 时间退化为单因子 (value only)</p>
    <p><b>OOS</b>: {_fmt_m(v4f_oos)} — 完全失效</p>
  </div>"""

    v5_strategy_card = f"""
  <div class="strategy-card">
    <h4>v5 量价 <span class="legend-box legend-good">Stage 22 基础版</span></h4>
    <p><b>类型</b>: 11 量价因子复合 | <b>因子</b>: 6 大类 (动量/交易波动/换手率/多空对比/量价背离/量幅同向)</p>
    <p><b>核心</b>: 截面 z-score + 复合因子加权 (z 加权 vs 等权在 52 池差异 <b>0.5%</b>, Stage 24 ablation 结论)</p>
    <p><b>加权</b>: 等权 | <b>调仓</b>: 月度</p>
    <p><b>OOS</b>: {_fmt_m(v5_oos)}</p>
    <p><b>说明</b>: v5 已被 v5.1 (逆波动) 替代, 此卡保留作为演进基线</p>
  </div>"""

    v51_strategy_card = f"""
  <div class="strategy-card">
    <h4>v5.1 量价 (逆波动) <span class="legend-box legend-good">Stage 23+ 推荐</span></h4>
    <p><b>类型</b>: 11 量价因子 (同 v5) | <b>加权</b>: 逆波动 (60d 窗口, vol_floor=0.01, max_weight=0.25)</p>
    <p><b>核心</b>: 等权 → 逆波动加权, T+1 调仓, 消融 look-ahead (Stage 24)</p>
    <p><b>OOS</b>: {_fmt_m(v51_oos_full)}</p>
    <p><b>说明</b>: 历史上绝对收益冠军 (年化 12.32%), 但 DD -18.04% 较大; 已被 v7.10 TV-PR (Calmar 2.144, DD -14.3%) 超越</p>
  </div>"""

    # v6.1 IC12 卡片 (Stage 27 RECOMMENDED)
    v61_strategy_card = ""
    if v61_oos:
        v61_strategy_card = f"""
  <div class="strategy-card" style="background: #FFF8E1; border-color: #D4A017;">
    <h4>v6.1 IC12 <span class="legend-box legend-good">Stage 27 BASELINE (v7.10 已超越)</span></h4>
    <p><b>类型</b>: <b>v5 因子排序</b> + <b>IC 加权 (12 月 expanding)</b> + v5.1 逆波动加权</p>
    <p><b>核心</b>: 用截至 d<sub>i-1</sub> 的 12 月 IC 加权替代等权, 给近期有效因子更高权重, 自动剔除失效因子 (负 IC)</p>
    <p><b>OOS</b>: ret {v61_oos['ann_return']*100:+.2f}% / Sharpe {v61_oos['sharpe']:.2f} / DD {v61_oos['max_dd']*100:.2f}% / <b>Calmar {v61_oos['calmar']:.3f}</b> ⭐</p>
    <p><b>稳定性</b>: 跨期稳健, DD -15.4% (v5.1.1 -18.0%); Stage 31 起降为 BASELINE, v7.10 已超越 (DD -14.3%, Calmar 2.144)</p>
  </div>"""

    # HS300 基准卡片 (hs300_oos 在下方 metrics 重算后才有, 此处先用占位, 后填充)
    hs300_card = ""

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

    print("[curve] 生成 7 个业绩图表...", flush=True)
    import time as _t
    import json as _json
    _t0 = _t.time()

    # 缓存支持: 复用已生成的图表 JSON, 避免每次重跑 HTML 都重新计算图表
    _CACHE_DIR = Path(__file__).parent / "_chart_cache"
    _CACHE_DIR.mkdir(exist_ok=True)
    _use_cache = os.environ.get("QN_HTML_USE_CACHE", "1") == "1"
    _refresh = os.environ.get("QN_HTML_REFRESH_CACHE", "0") == "1"

    figs = {}
    _chart_specs = [
        ("all_curves", chart_all_curves, navs_A_with_bench),
        ("grouped", chart_grouped_curves, navs_A_with_bench),
        ("alpha", chart_alpha_curves, navs_A_with_bench),
        ("drawdown", chart_drawdown_compare, navs_A_with_bench),
        ("period_compare", chart_period_compare, navs_A_with_bench),
        ("radar", chart_radar, navs_A_with_bench),
        ("monthly_heatmap", chart_monthly_heatmap, navs_A_with_bench),
    ]
    for key, fn, arg in _chart_specs:
        cache_path = _CACHE_DIR / f"{key}.json"
        if _use_cache and not _refresh and cache_path.exists():
            import plotly.io as _pio
            figs[key] = _pio.read_json(str(cache_path))
            print(f"  [chart {len(figs)}/7] {key}: cache hit ({cache_path.stat().st_size//1024} KB)", flush=True)
        else:
            figs[key] = fn(arg)
            figs[key].write_json(str(cache_path))
            print(f"  [chart {len(figs)}/7] {key}: generated ({cache_path.stat().st_size//1024} KB)", flush=True)

    print(f"[curve] 7 图表就绪, 耗时 {_t.time()-_t0:.1f}s", flush=True)

    # 嵌入 plotly.js
    if PLOTLY_JS.exists():
        plotly_src = PLOTLY_JS.read_text(encoding="utf-8")
    else:
        print("  [WARN] plotly.min.js not found, 使用 CDN")
        plotly_src = ""

    # 计算指标
    full_metrics = {col: metrics(navs_A_with_bench[col]) for col in navs_A_with_bench.columns}
    # v6 是在 navs_A_with_bench 之后才加入的, 用 navs_A_with_bench.loc[OOS_START:] 重新切,
    # 避免老 oos (v6 加载前) 不含 v6 导致回退到 full_metrics (即 'OOS 全期相同' bug)
    oos_full = navs_A_with_bench.loc[OOS_START:]
    oos_metrics = {col: metrics(oos_full[col]) for col in navs_A_with_bench.columns}
    # 排序只对策略列 (排除 HS300 基准)
    strategy_cols = [c for c in navs_A_with_bench.columns if c != "HS300 基准"]
    oos_sorted = sorted(
        [(c, oos_metrics[c]) for c in strategy_cols],
        key=lambda x: x[1]["calmar"], reverse=True
    )
    # HS300 单独放最后
    hs300_oos = oos_metrics.get("HS300 基准", full_metrics.get("HS300 基准"))
    if hs300_oos:
        hs300_card = f"""
  <div class="strategy-card" style="background: #F0F0F0; border-color: #999;">
    <h4>HS300 基准 📊 <span class="legend-box" style="background:#999;color:#fff;">基准</span></h4>
    <p><b>类型</b>: 被动指数 (沪深 300 ETF 510300) | <b>说明</b>: 无主动管理, 纯市场暴露, 用于 α 超额参考</p>
    <p><b>OOS</b>: ret {hs300_oos['ann_return']*100:+.2f}% / Sharpe {hs300_oos['sharpe']:.2f} / DD {hs300_oos['max_dd']*100:.2f}% / <b>Calmar {hs300_oos['calmar']:.3f}</b></p>
  </div>"""
    else:
        hs300_card = ""

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
    chart_descriptions = {
        "all_curves": "13 策略 NAV 曲线叠加, 重点策略用实线, 参考策略用虚线变淡. 蓝色高亮为 OOS 区间 (2022-2026), 红色虚线标记 2024-09 政策利好, 灰色虚线标记 2022 熊市底. ⭐=v1.0 locked (历史 OOS 最佳 1.791), 🚀=v7.10 TV-PR (Stage 31 NEW, OOS Calmar 2.144 当前最优). 任何策略 NAV 持续高于 HS300 基准 (深灰 dashdot) 即代表跑赢大盘.",
        "grouped": "2×2 网格分组对比: 左上是 v1.0 演进路径 (Stage 8→v1.0, 看每个优化的贡献), 右上和下排是不同风格策略对比. 每组独立 Y 轴, 重点看相对走势而非绝对值.",
        "alpha": "α = 策略 NAV / HS300 NAV - 1. 持续为正代表跑赢大盘. v1.0 locked 在大多数时间跑赢 (绿色填充区域), v3 长期稳定正 α, v5 在 2022 后 α 显著为正.",
        "drawdown": "从历史峰值的最大回撤. v1.0 locked (绿色填充) 几乎无回撤, v0.1+VT 类似. v4 因子/风格在 2022 熊市时回撤达 -38%, 是结构性风险. v6.2 ir_expanding DD -16.7% 仍可接受. v7.10 TV-PR (标准化+CV) DD -14.3% 较优. HS300 基准 (深灰) 用于对比大盘回撤.",
        "period_compare": "全期 vs OOS 双柱状图, 金边标记 v1.0 locked (OOS 0.791). 可看出哪些策略在 OOS 期间仍保持稳定表现 (v1.0/v3/v6.2), 哪些出现衰减.",
        "radar": "6 维归一化指标, 越靠外越好: 年化收益 / Sharpe / Calmar / 1/|DD| (回撤倒数) / 低波动 / 稳定性. v1.0 locked (绿色填充) 在 DD/Calmar 维度最突出, v5 在收益维度最突出.",
        "monthly_heatmap": "月度收益热图, 13 策略 × 101 月. 绿色 = 正收益, 红色 = 负收益. v1.0 全期基本无红格 (DD 控制好), v4 因子在 2022 集中出现红格.",
    }
    for key, fig in figs.items():
        desc = chart_descriptions.get(key, "")
        title_text = _extract_chart_title(fig)
        sections.append(f"""
        <section id="{key}">
          <h2>{title_text}</h2>
          {fig.to_html(full_html=False, include_plotlyjs=False, div_id=key)}
          <p style="font-size:13px;color:#555;background:#F8F9FA;padding:10px 14px;border-left:3px solid #1F77B4;border-radius:4px;margin-top:12px;">
            <b>解读</b>: {desc}
          </p>
        </section>""")

    # 策略简述 (只在完整版显示, STRATEGY_ITERATION_RECORD.html; v2 简化版不显示)
    strategies_section_html = ""
    if include_strategies:
        strategies_section_html = f"""
<section id="strategies">
  <h2>v0 - v7.10 策略简述 (13 策略 + HS300 基准, 按时间顺序)</h2>
  <p style="font-size:12px;color:#888;">从 Stage 8 (CICC 原始复现) 到 v7.10 (TV-PR 标准化+CV, Cui 2025), 完整记录量化策略演进轨迹. 主文件 (内部跟踪) 完整呈现; 报告版 (STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html) 省略本节.</p>
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">📚 基础阶段 (Stage 8 - 22): CICC → v5 量价</h3>
  {v0_strategy_card}
  {v01_strategy_card}
  {v02_strategy_card}
  {v1_strategy_card}
  {v3_strategy_card}
  {v4_style_card}
  {v4_factor_card}
  {v5_strategy_card}
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">🚀 加权与风控阶段 (Stage 23+ - 26): v5.1 → v6 TF</h3>
  {v51_strategy_card}
  {v6_strategy_card}
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">🎯 智能加权阶段 (Stage 27 - 31): v6.1 IC12 → v6.2 ir_expanding → v7.10 TV-PR (标准化+CV)</h3>
  {v61_strategy_card}
  {v62_strategy_card}
  {v710_strategy_card}
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">📊 基准</h3>
  {hs300_card}
</section>
"""

    # 全期 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>v1-v7 业绩曲线对比</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1280px; margin: 20px auto; padding: 0 20px; background: #fafafa; color: #2C3E50; line-height: 1.65; }}
h1 {{ color: #1F77B4; border-bottom: 3px solid #1F77B4; padding-bottom: 8px; margin-top: 0; font-size: 28px; }}
h2 {{ color: #2C3E50; margin-top: 40px; border-left: 5px solid #1F77B4; padding-left: 10px; font-size: 20px; }}
h3 {{ color: #34495E; margin-top: 24px; font-size: 16px; border-bottom: 1px dashed #ddd; padding-bottom: 4px; }}
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
.methodology {{ background: #F8F9FA; padding: 18px; border-radius: 8px; margin: 12px 0; border-left: 4px solid #1F77B4; font-size: 14px; }}
.methodology b {{ color: #1F77B4; }}
.methodology code {{ background: #e9ecef; padding: 1px 5px; border-radius: 3px; font-size: 13px; color: #c7254e; }}
.legend-box {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 12px; margin-right: 6px; }}
.legend-best {{ background: #FFF9E6; color: #B8860B; border: 1px solid #FFD700; }}
.legend-bench {{ background: #F0F0F0; color: #555; border: 1px solid #999; font-style: italic; }}
.legend-good {{ background: #E8F5E9; color: #2E7D32; border: 1px solid #66BB6A; }}
.legend-bad {{ background: #FFEBEE; color: #C62828; border: 1px solid #EF5350; }}
.strategy-card {{ background: #FAFBFC; border: 1px solid #E0E4E8; border-radius: 6px; padding: 12px 16px; margin: 10px 0; }}
.strategy-card h4 {{ margin: 0 0 6px 0; color: #1F77B4; font-size: 14px; }}
.strategy-card p {{ margin: 4px 0; font-size: 13px; color: #555; }}
.navbar {{ position: sticky; top: 0; background: rgba(255,255,255,0.95); padding: 12px; border-bottom: 1px solid #ddd; z-index: 100; backdrop-filter: blur(10px); }}
.navbar a {{ margin-right: 14px; color: #1F77B4; text-decoration: none; font-size: 14px; padding: 4px 8px; border-radius: 4px; }}
.navbar a:hover {{ background: #E8F5E9; text-decoration: none; }}
.color-tag {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }}
.toc {{ background: #FFFEF7; border: 1px solid #FFE082; border-radius: 6px; padding: 12px 18px; margin: 16px 0; font-size: 13px; }}
.toc b {{ color: #F57C00; }}
</style>
</head>
<body>

<h1>v1-v7 业绩曲线对比</h1>

<div class="navbar">
  <a href="#all_curves">主曲线</a>
  <a href="#grouped">分组曲线</a>
  <a href="#alpha">α 超额</a>
  <a href="#drawdown">回撤对比</a>
  <a href="#period_compare">全期 vs OOS</a>
  <a href="#radar">雷达图</a>
  <a href="#monthly_heatmap">月度热图</a>
  <a href="#metrics_table">指标表</a>
  {f'<a href="#strategies">策略简述</a>' if include_strategies else ''}
  <a href="#events">关键事件</a>
</div>

<div class="key-finding">
  <strong>核心发现 (口径 A OOS 2022-2026, 含 HS300 基准):</strong><br>
  • <b>风险调整冠军 (当前最优 🚀)</b>: v7.10 TV-PR (Stage 31 NEW) — OOS Calmar <b>{v710_oos.get('calmar', 0):.3f}</b>, Sharpe <b>{v710_oos.get('sharpe', 0):.2f}</b>, DD -{v710_dd_pct:.1f}% (weekly CV Calmar 2.144 ⭐)<br>
  • <b>风险调整冠军 (历史)</b>: v1.0 locked — OOS Calmar 1.791, Sharpe 1.51, DD -1.94% (VT 控波动优势)<br>
  • <b>绝对收益冠军</b>: v7.10 TV-PR — OOS 年化 <b>{v710_oos.get('ann_return', 0)*100:+.2f}%</b> (HS300 同期 {hs300_oos['ann_return']*100:+.2f}%, α {v710_oos.get('ann_return', 0)*100-hs300_oos['ann_return']*100:+.1f}%)<br>
  • <b>最均衡 (稳健)</b>: v3 (52 池) — OOS 年化 7.69%, Sharpe 1.08, DD -9.89%<br>
  • <b>v7.10 vs v6.2</b>: Calmar {v710_oos.get('calmar', 0):.3f} vs {v62_oos.get('calmar', 0):.3f} (+{(v710_oos.get('calmar', 0)/v62_oos.get('calmar', 0)-1)*100:+.1f}%); DD -{v710_dd_pct:.1f}% vs -{abs(v62_oos.get('max_dd', 0))*100:.1f}%<br>
  • <b>v6.x 系列</b>: v6.1 IC12 Stage 27 BASELINE (稳健) → v6.2 ir_expanding Stage 29 PROMISING (强, 已被 v7.10 超越)<br>
  • <b>HS300 基准</b>: OOS 年化 {hs300_oos['ann_return']*100:+.2f}%, DD {hs300_oos['max_dd']*100:.2f}%, Calmar {hs300_oos['calmar']:.3f}<br>
  • <b>推荐</b>: <b>v7.10 单策略</b> (最优) 或 v1.0 80% + <b>v7.10 20%</b> (保守+进攻, 兼顾低 DD 与高 α)
</div>

<div class="key-finding" style="background: linear-gradient(135deg, #FFE4B5 0%, #FFF9E6 100%); border-left-color: #FF4500; margin-top: 12px;">
  <strong>🚀 Stage 31 进展 (2026-07-17): v7.10 TV-PR (标准化+CV) 上线</strong><br>
  • <b>方法</b>: 17 macro + 19 量价因子, 时变 β_t (TV-PR, Cui 2025), 混合标准化 (宏观=时间序列Z, PV=截面Z+Winsorize) + 两阶段 CV (粗搜 10 → 细搜 25) + expanding window<br>
  • <b>诊断</b>: 条件数 2.17e+10 → 1.32e+02 (改善 8 个数量级); Beta TV Norm 166.78 → 56.44; β[t-1] 优于 β[t] (+4.8% Sharpe); step=4 优于 step=1 (+28% Sharpe, DD_duration 374→136 天)<br>
  • <b>OOS</b>: weekly Sharpe 1.60, Calmar 2.144 ⭐; daily Sharpe 1.47, Calmar 1.636, DD -14.8%, DD_duration 136 天<br>
  • <b>超越 v6.2</b>: Calmar +{(v710_oos.get('calmar', 0)/v62_oos.get('calmar', 0)-1)*100:+.1f}% (daily); weekly CV +161% (2.144 vs 0.821)<br>
  • <b>决策</b>: v7.10 为新默认 RECOMMENDED (稳健+最优), v6.1 IC12 降为 BASELINE (历史稳健 baseline), v6.2 ir_expanding 降为 PROMISING 备份. 详见 <code>reports/momentum_etf_rotation/v7_10_oos_validation.md</code>
</div>

<div class="toc">
  <b>📖 图表阅读指南:</b> 实线粗线 = 重点策略 | 虚线 = 参考策略 | 深灰虚线 = HS300 基准 | 金边/绿色填充 = 最佳 (v1.0 locked) | 蓝色高亮 = OOS 区间<br>
  排名按 <b>OOS Calmar</b> (年化收益 / |最大回撤|) 降序排列, 该指标兼顾收益与回撤控制, 是评价策略稳健性的核心指标
</div>

<section id="methodology">
  <h2>方法论与口径说明</h2>

  <h3>1. 统一 ETF 池 (52 只)</h3>
  <div class="methodology">
    <b>主池 44 只</b> (来自 <code>common/universe.py</code>): 6 只 A 股宽基 + 20 只 A 股行业 + 5 只港股 + 6 只商品 + 6 只海外 + 1 只国债<br>
    <b>SmartBeta 8 只</b> (来自 <code>v4/universe_v4.py</code>): 红利低波/低波/质量/价值/现金流等<br>
    <b>例外</b>: v4 子策略固定使用 12 只 SmartBeta (设计意图) | v5 使用 44 只 OHLCV (需要 volume 数据)
  </div>

  <h3>2. 统一时间区间</h3>
  <div class="methodology">
    <b>回测区间</b>: 2018-01-01 ~ 2026-06-30 (8.5 年, 2058 个交易日)<br>
    <b>OOS 区间</b>: 2022-01-01 ~ 2026-06-30 (4.5 年) — Walk-Forward 验证, 无 look-ahead bias<br>
    <b>预热期</b>: 252 天 (1 年) — 满足 v1.0 斜率×R² 与 v5 量价因子的最小历史需求
  </div>

  <h3>3. 风险控制 (CICC 规则)</h3>
  <div class="methodology">
    A 股宽基 + 行业 ≤ <b>3</b> 只 | 港股 ≤ 1 只 | 必含商品 + 海外 | 单边成本 <b>5bp</b> (v1.0 含 5+10bp 全部)<br>
    v1.0 额外启用 <b>波动率目标 (VT)</b>: target_vol=15%, 将仓位缩放到目标波动, 极端时只保留 30% 仓位
  </div>

  <h3>4. 指标定义</h3>
  <div class="methodology">
    <b>年化收益</b>: NAV 终值 / NAV 初值 ^ (252 / 实际交易日数) - 1<br>
    <b>年化波动</b>: 日收益标准差 × √252<br>
    <b>Sharpe</b>: 年化收益/年化波动 (无风险利率 = 0)<br>
    <b>最大回撤</b>: 历史 NAV 从峰值到谷底的最大跌幅<br>
    <b>Calmar</b>: 年化收益 / |最大回撤| — 风险调整核心指标
  </div>
</section>

<div class="events">
"""

    for date, label in events:
        html += f'  <div class="event"><span class="event-date">{date.strftime("%Y-%m-%d")}</span> {label}</div>\n'

    html += f"""</div>

<section id="metrics_table">
  <h2>13 策略 OOS 业绩表 (按 Calmar 排序)</h2>
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

{strategies_section_html}
<section id="events">
  <h2>关键事件时间线</h2>
  <table>
    <tr><th>日期</th><th>事件</th><th>市场背景</th><th>对策略的影响</th></tr>
    <tr><td>2018-01-29</td><td>2018 春节前</td><td>A 股 2018 年初快速冲高后回落</td><td>v3 / v4 启动期, v1.0 早期动量信号建立</td></tr>
    <tr><td>2020-03-23</td><td>疫情底</td><td>新冠疫情全球爆发, A 股急跌后反弹</td><td>v5 量价在反弹后 +40% 收益, v3 抓住回升</td></tr>
    <tr><td>2022-04-26</td><td>2022 熊市底</td><td>俄乌冲突 + 美联储加息, A 股深度调整</td><td>v4 因子 -23.59% (IC 信号失效), v1.0 VT 保护仅 -0.71%</td></tr>
    <tr><td>2022-10-31</td><td>2022 反弹</td><td>10 月后政策预期转向, 风险偏好回升</td><td>v3 反弹 +25%, v1.0 +5% (VT 仍保守)</td></tr>
    <tr><td>2024-09-23</td><td>政策利好</td><td>央行/证监会组合政策, 股市快速反弹</td><td>v5 量价 +35% (年化), v3 +12%, v1.0 仅 +5% (VT 限制)</td></tr>
    <tr><td>2025-09-30</td><td>2025 Q3 末</td><td>市场震荡上行, 风格分化</td><td>v5 +32.29% (年度最佳), v1.0 +5.26% (低波动防御)</td></tr>
  </table>
</section>

<section id="recommendations">
  <h2>策略推荐与组合方案</h2>

  <h3>1. 单策略推荐 (按风险偏好)</h3>
  <table>
    <tr><th>风险偏好</th><th>推荐策略</th><th>理由</th><th>OOS Calmar</th></tr>
    <tr><td>🛡️ 极保守</td><td>v1.0 locked</td><td>波动率目标限制 DD 至 -1.94%, 适合低风险偏好</td><td><b>1.791</b></td></tr>
    <tr><td>⚖️ 均衡</td><td>v3 (52 池)</td><td>三子策略互补, 风险与收益平衡</td><td>0.778</td></tr>
    <tr><td>🚀 进取 (历史)</td><td>v5.1 逆波动</td><td>11 量价因子 + 逆波动, 年化 {oos_metrics['v5.1 量价 (逆波动)']['ann_return']*100:.2f}%, 风险调整优 (已被 v7.10 超越)</td><td><b>{oos_metrics['v5.1 量价 (逆波动)']['calmar']:.3f}</b></td></tr>
    <tr><td>🚀 进取 (最优 🚀)</td><td>v7.10 TV-PR 🚀 Stage 31 NEW</td><td>17 macro + 19 量价因子 + 混合标准化 + 两阶段 CV, DD 仅 -{v710_dd_pct:.1f}%, 年化 {v710_oos.get('ann_return', 0)*100:+.2f}%</td><td><b>{v710_oos.get('calmar', 0):.3f}</b></td></tr>{v6_aggressive_row}{v6_1_row}
    <tr><td>📊 基准</td><td>HS300</td><td>被动指数, 无主动管理成本</td><td>{hs300_oos['calmar']:.3f}</td></tr>
  </table>

  <h3>2. 组合推荐 (多策略分散)</h3>
  <div class="methodology">
    <b>🚀 v7.10 单策略</b> (最优推荐 🚀) — OOS Calmar <b>{v710_oos.get('calmar', 0):.3f}</b>, Sharpe {v710_oos.get('sharpe', 0):.2f}, DD -{v710_dd_pct:.1f}%<br>
    优势: 当前风险调整最优 (weekly CV Calmar 2.144), 条件数改善 8 个数量级, 跨期稳健<br>
    <br>
    <b>v1.0 80% + v7.10 20%</b> (保守+进攻) — 全期 Calmar 估算 ≥ 1.4, OOS 估算 ≥ 1.5 ⭐ (兼顾低 DD 与高 α)<br>
    优势: 利用 v1.0 (低 DD) + v7.10 (高 α+低 DD) 的低相关性, 跨入'1.5 俱乐部'<br>
    <br>
    <b>⭐ v1.0 80% + v5.1 20%</b> (历史推荐) — 全期 Calmar <b>1.146</b>, OOS <b>1.015</b> ⭐, OOS Sharpe 0.95<br>
    优势: 利用 v1.0 (低 DD) + v5.1 (高收益+逆波动) 的低相关性, 跨入'1.0 俱乐部'<br>
    <br>
    <b>v1.0 50% + v3 25% + v5.1 25%</b> — 全期 Calmar 0.958, OOS 0.951, OOS Sharpe 0.91<br>
    优势: 三策略分散, 风险/收益/胜率都平衡<br>
    <br>
    <b>v3 50% + v5.1 50%</b> — 全期 Calmar 0.764, OOS 0.765, OOS Sharpe 0.80<br>
    优势: 无 VT 拖累, 进攻性较强<br>
    <br>
    <i>v5 (等权) 旧组合作为参考: v1.0 80% + v5 20% — OOS Calmar 0.886 (v5.1 升级后 +14.6%)</i>
  </div>

  <h3>3. 风险提示</h3>
  <div class="methodology" style="border-left-color: #F57C00; background: #FFF8E1;">
    <b>⚠️ 实盘前需注意:</b><br>
    1. <b>回测非预测</b>: 8.5 年样本量有限, 实盘可能与回测差异较大<br>
    2. <b>交易成本</b>: v5 月换手 161%, 实际成本可能侵蚀 2-3% 年化<br>
    3. <b>流动性</b>: 部分 SmartBeta ETF 流动性低, 大资金调仓冲击大<br>
    4. <b>政策风险</b>: 2022/2024 政策窗口期表现差异巨大, 未来不可预测<br>
    5. <b>数据依赖</b>: v5 需 OHLCV 数据, 缺数据期间退化为 v3 逻辑
  </div>
</section>

<footer style="margin-top: 40px; padding: 20px; background: #F0F4F8; border-radius: 8px; font-size: 12px; color: #666;">
  <b>📊 数据来源</b>: <code>combo/unified_v1v5_compare.py</code> 生成 NAV, <code>combo/nav_curves_html.py</code> 生成此 HTML<br>
  {'<b>📁 主文件 (内部跟踪)</b>: <code>STRATEGY_ITERATION_RECORD.html</code> (完整版, 含策略简述, 每次跑都覆盖)<br>' if include_strategies else '<b>📤 报告版 (对外汇报)</b>: <code>STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html</code> (简化版, 无策略简述, 带日期)<br>'}
  <b>🔗 相关文件</b>: <code>UNIFIED_V1V5_EVOLUTION.html</code> (v1-v5 演进专题) | <code>UNIFIED_V1V5_REPORT.md</code> (完整 markdown 报告)<br>
  <b>⚙️ 统一口径</b>: 52 ETF 池 | 2018-2026 | 5bp 单边成本 | A 股 cap=3 (CICC 规则) | v4 用 12 SmartBeta 子集
</footer>

</body>
</html>"""

    # 嵌入 plotly.js
    if plotly_src:
        html = html.replace("</head>", f'<script>{plotly_src}</script></head>')

    return html


def main_dispatcher():
    """生成两个版本:
    1. STRATEGY_ITERATION_RECORD.html — 主文件, 完整版 (含策略简述), 内部跟踪用
    2. STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html — 报告版, 简化版 (无策略简述), 对外汇报用
    """
    from datetime import datetime
    today_str = datetime.now().strftime("%Y%m%d")

    # 1. 主文件 (完整版)
    print("\n[1/2] 生成主文件 STRATEGY_ITERATION_RECORD.html (完整版, 内部跟踪) ...")
    html_full = main(include_strategies=True)
    out_main = OUT_DIR / "STRATEGY_ITERATION_RECORD.html"
    out_main.write_text(html_full, encoding="utf-8")
    size_mb = out_main.stat().st_size / 1024 / 1024
    print(f"[save] {out_main} ({size_mb:.2f} MB)")
    print(f"      7 个图表: 主曲线 / 分组 / α超额 / 回撤 / 全期对比 / 雷达 / 月度热图")
    print(f"      指标表 (含 HS300 基准行) + 13 个策略简述 (v0-v7.10) + HS300 基准 + 关键事件时间线")

    # 2. 报告版 (简化版, 带日期)
    print(f"\n[2/2] 生成报告版 STRATEGY_ITERATION_RECORD_v2_{today_str}.html (简化版, 对外汇报) ...")
    html_simp = main(include_strategies=False)
    out_report = OUT_DIR / f"STRATEGY_ITERATION_RECORD_v2_{today_str}.html"
    out_report.write_text(html_simp, encoding="utf-8")
    size_mb = out_report.stat().st_size / 1024 / 1024
    print(f"[save] {out_report} ({size_mb:.2f} MB)")
    print(f"      7 个图表: 主曲线 / 分组 / α超额 / 回撤 / 全期对比 / 雷达 / 月度热图")
    print(f"      指标表 (含 HS300 基准行) + 关键事件时间线")
    print(f"      ⚠️  无策略简述 (本版为对外汇报, 内容聚焦曲线和指标)")
    print(f"\n[summary]")
    print(f"  主文件 (内部跟踪): {out_main}")
    print(f"  报告版 (对外汇报): {out_report}")


if __name__ == "__main__":
    main_dispatcher()
