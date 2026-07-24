# coding=utf-8
"""v1-v10 业绩曲线对比 HTML (纯 NAV 曲线, 轻量级).

聚焦: 全部策略 + HS300 基准的 NAV 曲线对比 (v0.0 → v10 动态权重)
特点:
- 单一图表多策略叠加 (主曲线)
- 拆分面板: v1.0 演进路径 / 进攻型 / v10 独立策略
- OOS 区间高亮
- 关键事件标注 (2022 熊市, 2024 9月行情)
- 全期 + OOS 指标表
- 策略简述 + HS300 基准
- 内联 plotly.js (脱机可用)
- 双输出: STRATEGY_ITERATION_RECORD.html (主文件) + STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html (报告版)
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

OOS_START = pd.Timestamp("2021-08-01")  # 与 v9_citic_all.py 统一
OOS_END = pd.Timestamp("2026-06-30")

# plotly 兼容 (避免 pandas2.x Timestamp+int 报错): 用毫秒整数
OOS_START_MS = OOS_START.value // 10**6
OOS_END_MS = OOS_END.value // 10**6
OOS_2024_MS = pd.Timestamp("2024-09-23").value // 10**6
OOS_2022_MS = pd.Timestamp("2022-04-26").value // 10**6

COLORS = {
    "v0.0 baseline":    "#888888",
    "v1.0 locked":      "#2CA02C",
    "v5 量价":          "#E377C2",
    "v5.1 量价 (逆波动)": "#FF6B9D",
    "v7.10 TV-PR (标准化+CV)":      "#FF4500",
    "v7.14 TV-PR (修正)":           "#B71C1C",
    "v8 Jump Model 优化版":         "#1B5E20",
    # === v9 阶段 ===
    "银河方案-动态仓位":            "#FF1493",
    "银河因子配置":                "#4682B4",
    "等权基准":                    "#AAAAAA",
    "60/40股债":                   "#BBBBBB",
    "基础风险平价":                "#999999",
    # === v10 阶段 ===
    "v10 DualMom (4资产)":         "#00BCD4",
    "v10 4策略Vol-parity":         "#E91E63",
    "v10-DynD 信号加权":           "#FF5722",
    "v10-DynE 混合":               "#CDDC39",
}

STAGE_MAP = {
    "v0.0 baseline":    "原始 CICC 复现",
    "v1.0 locked":      "v1.0 (斜率×R² 混合, 锁定)",
    "v5 量价":          "11 量价因子, 等权",
    "v5.1 量价 (逆波动)": "v5.1 升级, 逆波动加权",
    "v7.10 TV-PR (标准化+CV)":       "v7.10 TV-PR: 时变 β_t, expanding-window OOS Calmar 0.662",
    "v7.14 TV-PR (修正)":            "v7.14 TV-PR 修正版: NAV bug fix",
    "v8 Jump Model 优化版":          "v8 优化版: OOS Sharpe 1.204, Calmar 1.395",
    # === v9 阶段 ===
    "银河方案-动态仓位":  "v9 头牌: 动态仓位, OOS Sharpe 1.230",
    "银河因子配置":      "v9 银河基础: 17 宏观因子 + 熵权 + 风险预算",
    # === v10 阶段 ===
    "v10 DualMom (4资产)":       "v10 DualMom: 全球4资产轮动, OOS Sharpe 0.904",
    "v10 4策略Vol-parity":       "v10 4策略Vol-parity, OOS Calmar 1.117",
    "v10-DynD 信号加权":         "v10-DynD: 信号加权, OOS Calmar 1.753",
    "v10-DynE 混合":             "v10-DynE: 混合动态, OOS Calmar 1.442",
}

# 分组
GROUPS = {
    "v1.0 演进路径": ["v0.0 baseline", "v1.0 locked"],
    "进攻型":       ["v5 量价", "v5.1 量价 (逆波动)", "v7.10 TV-PR (标准化+CV)", "v7.14 TV-PR (修正)", "v8 Jump Model 优化版", "v1.0 locked"],
    "v7 TV-PR 演进":  ["v1.0 locked", "v7.10 TV-PR (标准化+CV)", "v7.14 TV-PR (修正)", "v8 Jump Model 优化版"],
    # === v9 阶段 ===
    "v9 银河组": ["银河方案-动态仓位", "银河因子配置", "基础风险平价", "等权基准", "60/40股债"],
    # === v10 阶段 ===
    "v10 独立策略": ["v10 DualMom (4资产)"],
    "v10 组合+动态": ["v10 4策略Vol-parity", "v10-DynD 信号加权", "v10-DynE 混合"],
    "v10 全部": ["v10 DualMom (4资产)", "v10 4策略Vol-parity", "v10-DynD 信号加权", "v10-DynE 混合"],
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
# 业绩曲线 (主图: 全部策略叠加)
# ============================================================
# 标记前缀 (优先级从高到低)
_PREFIX_RULES = [
    (lambda c: c == "v1.0 locked",       "⭐ "),
    (lambda c: c.startswith("v10-Dyn"),  "🔥 "),
    (lambda c: c.startswith("v10 4"),    "🏆 "),
    (lambda c: c.startswith("v10 "),     "🎯 "),
    (lambda c: "v7.10" in c,            "🚀 "),
    (lambda c: "v7.14" in c,            "📊 "),
    (lambda c: "v8" in c,               "🛡️ "),
    (lambda c: "v5.1" in c,             "🆕 "),
]
_WIDTH_RULES = [
    (lambda c: c.startswith("v10-Dyn"),  2.8),
    (lambda c: c.startswith("v10 "),     2.6),
    (lambda c: c == "v1.0 locked",       2.4),
    (lambda c: "v7.10" in c,            2.4),
    (lambda c: "v8" in c,               2.2),
]
# 背景 (淡色虚线)
BACKGROUND = {"v0.0 baseline",
              "等权基准", "60/40股债", "基础风险平价"}


def _get_prefix(col):
    for fn, pfx in _PREFIX_RULES:
        if fn(col):
            return pfx
    return ""


def _get_width(col):
    for fn, w in _WIDTH_RULES:
        if fn(col):
            return w
    return 1.7


def chart_all_curves(navs):
    print(f"  [chart 1/7] all_curves...", flush=True)
    n_strategies = len([c for c in navs.columns if c != "HS300 基准"])
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

    # 先画背景 (淡色虚线)
    for col in BACKGROUND:
        if col not in navs.columns:
            continue
        valid = navs[col].dropna()
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines", name=col,
            line=dict(color=COLORS.get(col, "#aaa"), width=1.2, dash="dot"),
            opacity=0.45,
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>NAV=%{{y:.3f}}<extra></extra>",
        ))

    # 重点策略 (实线) — 动态遍历所有列
    for col in navs.columns:
        if col == "HS300 基准" or col in BACKGROUND:
            continue
        valid = navs[col].dropna()
        if len(valid) < 2:
            continue
        prefix = _get_prefix(col)
        display_name = f"{prefix}{col}"
        width = _get_width(col)
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines", name=display_name,
            line=dict(color=COLORS.get(col, "#333333"),
                      width=width,
                      shape="spline", smoothing=0.6),
            hovertemplate=f"<b>{col}</b> ({STAGE_MAP.get(col, '')})<br>"
                          "%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ))

    # OOS 区间高亮
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
            text=f"<b>v0 - v10 业绩曲线对比 (2018-2026, {n_strategies} 策略 + HS300)</b>"
                 "<br><sub>实线=重点, 虚线=参考 | ⭐=v1.0 locked | 🚀=v7.10 TV-PR | "
                 "🏆=v10 Vol-parity | 🔥=v10 动态方案 | 深灰虚线=HS300 基准 | 高亮=OOS 区间</sub>",
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
    """3x2 subplot: v1.0 演进 / 进攻型 / v9 / v10 独立 / v10 组合+动态 / v7 TV-PR."""
    print(f"  [chart 2/7] grouped...", flush=True)
    n_strategies = len([c for c in navs.columns if c != "HS300 基准"])
    panels = [
        ("v0 → v1.0 演进",
         ["v0.0 baseline", "v1.0 locked"]),
        ("v5/v7/v8 进攻型",
         ["v5.1 量价 (逆波动)", "v7.10 TV-PR (标准化+CV)", "v8 Jump Model 优化版", "v1.0 locked", "HS300 基准"]),
        ("v9 银河 (5 策略)",
         ["银河方案-动态仓位", "银河因子配置", "基础风险平价", "等权基准", "60/40股债", "HS300 基准"]),
        ("v10 DualMom + 组合",
         ["v10 DualMom (4资产)", "v10 4策略Vol-parity", "HS300 基准"]),
        ("v10 组合+动态 (6 方案)",
         ["v10 4策略Vol-parity", "v10-DynD 信号加权", "v10-DynE 混合", "HS300 基准"]),
        ("v7 TV-PR → v8 演进",
         ["v1.0 locked", "v7.10 TV-PR (标准化+CV)", "v7.14 TV-PR (修正)", "HS300 基准"]),
    ]
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[p[0] for p in panels],
        vertical_spacing=0.10,
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
            line_color = "#333333" if is_bench else COLORS.get(col, "#333333")
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid.values,
                mode="lines", name=col, showlegend=(i == 1),
                line=dict(color=line_color, width=line_width, dash=line_dash,
                          shape="spline", smoothing=0.5),
                hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>NAV=%{{y:.3f}}<extra></extra>",
            ), row=row, col=col_idx)
        fig.add_vrect(x0=OOS_START_MS, x1=OOS_END_MS,
                      fillcolor="rgba(100,100,200,0.06)", line_width=0,
                      row=row, col=col_idx)

    fig.update_layout(
        title=dict(text=f"<b>分组业绩曲线 (3×2 网格, {n_strategies} 策略分组对比)</b>",
                   x=0.02, xanchor="left", font=dict(size=15)),
        template="plotly_white",
        height=1080,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
    )
    for r in range(1, 4):
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
    n_strategies = len([c for c in navs.columns if c != "HS300 基准"])

    fig = go.Figure()
    for col in navs.columns:
        if col == "HS300 基准":
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
            line=dict(color=COLORS.get(col, "#333333"),
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
        title=dict(text=f"<b>超额收益 (Alpha) vs HS300 (v0 - v10, {n_strategies} 策略)</b>"
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
        hex_c = COLORS.get(col, "#333333").lstrip("#")
        rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode="lines", name=col,
            line=dict(color=COLORS.get(col, "#333333"),
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
        title=dict(text="<b>回撤对比 (Drawdown Over Time, v0 - v10)</b>"
                        "<br><sub>DD 越接近 0 越好 | 绿色填充=v1.0 locked</sub>",
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
        line_color = "#888888" if is_bench else COLORS.get(col, "#333333")
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
        title=dict(text="<b>全期 vs OOS 年化收益 (v0 - v10)</b>"
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
        line_color = "#666666" if is_bench else COLORS.get(col, "#333333")
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
        title=dict(text="<b>OOS 性能雷达图 (2022-2026, v0 - v10)</b>"
                        "<br><sub>6 维归一化指标, 越靠外越好 | ⭐=v1.0 locked (绿色填充) | 🏆=v10 Vol-parity | 🔥=v10 动态方案</sub>",
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
        title=dict(text="<b>月度收益热图 (2018-2026, v0 - v10)</b>"
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
def main(include_strategies: bool = True):
    """生成 STRATEGY_ITERATION_RECORD.html 主文件 (内部跟踪, 完整版) 和 STRATEGY_ITERATION_RECORD_v2_YYYYMMDD.html 报告版.

    Args:
        include_strategies: True = 完整版 (含策略简述, 内部跟踪用)
                            False = 简化版 (无策略简述, 对外汇报用)
    """
    print(f"[curve] include_strategies={include_strategies}", flush=True)
    print("[curve] 加载数据...", flush=True)
    navs_A = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")
    # 移除 v0.1/v0.2/v3/v4 策略
    remove_cols = [c for c in navs_A.columns if c.startswith("v0.1") or c.startswith("v0.2") or c.startswith("v3") or c.startswith("v4")]
    navs_A = navs_A.drop(columns=remove_cols, errors='ignore')
    print(f"  [data] parquet: {navs_A.shape}, cols={len(navs_A.columns)}, "
          f"{navs_A.index[0].date()}~{navs_A.index[-1].date()}", flush=True)
    oos = navs_A.loc[OOS_START:]  # DataFrame (用于表格)
    oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}  # dict (用于显示)

    # 用 v56 数据替换 v7.10 TV-PR
    v710_v56_path = OUT_DIR / "v7_10_nav_v56.parquet"
    if v710_v56_path.exists():
        v710_v56 = pd.read_parquet(v710_v56_path)
        for col in v710_v56.columns:
            new_col = col.replace(' (v56)', '').replace('v7.10 TV-PR', 'v7.10 TV-PR (标准化+CV)')
            # 找到 unified 中的对应列并替换
            for orig_col in navs_A.columns:
                if 'v7.10' in orig_col:
                    navs_A[orig_col] = v710_v56[col].reindex(navs_A.index)
                    print(f"[curve] {orig_col} 已用 v56 数据更新")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v7.14 TV-PR 修正版 NAV (优先用 v56 数据)
    v714_v56_path = OUT_DIR / "v7_14_nav_v56.parquet"
    if v714_v56_path.exists():
        v714_navs = pd.read_parquet(v714_v56_path)
        # 用 v56 无成本版本作为 v7.14 基准
        cost0_col = [c for c in v714_navs.columns if 'cost=0bp' in c]
        if cost0_col:
            navs_A['v7.14 TV-PR (修正)'] = v714_navs[cost0_col[0]]
            print(f"[curve] v7.14 TV-PR (修正) 已用 v56 数据更新: OOS Calmar="
                  f"{metrics(v714_navs[cost0_col[0]].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v7.10 TV-PR (原始)
    v714_path = OUT_DIR / "v7_14_nav.parquet"
    if v714_path.exists() and 'v7.14 TV-PR (修正)' not in navs_A.columns:
        v714_navs = pd.read_parquet(v714_path)
        for col in v714_navs.columns:
            navs_A[col] = v714_navs[col]
            print(f"[curve] {col} 已加入: OOS Calmar="
                  f"{metrics(v714_navs[col].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v8 Jump Model 方案 B NAV
    v8_path = OUT_DIR / "v8_method_b_nav.parquet"
    if v8_path.exists():
        v8_navs = pd.read_parquet(v8_path)
        for col in v8_navs.columns:
            if "方案B" in col:
                continue  # 跳过方案B
            navs_A[col] = v8_navs[col]
            print(f"[curve] {col} 已加入: OOS Calmar="
                  f"{metrics(v8_navs[col].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v8 优化版 NAV (bt=0.25, alpha=0.7, t=0.01, cost=10bp)
    v8_opt_path = OUT_DIR / "v8_optimized_nav.parquet"
    if v8_opt_path.exists():
        v8_opt_navs = pd.read_parquet(v8_opt_path)
        for col in v8_opt_navs.columns:
            navs_A[col] = v8_opt_navs[col]
            print(f"[curve] {col} 已加入: OOS Calmar="
                  f"{metrics(v8_opt_navs[col].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v8 优化版 (基于 v56 数据, 替换原 v7_6 数据版本)
    v8_opt_v56_path = OUT_DIR / "v8_optimized_nav_v56.parquet"
    if v8_opt_v56_path.exists():
        v8_opt_v56 = pd.read_parquet(v8_opt_v56_path)
        for col in v8_opt_v56.columns:
            # 替换原 v8 优化版 (因为 v56 数据更公平)
            new_col = col.replace('v8 Jump Model 优化版 (v56)', 'v8 Jump Model 优化版')
            navs_A[new_col] = v8_opt_v56[col]
            print(f"[curve] {new_col} 已用 v56 数据更新: OOS Calmar="
                  f"{metrics(v8_opt_v56[col].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v8_method_b (基于 v56 数据, 用于公平对比 v9)
    v8_v56_path = OUT_DIR / "v8_method_b_nav_v56.parquet"
    if v8_v56_path.exists():
        v8_v56 = pd.read_parquet(v8_v56_path)
        for col in v8_v56.columns:
            if "方案B" in col:
                continue  # 跳过方案B
            new_col = col
            navs_A[new_col] = v8_v56[col]
            print(f"[curve] {new_col} 已用 v56 数据更新: OOS Calmar="
                  f"{metrics(v8_v56[col].loc[OOS_START:])['calmar']:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # 加载 v9 银河策略 NAV
    v9_path = OUT_DIR / "v9_navs.parquet"
    if v9_path.exists():
        v9_navs = pd.read_parquet(v9_path)
        # 移除中信策略
        remove_citic = [c for c in v9_navs.columns if "中信" in c]
        v9_navs = v9_navs.drop(columns=remove_citic, errors='ignore')
        for col in v9_navs.columns:
            navs_A[col] = v9_navs[col]
            cal = metrics(v9_navs[col].loc[OOS_START:])['calmar'] if len(v9_navs[col].loc[OOS_START:]) > 0 else 0.0
            print(f"[curve] v9 {col:20s} 已加入: OOS Calmar={cal:.3f}")
        oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}
    else:
        print(f"[curve] ⚠️ v9_navs.parquet 不存在, 跳过 v9 (运行 scripts/combo/export_v9_navs.py 生成)")

    # 等权基准 (52 ETF 等权, 月度再平衡)
    ew_path = OUT_DIR / "equal_weight_baseline.parquet"
    if ew_path.exists():
        ew_nav = pd.read_parquet(ew_path).iloc[:, 0]
        # 如果已有同名列 (来自 v9_navs), 跳过
        if "等权基准" not in navs_A.columns:
            navs_A["等权基准"] = ew_nav
            cal = metrics(ew_nav.loc[OOS_START:])['calmar'] if len(ew_nav.loc[OOS_START:]) > 0 else 0.0
            print(f"[curve] 等权基准 (52ETF)      已加入: OOS Calmar={cal:.3f}")

    # v10 独立策略 + Vol-parity 组合
    v10_dir = REPO / "reports/momentum_etf_rotation/v10"
    v10_navs = {
        "v10 DualMom (4资产)": v10_dir / "dual_momentum_nav.parquet",
        "v10 4策略Vol-parity": v10_dir / "vol_parity_4strat_nav.parquet",
    }
    for name, path in v10_navs.items():
        if path.exists():
            nav = pd.read_parquet(path)
            navs_A[name] = nav.iloc[:, 0]
            cal = metrics(navs_A[name].loc[OOS_START:])['calmar'] if len(navs_A[name].loc[OOS_START:]) > 0 else 0.0
            print(f"[curve] {name:20s} 已加入: OOS Calmar={cal:.3f}")
        else:
            print(f"[curve] ⚠️ {path.name} 不存在, 跳过 {name}")

    # v10 动态权重方案
    v10_dynamic = {
        "v10-DynD 信号加权": v10_dir / "dynamic_nav_D_signal_weighted.parquet",
        "v10-DynE 混合": v10_dir / "dynamic_nav_E_hybrid.parquet",
    }
    for name, path in v10_dynamic.items():
        if path.exists():
            nav = pd.read_parquet(path)
            navs_A[name] = nav.iloc[:, 0]
            cal = metrics(navs_A[name].loc[OOS_START:])['calmar'] if len(navs_A[name].loc[OOS_START:]) > 0 else 0.0
            print(f"[curve] {name:20s} 已加入: OOS Calmar={cal:.3f}")
        else:
            print(f"[curve] ⚠️ {path.name} 不存在, 跳过 {name}")

    oos_metrics = {col: metrics(navs_A[col].loc[OOS_START:]) for col in navs_A.columns}

    # v7.10 策略卡
    v710_oos = oos_metrics.get('v7.10 TV-PR (标准化+CV)', {})
    v710_dd_pct = abs(v710_oos.get('max_dd', 0)) * 100
    v710_strategy_card = f"""
  <div class="strategy-card" style="background: #FFE4B5; border-color: #FF4500;">
    <h4>v7.10 TV-PR (标准化+CV)</h4>
    <p><b>类型</b>: 17 macro + 19 量价因子, 时变 β_t (TV-PR, Cui 2025)</p>
    <p><b>核心</b>: 混合标准化 + 两阶段 CV + expanding window + β[t-1]</p>
    <p><b>OOS</b>: ret {v710_oos.get('ann_return', 0)*100:+.2f}% / Sharpe {v710_oos.get('sharpe', 0):.2f} / DD -{v710_dd_pct:.1f}% / <b>Calmar {v710_oos.get('calmar', 0):.3f}</b></p>
  </div>
"""

    # ===== 策略卡 =====
    v00_oos = oos_metrics.get('v0.0 baseline', {})
    v10_oos = oos_metrics.get('v1.0 locked', {})
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

    v1_strategy_card = f"""
  <div class="strategy-card">
    <h4>v1.0 locked <span class="legend-box legend-best">⭐ OOS 最佳 (历史)</span></h4>
    <p><b>类型</b>: v1.0 锁定配置 | <b>增强</b>: 斜率×R² 混合动量 + VT + 成本 (5bp+10bp)</p>
    <p><b>核心</b>: 用 (价格动量 × 0.5 + 斜率×R² × 0.5) 替代纯价格动量, 信号更稳定; VT 缩放降低 DD; 含完整交易成本</p>
    <p><b>OOS</b>: {_fmt_m(v10_oos)} — 风险调整冠军 (12 SmartBeta 子集, 口径 B)</p>
  </div>"""

    v3_strategy_card = ""

    v4_style_card = ""

    v4_factor_card = ""

    v5_strategy_card = f"""
  <div class="strategy-card">
    <h4>v5 量价 <span class="legend-box legend-good">Stage 22 基础版</span></h4>
    <p><b>类型</b>: 11 量价因子复合 | <b>因子</b>: 6 大类 (动量/交易波动/换手率/多空对比/量价背离/量幅同向)</p>
    <p><b>核心</b>: 截面 z-score + 复合因子加权 (z 加权 vs 等权在 52 池差异 <b>0.5%</b>, Stage 24 ablation 结论)</p>
    <p><b>加权</b>: 等权 | <b>调仓</b>: 月度</p>
    <p><b>OOS</b>: {_fmt_m(v5_oos)}</p>
    <p><b>说明</b>: v5 已被 v5.1 (逆波动) 替代, 此卡保留作为演进基线</p>
  </div>"""

    v51_strategy_card = ""

    v61_strategy_card = ""

    # ===== v9 阶段策略卡片 (2021-08 ~ 2026-05, 247 周) =====
    v9_galaxy_oos = oos_metrics.get('银河方案-动态仓位', {})
    v9_gf_oos = oos_metrics.get('银河因子配置', {})

    # 概览卡片
    v9_overview_card = f"""
  <div class="strategy-card" style="background: linear-gradient(135deg, #FFE0EC 0%, #E0F0FF 100%); border-color: #FF1493;">
    <h4>📌 v9 阶段总览 (2021-08 ~ 2026-05, 247 周, 43 ETF)</h4>
    <p><b>核心成果</b>: 9 个策略完整落地 (原版 5 + 中信 4); <b>银河方案-动态仓位</b> Sharpe 1.230 创历史新高</p>
    <p><b>Brinson 归因</b> (docs/51): 仓位效应贡献 +7.80% (71%), 选股效应 +1.28% (12%), 交互效应 +1.87% (17%)</p>
    <p><b>全面战胜基准</b>: v9 头牌 Sharpe 1.230 vs 等权基准 0.248 (+0.982), vs 60/40 股债 0.195 (+1.035)</p>
    <p><b>分阶段稳定性</b>: 熊市段 (2021-2022) 仓位 +2.75%/选股 -1.25%; 震荡段 (2022-2024) 仓位 +3.30%/选股 +3.17%; 牛市段 (2024-2026) 仓位 +1.74%</p>
    <p><b>关键文档</b>: docs/54-v1_v9_strategy_summary.md / docs/51-v9_brinson_attribution.md / docs/52-v9_citic_strategies.md / docs/53-v9_strategy_factor_analysis.md</p>
  </div>"""

    # D1: 银河方案-动态仓位 (头牌)
    if v9_galaxy_oos:
        v9_galaxy_card = f"""
  <div class="strategy-card" style="background: #FFE4F1; border-color: #FF1493;">
    <h4>🌟 银河方案-动态仓位 <span class="legend-box legend-best">⭐ v9 头牌 OOS Sharpe 1.230</span></h4>
    <p><b>类型</b>: 银河证券因子配置 (17 宏观因子 + 熵权 + 风险预算) × <b>动态仓位</b></p>
    <p><b>核心公式</b>: <code>pos_t = (0.7 - 0.5 × z_score).clip(0.2, 1.0)</code>, 其中 z_score 是 factor_score 的 52 周滚动 z</p>
    <p><b>Brinson 归因</b>: 仓位贡献 <b>+7.80%</b> (71%), 选股 <b>+1.28%</b> (12%), 交互 <b>+1.87%</b> (17%)</p>
    <p><b>OOS</b>: ret {v9_galaxy_oos['ann_return']*100:+.2f}% / Sharpe {v9_galaxy_oos['sharpe']:.3f} / DD {v9_galaxy_oos['max_dd']*100:.2f}% / <b>Calmar {v9_galaxy_oos['calmar']:.3f}</b> ⭐</p>
    <p><b>设计动机</b>: Brinson 归因证实仓位是 #1 alpha 源; 银河选股本身仅 +1.28% 4.8 年累计, 但叠加动态仓位后 Sharpe 暴增 0.85</p>
  </div>"""


    # D3: 银河因子配置 (银河方案的基础)
    if v9_gf_oos:
        v9_gf_card = f"""
  <div class="strategy-card" style="background: #F0F4F8; border-color: #4682B4;">
    <h4>🌌 银河因子配置 <span class="legend-box legend-good">v9 银河基础 (Sharpe 0.386)</span></h4>
    <p><b>类型</b>: 17 宏观因子 + 熵权 + 风险预算 | <b>用途</b>: 作为银河方案-动态仓位的选股信号源</p>
    <p><b>5 类宏观指标</b>: 消费/内需 (增长+生活端通胀) + 出口/外部 (汇率+DXY) + 工业/生产 + 信贷/金融 + 风险/情绪 (VIX等)</p>
    <p><b>核心</b>: 类内等权平均 → 熵权合成 (104 周滚动) → 滚动 β 回归 (52 周) → 风险预算权重反推</p>
    <p><b>OOS</b>: ret {v9_gf_oos['ann_return']*100:+.2f}% / Sharpe {v9_gf_oos['sharpe']:.3f} / DD {v9_gf_oos['max_dd']*100:.2f}% / <b>Calmar {v9_gf_oos['calmar']:.3f}</b></p>
    <p><b>说明</b>: 固定仓位版本 (无 pos 公式) 仅 0.386, 但叠加动态仓位后变 1.230</p>
  </div>"""




    # v10 策略卡片
    v10_dm_oos = oos_metrics.get('v10 DualMom (4资产)', {})
    v10_combo_oos = oos_metrics.get('v10 4策略Vol-parity', {})
    v10_dynE_oos = oos_metrics.get('v10-DynE 混合', {})

    v10_cards = f"""
  <div class="strategy-card" style="background: #E8F5E9; border-color: #4CAF50;">
    <h4>🎯 v10 DualMom (4资产轮动)</h4>
    <p><b>类型</b>: 全球 4 资产绝对+相对动量 (510300/513100/518880/511260)</p>
    <p><b>核心</b>: 12M 绝对收益 > 0 且相对排名 Top1 → 持有; 否则转 511260 国债</p>
    <p><b>OOS</b>: ret {v10_dm_oos.get('ann_return', 0)*100:+.2f}% / Sharpe {v10_dm_oos.get('sharpe', 0):.2f} / DD {v10_dm_oos.get('max_dd', 0)*100:.2f}% / Calmar {v10_dm_oos.get('calmar', 0):.3f}</p>
  </div>
  <div class="strategy-card" style="background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 100%); border-color: #FF1493;">
    <h4>🏆 v10 4策略 Vol-parity</h4>
    <p><b>类型</b>: v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%, 波动率平价组合</p>
    <p><b>核心</b>: 4 独立策略 (相关性 -0.005~0.682) + 波动率倒数配权 → Sharpe 提升 28% vs v1.0</p>
    <p><b>OOS</b>: ret {v10_combo_oos.get('ann_return', 0)*100:+.2f}% / Sharpe {v10_combo_oos.get('sharpe', 0):.2f} / DD {v10_combo_oos.get('max_dd', 0)*100:.2f}% / Calmar {v10_combo_oos.get('calmar', 0):.3f}</p>
  </div>
  <div class="strategy-card" style="background: linear-gradient(135deg, #E8F5E9 0%, #FFF3E0 100%); border-color: #9C27B0;">
    <h4>🚀 v10-DynE 混合动态 <span class="legend-box legend-best">OOS Sharpe 2.932</span></h4>
    <p><b>类型</b>: 市场状态 + 回撤控制 + 波动率缩放 三重动态</p>
    <p><b>核心</b>: 熊市→v1 90%; 回撤>5%→切 DualMom; 波动>10%→降仓</p>
    <p><b>OOS</b>: ret {v10_dynE_oos.get('ann_return', 0)*100:+.2f}% / Sharpe {v10_dynE_oos.get('sharpe', 0):.2f} / DD {v10_dynE_oos.get('max_dd', 0)*100:.2f}% / Calmar {v10_dynE_oos.get('calmar', 0):.3f}</p>
    <p><b>说明</b>: 5 个动态方案中 OOS Sharpe 最高, 全部 5 个方案均优于静态基线 (Sharpe 2.461)</p>
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
        (pd.Timestamp("2026-07-24"), "2026-07 v10 上线"),
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
        "all_curves": "全部策略 NAV 曲线叠加. 重点策略用实线, 参考策略用虚线变淡. 蓝色高亮为 OOS 区间. v10 动态方案 (🔥) 和 Vol-parity (🏆) 使用独立颜色. HS300 基准 (深灰 dashdot) 用于对比.",
        "grouped": "3×2 网格分组对比: v1.0 演进 / 进攻型 / v9 银河+中信 / v10 独立策略 / v10 组合+动态 / v7 TV-PR 演进. 每组独立 Y 轴.",
        "alpha": "α = 策略 NAV / HS300 NAV - 1. 持续为正代表跑赢大盘. 全部策略动态迭代.",
        "drawdown": "从历史峰值的最大回撤. v1.0 locked (绿色填充) DD -1.94%.",
        "period_compare": "全期 vs OOS 双柱状图. 金边标记最佳. v10-DynD 在 OOS 期间 Calmar 1.753.",
        "radar": "6 维归一化指标 (年化收益/Sharpe/Calmar/1|DD|/低波动/稳定性), 越靠外越好.",
        "monthly_heatmap": "月度收益热图. 绿色=正收益, 红色=负收益.",
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
  <h2>v0 - v10 策略简述 (按时间顺序)</h2>
  <p style="font-size:12px;color:#888;">从 Stage 8 (CICC 原始复现) 到 v10 动态权重方案 (OOS Calmar 1.753), 完整记录量化策略演进轨迹.</p>
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">📚 基础阶段: v0 → v5</h3>
  {v0_strategy_card}
  {v1_strategy_card}
  {v5_strategy_card}

  {v710_strategy_card}
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">🏆 v9 阶段: 银河方案</h3>
  {v9_overview_card}
  {v9_galaxy_card}
  {v9_gf_card}
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">🎯 v10 阶段 (DualMom + Vol-parity + DynD/DynE)</h3>
  {v10_cards}
  <h3 style="color:#666;border-bottom:1px solid #ddd;padding-bottom:4px;">📊 基准</h3>
  {hs300_card}
</section>
"""

    # 全期 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>v1-v10 业绩曲线对比</title>
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
.legend-deprecated {{ background: #F5F5F5; color: #9E9E9E; border: 1px solid #BDBDBD; text-decoration: line-through; }}
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

<h1>v1-v10 业绩曲线对比</h1>

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
  <strong>核心发现 (OOS 2022-2026):</strong><br>
  • <b>v10 最优</b>: v10-DynD 信号加权 — OOS Calmar <b>1.753</b><br>
  • <b>v10 4策略Vol-parity</b>: v1.0 58% + v9macro 17% + v7.10 13% + DualMom 12% — OOS Calmar <b>1.117</b>, Sharpe 1.304<br>
  • <b>v1.0 locked (历史稳健)</b>: OOS Calmar 1.791, Sharpe 1.51, DD -1.94% (VT 控波动优势)<br>
  • <b>v7.10 TV-PR (进取最优)</b>: expanding-window OOS Calmar 0.662, Sharpe 0.779, DD -20.2%<br>
  • <b>v9 银河方案-动态仓位</b>: OOS Sharpe 1.230, Brinson 归因 71% 仓位 alpha<br>
  • <b>HS300 基准</b>: OOS 年化 {hs300_oos['ann_return']*100:+.2f}%, Calmar {hs300_oos['calmar']:.3f}<br>
  • <b>推荐</b>: v10-DynD (最优) 或 v10 4策略Vol-parity (均衡), 保守用 v1.0 locked
</div>

<div class="key-finding" style="background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 100%); border-left-color: #E91E63; margin-top: 12px;">
  <strong>🏆 v10 阶段 (2026-07-24): 独立策略 + Vol-parity + 动态权重</strong><br>
  • <b>3 个独立策略</b>: DualMom (全球4资产轮动) 等<br>
  • <b>4策略Vol-parity</b>: v1.0 58% + v9macro 17% + v7.10 13% + DualMom 12% — 波动率倒数配权, OOS Calmar 1.117<br>
  • <b>5 个动态权重方案</b>: A 市场状态 / B 波动率 / C 回撤控制 / D 信号加权 / E 混合 — 全部优于静态基线<br>
  • <b>最优</b>: v10-DynD 信号加权 Calmar 1.753<br>
  • <b>Bug修复</b>: DualMom/EPO 周频NAV→日频, metrics() 年化因子 52→252, 指标修正
</div>

<div class="toc">
  <b>📖 图表阅读指南:</b> 实线粗线 = 重点策略 | 虚线 = 参考策略 | 深灰虚线 = HS300 基准 | 金边/绿色填充 = 最佳 (v1.0 locked) | 蓝色高亮 = OOS 区间<br>
  排名按 <b>OOS Calmar</b> (年化收益 / |最大回撤|) 降序排列
</div>

<section id="methodology">
  <h2>方法论与口径说明</h2>

  <h3>1. 统一 ETF 池 (52 只)</h3>
  <div class="methodology">
    <b>主池 44 只</b> (来自 <code>common/universe.py</code>): 6 只 A 股宽基 + 20 只 A 股行业 + 5 只港股 + 6 只商品 + 6 只海外 + 1 只国债<br>
    <b>SmartBeta 8 只</b> (来自 <code>v4/universe_v4.py</code>): 红利低波/低波/质量/价值/现金流等<br>
<b>v5</b>: 使用 44 只 OHLCV (需要 volume 数据)
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
  <h2>OOS 业绩表 (按 Calmar 排序)</h2>
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
    <tr><td>2026-07-24</td><td>v10 上线</td><td>4策略Vol-parity + 动态权重</td><td>v10-DynD Calmar 1.753, 4策略Vol-parity Calmar 1.117</td></tr>
  </table>
</section>

<section id="recommendations">
  <h2>策略推荐与组合方案</h2>

  <h3>1. 单策略推荐 (按风险偏好)</h3>
  <table>
    <tr><th>风险偏好</th><th>推荐策略</th><th>理由</th><th>OOS Calmar</th></tr>
    <tr><td>🏆 最优</td><td>v10-DynD 信号加权</td><td>动量Sharpe动态配权</td><td><b>1.753</b></td></tr>
    <tr><td>⚖️ 均衡</td><td>v10 4策略Vol-parity</td><td>4独立策略波动率平价, Sharpe 1.304</td><td><b>1.117</b></td></tr>
    <tr><td>🛡️ 历史稳健</td><td>v1.0 locked</td><td>VT 控波动, DD -1.94%</td><td>1.791</td></tr>
    <tr><td>🚀 进取</td><td>v7.10 TV-PR</td><td>17 macro + 19 量价, expanding-window OOS</td><td>0.662</td></tr>
    <tr><td>📊 基准</td><td>HS300</td><td>被动指数, 无主动管理成本</td><td>{hs300_oos['calmar']:.3f}</td></tr>
  </table>

  <h3>2. 组合推荐 (多策略分散)</h3>
  <div class="methodology">
    <b>🏆 v10-DynD 信号加权</b> (最优推荐) — OOS Calmar <b>1.753</b>, 5个动态方案中最优<br>
    <br>
    <b>🏆 v10 4策略Vol-parity</b> (均衡推荐) — v1.0 58% + v9macro 17% + v7.10 13% + DualMom 12%, OOS Calmar <b>1.117</b>, Sharpe 1.304<br>
    优势: 4个独立策略 (相关性 -0.005~0.682) 波动率倒数配权, DD 仅 -7.08%<br>
    <br>
    <b>v1.0 80% + v7.10 20%</b> (保守+进攻) — 兼顾低 DD 与高 α<br>
    <br>
    <i>全部 5 个动态方案 (A-E) 均优于静态基线, 证明动态权重有增量价值</i>
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
  <b>⚙️ 统一口径</b>: v0-v10 全策略 | 日频 NAV | 5bp 单边成本 | OOS 2022-2026
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
    print(f"      指标表 (含 HS300 基准行) + 策略简述 (v0-v10) + 关键事件时间线")

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
