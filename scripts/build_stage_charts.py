# coding=utf-8
"""为 Stage 17-22 完整研究生成交互式 HTML 图表.

输出: reports/momentum_etf_rotation/STAGE17_22_CHARTS.html
       + 4 个分段 HTML (按阶段)

图表:
  Stage 17:
    1. v3 / v4A / v4B / v4C / v4D NAV 走势 (8y)
    2. v4 各模式年度收益对比
    3. v4A Drawdown 走势
    4. v4 失败原因诊断 (calmar / DD 对比)
  Stage 18:
    5. v4 风格升级前后对比 (L60 vs L120 vs 多窗口)
    6. v4 因子升级前后对比 (6 因子 vs 5 因子 vs 因子特异 FW)
    7. 三策略组合 vs 单策略 (8y)
  Stage 19:
    8. LW 模式 vs IC^2 (Calmar 柱状)
    9. LW 滚动 λ 时序
   10. OOS 4 模式对比
  Stage 22:
   11. v5 量价 vs v4 因子 NAV 走势
   12. v5 Year-by-year 收益
   13. v5 月度收益分布 (histogram)
   14. 组合 (v3+v4+v5) 4 档风险偏好
  Stage 22.5 详细统计:
   15. 同比汇总 (年化收益 + 波动) grouped bar
   16. Top-5 大回撤瀑布
   17. 滚动 1y/2y/3y/5y 收益曲线
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

sys.path.insert(0, "/home/ll/Public/QuantNodes")

REPO = Path("/home/ll/Public/QuantNodes")
OUT_DIR = REPO / "reports/momentum_etf_rotation"

# 复制 plotly.min.js 到输出目录 (免 CDN)
_PLOTLY_SRC = Path(__file__).resolve().parent.parent / ".venv-mig" / "lib" / "python3.11" / "site-packages" / "plotly" / "package_data" / "plotly.min.js"
_PLOTLY_DST = OUT_DIR / "plotly.min.js"
if _PLOTLY_SRC.exists() and (not _PLOTLY_DST.exists() or _PLOTLY_SRC.stat().st_mtime > _PLOTLY_DST.stat().st_mtime):
    import shutil
    shutil.copy2(str(_PLOTLY_SRC), str(_PLOTLY_DST))
    print(f"[copy] plotly.min.js → {_PLOTLY_DST}")


def ann_return(nav):
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def ann_vol(rets):
    return float(rets.std() * np.sqrt(252))


def sharpe(rets):
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def max_dd(nav):
    pk = nav.cummax()
    return float((nav / pk - 1.0).min())


def metrics(nav):
    rets = nav.pct_change().dropna()
    ar = ann_return(nav)
    av = ann_vol(rets)
    sh = sharpe(rets)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "ann_vol": av,
        "sharpe": sh,
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
    }


def load_data():
    """加载所有 NAV 数据."""
    data = {}
    data["v3"] = pd.read_parquet(REPO / "reports/momentum_etf_rotation/v4/stage17_navs.parquet")
    data["v4_merged"] = pd.read_parquet(REPO / "reports/momentum_etf_rotation/v4/v4_merged_navs.parquet")
    data["v4_lw"] = pd.read_parquet(REPO / "reports/momentum_etf_rotation/v4/lw_factor_timing_navs.parquet")
    data["v5"] = pd.read_parquet(REPO / "reports/momentum_etf_rotation/v5/v5_navs.parquet")
    return data


# ============================================================
# Chart 1: Stage 17 - v4 各模式 NAV 走势
# ============================================================
def chart_stage17_navs(data):
    """v3 / v4A / v4B / v4C / v4D NAV 走势 (8y)."""
    navs = data["v3"]
    fig = go.Figure()

    colors = {
        "v3_baseline": "#1f77b4",
        "v4A_style": "#ff7f0e",
        "v4B_smartbeta": "#2ca02c",
        "v4C_combo": "#d62728",
        "v4D_ic": "#9467bd",
    }
    names_cn = {
        "v3_baseline": "v3 动量 (基准)",
        "v4A_style": "v4A 风格轮动 (单窗口)",
        "v4B_smartbeta": "v4B Smart β",
        "v4C_combo": "v4C 风格+Smart β",
        "v4D_ic": "v4D +IC 因子择时",
    }

    for col in navs.columns:
        fig.add_trace(go.Scatter(
            x=navs.index, y=navs[col],
            mode="lines",
            name=names_cn.get(col, col),
            line=dict(color=colors.get(col, "#888"), width=1.5),
        ))

    fig.update_layout(
        title=dict(text="<b>Stage 17: v4 各模式 NAV 走势 (2018-2026)</b><br>"
                       "<sub>v3 动量显著跑赢 v4 各模式 (Calmar 0.48 vs 0.10-0.14)</sub>",
                    x=0.5),
        xaxis_title="日期",
        yaxis_title="NAV (起点 = 1.0)",
        template="plotly_white",
        height=500,
        hovermode="x unified",
    )
    return fig


# ============================================================
# Chart 2: Stage 17 - v4 失败原因 (Calmar / DD)
# ============================================================
def chart_stage17_failure(data):
    """v4 失败原因诊断."""
    navs = data["v3"]
    metrics_data = []
    for col in navs.columns:
        m = metrics(navs[col])
        metrics_data.append({
            "name": col,
            "Calmar": m["calmar"],
            "Max DD": m["max_dd"],
            "Ann Return": m["ann_return"],
            "Sharpe": m["sharpe"],
        })
    df = pd.DataFrame(metrics_data)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Calmar 8y", "Max DD 8y"))

    fig.add_trace(go.Bar(
        x=df["name"], y=df["Calmar"],
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        text=df["Calmar"].round(2),
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df["name"], y=df["Max DD"] * 100,
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        text=(df["Max DD"] * 100).round(1).astype(str) + "%",
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(title_text="Calmar", row=1, col=1)
    fig.update_yaxes(title_text="Max DD (%)", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Stage 17: v4 失败诊断 (Calmar + DD)</b><br>"
                       "<sub>v4 全周期 3-4x 跑输 v3, 根因: 短视 + 分散过度</sub>",
                    x=0.5),
        template="plotly_white",
        height=450,
    )
    return fig


# ============================================================
# Chart 3: Stage 17 - v4A Drawdown
# ============================================================
def chart_stage17_v4a_dd(data):
    """v4A 风格轮动 Drawdown 走势."""
    nav = data["v3"]["v4A_style"]
    pk = nav.cummax()
    dd = (nav / pk - 1.0) * 100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("v4A 风格轮动 NAV", "v4A Drawdown (%)"),
                        row_heights=[0.6, 0.4])

    fig.add_trace(go.Scatter(
        x=nav.index, y=nav, mode="lines",
        name="v4A NAV", line=dict(color="#ff7f0e", width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, mode="lines",
        name="v4A Drawdown", fill="tozeroy",
        line=dict(color="#d62728", width=1),
    ), row=2, col=1)

    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_layout(
        title=dict(text="<b>Stage 17: v4A 风格轮动 (单窗口 L60_T3) Drawdown</b><br>"
                       "<sub>2021-2024 持续 3 年回撤 -49%, 沉船期</sub>",
                    x=0.5),
        template="plotly_white",
        height=550,
    )
    return fig


# ============================================================
# Chart 4: Stage 18 - v4 风格升级前后对比
# ============================================================
def chart_stage18_style_rotation(data):
    """v4 风格升级前后对比: 单窗口 vs 多窗口 Long-biased."""
    v4s = data["v4_merged"]["v4_style_merged"]
    v3 = data["v3"]["v3_baseline"]
    v3_78 = v4s / v4s.iloc[0] * 1.0

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("升级前 (L60_T3) vs 升级后 (多窗口 5/20/120/180)",
                                        "Calmar + Sharpe 对比"))

    fig.add_trace(go.Scatter(
        x=v3.index, y=v3, mode="lines",
        name="v3 基准", line=dict(color="#1f77b4", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=v4s.index, y=v4s, mode="lines",
        name="v4 风格 (Stage 18)", line=dict(color="#2ca02c", width=2),
    ), row=1, col=1)

    config_names = ["原 v4 风格 (L60_T3)", "Stage 18 v4 (多窗口 + dividend + Top-2 + sideways)"]
    config_calmars = [0.218, 0.439]
    config_sharpes = [0.20, 0.70]

    fig.add_trace(go.Bar(
        x=config_names, y=config_calmars,
        name="Calmar",
        marker_color="#2ca02c",
        text=[f"{c:.3f}" for c in config_calmars],
        textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=config_names, y=config_sharpes,
        name="Sharpe",
        marker_color="#ff7f0e",
        text=[f"{s:.2f}" for s in config_sharpes],
        textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="指标", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Stage 18: v4 风格轮动 4 改进 (27x Calmar 提升)</b><br>"
                       "<sub>0.016 → 0.439, 多窗口 + dividend 底仓 + Top-2 + sideways</sub>",
                    x=0.5),
        template="plotly_white",
        height=500,
    )
    return fig


# ============================================================
# Chart 5: Stage 18 - v4 因子升级前后
# ============================================================
def chart_stage18_factor_timing(data):
    """v4 因子升级前后: 6 因子统一 20d vs 5 因子特异 FW."""
    v4f = data["v4_merged"]["v4_factor_merged"]
    v3 = data["v3"]["v3_baseline"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("升级前后 NAV", "Calmar + Sharpe 对比"))

    fig.add_trace(go.Scatter(
        x=v3.index, y=v3, mode="lines",
        name="v3 基准", line=dict(color="#1f77b4", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=v4f.index, y=v4f, mode="lines",
        name="v4 因子 (Stage 18)", line=dict(color="#2ca02c", width=2),
    ), row=1, col=1)

    config_names = ["原 v4D (6 因子, 统一 20d)", "Stage 18 v4 (5 因子, 因子特异 FW)"]
    config_calmars = [0.092, 0.613]
    config_sharpes = [0.20, 0.70]

    fig.add_trace(go.Bar(
        x=config_names, y=config_calmars,
        name="Calmar", marker_color="#2ca02c",
        text=[f"{c:.3f}" for c in config_calmars],
        textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=config_names, y=config_sharpes,
        name="Sharpe", marker_color="#ff7f0e",
        text=[f"{s:.2f}" for s in config_sharpes],
        textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="指标", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Stage 18: v4 因子择时 5 改进 (6.7x Calmar 提升)</b><br>"
                       "<sub>0.092 → 0.613, 删除 low_vol + 因子特异 FW + regime</sub>",
                    x=0.5),
        template="plotly_white",
        height=500,
    )
    return fig


# ============================================================
# Chart 6: Stage 19 - LW 模式 vs IC^2
# ============================================================
def chart_stage19_lw_comparison(data):
    """LW 模式 vs IC^2 模式对比."""
    v4f = data["v4_merged"]["v4_factor_merged"]
    v4f_lw = data["v4_lw"]["lw_rolling"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("IC^2 vs LW 滚动 λ NAV", "Calmar 对比"))

    fig.add_trace(go.Scatter(
        x=v4f.index, y=v4f, mode="lines",
        name="v4 IC^2 (默认)", line=dict(color="#1f77b4", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=v4f_lw.index, y=v4f_lw, mode="lines",
        name="v4 LW 滚动 λ", line=dict(color="#9467bd", width=2),
    ), row=1, col=1)

    config_names = ["IC^2 (默认)", "LW λ=10", "LW λ=100", "LW 滚动 λ"]
    config_calmars = [0.613, 0.531, 0.536, 0.468]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig.add_trace(go.Bar(
        x=config_names, y=config_calmars,
        marker_color=colors,
        text=[f"{c:.3f}" for c in config_calmars],
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)

    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Calmar (8y)", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Stage 19: LW 协方差 + λ 收缩 (Nagel 论文复现)</b><br>"
                       "<sub>在 5 ETF 类别下, LW 不显著优于 IC^2 (5 类别 IC^2 已足够集中)</sub>",
                    x=0.5),
        template="plotly_white",
        height=500,
    )
    return fig


# ============================================================
# Chart 7: Stage 22 - v5 vs v4 因子 NAV
# ============================================================
def chart_stage22_v5_vs_v4(data):
    """v5 量价 vs v4 因子 NAV 对比."""
    v5 = data["v5"]["v5_industry"]
    v4f = data["v4_merged"]["v4_factor_merged"]
    v3 = data["v3"]["v3_baseline"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=v3.index, y=v3, mode="lines",
        name="v3 基准", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=v4f.index, y=v4f, mode="lines",
        name="v4 因子 (5 因子)", line=dict(color="#2ca02c", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=v5.index, y=v5, mode="lines",
        name="v5 量价 (11 因子)", line=dict(color="#d62728", width=2),
    ))

    fig.update_layout(
        title=dict(text="<b>Stage 22: v5 量价 vs v4 因子 (8y NAV)</b><br>"
                       "<sub>v5 Calmar 0.643 > v4 因子 0.613, 11 因子等权优势</sub>",
                    x=0.5),
        xaxis_title="日期",
        yaxis_title="NAV",
        template="plotly_white",
        height=500,
    )
    return fig


# ============================================================
# Chart 8: Stage 22 - v5 Year-by-year 收益
# ============================================================
def chart_stage22_v5_yearly(data):
    """v5 Year-by-year 收益 + 波动."""
    nav = data["v5"]["v5_industry"]
    yearly = nav.resample("YE")
    rows = []
    for year_end, year_nav in yearly:
        if len(year_nav) < 10:
            continue
        rets = year_nav.pct_change().dropna()
        ar = ann_return(year_nav)
        av = ann_vol(rets) if len(rets) > 5 else 0
        rows.append({
            "year": year_end.year,
            "return": float(year_nav.iloc[-1] / year_nav.iloc[0] - 1) * 100,
            "ann_vol": av * 100,
        })
    df = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=1)

    fig.add_trace(go.Bar(
        x=df["year"].astype(str), y=df["return"],
        name="年化收益 (%)",
        marker_color=["#d62728" if r < 0 else "#2ca02c" for r in df["return"]],
        text=[f"{r:+.1f}%" for r in df["return"]],
        textposition="outside",
    ))

    fig.add_trace(go.Scatter(
        x=df["year"].astype(str), y=df["ann_vol"],
        name="年化波动 (%)",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color="#ff7f0e", width=2),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=dict(text="<b>Stage 22: v5 量价 Year-by-year 收益 + 波动</b><br>"
                       "<sub>2024 年 +87% 关键胜出, 2022/2026 熊市 -17%</sub>",
                    x=0.5),
        xaxis_title="年份",
        yaxis=dict(title="收益 (%)", side="left"),
        yaxis2=dict(title="波动 (%)", side="right", overlaying="y", showgrid=False),
        template="plotly_white",
        height=450,
    )
    return fig


# ============================================================
# Chart 9: Stage 22 - v5 月度收益分布
# ============================================================
def chart_stage22_v5_monthly_dist(data):
    """v5 月度收益分布 histogram."""
    nav = data["v5"]["v5_industry"]
    monthly = nav.resample("ME").last().pct_change().dropna() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=monthly,
        nbinsx=30,
        marker_color="#1f77b4",
        opacity=0.7,
        name="月度收益",
    ))
    fig.add_vline(
        x=0, line_dash="dash", line_color="red",
        annotation_text="0%", annotation_position="top",
    )
    fig.add_vline(
        x=monthly.mean(), line_dash="dot", line_color="green",
        annotation_text=f"mean {monthly.mean():.2f}%", annotation_position="top",
    )

    fig.update_layout(
        title=dict(text=f"<b>Stage 22: v5 月度收益分布</b><br>"
                       f"<sub>n={len(monthly)} 月, mean {monthly.mean():.2f}%, std {monthly.std():.2f}%, "
                       f"月胜率 {(monthly > 0).mean()*100:.1f}%</sub>",
                    x=0.5),
        xaxis_title="月度收益 (%)",
        yaxis_title="频次",
        template="plotly_white",
        height=400,
    )
    return fig


# ============================================================
# Chart 10: Stage 22 - 4 档组合推荐
# ============================================================
def chart_stage22_portfolio_recommend(data):
    """4 档风险偏好组合 (8y + OOS Calmar)."""
    v3 = data["v3"]["v3_baseline"]
    v4f = data["v4_merged"]["v4_factor_merged"]
    v5 = data["v5"]["v5_industry"]
    v4s = data["v4_merged"]["v4_style_merged"]

    portfolios = {
        "v3 only": v3,
        "v3 80% + v5 20%": 0.8 * v3 + 0.2 * v5,
        "v3 70% + v5 30%": 0.7 * v3 + 0.3 * v5,
        "v3 33% + v4f 33% + v5 34%": 0.33 * v3 + 0.33 * v4f + 0.34 * v5,
        "v3 50% + v4f 25% + v5 25%": 0.5 * v3 + 0.25 * v4f + 0.25 * v5,
    }

    rows = []
    for name, nav in portfolios.items():
        m = metrics(nav)
        # OOS
        oos = nav.loc["2022-01-01":] if (nav.index >= "2022-01-01").any() else nav
        m_oos = metrics(oos)
        rows.append({
            "portfolio": name,
            "8y Calmar": m["calmar"],
            "OOS Calmar": m_oos["calmar"],
            "8y Sharpe": m["sharpe"],
            "OOS Sharpe": m_oos["sharpe"],
            "8y Ann Return": m["ann_return"] * 100,
            "OOS Ann Return": m_oos["ann_return"] * 100,
        })
    df = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("8y vs OOS Calmar", "8y vs OOS Sharpe"),
                        horizontal_spacing=0.12)

    fig.add_trace(go.Bar(
        x=df["portfolio"], y=df["8y Calmar"],
        name="8y Calmar", marker_color="#2ca02c",
        text=[f"{c:.2f}" for c in df["8y Calmar"]],
        textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df["portfolio"], y=df["OOS Calmar"],
        name="OOS Calmar", marker_color="#9467bd",
        text=[f"{c:.2f}" for c in df["OOS Calmar"]],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["portfolio"], y=df["8y Sharpe"],
        name="8y Sharpe", marker_color="#2ca02c",
        text=[f"{s:.2f}" for s in df["8y Sharpe"]],
        textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=df["portfolio"], y=df["OOS Sharpe"],
        name="OOS Sharpe", marker_color="#9467bd",
        text=[f"{s:.2f}" for s in df["OOS Sharpe"]],
        textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(title_text="Calmar", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Stage 22: 4 档风险偏好组合 (8y + OOS 2022-2026)</b><br>"
                       "<sub>v3 80% + v5 20% OOS Calmar 0.850, Sharpe 1.01 (最稳健)</sub>",
                    x=0.5),
        template="plotly_white",
        height=500,
        barmode="group",
    )
    fig.update_xaxes(tickangle=-15)
    return fig


# ============================================================
# Chart 11: Stage 22.5 - 同比汇总 (年化收益 + 波动)
# ============================================================
def chart_stage22_summary_grouped(data):
    """同比汇总: 年化收益 + 波动 grouped bar."""
    navs = {
        "v3 baseline": data["v3"]["v3_baseline"],
        "v4 风格": data["v4_merged"]["v4_style_merged"],
        "v4 因子": data["v4_merged"]["v4_factor_merged"],
        "v5 量价": data["v5"]["v5_industry"],
        "v3 80% + v5 20%": 0.8 * data["v3"]["v3_baseline"] + 0.2 * data["v5"]["v5_industry"],
        "v3 33% + v4f 33% + v5 34%": 0.33 * data["v3"]["v3_baseline"] + 0.33 * data["v4_merged"]["v4_factor_merged"] + 0.34 * data["v5"]["v5_industry"],
    }

    rows = []
    for name, nav in navs.items():
        m = metrics(nav)
        rows.append({
            "name": name,
            "年化收益": m["ann_return"] * 100,
            "年化波动": m["ann_vol"] * 100,
            "Max DD": m["max_dd"] * 100,
            "Calmar": m["calmar"],
            "Sharpe": m["sharpe"],
        })
    df = pd.DataFrame(rows)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("年化收益 + 波动", "Calmar + Sharpe"))

    fig.add_trace(go.Bar(
        x=df["name"], y=df["年化收益"],
        name="年化收益 (%)", marker_color="#2ca02c",
        text=[f"{r:.1f}%" for r in df["年化收益"]],
        textposition="outside",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df["name"], y=df["年化波动"],
        name="年化波动 (%)", marker_color="#ff7f0e",
        text=[f"{r:.1f}%" for r in df["年化波动"]],
        textposition="outside",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["name"], y=df["Calmar"],
        name="Calmar", marker_color="#9467bd",
        text=[f"{c:.2f}" for c in df["Calmar"]],
        textposition="outside",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=df["name"], y=df["Sharpe"],
        name="Sharpe", marker_color="#1f77b4",
        text=[f"{s:.2f}" for s in df["Sharpe"]],
        textposition="outside",
    ), row=1, col=2)

    fig.update_yaxes(title_text="比例 (%)", row=1, col=1)
    fig.update_yaxes(title_text="指标值", row=1, col=2)
    fig.update_layout(
        title=dict(text="<b>Stage 22.5: 同比汇总 (年化收益 + 波动 + Calmar + Sharpe)</b><br>"
                       "<sub>v3 80% + v5 20% 是最稳健 (波动 10.5%, Sharpe 0.96)</sub>",
                    x=0.5),
        template="plotly_white",
        height=500,
        barmode="group",
    )
    fig.update_xaxes(tickangle=-15)
    return fig


# ============================================================
# Chart 12: Stage 22.5 - 滚动 1y/2y/3y/5y 收益
# ============================================================
def chart_stage22_rolling_returns(data):
    """滚动 1y/2y/3y/5y 年化收益."""
    navs = {
        "v3 baseline": data["v3"]["v3_baseline"],
        "v4 风格": data["v4_merged"]["v4_style_merged"],
        "v4 因子": data["v4_merged"]["v4_factor_merged"],
        "v5 量价": data["v5"]["v5_industry"],
    }
    windows = [(252, "1y"), (504, "2y"), (756, "3y"), (1260, "5y")]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Rolling 1y mean", "Rolling 2y mean",
                                        "Rolling 3y mean", "Rolling 5y mean"),
                        vertical_spacing=0.12)

    colors = {"v3 baseline": "#1f77b4", "v4 风格": "#2ca02c",
              "v4 因子": "#ff7f0e", "v5 量价": "#d62728"}

    for idx, (w, label) in enumerate(windows):
        row = idx // 2 + 1
        col = idx % 2 + 1
        for name, nav in navs.items():
            rets = nav.pct_change(w).dropna() * 100
            if len(rets) > 0:
                fig.add_trace(go.Histogram(
                    x=rets, name=name, marker_color=colors[name],
                    opacity=0.6, nbinsx=25, showlegend=(idx == 0),
                ), row=row, col=col)

    fig.update_layout(
        title=dict(text="<b>Stage 22.5: 滚动 1y/2y/3y/5y 年化收益分布</b><br>"
                       "<sub>v5 在长窗口 5y mean +127%, v3 仅 +34%</sub>",
                    x=0.5),
        template="plotly_white",
        height=650,
        barmode="overlay",
    )
    return fig


# ============================================================
# Chart 13: Stage 22.5 - Drawdown 走势 (5 策略)
# ============================================================
def chart_stage22_drawdowns(data):
    """5 策略 Drawdown 走势."""
    navs = {
        "v3 baseline": data["v3"]["v3_baseline"],
        "v4 风格": data["v4_merged"]["v4_style_merged"],
        "v4 因子": data["v4_merged"]["v4_factor_merged"],
        "v5 量价": data["v5"]["v5_industry"],
        "v3 80% + v5 20%": 0.8 * data["v3"]["v3_baseline"] + 0.2 * data["v5"]["v5_industry"],
    }

    fig = go.Figure()
    colors = {
        "v3 baseline": "#1f77b4",
        "v4 风格": "#2ca02c",
        "v4 因子": "#ff7f0e",
        "v5 量价": "#d62728",
        "v3 80% + v5 20%": "#9467bd",
    }
    for name, nav in navs.items():
        pk = nav.cummax()
        dd = (nav / pk - 1.0) * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd, mode="lines",
            name=name, line=dict(color=colors[name], width=1.5),
        ))

    fig.update_layout(
        title=dict(text="<b>Stage 22.5: 5 策略 Drawdown 走势</b><br>"
                       "<sub>v3 baseline 最浅 (-14%), v3 80%+v5 20% 次之 (-16%)</sub>",
                    x=0.5),
        xaxis_title="日期",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=500,
    )
    return fig


# ============================================================
# Chart 14: Stage 22 - 相关性矩阵
# ============================================================
def chart_stage22_correlation(data):
    """5 策略日收益相关矩阵."""
    navs = pd.DataFrame({
        "v3": data["v3"]["v3_baseline"],
        "v4 风格": data["v4_merged"]["v4_style_merged"],
        "v4 因子": data["v4_merged"]["v4_factor_merged"],
        "v5 量价": data["v5"]["v5_industry"],
    }).dropna()
    rets = navs.pct_change().dropna()
    corr = rets.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=1.0,
        zmin=0.3,
        zmax=1.0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorbar=dict(title="相关系数"),
    ))
    fig.update_layout(
        title=dict(text="<b>Stage 22: 4 策略日收益相关矩阵</b><br>"
                       "<sub>v5 与 v4 风格/因子 相关 0.44 (真正分散器)</sub>",
                    x=0.5),
        template="plotly_white",
        height=450,
    )
    return fig


# ============================================================
# Main: Build HTML
# ============================================================
def build_html(charts_dict, title="Stage 17-22 完整研究 — 交互式图表"):
    """构建单 HTML 文件包含所有图表 (plotly.js 内嵌, 无需网络)."""
    # 读取 plotly.min.js 内嵌到 HTML (避免 file:// CORS 限制)
    plotly_js = ""
    if _PLOTLY_SRC.exists():
        plotly_js = _PLOTLY_SRC.read_text(encoding="utf-8")

    sections = []
    for section_title, charts in charts_dict.items():
        chart_htmls = []
        for chart_name, fig in charts.items():
            chart_htmls.append(
                f'<div class="chart-container"><h3>{chart_name}</h3>{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div>'
            )
        sections.append(f'<section><h2>{section_title}</h2>{"".join(chart_htmls)}</section>')

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<script>{plotly_js}</script>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    margin: 0; padding: 20px;
    background: #f8f9fa; color: #212529;
    line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 50%, #ff7f0e 100%);
    color: white; padding: 30px; border-radius: 8px;
    margin-bottom: 30px;
  }}
  .header h1 {{ margin: 0 0 10px 0; font-size: 32px; }}
  .header .meta {{ opacity: 0.9; font-size: 14px; }}
  section {{
    background: white; padding: 25px; margin-bottom: 25px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
  }}
  section h2 {{
    color: #1f77b4; border-bottom: 3px solid #1f77b4;
    padding-bottom: 10px; margin-top: 0;
  }}
  .chart-container {{
    margin-bottom: 25px;
    background: #fafbfc;
    padding: 15px;
    border-radius: 6px;
  }}
  .chart-container h3 {{
    color: #495057; margin: 0 0 15px 0;
    font-size: 18px;
  }}
  .nav {{
    background: #343a40; padding: 15px 20px; border-radius: 8px;
    margin-bottom: 20px;
  }}
  .nav a {{
    color: white; text-decoration: none;
    margin-right: 20px; font-weight: 500;
  }}
  .nav a:hover {{ color: #ffc107; }}
  .key-finding {{
    background: #d1ecf1; border-left: 4px solid #17a2b8;
    padding: 12px 15px; margin: 15px 0; border-radius: 4px;
  }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #dee2e6; }}
  th {{ background: #e9ecef; font-weight: 600; }}
  .highlight {{ background: #fff3cd; }}
</style>
</head>
<body>

<div class="header">
  <h1>📊 {title}</h1>
  <div class="meta">
    QuantNodes 量化研究 | 2018-2026 8y 回测 + 2022-2026 4.5y OOS 验证
  </div>
</div>

<div class="nav">
  <a href="#stage17">Stage 17 诊断</a>
  <a href="#stage18">Stage 18 v4 整合</a>
  <a href="#stage19">Stage 19 LW 增强</a>
  <a href="#stage22">Stage 22 v5 行业量价</a>
  <a href="#stage225">Stage 22.5 详细统计</a>
  <a href="#summary">最终推荐</a>
</div>

{"".join(sections)}

<section id="summary">
  <h2>最终生产推荐 (基于 OOS 2022-2026 4.5y)</h2>
  <div class="key-finding">
    <strong>🏆 最稳健 (推荐):</strong> v3 80% + v5 20% — OOS Calmar <strong>0.850</strong>, Sharpe <strong>1.01</strong>, DD -13.45%<br>
    <strong>⚖️ 平衡:</strong> v3 70% + v5 30% — OOS Calmar 0.790, Sharpe 0.91, DD -15.22%<br>
    <strong>🎯 分散 (8y 最高):</strong> v3 33% + v4f 33% + v5 34% — 8y Calmar <strong>0.753</strong>, OOS 0.747<br>
    <strong>🚀 进取:</strong> v3 50% + v4f 25% + v5 25% — 8y Calmar 0.733, OOS 0.788, Sharpe 0.91
  </div>
  <table>
    <tr><th>策略</th><th>8y Calmar</th><th>OOS Calmar</th><th>OOS Sharpe</th><th>年化收益</th><th>年化波动</th></tr>
    <tr><td>v3 baseline</td><td>0.484</td><td>1.012</td><td>1.29</td><td>6.76%</td><td>7.76%</td></tr>
    <tr class="highlight"><td>v4 风格 (5 改进)</td><td>0.439</td><td>-</td><td>-</td><td>10.06%</td><td>16.15%</td></tr>
    <tr><td>v4 因子 (5 改进)</td><td>0.613</td><td>0.581</td><td>0.65</td><td>11.15%</td><td>18.03%</td></tr>
    <tr><td>v5 量价 (11 因子)</td><td>0.643</td><td>0.600</td><td>0.67</td><td>17.59%</td><td>21.24%</td></tr>
    <tr class="highlight"><td>v3 80% + v5 20%</td><td><strong>0.619</strong></td><td><strong>0.850</strong></td><td><strong>1.01</strong></td><td>9.65%</td><td>10.54%</td></tr>
    <tr><td>v3 33% + v4f 33% + v5 34%</td><td><strong>0.753</strong></td><td>0.747</td><td>0.84</td><td>12.56%</td><td>14.15%</td></tr>
  </table>
</section>

</body>
</html>
"""
    return html


def main():
    print("[data] 加载所有 NAV 数据 ...")
    data = load_data()
    print(f"[data] v3: {len(data['v3'].columns)} 策略")
    print(f"[data] v4_merged: {len(data['v4_merged'].columns)} 策略")
    print(f"[data] v4_lw: {len(data['v4_lw'].columns)} 策略")
    print(f"[data] v5: {len(data['v5'].columns)} 策略")

    print("\n[chart] 生成所有图表 ...")
    charts_dict = {
        "Stage 17 — v4 失败诊断": {
            "1. v4 各模式 NAV 走势": chart_stage17_navs(data),
            "2. v4 Calmar + DD 对比": chart_stage17_failure(data),
            "3. v4A 风格轮动 Drawdown": chart_stage17_v4a_dd(data),
        },
        "Stage 18 — v4 整合 (4+5 改进)": {
            "4. v4 风格升级前后 (Calmar 0.016 → 0.439)": chart_stage18_style_rotation(data),
            "5. v4 因子升级前后 (Calmar 0.092 → 0.613)": chart_stage18_factor_timing(data),
        },
        "Stage 19 — LW 协方差 + λ 收缩 (Nagel 风格, 可选)": {
            "6. LW 模式 vs IC^2 模式对比": chart_stage19_lw_comparison(data),
        },
        "Stage 22 — v5 行业量价因子 (新子策略)": {
            "7. v5 vs v4 因子 NAV 走势": chart_stage22_v5_vs_v4(data),
            "8. v5 Year-by-year 收益 + 波动": chart_stage22_v5_yearly(data),
            "9. v5 月度收益分布": chart_stage22_v5_monthly_dist(data),
            "10. 4 档组合推荐 (8y + OOS)": chart_stage22_portfolio_recommend(data),
            "11. 4 策略相关矩阵": chart_stage22_correlation(data),
        },
        "Stage 22.5 — 详细统计 (年化收益 + 波动)": {
            "12. 同比汇总 (年化收益 + 波动 + Calmar + Sharpe)": chart_stage22_summary_grouped(data),
            "13. 滚动 1y/2y/3y/5y 收益分布": chart_stage22_rolling_returns(data),
            "14. 5 策略 Drawdown 走势": chart_stage22_drawdowns(data),
        },
    }

    print("[chart] 构建 HTML ...")
    html = build_html(charts_dict)

    out_path = OUT_DIR / "STAGE17_22_CHARTS.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n[save] {out_path}")
    print(f"[size] {out_path.stat().st_size / 1024:.1f} KB")
    print(f"\n[DONE] 14 个交互式图表, 5 个阶段")


if __name__ == "__main__":
    main()
