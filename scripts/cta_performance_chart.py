# coding=utf-8
"""cta_performance_chart.py — CTA 业绩曲线图 (plotly).

面板 (本版: NAV净值曲线; 回撤/逐年Sharpe面板后续扩展):
    P1: NAV 2018-2025, 起点归一=1, 对数轴
        主线: 主CTA(含EC, FINAL_LUMP_CFG定版close) / 主CTA(剔EC保守) / R2组合(open净口径)
        细线: 4个子策略 (legend点击可隐藏)
        标注: 2024-07→09 能化whipsaw回撤窗口; EC依赖注记

数据源 (全部已落盘, 零回测):
    主CTA含EC  : overfit_final_navs/基线_严格OOS0.5,60_2018.parquet (重构对账golden)
    剔EC       : tv150_calibrated_exEC_nav.csv
    R2         : portfolio_R2_nav.csv (252计数净口径, 与audit文档同源性存疑, 仅作曲线)
    子策略     : rebuild_subnav_{CU溢价TC,T10y,jjm_coke,ZN库存}_open.csv (PIT修正版)

输出: reports/cta/cta_performance_curve.html (交互) / .png (kaleido)
      reports/cta/cta_performance_curve_nav.csv    (7列NAV宽表, 与图同源)
      reports/cta/cta_performance_curve_yearly.csv (年度收益/Sharpe长表)

用法: python3 scripts/cta_performance_chart.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, "/home/ll/Public/QuantNodes")

REPORT = Path("reports/cta")

MAIN_STYLES = {
    "主CTA (含EC, 定版close)": ("#1f77b4", 2.6),
    "主CTA (剔EC保守)": ("#ff7f0e", 2.2),
    "R2组合 (open净口径)": ("#2ca02c", 2.2),
}
SUB_FILES = {
    "子策略: CU溢价TC": "rebuild_subnav_CU溢价TC_open.csv",
    "子策略: T10y": "rebuild_subnav_T10y_open.csv",
    "子策略: jjm_coke": "rebuild_subnav_jjm_coke_open.csv",
    "子策略: ZN库存": "rebuild_subnav_ZN库存_open.csv",
}


def load_navs() -> dict[str, pd.Series]:
    """读全部 NAV, 起点归一 = 1, 统一截断到 2025-12-31 (评估口径).

    注: 部分子策略文件被后续重跑覆盖过 (含2026数据), 与主线区间不一致,
    故统一截断, 保证时间轴对齐.
    """
    navs: dict[str, pd.Series] = {}
    navs["主CTA (含EC, 定版close)"] = pd.read_parquet(
        REPORT / "overfit_final_navs" / "基线_严格OOS0.5,60_2018.parquet")["nav"]
    navs["主CTA (剔EC保守)"] = pd.read_csv(
        REPORT / "tv150_calibrated_exEC_nav.csv", index_col=0, parse_dates=True).iloc[:, 0]
    navs["R2组合 (open净口径)"] = pd.read_csv(
        REPORT / "portfolio_R2_nav.csv", index_col=0, parse_dates=True).iloc[:, 0]
    for name, f in SUB_FILES.items():
        navs[name] = pd.read_csv(REPORT / f, index_col=0, parse_dates=True).iloc[:, 0]
    return {k: (v[v.index <= "2025-12-31"] / v.iloc[0]) for k, v in navs.items()}


def nav_panel(navs: dict[str, pd.Series]) -> go.Figure:
    """P1: 净值曲线 (对数轴 + 年度分隔 + 事件标注)."""
    fig = go.Figure()
    for name, nav in navs.items():
        if name in MAIN_STYLES:
            color, width = MAIN_STYLES[name]
            dash = None
            opacity = 1.0
        else:
            color, width, dash, opacity = "#8c8c8c", 1.1, "dot", 0.65
        fig.add_trace(go.Scatter(
            x=nav.index, y=nav.values, name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash), opacity=opacity,
            hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
        ))

    # 年度分隔线
    years = pd.date_range("2019-01-01", "2026-01-01", freq="YS")
    for y in years:
        fig.add_vline(x=y.isoformat(), line_width=1, line_color="#e0e0e0")

    # 2024 能化 whipsaw 回撤窗口 (2024-07-04 → 09-10, -6.14%, 11个月修复)
    fig.add_vrect(
        x0="2024-07-04", x1="2024-09-10", fillcolor="#d62728", opacity=0.08,
        line_width=0,
        annotation_text="能化whipsaw<br>-6.1%", annotation_position="top left",
        annotation_font_size=11, annotation_font_color="#d62728",
    )

    fig.update_layout(
        title=dict(
            text=("CTA 业绩净值曲线 (2018-2025, 起点归一=1, 对数轴)<br>"
                  "<sup>主CTA=66品种 FINAL_LUMP_CFG定版 close口径 | 剔EC=保守对照 | "
                  "R2=55%主CTA×动态杠杆+CU25%+T10y10%+jjm10% (open净口径) | "
                  "灰虚线=子策略(legend点击隐藏) | 主CTA 2018为TVPR预热期≈平</sup>"),
            x=0.01, font=dict(size=17),
        ),
        yaxis=dict(type="log", title="NAV (对数)",
                   range=[np.log10(0.68), np.log10(4.6)],  # log轴range用log10值
                   gridcolor="#f0f0f0"),
        xaxis=dict(title="", gridcolor="#f0f0f0"),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=680, width=1440,
        font=dict(family="Noto Serif CJK SC, sans-serif", size=12),
    )

    fig.add_annotation(
        x=0.02, y=0.97, xref="paper", yref="paper", showarrow=False,
        xanchor="left", yanchor="top",
        text=("注: 蓝线2024Q1垂直跳升 = EC红海行情 (当年正盈利68%来自EC单品种,<br>"
              "剔EC橙色线当年≈0); 红区 = 2024-07→09能化whipsaw回撤 -6.1% (11个月修复)"),
        font=dict(size=12, color="#8c564b"), bgcolor="rgba(255,255,255,0.85)",
        align="left", bordercolor="#e8d4c8", borderwidth=1,
    )
    return fig


def export_nav_csv(navs: dict[str, pd.Series]) -> pd.DataFrame:
    """导出 NAV 宽表 CSV (与图同源, 起点归一=1, 截至2025-12-31)."""
    nav_df = pd.DataFrame(navs).round(6)
    nav_df.index.name = "date"
    out = REPORT / "cta_performance_curve_nav.csv"
    nav_df.to_csv(out)
    print(f"已保存: {out}  ({len(nav_df)}行 × {nav_df.shape[1]}列)")
    return nav_df


def export_yearly_csv(navs: dict[str, pd.Series]) -> None:
    """导出年度收益/Sharpe 长表 (tidy格式)."""
    rows = []
    for name, nav in navs.items():
        ret = nav.pct_change().dropna()
        yret = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
        yshp = ret.groupby(ret.index.year).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else np.nan)
        for y in yret.index:
            rows.append({"year": int(y), "series": name,
                         "yearly_ret": round(float(yret.loc[y]), 6),
                         "yearly_sharpe": (round(float(yshp.loc[y]), 4)
                                           if pd.notna(yshp.loc[y]) else "")})
    out = REPORT / "cta_performance_curve_yearly.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"已保存: {out}  ({len(rows)}行)")


def main() -> None:
    navs = load_navs()
    print("已加载 NAV 序列:")
    for name, nav in navs.items():
        print(f"  {name}: {nav.index.min().date()}~{nav.index.max().date()} "
              f"终值 {nav.iloc[-1]:.3f}")
    export_nav_csv(navs)
    export_yearly_csv(navs)
    fig = nav_panel(navs)
    out = REPORT / "cta_performance_curve"
    fig.write_html(str(out) + ".html", include_plotlyjs=True)
    fig.write_image(str(out) + ".png", width=1440, height=680, scale=2)
    print(f"已保存: {out}.html / {out}.png")


if __name__ == "__main__":
    main()
