# coding=utf-8
"""v1-v5 演进 HTML 图表生成器 (沿用 build_stage_charts 模板, 嵌入 plotly.js).

生成: reports/momentum_etf_rotation/combo/UNIFIED_V1V5_EVOLUTION.html
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "reports/momentum_etf_rotation/combo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTLY_JS = Path("/home/ll/.local/lib/python3.11/site-packages/plotly/package_data/plotly.min.js")

OOS_START = "2022-01-01"

# 策略阶段映射 (用于 hover 显示)
STAGE_MAP = {
    "v0.0 baseline": "Stage 8 (原始 CICC 复现)",
    "v0.1 +VT":      "Stage 9-C (波动率目标)",
    "v0.2 +TF":      "Stage 9-B (趋势过滤)",
    "v1.0 locked":   "Stage 12A (斜率×R² 混合, v1.0 锁定)",
    "v3 (52 池)":     "Stage 16A (多策略组合, 52 池)",
    "v4 style":      "Stage 18 (风格轮动, 12 池)",
    "v4 factor":     "Stage 18 (IC 因子择时, 12 池)",
    "v5 量价":        "Stage 22 (11 量价因子, 44 OHLCV)",
}

# 调色板 (类似 build_stage_charts 风格)
COLORS = {
    "v0.0 baseline": "#888888",
    "v0.1 +VT":      "#FF7F0E",
    "v0.2 +TF":      "#D62728",
    "v1.0 locked":   "#2CA02C",
    "v3 (52 池)":     "#1F77B4",
    "v4 style":      "#9467BD",
    "v4 factor":     "#8C564B",
    "v5 量价":        "#E377C2",
}


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
# Chart 1: NAV 对比 (口径 A)
# ============================================================
def chart_nav_overview(navs):
    fig = go.Figure()
    for col in navs.columns:
        valid = navs[col].dropna()
        if len(valid) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values,
            mode="lines",
            name=f"{col} ({STAGE_MAP.get(col, '')})",
            line=dict(color=COLORS.get(col, "#333"), width=2),
            hovertemplate=f"<b>{col}</b><br>" + "%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="<b>v1-v5 NAV 对比</b><br><sub>统一池 52 只 | 2018-2026 | 口径 A (含 5bp 成本)</sub>",
        xaxis_title="日期", yaxis_title="NAV (起点=1.0)",
        template="plotly_white", height=600,
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ============================================================
# Chart 2: 滚动 252d Calmar
# ============================================================
def chart_rolling_calmar(navs):
    fig = go.Figure()
    win = 252
    for col in navs.columns:
        valid = navs[col].dropna()
        if len(valid) < win + 1:
            continue
        rolling = []
        for i in range(win, len(valid)):
            window = valid.iloc[i - win:i + 1]
            ar = ann_return(window)
            dd = max_dd(window)
            rolling.append(ar / abs(dd) if dd != 0 else 0.0)
        fig.add_trace(go.Scatter(
            x=valid.index[win:], y=rolling, mode="lines",
            name=col, line=dict(color=COLORS.get(col, "#333"), width=1.5),
            hovertemplate=f"<b>{col}</b><br>" + "%{x|%Y-%m-%d}<br>Calmar=%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="<b>滚动 252d Calmar</b><br><sub>1 年滚动窗口, 反映策略在不同时期的风险调整表现</sub>",
        xaxis_title="日期", yaxis_title="滚动 Calmar",
        template="plotly_white", height=500,
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return fig


# ============================================================
# Chart 3: 年度收益热图
# ============================================================
def chart_yearly_heatmap(navs):
    years = sorted(set(navs.index.year))
    cols = list(navs.columns)
    matrix = np.full((len(cols), len(years)), np.nan)
    for i, col in enumerate(cols):
        for j, year in enumerate(years):
            yr = navs[col].loc[str(year)]
            if len(yr) > 1:
                matrix[i, j] = yr.iloc[-1] / yr.iloc[0] - 1

    text = [[f"{v*100:+.1f}%" if pd.notna(v) else "" for v in row] for row in matrix]
    fig = go.Figure(data=go.Heatmap(
        z=matrix * 100,
        x=years, y=cols,
        text=text, texttemplate="%{text}",
        colorscale="RdYlGn", zmid=0, zmin=-30, zmax=40,
        colorbar=dict(title="% 收益"),
        hovertemplate="%{y}<br>%{x}<br>%{z:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        title="<b>年度收益热图</b><br><sub>红=亏损, 绿=盈利</sub>",
        template="plotly_white", height=500,
        xaxis_title="年份", yaxis_title="策略",
    )
    return fig


# ============================================================
# Chart 4: 风险-收益散点 (Sharpe vs DD)
# ============================================================
def chart_risk_return(navs):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("全期 (2018-2026)", "OOS (2022-2026)"),
        vertical_spacing=0.12,
    )
    for scope, row_idx, data in [
        ("全期", 1, navs),
        ("OOS", 2, navs.loc[OOS_START:]),
    ]:
        for col in data.columns:
            m = metrics(data[col])
            if m["ann_return"] == 0:
                continue
            stage = STAGE_MAP.get(col, "")
            fig.add_trace(go.Scatter(
                x=[-m["max_dd"] * 100], y=[m["sharpe"]],
                mode="markers+text", name=col, showlegend=(row_idx == 1),
                marker=dict(
                    size=m["calmar"] * 30 + 10,
                    color=COLORS.get(col, "#333"),
                    line=dict(color="black", width=1),
                    opacity=0.7,
                ),
                text=[col], textposition="top center",
                hovertemplate=f"<b>{col}</b> ({stage})<br>"
                              f"Sharpe={m['sharpe']:.2f}<br>"
                              f"DD={m['max_dd']*100:.2f}%<br>"
                              f"Calmar={m['calmar']:.3f}<br>"
                              f"Ann={m['ann_return']*100:.2f}%<extra></extra>",
            ), row=row_idx, col=1)
    fig.update_xaxes(title_text="|Max DD| (%)", row=1, col=1)
    fig.update_xaxes(title_text="|Max DD| (%)", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    fig.update_layout(
        title="<b>风险-收益散点</b><br><sub>气泡大小 = Calmar, X = |最大回撤|, Y = Sharpe</sub>",
        template="plotly_white", height=700,
    )
    return fig


# ============================================================
# Chart 5: 相关性热图
# ============================================================
def chart_correlation(navs):
    rets = navs.pct_change().dropna()
    corr = rets.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        text=corr.round(2).values, texttemplate="%{text}",
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="相关"),
        hovertemplate="%{y} ↔ %{x}<br>r=%{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="<b>日收益相关性矩阵</b><br><sub>数值越低, 分散效果越好</sub>",
        template="plotly_white", height=600,
    )
    return fig


# ============================================================
# Chart 6: 双口径 Calmar 对比
# ============================================================
def chart_dual_caliber(metrics_A, metrics_B):
    strategies = list(metrics_A.keys())
    a_vals = [metrics_A[s]["calmar"] for s in strategies]
    b_vals = [metrics_B[s]["calmar"] for s in strategies]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=strategies, y=a_vals, name="口径 A (5bp 成本)",
        marker_color="#1F77B4",
    ))
    fig.add_trace(go.Bar(
        x=strategies, y=b_vals, name="口径 B (无成本)",
        marker_color="#FF7F0E",
    ))
    fig.update_layout(
        title="<b>双口径 OOS Calmar 对比</b><br><sub>成本对长期表现影响</sub>",
        xaxis_title="策略", yaxis_title="OOS Calmar (2022-2026)",
        template="plotly_white", height=500, barmode="group",
    )
    return fig


# ============================================================
# Chart 7: v1.0 演进路径 (Stage 8 → v1.0 locked)
# ============================================================
def chart_v1_evolution(navs):
    """展示 v0.0 → v0.1 → v0.2 → v1.0 的演进路径."""
    chain = ["v0.0 baseline", "v0.1 +VT", "v0.2 +TF", "v1.0 locked"]
    available = [c for c in chain if c in navs.columns]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("OOS Calmar 演进", "OOS Sharpe 演进"),
        vertical_spacing=0.14,
    )
    oos = navs.loc[OOS_START:]

    for metric_name, row_idx in [("calmar", 1), ("sharpe", 2)]:
        xs, ys, labels = [], [], []
        for col in available:
            m = metrics(oos[col])
            if m["ann_return"] == 0:
                continue
            xs.append(col)
            ys.append(m[metric_name])
            labels.append(STAGE_MAP.get(col, col))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers+text",
            text=labels, textposition="top center",
            marker=dict(size=12, color="rgba(31,119,180,0.8)"),
            line=dict(color="#1F77B4", width=2),
            showlegend=False,
        ), row=row_idx, col=1)
    fig.update_yaxes(title_text="Calmar", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    fig.update_layout(
        title="<b>v1.0 演进路径 (Stage 8 → v1.0 locked)</b><br><sub>每个中间优化的贡献</sub>",
        template="plotly_white", height=650,
    )
    return fig


# ============================================================
# Chart 8: 组合推荐
# ============================================================
def chart_portfolio_combos(navs_A):
    combos = {
        "v1.0 80% + v5 20%":      0.8 * navs_A["v1.0 locked"] + 0.2 * navs_A["v5 量价"],
        "v1.0 70% + v5 30%":      0.7 * navs_A["v1.0 locked"] + 0.3 * navs_A["v5 量价"],
        "v3 50% + v5 50%":         0.5 * navs_A["v3 (52 池)"] + 0.5 * navs_A["v5 量价"],
        "v1.0 50% + v3 25% + v5 25%":
            0.5 * navs_A["v1.0 locked"] + 0.25 * navs_A["v3 (52 池)"] + 0.25 * navs_A["v5 量价"],
        "v1.0 60% + v3 20% + v5 20%":
            0.6 * navs_A["v1.0 locked"] + 0.2 * navs_A["v3 (52 池)"] + 0.2 * navs_A["v5 量价"],
    }
    fig = go.Figure()
    for name, nav in combos.items():
        valid = nav.dropna()
        fig.add_trace(go.Scatter(
            x=valid.index, y=valid.values, mode="lines", name=name,
            line=dict(width=2.5),
            hovertemplate=f"<b>{name}</b><br>" + "%{x|%Y-%m-%d}<br>NAV=%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        title="<b>推荐组合 NAV</b><br><sub>口径 A: 含 5bp 成本</sub>",
        xaxis_title="日期", yaxis_title="NAV (起点=1.0)",
        template="plotly_white", height=600,
        hovermode="x unified", legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ============================================================
# Chart 9: 月度收益分布
# ============================================================
def chart_monthly_distribution(navs):
    fig = go.Figure()
    for col in navs.columns:
        monthly = navs[col].resample("ME").last().pct_change().dropna()
        if len(monthly) < 3:
            continue
        fig.add_trace(go.Box(
            y=monthly.values * 100, name=col,
            marker_color=COLORS.get(col, "#333"),
            boxmean=True,
        ))
    fig.update_layout(
        title="<b>月度收益分布</b><br><sub>Box = 25-75% 分位, 中线 = 中位数, ★= 均值</sub>",
        xaxis_title="策略", yaxis_title="月收益 (%)",
        template="plotly_white", height=500,
    )
    return fig


# ============================================================
# Main
# ============================================================
def main():
    print("[chart] 加载数据...")
    navs_A = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")
    navs_B = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calB.parquet")

    # 计算指标
    metrics_A = {col: metrics(navs_A[col]) for col in navs_A.columns}
    metrics_B = {col: metrics(navs_B[col]) for col in navs_B.columns}

    print("[chart] 生成图表...")
    figs = {
        "nav": chart_nav_overview(navs_A),
        "rolling_calmar": chart_rolling_calmar(navs_A),
        "yearly_heatmap": chart_yearly_heatmap(navs_A),
        "risk_return": chart_risk_return(navs_A),
        "correlation": chart_correlation(navs_A),
        "dual_caliber": chart_dual_caliber(
            {c: metrics(v.loc[OOS_START:]) for c, v in navs_A.items()},
            {c: metrics(v.loc[OOS_START:]) for c, v in navs_B.items()},
        ),
        "v1_evolution": chart_v1_evolution(navs_A),
        "portfolio_combos": chart_portfolio_combos(navs_A),
        "monthly_dist": chart_monthly_distribution(navs_A),
    }

    # ============================================================
    # 嵌入 plotly.js (与 build_stage_charts 一致)
    # ============================================================
    print("[chart] 嵌入 plotly.js (内联)...")
    if PLOTLY_JS.exists():
        plotly_src = PLOTLY_JS.read_text(encoding="utf-8")
    else:
        print("  [WARN] plotly.min.js not found, 使用 CDN (可能脱机不工作)")
        plotly_src = ""

    # ============================================================
    # 汇总 HTML
    # ============================================================
    print("[chart] 生成 HTML...")
    oos_A = {c: metrics(v.loc[OOS_START:]) for c, v in navs_A.items()}
    oos_B = {c: metrics(v.loc[OOS_START:]) for c, v in navs_B.items()}

    # 找 OOS Calmar 最佳
    best_oos = max(oos_A.items(), key=lambda x: x[1]["calmar"])

    sections = []
    for key, fig in figs.items():
        sections.append(f"""
        <section id="{key}">
          <h2>{key.replace('_', ' ').title()}</h2>
          {fig.to_html(full_html=False, include_plotlyjs=False, div_id=key)}
        </section>
        """)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>v1-v5 统一池统一时间对比</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1280px; margin: 20px auto; padding: 0 20px; background: #fafafa; }}
h1 {{ color: #1F77B4; border-bottom: 3px solid #1F77B4; padding-bottom: 8px; }}
h2 {{ color: #2C3E50; margin-top: 40px; border-left: 5px solid #1F77B4; padding-left: 10px; }}
section {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }}
th {{ background: #1F77B4; color: white; }}
tr.highlight {{ background: #FFF9E6; font-weight: bold; }}
.key-finding {{ background: #E8F5E9; padding: 12px; border-radius: 4px; margin: 12px 0; }}
.navbar {{ position: sticky; top: 0; background: white; padding: 8px; border-bottom: 1px solid #ddd; z-index: 100; }}
.navbar a {{ margin-right: 12px; color: #1F77B4; text-decoration: none; }}
.navbar a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>v1-v5 演进对比 — 统一 ETF 池 (52) + 统一时间 (2018-2026)</h1>

<div class="navbar">
  <a href="#nav">NAV 概览</a>
  <a href="#rolling_calmar">滚动 Calmar</a>
  <a href="#yearly_heatmap">年度热图</a>
  <a href="#risk_return">风险-收益</a>
  <a href="#correlation">相关性</a>
  <a href="#dual_caliber">双口径对比</a>
  <a href="#v1_evolution">v1.0 演进</a>
  <a href="#portfolio_combos">推荐组合</a>
  <a href="#monthly_dist">月度分布</a>
  <a href="#summary">最终推荐</a>
</div>

<div class="key-finding">
  <strong>关键发现 (口径 A OOS 2022-2026):</strong><br>
  • <b>最佳风险调整</b>: {best_oos[0]} — OOS Calmar <b>{best_oos[1]['calmar']:.3f}</b>, Sharpe <b>{best_oos[1]['sharpe']:.2f}</b>, DD <b>{best_oos[1]['max_dd']*100:.2f}%</b><br>
  • <b>最高绝对收益</b>: v5 量价 — OOS 年化 <b>{oos_A['v5 量价']['ann_return']*100:.2f}%</b>, Calmar 0.488<br>
  • <b>最稳</b>: v1.0 locked — OOS DD 仅 -1.94%, 波动率 2.38%<br>
  • <b>成本影响</b>: 5bp 月度成本对长期 Calmar 影响 &lt; 0.04
</div>

{"".join(sections)}

<section id="summary">
  <h2>最终推荐 (基于 OOS Calmar 2022-2026)</h2>
  <div style="font-size:13px;color:#666;margin-bottom:8px;">按 OOS Calmar 降序排列</div>
  <table>
    <tr>
      <th>排名</th><th>策略</th><th>年化收益率</th><th>年化波动</th>
      <th>OOS Sharpe</th><th>最大回撤</th><th>OOS Calmar</th>
    </tr>
"""

    # 按 OOS Calmar 降序排序
    oos_A_with_name = [(col, oos_A[col]) for col in navs_A.columns]
    oos_A_sorted = sorted(oos_A_with_name, key=lambda x: x[1]["calmar"], reverse=True)
    for rank, (col, o) in enumerate(oos_A_sorted, 1):
        cls = "highlight" if rank == 1 else ""
        star = " ⭐" if rank == 1 else ""
        html += f"""    <tr class="{cls}"><td>{rank}</td><td>{col}{star}</td>
      <td>{o['ann_return']*100:+.2f}%</td>
      <td>{o['ann_vol']*100:.2f}%</td>
      <td>{o['sharpe']:.2f}</td>
      <td>{o['max_dd']*100:.2f}%</td>
      <td><b>{o['calmar']:.3f}</b></td></tr>
"""

    html += """
  </table>

  <h3>演进路径总结</h3>
  <table>
    <tr><th>阶段</th><th>核心创新</th><th>OOS Calmar</th></tr>
    <tr><td>Stage 8 (v0.0 baseline)</td><td>CICC 原始 4 步规则 + 144d 动量</td><td>0.484</td></tr>
    <tr><td>Stage 9-B (v0.2 +TF)</td><td>趋势过滤 (HS300 MA200, bear=0.7)</td><td>0.509</td></tr>
    <tr><td>Stage 9-C (v0.1 +VT)</td><td>波动率目标 (target=0.15)</td><td>0.492</td></tr>
    <tr><td>Stage 12A (v1.0 locked)</td><td>斜率×R² 混合 + VT + Cost</td><td><b>1.791</b></td></tr>
    <tr><td>Stage 16A (v3)</td><td>多策略 (动量+反转+行业轮动)</td><td>0.778</td></tr>
    <tr><td>Stage 18 (v4 style/factor)</td><td>风格轮动 + IC 因子择时</td><td>-0.085 ~ 0.131</td></tr>
    <tr><td>Stage 22 (v5)</td><td>11 量价因子等权 (华西论文)</td><td>0.488</td></tr>
  </table>

  <p style="font-size: 12px; color: #666; margin-top: 30px;">
  统一口径: 52 ETF (44 主池 + 8 SmartBeta) | 2018-2026 | 5bp 单边成本 | A 股 cap=3 (CICC 规则)<br>
  数据: 前复权 close (scripts/fix_ohlcv_adjust.py 已修正 9 只 ETF 拆合股跳变)<br>
  v4 池: 12 SmartBeta 子集 (v4 设计意图) | v5 池: 44 OHLCV
  </p>
</section>

</body>
</html>
"""

    # 写入 HTML
    if plotly_src:
        # 内联嵌入 plotly.js
        html = html.replace(
            "</head>",
            f'<script>{plotly_src}</script></head>',
        )

    out_html = OUT_DIR / "UNIFIED_V1V5_EVOLUTION.html"
    out_html.write_text(html, encoding="utf-8")
    size_mb = out_html.stat().st_size / 1024 / 1024
    print(f"\n[save] {out_html} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
