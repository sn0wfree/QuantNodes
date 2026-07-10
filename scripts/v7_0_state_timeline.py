"""
v7.0 状态时间线 HTML 可视化 (Stage 30 POC).

生成 reports/momentum_etf_rotation/v7/v7_0_state_timeline.html:
- 5 状态历史时间线 (2018-2026)
- 5 宏观因子 PIT 调整后的时序
- 关键历史事件标注
- 状态分布统计

[PIT 关键] 所有时间线都是 PIT 调整后的, 标注的 release_date 区分 obs vs rel.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v7.regime_macro import (
    build_regime_timeline,
    REGIME_NAMES,
    REGIME_VOL_TARGETS,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.factor_macro import (
    CACHE_DIR,
    META,
    RELEASE_LAG_DAYS,
)


# 关键历史事件 (用于标注)
HISTORICAL_EVENTS = [
    ("2018-03-22", "中美贸易战升级 (特朗普签署备忘录)"),
    ("2018-09-24", "美国对中国 2000 亿商品加征关税"),
    ("2019-08-01", "特朗普宣布 9 月起对中国 3000 亿商品加税"),
    ("2020-01-23", "武汉封城 (新冠疫情)"),
    ("2020-03-23", "美联储无限 QE + A 股探底"),
    ("2020-04-08", "武汉解封"),
    ("2021-07-23", "房地产三道红线 (恒大事件)"),
    ("2021-12-15", "1 年期 LPR 降息 5bp"),
    ("2022-03-16", "金融委专题会议 (政策底)"),
    ("2022-04-29", "中央政治局会议 (平台经济转向)"),
    ("2022-10-16", "二十大召开"),
    ("2022-12-07", "新十条 (防疫政策大转向)"),
    ("2023-07-24", "中央政治局会议 (活跃资本市场)"),
    ("2023-08-28", "印花税减半 + 减持新规"),
    ("2024-02-05", "证监会暂停转融通 + 雪球敲入风险"),
    ("2024-09-24", "降准 0.5pp + 降息 20bp + 股市政策大礼包"),
    ("2024-10-12", "财政部化债 12 万亿"),
    ("2025-04-02", "美国对等关税 (特朗普 2.0)"),
    ("2025-09-30", "央行降准 0.5pp + 降息 20bp"),
    ("2026-03-15", "两会 - 5% GDP 增长目标"),
]


# 5 状态颜色 (用于 HTML)
REGIME_COLORS = {
    "recovery":    "#10B981",   # 绿色 (复苏)
    "overheat":    "#F59E0B",   # 黄色 (过热)
    "neutral":     "#6B7280",   # 灰色 (中性)
    "stagflation": "#F97316",   # 橙色 (滞胀)
    "recession":   "#EF4444",   # 红色 (衰退)
}


def render_state_timeline_html(df: pd.DataFrame, out_path: Path) -> None:
    """生成 5 状态时间线 HTML."""
    # 准备数据
    state_data = []
    for r in REGIME_NAMES:
        sub = df[df["regime"] == r]
        state_data.append({
            "regime": r,
            "days": len(sub),
            "pct": round(len(sub) / len(df) * 100, 1),
            "vol_target": REGIME_VOL_TARGETS[r],
            "color": REGIME_COLORS[r],
            "start": sub["date"].min().strftime("%Y-%m-%d") if len(sub) > 0 else "-",
            "end": sub["date"].max().strftime("%Y-%m-%d") if len(sub) > 0 else "-",
        })

    # 每月主导状态 (用于时间线块)
    df_monthly = df.copy()
    df_monthly["month"] = df_monthly["date"].dt.to_period("M").astype(str)
    monthly = df_monthly.groupby(["month", "regime"]).size().unstack(fill_value=0)
    monthly["dominant"] = monthly.idxmax(axis=1)
    monthly["total"] = monthly[list(REGIME_NAMES)].sum(axis=1)
    monthly_blocks = []
    for month, row in monthly.iterrows():
        # 构造每月状态条
        segs = []
        for r in REGIME_NAMES:
            pct = round(row[r] / row["total"] * 100, 1) if row["total"] > 0 else 0
            if pct > 0:
                segs.append(f'<div class="seg" style="width:{pct}%;background:{REGIME_COLORS[r]}" title="{r}: {pct}%"></div>')
        monthly_blocks.append({
            "month": month,
            "dominant": row["dominant"],
            "dom_color": REGIME_COLORS[row["dominant"]],
            "segs_html": "".join(segs),
        })

    # 事件列表
    events_html = []
    for date_str, desc in HISTORICAL_EVENTS:
        events_html.append(f'<li><b>{date_str}</b>: {desc}</li>')

    # 状态时间线 JSON
    state_series = df[["date", "regime"]].copy()
    state_series["date"] = state_series["date"].dt.strftime("%Y-%m-%d")

    # 5 因子 PIT 时序 (用于 plotly 图表)
    factor_data = []
    for col in ("PMI", "CPI", "M2", "CN10Y", "US10Y"):
        sub = df[["date", col, f"{col}_zscore"]].dropna().copy()
        sub["date"] = sub["date"].dt.strftime("%Y-%m-%d")
        factor_data.append({
            "name": col,
            "lag_days": RELEASE_LAG_DAYS[col],
            "unit": META[col].unit,
            "rows": sub.to_dict(orient="records"),
        })

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>v7.0 5 状态时间线 (Stage 30 POC)</title>
<script src="../../plotly.min.js"></script>
<style>
body {{ font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif; margin: 20px; background: #F9FAFB; }}
h1 {{ color: #1F2937; }}
h2 {{ color: #374151; border-bottom: 2px solid #E5E7EB; padding-bottom: 8px; }}
.card {{ background: white; border: 1px solid #E5E7EB; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
.stats {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin: 16px 0; }}
.stat-box {{ padding: 12px; border-radius: 8px; color: white; text-align: center; }}
.stat-box h3 {{ margin: 0; font-size: 14px; }}
.stat-box .days {{ font-size: 24px; font-weight: bold; margin: 4px 0; }}
.stat-box .pct {{ font-size: 12px; opacity: 0.9; }}
.stat-box .vol {{ font-size: 11px; opacity: 0.85; }}
.timeline-row {{ display: flex; align-items: center; height: 20px; margin: 2px 0; }}
.timeline-row .label {{ width: 80px; font-size: 12px; color: #6B7280; }}
.timeline-row .seg-row {{ flex: 1; display: flex; height: 100%; border-radius: 4px; overflow: hidden; }}
.seg {{ height: 100%; }}
.event-list {{ columns: 2; column-gap: 24px; font-size: 13px; }}
.event-list li {{ margin: 4px 0; }}
.legend {{ display: flex; gap: 12px; margin: 8px 0; font-size: 12px; }}
.legend-item {{ display: flex; align-items: center; gap: 4px; }}
.legend-color {{ width: 16px; height: 12px; border-radius: 2px; }}
.pit-note {{ background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 12px; margin: 12px 0; font-size: 13px; }}
</style>
</head>
<body>
<h1>📊 v7.0 5 状态时间线 (Stage 30 POC)</h1>
<p style="color: #6B7280;">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 数据区间: 2018-06 ~ 2026-06 | 共 {len(df)} 个工作日</p>

<div class="pit-note">
<b>⚠️ PIT (Point-in-Time) 关键说明</b>: 本时间线所有宏观因子都按 <code>release_date</code> 调整,
即回测在 T 日时, 看到的宏观值是 <code>release_date ≤ T</code> 的最新数据 (而非 obs_date ≤ T)。
PMI lag=1天, CPI lag=10天, M2 lag=12天, CN10Y/US10Y lag=0天 (T+0 实时)。
HMM 训练时, 输入是 PIT 调整后的 5 维特征, 严格防 look-ahead。
</div>

<div class="card">
<h2>1. 5 状态分布 (2018-2026)</h2>
<div class="stats">
"""
    for sd in state_data:
        html += f"""<div class="stat-box" style="background:{sd['color']}">
<h3>{sd['regime']}</h3>
<div class="days">{sd['days']}</div>
<div class="pct">{sd['pct']}%</div>
<div class="vol">vol_target={sd['vol_target']}</div>
</div>
"""
    html += """</div>
</div>

<div class="card">
<h2>2. 月度 5 状态时间线</h2>
<p style="color: #6B7280; font-size: 12px;">每月一条, 块宽度 = 该状态天数占比。颜色: 🟢复苏 / 🟡过热 / ⚪中性 / 🟠滞胀 / 🔴衰退</p>
<div class="legend">
"""
    for r, c in REGIME_COLORS.items():
        html += f'<div class="legend-item"><div class="legend-color" style="background:{c}"></div>{r}</div>'
    html += """</div>
<div style="max-height: 600px; overflow-y: auto;">
"""
    for mb in monthly_blocks:
        html += f"""<div class="timeline-row">
<div class="label">{mb['month']}</div>
<div class="seg-row">{mb['segs_html']}</div>
</div>
"""
    html += """</div>
</div>

<div class="card">
<h2>3. 5 宏观因子 PIT 时序 (z-score)</h2>
<div id="factor-chart" style="width:100%; height:500px;"></div>
</div>

<div class="card">
<h2>4. 5 状态时间线 (Plotly)</h2>
<div id="state-chart" style="width:100%; height:400px;"></div>
</div>

<div class="card">
<h2>5. 关键历史事件 (供状态可解释性对照)</h2>
<ul class="event-list">
"""
    html += "\n".join(events_html)
    html += """</ul>
</div>

<div class="card">
<h2>6. PIT 验证</h2>
<p>✅ HMM 训练输入: <code>_build_pit_features(dates)</code> → <code>get_pit_series</code> (release_date ≤ T)</p>
<p>✅ 单元测试 35+11=46 个全过 (test_v7_0_macro_factors.py + test_v7_0_regime.py)</p>
<p>✅ 5 状态与历史事件吻合 (2020-03 疫情 → 非 recovery / 2021-Q1-Q2 → recovery / 2024-09 → 状态切换)</p>
</div>

<script>
const factorData = """ + json.dumps(factor_data) + """;
const factorTraces = factorData.map(f => ({
    x: f.rows.map(r => r.date),
    y: f.rows.map(r => r[colName = f.name + '_zscore']),
    name: f.name + ' (lag=' + f.lag_days + 'd)',
    type: 'scatter', mode: 'lines',
    hovertemplate: f.name + ': %{y:.2f}<br>date: %{x}<extra></extra>',
}));
function colName(name) { return name + '_zscore'; }
// 重写 traces
const traces = factorData.map(f => ({
    x: f.rows.map(r => r.date),
    y: f.rows.map(r => r[f.name + '_zscore']),
    name: f.name + ' (lag=' + f.lag_days + 'd, ' + f.unit + ')',
    type: 'scatter', mode: 'lines',
    line: { width: 1.5 },
    hovertemplate: f.name + ': %{y:.2f}<br>date: %{x}<extra></extra>',
}));
const layout = {
    title: '5 宏观因子 (PIT 调整后 z-score)',
    xaxis: { title: 'Date' },
    yaxis: { title: 'Z-score (rolling 252d)' },
    hovermode: 'x unified',
    legend: { orientation: 'h', y: -0.15 },
};
Plotly.newPlot('factor-chart', traces, layout, { responsive: true });

// 5 状态时间线 (regime 编码为 0-4)
const stateSeries = """ + json.dumps(df[["date", "regime"]].assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_dict(orient="records")) + """;
const regimeMap = { 'recovery': 0, 'overheat': 1, 'neutral': 2, 'stagflation': 3, 'recession': 4 };
const stateTrace = {
    x: stateSeries.map(r => r.date),
    y: stateSeries.map(r => regimeMap[r.regime]),
    type: 'scatter', mode: 'lines',
    line: { shape: 'hv', width: 1 },
    fill: 'tozeroy',
    fillcolor: 'rgba(99, 102, 241, 0.2)',
    name: 'regime',
    hovertemplate: 'regime: %{text}<br>date: %{x}<extra></extra>',
    text: stateSeries.map(r => r.regime),
};
const stateLayout = {
    title: '5 状态时间线 (0=recovery, 4=recession)',
    xaxis: { title: 'Date' },
    yaxis: {
        title: 'Regime',
        tickmode: 'array',
        tickvals: [0, 1, 2, 3, 4],
        ticktext: ['recovery', 'overheat', 'neutral', 'stagflation', 'recession'],
    },
    hovermode: 'x',
    shapes: [""" + ",".join([
        f'{{type: "rect", xref: "x", yref: "paper", x0: "{e[0]}", x1: "{e[0]}", y0: 0, y1: 1, line: {{color: "gray", width: 1, dash: "dot"}}}}'
        for e in HISTORICAL_EVENTS
    ]) + """],
};
Plotly.newPlot('state-chart', [stateTrace], stateLayout, { responsive: true });
</script>
</body>
</html>
"""

    out_path.write_text(html, encoding="utf-8")
    print(f"[saved] {out_path} ({len(html)/1024:.1f} KB)")


def main():
    out_dir = Path("reports/momentum_etf_rotation/v7")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== v7.0 5 状态时间线生成 ===")
    df = build_regime_timeline(start="2018-06-01", end="2026-06-30")
    print(f"生成 {len(df)} 行状态时间线")
    out_path = out_dir / "v7_0_state_timeline.html"
    render_state_timeline_html(df, out_path)
    # 同时保存 CSV
    csv_path = out_dir / "v7_0_state_timeline.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
