# coding=utf-8
"""诊断报告生成 (Markdown + HTML 仪表盘).

输出:
    - Markdown 报告 (用于 docs/50 模板填充)
    - HTML 仪表盘 (含 6 个面板的交互式可视化)
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from .cycle_position import CycleState


def generate_markdown_report(
    state: CycleState,
    historical_phases: pd.Series | None = None,
    hs300_history: pd.Series | None = None,
) -> str:
    """生成 Markdown 报告 (填充 docs/50 模板).

    参数:
        state: 当前 CycleState
        historical_phases: 历史美林时钟阶段 (用于回测验证)
        hs300_history: 沪深300 历史收益 (用于回测验证)

    返回:
        Markdown 字符串
    """
    report = []
    report.append("# v9 当前周期状态诊断报告")
    report.append(f"> **报告日期**: {state.report_date.strftime('%Y-%m-%d') if state.report_date else 'N/A'}")
    report.append(f"> **数据截至**: {state.data_through.strftime('%Y-%m-%d') if state.data_through is not None else 'N/A'}")
    report.append(f"> **关联**: docs/49-v9_cycle_timing.md, docs/49a-v9_cycle_diagnosis.md")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## ⚠️ 重要说明")
    report.append("")
    report.append("**本报告是基于 v9 周期诊断框架生成的当前市场状态评估**。")
    report.append("")
    report.append("### 局限")
    report.append("")
    report.append("1. **数据长度**: A 股可用数据 1990-至今 ~34 年, 不够 1 个完整 Kondratieff (50-60 年)")
    report.append("2. **西方框架**: 美林时钟/Pring 周期是西方成熟市场理论, A 股适用性需自行验证")
    report.append("3. **未来数据**: 所有判定仅基于历史数据, 不构成投资建议")
    report.append("")
    report.append("### 评分解读")
    report.append("")
    report.append("| 评分区间 | 含义 | 建议 |")
    report.append("|---------|------|------|")
    report.append("| 0-30 | 弱势/震荡 | 谨慎, 降低仓位 |")
    report.append("| 30-50 | 中性 | 标准配置 |")
    report.append("| 50-70 | 偏多 | 适度加仓 |")
    report.append("| 70-100 | 强势 | 重仓 |")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## 一、美林时钟定位")
    report.append("")
    report.append("### 1.1 当前阶段")
    report.append("")
    report.append("| 指标 | 当前值 (z-score) | 阈值 | 方向 |")
    report.append("|------|------------------|------|------|")
    report.append(f"| 经济增长 ({state.gdp_proxy_name}) | {state.growth_zscore:.3f} | 滚动 36 周中位数 | {'↑' if state.growth_zscore > 0 else '↓'} |")
    report.append(f"| 通胀 ({state.cpi_proxy_name}) | {state.inflation_zscore:.3f} | 滚动 36 周中位数 | {'↑' if state.inflation_zscore > 0 else '↓'} |")
    report.append("")
    report.append(f"**当前判定**: {state.merrill_phase} ({state.merrill_phase_cn})")
    report.append("")
    report.append(f"**置信度**: {state.merrill_confidence:.2%}")
    report.append("")
    report.append(f"**判定时间**: {state.data_through.strftime('%Y-%m') if state.data_through is not None else 'N/A'}")
    report.append("")
    report.append("### 1.3 推荐配置")
    report.append("")
    report.append("基于当前美林阶段:")
    report.append("")
    report.append("| 资产 | 权重 |")
    report.append("|------|------|")
    for asset, weight in state.recommended_allocation.items():
        report.append(f"| {asset} | {weight:.0%} |")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## 二、Pring 周期定位")
    report.append("")
    report.append("### 2.1 10 年周期位置")
    report.append("")
    report.append(f"- **当前年份**: {state.pring_year}")
    report.append(f"- **10 年周期位置**: 第 {state.pring_position} 年 (基准 2015 = 第 7 年)")
    report.append(f"- **季节性偏好**: {state.pring_seasonality_cn} ({state.pring_seasonality})")
    report.append("")
    report.append("### 2.3 多周期叠加")
    report.append("")
    report.append("| 周期 | 当前相位 | 趋势 |")
    report.append("|------|----------|------|")
    for name, phase in state.multi_cycle_phases.items():
        direction = "↑ 上行" if phase == "up" else ("↓ 下行" if phase == "down" else "→ 转换")
        report.append(f"| {name} | {phase} | {direction} |")
    report.append("")
    report.append(f"**综合判定**: {state.composite_phase}")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## 四、当前评分")
    report.append("")
    report.append("### 4.1 评分分解")
    report.append("")
    report.append("| 维度 | 评分 | 满分 | 占比 |")
    report.append("|------|------|------|------|")
    report.append(f"| 周期趋势 (cycle) | {state.cycle_score:.1f} | 40 | {state.cycle_score/40:.1%} |")
    report.append(f"| 周期耦合 (coupling) | {state.coupling_score:.1f} | 40 | {state.coupling_score/40:.1%} |")
    report.append(f"| VIX 分 (vix) | {state.vix_score:.1f} | 20 | {state.vix_score/20:.1%} |")
    report.append(f"| **总分** | **{state.total_score:.1f}** | **100** | **{state.total_score/100:.1%}** |")
    report.append("")
    report.append("### 4.3 信号判定")
    report.append("")
    report.append(f"- **大盘信号**: {state.v9_signal} (0=空仓, 1=满仓)")
    report.append(f"- **迟滞状态**: {state.signal_label}")
    if state.vix_value is not None:
        report.append(f"- **VIX 当前**: {state.vix_value:.2f}")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## 五、择时建议")
    report.append("")
    report.append("### 5.1 大盘择时")
    report.append("")
    report.append(f"| 信号 | 当前 |")
    report.append("|------|------|")
    report.append(f"| 大盘信号 | {state.v9_signal} |")
    report.append(f"| 总分 | {state.total_score:.1f} |")
    report.append("")
    report.append(f"**建议**: {state.signal_label}")
    report.append("")
    report.append("### 5.2 资产配置建议")
    report.append("")
    report.append("基于美林时钟 + 多周期综合:")
    report.append("")
    report.append("| 资产 | 建议权重 |")
    report.append("|------|----------|")
    for asset, weight in state.recommended_allocation.items():
        report.append(f"| {asset} | {weight:.0%} |")
    report.append("")
    report.append("---")
    report.append("")

    report.append("## 七、附录")
    report.append("")
    report.append("### 7.1 数据说明")
    report.append("")
    report.append("| 数据 | 来源 | 频率 |")
    report.append("|------|------|------|")
    report.append(f"| {state.gdp_proxy_name} | v7_14_X_panel 第 1 列 | 周频 |")
    report.append(f"| {state.cpi_proxy_name} | v7_14_X_panel 第 2 列 | 周频 |")
    report.append("| VIX (美股) | macro_vix_daily.parquet | 日频 |")
    report.append("")
    report.append("### 7.2 方法说明")
    report.append("")
    report.append("- **HP 滤波**: λ=100 (周频标准)")
    report.append("- **美林时钟**: 6 周平滑 + 36 周中位数阈值")
    report.append("- **评分权重**: 周期趋势 (40%) + 周期耦合 (40%) + VIX (20%)")
    report.append("- **迟滞阈值**: 50 (多) / 30 (空)")
    report.append("")
    report.append("### 7.3 免责声明")
    report.append("")
    report.append("**本报告仅供参考, 不构成投资建议**。")
    report.append("")
    report.append("- 所有判定基于历史数据, 未来可能完全不同")
    report.append("- 美林/Pring 框架是西方理论, A 股适用性需自行验证")
    report.append("- 评分模型未在样本外充分验证")
    report.append("- 投资有风险, 决策需谨慎")
    report.append("")
    report.append("---")
    report.append("")
    report.append("## 八、版本与变更")
    report.append("")
    report.append("| 日期 | 版本 | 变更 |")
    report.append("|------|------|------|")
    report.append(f"| {state.report_date.strftime('%Y-%m-%d') if state.report_date else 'N/A'} | v1.0 | 初始报告 |")
    report.append("")

    return "\n".join(report)


def generate_html_dashboard(
    state: CycleState,
    output_path: str | Path,
    imfs_history: np.ndarray | None = None,
    phase_history: pd.Series | None = None,
    hs300_history: pd.Series | None = None,
    imf_dates: pd.DatetimeIndex | None = None,
) -> Path:
    """生成 HTML 仪表盘.

    参数:
        state: 当前 CycleState
        output_path: HTML 输出路径
        imfs_history: (K, T) IMF 时序 (可选, 用于面板 4)
        phase_history: 历史美林时钟阶段 (可选, 用于面板 1)
        hs300_history: 沪深300 历史 (可选, 用于多个面板)
        imf_dates: IMF 时间索引 (可选)

    返回:
        输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signal_color = (
        "#22c55e" if state.v9_signal == 1
        else "#ef4444" if state.v9_signal == 0
        else "#f59e0b"
    )

    phase_color = {
        0: "#10b981",
        1: "#f59e0b",
        2: "#ef4444",
        3: "#3b82f6",
    }.get(state.merrill_phase_num, "#6b7280")

    score_pct = state.total_score
    if score_pct >= 70:
        score_color = "#22c55e"
    elif score_pct >= 50:
        score_color = "#84cc16"
    elif score_pct >= 30:
        score_color = "#f59e0b"
    else:
        score_color = "#ef4444"

    imf_chart_data = ""
    if imfs_history is not None and imfs_history.size > 0:
        K, T = imfs_history.shape
        if imf_dates is None:
            imf_dates = pd.date_range(end=state.data_through, periods=T, freq="W")

        color_list = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
        imf_datasets = []
        for k in range(K):
            color = color_list[k % 4]
            data_str = ", ".join(f"{v:.4f}" for v in imfs_history[k])
            imf_datasets.append(
                f'{{label: "IMF{k+1}", data: [{data_str}], borderColor: "{color}", fill: false, pointRadius: 0, borderWidth: 1.5}}'
            )
        labels_list = [d.strftime("%Y-%m") for d in imf_dates]
        imf_labels = "[" + ", ".join('"' + l + '"' for l in labels_list) + "]"
        imf_chart_data = f"""
        new Chart(document.getElementById('imfChart'), {{
            type: 'line',
            data: {{
                labels: {imf_labels},
                datasets: [{", ".join(imf_datasets)}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'top' }} }},
                scales: {{ x: {{ ticks: {{ maxTicksLimit: 10 }} }} }}
            }}
        }});"""

    phase_chart_data = ""
    if phase_history is not None and len(phase_history) > 0:
        phase_labels_list = [d.strftime("%Y-%m") for d in phase_history.index]
        phase_labels = "[" + ", ".join('"' + l + '"' for l in phase_labels_list) + "]"
        phase_values = phase_history.tolist()
        phase_colors = [
            {0: "#10b981", 1: "#f59e0b", 2: "#ef4444", 3: "#3b82f6"}.get(v, "#6b7280")
            for v in phase_values
        ]
        phase_chart_data = f"""
        new Chart(document.getElementById('phaseChart'), {{
            type: 'bar',
            data: {{
                labels: {phase_labels},
                datasets: [{{
                    label: '美林阶段',
                    data: {phase_values},
                    backgroundColor: {json.dumps(phase_colors)},
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ y: {{ ticks: {{ callback: v => {{ const m = {{0:'Recovery',1:'Overheat',2:'Stagflation',3:'Recession'}}; return m[v] || ''; }} }} }} }}
            }}
        }});"""

    nav_chart_data = ""
    if hs300_history is not None and len(hs300_history) > 0:
        nav_labels_list = [d.strftime("%Y-%m") for d in hs300_history.index]
        nav_labels = "[" + ", ".join('"' + l + '"' for l in nav_labels_list) + "]"
        nav_values = ((1 + hs300_history).cumprod() * 100).tolist()
        nav_chart_data = f"""
        new Chart(document.getElementById('navChart'), {{
            type: 'line',
            data: {{
                labels: {nav_labels},
                datasets: [{{
                    label: '沪深300等权 (NAV)',
                    data: {nav_values},
                    borderColor: '#3b82f6',
                    fill: false,
                    pointRadius: 0,
                    borderWidth: 1.5
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'top' }} }},
                scales: {{ x: {{ ticks: {{ maxTicksLimit: 10 }} }} }}
            }}
        }});"""

    allocation_data = {
        "labels": list(state.recommended_allocation.keys()),
        "values": list(state.recommended_allocation.values()),
    }

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>v9 周期择时诊断仪表盘 — {state.data_through.strftime('%Y-%m-%d') if state.data_through is not None else ''}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    padding: 24px;
    line-height: 1.6;
}}
.container {{ max-width: 1400px; margin: 0 auto; }}
header {{
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 24px;
    border: 1px solid #334155;
}}
h1 {{ font-size: 28px; margin-bottom: 8px; }}
.subtitle {{ color: #94a3b8; font-size: 14px; }}
.row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 16px; }}
.card {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
}}
.card-title {{ font-size: 12px; text-transform: uppercase; color: #94a3b8; letter-spacing: 1px; margin-bottom: 8px; }}
.card-value {{ font-size: 32px; font-weight: 700; }}
.card-sub {{ font-size: 14px; color: #94a3b8; margin-top: 4px; }}
.phase-tag {{
    display: inline-block;
    padding: 6px 16px;
    border-radius: 6px;
    color: white;
    font-weight: 600;
    margin-right: 8px;
}}
.score-bar {{
    background: #334155;
    height: 24px;
    border-radius: 12px;
    overflow: hidden;
    margin: 12px 0;
}}
.score-fill {{
    height: 100%;
    transition: width 0.6s ease-out;
    display: flex;
    align-items: center;
    padding-left: 12px;
    color: white;
    font-weight: 600;
    font-size: 13px;
}}
table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #334155; font-size: 14px; }}
th {{ color: #94a3b8; font-weight: 500; font-size: 12px; text-transform: uppercase; }}
.chart-container {{ height: 280px; position: relative; margin-top: 16px; }}
.allocation-bar {{ display: flex; height: 32px; border-radius: 8px; overflow: hidden; margin-top: 8px; }}
.allocation-bar > div {{ display: flex; align-items: center; justify-content: center; color: white; font-size: 12px; font-weight: 600; }}
.bull {{ color: #22c55e; }}
.bear {{ color: #ef4444; }}
.neutral {{ color: #f59e0b; }}
.up {{ color: #22c55e; }}
.down {{ color: #ef4444; }}
.transition {{ color: #f59e0b; }}
footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 32px; padding: 16px; }}
</style>
</head>
<body>
<div class="container">
    <header>
        <h1>📊 v9 周期择时诊断仪表盘</h1>
        <div class="subtitle">报告日期: {state.report_date.strftime('%Y-%m-%d %H:%M') if state.report_date else 'N/A'} | 数据截至: {state.data_through.strftime('%Y-%m-%d') if state.data_through is not None else 'N/A'}</div>
    </header>

    <div class="row">
        <div class="card">
            <div class="card-title">🎯 美林时钟</div>
            <div class="card-value">
                <span class="phase-tag" style="background: {phase_color}">{state.merrill_phase}</span>
            </div>
            <div class="card-sub">{state.merrill_phase_cn} | 置信度 {state.merrill_confidence:.0%}</div>
            <div style="margin-top: 12px; font-size: 13px;">
                经济增长 z: <span class="{'up' if state.growth_zscore > 0 else 'down'}">{state.growth_zscore:+.2f}</span><br>
                通胀 z: <span class="{'up' if state.inflation_zscore > 0 else 'down'}">{state.inflation_zscore:+.2f}</span>
            </div>
        </div>

        <div class="card">
            <div class="card-title">📅 Pring 10 年周期</div>
            <div class="card-value">{state.pring_position}/10</div>
            <div class="card-sub">{state.pring_seasonality_cn} ({state.pring_year} 年)</div>
        </div>

        <div class="card">
            <div class="card-title">🌊 多周期叠加</div>
            <div class="card-value">{state.composite_phase.replace('_', ' ')}</div>
            <div class="card-sub">基于 4 IMF 综合判定</div>
            <div style="margin-top: 12px; font-size: 13px;">
                {''.join('<span class="cycle-pill"><span class="' + ('up' if v == 'up' else 'down' if v == 'down' else 'transition') + '">' + k + '=' + v + '</span></span> ' for k, v in state.multi_cycle_phases.items())}
            </div>
        </div>

        <div class="card">
            <div class="card-title">⚡ V9 大盘信号</div>
            <div class="card-value" style="color: {signal_color};">
                {state.signal_label.split(' ')[0]}
            </div>
            <div class="card-sub">{state.signal_label}</div>
            <div style="margin-top: 12px;">
                <div style="font-size: 12px; color: #94a3b8;">总分 (0-100)</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: {state.total_score}%; background: {score_color};">
                        {state.total_score:.1f}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="row">
        <div class="card">
            <div class="card-title">📈 评分分解 (40+40+20)</div>
            <table>
                <thead><tr><th>维度</th><th>评分</th><th>占比</th></tr></thead>
                <tbody>
                    <tr><td>周期趋势 (cycle)</td><td>{state.cycle_score:.1f} / 40</td><td>{state.cycle_score/40:.0%}</td></tr>
                    <tr><td>周期耦合 (coupling)</td><td>{state.coupling_score:.1f} / 40</td><td>{state.coupling_score/40:.0%}</td></tr>
                    <tr><td>VIX 分 (vix)</td><td>{state.vix_score:.1f} / 20</td><td>{state.vix_score/20:.0%}</td></tr>
                    <tr style="font-weight: 700; color: {score_color};"><td>总分</td><td>{state.total_score:.1f} / 100</td><td>{state.total_score/100:.0%}</td></tr>
                </tbody>
            </table>
            {f'<div style="margin-top: 12px; font-size: 12px; color: #94a3b8;">VIX 当前: <b>{state.vix_value:.2f}</b> (作为全球风险偏好代理)</div>' if state.vix_value is not None else ''}
        </div>

        <div class="card">
            <div class="card-title">💼 推荐资产配置</div>
            <div class="allocation-bar">
                {''.join(f'<div style="width: {w*100}%; background: {c};">{k} {w:.0%}</div>' for (k, w), c in zip(state.recommended_allocation.items(), ['#3b82f6', '#10b981', '#f59e0b', '#6b7280']))}
            </div>
            <div style="margin-top: 16px; font-size: 13px; color: #94a3b8;">
                基于美林阶段 {state.merrill_phase} 的标准配置
            </div>
        </div>
    </div>

    {f'''
    <div class="row">
        <div class="card" style="grid-column: span 2;">
            <div class="card-title">🌊 历史美林时钟阶段</div>
            <div class="chart-container">
                <canvas id="phaseChart"></canvas>
            </div>
        </div>
    </div>
    ''' if phase_chart_data else ''}

    {f'''
    <div class="row">
        <div class="card" style="grid-column: span 2;">
            <div class="card-title">📊 多周期 IMF 分解</div>
            <div class="chart-container">
                <canvas id="imfChart"></canvas>
            </div>
        </div>
    </div>
    ''' if imf_chart_data else ''}

    {f'''
    <div class="row">
        <div class="card" style="grid-column: span 2;">
            <div class="card-title">📈 沪深300等权历史 NAV</div>
            <div class="chart-container">
                <canvas id="navChart"></canvas>
            </div>
        </div>
    </div>
    ''' if nav_chart_data else ''}

    <div class="row">
        <div class="card" style="grid-column: span 2;">
            <div class="card-title">📋 评分方法与局限</div>
            <div style="font-size: 13px; color: #94a3b8; margin-top: 8px;">
                <p><b>方法</b>: 美林时钟 (6 周平滑 + 36 周中位数阈值) + Pring 10 年周期 (基准 2015=第 7 年) + 多周期叠加 (Kitchin/Juglar/Kuznets/Kondratieff) + 评分 (40+40+20 学术默认权重)</p>
                <p style="margin-top: 8px;"><b>局限</b>: A 股数据 ~34 年不够 1 个完整 Kondratieff; 美林/Pring 框架是西方成熟市场理论, A 股适用性需自行验证; 评分模型未充分样本外验证</p>
                <p style="margin-top: 8px;"><b>声明</b>: 本报告仅供参考, 不构成投资建议</p>
            </div>
        </div>
    </div>

    <footer>
        v9.0 · QuantNodes · 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </footer>
</div>

<script>
{imf_chart_data}
{phase_chart_data}
{nav_chart_data}
</script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    return output_path