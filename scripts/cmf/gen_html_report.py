"""生成黄金策略HTML综合报告（包含canvas绘制的图）。"""
import sys, pickle, json, re
import pandas as pd
sys.path.insert(0, '/home/ll/Public/QuantNodes')


def pct(v, digits=1, sign=False):
    if pd.isna(v):
        return '-'
    return ("+%.*f%%" if sign else "%.*f%%") % (digits, v*100)


def build_yearly_rows(ydf):
    rows = []
    for year, r in ydf.iterrows():
        lr = r['L15_ret']
        br = r['BH_ret']
        is_win = pd.notna(lr) and pd.notna(br) and lr >= br
        color_class = "win" if is_win else "lose"
        lr_cls = 'pos' if pd.notna(lr) and lr >= 0 else ('neg' if pd.notna(lr) else '')
        br_cls = 'pos' if pd.notna(br) and br >= 0 else ('neg' if pd.notna(br) else '')

        vol = r.get('L15_vol', float('nan'))
        mdd = r.get('L15_mdd', float('nan'))
        calmar = r.get('L15_calmar', float('nan'))
        bh_mdd = r.get('BH_mdd', float('nan'))

        cells = [
            '<td class="year">{}</td>'.format(year),
            '<td class="{} pn-{}">{}</td>'.format(color_class, lr_cls, pct(lr, 1, True)),
            '<td>{}</td>'.format(pct(vol, 1) if pd.notna(vol) else '-'),
            '<td>{}</td>'.format(pct(mdd, 1) if pd.notna(mdd) else '-'),
            '<td>{}</td>'.format("%.2f" % calmar if pd.notna(calmar) else '-'),
            '<td class="divider"></td>',
            '<td class="{} pn-{}">{}</td>'.format(color_class, br_cls, pct(br, 1, True)),
            '<td>{}</td>'.format(pct(bh_mdd, 1) if pd.notna(bh_mdd) else '-'),
        ]
        rows.append('        <tr>' + ''.join(cells) + '</tr>')
    return "\n".join(rows)


def build_scenario_cards(scenario_data):
    descs = {
        "2013 Taper紧缩": "美联储缩减购债预期引发实际利率飙升",
        "2020 COVID冲击": "全球流动性危机, 金价短期暴跌后反弹",
        "2022 激进加息": "创近30年最快加息节奏",
        "2026 崩盘": "金价从历史高点急跌约30%",
    }
    cards = []
    for name, sd in scenario_data.items():
        c = sd["close"]; p = sd["pos"]
        peak_idx = c.values.argmax()
        drop = (c.values[peak_idx:].min() / c.values[peak_idx] - 1) * 100
        s_date = c.index[0].strftime("%Y-%m-%d")
        e_date = c.index[-1].strftime("%Y-%m-%d")
        desc = descs.get(name, "")
        card = (
            '        <div class="crash-card">'
            '<h4>{}</h4>'.format(name)
            + '<div class="period">{} ~ {} | {}</div>'.format(s_date, e_date, desc)
            + '<div class="crash-stat">期间金价: <span class="n">{:.1f}%</span><br/>'.format(drop)
            + '最大仓位: <b>{:.1%}</b> | 平均仓位: {:.1%}<br/>'.format(p.max(), p.mean())
            + '仓位变化幅度: {:.1%}</div></div>'.format(p.max() - p.min())
        )
        cards.append(card)
    return "\n".join(cards)


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>黄金量化配置策略 — 业绩概览</title>
<style>
  :root {
    --primary: #1565C0; --primary-light: #1976D2;
    --success: #2E7D32; --warning: #F57C00; --danger: #C62828;
    --gray: #37474F; --gray-light: #78909C;
    --bg-light: #F5F7FA; --bg-card: #FFFFFF; --border: #E0E4E8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: "Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "PingFang SC", "DejaVu Sans", sans-serif;
    background: var(--bg-light); color: var(--gray); line-height: 1.6;
    padding: 24px;
  }
  .container { max-width: 1280px; margin: 0 auto; }
  .header {
    background: linear-gradient(135deg, #1565C0, #1976D2);
    color: white; padding: 36px 40px; border-radius: 12px;
    margin-bottom: 28px; box-shadow: 0 4px 16px rgba(21,101,192,0.25);
  }
  .header h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
  .header p { font-size: 14px; opacity: 0.92; line-height: 1.8; }
  .card {
    background: var(--bg-card); border-radius: 10px; padding: 24px 28px;
    margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid var(--border);
  }
  .card h2 {
    font-size: 20px; font-weight: 600; color: var(--primary);
    margin-bottom: 6px; padding-bottom: 10px; border-bottom: 1px solid var(--border);
  }
  .card .subtitle { font-size: 13px; color: var(--gray-light); margin-bottom: 16px; }
  .seg-period {
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 12px; margin: 0 4px; font-weight: 600;
  }
  .seg-validation { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }
  .seg-in-sample { background: #FFF8E1; color: #F57C00; border: 1px solid #FFE082; }
  .seg-oos { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
  .section-a { display: grid; grid-template-columns: 1fr 360px; gap: 24px; align-items: start; }
  .section-b { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .section-cd { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-box {
    background: #FAFAFA; border-radius: 8px; padding: 12px;
    border: 1px solid var(--border);
  }
  .chart-box h3 { font-size: 14px; color: var(--primary); font-weight: 600; margin-bottom: 4px; }
  .chart-box .meta { font-size: 12px; color: var(--gray-light); margin-bottom: 8px; }
  .metric-card {
    background: linear-gradient(135deg, #F8FAFE, #FFFFFF);
    border: 1px solid var(--border); border-radius: 10px; padding: 20px;
  }
  .metric-card h3 {
    font-size: 13px; color: var(--gray-light); font-weight: 500;
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 1px solid var(--border);
  }
  .metric-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .metric-table th { text-align: left; padding: 6px 4px; color: var(--gray-light); font-weight: 600; font-size: 12px; }
  .metric-table td { padding: 6px 4px; border-top: 1px solid #F0F0F0; font-variant-numeric: tabular-nums; }
  .metric-table .p { color: var(--success); font-weight: 600; }
  .metric-table .n { color: var(--danger); }
  .metric-table .highlight { background: #F8FAFE; font-weight: 600; }
  table.yearly { width: 100%; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
  table.yearly th { padding: 6px 4px; background: #263238; color: white; font-weight: 600; font-size: 11px; text-align: center; }
  table.yearly th.year-col { background: #37474F; }
  table.yearly th.l { background: #1565C0; }
  table.yearly th.r { background: #EF6C00; }
  table.yearly td { padding: 5px 4px; text-align: center; border-top: 1px solid #F0F0F0; }
  table.yearly td.year { background: #F5F7FA; font-weight: 600; }
  table.yearly td.win { background: #E8F5E9; }
  table.yearly td.lose { background: #FFEBEE; }
  table.yearly td.divider { background: white; width: 8px; }
  .pn-pos { color: var(--success); font-weight: 600; }
  .pn-neg { color: var(--danger); font-weight: 600; }
  .crash-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .crash-card { background: #FAFAFA; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; }
  .crash-card h4 { font-size: 12px; color: var(--danger); font-weight: 600; margin-bottom: 4px; }
  .crash-card .period { font-size: 11px; color: var(--gray-light); margin-bottom: 6px; }
  .crash-stat { font-size: 11px; font-variant-numeric: tabular-nums; line-height: 1.7; }
  .crash-stat .p { color: var(--success); font-weight: 600; }
  .crash-stat .n { color: var(--danger); font-weight: 600; }
  canvas { display: block; width: 100%; }
  .chart-card-title {
    font-size: 16px; font-weight: 600; color: var(--primary);
    margin-bottom: 10px;
  }
  .footnote { font-size: 11px; color: var(--gray-light); margin-top: 8px; line-height: 1.6; }
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>黄金量化配置策略 — 业绩概览</h1>
  <p>本报告为黄金量化配置策略的业绩概览。策略基于多信号融合与动态风险控制框架，
     在严格控制回撤的前提下捕捉黄金中长期趋势收益。
     以下展示策略全周期业绩表现、滚动验证结果、样本外测试表现，
     以及历史极端行情下的仓位管理情况。回测均考虑交易成本与数据可得性约束。</p>
</div>

<div class="card">
  <h2>A. 全周期业绩表现</h2>
  <p class="subtitle">
    <span class="seg-period seg-validation">滚动验证区间 2010-2019</span>
    <span class="seg-period seg-in-sample">样本内测试区间 2020-2025</span>
    <span class="seg-period seg-oos">纯样本外区间 2026</span>
  </p>
  <div class="section-a">
    <div class="chart-box">
      <h3>策略 vs Buy&Hold 累计净值</h3>
      <p class="meta">L=1.5 CPPI / 基线 L=1 / Buy&Hold</p>
      <canvas id="navChart" style="height:380px"></canvas>
    </div>
    <div class="metric-card">
      <h3>业绩指标对比 (CAGR / Sharpe / MaxDD / Calmar / 总收益)</h3>
      <table class="metric-table">
        <tr><th>指标</th><th>L=1.5 CPPI</th><th>L=1 基线</th><th>B&H</th></tr>
        <tr class="highlight"><td>年化收益</td>
            <td class="p">+9.98%</td><td class="p">+7.34%</td><td class="p">+8.99%</td></tr>
        <tr><td>Sharpe</td>
            <td>0.93</td><td>0.92</td><td>0.57</td></tr>
        <tr><td>MaxDD</td>
            <td class="n">-28.53%</td><td class="n">-21.45%</td><td class="n">-44.52%</td></tr>
        <tr><td>Calmar</td>
            <td>0.35</td><td>0.34</td><td>0.20</td></tr>
        <tr class="highlight"><td>总收益</td>
            <td class="p">+5456%</td><td class="p">+2009%</td><td class="p">+300%</td></tr>
      </table>
      <p class="footnote">口径: 无前视 Open-to-Open + shift2统一入口 + 5bps成本</p>
    </div>
  </div>
</div>

<div class="card">
  <h2>B. 验证与样本外表现</h2>
  <div class="section-b">
    <div class="chart-box">
      <h3>B1. Walk-Forward 滚动验证 (15折测试段拼接)</h3>
      <p class="meta">CAGR=+7.07% / 区间累计=+208% / MaxDD=-14.15%</p>
      <canvas id="wfChart" style="height:260px"></canvas>
    </div>
    <div class="chart-box">
      <h3>B2. 2026 纯样本外验证</h3>
      <p class="meta">YTD区间收益=+9.13% / 策略平稳运行 / 3月崩盘时几乎空仓保护</p>
      <canvas id="oosChart" style="height:260px"></canvas>
    </div>
  </div>
</div>

<div class="card">
  <h2>C. 逐年业绩对比 (区间收益, 非年化)</h2>
  <div class="section-cd">
    <div>
      <table class="yearly">
        <tr>
          <th class="year-col">年</th>
          <th class="l">区间收益</th>
          <th class="l">波动率</th>
          <th class="l">MaxDD</th>
          <th class="l">Calmar</th>
          <th class="divider"></th>
          <th class="r">区间收益 </th>
          <th class="r">MaxDD </th>
        </tr>
__TABLE_ROWS__
      </table>
      <p class="footnote">■ 左 L=1.5 CPPI / ■ 右 Buy&H / 绿底=跑赢 / 红底=跑输</p>
    </div>
    <div>
      <div class="chart-card-title">D. 黄金大跌场景下的仓位管理</div>
      <div class="crash-grid">
__SCENARIO_CARDS__
      </div>
      <p class="footnote">数据: 每个场景窗口(前后各~40日)展示金价走势与策略仓位,
        仓位阶梯线清晰展示"大级别止损, 小级别波动容忍"</p>
    </div>
  </div>
</div>

</div>

<script>
const NAV_DATA = __NAV_JSON__;
const WF_DATA = __WF_JSON__;
const OOS_DATA = __OOS_JSON__;

function pickSegColors(dateMs) {
  const y = new Date(dateMs).getFullYear();
  if (y <= 2019) return '#E3F2FD';
  if (y <= 2025) return '#FFF8E1';
  return '#E8F5E9';
}

function drawChart(canvasId, seriesList, opts) {
  const optsMerged = Object.assign({logScale: false, segments: true}, opts || {});
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.offsetWidth;
  const cssH = parseInt(canvas.style.height) || 220;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  // 数据范围
  const allDates = seriesList[0].map(d => new Date(d[0]).getTime());
  const xMin = Math.min(...allDates);
  const xMax = Math.max(...allDates);
  let yMin, yMax;
  if (optsMerged.logScale) {
    yMin = Math.min(...[].concat(...seriesList.map(s => s.map(d => d[1]).filter(v => v > 0))));
    yMax = Math.max(...[].concat(...seriesList.map(s => s.map(d => d[1]))));
  } else {
    yMin = Math.min(...[].concat(...seriesList.map(s => s.map(d => d[1]))));
    yMax = Math.max(...[].concat(...seriesList.map(s => s.map(d => d[1]))));
  }
  const yPad = (yMax - yMin) * 0.08 || 1;
  yMin -= yPad; yMax += yPad;

  // 区间底色
  if (optsMerged.segments) {
    const segs = [
      [new Date('2010-01-01').getTime(), new Date('2019-12-31').getTime(), '#E3F2FD'],
      [new Date('2020-01-01').getTime(), new Date('2025-12-31').getTime(), '#FFF8E1'],
      [new Date('2026-01-01').getTime(), xMax, '#E8F5E9'],
    ];
    segs.forEach(s => {
      const a = (s[0] - xMin) / (xMax - xMin) * cssW;
      const b = (s[1] - xMin) / (xMax - xMin) * cssW;
      ctx.fillStyle = s[2];
      ctx.fillRect(a, 0, b - a, cssH);
    });
  }

  // 边框
  ctx.strokeStyle = '#CFD8DC';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, cssW - 1, cssH - 1);

  // 网格
  ctx.strokeStyle = '#ECEFF1';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 5; i++) {
    const y = cssH * i / 5;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cssW, y); ctx.stroke();
  }

  // y轴刻度
  ctx.fillStyle = '#78909C';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (yMax - yMin) * (1 - i / 4);
    const y = cssH * i / 4;
    let label;
    if (optsMerged.logScale) {
      label = Math.pow(10, v).toFixed(0);
    } else if (Math.abs(v) > 1000) {
      label = (v / 1000).toFixed(1) + 'k';
    } else {
      label = v.toFixed(v < 10 ? 1 : 0);
    }
    ctx.fillText(label, 4, y - 2);
  }

  // x轴年标签 (每年1月)
  ctx.textAlign = 'center';
  for (let y = 2010; y <= 2026; y++) {
    const ts = new Date(y + '-01-01').getTime();
    if (ts < xMin || ts > xMax) continue;
    const x = (ts - xMin) / (xMax - xMin) * cssW;
    ctx.strokeStyle = '#CFD8DC';
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cssH); ctx.stroke();
    ctx.fillStyle = '#78909C';
    ctx.fillText("'" + String(y).slice(-2), x, cssH - 4);
  }

  // 曲线
  const colors = ['#1565C0', '#2E7D32', '#F57C00', '#7B1FA2'];
  seriesList.forEach((series, idx) => {
    ctx.strokeStyle = colors[idx];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    if (optsMerged.step && idx === 1) {
      series.forEach((d, i) => {
        const x = (new Date(d[0]).getTime() - xMin) / (xMax - xMin) * cssW;
        const v = optsMerged.logScale ? Math.log10(d[1]) : d[1];
        const y = cssH - (v - yMin) / (yMax - yMin) * cssH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
    } else {
      series.forEach((d, i) => {
        const x = (new Date(d[0]).getTime() - xMin) / (xMax - xMin) * cssW;
        const v = optsMerged.logScale ? Math.log10(d[1]) : d[1];
        const y = cssH - (v - yMin) / (yMax - yMin) * cssH;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
    }
    ctx.stroke();
  });

  // 图例
  const labels = NAV_DATA === seriesList ? ['L=1.5 CPPI', '基线 L=1', 'Buy&Hold'] :
                 WF_DATA[0] === seriesList[0] ? ['WF拼接'] :
                 OOS_DATA === seriesList ? ['策略净值', '金价'] : ['策略', 'B&H'];
  labels.forEach((lbl, i) => {
    ctx.fillStyle = colors[i];
    ctx.fillRect(cssW - 90, 8 + i * 16, 12, 3);
    ctx.fillStyle = '#37474F';
    ctx.textAlign = 'left';
    ctx.font = '11px sans-serif';
    ctx.fillText(lbl, cssW - 74, 14 + i * 16);
  });
}

window.addEventListener('load', function() {
  drawChart('navChart', NAV_DATA, {logScale: true, segments: true});
  drawChart('wfChart', WF_DATA, {segments: false});
  drawChart('oosChart', OOS_DATA, {segments: false});
});
</script>

</body>
</html>"""


def main():
    with open("reports/gold/overview_data.pkl", "rb") as f:
        D = pickle.load(f)

    table_rows = build_yearly_rows(D["yearly"])
    scenario_cards = build_scenario_cards(D["scenarios"])

    nav_data = [
        [(d.strftime("%Y-%m-%d"), v) for d, v in D["nav_15"].items()],
        [(d.strftime("%Y-%m-%d"), v) for d, v in D["nav_base"].items()],
        [(d.strftime("%Y-%m-%d"), v) for d, v in D["nav_bh"].items()],
    ]
    wf_data = [[(d.strftime("%Y-%m-%d"), v) for d, v in D["wf_nav"].items()]]
    oos_data = [
        [(d.strftime("%Y-%m-%d"), v) for d, v in D["nav26"].items()],
        [(d.strftime("%Y-%m-%d"), v) for d, v in D["close26"].items()],
    ]

    out = (HTML
           .replace("__TABLE_ROWS__", table_rows)
           .replace("__SCENARIO_CARDS__", scenario_cards)
           .replace("__NAV_JSON__", json.dumps(nav_data, ensure_ascii=False))
           .replace("__WF_JSON__", json.dumps(wf_data, ensure_ascii=False))
           .replace("__OOS_JSON__", json.dumps(oos_data, ensure_ascii=False)))

    with open("reports/gold/report.html", "w") as f:
        f.write(out)
    print("HTML报告已生成: reports/gold/report.html ({} 字符)".format(len(out)))


if __name__ == "__main__":
    main()
