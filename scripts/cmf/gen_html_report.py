"""生成黄金策略HTML综合报告 (matplotlib SVG inline嵌入, 单一HTML文件)."""
import sys, os
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')


def pct(v, digits=1, sign=False):
    if pd.isna(v):
        return '-'
    return ("+%.*f%%" if sign else "%.*f%%") % (digits, v*100)


HTML_HEAD = r"""<!DOCTYPE html>
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
  .seg-insample { background: #FFF8E1; color: #F57C00; border: 1px solid #FFE082; }
  .seg-validation { background: #E3F2FD; color: #1565C0; border: 1px solid #90CAF9; }
  .seg-oos { background: #E8F5E9; color: #2E7D32; border: 1px solid #A5D6A7; }
  .section-a { display: grid; grid-template-columns: 1fr 360px; gap: 24px; align-items: start; }
  .section-b { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .section-cd { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-svg {
    width: 100%; height: auto; border-radius: 6px;
    border: 1px solid var(--border); display: block; background: white;
  }
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
  .chart-title {
    font-size: 14px; color: var(--primary); font-weight: 600; margin-bottom: 8px;
  }
  .chart-meta { font-size: 12px; color: var(--gray-light); margin-bottom: 8px; }
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
  <p>本报告为黄金量化配置策略的业绩概览。策略基于多信号融合与动态风险控制框架,
     在严格控制回撤的前提下捕捉黄金中长期趋势收益。
     以下展示策略全周期业绩表现、滚动验证结果、样本外测试表现,
     以及历史极端行情下的仓位管理情况。回测均考虑交易成本与数据可得性约束。</p>
</div>

<div class="card">
  <h2>A. 全周期业绩表现</h2>
  <p class="subtitle">
    <span class="seg-period seg-insample">样本内测试区间 2010-2019</span>
    <span class="seg-period seg-validation">滚动验证区间 2020-2025</span>
    <span class="seg-period seg-oos">纯样本外区间 2026</span>
  </p>
  <div class="section-a">
    <div>
__SVG_FULL_NAV__
    </div>
    <div class="metric-card">
      <h3>业绩指标对比 (CAGR / Sharpe / MaxDD / Calmar / 总收益)</h3>
      <table class="metric-table">
        __METRICS_ROWS__
      </table>
      <p class="footnote">口径: 无前视 Open-to-Open + shift2统一入口 + 5bps成本</p>
      <p class="footnote" style="margin-top:6px"><b>区间语义</b>: 2010-2019 用于样本内参数确定与早期验证;
        2020-2025 滚动验证(每折训练集=该折之前所有样本); 2026 纯样本外盲测。</p>
    </div>
  </div>
</div>

<div class="card">
  <h2>B. 验证与样本外表现</h2>
  <div class="section-b">
    <div>
      <div class="chart-title">B1. Walk-Forward 滚动验证 (15折测试段拼接)</div>
      <p class="chart-meta">CAGR=+7.07% / 区间累计=+208% / MaxDD=-14.15%</p>
__SVG_WF__
    </div>
    <div>
      <div class="chart-title">B2. 2026 纯样本外验证</div>
      <p class="chart-meta">YTD区间收益=+9.13% / 策略平稳运行 / 3月崩盘时几乎空仓保护</p>
__SVG_OOS__
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
__SVG_SCENARIOS__
      <p class="footnote">数据: 每个场景窗口展示金价走势与策略仓位,
        仓位阶梯线清晰展示"大级别止损, 小级别波动容忍"</p>
    </div>
  </div>
</div>

</div>

</body>
</html>"""


def build_yearly_rows(ydf):
    rows = []
    for year, r in ydf.iterrows():
        lr = r['L15_ret']; br = r['BH_ret']
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


def main():
    import pickle
    from gen_html_assets import plot_full_nav, plot_wf, plot_oos, plot_scenarios

    with open("reports/gold/overview_data.pkl", "rb") as f:
        D = pickle.load(f)

    m15 = D["metrics_15"]
    mb = D["metrics_base"]
    mbh = D["metrics_bh"]

    def fmt_pct(v): return f"{v*100:.2f}"
    def fmt_f(v, d=2): return f"{v:.{d}f}"

    metrics_html = f"""        <tr><th>指标</th><th>L=1.5 CPPI</th><th>L=1 基线</th><th>B&H</th></tr>
        <tr class="highlight"><td>年化收益</td>
            <td class="p">+{fmt_pct(m15['cagr'])}%</td><td class="p">+{fmt_pct(mb['cagr'])}%</td><td class="p">+{fmt_pct(mbh['cagr'])}%</td></tr>
        <tr><td>Sharpe</td>
            <td>{fmt_f(m15['sharpe'])}</td><td>{fmt_f(mb['sharpe'])}</td><td>{fmt_f(mbh['sharpe'])}</td></tr>
        <tr><td>MaxDD</td>
            <td class="n">{fmt_pct(m15['mdd'])}%</td><td class="n">{fmt_pct(mb['mdd'])}%</td><td class="n">{fmt_pct(mbh['mdd'])}%</td></tr>
        <tr><td>Calmar</td>
            <td>{fmt_f(m15['calmar'])}</td><td>{fmt_f(mb['calmar'])}</td><td>{fmt_f(mbh['calmar'])}</td></tr>
        <tr><td>Vol</td>
            <td>{fmt_pct(m15['vol'])}%</td><td>{fmt_pct(mb['vol'])}%</td><td>{fmt_pct(mbh['vol'])}%</td></tr>
        <tr class="highlight"><td>总收益</td>
            <td class="p">+{fmt_pct(m15['total'])}%</td><td class="p">+{fmt_pct(mb['total'])}%</td><td class="p">+{fmt_pct(mbh['total'])}%</td></tr>"""

    table_rows = build_yearly_rows(D["yearly"])

    svg_full = plot_full_nav()
    svg_wf = plot_wf()
    svg_oos = plot_oos()
    svg_scenarios = plot_scenarios()

    # 替换占位符
    out = HTML_HEAD.replace("__METRICS_ROWS__", metrics_html)
    out = out.replace("__TABLE_ROWS__", table_rows)
    out = out.replace("__SVG_FULL_NAV__", '<div class="chart-svg">' + svg_full + '</div>')
    out = out.replace("__SVG_WF__", '<div class="chart-svg">' + svg_wf + '</div>')
    out = out.replace("__SVG_OOS__", '<div class="chart-svg">' + svg_oos + '</div>')
    out = out.replace("__SVG_SCENARIOS__", '<div class="chart-svg">' + svg_scenarios + '</div>')

    with open("reports/gold/report.html", "w") as f:
        f.write(out)
    print("HTML报告已生成: reports/gold/report.html ({:.0f} KB)".format(len(out) / 1024))


if __name__ == "__main__":
    main()
