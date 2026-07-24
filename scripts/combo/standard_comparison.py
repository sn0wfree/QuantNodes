"""21 策略 × 9 区间 × 9 指标 标准化对比.

9 指标:
  Sharpe / Calmar / MaxDD / AnnRet / Vol / WinRate (原 6)
  Sortino / DownsideVol / MaxDDDays / PayoffRatio (新 4)

输出:
  standard_comparison_wide.csv        (21 × 82 列宽格式)
  standard_comparison_long.csv        (long format: 21×9=189 行)
  standard_comparison_summary.csv     (策略总结: Top5 次数, 最佳场景等)
  standard_comparison_underwater.csv  (各策略 under-water days 历史)
  standard_comparison_4strats_monthly.csv (4 策略 × OOS 月度)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"
FIGS_DIR = OUT_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# 9 个测试区间
PERIODS = [
    ('Full Sample',     '2018-01-03', '2026-05-29'),
    ('OOS 22-26',       '2022-01-01', '2026-05-29'),
    ('2018 贸易战大跌', '2018-01-22', '2018-12-28'),
    ('2019 春燥行情',   '2019-01-02', '2019-04-26'),
    ('2020 疫情牛',     '2020-03-23', '2021-02-10'),
    ('2021 白马崩盘',   '2021-09-13', '2022-04-29'),
    ('2022 慢熊',       '2022-01-01', '2022-12-31'),
    ('2023 慢熊延续',   '2023-01-01', '2023-12-31'),
    ('2024 政策牛',     '2024-09-24', '2024-10-08'),
]

PERIOD_FREQ = 252  # 日频年化


def metrics_extended(nav: pd.Series, start: str, end: str, freq=PERIOD_FREQ) -> dict | None:
    """标准化 9 指标."""
    seg = nav.loc[start:end].dropna()
    if len(seg) < 30:
        return None

    rets = seg.pct_change().dropna()
    if len(rets) < 5:
        return None

    total = seg.iloc[-1] / seg.iloc[0] - 1
    n_years = len(rets) / freq
    if n_years < 1e-6:
        n_years = 1.0 / freq

    ann_ret = (1 + total) ** (1 / max(n_years, 1e-9)) - 1
    vol = float(rets.std() * np.sqrt(freq))
    sharpe = float(ann_ret / vol) if vol > 1e-9 else 0.0

    peak = seg.cummax()
    dd = seg / peak - 1
    max_dd = float(dd.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < -1e-6 else 0.0

    win_rate = float((rets > 0).mean())

    # === 新 4 指标 ===
    # Downside vol: 仅负收益的 std
    neg_rets = rets[rets < 0]
    downside_vol = float(neg_rets.std() * np.sqrt(freq)) if len(neg_rets) > 1 else 0.0
    sortino = float(ann_ret / downside_vol) if downside_vol > 1e-9 else 0.0

    # MaxDDDays: 历史最长 under-water days (回撤从开始到恢复新高经历的天数)
    underwater = dd < 0
    if underwater.any():
        # 计算连续 under-water 段
        groups = (underwater != underwater.shift()).cumsum()
        ud_counts = underwater.groupby(groups).sum()
        max_dd_days = int(ud_counts[underwater.groupby(groups).first()].max())
    else:
        max_dd_days = 0

    # Payoff Ratio: 平均盈利 / |平均亏损|
    pos_rets = rets[rets > 0]
    mean_pos = float(pos_rets.mean()) if len(pos_rets) > 0 else 0.0
    mean_neg = float(neg_rets.mean()) if len(neg_rets) > 0 else 0.0
    payoff = float(mean_pos / abs(mean_neg)) if abs(mean_neg) > 1e-9 else 0.0

    return {
        'AnnRet': float(ann_ret),
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'MaxDD': max_dd,
        'MaxDDDays': max_dd_days,
        'Vol': vol,
        'DownsideVol': downside_vol,
        'WinRate': win_rate,
        'PayoffRatio': payoff,
        'N_Days': len(seg),
    }


def load_strategies() -> dict[str, pd.Series]:
    """加载 21 策略 NAV."""
    navs = {}

    df = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")
    keep = ['v0.0 baseline', 'v0.1 +VT', 'v0.2 +TF', 'v1.0 locked', 'v3 (52 池)',
            'v4 style', 'v4 factor',
            'v5 量价', 'v5.1 量价 (逆波动)',
            'v7.10 TV-PR (标准化+CV)']
    for col in keep:
        if col in df.columns:
            navs[col] = df[col]

    # === v7.10 TV-PR 4 档成本补足 (NEW 2026-07-24) ===
    for cost_label in ['5bp', '10bp', '15bp', '20bp']:
        path = OUT_DIR / f'v7_10_v56_{cost_label}.parquet'
        if path.exists():
            navs[f'v7.10 TV-PR ({cost_label})'] = pd.read_parquet(path).iloc[:, 0]

    df = pd.read_parquet(OUT_DIR / "v6_navs.parquet")
    for col in ['v6 全风控']:
        if col in df.columns:
            navs[col] = df[col]

    df = pd.read_parquet(OUT_DIR / "v8_all_navs_v56.parquet")
    if 'v8_prob_3state_cost10bp' in df.columns:
        navs['v8 prob 3state 10bp'] = df['v8_prob_3state_cost10bp']
    if 'v8_prob_2state_cost10bp' in df.columns:
        navs['v8 prob 2state 10bp'] = df['v8_prob_2state_cost10bp']

    df = pd.read_parquet(OUT_DIR / "v8_method_b_nav_v56.parquet")
    navs['v8 method_b (有未来)'] = df.iloc[:, 0] 

    df = pd.read_parquet(OUT_DIR / "v8_per_asset_C1_5bp.parquet")
    navs['v8 per-asset 5bp'] = df.iloc[:, 0]

    df = pd.read_parquet(OUT_DIR / "v9_navs.parquet")
    if '银河方案-动态仓位' in df.columns:
        navs['v9 银河方案-动态仓位'] = df['银河方案-动态仓位']
    if '银河因子配置' in df.columns:
        navs['v9 银河因子配置'] = df['银河因子配置']
    if '等权基准' in df.columns:
        navs['v9 等权基准'] = df['等权基准']

    df = pd.read_parquet(OUT_DIR / "v9_macro_best_C5.parquet")
    navs['⭐ v8+v9 macro 5bp (NEW)'] = df.iloc[:, 0]

    df = pd.read_parquet(OUT_DIR / "v9_macro_best_C10.parquet")
    navs['v8+v9 macro 10bp (NEW)'] = df.iloc[:, 0]

    bench_price = pd.read_parquet('data/high_freq_macro/v9_benchmark_沪深300.parquet')['沪深300指数']
    bench_ret = bench_price.pct_change()
    bench_nav = (1 + bench_ret).cumprod()
    bench_nav.iloc[0] = 1.0
    navs['沪深300 (benchmark)'] = bench_nav

    return navs


def main():
    print("=" * 70)
    print("21 策略 × 9 区间 × 9 指标 标准化对比")
    print("=" * 70)

    navs = load_strategies()
    print(f"\n加载 {len(navs)} 个策略 NAV")

    # === Wide 表 (策略 × 区间 × 指标) ===
    wide_rows = []
    for name, nav in navs.items():
        row = {'Strategy': name}
        for pname, ps, pe in PERIODS:
            m = metrics_extended(nav, ps, pe)
            if m is None:
                for k in ['Sharpe', 'Sortino', 'Calmar', 'MaxDD', 'MaxDDDays',
                         'AnnRet', 'Vol', 'DownsideVol', 'WinRate', 'PayoffRatio']:
                    row[f"{pname}_{k}"] = None
            else:
                for k, v in m.items():
                    if k == 'N_Days':
                        continue
                    row[f"{pname}_{k}"] = v
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_csv = OUT_DIR / "standard_comparison_wide.csv"
    wide_df.to_csv(wide_csv, index=False)
    print(f"\n[WIDE] {wide_csv}: {wide_df.shape}")

    # === Long 表 (策略 × 区间 一行) ===
    long_rows = []
    for name, nav in navs.items():
        for pname, ps, pe in PERIODS:
            m = metrics_extended(nav, ps, pe)
            if m is None:
                continue
            row = {'Strategy': name, 'Period': pname, 'Start': ps, 'End': pe}
            row.update(m)
            long_rows.append(row)

    long_df = pd.DataFrame(long_rows)
    long_csv = OUT_DIR / "standard_comparison_long.csv"
    long_df.to_csv(long_csv, index=False)
    print(f"[LONG] {long_csv}: {long_df.shape}")

    # === Summary 表 (每策略 × Top5 计数 + 最佳场景) ===
    summary_rows = []
    for pname, _, _ in PERIODS:
        # 对每个 Sharpe/Scenario 列做 rank
        col = f"{pname}_Sharpe"
        if col not in wide_df.columns:
            continue
        ranked = wide_df[['Strategy', col]].dropna().copy()
        if len(ranked) == 0:
            continue
        ranked['Rank'] = ranked[col].rank(ascending=False)
        top5 = ranked[ranked['Rank'] <= 5]['Strategy'].tolist()
        summary_rows.append({
            'Period': pname,
            'Top5_Sharpe': ' / '.join(top5),
            'Best_Sharpe_Strategy': ranked.iloc[ranked[col].argmax()]['Strategy'],
            'Best_Sharpe_Value': ranked[col].max(),
            'Worst_Sharpe_Strategy': ranked.iloc[ranked[col].argmin()]['Strategy'],
            'Worst_Sharpe_Value': ranked[col].min(),
            'Median_Sharpe': ranked[col].median(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = OUT_DIR / "standard_comparison_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[SUMMARY] {summary_csv}: {summary_df.shape}")

    # === Underwater 历史 (策略 × 全期 MaxDDDays) ===
    # 计算每个策略在 Full Sample 内的 under-water 时间序列
    uw_rows = {}
    for name, nav in navs.items():
        seg = nav.loc['2018-01-03':'2026-05-29'].dropna()
        peak = seg.cummax()
        dd = seg / peak - 1
        underwater = (dd < 0).astype(int)
        uw_rows[name] = underwater
    uw_df = pd.DataFrame(uw_rows)
    uw_csv = OUT_DIR / "standard_comparison_underwater.csv"
    uw_df.to_csv(uw_csv)
    print(f"[UNDERWATER] {uw_csv}: {uw_df.shape}")

    # === 月度 sub-period (4 策略 × OOS 月) ===
    key_strats = ['⭐ v8+v9 macro 5bp (NEW)', 'v8 per-asset 5bp', 'v7.10 TV-PR (标准化+CV)', 'v1.0 locked']
    monthly_rows = []
    months = pd.date_range('2022-01-01', '2026-05-01', freq='MS')
    for name in key_strats:
        if name not in navs:
            continue
        nav = navs[name]
        for month_start in months:
            month_end = month_start + pd.offsets.MonthEnd(0)
            seg = nav.loc[month_start:month_end].dropna()
            if len(seg) < 5:
                continue
            rets = seg.pct_change().dropna()
            monthly_ret = seg.iloc[-1] / seg.iloc[0] - 1

            row = {
                'Strategy': name,
                'Month': month_start.strftime('%Y-%m'),
                'TotalRet': float(monthly_ret),
                'Sharpe_equiv': float(monthly_ret / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0,
            }
            monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)
    pivot_monthly = monthly_df.pivot_table(
        index='Strategy', columns='Month', values='TotalRet'
    )
    pivot_csv = OUT_DIR / "standard_comparison_4strats_monthly.csv"
    pivot_monthly.to_csv(pivot_csv)
    print(f"[MONTHLY 4-STRATS] {pivot_csv}: {pivot_monthly.shape}")

    # === OOS 全期 Sharpe 排名 ===
    print("\n=== OOS 22-26 Sharpe 排名 ===")
    ranked_oos = wide_df[['Strategy', 'OOS 22-26_Sharpe']].dropna().sort_values('OOS 22-26_Sharpe', ascending=False)
    for i, (_, r) in enumerate(ranked_oos.iterrows(), 1):
        marker = ' ⭐' if 'NEW' in r['Strategy'] or 'v1.0' in r['Strategy'] else ''
        print(f"  {i:>2d}. {r['Strategy']:<35} Sharpe={r['OOS 22-26_Sharpe']:.3f}{marker}")

    print("\n=== Full Sample Sharpe 排名 ===")
    ranked_full = wide_df[['Strategy', 'Full Sample_Sharpe']].dropna().sort_values('Full Sample_Sharpe', ascending=False)
    for i, (_, r) in enumerate(ranked_full.iterrows(), 1):
        marker = ' ⭐' if 'NEW' in r['Strategy'] or 'v1.0' in r['Strategy'] else ''
        print(f"  {i:>2d}. {r['Strategy']:<35} Sharpe={r['Full Sample_Sharpe']:.3f}{marker}")

    print("\n✅ 5 份 CSV 已保存到 reports/momentum_etf_rotation/combo/")
    print("接下来: 生成 3 张可视化 (Standard Viz)")


if __name__ == "__main__":
    main()
