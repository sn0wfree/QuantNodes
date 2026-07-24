"""全样本 + OOS 关键期: 18 策略综合对比.

时段:
  - 全样本 2018-2026 (有 warm-up)
  - 2022-2026 (典型 OOS, 涵盖 2022 慢熊 + 924 反弹 + 慢牛)
  - 2024 Q3-Q4 (924 周期专项)
  - 2022 慢熊专项
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"


def metrics_for_period(nav: pd.Series, period_start: str, period_end: str, freq=252):
    """计算某时间段的 Sharpe/Calmar/MaxDD/AnnRet/Vol/WinRate."""
    seg = nav.loc[period_start:period_end].dropna()
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
    vol = rets.std() * np.sqrt(freq)
    sharpe = ann_ret / vol if vol > 0 else 0.0

    peak = seg.cummax()
    dd = seg / peak - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-6 else 0.0

    win_rate = float((rets > 0).mean())

    return {
        'Total_Return': float(total),
        'Ann_Ret': float(ann_ret),
        'Vol': float(vol),
        'Sharpe': float(sharpe),
        'Calmar': float(calmar),
        'MaxDD': max_dd,
        'WinRate': win_rate,
        'N_Days': len(seg),
        'N_Years': float(n_years),
    }


def build_strategy_nav_table() -> dict:
    """从各 parquet 文件构建 (name -> nav Series) 字典."""
    navs = {}

    # 1. unified_v1v5_navs_calA (v0.0, v0.1, v0.2, v1.0 locked, v3, v4, v5, v5.1, v7.10)
    df = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")
    for col in df.columns:
        navs[col] = df[col]

    # 2. v6 全风控 (从 v6_navs)
    df = pd.read_parquet(OUT_DIR / "v6_navs.parquet")
    navs['v6 全风控'] = df['v6 全风控']

    # 3. v8 method_b (有未来函数)
    df = pd.read_parquet(OUT_DIR / "v8_method_b_nav_v56.parquet")
    navs['v8 method_b (有未来)'] = df.iloc[:, 0]

    # 4. v8 prob 3state cost 10bp (无未来, v8 早期方案)
    df = pd.read_parquet(OUT_DIR / "v8_all_navs_v56.parquet")
    # 取 cost 10bp 中间档
    col = 'v8_prob_3state_cost10bp'
    if col in df.columns:
        navs['v8 prob 3state 10bp'] = df[col]

    # 5. v8 prob 2state cost 10bp
    col = 'v8_prob_2state_cost10bp'
    if col in df.columns:
        navs['v8 prob 2state 10bp'] = df[col]

    # 6. v8 per-asset 5bp
    df = pd.read_parquet(OUT_DIR / "v8_per_asset_C1_5bp.parquet")
    navs['v8 per-asset 5bp'] = df.iloc[:, 0]

    # 7. v9 银河方案-动态仓位 (主要 v9 对比)
    df = pd.read_parquet(OUT_DIR / "v9_navs.parquet")
    if '银河方案-动态仓位' in df.columns:
        navs['v9 银河方案-动态仓位'] = df['银河方案-动态仓位']
    if '银河因子配置' in df.columns:
        navs['v9 银河因子配置'] = df['银河因子配置']
    if '等权基准' in df.columns:
        navs['v9 等权基准'] = df['等权基准']

    # 8. **NEW: v8 + v9 macro LEVEL 5bp**
    df = pd.read_parquet(OUT_DIR / "v9_macro_best_C5.parquet")
    navs['⭐ v8+v9 macro 5bp (NEW)'] = df.iloc[:, 0]

    # 9. **NEW: v8 + v9 macro LEVEL 10bp**
    df = pd.read_parquet(OUT_DIR / "v9_macro_best_C10.parquet")
    navs['v8+v9 macro 10bp (NEW)'] = df.iloc[:, 0] 

    # 10. 沪深300
    df = pd.read_parquet("data/high_freq_macro/v9_benchmark_沪深300.parquet")
    bench_price = df['沪深300指数']
    bench_ret = bench_price.pct_change()
    bench_nav = (1 + bench_ret).cumprod()
    bench_nav.iloc[0] = 1.0
    navs['沪深300 (benchmark)'] = bench_nav

    return navs


def main():
    navs = build_strategy_nav_table()
    print(f"加载 {len(navs)} 个策略 NAV\n")
    print(f"{'策略':<35} {'区间':<10}")
    for name, nav in navs.items():
        print(f"{name:<35} {nav.index[0].date()!s:>10} ~ {nav.index[-1].date()!s:>10}")

    # 各时段
    periods = [
        ('Full 18-26', '2018-01-01', '2026-06-30'),
        ('OOS 21-26',  '2021-08-01', '2026-06-30'),
        ('OOS 22-26',  '2022-01-01', '2026-06-30'),
        ('Slow Bear 22', '2022-01-01', '2022-12-31'),
        ('924 周期 24-08~24-10', '2024-08-01', '2024-10-31'),
        ('2025 慢牛',   '2025-01-01', '2025-12-31'),
    ]

    rows = []
    for name, nav in navs.items():
        row = {'Strategy': name}
        for pname, ps, pe in periods:
            m = metrics_for_period(nav, ps, pe)
            if m:
                row[f"{pname}_Sharpe"] = m['Sharpe']
                row[f"{pname}_Calmar"] = m['Calmar']
                row[f"{pname}_MaxDD"] = m['MaxDD']
                row[f"{pname}_AnnRet"] = m['Ann_Ret']
                row[f"{pname}_Vol"] = m['Vol']
                row[f"{pname}_WinRate"] = m['WinRate']
                row[f"{pname}_TotalRet"] = m['Total_Return']
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "full_sample_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n保存至: {csv_path}")

    # 关键时段输出表
    print('\n' + '=' * 140)
    print('📊 Full Sample 2018-2026 + 关键 OOS 区间 (Sharpe)')
    print('=' * 140)
    cols_to_show = ['Strategy']
    for pname, _, _ in periods:
        col = f"{pname}_Sharpe"
        if col in df.columns:
            cols_to_show.append(col)
    show = df[cols_to_show].copy()
    print(show.sort_values('Full 18-26_Sharpe', ascending=False).to_string(index=False))

    print('\n' + '=' * 140)
    print('📊 MaxDD (回撤)')
    print('=' * 140)
    cols_dd = ['Strategy']
    for pname, _, _ in periods:
        col = f"{pname}_MaxDD"
        if col in df.columns:
            cols_dd.append(col)
    show = df[cols_dd].copy()
    print(show.sort_values('Full 18-26_MaxDD').to_string(index=False))

    print('\n' + '=' * 140)
    print('📊 AnnRet (年化)')
    print('=' * 140)
    cols_ret = ['Strategy']
    for pname, _, _ in periods:
        col = f"{pname}_AnnRet"
        if col in df.columns:
            cols_ret.append(col)
    show = df[cols_ret].copy()
    print(show.sort_values('Full 18-26_AnnRet', ascending=False).to_string(index=False))

    # 重点: 924 周期对比
    print('\n' + '=' * 140)
    print('🔥 924 周期专项 (2024-08-01 ~ 2024-10-31)')
    print('=' * 140)
    print(f"\n{'Strategy':<35} {'Sharpe':>8s} {'AnnRet':>10s} {'MaxDD':>8s} {'WinRate':>8s} {'TotalRet':>10s}")
    for name, nav in navs.items():
        m = metrics_for_period(nav, '2024-08-01', '2024-10-31')
        if m:
            star = ' ⭐' if 'NEW' in name or 'v1.0 locked' in name else ''
            print(f"{name:<35} {m['Sharpe']:>8.3f} {m['Ann_Ret']:>10.2%} {m['MaxDD']:>8.2%} {m['WinRate']:>8.1%} {m['Total_Return']:>10.2%}{star}")

    # 2022 慢熊专项
    print('\n' + '=' * 140)
    print('🐻 2022 慢熊专项 (2022-01-01 ~ 2022-12-31)')
    print('=' * 140)
    print(f"\n{'Strategy':<35} {'Sharpe':>8s} {'AnnRet':>10s} {'MaxDD':>8s} {'WinRate':>8s} {'TotalRet':>10s}")
    for name, nav in navs.items():
        m = metrics_for_period(nav, '2022-01-01', '2022-12-31')
        if m:
            star = ' ⭐' if 'NEW' in name or 'v1.0 locked' in name else ''
            print(f"{name:<35} {m['Sharpe']:>8.3f} {m['Ann_Ret']:>10.2%} {m['MaxDD']:>8.2%} {m['WinRate']:>8.1%} {m['Total_Return']:>10.2%}{star}")


if __name__ == "__main__":
    main()
