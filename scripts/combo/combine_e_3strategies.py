"""3 策略组合 (v7.10 TV-PR + v8+v9 macro + v1.0 locked).

逻辑:
  v1.0 locked: 极致防御 / 现金替代 (Sharpe 1.596, MaxDD -1.94%)
  v8+v9 macro 5bp: 中度 macro timing (Sharpe 1.165, MaxDD -17.29%)
  v7.10 TV-PR 5bp: 激进 alpha (Sharpe 1.238, AnnRet 25.43%)

3 策略 "完全互补", 组合后理论上 Sharpe 改善 (低相关 + alpha 加成).

方法: 月度 rebalance, 多种权重.
"""
import sys, time, logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))

OUT_DIR = REPO / 'reports/momentum_etf_rotation/combo'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def metrics(nav, ps='2022-01-01', pe='2026-05-29'):
    seg = nav.loc[ps:pe].dropna()
    rets = seg.pct_change().dropna()
    total = seg.iloc[-1] / seg.iloc[0] - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total) ** (1 / max(n_years, 1e-9)) - 1
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(ann_ret / vol) if vol > 0 else 0.0
    peak = seg.cummax()
    max_dd = float((seg / peak - 1).min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < -1e-6 else 0.0
    neg_rets = rets[rets < 0]
    downside_vol = float(neg_rets.std() * np.sqrt(252)) if len(neg_rets) > 1 else 0.0
    sortino = float(ann_ret / downside_vol) if downside_vol > 1e-9 else 0.0
    underwater = (seg < peak).astype(int)
    max_dd_days = int(underwater.groupby((underwater != underwater.shift()).cumsum()).sum().max()) if underwater.any() else 0
    win_rate = float((rets > 0).mean())
    pos_rets = rets[rets > 0]
    payoff = float(pos_rets.mean() / abs(neg_rets.mean())) if len(neg_rets) > 0 else 0.0
    return {'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
            'MaxDD': max_dd, 'MaxDDDays': max_dd_days,
            'AnnRet': ann_ret, 'Vol': vol, 'DownsideVol': downside_vol,
            'WinRate': win_rate, 'PayoffRatio': payoff}


def compute_combined_nav(navs: dict, weights: dict, rebal_freq='M'):
    """组合多个策略 NAV 加权, 按 rebal_freq 调仓.

    navs: {name: Series}
    weights: {name: float} (sum to 1)
    """
    common = list(navs.values())[0].index
    for nav in navs.values():
        common = common.intersection(nav.index)
    common = common.sort_values()

    aligned_navs = {name: nav.reindex(common) for name, nav in navs.items()}
    n_total = aligned_navs[list(aligned_navs.keys())[0]].iloc[0]

    # 月度调仓日
    common_dti = pd.DatetimeIndex(common)
    rebal_dates = common_dti.to_series().resample(rebal_freq).last().index
    rebal_idx = pd.DatetimeIndex(rebal_dates)

    nav_combined = pd.Series(1.0, index=common, dtype=float)
    # 每月调仓, 中间持有
    last_w = {name: 0.0 for name in weights}

    # 归一化初始权重
    total_w = sum(weights.values())
    norm_w = {k: v / total_w for k, v in weights.items()}

    for i, d in enumerate(common):
        # 检查是否调仓日
        if d in rebal_idx:
            # 在调仓日, 重置权重 (基于当前 NAV)
            total_nav = 0.0
            current = {name: aligned_navs[name].loc[d] for name in weights}
            cur_total = sum(current[name] for name in weights)
            last_w = {name: norm_w[name] for name in weights}
            nav_today = cur_total  # 全部归一化: 起始时 = 上一期 NAV

        # 计算今日 NAV = sum(各策略今日 NAV / 该策略上期 NAV * 各策略权重 * 组合上期 NAV)
        # 简化: 直接按各策略今日 NAV 与上期 NAV 比, 加权
        if i == 0:
            nav_combined.iloc[i] = 1.0
        else:
            port_ret = 0.0
            for name in weights:
                if pd.isna(aligned_navs[name].iloc[i]) or pd.isna(aligned_navs[name].iloc[i-1]):
                    r = 0.0
                else:
                    r = aligned_navs[name].iloc[i] / aligned_navs[name].iloc[i-1] - 1
                port_ret += last_w[name] * r
            nav_combined.iloc[i] = nav_combined.iloc[i-1] * (1 + port_ret)

    return nav_combined


def main():
    log('=' * 70)
    log('3 策略组合: v7.10 TV-PR + v8+v9 macro + v1.0 locked')
    log('=' * 70)

    # === 加载 3 策略 NAV ===
    df_a = pd.read_parquet('reports/momentum_etf_rotation/combo/unified_v1v5_navs_calA.parquet')
    v1 = df_a['v1.0 locked']
    v710 = pd.read_parquet('reports/momentum_etf_rotation/combo/v7_10_v56_5bp.parquet').iloc[:, 0]
    v9macro = pd.read_parquet('reports/momentum_etf_rotation/combo/v9_macro_best_C5.parquet')['nav']

    # 对齐
    common = v1.index.intersection(v710.index).intersection(v9macro.index)
    common = common.sort_values()
    v1 = v1.reindex(common)
    v710 = v710.reindex(common)
    v9macro = v9macro.reindex(common)
    navs = {'v1.0': v1, 'v9macro': v9macro, 'v710': v710}
    log(f'  共同区间: {len(common)} days, {common[0].date()} ~ {common[-1].date()}')

    # === 单策略 baseline ===
    log('\n=== 各策略 baseline (OOS 22-26) ===')
    for name, nav in navs.items():
        m = metrics(nav)
        log(f'  {name}: Sharpe={m["Sharpe"]:.3f} AnnRet={m["AnnRet"]:.2%} '
            f'MaxDD={m["MaxDD"]:.2%} MaxDDDays={m["MaxDDDays"]:.0f}')

    # === 11 个权重组合 ===
    weight_grid = [
        ('Equal 1/3',           {'v1.0': 1/3, 'v9macro': 1/3, 'v710': 1/3}),
        ('v1 重 0.50',          {'v1.0': 0.50, 'v9macro': 0.25, 'v710': 0.25}),
        ('v1 重 0.40',          {'v1.0': 0.40, 'v9macro': 0.30, 'v710': 0.30}),
        ('v1 重 0.30',          {'v1.0': 0.30, 'v9macro': 0.35, 'v710': 0.35}),
        ('v1 重 0.20',          {'v1.0': 0.20, 'v9macro': 0.40, 'v710': 0.40}),
        ('v1 重 0.10',          {'v1.0': 0.10, 'v9macro': 0.45, 'v710': 0.45}),
        ('v710 重 0.60',        {'v1.0': 0.20, 'v9macro': 0.20, 'v710': 0.60}),
        ('v710 重 0.70',        {'v1.0': 0.15, 'v9macro': 0.15, 'v710': 0.70}),
        ('v710 重 0.80',        {'v1.0': 0.10, 'v9macro': 0.10, 'v710': 0.80}),
        ('v710 重 0.90',        {'v1.0': 0.05, 'v9macro': 0.05, 'v710': 0.90}),
        ('Vol-parity 倒数 vol', None),  # 特殊处理
    ]

    log('\n=== 3 策略组合 (月度 rebalance) ===')
    rows = []
    for name, weights in weight_grid:
        if weights is None:
            # Vol-parity inverse-vol weights
            weights = {}
            target_vol = 0.08  # target 8% annual vol
            for n in navs:
                vol_n = navs[n].pct_change().std() * np.sqrt(252)
                w = (target_vol / 3) / vol_n  # each ~ target/3
                weights[n] = w
            # normalize
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            log(f'\n  Vol-parity weights: {weights}')

        t0 = time.time()
        nav_combined = compute_combined_nav(navs, weights, rebal_freq='M')
        elapsed = time.time() - t0

        m = metrics(nav_combined)

        # Find best (across periods)
        log(f'\n  [{name}] elapsed={elapsed:.1f}s')
        log(f'    weights: {weights}')
        log(f'    OOS Sharpe={m["Sharpe"]:.3f} AnnRet={m["AnnRet"]:.2%} '
            f'MaxDD={m["MaxDD"]:.2%} MaxDDDays={m["MaxDDDays"]:.0f}')
        log(f'    Sortino={m["Sortino"]:.3f} Calmar={m["Calmar"]:.3f}')

        row = {'config': name}
        for name_, w in weights.items():
            row[f'w_{name_}'] = w
        row.update(m)
        rows.append(row)

        # 保存最优 NAV
        if 'v710' in weights and weights['v710'] >= 0.5:
            nav_path = OUT_DIR / f'combine_e_3strat_{name.replace(" ", "_").replace("/","")}_C5.parquet'
            nav_combined.to_frame('nav').to_parquet(nav_path)
            log(f'    → saved: {nav_path}')

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'combine_e_3strategies_grid.csv'
    df.to_csv(csv_path, index=False)
    log(f'\n[完成] {csv_path}')

    log('\n=== 全部组合汇总 (OOS 22-26) ===')
    show_cols = ['config', 'w_v1.0', 'w_v9macro', 'w_v710', 'Sharpe', 'Sortino', 'Calmar',
                 'AnnRet', 'MaxDD', 'MaxDDDays']
    log(df.sort_values('Sharpe', ascending=False)[show_cols].to_string(index=False))

    log('\n=== 3 项标准检查 ===')
    for _, r in df.iterrows():
        ok_sharpe = r['Sharpe'] >= 1.20
        ok_annret = r['AnnRet'] >= 0.25
        ok_mddd = r['MaxDDDays'] <= 136
        ok = ok_sharpe and ok_annret and ok_mddd
        if ok or r['Sharpe'] >= 1.30:  # 显示 Sharpe 突破 1.30 的
            marker = ' ✅' if ok else ' (partial)'
            log(f'  {r["config"]:<20s} Sharpe={r["Sharpe"]:.3f}{"✅" if ok_sharpe else "❌"} '
                f'AnnRet={r["AnnRet"]:.2%}{"✅" if ok_annret else "❌"} '
                f'MaxDDDays={r["MaxDDDays"]:.0f}{"✅" if ok_mddd else "❌"} {marker}')


if __name__ == '__main__':
    main()
