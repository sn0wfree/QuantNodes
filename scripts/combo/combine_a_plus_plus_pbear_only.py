"""方案 A++: v7.10 weekly + 仅 P_bear per-asset 月末调仓 (无 risk_scalar, no Layer 2).

这是 "v7.10 选股 + v8 Layer 1 per-asset 月末" 组合, 无 Layer 2 整体仓位调整.

Layer 1: per-asset sigmoid 月末 (P_bear 信号)
无 Layer 2 (rs = 1.0 恒等)

输出: 让 v7.10 的不衰减 dyna + v8 P_bear 月末防御结合
"""
import sys, time, logging, pickle, importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

OUT_DIR = REPO / 'reports/momentum_etf_rotation/combo'
OUT_DIR.mkdir(exist_ok=True)

from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import expanding_window_tvpr
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio, calculate_daily_nav,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_risk_scalar,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def sigmoid_adj(P_bear, threshold=0.50, steepness=10):
    if pd.isna(P_bear):
        return 1.0
    x = (P_bear - threshold) * steepness
    return 1.0 / (1.0 + np.exp(x))


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


def compute_two_layer_v7_pbear(
    weekly_weights, daily_returns, signals, cost_bp,
    sigmoid_threshold=0.50, sigmoid_steepness=10,
    clip_low=0.5, clip_high=1.2,
):
    """per-asset sigmoid 月末 × static rs=1.0 (即无 Layer 2 调整).

    final_position = per_asset_adj × 1.0 = per_asset_adj
    """
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear_pct[code] = bear_pct.reindex(weekly_dates, method='ffill')

    date_to_adjusted_weights = {}
    last_ww = None
    last_per_asset_adj = {code: 1.0 for code in common_codes}

    for i, wd in enumerate(weekly_dates):
        after = all_dates[all_dates > wd]
        if len(after) == 0:
            continue
        start = after[0]
        if i + 1 < len(weekly_dates):
            next_wd = weekly_dates[i + 1]
            before_next = all_dates[all_dates <= next_wd]
            if len(before_next) == 0:
                continue
            end = before_next[-1]
        else:
            end = all_dates[-1]

        is_month_end = (i + 1 >= len(weekly_dates)) or (wd.month != next_wd.month)

        if is_month_end:
            last_ww = weekly_weights.loc[wd].copy()
            for asset in common_codes:
                if asset not in weekly_bear_pct:
                    continue
                p_bear = weekly_bear_pct[asset].loc[wd]
                if pd.isna(p_bear):
                    p_bear = 0.0
                last_per_asset_adj[asset] = sigmoid_adj(p_bear, sigmoid_threshold, sigmoid_steepness)

        if last_ww is not None:
            adj_weights = last_ww.copy()
        else:
            adj_weights = weekly_weights.loc[wd].copy()

        for asset in common_codes:
            if asset in last_per_asset_adj:
                adj_weights[asset] *= last_per_asset_adj[asset]

        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adj_weights.copy()

    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_adjusted_weights.get(d)
        if w is not None:
            row = daily_returns.loc[d]
            if row[common_codes].isna().all():
                nav.iloc[i] = nav.iloc[i - 1]
            else:
                ret = row.fillna(0.0)
                port_ret = float((w * ret).sum())
                cost_factor = 1.0
                if cost_bp > 0:
                    turnover = float((w - prev_w).abs().sum())
                    cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
                nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
                prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]

    return nav


def main():
    log('=' * 70)
    log('方案 A++: v7.10 weekly + P_bear per-asset 月末 (无 risk_scalar)')
    log('=' * 70)

    X, Y, codes = load_v7_10_data()
    daily_returns = load_daily_etf_returns()
    with open('scripts/combo/signals_prob.pkl', 'rb') as f:
        signals = pickle.load(f)
    log(f'  v7.10 X={X.shape}, Y={Y.shape}, signals={len(signals)}')

    log('\n[Step 1] 训练 v7.10 β ...')
    beta = expanding_window_tvpr(
        Y, X, 0.06, 0.105,
        min_history=52, max_iter=200, tol=1e-5, step=4,
    )
    log(f'  β: {beta.shape}')

    log('\n[Step 2] weekly_weights ...')
    cfg = V7_6Config()
    _, weekly_weights = construct_portfolio(Y, X, beta, cfg, return_weights=True)
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    rows = []
    # 1 网格 + 4 成本档
    for cost_bp in [5, 10, 15, 20]:
        t0 = time.time()
        nav = compute_two_layer_v7_pbear(
            weekly_weights, daily_returns, signals, cost_bp,
        )
        elapsed = time.time() - t0

        for ps_name, ps_date, pe_date in [
            ('Full Sample', '2018-01-03', '2026-05-29'),
            ('OOS 22-26', '2022-01-01', '2026-05-29'),
        ]:
            m = metrics(nav, ps_date, pe_date)
            row = {
                'version': 'A++_v7_10_pbear_only',
                'cost_bp': cost_bp,
                'period': ps_name,
            }
            row.update(m)
            rows.append(row)
        log(f'  cost={cost_bp}bp ({elapsed:.1f}s), NAV end={nav.iloc[-1]:.4f}')

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'combine_a_plus_plus_pbear_only_grid.csv'
    df.to_csv(csv_path, index=False)
    log(f'\n[完成] {csv_path}')

    oos = df[df['period'] == 'OOS 22-26'].copy()
    log('\n=== 方案 A++ (OOS 22-26) ===')
    log(oos[['cost_bp', 'Sharpe', 'AnnRet', 'MaxDD', 'MaxDDDays']].to_string(index=False))

    if len(oos) > 0:
        best = oos.sort_values('Sharpe', ascending=False).iloc[0]
        log(f'\n=== A++ 最优通过检查 ===')
        log(f'  cost={best["cost_bp"]}bp')
        log(f'  Sharpe={best["Sharpe"]:.3f} (≥ 1.20? {"✅" if best["Sharpe"] >= 1.20 else "❌"})')
        log(f'  AnnRet={best["AnnRet"]:.2%} (≥ 25%? {"✅" if best["AnnRet"] >= 0.25 else "❌"})')
        log(f'  MaxDDDays={best["MaxDDDays"]:.0f} (≤ 136? {"✅" if best["MaxDDDays"] <= 136 else "❌"})')

        # 保存最优 NAV
        cost_bp = int(best['cost_bp'])
        # 重新算最优 NAV
        nav = compute_two_layer_v7_pbear(
            weekly_weights, daily_returns, signals, cost_bp,
        )
        nav_path = OUT_DIR / f'combine_a_plus_plus_pbear_only_C{cost_bp}.parquet'
        nav.to_frame('nav').to_parquet(nav_path)
        log(f'  {nav_path}')


if __name__ == '__main__':
    main()
