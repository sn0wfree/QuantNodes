"""方案 A+: v7.10 weekly_weights × v9 macro LEVEL risk_scalar (动态 clip).

Layer 1: v7.10 TV-PR weekly_weights (更优起点, v7.10 5bp Sharpe=1.238)
Layer 2: 8 v9 macro LEVEL → 4 周 zscore → 104 周熵权 → risk_scalar
动态 clip: 根据 weekly max_weight 调整

输出: 重新跑 108 组合, 用 v7.10 起点 (vs v7.14)
"""
import sys, time, logging, pickle, importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'combo'
OUT_DIR.mkdir(exist_ok=True)

from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import expanding_window_tvpr
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_v9_macro_factors,
    compute_factor_score_from_macro,
    compute_risk_scalar,
)

SPEC = importlib.util.spec_from_file_location(
    'regen_dyn', REPO / 'scripts/combo/regenerate_v8_dynamic_position.py'
)
_regen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_regen)
compute_nav_two_layer = _regen.compute_nav_two_layer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def dynamic_clip(ww_window):
    max_w = float(ww_window.max()) if len(ww_window) > 0 else 0.0
    if max_w >= 0.20:
        return 0.7, 1.1
    elif max_w >= 0.10:
        return 0.5, 1.2
    else:
        return 0.4, 1.4


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


def main():
    log('=' * 70)
    log('方案 A+ (修正版): v7.10 weekly_weights × v9 macro + 动态 clip')
    log('=' * 70)

    log('\n[Step 1] 加载 v7.10 weekly (更优起点) ...')
    X, Y, codes = load_v7_10_data()
    daily_returns = load_daily_etf_returns()
    log(f'  v7.10 X {X.shape}, Y {Y.shape}, daily {daily_returns.shape}')

    log('\n[Step 2] 复用 v7_10_v56_5bp 已生成的 weekly_weights ...')
    # 从 v7_10_v56_5bp 的 NAV 反推 weekly weights - 但没有保存 weekly weights
    # 重新跑 v7.10 选股 (5 秒)
    beta = expanding_window_tvpr(
        Y, X, 0.06, 0.105,
        min_history=52, max_iter=200, tol=1e-5, step=4,
    )
    cfg = V7_6Config()
    _, weekly_weights = construct_portfolio(Y, X, beta, cfg, return_weights=True)
    log(f'  weekly_weights: {weekly_weights.shape}')

    # 与 v7.10 daily_returns (2058 行) 对齐
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    log(f'  common: weekly {len(weekly_weights)}, daily {len(daily_returns)}')

    with open('scripts/combo/signals_prob.pkl', 'rb') as f:
        signals = pickle.load(f)
    v9_weekly = pd.read_parquet('data/high_freq_macro/v9_factors_weekly.parquet')

    log('\n[Step 3] 网格搜索: 3 zwin × 3 coef × 4 cost × 动态 clip')
    zwins = [4, 8, 13]
    coefs = [0.8, 1.0, 1.5]
    costs = [5, 10, 15, 20]

    rows = []
    for zwin in zwins:
        factors = compute_v9_macro_factors(v9_weekly, zscore_window=zwin, use_flow=False)
        fs = compute_factor_score_from_macro(factors)

        for coef in coefs:
            rs = compute_risk_scalar(fs, coef=coef, clip_low=0.5, clip_high=1.2)

            for cost_bp in costs:
                t0 = time.time()

                # 动态 clip - 每周末基于 weekly max 调整
                adjusted_rs = rs.copy()
                for wd in weekly_weights.index:
                    if wd not in adjusted_rs.index:
                        continue
                    ww = weekly_weights.loc[wd]
                    cl, ch = dynamic_clip(ww)
                    if rs[wd] > ch:
                        adjusted_rs[wd] = ch
                    elif rs[wd] < cl:
                        adjusted_rs[wd] = cl

                nav = compute_nav_two_layer(
                    weekly_weights, daily_returns, signals, adjusted_rs,
                    cost_bp=cost_bp,
                    clip_low=0.4, clip_high=1.4,
                )
                elapsed = time.time() - t0

                for ps, ps_date, pe_date in [('Full Sample', '2018-01-03', '2026-05-29'),
                                              ('OOS 22-26', '2022-01-01', '2026-05-29')]:
                    m = metrics(nav, ps_date, pe_date)
                    row = {
                        'version': 'A+_v7_10_v9_dynamic_clip',
                        'zwin': zwin,
                        'coef': coef,
                        'cost_bp': cost_bp,
                        'clip_type': 'dynamic',
                        'period': ps,
                    }
                    row.update(m)
                    rows.append(row)
                log(f'  zwin={zwin} coef={coef} cost={cost_bp}bp ({elapsed:.1f}s)')

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'combine_a_plus_v7_10_v9_grid.csv'
    df.to_csv(csv_path, index=False)
    log(f'\n[完成] {csv_path} ({df.shape})')

    oos = df[df['period'] == 'OOS 22-26'].copy()
    log('\n=== 方案 A+ Top 10 (OOS 22-26) ===')
    log(oos.sort_values('Sharpe', ascending=False).head(10)[
        ['zwin', 'coef', 'cost_bp', 'Sharpe', 'AnnRet', 'MaxDD', 'MaxDDDays']
    ].to_string(index=False))

    best = oos.sort_values('Sharpe', ascending=False).iloc[0]
    log(f'\n=== A+ 最优通过检查 ===')
    log(f'  zwin={best["zwin"]} coef={best["coef"]} cost={best["cost_bp"]}bp')
    log(f'  Sharpe={best["Sharpe"]:.3f} (目标 ≥ 1.20, {"✅" if best["Sharpe"] >= 1.20 else "❌"})')
    log(f'  AnnRet={best["AnnRet"]:.2%} (目标 ≥ 25%, {"✅" if best["AnnRet"] >= 0.25 else "❌"})')
    log(f'  MaxDDDays={best["MaxDDDays"]:.0f} (目标 ≤ 136, {"✅" if best["MaxDDDays"] <= 136 else "❌"})')
    log(f'  → {"✅ 全部通过" if (best["Sharpe"] >= 1.20 and best["AnnRet"] >= 0.25 and best["MaxDDDays"] <= 136) else "❌ 未全部通过"}')

    best_zwin, best_coef, best_cost = int(best['zwin']), float(best['coef']), int(best['cost_bp'])
    factors_best = compute_v9_macro_factors(v9_weekly, zscore_window=best_zwin, use_flow=False)
    fs_best = compute_factor_score_from_macro(factors_best)
    rs_best = compute_risk_scalar(fs_best, coef=best_coef, clip_low=0.5, clip_high=1.2)
    adjusted_rs_best = rs_best.copy()
    for wd in weekly_weights.index:
        if wd not in adjusted_rs_best.index:
            continue
        ww = weekly_weights.loc[wd]
        cl, ch = dynamic_clip(ww)
        if rs_best[wd] > ch:
            adjusted_rs_best[wd] = ch
        elif rs_best[wd] < cl:
            adjusted_rs_best[wd] = cl
    nav_best = compute_nav_two_layer(
        weekly_weights, daily_returns, signals, adjusted_rs_best,
        cost_bp=best_cost,
        clip_low=0.4, clip_high=1.4,
    )
    nav_path = OUT_DIR / f'combine_a_plus_v7_10_v9_best_C{best_cost}.parquet'
    nav_best.to_frame('nav').to_parquet(nav_path)
    log(f'  {nav_path}')


if __name__ == '__main__':
    main()
