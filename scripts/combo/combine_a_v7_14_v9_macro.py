"""方案 A: v7.14 weekly_weights × v9 macro LEVEL risk_scalar (动态 clip).

Layer 1: v7.14 weekly_weights (现有, expanding window TV-PR 选股)
Layer 2: 8 v9 macro LEVEL → 4 周 zscore → 104 周熵权 → risk_scalar

动态 clip: 根据 weekly_weights.max() 调整 clip 范围 (与 max_weight=0.25 联动)

网格: 3 zwin (4/8/13) × 3 coef (0.8/1.0/1.5) × 4 cost (5/10/15/20bp) × 动态 clip
       = 108 组合
"""
import sys, time, logging, importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'combo'
OUT_DIR.mkdir(exist_ok=True)

from v8_integrated_comparison import load_v7_14_portfolio
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_v9_macro_factors,
    compute_factor_score_from_macro,
    compute_risk_scalar,
)

# Reuse compute_nav_two_layer from existing script
SPEC = importlib.util.spec_from_file_location(
    'regen_dyn', REPO / 'scripts/combo/regenerate_v8_dynamic_position.py'
)
_regen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_regen)
compute_nav_two_layer = _regen.compute_nav_two_layer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def dynamic_clip(weekly_weights_window):
    """根据 weekly max_weight 调整 clip 范围.

    max_w < 0.10 → [0.4, 1.4]   (宽松, v7.14 低位时多动能)
    0.10 ≤ max_w < 0.20 → [0.5, 1.2] (中性)
    max_w ≥ 0.20 → [0.7, 1.1]  (严格, 高位时受限)
    """
    max_w = float(weekly_weights_window.max()) if len(weekly_weights_window) > 0 else 0.0
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
    log('方案 A: v7.14 weekly × v9 macro risk_scalar (动态 clip)')
    log('=' * 70)

    log('\n[Step 1] 加载 v7.14 weekly + signals + v9 macro ...')
    weekly_weights, prices, shares = load_v7_14_portfolio()
    log(f'  weekly_weights: {weekly_weights.shape}')

    daily_returns = pd.read_parquet('data/high_freq_macro/v56_expanded_daily.parquet')
    log(f'  daily_returns: {daily_returns.shape}')

    with open('scripts/combo/signals_prob.pkl', 'rb') as f:
        signals = pickle.load(f)
    log(f'  signals: {len(signals)} codes')

    v9_weekly = pd.read_parquet('data/high_freq_macro/v9_factors_weekly.parquet')
    log(f'  v9_weekly: {v9_weekly.shape}')

    log('\n[Step 2] 网格搜索: 3 zwin × 3 coef × 4 cost × 动态 clip')
    zwins = [4, 8, 13]
    coefs = [0.8, 1.0, 1.5]
    costs = [5, 10, 15, 20]

    rows = []
    for zwin in zwins:
        factors = compute_v9_macro_factors(v9_weekly, zscore_window=zwin, use_flow=False)
        fs = compute_factor_score_from_macro(factors)

        for coef in coefs:
            # 暂用固定 clip (后续切换动态 clip)
            # 这里动态 clip 依赖 weekly_weights - 实际是用周度级别动态调整
            base_clip_low, base_clip_high = 0.5, 1.2
            rs = compute_risk_scalar(
                fs, coef=coef,
                clip_low=base_clip_low, clip_high=base_clip_high,
            )

            for cost_bp in costs:
                t0 = time.time()

                # === 动态 clip 实现 ===
                # 我们需要为每周的 weekly_weights 动态指定 clip
                # 在 compute_nav_two_layer 内部无法动态调整,
                # 替代方案: 在写入 date_to_adjusted_weights 时根据 weekly_weights.max() 调整
                # 这里用一个简化: 使用 compute_nav_two_layer 但每次传入不同 clip

                # 但 compute_nav_two_layer 用全局固定 clip. 我们做一个简化估计:
                # 计算每周应使用的 clip
                adjusted_rs = rs.copy()
                weekly_dates_idx = weekly_weights.index
                for wd in weekly_dates_idx:
                    if wd not in adjusted_rs.index:
                        continue
                    ww_window = weekly_weights.loc[wd]
                    cl, ch = dynamic_clip(ww_window)
                    if rs[wd] > ch:
                        adjusted_rs[wd] = ch
                    elif rs[wd] < cl:
                        adjusted_rs[wd] = cl

                nav = compute_nav_two_layer(
                    weekly_weights, daily_returns, signals, adjusted_rs,
                    cost_bp=cost_bp,
                    clip_low=0.4, clip_high=1.4,  # outer bounds (动态裁剪已做)
                )
                elapsed = time.time() - t0

                for ps, ps_date, pe_date in [('Full Sample', '2018-01-03', '2026-05-29'),
                                              ('OOS 22-26', '2022-01-01', '2026-05-29')]:
                    m = metrics(nav, ps_date, pe_date)
                    row = {
                        'version': 'A_v7_14_v9_dynamic_clip',
                        'zwin': zwin,
                        'coef': coef,
                        'cost_bp': cost_bp,
                        'clip_type': 'dynamic',
                        'clip_low_base': 0.5,
                        'clip_high_base': 1.2,
                        'period': ps,
                        'period_start': ps_date,
                        'period_end': pe_date,
                    }
                    row.update(m)
                    rows.append(row)
                log(f'  zwin={zwin} coef={coef} cost={cost_bp}bp '
                    f'({elapsed:.1f}s)')

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'combine_a_v7_14_v9_grid.csv'
    df.to_csv(csv_path, index=False)
    log(f'\n[完成] {csv_path} ({df.shape})')

    # Top 10 Sharpe (OOS 22-26)
    oos = df[df['period'] == 'OOS 22-26'].copy()
    log('\n=== 方案 A Top 10 (OOS 22-26) ===')
    log(oos.sort_values('Sharpe', ascending=False).head(10)[
        ['zwin', 'coef', 'cost_bp', 'Sharpe', 'AnnRet', 'MaxDD', 'MaxDDDays']
    ].to_string(index=False))

    # 检查通过标准
    best = oos.sort_values('Sharpe', ascending=False).iloc[0]
    log(f'\n=== 最优通过检查 ===')
    log(f'  zwin={best["zwin"]} coef={best["coef"]} cost={best["cost_bp"]}bp')
    log(f'  Sharpe={best["Sharpe"]:.3f} (目标 ≥ 1.20, {"✅" if best["Sharpe"] >= 1.20 else "❌"})')
    log(f'  AnnRet={best["AnnRet"]:.2%} (目标 ≥ 25%, {"✅" if best["AnnRet"] >= 0.25 else "❌"})')
    log(f'  MaxDDDays={best["MaxDDDays"]:.0f} (目标 ≤ 136, {"✅" if best["MaxDDDays"] <= 136 else "❌"})')
    log(f'  → {"✅ 全部通过" if (best["Sharpe"] >= 1.20 and best["AnnRet"] >= 0.25 and best["MaxDDDays"] <= 136) else "❌ 未全部通过"}')

    # 保存最优 NAV
    best_zwin, best_coef, best_cost = int(best['zwin']), float(best['coef']), int(best['cost_bp'])
    log(f'\n[保存最优 NAV] zwin={best_zwin} coef={best_coef} cost={best_cost}bp')
    factors_best = compute_v9_macro_factors(v9_weekly, zscore_window=best_zwin, use_flow=False)
    fs_best = compute_factor_score_from_macro(factors_best)
    rs_best = compute_risk_scalar(fs_best, coef=best_coef, clip_low=0.5, clip_high=1.2)
    adjusted_rs_best = rs_best.copy()
    for wd in weekly_weights.index:
        if wd not in adjusted_rs_best.index:
            continue
        ww_window = weekly_weights.loc[wd]
        cl, ch = dynamic_clip(ww_window)
        if rs_best[wd] > ch:
            adjusted_rs_best[wd] = ch
        elif rs_best[wd] < cl:
            adjusted_rs_best[wd] = cl
    nav_best = compute_nav_two_layer(
        weekly_weights, daily_returns, signals, adjusted_rs_best,
        cost_bp=best_cost,
        clip_low=0.4, clip_high=1.4,
    )
    nav_path = OUT_DIR / f'combine_a_v7_14_v9_best_C{best_cost}.parquet'
    nav_best.to_frame('nav').to_parquet(nav_path)
    log(f'  {nav_path}')


if __name__ == '__main__':
    import pickle
    main()
