"""方案 B1: v7.10 β 重训 + v9 macro 加为 X 因子.

把 8 v9 macro zscore 作为额外因子输入到 TV-PR β 估计, 让选股吸收宏观信息.

X_panel_enhanced: 在原始 K 维后追加 8 v9 macro zscore (T, N, K+8)
不影响 β 估计本身的算法, 只是在输入端加维度.

输出: 重新训练, 计算不同 v9 macro zwin (4/8/13 周) 的对比
"""
import sys, time, logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

OUT_DIR = REPO / 'reports/momentum_etf_rotation/combo'
OUT_DIR.mkdir(exist_ok=True)

from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config, construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_v9_macro_factors,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def build_v9_macro_panel(v9_zscore_df, Y_codes, weekly_dates):
    """把 v9 macro zscore (周频) 扩展到 (T, N, 8) 形式.

    v9 macro 是 weekly level, 每个 macro 对应一个 (T, 8) 时序.
    我们把它"广播"到每个 asset, 使得 β_t 同时估计 asset-specific 和 macro-common 维度.

    返回: (T_aligned, N, 8) 数组. macro 值每周一行.
    """
    aligned = v9_zscore_df.reindex(weekly_dates)
    aligned_filled = aligned.ffill().fillna(0.0)
    T_aligned = len(weekly_dates)
    N = len(Y_codes)
    K_macro = aligned_filled.shape[1]
    panel = np.zeros((T_aligned, N, K_macro))
    # 把 macro 值放在每一只 asset 的同一行 (asset-independent macro factor)
    for t in range(T_aligned):
        for k in range(K_macro):
            panel[t, :, k] = aligned_filled.iloc[t, k]
    return panel


def expand_X_panel_v9(X_panel, v9_panel):
    """在 K 维后追加 8 v9 macro.

    X_panel: (T, N, K_original)
    v9_panel: (T, N, K_macro=8)
    → result: (T, N, K_original + 8)
    """
    return np.concatenate([X_panel, v9_panel], axis=2)


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
    log('方案 B1: v7.10 β 重训 + 8 v9 macro 加为 X 因子')
    log('=' * 70)

    X, Y, codes = load_v7_10_data()
    daily_returns = load_daily_etf_returns()
    v9_weekly = pd.read_parquet('data/high_freq_macro/v9_factors_weekly.parquet')
    log(f'  v7.10 X {X.shape}, Y {Y.shape}')
    log(f'  v9_weekly: {v9_weekly.shape}')

    # Y 的 index 是 weekly 频率 (周频)
    weekly_dates = Y.index

    rows = []

    # --- Baseline: 不加 v9 macro (v7.10 原版) ---
    log('\n[Baseline] v7.10 β (无 v9 macro)...')
    beta_baseline = expanding_window_tvpr(
        Y, X, 0.06, 0.105,
        min_history=52, max_iter=200, tol=1e-5, step=4,
    )
    cfg = V7_6Config()
    _, ww_baseline = construct_portfolio(Y, X, beta_baseline, cfg, return_weights=True)
    log(f'  baseline weekly_weights: {ww_baseline.shape}')

    # --- 3 个 v9 macro zwin 网格 ---
    for v9_zwin in [4, 8, 13]:
        log(f'\n[v9 macro zwin={v9_zwin}] 计算 + 拼接 X_panel')
        v9_factors = compute_v9_macro_factors(v9_weekly, zscore_window=v9_zwin, use_flow=False)
        v9_panel = build_v9_macro_panel(v9_factors, codes, weekly_dates)
        log(f'  v9_panel: {v9_panel.shape}')

        # 拼接 X_panel: 把 8 个 macro zscore 沿 K 维追加
        X_enhanced = expand_X_panel_v9(X, v9_panel)
        log(f'  X_enhanced: {X_enhanced.shape}')

        # 重训 β (用相同 BEST_LAMBDA_TV=0.06, BEST_LAMBDA_L1=0.105)
        t0 = time.time()
        beta_enhanced = expanding_window_tvpr(
            Y, X_enhanced, 0.06, 0.105,
            min_history=52, max_iter=200, tol=1e-5, step=4,
        )
        log(f'  β estimate: {time.time()-t0:.1f}s, shape={beta_enhanced.shape}')

        # beta_enhanced 前 36 维是 v7.10 原 β (TV-PR 学到的), 后 8 维是 v9 macro β
        v7_part = beta_enhanced.iloc[:, :36]
        cfg = V7_6Config()
        _, ww_enhanced = construct_portfolio(Y, X_enhanced, beta_enhanced, cfg, return_weights=True)
        log(f'  weekly_weights (enhanced): {ww_enhanced.shape}')

        # 计算 NAV (使用 v7.10 原版 cost 处理)
        common_codes = [c for c in ww_enhanced.columns if c in daily_returns.columns]
        ww_e = ww_enhanced[common_codes]
        dr_e = daily_returns[common_codes]

        # 复用 v7_10_4costs 的 cost logic
        weights_long = ww_e.reset_index().melt(
            id_vars=ww_e.index.name or 'index',
            var_name='code', value_name='weight',
        ).rename(columns={ww_e.index.name or 'index': 'date'})
        from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import calculate_daily_nav
        for cost_bp, cname in [(5, '5bp'), (10, '10bp'), (15, '15bp'), (20, '20bp')]:
            cfg_t = V7_6Config(commission_bp=cost_bp/2, slippage_bp=cost_bp/2)
            nav = calculate_daily_nav(weights_long, dr_e, cfg_t)
            nav = nav / nav.iloc[0]

            for ps_name, ps_date, pe_date in [('Full Sample', '2018-01-03', '2026-05-29'),
                                              ('OOS 22-26', '2022-01-01', '2026-05-29')]:
                m = metrics(nav, ps_date, pe_date)
                row = {
                    'version': 'B1_v9_added_to_X',
                    'v9_zwin': v9_zwin,
                    'cost_bp': cost_bp,
                    'period': ps_name,
                }
                # β 维度统计
                beta_v9_part = beta_enhanced.iloc[:, 36:]  # 后 8 列是 v9 macro β
                for col in beta_v9_part.columns:
                    row[f'beta_{col[:10]}_mean'] = float(beta_v9_part[col].mean())
                row.update(m)
                rows.append(row)

            # 输出 NAV
            nav_path = OUT_DIR / f'combine_b1_v9zwin{v9_zwin}_C{cost_bp}.parquet'
            nav.to_frame('nav').to_parquet(nav_path)

    # --- Baseline (v7.10 原版, 不加 v9 macro) ---
    macro_cols_short = ['宏观增长因子', '宏观通胀因子_生活端', '宏观汇率因子', '无风险收益率']
    for cost_bp, cname in [(5, '5bp'), (10, '10bp'), (15, '15bp'), (20, '20bp')]:
        baseline_path = OUT_DIR / f'v7_10_v56_{cname}.parquet'
        baseline_df = pd.read_parquet(baseline_path)
        nav = baseline_df.iloc[:, 0]

        for ps_name, ps_date, pe_date in [('Full Sample', '2018-01-03', '2026-05-29'),
                                          ('OOS 22-26', '2022-01-01', '2026-05-29')]:
            m = metrics(nav, ps_date, pe_date)
            row = {
                'version': 'B1_baseline_NO_V9',
                'v9_zwin': 0,
                'cost_bp': cost_bp,
                'period': ps_name,
            }
            for col in macro_cols_short:
                row[f'beta_{col[:10]}_mean'] = np.nan
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'combine_b1_v9_added_x_grid.csv'
    df.to_csv(csv_path, index=False)
    log(f'\n[完成] {csv_path}')

    oos = df[df['period'] == 'OOS 22-26'].copy()
    log('\n=== 方案 B1 (OOS 22-26) 全组合 ===')
    cols_show = ['version', 'v9_zwin', 'cost_bp', 'Sharpe', 'AnnRet', 'MaxDD', 'MaxDDDays']
    log(oos.sort_values('Sharpe', ascending=False).head(20)[cols_show].to_string(index=False))

    # 检查通过标准 (基于 v9_zwin=4 最佳)
    if len(oos[oos['v9_zwin'] == 4]) > 0:
        sub = oos[oos['v9_zwin'] == 4].copy()
        if len(sub) > 0:
            best = sub.sort_values('Sharpe', ascending=False).iloc[0]
            log(f'\n=== B1 zwin=4 最优 ===')
            log(f'  v9_zwin={best["v9_zwin"]} cost={best["cost_bp"]}bp')
            log(f'  Sharpe={best["Sharpe"]:.3f} (≥ 1.20? {"✅" if best["Sharpe"] >= 1.20 else "❌"})')
            log(f'  AnnRet={best["AnnRet"]:.2%} (≥ 25%? {"✅" if best["AnnRet"] >= 0.25 else "❌"})')
            log(f'  MaxDDDays={best["MaxDDDays"]:.0f} (≤ 136? {"✅" if best["MaxDDDays"] <= 136 else "❌"})')
            log(f'  → {"✅ 全部通过" if (best["Sharpe"] >= 1.20 and best["AnnRet"] >= 0.25 and best["MaxDDDays"] <= 136) else "❌ 未全部通过"}')


if __name__ == '__main__':
    main()
