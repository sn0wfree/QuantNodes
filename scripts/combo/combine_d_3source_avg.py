"""方案 D: v7.10 weekly × 3 源均权综合 dynamic 仓位.

3 源:
  Source 1 (1/3): v9 macro LEVEL factor_score (4 周 zscore)
  Source 2 (1/3): 每周 P_bear 均值 (P_bear 越高 → market 更 bearish → risk_scalar 越低)
  Source 3 (1/3): v7.10 TV-PR β 每日绝对值总和 (越高越主动)

输出: 3 source 均权 zscore 综合 → risk_scalar → 整体仓位调整

核心创新: 多源 zscore 加权, 比单 v9 macro 更稳定, 包括 P_bear 短期信号 + β 主动信号
"""
import sys, time, logging, pickle
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
    V7_6Config, construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_v9_macro_factors,
    compute_factor_score_from_macro,
)

import importlib.util
SPEC = importlib.util.spec_from_file_location(
    'regen_dyn', REPO / 'scripts/combo/regenerate_v8_dynamic_position.py'
)
_regen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_regen)
compute_nav_two_layer = _regen.compute_nav_two_layer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def compute_3_source_score(v9_weekly, signals, beta, weekly_dates, window=52, coef=0.5):
    """3 源 zscore 综合 → risk_scalar.

    Source 1: v9 macro factor_score (4 周 zscore)
    Source 2: P_bear 周均反向 (P_bear 高 → market 弱)
    Source 3: β 绝对值 (|β| 大 → 主动调整, 与 risk_scalar 弱关联 - 我们用 β 累积 zscore 作 'confidence')

    每个 source 先 zscore (滚动 52 周), 然后 1/3 + 1/3 + 1/3 综合.
    """
    # === Source 1: v9 macro LEVEL ===
    factors = compute_v9_macro_factors(v9_weekly, zscore_window=4, use_flow=False)
    f1 = compute_factor_score_from_macro(factors)  # 周频 Series
    f1 = f1.reindex(weekly_dates).ffill()
    f1_z = (f1 - f1.rolling(window).mean()) / (f1.rolling(window).std() + 1e-10)
    log(f'  Source 1 (v9 macro): mean={f1.mean():.3f}, std={f1.std():.3f}')

    # === Source 2: P_bear 周均反向 ===
    # 先合并 signals 的 P_bear (2007 天 × 43 资产)
    p_bear_df = pd.DataFrame({k: v['P_bear'] for k, v in signals.items()}).sort_index()
    # 每周平均 P_bear (用 resample)
    p_bear_weekly = p_bear_df.resample('W').last().mean(axis=1)
    p_bear_weekly = p_bear_weekly.reindex(weekly_dates).ffill()
    # 反向: P_bear 高意味着 market 看空 → risk_scalar 应该低
    f2 = -p_bear_weekly
    f2_z = (f2 - f2.rolling(window).mean()) / (f2.rolling(window).std() + 1e-10)
    log(f'  Source 2 (-P_bear weekly): mean={f2.mean():.3f}, std={f2.std():.3f}')

    # === Source 3: |β| 信号 (高 → 主动调整多) ===
    beta_abs_sum = beta.abs().sum(axis=1)  # (T,)
    beta_abs_sum = beta_abs_sum.reindex(weekly_dates).ffill()
    # 用 (mean - beta_abs_sum) 表示 "稳定 → 高 risk_scalar"
    f3 = -beta_abs_sum  # β 大就反向 (主动加仓时减少叠加)
    f3_z = (f3 - f3.rolling(window).mean()) / (f3.rolling(window).std() + 1e-10)
    log(f'  Source 3 (-|β| sum): mean={f3.mean():.3f}, std={f3.std():.3f}')

    # 3 源均权
    composite = (f1_z + f2_z + f3_z) / 3.0
    composite = composite.fillna(0.0)
    # clip & risk_scalar
    risk_scalar = (1 + coef * composite).clip(0.5, 1.2)
    return risk_scalar


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
    log('方案 D: v7.10 weekly × 3 源均权综合 risk_scalar')
    log('=' * 70)

    X, Y, codes = load_v7_10_data()
    daily_returns = load_daily_etf_returns()
    v9_weekly = pd.read_parquet('data/high_freq_macro/v9_factors_weekly.parquet')
    with open('scripts/combo/signals_prob.pkl', 'rb') as f:
        signals = pickle.load(f)

    log(f'  v7.10 X={X.shape}, Y={Y.shape}')
    log(f'  v9_weekly: {v9_weekly.shape}, signals: {len(signals)}')

    log('\n[Step 1] 训练 v7.10 β ...')
    beta = expanding_window_tvpr(
        Y, X, 0.06, 0.105,
        min_history=52, max_iter=200, tol=1e-5, step=4,
    )
    log(f'  β: {beta.shape}')

    log('\n[Step 2] 构造 weekly_weights ...')
    cfg = V7_6Config()
    _, weekly_weights = construct_portfolio(Y, X, beta, cfg, return_weights=True)
    log(f'  weekly_weights: {weekly_weights.shape}')

    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    # 网格: 3 coef × 4 cost = 12 组合
    coefs = [0.3, 0.5, 0.8]
    costs = [5, 10, 15, 20]
    weekly_dates = weekly_weights.index

    log('\n[Step 3] 计算 3 源综合 risk_scalar ...')
    risk_scalar = compute_3_source_score(
        v9_weekly, signals, beta, weekly_dates,
        window=52, coef=0.5,  # 此处 coef 不影响, 因为后面单独 grid
    )
    log(f'  risk_scalar shape: {risk_scalar.shape}, range: [{risk_scalar.min():.3f}, {risk_scalar.max():.3f}]')

    rows = []
    for coef in coefs:
        # 实际 risk_scalar 用 coef 缩放
        # 简单做法: 在 compute_3_source_score 内部 coef, 这里重新计算
        rs_scaled = compute_3_source_score(
            v9_weekly, signals, beta, weekly_dates, window=52, coef=coef,
        )
        for cost_bp in costs:
            t0 = time.time()
            nav = compute_nav_two_layer(
                weekly_weights, daily_returns, signals, rs_scaled,
                cost_bp=cost_bp,
                clip_low=0.5, clip_high=1.2,
            )
            elapsed = time.time() - t0

            for ps_name, ps_date, pe_date in [
                ('Full Sample', '2018-01-03', '2026-05-29'),
                ('OOS 22-26', '2022-01-01', '2026-05-29'),
            ]:
                m = metrics(nav, ps_date, pe_date)
                row = {
                    'version': 'D_3_source_avg',
                    'coef': coef,
                    'cost_bp': cost_bp,
                    'period': ps_name,
                }
                row.update(m)
                rows.append(row)
            log(f'  coef={coef} cost={cost_bp}bp ({elapsed:.1f}s)')

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / 'combine_d_3source_grid.csv'
    df.to_csv(csv_path, index=False)
    log(f'\n[完成] {csv_path}')

    oos = df[df['period'] == 'OOS 22-26'].copy()
    log('\n=== 方案 D (OOS 22-26) ===')
    log(oos.sort_values('Sharpe', ascending=False)[
        ['coef', 'cost_bp', 'Sharpe', 'AnnRet', 'MaxDD', 'MaxDDDays']
    ].to_string(index=False))

    # 通过检查
    for _, row in oos.iterrows():
        if row['cost_bp'] == 5:  # 只看 5bp
            log(f'\n=== D best (5bp) ===')
            log(f'  coef={row["coef"]} Sharpe={row["Sharpe"]:.3f}')
            log(f'  AnnRet={row["AnnRet"]:.2%}')
            log(f'  MaxDDDays={row["MaxDDDays"]:.0f}')
            log(f'  → Sharpe ≥ 1.20? {"✅" if row["Sharpe"] >= 1.20 else "❌"}')
            log(f'  → AnnRet ≥ 25%? {"✅" if row["AnnRet"] >= 0.25 else "❌"}')
            log(f'  → MaxDDDays ≤ 136? {"✅" if row["MaxDDDays"] <= 136 else "❌"}')


if __name__ == '__main__':
    main()
