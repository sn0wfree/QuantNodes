"""v7.10 TV-PR 4 档成本补足 (5/10/15/20bp 单边).

✅ 无未来函数: 复用 expanding_window_tvpr (OOS 因果估计)
✅ 成本: V7_6Config 默认 commission_bp=5 + slippage_bp=5 = 10bp 单边

直接复用 v7_10_gen_nav.py 的精确调用链 (load_v7_10_data + BEST_LAMBDA_TV=0.06, BEST_LAMBDA_L1=0.105).
"""
import sys, time, logging
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
    calculate_daily_nav,
)

OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"

# v7.10 优化参数 (与 v7_10_gen_nav.py 一致)
BEST_LAMBDA_TV = 0.06
BEST_LAMBDA_L1 = 0.105
MIN_HISTORY = 52

# 4 档成本 (commission = slippage = 各一半)
COST_TIERS = [
    {'cost_bp': 5,  'commission_bp': 2.5, 'slippage_bp': 2.5, 'name': '5bp',
     'desc': '乐观 (各半)'},
    {'cost_bp': 10, 'commission_bp': 5.0, 'slippage_bp': 5.0, 'name': '10bp',
     'desc': '默认 (各半)'},
    {'cost_bp': 15, 'commission_bp': 7.5, 'slippage_bp': 7.5, 'name': '15bp',
     'desc': '保守 (各半)'},
    {'cost_bp': 20, 'commission_bp': 10.0, 'slippage_bp': 10.0, 'name': '20bp',
     'desc': '最坏 (各半)'},
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.info


def metrics(nav, period_start='2022-01-01', period_end='2026-05-29'):
    seg = nav.loc[period_start:period_end].dropna()
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
    if underwater.any():
        groups = (underwater != underwater.shift()).cumsum()
        max_dd_days = int(underwater.groupby(groups).sum().max())
    else:
        max_dd_days = 0
    win_rate = float((rets > 0).mean())
    pos_rets = rets[rets > 0]
    payoff = float(pos_rets.mean() / abs(neg_rets.mean())) if len(neg_rets) > 0 else 0.0

    return {
        'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar,
        'MaxDD': max_dd, 'MaxDDDays': max_dd_days,
        'AnnRet': ann_ret, 'Vol': vol, 'DownsideVol': downside_vol,
        'WinRate': win_rate, 'PayoffRatio': payoff, 'N_Days': len(seg),
    }


def calc_weekly_turnover(weekly_weights_wide: pd.DataFrame) -> float:
    avg_turnover = 0.0
    prev_w = None
    count = 0
    for d in weekly_weights_wide.index:
        w = weekly_weights_wide.loc[d]
        if prev_w is not None:
            avg_turnover += float((w - prev_w).abs().sum())
            count += 1
        prev_w = w
    return avg_turnover / max(count, 1)


def main():
    log("=" * 70)
    log("v7.10 TV-PR 4 档成本补足 (复用 v7_10_gen_nav.py 算法)")
    log("=" * 70)

    log("[Step 1] 加载 v7.10 数据...")
    X, Y, codes = load_v7_10_data()
    daily_ret = load_daily_etf_returns()
    log(f"  X {X.shape}, Y {Y.shape}, {len(codes)} codes, daily {daily_ret.shape}")

    log("\n[Step 2] β 估计 (expanding window, OOS, 无前视, BEST_LAMBDA_TV=0.06)...")
    t0 = time.time()
    beta = expanding_window_tvpr(
        Y, X, BEST_LAMBDA_TV, BEST_LAMBDA_L1,
        min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4,
    )
    log(f"  Beta {beta.shape}, {time.time()-t0:.1f}s")

    rows = []
    for tier in COST_TIERS:
        cost_bp = tier['cost_bp']
        commission_bp = tier['commission_bp']
        slippage_bp = tier['slippage_bp']
        name = tier['name']

        cfg = V7_6Config(
            commission_bp=commission_bp,
            slippage_bp=slippage_bp,
        )

        t0 = time.time()
        # construct_portfolio 返回 (nav_series, weights_df_wide)
        nav_w_series, weights_df_wide = construct_portfolio(Y, X, beta, cfg, return_weights=True)
        # weights_df_wide: index=date, columns=code (wide format)
        # calculate_daily_nav 期望 long format (date, code, weight)
        weights_long = weights_df_wide.reset_index().melt(
            id_vars=weights_df_wide.index.name or 'index',
            var_name='code',
            value_name='weight',
        ).rename(columns={weights_df_wide.index.name or 'index': 'date'})
        nav_d = calculate_daily_nav(weights_long, daily_ret, cfg)
        nav_d = nav_d / nav_d.iloc[0]
        log(f"  [{name}] cost={cost_bp}bp (comm={commission_bp}+slip={slippage_bp}) "
            f"{time.time()-t0:.1f}s, NAV last 5: {nav_d.iloc[-5:].tolist()}")

        # 输出 NAV
        out_path = OUT_DIR / f"v7_10_v56_{name}.parquet"
        nav_d.to_frame(f'v7.10_{name}').to_parquet(out_path)

        # 计算 turnover (用 wide-format weights)
        weekly_turnover = calc_weekly_turnover(weights_df_wide)
        ann_cost_bp = weekly_turnover * cost_bp * 52

        for period_name, ps, pe in [('Full Sample', '2018-01-03', '2026-05-29'),
                                     ('OOS 22-26', '2022-01-01', '2026-05-29')]:
            m = metrics(nav_d, ps, pe)
            row = {
                'cost_bp': cost_bp,
                'commission_bp': commission_bp,
                'slippage_bp': slippage_bp,
                'name': name,
                'period': period_name,
                'desc': tier['desc'],
            }
            row.update(m)
            row['annualized_cost_pct'] = ann_cost_bp / 100
            row['avg_weekly_turnover'] = weekly_turnover
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "v7_10_v56_4costs_comparison.csv"
    df.to_csv(csv_path, index=False)
    log(f"\n对比表: {csv_path}")

    log("\n" + "=" * 70)
    log("v7.10 TV-PR 4 档 OOS 22-26")
    log("=" * 70)
    oos = df[df['period'] == 'OOS 22-26']
    cols_show = ['name', 'cost_bp', 'commission_bp', 'slippage_bp',
                 'Sharpe', 'Sortino', 'Calmar', 'AnnRet', 'MaxDD', 'MaxDDDays',
                 'avg_weekly_turnover', 'annualized_cost_pct']
    log(oos[cols_show].to_string(index=False))

    log("\n" + "=" * 70)
    log("v7.10 TV-PR 4 档 Full Sample")
    log("=" * 70)
    full = df[df['period'] == 'Full Sample']
    log(full[cols_show].to_string(index=False))

    log("\n✅ 4 档 NAV + 对比表已保存")


if __name__ == '__main__':
    main()
