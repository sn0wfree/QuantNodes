# coding=utf-8
"""v8 Jump Model: probabilistic + 月频 P_bear 调仓.

核心改动: P_bear 仓位调整从每周改为月频 (每月只调整 1 次),
    大幅降低换手率.

用法:
  python3 scripts/combo/regenerate_v8_monthly.py

产出:
  reports/momentum_etf_rotation/combo/v8_monthly_*.parquet
  reports/momentum_etf_rotation/combo/v8_monthly_comparison.csv
"""
import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import compute_features
from v8_probabilistic_experiment import probabilistic_jump_model
from v8_integrated_comparison import load_v7_14_portfolio
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics

OOS_START = pd.Timestamp('2021-08-01')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"
HF_DIR = REPO / "data" / "high_freq_macro"
SIGNAL_PKL = Path(__file__).resolve().parent / "signals_prob.pkl"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_v56():
    return pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")


def sigmoid_adj(P_bear, threshold=0.40, steepness=10):
    """Sigmoid 仓位函数."""
    x = (P_bear - threshold) * steepness
    return 1.0 / (1.0 + np.exp(x))


def load_signals(weekly_weights):
    """加载已有信号."""
    if SIGNAL_PKL.exists():
        log(f"加载已有信号: {SIGNAL_PKL}")
        with open(SIGNAL_PKL, 'rb') as f:
            return pickle.load(f)
    log("错误: 信号文件不存在, 请先运行 regenerate_v8_sigmoid.py")
    sys.exit(1)


def compute_nav_monthly(weekly_weights, daily_returns, signals,
                         freq='M', threshold=0.40, steepness=10, cost_bp=20):
    """月频/双周/季频 P_bear 调仓.

    Args:
        freq: 'W' (原,每周), '2W' (双周), 'M' (月), 'Q' (季)
    """
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建 weekly_bear_pct
    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear = bear_pct.reindex(weekly_dates, method='ffill')
            weekly_bear_pct[code] = weekly_bear

    date_to_adjusted_weights = {}
    # 上次调仓的 adj (只在 P_bear 调仓日更新)
    last_adj = {asset: 1.0 for asset in common_codes}
    last_rebal_week = -999  # 上次调仓的周索引

    n_rebalances = 0  # 调试用: 统计实际调仓次数

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

        # 判断是否需要调 P_bear 仓位
        need_pbear_rebal = False
        if freq == 'W':
            need_pbear_rebal = True
        elif freq == '2W':
            if i - last_rebal_week >= 2:
                need_pbear_rebal = True
        elif freq == 'M':
            # 每月调一次: 检查 next_wd 月份是否变化
            if i + 1 < len(weekly_dates):
                need_pbear_rebal = (wd.month != next_wd.month)
            else:
                need_pbear_rebal = True
        elif freq == 'Q':
            # 每季调一次: 检查季度变化
            if i + 1 < len(weekly_dates):
                q_now = (wd.year, wd.month // 3)
                q_next = (next_wd.year, next_wd.month // 3)
                need_pbear_rebal = (q_now != q_next)
            else:
                need_pbear_rebal = True

        if need_pbear_rebal:
            # 计算新的 adj
            for asset in common_codes:
                if asset in weekly_bear_pct:
                    current_bear = weekly_bear_pct[asset].loc[wd]
                else:
                    current_bear = 0.0
                if pd.isna(current_bear):
                    current_bear = 0.0
                last_adj[asset] = sigmoid_adj(current_bear, threshold, steepness)
            last_rebal_week = i
            n_rebalances += 1

        # 每周都更新 weekly_weights, 但 adj 用 last_adj
        adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            adj_weights[asset] *= last_adj[asset]

        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adj_weights.copy()

    # 计算 NAV
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

    nav_rets = nav.pct_change().dropna()
    cost_drag = nav_rets.mean() * 252
    implied_turnover = cost_drag / (cost_bp / 10000) if cost_bp > 0 else 0

    log(f"  P_bear 实际调仓次数: {n_rebalances}")
    return nav, implied_turnover, n_rebalances


def main():
    log("=" * 70)
    log("v8 probabilistic + 月频 P_bear 调仓")
    log("=" * 70)

    log("加载数据...")
    daily_returns = load_v56()
    log(f"v56: {daily_returns.shape}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    signals = load_signals(weekly_weights)

    # 4 种 P_bear 调仓频率
    configs = [
        {'freq': 'W', 'name': 'F1_Week', 'desc': '每周 (原)'},
        {'freq': '2W', 'name': 'F2_BiWeek', 'desc': '双周'},
        {'freq': 'M', 'name': 'F3_Month', 'desc': '月频'},
        {'freq': 'Q', 'name': 'F4_Quarter', 'desc': '季频'},
    ]

    results = []
    for cfg in configs:
        log(f"\n=== {cfg['name']} ({cfg['desc']}) ===")
        t0 = time.time()
        nav, turnover, n_rebals = compute_nav_monthly(
            weekly_weights, daily_returns, signals,
            freq=cfg['freq'],
            threshold=0.40, steepness=10,
            cost_bp=20,
        )
        elapsed = time.time() - t0

        oos = nav.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')

        out_path = OUT_DIR / f"v8_monthly_{cfg['name']}.parquet"
        nav.to_frame('v8_monthly').to_parquet(out_path)

        log(f"  Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} "
            f"AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")
        log(f"  隐含换手率: {turnover:.1f}x  P_bear 调仓次数: {n_rebals}  耗时: {elapsed:.1f}s")

        results.append({
            'name': cfg['name'],
            'freq': cfg['freq'],
            'desc': cfg['desc'],
            'cost_bp': 20,
            'Sharpe': m['Sharpe'],
            'Calmar': m['Calmar'],
            'AnnRet': m['AnnRet'],
            'MaxDD': m['MaxDD'],
            'turnover_x': turnover,
            'n_pbear_rebal': n_rebals,
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_monthly_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("全部完成!")
    log(f"对比表: {csv_path}")
    log("\n=== 总结 ===")
    log(df.to_string(index=False))
    log("\n=== 对比基线 ===")
    log(f"原 v8 probabilistic 20bp (周频): Sharpe=0.354, 换手率=47x")
    log(f"v7.10 TV-PR (周频选股, 无 v8): Sharpe=0.922, 换手率低 (~10-15x 估算)")
    log("=" * 70)


if __name__ == "__main__":
    main()