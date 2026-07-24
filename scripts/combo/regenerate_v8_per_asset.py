# coding=utf-8
"""v8 Jump Model: per-asset 月末调仓 (Sigmoid 修复).

核心修复:
  1. 仓位公式: BEAR_THRESHOLD=0.25 -> Sigmoid(threshold=0.50, steepness=10)
     修复前 86% OOS 周触发 -> 修复后 ~5% OOS 周触发
  2. 调仓频率: 周频 -> 月末评估 P_bear, 月内保持
  3. per-asset: 每只 ETF 独立评估, 不聚合

为什么不用 signal_composer:
  composer 把多个 ETF 聚合成 composite signal, 违反 per-asset 原则.

输入:  scripts/combo/signals_prob.pkl (已有 P_bear 信号)
输出:  reports/momentum_etf_rotation/combo/v8_per_asset_*.parquet
       reports/momentum_etf_rotation/combo/v8_per_asset_comparison.csv
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

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


def load_signals():
    """加载 P_bear 信号 (pickle 复用)."""
    if not SIGNAL_PKL.exists():
        log(f"错误: 信号文件不存在 {SIGNAL_PKL}")
        sys.exit(1)
    log(f"加载已有信号: {SIGNAL_PKL}")
    with open(SIGNAL_PKL, 'rb') as f:
        return pickle.load(f)


def sigmoid_adj(P_bear, threshold=0.50, steepness=10):
    """per-asset Sigmoid 仓位函数.

    设计:
      P_bear < 0.35:  adj ≈ 1.0   (几乎满仓)
      P_bear ≈ 0.50:  adj = 0.5   (减半)
      P_bear > 0.65:  adj ≈ 0.0   (几乎空仓)

    对比原线性公式 (BEAR_THRESHOLD=0.25):
      原: 86% OOS 周触发 -> adj 平均 ~0.94 (永远轻微减仓)
      新: ~5% OOS 周触发 -> adj 平均 ~0.99 (基本满仓)
    """
    if pd.isna(P_bear):
        return 1.0
    x = (P_bear - threshold) * steepness
    return 1.0 / (1.0 + np.exp(x))


def compute_nav_per_asset(weekly_weights, daily_returns, signals,
                           threshold=0.50, steepness=10,
                           cost_bp=20):
    """per-asset P_bear 月末调仓.

    核心逻辑:
      - 每月最后一周更新 weekly_weights
      - 每月最后一周重新评估每只 ETF 的 P_bear
      - 月内保持 weekly_weights 和 adj 不变
      - 每只 ETF 独立计算 adj (per-asset)
    """
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建 weekly_bear_pct (per-asset)
    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear = bear_pct.reindex(weekly_dates, method='ffill')
            weekly_bear_pct[code] = weekly_bear

    date_to_adjusted_weights = {}

    # 上月末状态 (初始化)
    last_ww = None
    last_per_asset_adj = {code: 1.0 for code in common_codes}

    n_monthly_rebals = 0  # 月末调仓次数
    n_bear_reduce_events = 0  # per-asset adj<1 的总数

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

        # 判断月末
        is_month_end = False
        if i + 1 < len(weekly_dates):
            is_month_end = (wd.month != next_wd.month)
        else:
            is_month_end = True

        if is_month_end:
            # 月末: 1) 更新 weekly_weights  2) 重新评估 per-asset adj
            last_ww = weekly_weights.loc[wd].copy()
            n_monthly_rebals += 1

            for asset in common_codes:
                if asset not in weekly_bear_pct:
                    continue
                p_bear = weekly_bear_pct[asset].loc[wd]
                if pd.isna(p_bear):
                    p_bear = 0.0

                # per-asset sigmoid adj
                last_per_asset_adj[asset] = sigmoid_adj(p_bear, threshold, steepness)
                if last_per_asset_adj[asset] < 0.99:
                    n_bear_reduce_events += 1

        # 用 last_ww 和 last_per_asset_adj 构造 weekly 仓位
        if last_ww is not None:
            adj_weights = last_ww.copy()
        else:
            adj_weights = weekly_weights.loc[wd].copy()

        # 应用 per-asset adj
        for asset in common_codes:
            if asset in last_per_asset_adj:
                adj_weights[asset] *= last_per_asset_adj[asset]

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

    return nav, implied_turnover, n_monthly_rebals, n_bear_reduce_events


def main():
    log("=" * 70)
    log("v8 per-asset 月末调仓 (Sigmoid threshold=0.50)")
    log("=" * 70)

    log("加载数据...")
    daily_returns = load_v56()
    log(f"v56: {daily_returns.shape}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    signals = load_signals()

    # 3 种成本档
    cost_tiers = [
        {'cost_bp': 5, 'name': 'C1_5bp', 'desc': '5bp 现实成本'},
        {'cost_bp': 10, 'name': 'C2_10bp', 'desc': '10bp 保守成本'},
        {'cost_bp': 20, 'name': 'C3_20bp', 'desc': '20bp 最坏情况'},
    ]

    results = []
    for cfg in cost_tiers:
        log(f"\n=== {cfg['name']} ({cfg['desc']}) ===")
        t0 = time.time()
        nav, turnover, n_rebals, n_bear = compute_nav_per_asset(
            weekly_weights, daily_returns, signals,
            threshold=0.50, steepness=10,
            cost_bp=cfg['cost_bp'],
        )
        elapsed = time.time() - t0

        oos = nav.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')

        # 用真实 NAV 计算 max_dd (不用 compute_metrics 重建的 NAV)
        peak = oos.cummax()
        dd = oos / peak - 1
        max_dd_true = dd.min()

        out_path = OUT_DIR / f"v8_per_asset_{cfg['name']}.parquet"
        nav.to_frame('v8_per_asset').to_parquet(out_path)

        log(f"  Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} "
            f"AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")
        log(f"  换手率: {turnover:.1f}x  月末调仓: {n_rebals}次  "
            f"per-asset adj<1: {n_bear}次")

        results.append({
            'name': cfg['name'],
            'cost_bp': cfg['cost_bp'],
            'desc': cfg['desc'],
            'Sharpe': m['Sharpe'],
            'Calmar': m['Calmar'],
            'AnnRet': m['AnnRet'],
            'MaxDD': m['MaxDD'],
            'turnover_x': turnover,
            'n_rebals': n_rebals,
            'n_bear_adjust': n_bear,
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_per_asset_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("全部完成!")
    log(f"对比表: {csv_path}")
    log("\n=== 总结 ===")
    log(df.to_string(index=False))
    log("\n=== 对比基线 ===")
    log(f"v7.10 单独 (5bp):                Sharpe=0.922")
    log(f"v8 probabilistic 5bp (修复前):   Sharpe=0.767")
    log(f"v8 method_b (有未来, 不可实盘):  Sharpe=1.045")
    log(f"v8 per_asset sigmoid 0.50 (新):  Sharpe=????")
    log("=" * 70)


if __name__ == "__main__":
    main()