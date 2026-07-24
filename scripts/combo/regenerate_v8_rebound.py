# coding=utf-8
"""v8 Jump Model: probabilistic + 反弹检测器.

核心思路: 检测 P_bear 短期快速下降, 判定为反弹,
    反弹时强制满仓, 不让 v8 减仓错过底部.

为什么有效:
  v8 P_bear 在熊市反弹时仍维持高仓位, 错过底部.
  反弹检测器发现 P_bear 短期下降 -> 强制满仓.

用法:
  python3 scripts/combo/regenerate_v8_rebound.py

产出:
  reports/momentum_etf_rotation/combo/v8_rebound_*.parquet
  reports/momentum_etf_rotation/combo/v8_rebound_comparison.csv
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


def load_signals():
    """加载 P_bear 信号."""
    if SIGNAL_PKL.exists():
        log(f"加载已有信号: {SIGNAL_PKL}")
        with open(SIGNAL_PKL, 'rb') as f:
            return pickle.load(f)
    log("错误: 信号文件不存在")
    sys.exit(1)


def check_rebound(p_bear_series, lookback=10, threshold=0.10):
    """检查是否处于反弹中.

    Args:
        p_bear_series: P_bear 序列 (从最早到最新)
        lookback: 短期窗口
        threshold: P_bear 下降幅度阈值

    Returns:
        bool: 是否处于反弹中
    """
    if len(p_bear_series) < lookback + 1:
        return False

    recent = p_bear_series.iloc[-lookback:]
    p_now = recent.iloc[-1]
    p_past = recent.iloc[0]

    return (p_past - p_now) > threshold


def compute_nav_rebound(weekly_weights, daily_returns, signals,
                        rebound_lookback=10, rebound_threshold=0.10,
                        sigmoid_threshold=0.40, sigmoid_steepness=10,
                        ww_freq='M', cost_bp=20):
    """v8 + 反弹检测器."""

    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # P_bear 周度 (用于 sigmoid 仓位)
    weekly_bear_pct = {}
    # P_bear 日度 (用于反弹检测)
    daily_bear_series = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear = bear_pct.reindex(weekly_dates, method='ffill')
            weekly_bear_pct[code] = weekly_bear
            daily_bear_series[code] = bear_pct

    date_to_adjusted_weights = {}
    last_ww = None
    last_ww_week = -999

    n_ww_rebals = 0
    n_rebound_detections = 0

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

        # 1. 更新 weekly_weights (按频率)
        need_ww_rebal = False
        if ww_freq == 'W':
            need_ww_rebal = True
        elif ww_freq == 'M':
            if i + 1 < len(weekly_dates):
                need_ww_rebal = (wd.month != next_wd.month)
            else:
                need_ww_rebal = True
        if need_ww_rebal:
            last_ww = weekly_weights.loc[wd].copy()
            last_ww_week = i
            n_ww_rebals += 1

        # 2. 计算每周调整 (带反弹检测)
        adj_weights = (last_ww if last_ww is not None else weekly_weights.loc[wd]).copy()

        for asset in common_codes:
            if asset not in weekly_bear_pct:
                continue

            # 当前 P_bear
            current_bear = weekly_bear_pct[asset].loc[wd]
            if pd.isna(current_bear):
                current_bear = 0.0

            # 检查反弹: 用该 ETF 自己的日度 P_bear 序列
            rebound = False
            if asset in daily_bear_series:
                # 取到 wd 为止的 P_bear 序列
                daily_bear = daily_bear_series[asset]
                valid = daily_bear[daily_bear.index <= wd]
                if check_rebound(valid, rebound_lookback, rebound_threshold):
                    rebound = True
                    n_rebound_detections += 1

            if rebound:
                # 反弹中: 强制满仓
                adj = 1.0
            elif current_bear > sigmoid_threshold:
                # 熊市: sigmoid 减仓
                adj = sigmoid_adj(current_bear, sigmoid_threshold, sigmoid_steepness)
            else:
                # 平稳: 满仓
                adj = 1.0

            adj_weights[asset] *= adj

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

    return nav, implied_turnover, n_ww_rebals, n_rebound_detections


def main():
    log("=" * 70)
    log("v8 probabilistic + 反弹检测器")
    log("=" * 70)

    log("加载数据...")
    daily_returns = load_v56()
    log(f"v56: {daily_returns.shape}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    signals = load_signals()

    # 基线: 不加反弹检测 (月频)
    log("\n=== 基线: C2_WW_M (无反弹检测) ===")
    t0 = time.time()
    nav_base, _, _, _ = compute_nav_rebound(
        weekly_weights, daily_returns, signals,
        rebound_lookback=999, rebound_threshold=999,  # 永不触发
        ww_freq='M', cost_bp=20,
    )
    m_base = compute_metrics(nav_base.loc[OOS_START:].pct_change().dropna(), freq='D')
    log(f"  Sharpe={m_base['Sharpe']:.3f} Calmar={m_base['Calmar']:.3f} MaxDD={m_base['MaxDD']:.2%}")

    # 4 种反弹检测参数
    configs = [
        {'lookback': 10, 'threshold': 0.10, 'name': 'Y1', 'desc': '10d/0.10 (标准)'},
        {'lookback': 15, 'threshold': 0.15, 'name': 'Y2', 'desc': '15d/0.15 (略宽松)'},
        {'lookback': 5, 'threshold': 0.05, 'name': 'Y3', 'desc': '5d/0.05 (灵敏)'},
        {'lookback': 20, 'threshold': 0.20, 'name': 'Y4', 'desc': '20d/0.20 (宽松)'},
    ]

    results = []
    results.append({
        'name': 'Base',
        'lookback': 999, 'threshold': 999,
        'desc': '基线 (无反弹检测, C2_WW_M)',
        'cost_bp': 20,
        'Sharpe': m_base['Sharpe'],
        'Calmar': m_base['Calmar'],
        'AnnRet': m_base['AnnRet'],
        'MaxDD': m_base['MaxDD'],
        'turnover_x': 0.0,
        'n_ww_rebal': 0,
        'n_rebound': 0,
    })

    for cfg in configs:
        log(f"\n=== {cfg['name']}: {cfg['desc']} ===")
        t0 = time.time()
        nav, turnover, n_ww, n_rebound = compute_nav_rebound(
            weekly_weights, daily_returns, signals,
            rebound_lookback=cfg['lookback'],
            rebound_threshold=cfg['threshold'],
            ww_freq='M', cost_bp=20,
        )
        elapsed = time.time() - t0

        oos = nav.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')

        out_path = OUT_DIR / f"v8_rebound_{cfg['name']}.parquet"
        nav.to_frame('v8_rebound').to_parquet(out_path)

        log(f"  Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} "
            f"AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")
        log(f"  隐含换手率: {turnover:.1f}x  反弹检测: {n_rebound}次  耗时: {elapsed:.1f}s")

        results.append({
            'name': cfg['name'],
            'lookback': cfg['lookback'],
            'threshold': cfg['threshold'],
            'desc': cfg['desc'],
            'cost_bp': 20,
            'Sharpe': m['Sharpe'],
            'Calmar': m['Calmar'],
            'AnnRet': m['AnnRet'],
            'MaxDD': m['MaxDD'],
            'turnover_x': turnover,
            'n_ww_rebal': n_ww,
            'n_rebound': n_rebound,
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_rebound_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("全部完成!")
    log(f"对比表: {csv_path}")
    log("\n=== 总结 ===")
    log(df.to_string(index=False))
    log("\n=== 对比 ===")
    log("v7.10 单独 (双周频 5bp): Sharpe=0.922")
    log("C2_WW_M (月频, 无反弹检测): Sharpe=0.544")
    log("=" * 70)


if __name__ == "__main__":
    main()