# coding=utf-8
"""v8 Jump Model: probabilistic + P_bear 信号阈值验证."""
import sys
import time
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


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_v56():
    return pd.read_parquet(HF_DIR / "v56_expanded_daily.parquet")


def build_probabilistic_signals(weekly_weights, daily_returns):
    signals = {}
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    for i, code in enumerate(common_codes):
        returns = daily_returns[code].dropna()
        if len(returns) < 1000:
            continue
        feats = compute_features(returns).dropna()
        common = returns.index.intersection(feats.index)
        rets_aligned = returns.loc[common]
        feats_aligned = feats.loc[common]
        states, probs = probabilistic_jump_model(rets_aligned, feats_aligned, retrain_every=30)
        cols = ['P_bull', 'P_bear'] if probs.shape[1] == 2 else ['P_bull', 'P_neutral', 'P_bear']
        probs_df = pd.DataFrame(probs, index=feats_aligned.index, columns=cols[:probs.shape[1]])
        signals[code] = probs_df
        if (i + 1) % 10 == 0:
            log(f"  probabilistic: {i+1}/{len(common_codes)} ETF 完成")
    return signals


def compute_nav_with_threshold(weekly_weights, daily_returns, signals, threshold=0.15, cost_bp=20):
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 关键修改: 为每个 ETF 创建 weekly_date → P_bear 的映射
    # 直接用 weekly_dates 的日期, 不用 resample
    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals:
            probs_df = signals[code]
            if 'P_bear' in probs_df.columns:
                # 用 reindex 对齐到 weekly_dates, 用 ffill 填充
                bear_pct = probs_df['P_bear']
                # 找到每个 weekly_date 对应的最近的 P_bear
                weekly_bear = bear_pct.reindex(weekly_dates, method='ffill')
                weekly_bear_pct[code] = weekly_bear

    date_to_adjusted_weights = {}
    prev_week_bear = {code: None for code in common_codes}  # 上周的 P_bear (每周都更新)

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

        need_rebalance = False
        adj = {}
        for asset in common_codes:
            # 直接从 weekly_bear_pct 获取当前 P_bear
            if asset in weekly_bear_pct:
                current_bear = weekly_bear_pct[asset].loc[wd]
            else:
                current_bear = 0.0
            if pd.isna(current_bear):
                current_bear = 0.0

            # 阈值判断: 比较当前值与上周值
            if prev_week_bear[asset] is not None:
                delta = abs(current_bear - prev_week_bear[asset])
                if delta > threshold:
                    need_rebalance = True

            # 关键: 每周都更新 prev_week_bear (无论是否调仓)
            prev_week_bear[asset] = current_bear

            if current_bear > 0.25:
                rf = 1.0 - (current_bear - 0.25) / (1.0 - 0.25)
                adj[asset] = max(rf, 0.0)
            else:
                adj[asset] = 1.0

        if not need_rebalance and i > 0:
            prev_wd = weekly_dates[i - 1]
            if prev_wd in date_to_adjusted_weights:
                for d in all_dates[(all_dates >= start) & (all_dates <= end)]:
                    date_to_adjusted_weights[d] = date_to_adjusted_weights[prev_wd].copy()
                continue

        adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            if asset in adj:
                adj_weights[asset] *= adj[asset]
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

    nav_rets = nav.pct_change().dropna()
    cost_drag = nav_rets.mean() * 252
    implied_turnover = cost_drag / (cost_bp / 10000) if cost_bp > 0 else 0

    return nav, implied_turnover


def main():
    log("=" * 70)
    log("v8 probabilistic + P_bear 信号阈值验证")
    log("=" * 70)
    log("加载数据...")
    daily_returns = load_v56()
    log(f"v56: {daily_returns.shape}")
    weekly_weights, _, _ = load_v7_14_portfolio()
    log(f"v7.14: {weekly_weights.shape}")

    log("生成 probabilistic 信号...")
    t0 = time.time()
    signals = build_probabilistic_signals(weekly_weights, daily_returns)
    log(f"信号生成耗时: {time.time()-t0:.0f}s, {len(signals)} ETF")

    thresholds = [0.005, 0.01, 0.015, 0.02]
    results = []

    for thresh in thresholds:
        log(f"\n=== 阈值: {thresh} ===")
        t0 = time.time()
        nav, turnover = compute_nav_with_threshold(
            weekly_weights, daily_returns, signals,
            threshold=thresh, cost_bp=20,
        )
        elapsed = time.time() - t0
        oos = nav.loc[OOS_START:].dropna()
        rets = oos.pct_change().dropna()
        m = compute_metrics(rets, freq='D')

        out_path = OUT_DIR / f"v8_probabilistic_thresh{thresh}.parquet"
        nav.to_frame('v8_probabilistic').to_parquet(out_path)

        log(f"  Sharpe={m['Sharpe']:.3f} Calmar={m['Calmar']:.3f} "
            f"AnnRet={m['AnnRet']:.2%} MaxDD={m['MaxDD']:.2%}")
        log(f"  隐含换手率: {turnover:.1f}x 耗时: {elapsed:.0f}s")

        results.append({
            'threshold': thresh, 'cost_bp': 20,
            'Sharpe': m['Sharpe'], 'Calmar': m['Calmar'],
            'AnnRet': m['AnnRet'], 'MaxDD': m['MaxDD'],
            'turnover_x': turnover, 'elapsed_s': elapsed,
        })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / "v8_probabilistic_threshold_comparison.csv"
    df.to_csv(csv_path, index=False)

    log("\n" + "=" * 70)
    log("全部完成!")
    log(f"对比表: {csv_path}")
    log("\n=== 总结 ===")
    log(df.to_string(index=False))
    log("\n=== 对比基线 ===")
    log("无阈值 20bp: Sharpe=0.237, 换手率=21.5x")
    log("=" * 70)


if __name__ == "__main__":
    main()