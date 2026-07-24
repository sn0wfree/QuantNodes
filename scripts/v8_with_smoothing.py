#!/usr/bin/env python3
# coding=utf-8
"""v8 集成策略 + 平滑测试.

Part 1: 4 版本基线 (no smooth) × 3 成本 = 12
Part 2: 4 版本 + smooth(alpha=0.7, t=0.01) × 3 成本 = 12
Part 3: smooth 参数网格 (alpha × threshold) × 3 成本 = 27

OOS: 2022-01-01 ~ 2026-06-30
"""
from __future__ import annotations

import logging
import sys
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v8_with_smoothing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOS_START = pd.Timestamp("2022-01-01")
OOS_END = pd.Timestamp("2026-06-30")

VERSIONS = ["v8_method_b", "v8_prob_2state", "v8_prob_3state", "v8_uniform"]
VERSION_COLORS = {
    "v8_method_b": "#B71C1C",
    "v8_prob_2state": "#0D47A1",
    "v8_prob_3state": "#1B5E20",
    "v8_uniform": "#757575",
}
COST_CANDIDATES = [0, 10, 20]

JUMP_PENALTY = 50.0
N_RESTARTS = 5
N_ITER = 10
RANDOM_STATE = 42
TRAIN_WINDOW = 1000
BEAR_THRESHOLD = 0.25

POSITION_WEIGHTS_3STATE = np.array([1.0, 0.6, 0.0])
POSITION_WEIGHTS_2STATE = np.array([1.0, 0.0])

# smooth 默认参数 (主代码默认)
DEFAULT_SMOOTH_ALPHA = 0.7
DEFAULT_SMOOTH_THRESHOLD = 0.01

# smooth 参数网格
SMOOTH_ALPHA_GRID = [0.5, 0.7, 0.9]
SMOOTH_THRESHOLD_GRID = [0.01, 0.02, 0.05]


# ============================================================
# 数据加载与复用
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")


def load_v7_14_portfolio():
    from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
        V7_6Config, construct_portfolio_components,
    )
    from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import expanding_window_tvpr
    from QuantNodes.strategy.momentum_etf_rotation.v7.adapters import load_v7_14_data_uniform

    X, Y, codes = load_v7_14_data_uniform()
    cfg = V7_6Config()
    beta = expanding_window_tvpr(
        Y, X, cfg.lambda_tv, cfg.lambda_l1,
        min_history=cfg.min_history, step=cfg.step,
    )
    shares, prices, weekly_weights = construct_portfolio_components(Y, X, beta, cfg)
    return weekly_weights, prices, shares


def _train_jump_model_with_probs(
    returns: pd.Series,
    feats: pd.DataFrame,
    n_states: int = 2,
):
    from scripts.v8_probabilistic_experiment import _dp_with_probs, _classify_states

    np.random.seed(RANDOM_STATE)
    feats_z = (feats.values - feats.values.mean(axis=0)) / (feats.values.std(axis=0) + 1e-10)

    best_cost = np.inf
    best_states = None
    best_centroids = None
    for _ in range(N_RESTARTS):
        centroids = np.random.randn(n_states, feats_z.shape[1])
        for _ in range(N_ITER):
            st, _ = _dp_with_probs(feats_z, centroids, JUMP_PENALTY, n_states, 50.0)
            for s in range(n_states):
                mask = st == s
                if mask.sum() > 0:
                    centroids[s] = feats_z[mask].mean(axis=0)
        cost = 0.0
        for tt in range(len(st)):
            cost += np.sum((feats_z[tt] - centroids[st[tt]]) ** 2)
        for tt in range(1, len(st)):
            if st[tt] != st[tt - 1]:
                cost += JUMP_PENALTY
        if cost < best_cost:
            best_cost = cost
            best_states = st.copy()
            best_centroids = centroids.copy()

    state_labels = _classify_states(best_states, returns, n_states)
    states_series = pd.Series(best_states, index=feats.index)

    _, probs = _dp_with_probs(feats_z, best_centroids, JUMP_PENALTY, n_states, 50.0)
    perm = np.zeros(n_states, dtype=int)
    for raw_id, sem_id in state_labels.items():
        perm[sem_id] = raw_id
    probs = probs[:, perm]
    return states_series, state_labels, probs


def compute_per_asset_signals(weekly_weights: pd.DataFrame, daily_returns: pd.DataFrame) -> dict:
    from scripts.v8_probabilistic_experiment import compute_features

    signals = {}
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    for code in common_codes:
        returns = daily_returns[code].dropna()
        if len(returns) < TRAIN_WINDOW:
            continue
        feats = compute_features(returns).dropna()
        common = returns.index.intersection(feats.index)
        rets_aligned = returns.loc[common]
        feats_aligned = feats.loc[common]

        states_2, _, probs_2 = _train_jump_model_with_probs(rets_aligned, feats_aligned, n_states=2)
        _, _, probs_3 = _train_jump_model_with_probs(rets_aligned, feats_aligned, n_states=3)
        bear_pct = states_2.rolling(60, min_periods=1).mean()

        signals[code] = {
            "bear_pct": bear_pct,
            "prob_2state": pd.DataFrame(probs_2, index=feats_aligned.index,
                                          columns=["P_bull", "P_bear"]),
            "prob_3state": pd.DataFrame(probs_3, index=feats_aligned.index,
                                          columns=["P_bull", "P_neutral", "P_bear"]),
        }
    return signals


def compute_position_adjustment(version: str, signal: dict, date: pd.Timestamp) -> dict:
    adjustments = {}
    for asset, sig in signal.items():
        bear_pct = sig["bear_pct"]
        prob_2 = sig["prob_2state"]
        prob_3 = sig["prob_3state"]

        if version == "v8_method_b":
            if date in bear_pct.index:
                bp = bear_pct.loc[date]
            else:
                before = bear_pct[bear_pct.index <= date]
                bp = float(before.iloc[-1]) if len(before) > 0 else 0.0
            if pd.isna(bp):
                bp = 0.0
            if bp > BEAR_THRESHOLD:
                rf = 1.0 - (bp - BEAR_THRESHOLD) / (1.0 - BEAR_THRESHOLD)
                adjustments[asset] = max(rf, 0.0)
            else:
                adjustments[asset] = 1.0

        elif version == "v8_prob_2state":
            if date in prob_2.index:
                p = prob_2.loc[date].values
            else:
                before = prob_2[prob_2.index <= date]
                p = before.iloc[-1].values if len(before) > 0 else np.array([1.0, 0.0])
            adjustments[asset] = float(p @ POSITION_WEIGHTS_2STATE)

        elif version == "v8_prob_3state":
            if date in prob_3.index:
                p = prob_3.loc[date].values
            else:
                before = prob_3[prob_3.index <= date]
                p = before.iloc[-1].values if len(before) > 0 else np.array([1.0, 0.6, 0.0])
            adjustments[asset] = float(p @ POSITION_WEIGHTS_3STATE)

        elif version == "v8_uniform":
            adjustments[asset] = 1.0
    return adjustments


def smooth_weekly_weights(weights_df: pd.DataFrame, alpha: float = 0.7,
                          min_trade_threshold: float = 0.02) -> pd.DataFrame:
    """主代码 smooth 函数复制."""
    smoothed = weights_df.copy()
    for t in range(1, len(smoothed)):
        prev_w = smoothed.iloc[t - 1]
        new_w = weights_df.iloc[t]
        blended = alpha * new_w + (1 - alpha) * prev_w
        diff = blended - prev_w
        diff[diff.abs() < min_trade_threshold] = 0.0
        smoothed.iloc[t] = prev_w + diff
    row_sums = smoothed.sum(axis=1)
    mask = row_sums > 1.0
    smoothed.loc[mask] = smoothed.loc[mask].div(row_sums[mask], axis=0)
    return smoothed


def compute_integrated_nav(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    signals: dict,
    version: str,
    cost_bp: float = 0.0,
    smooth_alpha: float = None,
    smooth_threshold: float = None,
) -> tuple[pd.Series, float]:
    """计算集成策略 NAV.

    如果提供 smooth_alpha 和 smooth_threshold, 对调整后权重做平滑.
    Returns: (nav, weekly_turnover)
    """
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建调整后权重
    adjusted_w_list = []
    weekly_dates_used = []
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

        adj = compute_position_adjustment(version, signals, wd)
        adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            if asset in adj:
                adj_weights[asset] *= adj[asset]
        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        adjusted_w_list.append(adj_weights)
        weekly_dates_used.append(wd)

    adjusted_weights = pd.DataFrame(adjusted_w_list, index=weekly_dates_used)

    # 应用平滑 (如果有)
    if smooth_alpha is not None and smooth_threshold is not None:
        adjusted_weights = smooth_weekly_weights(
            adjusted_weights, alpha=smooth_alpha, min_trade_threshold=smooth_threshold,
        )

    # 计算 weekly turnover (平滑后)
    weekly_turnover = adjusted_weights.diff().abs().sum(axis=1).dropna().mean() * 52

    # 构建 date → weights 映射
    date_to_adjusted_weights = {}
    for i, wd in enumerate(weekly_dates_used):
        if i + 1 < len(weekly_dates_used):
            next_wd = weekly_dates_used[i + 1]
            before_next = all_dates[all_dates <= next_wd]
            if len(before_next) == 0:
                continue
            end = before_next[-1]
        else:
            end = all_dates[-1]
        after = all_dates[all_dates > wd]
        if len(after) == 0:
            continue
        start = after[0]
        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adjusted_weights.loc[wd].copy()

    # 计算 NAV
    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_adjusted_weights.get(d)
        if w is not None:
            ret = daily_returns.loc[d].fillna(0.0)
            port_ret = float((w * ret).sum())
            cost_factor = 1.0
            if cost_bp > 0:
                turnover = float((w - prev_w).abs().sum())
                cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
            nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
            prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]
    return nav, weekly_turnover


def performance_metrics(nav: pd.Series, freq: int = 252) -> dict:
    if nav.empty or len(nav) < 2:
        return {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "calmar": 0.0}
    n_years = len(rets) / freq
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    dd = nav / nav.cummax() - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0
    return {
        "ann_return": round(ann_ret, 4),
        "vol": round(vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
    }


def run_single(version, cost_bp, smooth_alpha, smooth_threshold, weekly_weights, daily_returns, signals):
    nav, weekly_to = compute_integrated_nav(
        weekly_weights, daily_returns, signals, version,
        cost_bp=cost_bp,
        smooth_alpha=smooth_alpha,
        smooth_threshold=smooth_threshold,
    )
    nav_oos = nav.loc[OOS_START:OOS_END]
    m = performance_metrics(nav_oos)
    return {**m, "weekly_turnover": round(weekly_to, 4)}


def main():
    logging.info("=" * 70)
    logging.info("v8 集成策略 + 平滑测试")
    logging.info("=" * 70)

    daily_returns = load_daily_returns()
    weekly_weights, prices, shares = load_v7_14_portfolio()
    logging.info(f"weekly_weights: {weekly_weights.shape}")
    logging.info(f"OOS: {OOS_START.date()} ~ {OOS_END.date()}")

    logging.info("\n计算每资产 Jump Model 信号...")
    signals = compute_per_asset_signals(weekly_weights, daily_returns)
    logging.info(f"共 {len(signals)} 个资产有信号")

    # ============================================================
    # Part 1+2: 4 版本 × smooth 开关 × 3 成本 = 24 测试
    # ============================================================
    logging.info("\n" + "=" * 70)
    logging.info("Part 1+2: 4 版本 × smooth 开关 × 3 成本 = 24 测试")
    logging.info("=" * 70)

    part12_results = []
    smooth_configs = [
        ("OFF", None, None),
        ("ON", DEFAULT_SMOOTH_ALPHA, DEFAULT_SMOOTH_THRESHOLD),
    ]

    for version in VERSIONS:
        for smooth_label, smooth_alpha, smooth_threshold in smooth_configs:
            for cost_bp in COST_CANDIDATES:
                logging.info(f"  {version} | smooth={smooth_label} | cost={cost_bp}bp ...")
                res = run_single(
                    version, cost_bp, smooth_alpha, smooth_threshold,
                    weekly_weights, daily_returns, signals,
                )
                part12_results.append({
                    "version": version,
                    "smooth": smooth_label,
                    "smooth_alpha": smooth_alpha if smooth_alpha else "N/A",
                    "smooth_threshold": smooth_threshold if smooth_threshold else "N/A",
                    "cost_bp": cost_bp,
                    **res,
                })
                logging.info(
                    f"    Sharpe={res['sharpe']:.3f}, Calmar={res['calmar']:.3f}, "
                    f"AnnRet={res['ann_return']*100:.2f}%, MaxDD={res['max_drawdown']*100:.2f}%, "
                    f"Turnover={res['weekly_turnover']:.2f}x"
                )

    df_part12 = pd.DataFrame(part12_results)
    df_part12.to_csv(OUTPUT_DIR / "comparison_smooth.csv", index=False)
    logging.info(f"\nPart 1+2 已保存: {OUTPUT_DIR / 'comparison_smooth.csv'}")

    # ============================================================
    # Part 3: smooth 参数网格 (针对 v8_method_b)
    # ============================================================
    logging.info("\n" + "=" * 70)
    logging.info("Part 3: smooth 参数网格 (alpha × threshold × cost)")
    logging.info("=" * 70)

    part3_results = []
    for alpha in SMOOTH_ALPHA_GRID:
        for threshold in SMOOTH_THRESHOLD_GRID:
            for cost_bp in COST_CANDIDATES:
                logging.info(f"  alpha={alpha} threshold={threshold} cost={cost_bp}bp ...")
                res = run_single(
                    "v8_method_b", cost_bp, alpha, threshold,
                    weekly_weights, daily_returns, signals,
                )
                part3_results.append({
                    "version": "v8_method_b",
                    "alpha": alpha,
                    "threshold": threshold,
                    "cost_bp": cost_bp,
                    **res,
                })
                logging.info(
                    f"    Sharpe={res['sharpe']:.3f}, Calmar={res['calmar']:.3f}, "
                    f"Turnover={res['weekly_turnover']:.2f}x"
                )

    df_part3 = pd.DataFrame(part3_results)
    df_part3.to_csv(OUTPUT_DIR / "smooth_param_grid.csv", index=False)
    logging.info(f"\nPart 3 已保存: {OUTPUT_DIR / 'smooth_param_grid.csv'}")

    # ============================================================
    # 生成综合报告
    # ============================================================
    generate_summary(df_part12, df_part3, OUTPUT_DIR)
    generate_charts(df_part12, OUTPUT_DIR)
    logging.info(f"\n报告已保存到: {OUTPUT_DIR}")


def generate_summary(df_part12: pd.DataFrame, df_part3: pd.DataFrame, output_dir: Path):
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# v8 集成策略 + 平滑测试报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**OOS 区间**: {OOS_START.date()} ~ {OOS_END.date()}\n\n")

        f.write("## Part 1+2: smooth 开关效果对比\n\n")
        f.write(f"**smooth ON 参数**: alpha={DEFAULT_SMOOTH_ALPHA}, threshold={DEFAULT_SMOOTH_THRESHOLD}\n\n")
        f.write("### 0bp 成本下\n\n")
        f.write("| 版本 | smooth | AnnRet | Vol | Sharpe | MaxDD | Calmar | Turnover |\n")
        f.write("|------|--------|--------|-----|--------|-------|--------|----------|\n")
        sub = df_part12[df_part12["cost_bp"] == 0]
        for _, r in sub.iterrows():
            f.write(
                f"| {r['version']} | {r['smooth']} | {r['ann_return']*100:.2f}% | "
                f"{r['vol']*100:.2f}% | **{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                f"**{r['calmar']:.3f}** | {r['weekly_turnover']:.2f}x |\n"
            )

        f.write("\n### 10bp 成本下 (标准)\n\n")
        f.write("| 版本 | smooth | AnnRet | Vol | Sharpe | MaxDD | Calmar | Turnover |\n")
        f.write("|------|--------|--------|-----|--------|-------|--------|----------|\n")
        sub = df_part12[df_part12["cost_bp"] == 10]
        sub_sorted = sub.sort_values("sharpe", ascending=False)
        for _, r in sub_sorted.iterrows():
            f.write(
                f"| {r['version']} | {r['smooth']} | {r['ann_return']*100:.2f}% | "
                f"{r['vol']*100:.2f}% | **{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                f"**{r['calmar']:.3f}** | {r['weekly_turnover']:.2f}x |\n"
            )

        f.write("\n### 20bp 成本下\n\n")
        f.write("| 版本 | smooth | AnnRet | Vol | Sharpe | MaxDD | Calmar | Turnover |\n")
        f.write("|------|--------|--------|-----|--------|-------|--------|----------|\n")
        sub = df_part12[df_part12["cost_bp"] == 20]
        sub_sorted = sub.sort_values("sharpe", ascending=False)
        for _, r in sub_sorted.iterrows():
            f.write(
                f"| {r['version']} | {r['smooth']} | {r['ann_return']*100:.2f}% | "
                f"{r['vol']*100:.2f}% | **{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                f"**{r['calmar']:.3f}** | {r['weekly_turnover']:.2f}x |\n"
            )

        # smooth 效果总结
        f.write("\n## smooth 效果分析\n\n")
        f.write("### smooth 对 Sharpe 的提升 (10bp 成本)\n\n")
        f.write("| 版本 | OFF Sharpe | ON Sharpe | 提升 |\n")
        f.write("|------|-----------|-----------|------|\n")
        for v in VERSIONS:
            off = df_part12[(df_part12["version"] == v) & (df_part12["smooth"] == "OFF") & (df_part12["cost_bp"] == 10)]
            on = df_part12[(df_part12["version"] == v) & (df_part12["smooth"] == "ON") & (df_part12["cost_bp"] == 10)]
            if len(off) > 0 and len(on) > 0:
                sharpe_off = off.iloc[0]["sharpe"]
                sharpe_on = on.iloc[0]["sharpe"]
                gain = sharpe_on - sharpe_off
                f.write(f"| {v} | {sharpe_off:.3f} | {sharpe_on:.3f} | {gain:+.3f} |\n")

        # 换手率下降
        f.write("\n### smooth 对换手率的影响\n\n")
        f.write("| 版本 | OFF Turnover | ON Turnover | 下降 |\n")
        f.write("|------|---------------|-------------|------|\n")
        for v in VERSIONS:
            off = df_part12[(df_part12["version"] == v) & (df_part12["smooth"] == "OFF")]
            on = df_part12[(df_part12["version"] == v) & (df_part12["smooth"] == "ON")]
            if len(off) > 0 and len(on) > 0:
                to_off = off.iloc[0]["weekly_turnover"]
                to_on = on.iloc[0]["weekly_turnover"]
                reduction = (to_off - to_on) / to_off * 100 if to_off > 0 else 0
                f.write(f"| {v} | {to_off:.2f}x | {to_on:.2f}x | {reduction:.1f}% |\n")

        # Part 3: smooth 参数网格
        f.write("\n## Part 3: smooth 参数网格 (v8_method_b)\n\n")
        f.write("### 10bp 成本下 Sharpe 矩阵\n\n")
        f.write("| alpha \\\\ threshold | 0.01 | 0.02 | 0.05 |\n")
        f.write("|-------------------|------|------|------|\n")
        for alpha in SMOOTH_ALPHA_GRID:
            row = f"| {alpha} "
            for threshold in SMOOTH_THRESHOLD_GRID:
                sub = df_part3[(df_part3["alpha"] == alpha) &
                               (df_part3["threshold"] == threshold) &
                               (df_part3["cost_bp"] == 10)]
                if len(sub) > 0:
                    sharpe = sub.iloc[0]["sharpe"]
                    row += f"| {sharpe:.3f} "
                else:
                    row += "| - "
            row += "|\n"
            f.write(row)

        f.write("\n### 10bp 成本下 Calmar 矩阵\n\n")
        f.write("| alpha \\\\ threshold | 0.01 | 0.02 | 0.05 |\n")
        f.write("|-------------------|------|------|------|\n")
        for alpha in SMOOTH_ALPHA_GRID:
            row = f"| {alpha} "
            for threshold in SMOOTH_THRESHOLD_GRID:
                sub = df_part3[(df_part3["alpha"] == alpha) &
                               (df_part3["threshold"] == threshold) &
                               (df_part3["cost_bp"] == 10)]
                if len(sub) > 0:
                    calmar = sub.iloc[0]["calmar"]
                    row += f"| {calmar:.3f} "
                else:
                    row += "| - "
            row += "|\n"
            f.write(row)

        # 最优参数
        f.write("\n### 最优 smooth 参数 (按 Sharpe)\n\n")
        sub_10bp = df_part3[df_part3["cost_bp"] == 10]
        if not sub_10bp.empty:
            best = sub_10bp.sort_values("sharpe", ascending=False).iloc[0]
            f.write(f"**最优**: alpha={best['alpha']}, threshold={best['threshold']}\n")
            f.write(f"- Sharpe: {best['sharpe']:.3f}\n")
            f.write(f"- Calmar: {best['calmar']:.3f}\n")
            f.write(f"- AnnRet: {best['ann_return']*100:.2f}%\n")
            f.write(f"- MaxDD: {best['max_drawdown']*100:.2f}%\n")
            f.write(f"- Turnover: {best['weekly_turnover']:.2f}x\n")


def generate_charts(df_part12: pd.DataFrame, output_dir: Path):
    """生成对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for idx, cost_bp in enumerate([0, 10]):
        ax = axes[idx]
        sub = df_part12[df_part12["cost_bp"] == cost_bp]
        if sub.empty:
            continue

        # 每个版本两个柱 (OFF vs ON)
        x = np.arange(len(VERSIONS))
        width = 0.35
        for vi, smooth_label in enumerate(["OFF", "ON"]):
            vals = []
            for v in VERSIONS:
                r = sub[(sub["version"] == v) & (sub["smooth"] == smooth_label)]
                vals.append(r.iloc[0]["sharpe"] if len(r) > 0 else 0)
            offset = (vi - 0.5) * width
            color = "#B0BEC5" if smooth_label == "OFF" else "#1976D2"
            ax.bar(x + offset, vals, width, color=color,
                   label=f"smooth {smooth_label}", edgecolor="black", linewidth=0.5)
            for i, v in enumerate(vals):
                ax.text(i + offset, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

        ax.set_title(f"cost={cost_bp}bp — Sharpe (smooth 开关)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(VERSIONS, rotation=15, fontsize=9)
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)

    plt.suptitle("smooth 开关对 Sharpe 的影响 (4 版本 × 2 成本)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "nav_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()