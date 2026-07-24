#!/usr/bin/env python3
# coding=utf-8
"""v8 集成策略对比 — v8_method_b vs v8_prob vs v8_uniform.

在 v7.14 TV-PR 框架上对比 4 个版本:
  - v8_method_b       : 主代码默认 (硬分类 + bear_threshold=0.25)
  - v8_prob_2state    : 概率化 2 状态, 仓位=P(bull)*1.0+P(bear)*0.0
  - v8_prob_3state    : 概率化 3 状态, 仓位=P·[1.0, 0.6, 0.0]
  - v8_uniform        : 不调整, 100% 仓位

每个版本:
  1. 对 weekly_weights 中每个资产跑 Jump Model
  2. 根据版本计算仓位调整因子
  3. 调整 weekly_weights (weight_i * factor_i)
  4. 计算日频 NAV (含 0/10/20bp 成本)

OOS: 2022-02-17 ~ 2026-06-30

输出:
  reports/momentum_etf_rotation/v8_integrated/
    ├── comparison_integrated.csv
    ├── summary.md
    └── nav_comparison.png
"""
from __future__ import annotations

import logging
import sys
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

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v8_integrated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOS_START = pd.Timestamp("2022-02-17")
OOS_END = pd.Timestamp("2026-06-30")

VERSIONS = ["v8_method_b", "v8_prob_2state", "v8_prob_3state", "v8_uniform"]
VERSION_COLORS = {
    "v8_method_b": "#B71C1C",
    "v8_prob_2state": "#0D47A1",
    "v8_prob_3state": "#1B5E20",
    "v8_uniform": "#757575",
}

COST_CANDIDATES = [0, 10, 20]

# Jump Model 参数 (与主代码默认一致)
JUMP_PENALTY = 50.0
N_RESTARTS = 5  # 诊断显示 5 已足够
N_ITER = 10
RANDOM_STATE = 42
TRAIN_WINDOW = 1000
RETRAIN_EVERY = 30
BEAR_THRESHOLD = 0.25

POSITION_WEIGHTS_3STATE = np.array([1.0, 0.6, 0.0])
POSITION_WEIGHTS_2STATE = np.array([1.0, 0.0])


# ============================================================
# 数据加载
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")


def load_v7_14_portfolio():
    """加载 v7.14 TV-PR 框架生成的周频权重和日频收益."""
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


# ============================================================
# 单资产 Jump Model (基于诊断脚本)
# ============================================================
def _train_jump_model_with_probs(
    returns: pd.Series,
    feats: pd.DataFrame,
    n_states: int = 2,
    n_restarts: int = N_RESTARTS,
    n_iter: int = N_ITER,
    jump_penalty: float = JUMP_PENALTY,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, dict, np.ndarray]:
    from scripts.v8_probabilistic_experiment import _dp_with_probs, _classify_states

    np.random.seed(random_state)
    feats_z = (feats.values - feats.values.mean(axis=0)) / (feats.values.std(axis=0) + 1e-10)

    best_cost = np.inf
    best_states = None
    best_centroids = None
    for _ in range(n_restarts):
        centroids = np.random.randn(n_states, feats_z.shape[1])
        for _ in range(n_iter):
            st, _ = _dp_with_probs(feats_z, centroids, jump_penalty, n_states, 50.0)
            for s in range(n_states):
                mask = st == s
                if mask.sum() > 0:
                    centroids[s] = feats_z[mask].mean(axis=0)
        cost = 0.0
        for tt in range(len(st)):
            cost += np.sum((feats_z[tt] - centroids[st[tt]]) ** 2)
        for tt in range(1, len(st)):
            if st[tt] != st[tt - 1]:
                cost += jump_penalty
        if cost < best_cost:
            best_cost = cost
            best_states = st.copy()
            best_centroids = centroids.copy()

    state_labels = _classify_states(best_states, returns, n_states)
    states_series = pd.Series(best_states, index=feats.index)

    _, probs = _dp_with_probs(feats_z, best_centroids, jump_penalty, n_states, 50.0)
    perm = np.zeros(n_states, dtype=int)
    for raw_id, sem_id in state_labels.items():
        perm[sem_id] = raw_id
    probs = probs[:, perm]
    return states_series, state_labels, probs


def compute_per_asset_signals(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
) -> dict:
    """对 weekly_weights 中的每个资产计算:
        - bear_pct_series: 滚动 60 日 Bear%
        - prob_2state: 2 状态概率
        - prob_3state: 3 状态概率

    Returns:
        dict: {asset: {
            'bear_pct': pd.Series,
            'prob_2state': pd.DataFrame (T, 2),
            'prob_3state': pd.DataFrame (T, 3),
        }}
    """
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

        # 2 状态
        states_2, labels_2, probs_2 = _train_jump_model_with_probs(
            rets_aligned, feats_aligned, n_states=2,
        )
        # 3 状态
        states_3, labels_3, probs_3 = _train_jump_model_with_probs(
            rets_aligned, feats_aligned, n_states=3,
        )

        # bear_pct (60 日滚动)
        bear_pct = states_2.rolling(60, min_periods=1).mean()

        signals[code] = {
            "bear_pct": bear_pct,
            "prob_2state": pd.DataFrame(probs_2, index=feats_aligned.index,
                                          columns=["P_bull", "P_bear"]),
            "prob_3state": pd.DataFrame(probs_3, index=feats_aligned.index,
                                          columns=["P_bull", "P_neutral", "P_bear"]),
        }
        logging.info(f"  {code}: states_2 mean={states_2.mean():.3f}, "
                     f"states_3 mean={states_3.mean():.3f}, "
                     f"bear_pct mean={bear_pct.mean():.3f}")
    return signals


def compute_position_adjustment(version: str, signal: dict, date: pd.Timestamp) -> dict:
    """给定版本和信号, 返回每个资产在该日期的仓位调整因子.

    Returns:
        dict: {asset: adjustment_factor (0~1)}
    """
    adjustments = {}
    for asset, sig in signal.items():
        bear_pct = sig["bear_pct"]
        prob_2 = sig["prob_2state"]
        prob_3 = sig["prob_3state"]

        if version == "v8_method_b":
            # 硬分类 + 阈值
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
            # 概率化 2 状态
            if date in prob_2.index:
                p = prob_2.loc[date].values
            else:
                before = prob_2[prob_2.index <= date]
                p = before.iloc[-1].values if len(before) > 0 else np.array([1.0, 0.0])
            adjustments[asset] = float(p @ POSITION_WEIGHTS_2STATE)

        elif version == "v8_prob_3state":
            # 概率化 3 状态
            if date in prob_3.index:
                p = prob_3.loc[date].values
            else:
                before = prob_3[prob_3.index <= date]
                p = before.iloc[-1].values if len(before) > 0 else np.array([1.0, 0.6, 0.0])
            adjustments[asset] = float(p @ POSITION_WEIGHTS_3STATE)

        elif version == "v8_uniform":
            adjustments[asset] = 1.0

        else:
            raise ValueError(f"Unknown version: {version}")

    return adjustments


# ============================================================
# 集成 NAV 计算 (与主代码 _compute_daily_nav_from_weights 一致)
# ============================================================
def compute_integrated_nav(
    weekly_weights: pd.DataFrame,
    daily_returns: pd.DataFrame,
    signals: dict,
    version: str,
    cost_bp: float = 0.0,
) -> pd.Series:
    """计算集成策略 NAV.

    逻辑 (与主代码 _compute_daily_nav_from_weights 一致):
      1. 每个调仓日计算 position adjustment (基于 Jump Model)
      2. weekly_weight_adjusted = weekly_weight * adjustment
      3. 调仓日扣除 turnover × cost_bp / 10000
    """
    # 对齐资产
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]

    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # 构建 date → adjusted_weights 映射
    # 与主代码一致: 周五 t 生成, 生效于下周一
    date_to_adjusted_weights = {}
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

        # 在该生效期内, 用该调仓日生效时点的信号计算仓位
        # 简化: 使用 wd 当天或最近的信号
        # 实际应该用生效期第一天 start 的信号, 但为了与主代码一致, 用 wd
        # 注意: 这是避免未来函数
        adj = compute_position_adjustment(version, signals, wd)
        adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            if asset in adj:
                adj_weights[asset] *= adj[asset]
        # 归一化 (与主代码一致)
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
            # 中国假期判断: ETF 收益全 NaN 当日, 跳过 (与 v7_6 数据一致)
            if row[common_codes].isna().all():
                nav.iloc[i] = nav.iloc[i - 1]
            else:
                ret = row.fillna(0.0)
                port_ret = float((w * ret).sum())
                # 成本
                cost_factor = 1.0
                if cost_bp > 0:
                    turnover = float((w - prev_w).abs().sum())
                    cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
                nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
                prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]

    return nav


# ============================================================
# 性能指标 (与主代码一致)
# ============================================================
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


# ============================================================
# 主实验
# ============================================================
def main():
    logging.info("=" * 70)
    logging.info("v8 集成策略对比 (v8_method_b vs v8_prob vs v8_uniform)")
    logging.info("=" * 70)

    daily_returns = load_daily_returns()
    weekly_weights, prices, shares = load_v7_14_portfolio()
    logging.info(f"weekly_weights: {weekly_weights.shape}")
    logging.info(f"daily_returns: {daily_returns.shape}")

    # 计算每个资产的 Jump Model 信号
    logging.info("\n计算每资产 Jump Model 信号...")
    signals = compute_per_asset_signals(weekly_weights, daily_returns)
    logging.info(f"共 {len(signals)} 个资产有信号")

    # 计算集成 NAV
    results = []
    navs = {}
    for version in VERSIONS:
        for cost_bp in COST_CANDIDATES:
            logging.info(f"\n  {version} (cost={cost_bp}bp)...")
            nav = compute_integrated_nav(
                weekly_weights, daily_returns, signals, version, cost_bp
            )
            nav_oos = nav.loc[OOS_START:OOS_END]
            m = performance_metrics(nav_oos)
            results.append({
                "version": version,
                "cost_bp": cost_bp,
                **m,
            })
            navs[(version, cost_bp)] = nav_oos
            logging.info(
                f"    Sharpe={m['sharpe']:.3f}, Calmar={m['calmar']:.3f}, "
                f"AnnRet={m['ann_return']*100:.2f}%, MaxDD={m['max_drawdown']*100:.2f}%"
            )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "comparison_integrated.csv", index=False)
    logging.info(f"\n详细结果已保存: {OUTPUT_DIR / 'comparison_integrated.csv'}")

    # 生成报告
    generate_summary(df, navs, OUTPUT_DIR)
    generate_charts(df, navs, OUTPUT_DIR)
    logging.info(f"\n报告已保存到: {OUTPUT_DIR}")


def generate_summary(df: pd.DataFrame, navs: dict, output_dir: Path):
    """生成 summary.md"""
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# v8 集成策略对比报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**OOS 区间**: {OOS_START.date()} ~ {OOS_END.date()}\n\n")

        f.write("## 实验设计\n\n")
        f.write("- **集成框架**: v7.14 TV-PR (top-N 选股) + v8 Jump Model 仓位调整\n")
        f.write("- **测试资产数**: 44 个 ETF (覆盖权益/债券/商品)\n")
        f.write("- **Walk-Forward 起点**: 不适用 (使用主代码的全样本 Jump Model)\n")
        f.write("- **测试成本**: 0 / 10 / 20 bp/单边\n\n")

        f.write("## 4 个版本对比\n\n")
        f.write("| 版本 | 仓位计算 | 调参 |\n")
        f.write("|------|----------|------|\n")
        f.write("| v8_method_b | 硬分类 + bear_threshold=0.25 | bear_threshold=0.25 |\n")
        f.write("| v8_prob_2state | 概率化 2 状态, 仓位=P·[1.0, 0.0] | **零调参** |\n")
        f.write("| v8_prob_3state | 概率化 3 状态, 仓位=P·[1.0, 0.6, 0.0] | **零调参** |\n")
        f.write("| v8_uniform | 不调整, 100% 仓位 | **零调参** |\n\n")

        # 各成本档对比
        f.write("## 各成本档结果\n\n")
        for cost_bp in COST_CANDIDATES:
            sub = df[df["cost_bp"] == cost_bp]
            if sub.empty:
                continue
            f.write(f"### 成本 = {cost_bp} bp/单边\n\n")
            f.write("| 版本 | AnnRet | Vol | Sharpe | MaxDD | Calmar |\n")
            f.write("|------|--------|-----|--------|-------|--------|\n")
            sub_sorted = sub.sort_values("sharpe", ascending=False)
            for _, r in sub_sorted.iterrows():
                f.write(
                    f"| {r['version']} | {r['ann_return']*100:.2f}% | {r['vol']*100:.2f}% | "
                    f"**{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                    f"**{r['calmar']:.3f}** |\n"
                )
            f.write("\n")

        # 成本敏感性
        f.write("## 成本敏感性\n\n")
        f.write("| 版本 | 0bp Sharpe | 10bp Sharpe | 20bp Sharpe | 10bp 损失 | 20bp 损失 |\n")
        f.write("|------|-----------|-------------|-------------|-----------|----------|\n")
        for version in VERSIONS:
            row = f"| {version} "
            sharpes = {}
            for cb in COST_CANDIDATES:
                sub = df[(df["version"] == version) & (df["cost_bp"] == cb)]
                if sub.empty:
                    row += f"| - |"
                    continue
                sharpes[cb] = sub["sharpe"].mean()
                row += f"| {sharpes[cb]:.3f} "
            if 0 in sharpes and 10 in sharpes:
                row += f"| {sharpes[10]-sharpes[0]:+.3f} "
            else:
                row += f"| - "
            if 0 in sharpes and 20 in sharpes:
                row += f"| {sharpes[20]-sharpes[0]:+.3f} |\n"
            else:
                row += f"| - |\n"
            f.write(row)

        # 最终判定
        f.write("\n## 最终判定\n\n")
        sub_10 = df[df["cost_bp"] == 10]
        if not sub_10.empty:
            avg_10 = sub_10.sort_values("sharpe", ascending=False)
            f.write("**10bp 标准成本下排序**:\n\n")
            for rank, (_, r) in enumerate(avg_10.iterrows(), 1):
                f.write(f"{rank}. **{r['version']}**: Sharpe={r['sharpe']:.3f}, "
                        f"Calmar={r['calmar']:.3f}, MaxDD={r['max_drawdown']*100:.2f}%\n")
            f.write("\n")
            best_row = avg_10.iloc[0]
            base_row = avg_10[avg_10["version"] == "v8_method_b"].iloc[0]
            uniform_row = avg_10[avg_10["version"] == "v8_uniform"].iloc[0]
            f.write(f"**结论**: {best_row['version']} 在 10bp 成本下最优 "
                    f"(Sharpe={best_row['sharpe']:.3f})\n")
            f.write(f"- vs v8_method_b: {best_row['sharpe']-base_row['sharpe']:+.3f} Sharpe\n")
            f.write(f"- vs v8_uniform: {best_row['sharpe']-uniform_row['sharpe']:+.3f} Sharpe\n")


def generate_charts(df: pd.DataFrame, navs: dict, output_dir: Path):
    """生成对比图"""
    # 1. 各版本 NAV 对比 (10bp 成本)
    fig, ax = plt.subplots(figsize=(14, 7))
    for version in VERSIONS:
        nav = navs.get((version, 10))
        if nav is None or len(nav) == 0:
            continue
        ax.plot(nav.index, nav.values, color=VERSION_COLORS[version],
                label=version, linewidth=2.0, alpha=0.85)
    ax.set_title("集成策略 NAV 对比 (10bp 成本, OOS 2022-02 ~ 2026-06)",
                 fontsize=13)
    ax.set_xlabel("日期")
    ax.set_ylabel("NAV (起点=1.0)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "nav_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"nav_comparison.png 已保存")


if __name__ == "__main__":
    main()