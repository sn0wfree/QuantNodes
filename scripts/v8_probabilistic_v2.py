#!/usr/bin/env python3
# coding=utf-8
"""v8 概率化 Jump Model + Walk-Forward (修正版).

基于诊断结果 (Step 1-3) 的关键修正:
  - 数据过滤 bug 已修复: 用 iloc[:1000] 而非 loc[start:end]
  - bear 状态正常检测到 (不再是 0)
  - 保持 n_restarts=5 (诊断显示足够)
  - jump_penalty=50 默认 (参数评估确认合理)

4 个对比版本:
  v8_method_b       : 硬分类 + bear_threshold=0.25
  v8_prob_2state    : 概率化 2 状态, 仓位=P(bull)*1.0+P(bear)*0.0
  v8_prob_3state    : 概率化 3 状态, 仓位=P·[1.0, 0.6, 0.0]
  v8_uniform        : 等权 100%

测试:
  - 5 资产 × 3 walk-forward 起点
  - 3 档成本: 0 / 10 / 20 bp/单边
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

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v8_probabilistic_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_ASSETS = ["510300", "511260", "518880", "159915", "512760"]
ASSET_LABELS = {
    "510300": "沪深300", "511260": "国债", "518880": "黄金",
    "159915": "创业板", "512760": "半导体",
}
VERSIONS = ["v8_method_b", "v8_prob_2state", "v8_prob_3state", "v8_uniform"]
VERSION_COLORS = {
    "v8_method_b": "#B71C1C",
    "v8_prob_2state": "#0D47A1",
    "v8_prob_3state": "#1B5E20",
    "v8_uniform": "#757575",
}

WALK_FORWARD_STARTS = [
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2019-01-01"),
    pd.Timestamp("2020-01-01"),
]
TRAIN_WINDOW = 1000
TEST_WINDOW = 252
STEP = 60

JUMP_PENALTY = 50.0
N_RESTARTS = 5
N_ITER = 10
RANDOM_STATE = 42
BEAR_THRESHOLD = 0.25

COST_CANDIDATES = [0, 10, 20]
POSITION_WEIGHTS_3STATE = np.array([1.0, 0.6, 0.0])
POSITION_WEIGHTS_2STATE = np.array([1.0, 0.0])


# ============================================================
# 复用 v8_diagnostic 中的核心函数
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")


def compute_features_local(returns: pd.Series) -> pd.DataFrame:
    from scripts.v8_probabilistic_experiment import compute_features
    return compute_features(returns)


def _train_jump_model_with_probs(
    returns: pd.Series,
    feats: pd.DataFrame,
    n_states: int = 2,
    n_restarts: int = N_RESTARTS,
    n_iter: int = N_ITER,
    jump_penalty: float = JUMP_PENALTY,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, dict, np.ndarray]:
    """训练 Jump Model 并返回 states, state_mapping, final centroids."""
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
        # 计算 cost
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

    # 状态分类
    state_labels = _classify_states(best_states, returns, n_states)
    states_series = pd.Series(best_states, index=feats.index)

    # 计算概率
    _, probs = _dp_with_probs(feats_z, best_centroids, jump_penalty, n_states, 50.0)
    # 应用 semantic mapping (列重排)
    perm = np.zeros(n_states, dtype=int)
    for raw_id, sem_id in state_labels.items():
        perm[sem_id] = raw_id
    probs = probs[:, perm]
    return states_series, state_labels, probs


# ============================================================
# Walk-Forward 主函数
# ============================================================
def run_walk_forward_v2(
    asset: str,
    daily_returns: pd.DataFrame,
    version: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cost_bp: float = 0.0,
) -> dict:
    """修正版 walk-forward."""
    returns_full = daily_returns[asset].dropna().loc[start_date:end_date]

    all_test_returns = []
    all_positions = []
    all_costs = []
    prev_pos = 1.0

    for train_end_idx in range(TRAIN_WINDOW, len(returns_full) - TEST_WINDOW + 1, STEP):
        train_returns = returns_full.iloc[:train_end_idx].iloc[-TRAIN_WINDOW:]
        test_returns = returns_full.iloc[train_end_idx:train_end_idx + TEST_WINDOW]

        feats = compute_features_local(train_returns).dropna()
        common_idx = train_returns.index.intersection(feats.index)
        train_returns_aligned = train_returns.loc[common_idx]
        feats = feats.loc[common_idx]
        if len(feats) < 100:
            continue

        # 计算仓位
        if version == "v8_method_b":
            states, state_labels, _ = _train_jump_model_with_probs(
                train_returns_aligned, feats, n_states=2,
            )
            bear_pct = states.rolling(60, min_periods=1).mean().fillna(0.0)
            last_bear_pct = float(bear_pct.iloc[-1])
            if last_bear_pct > BEAR_THRESHOLD:
                rf = 1.0 - (last_bear_pct - BEAR_THRESHOLD) / (1.0 - BEAR_THRESHOLD)
                pos = max(rf, 0.0)
            else:
                pos = 1.0
            test_pos = np.full(len(test_returns), pos)

        elif version == "v8_prob_2state":
            _, _, probs = _train_jump_model_with_probs(
                train_returns_aligned, feats, n_states=2,
            )
            last_probs = probs[-1]
            pos = float(last_probs @ POSITION_WEIGHTS_2STATE)
            test_pos = np.full(len(test_returns), pos)

        elif version == "v8_prob_3state":
            _, _, probs = _train_jump_model_with_probs(
                train_returns_aligned, feats, n_states=3,
            )
            last_probs = probs[-1]
            pos = float(last_probs @ POSITION_WEIGHTS_3STATE)
            test_pos = np.full(len(test_returns), pos)

        elif version == "v8_uniform":
            test_pos = np.ones(len(test_returns))
        else:
            raise ValueError(f"Unknown version: {version}")

        # 调仓成本
        turnover = abs(test_pos[0] - prev_pos)
        if cost_bp > 0 and turnover > 0:
            cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
        else:
            cost_factor = 1.0

        all_test_returns.append(test_returns.values)
        all_positions.append(test_pos)
        all_costs.append(np.full(len(test_returns), 1.0))
        all_costs[-1][0] = cost_factor

        prev_pos = test_pos[-1]

    if not all_test_returns:
        return None

    # 拼接并计算 NAV
    test_rets_concat = np.concatenate(all_test_returns)
    positions_concat = np.concatenate(all_positions)
    costs_concat = np.concatenate(all_costs)

    adj_returns = test_rets_concat * positions_concat
    nav_vals = np.zeros(len(adj_returns))
    nav_vals[0] = (1 + np.nan_to_num(adj_returns[0], nan=0.0)) * costs_concat[0]
    for i in range(1, len(adj_returns)):
        nav_vals[i] = nav_vals[i - 1] * (1 + np.nan_to_num(adj_returns[i], nan=0.0)) * costs_concat[i]
    nav = pd.Series(nav_vals)
    nav = nav / nav.iloc[0]

    # 性能指标
    rets = nav.pct_change().dropna()
    if len(rets) < 2:
        return None
    n_years = len(rets) / 252
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(252))
    dd = nav / nav.cummax() - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0

    total_turnover = sum(
        abs(all_positions[i][0] - (all_positions[i - 1][-1] if i > 0 else 1.0))
        for i in range(len(all_positions))
    )

    return {
        "ann_return": round(ann_ret, 6),
        "vol": round(vol, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "n_windows": len(all_test_returns),
        "mean_position": round(float(np.mean(positions_concat)), 4),
        "position_std": round(float(np.std(positions_concat)), 4),
        "total_turnover": round(total_turnover, 4),
        "cost_bp": cost_bp,
    }


def main():
    logging.info("=" * 70)
    logging.info("v8 概率化 Jump Model + Walk-Forward (修正版)")
    logging.info("=" * 70)

    daily_returns = load_daily_returns()
    logging.info(f"ETF 数据: {daily_returns.shape}")

    results = []
    for asset in TEST_ASSETS:
        if asset not in daily_returns.columns:
            continue
        for start_date in WALK_FORWARD_STARTS:
            end_date = pd.Timestamp("2026-06-30")
            for version in VERSIONS:
                for cost_bp in COST_CANDIDATES:
                    logging.info(f"  {asset} ({ASSET_LABELS[asset]}) {start_date.date()} {version} cost={cost_bp}bp ...")
                    res = run_walk_forward_v2(asset, daily_returns, version, start_date, end_date, cost_bp)
                    if res is None:
                        continue
                    results.append({
                        "asset": asset,
                        "asset_name": ASSET_LABELS[asset],
                        "version": version,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        **res,
                    })
                    logging.info(
                        f"    Sharpe={res['sharpe']:.3f}, AnnRet={res['ann_return']*100:.2f}%, "
                        f"mean_pos={res['mean_position']:.3f}, turnover={res['total_turnover']:.2f}"
                    )

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "comparison_v2.csv", index=False)
    logging.info(f"\n详细结果已保存: {OUTPUT_DIR / 'comparison_v2.csv'}")

    # 生成报告
    generate_reports(df_results, OUTPUT_DIR)
    logging.info(f"\n报告已保存到: {OUTPUT_DIR}")


def generate_reports(df: pd.DataFrame, output_dir: Path):
    """生成综合报告."""
    # 1. 跨成本档对比图
    fig, axes = plt.subplots(len(COST_CANDIDATES), 1,
                              figsize=(14, 3.5 * len(COST_CANDIDATES)),
                              sharex=True)
    if len(COST_CANDIDATES) == 1:
        axes = [axes]
    for ci, cost_bp in enumerate(COST_CANDIDATES):
        ax = axes[ci]
        sub_cost = df[df["cost_bp"] == cost_bp]
        if sub_cost.empty:
            continue
        avg_metrics = sub_cost.groupby("version")["sharpe"].mean().sort_values(ascending=False)
        colors = [VERSION_COLORS.get(v, "gray") for v in avg_metrics.index]
        ax.bar(avg_metrics.index, avg_metrics.values, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.set_title(f"成本 = {cost_bp} bp/单边 — 跨资产平均 Sharpe (Walk-Forward OOS)",
                     fontsize=11)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        for i, (v, val) in enumerate(avg_metrics.items()):
            ax.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.suptitle("4 版本跨资产平均 Sharpe (按成本分面, 修正版)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_chart.png", dpi=100, bbox_inches="tight")
    plt.close()

    # 2. summary.md
    with open(output_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# v8 概率化 Jump Model 实验报告 (修正版)\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write("## 关键修正\n\n")
        f.write("基于 Step 1-3 诊断发现:\n")
        f.write("- 之前 walk-forward 实验数据过滤 bug (1400 calendar days → 928 trading days)\n")
        f.write("- 修正后: 用 iloc[:1000] 取训练期\n")
        f.write("- Jump Model 实际能检测到 bear 状态 (不再是 0)\n")
        f.write("- n_restarts=5 已足够, jump_penalty=50 是合理默认\n\n")

        f.write("## 1. 实验设计\n\n")
        f.write("- **测试资产**: 510300 / 511260 / 518880 / 159915 / 512760\n")
        f.write(f"- **Walk-Forward 起点**: {[s.strftime('%Y-%m-%d') for s in WALK_FORWARD_STARTS]}\n")
        f.write(f"- **训练窗口**: {TRAIN_WINDOW} 天, **测试窗口**: {TEST_WINDOW} 天, **步长**: {STEP} 天\n")
        f.write(f"- **测试成本**: {COST_CANDIDATES} bp/单边\n\n")

        f.write("## 2. 各成本档结果\n\n")
        for cost_bp in COST_CANDIDATES:
            sub = df[df["cost_bp"] == cost_bp]
            if sub.empty:
                continue
            f.write(f"### 成本 = {cost_bp} bp\n\n")
            avg = sub.groupby("version")[["ann_return", "vol", "sharpe", "max_drawdown", "calmar"]].mean()
            f.write("| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar | Avg Pos | Avg Turnover |\n")
            f.write("|------|-----------|---------|-----------|-----------|------------|---------|-------------|\n")
            for version in VERSIONS:
                if version not in avg.index:
                    continue
                r = avg.loc[version]
                sub_v = sub[sub["version"] == version]
                avg_pos = sub_v["mean_position"].mean()
                avg_to = sub_v["total_turnover"].mean()
                f.write(
                    f"| {version} | {r['ann_return']*100:.2f}% | {r['vol']*100:.2f}% | "
                    f"**{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                    f"**{r['calmar']:.3f}** | {avg_pos:.3f} | {avg_to:.2f} |\n"
                )
            f.write("\n")

        # 成本敏感性
        f.write("## 3. 成本敏感性\n\n")
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
                avg = sub["sharpe"].mean()
                sharpes[cb] = avg
                row += f"| {avg:.3f} "
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
        f.write("\n## 4. 最终判定\n\n")
        sub_10 = df[df["cost_bp"] == 10]
        if not sub_10.empty:
            avg_10 = sub_10.groupby("version")["sharpe"].mean().sort_values(ascending=False)
            f.write("**10bp 成本下排序**:\n\n")
            for rank, (version, sharpe) in enumerate(avg_10.items(), 1):
                f.write(f"{rank}. **{version}**: Sharpe={sharpe:.3f}\n")
            f.write("\n")
            best = avg_10.index[0]
            best_sharpe = avg_10.iloc[0]
            base_sharpe = avg_10.get("v8_method_b", 0)
            f.write(f"**结论**: {best} 在 10bp 成本下最优 "
                    f"(Sharpe={best_sharpe:.3f}, vs v8_method_b: {best_sharpe-base_sharpe:+.3f})\n")


if __name__ == "__main__":
    main()