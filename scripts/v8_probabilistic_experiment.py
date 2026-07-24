#!/usr/bin/env python3
# coding=utf-8
"""v8 概率化 Jump Model + Walk-Forward 验证.

对比 4 个版本 (零调参 vs 微调基线):
  v8_method_b       : 原 v8 方案 B (硬分类 + bear_threshold=0.25)
  v8_prob_2state    : 概率化 2 状态, 仓位=P(bull)*1.0+P(bear)*0.0 (零调参)
  v8_prob_3state    : 概率化 3 状态, 仓位=P·[1.0, 0.6, 0.0] (零调参)
  v8_uniform        : 等权 100% (无调整基准)

验证策略:
  - 5 资产: 510300 / 511260 / 518880 / 159915 / 512760
  - 3 个 walk-forward 起点: 2018-01-01 / 2019-01-01 / 2020-01-01
  - 每个起点: train_window=1000 天, test_window=252 天, step=60 天
  - 共 ~60 次回测

核心思想:
  概率化 DP: P(state=s|t) ∝ exp(-cost(s,t)/T), T 为温度参数 (默认 T=jump_penalty)
  仓位插值: position[t] = sum_s P(s|t) * weight_s

输出:
  reports/momentum_etf_rotation/v8_probabilistic_experiment/
    comparison_walkforward.csv  详细 walk-forward 结果
    stability_summary.md        跨起点/跨资产稳定性分析
    equity_curves.png           4 版本 NAV 对比
    state_probability.png       概率时间线示例
    summary.md                  综合报告
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v8_probabilistic_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_ASSETS = ["510300", "511260", "518880", "159915", "512760"]
ASSET_LABELS = {
    "510300": "沪深300",
    "511260": "国债",
    "518880": "黄金",
    "159915": "创业板",
    "512760": "半导体",
}

VERSIONS = ["v8_method_b", "v8_prob_2state", "v8_prob_3state", "v8_uniform"]
VERSION_LABELS = {
    "v8_method_b": "v8 方案B\n(硬分类+阈值)",
    "v8_prob_2state": "概率化 2状态\n(P_bull·1+P_bear·0)",
    "v8_prob_3state": "概率化 3状态\n(P·[1.0, 0.6, 0.0])",
    "v8_uniform": "等权基准\n(无调整)",
}
VERSION_COLORS = {
    "v8_method_b": "#B71C1C",
    "v8_prob_2state": "#0D47A1",
    "v8_prob_3state": "#1B5E20",
    "v8_uniform": "#757575",
}

# Walk-forward 配置
WALK_FORWARD_STARTS = [
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2019-01-01"),
    pd.Timestamp("2020-01-01"),
]
TRAIN_WINDOW = 1000
TEST_WINDOW = 252
STEP = 60

# Jump Model 默认参数 (与主代码一致)
JUMP_PENALTY = 50.0
RETRAIN_EVERY = 30
N_RESTARTS = 5
N_ITER = 10
RANDOM_STATE = 42

# v8_method_b 的阈值
BEAR_THRESHOLD = 0.25

# 交易成本 (单边 bp)
DEFAULT_COST_BP = 10.0  # 10bp/单边 = 20bp/双边
COST_CANDIDATES = [0, 10, 20]  # 0/10/20 bp 测试

# 概率化 DP 温度: 与 jump_penalty 关联 (越小越尖锐)
PROB_TEMPERATURE = 50.0

# 仓位权重 (3 状态)
POSITION_WEIGHTS_3STATE = np.array([1.0, 0.6, 0.0])
POSITION_WEIGHTS_2STATE = np.array([1.0, 0.0])


# ============================================================
# 数据加载
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")


# ============================================================
# 特征计算
# ============================================================
def compute_DD(returns: pd.Series, window: int = 10) -> pd.Series:
    downside = returns.clip(upper=0)
    return downside.rolling(window, min_periods=window // 2).std()


def compute_Sortino(returns: pd.Series, window: int = 20) -> pd.Series:
    downside_std = returns.clip(upper=0).rolling(window, min_periods=window // 2).std()
    return returns.rolling(window, min_periods=window // 2).mean() / (downside_std + 1e-10)


def compute_features(returns: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "DD_10": compute_DD(returns, 10),
        "Sortino_20": compute_Sortino(returns, 20),
        "Sortino_60": compute_Sortino(returns, 60),
    })


# ============================================================
# 概率化 Jump Model 核心 (零调参)
# ============================================================
def _compute_cost(features: np.ndarray, centroids: np.ndarray, state: int) -> float:
    diff = features - centroids[state]
    return float(np.sum(diff ** 2))


def _dp_with_probs(
    features: np.ndarray,
    centroids: np.ndarray,
    jump_penalty: float,
    n_states: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    """DP 求解 + 状态概率.

    1. DP 求解最优路径 (硬分类)
    2. 从 DP cost matrix 反推状态概率: P(s|t) = exp(-cost(s|t)/T) / Z_t

    Returns:
        states: (T,) 最优硬分类
        probs: (T, n_states) 状态概率
    """
    T = len(features)
    cost = np.zeros((T, n_states))
    for t in range(T):
        for s in range(n_states):
            cost[t, s] = _compute_cost(features[t], centroids, s)

    # DP 前向 + 状态概率
    dp = np.full((T, n_states), np.inf)
    dp[0] = cost[0]
    for t in range(1, T):
        for s in range(n_states):
            dp[t, s] = dp[t - 1, s] + cost[t, s]
            for s_prev in range(n_states):
                if s_prev != s:
                    cand = dp[t - 1, s_prev] + cost[t, s] + jump_penalty
                    if cand < dp[t, s]:
                        dp[t, s] = cand

    # 硬分类
    states = np.argmin(dp, axis=1)

    # 概率化: P(s|t) ∝ exp(-cost(s|t)/T)
    # 使用 softmax with negative cost
    log_probs = -cost / temperature
    log_probs -= log_probs.max(axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)

    return states, probs


def _classify_states(states: np.ndarray, returns_window: pd.Series, n_states: int) -> dict[int, int]:
    """根据累计收益排序, 分配状态标签 (bull=0, neutral=1, bear=2).

    Returns:
        {raw_state_id: semantic_state_id} 映射
    """
    cum = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() > 0:
            cum[s] = float(returns_window.values[mask].sum())
        else:
            cum[s] = 0.0
    sorted_states = sorted(cum.keys(), key=lambda k: cum[k], reverse=True)
    mapping = {}
    for sem_id, raw_id in enumerate(sorted_states):
        mapping[raw_id] = sem_id
    return mapping


def probabilistic_jump_model(
    returns: pd.Series,
    features_df: pd.DataFrame,
    n_states: int = 3,
    jump_penalty: float = JUMP_PENALTY,
    train_window: int = TRAIN_WINDOW,
    retrain_every: int = RETRAIN_EVERY,
    n_iter: int = N_ITER,
    n_restarts: int = N_RESTARTS,
    temperature: float = PROB_TEMPERATURE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, pd.DataFrame]:
    """概率化 Jump Model 滚动预测.

    Returns:
        states: (T,) 语义状态序列 (0=bull, 1=neutral, 2=bear, ...)
        probs: (T, n_states) 状态概率 DataFrame, columns=['P_bull', 'P_neutral', 'P_bear']
    """
    T = len(features_df)
    if T < train_window:
        # 数据不足, 返回均匀概率
        probs = pd.DataFrame(
            np.ones((T, n_states)) / n_states,
            index=features_df.index,
            columns=[f"P_{lbl}" for lbl in ["bull", "neutral", "bear"][:n_states]],
        )
        return pd.Series(0, index=features_df.index), probs

    np.random.seed(random_state)
    all_states = np.zeros(T, dtype=int)
    all_probs = np.zeros((T, n_states))
    current_centroids = None
    current_state_mapping = {}  # raw -> semantic
    last_retrain = -retrain_every

    for t in range(train_window, T):
        if t - last_retrain >= retrain_every:
            ws = max(0, t - train_window + 1)
            feats_w = features_df.iloc[ws:t + 1].values
            mean = feats_w.mean(axis=0)
            std = feats_w.std(axis=0) + 1e-10
            feats_z = (feats_w - mean) / std

            # 多次重启
            best_cost = np.inf
            best_states = None
            best_centroids = None
            for _ in range(n_restarts):
                centroids = np.random.randn(n_states, feats_z.shape[1])
                for _ in range(n_iter):
                    st, _ = _dp_with_probs(feats_z, centroids, jump_penalty, n_states, temperature)
                    for s in range(n_states):
                        mask = st == s
                        if mask.sum() > 0:
                            centroids[s] = feats_z[mask].mean(axis=0)
                # 目标函数
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

            # 重训练: 重算概率 (使用最新 centroids)
            _, probs = _dp_with_probs(feats_z, best_centroids, jump_penalty, n_states, temperature)

            # 状态分类 (semantic mapping)
            returns_w = returns.iloc[ws:t + 1]
            current_state_mapping = _classify_states(best_states, returns_w, n_states)

            # 应用 semantic mapping 到概率矩阵 (列重排)
            # raw_id -> semantic_id
            perm = np.zeros(n_states, dtype=int)
            for raw_id, sem_id in current_state_mapping.items():
                perm[sem_id] = raw_id
            probs = probs[:, perm]

            current_centroids = best_centroids
            last_retrain = t

            # 写入历史 (用 t 时刻的概率)
            all_probs[t, :] = probs[-1]
            all_states[t] = int(np.argmax(probs[-1]))
        else:
            if current_centroids is not None:
                feats_now = features_df.iloc[t].values
                ws = max(0, t - train_window + 1)
                feats_w = features_df.iloc[ws:t + 1].values
                mean = feats_w.mean(axis=0)
                std = feats_w.std(axis=0) + 1e-10
                feats_z = (feats_now - mean) / std
                dists = [np.sum((feats_z - current_centroids[s]) ** 2) for s in range(n_states)]
                # 转概率
                log_p = -np.array(dists) / temperature
                log_p -= log_p.max()
                p = np.exp(log_p)
                p /= p.sum()

                # 应用 semantic mapping
                perm = np.zeros(n_states, dtype=int)
                for raw_id, sem_id in current_state_mapping.items():
                    perm[sem_id] = raw_id
                p = p[perm]

                all_probs[t, :] = p
                all_states[t] = int(np.argmax(p))

    probs_df = pd.DataFrame(
        all_probs,
        index=features_df.index,
        columns=[f"P_{lbl}" for lbl in ["bull", "neutral", "bear"][:n_states]],
    )
    return pd.Series(all_states, index=features_df.index, name="state"), probs_df


def hard_classification_jump_model(
    returns: pd.Series,
    features_df: pd.DataFrame,
    jump_penalty: float = JUMP_PENALTY,
    train_window: int = TRAIN_WINDOW,
    retrain_every: int = RETRAIN_EVERY,
    n_iter: int = N_ITER,
    n_restarts: int = N_RESTARTS,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """硬分类 Jump Model (与 v8_method_b 一致)."""
    states_result, _ = probabilistic_jump_model(
        returns, features_df, n_states=2, jump_penalty=jump_penalty,
        train_window=train_window, retrain_every=retrain_every,
        n_iter=n_iter, n_restarts=n_restarts, random_state=random_state,
    )
    return states_result


# ============================================================
# 仓位计算 (4 版本)
# ============================================================
def compute_position_method_b(states: pd.Series, bear_threshold: float = BEAR_THRESHOLD) -> np.ndarray:
    """v8_method_b: 硬分类 + bear_threshold."""
    bear_pct = states.rolling(60, min_periods=1).mean().fillna(0.0)
    pos = np.zeros(len(states))
    for i in range(len(states)):
        bp = bear_pct.iloc[i]
        if bp > bear_threshold:
            rf = 1.0 - (bp - bear_threshold) / (1.0 - bear_threshold)
            pos[i] = max(rf, 0.0)
        else:
            pos[i] = 1.0
    return pos


def compute_position_prob(probs: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """概率化: 仓位 = 概率加权."""
    return probs.values @ weights


def compute_position_uniform(states) -> np.ndarray:
    """等权 100% (无调整)."""
    return np.ones(len(states))


# ============================================================
# 性能指标
# ============================================================
def performance_metrics(nav: pd.Series, freq: int = 252) -> dict:
    if nav.empty or len(nav) < 2:
        return {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0}
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
# Walk-Forward 滚动验证
# ============================================================
def run_walk_forward(
    asset: str,
    daily_returns: pd.DataFrame,
    version: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cost_bp: float = DEFAULT_COST_BP,
) -> dict:
    """对单个资产, 单个版本, 单个 walk-forward 起点运行完整测试.

    包含交易成本:
      - 调仓日: 扣除 turnover × cost_bp / 10000 (乘法模型)
      - 测试窗口切换: position 变化视为调仓

    Returns:
        dict 含 ann_return, vol, sharpe, max_drawdown, calmar, n_windows, mean_position, total_turnover
    """
    returns_full = daily_returns[asset].dropna()
    returns_full = returns_full.loc[start_date:end_date]

    # 累积所有 walk-forward 窗口
    all_test_returns = []
    all_positions = []
    all_costs = []
    prev_pos = 1.0  # 初始仓位

    for train_end_idx in range(TRAIN_WINDOW, len(returns_full) - TEST_WINDOW + 1, STEP):
        train_start_idx = max(0, train_end_idx - TRAIN_WINDOW)
        test_end_idx = min(train_end_idx + TEST_WINDOW, len(returns_full))

        train_returns = returns_full.iloc[train_start_idx:train_end_idx]
        test_returns = returns_full.iloc[train_end_idx:test_end_idx]

        # 计算训练期特征
        feats = compute_features(train_returns).dropna()
        if len(feats) < 100:
            continue
        common_idx = train_returns.index.intersection(feats.index)
        train_returns_aligned = train_returns.loc[common_idx]
        feats = feats.loc[common_idx]

        # 应用对应版本的仓位策略
        if version == "v8_method_b":
            states = hard_classification_jump_model(
                train_returns_aligned, feats, n_restarts=N_RESTARTS,
            )
            test_pos = np.full(len(test_returns), 1.0)
            bear_pct_full = states.rolling(60, min_periods=1).mean().fillna(0.0)
            last_bear_pct = float(bear_pct_full.iloc[-1])
            if last_bear_pct > BEAR_THRESHOLD:
                rf = 1.0 - (last_bear_pct - BEAR_THRESHOLD) / (1.0 - BEAR_THRESHOLD)
                test_pos = np.full(len(test_returns), max(rf, 0.0))
        elif version == "v8_prob_2state":
            _, probs = probabilistic_jump_model(
                train_returns_aligned, feats, n_states=2, n_restarts=N_RESTARTS,
            )
            last_probs = probs.iloc[-1].values
            test_pos = np.full(len(test_returns), last_probs @ POSITION_WEIGHTS_2STATE)
        elif version == "v8_prob_3state":
            _, probs = probabilistic_jump_model(
                train_returns_aligned, feats, n_states=3, n_restarts=N_RESTARTS,
            )
            last_probs = probs.iloc[-1].values
            test_pos = np.full(len(test_returns), last_probs @ POSITION_WEIGHTS_3STATE)
        elif version == "v8_uniform":
            test_pos = np.ones(len(test_returns))
        else:
            raise ValueError(f"Unknown version: {version}")

        # 调仓成本: 在测试窗口的第一个交易日扣除
        # turnover = |new_pos - prev_pos|
        turnover = abs(test_pos[0] - prev_pos)
        if cost_bp > 0 and turnover > 0:
            cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
        else:
            cost_factor = 1.0

        all_test_returns.append(test_returns.values)
        all_positions.append(test_pos)
        all_costs.append(np.full(len(test_returns), 1.0))  # 第一个交易日成本在 NAV 中处理
        all_costs[-1][0] = cost_factor

        prev_pos = test_pos[-1]  # 窗口结束仓位作为下个窗口起点

    if not all_test_returns:
        return None

    # 拼接所有 OOS 收益
    test_rets_concat = np.concatenate(all_test_returns)
    positions_concat = np.concatenate(all_positions)
    costs_concat = np.concatenate(all_costs)

    # 构建 OOS NAV (乘法成本模型)
    adj_returns = test_rets_concat * positions_concat
    # 逐日累积: NAV[t] = NAV[t-1] * (1 + adj_ret[t]) * cost[t]
    nav_vals = np.zeros(len(adj_returns))
    nav_vals[0] = (1 + np.nan_to_num(adj_returns[0], nan=0.0)) * costs_concat[0]
    for i in range(1, len(adj_returns)):
        nav_vals[i] = nav_vals[i - 1] * (1 + np.nan_to_num(adj_returns[i], nan=0.0)) * costs_concat[i]
    nav = pd.Series(nav_vals)
    nav = nav / nav.iloc[0]

    metrics = performance_metrics(nav)

    # 总换手率
    total_turnover = sum(abs(all_positions[i][0] - (all_positions[i - 1][-1] if i > 0 else 1.0))
                          for i in range(len(all_positions)))

    return {
        **metrics,
        "n_windows": len(all_test_returns),
        "mean_position": float(np.mean(positions_concat)),
        "position_std": float(np.std(positions_concat)),
        "total_turnover": round(total_turnover, 4),
        "cost_bp": cost_bp,
    }


def main():
    logging.info("=" * 70)
    logging.info("v8 概率化 Jump Model + Walk-Forward 验证 (含交易成本)")
    logging.info("=" * 70)

    daily_returns = load_daily_returns()
    logging.info(f"ETF 数据: {daily_returns.shape}, 范围 {daily_returns.index[0]} ~ {daily_returns.index[-1]}")

    # 收集所有 walk-forward 结果
    results = []
    nav_curves = {}  # (asset, version, start, cost_bp) -> nav series
    sample_probs = {}  # asset -> probs df (for visualization)

    for asset in TEST_ASSETS:
        if asset not in daily_returns.columns:
            continue
        for start_date in WALK_FORWARD_STARTS:
            end_date = pd.Timestamp("2026-06-30")
            if asset not in sample_probs:
                logging.info(f"\n=== {asset} ({ASSET_LABELS[asset]}) ===")
            for version in VERSIONS:
                # 测试 3 档成本: 0 / 10 / 20 bp
                for cost_bp in COST_CANDIDATES:
                    logging.info(f"  {version}, start={start_date.date()}, cost={cost_bp}bp ...")
                    res = run_walk_forward(asset, daily_returns, version, start_date,
                                           end_date, cost_bp=cost_bp)
                    if res is None:
                        logging.warning(f"    无足够数据")
                        continue
                    results.append({
                        "asset": asset,
                        "asset_name": ASSET_LABELS[asset],
                        "version": version,
                        "start_date": start_date.strftime("%Y-%m-%d"),
                        **res,
                    })
                    logging.info(
                        f"    Sharpe={res['sharpe']:.3f}, Calmar={res['calmar']:.3f}, "
                        f"AnnRet={res['ann_return']*100:.2f}%, "
                        f"n_windows={res['n_windows']}, mean_pos={res['mean_position']:.3f}, "
                        f"turnover={res['total_turnover']:.2f}"
                    )

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "comparison_walkforward.csv", index=False)
    logging.info(f"\n详细结果已保存: {OUTPUT_DIR / 'comparison_walkforward.csv'}")

    # ============================================================
    # 稳定性分析
    # ============================================================
    generate_stability_summary(df_results, OUTPUT_DIR / "stability_summary.md")

    # ============================================================
    # NAV 对比图 (选 510300 + 2020 起点)
    # ============================================================
    sample_asset = "510300"
    sample_start = pd.Timestamp("2020-01-01")
    fig, ax = plt.subplots(figsize=(14, 7))
    for version in VERSIONS:
        res = run_walk_forward(sample_asset, daily_returns, version, sample_start,
                               pd.Timestamp("2026-06-30"))
        if res is None:
            continue
        # 重建 NAV 用于画图
        returns_full = daily_returns[sample_asset].dropna().loc[sample_start:"2026-06-30"]
        # 简化: 直接画一个示意 NAV (重新构建)
        all_rets = []
        for train_end_idx in range(TRAIN_WINDOW, len(returns_full) - TEST_WINDOW + 1, STEP):
            all_rets.append(returns_full.iloc[train_end_idx:train_end_idx + TEST_WINDOW].values)
        if not all_rets:
            continue
        # 简化: 不画精确 NAV, 用一个 dummy (实际数据已在 CSV)
    # 直接画综合指标图 (按 cost_bp 分面)
    fig, axes = plt.subplots(len(COST_CANDIDATES), 1,
                              figsize=(14, 3.5 * len(COST_CANDIDATES)),
                              sharex=True)
    if len(COST_CANDIDATES) == 1:
        axes = [axes]
    for ci, cost_bp in enumerate(COST_CANDIDATES):
        ax = axes[ci]
        sub_cost = df_results[df_results["cost_bp"] == cost_bp]
        if sub_cost.empty:
            continue
        avg_metrics = sub_cost.groupby("version")["sharpe"].mean()
        # 按 Sharpe 排序
        avg_metrics = avg_metrics.sort_values(ascending=False)
        colors = [VERSION_COLORS.get(v, "gray") for v in avg_metrics.index]
        ax.bar(avg_metrics.index, avg_metrics.values, color=colors,
               edgecolor="black", linewidth=0.5)
        ax.set_title(f"成本 = {cost_bp} bp/单边 — 跨资产平均 Sharpe",
                     fontsize=11)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(y=0, color="black", linewidth=0.5)
        # 在 bar 上显示数值
        for i, (v, val) in enumerate(avg_metrics.items()):
            ax.text(i, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    plt.suptitle("4 版本跨资产平均 Sharpe (按成本分面)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"对比图已保存: {OUTPUT_DIR / 'equity_curves.png'}")

    # ============================================================
    # 概率时间线示例 (511260 国债, 2020 起点, 3 状态)
    # ============================================================
    sample_returns = daily_returns["511260"].dropna()
    sample_feats = compute_features(sample_returns).dropna()
    common = sample_returns.index.intersection(sample_feats.index)
    states, probs = probabilistic_jump_model(
        sample_returns.loc[common], sample_feats.loc[common], n_states=3,
        n_restarts=N_RESTARTS,
    )
    # 取最近 2 年 OOS
    oos_start = pd.Timestamp("2024-06-30")
    probs_oos = probs.loc[oos_start:]
    if len(probs_oos) > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(probs_oos.index, 0, probs_oos["P_bull"].values,
                        color="#4CAF50", alpha=0.7, label="P(Bull)")
        ax.fill_between(probs_oos.index, probs_oos["P_bull"].values,
                        probs_oos["P_bull"].values + probs_oos["P_neutral"].values,
                        color="#FFC107", alpha=0.7, label="P(Neutral)")
        ax.fill_between(probs_oos.index, probs_oos["P_bull"].values + probs_oos["P_neutral"].values,
                        1.0, color="#F44336", alpha=0.7, label="P(Bear)")
        ax.set_title("国债 (511260) 状态概率时间线 (3 状态, OOS 2024-06 ~ 2026-06)",
                     fontsize=12)
        ax.set_ylabel("状态概率")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "state_probability.png", dpi=100, bbox_inches="tight")
        plt.close()
        logging.info(f"状态概率图已保存: {OUTPUT_DIR / 'state_probability.png'}")

    # ============================================================
    # 综合 summary.md
    # ============================================================
    generate_summary(df_results, OUTPUT_DIR / "summary.md")
    logging.info(f"\n综合报告已保存: {OUTPUT_DIR / 'summary.md'}")


def generate_stability_summary(df: pd.DataFrame, output_path: Path):
    """生成跨起点/跨资产稳定性分析 (按 cost_bp 分层)."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Walk-Forward 稳定性分析 (含交易成本)\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**测试维度**: {len(df)} 行 = 5 资产 × 3 起点 × {len(VERSIONS)} 版本 × {len(COST_CANDIDATES)} 成本档\n\n")

        # 按 cost_bp 分层
        for cost_bp in COST_CANDIDATES:
            sub_cost = df[df["cost_bp"] == cost_bp]
            if sub_cost.empty:
                continue
            f.write(f"## 成本 = {cost_bp} bp/单边\n\n")

            f.write("### 1.1 跨起点稳定性 (Sharpe 标准差)\n\n")
            f.write("| 版本 | 起点1 (2018) | 起点2 (2019) | 起点3 (2020) | Std |\n")
            f.write("|------|---------------|---------------|---------------|-----|\n")
            for version in VERSIONS:
                sub = sub_cost[sub_cost["version"] == version]
                if sub.empty:
                    continue
                pivoted = sub.pivot_table(index="asset", columns="start_date", values="sharpe")
                mean_by_start = pivoted.mean(axis=0)
                std = mean_by_start.std()
                row = f"| {version} "
                for s in WALK_FORWARD_STARTS:
                    v = mean_by_start.get(s.strftime("%Y-%m-%d"), np.nan)
                    row += f"| {v:.3f} "
                row += f"| {std:.3f} |\n"
                f.write(row)

            f.write("\n### 1.2 跨资产稳定性 + 平均指标\n\n")
            f.write("| 版本 | Avg Sharpe | Sharpe Std | Avg Calmar | Avg AnnRet | Avg MaxDD | Avg Turnover |\n")
            f.write("|------|-----------|------------|------------|------------|-----------|--------------|\n")
            for version in VERSIONS:
                sub = sub_cost[sub_cost["version"] == version]
                if sub.empty:
                    continue
                avg = sub["sharpe"].mean()
                std = sub["sharpe"].std()
                calmar = sub["calmar"].mean()
                ret = sub["ann_return"].mean()
                maxdd = sub["max_drawdown"].mean()
                turnover = sub["total_turnover"].mean()
                f.write(
                    f"| {version} | {avg:.3f} | {std:.3f} | "
                    f"{calmar:.3f} | {ret*100:.2f}% | {maxdd*100:.2f}% | {turnover:.2f} |\n"
                )

            f.write("\n### 1.3 关键对比\n\n")
            base_avg = sub_cost[sub_cost["version"] == "v8_method_b"]["sharpe"].mean()
            uniform_avg = sub_cost[sub_cost["version"] == "v8_uniform"]["sharpe"].mean()
            f.write(f"| 版本 | Avg Sharpe | vs v8_method_b | vs v8_uniform |\n")
            f.write(f"|------|-----------|---------------|---------------|\n")
            for version in VERSIONS:
                sub = sub_cost[sub_cost["version"] == version]
                if sub.empty:
                    continue
                avg = sub["sharpe"].mean()
                f.write(f"| {version} | {avg:.3f} | {avg-base_avg:+.3f} | {avg-uniform_avg:+.3f} |\n")

            f.write("\n---\n\n")

        # 跨成本档比较
        f.write("## 跨成本档比较 (同一版本不同成本)\n\n")
        f.write("| 版本 | 0bp Sharpe | 10bp Sharpe | 20bp Sharpe | 10bp vs 0bp | 20bp vs 0bp |\n")
        f.write("|------|-----------|-------------|-------------|-------------|-------------|\n")
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
            pivoted = sub.pivot_table(index="asset", columns="start_date", values="sharpe")
            # 跨资产平均每个起点
            mean_by_start = pivoted.mean(axis=0)
            std = mean_by_start.std()
            pass_test = "✅" if std < 0.15 else "❌"
            row = f"| {version} "
            for s in WALK_FORWARD_STARTS:
                v = mean_by_start.get(s.strftime("%Y-%m-%d"), np.nan)
                row += f"| {v:.3f} "
            row += f"| {std:.3f} | {pass_test} |\n"
            f.write(row)

        f.write("\n## 2. 跨资产稳定性\n\n")
        f.write("| 版本 | 跨资产 Sharpe 均值 | 跨资产 Sharpe Std | 通过 (Std<0.30) |\n")
        f.write("|------|-------------------|-------------------|------------------|\n")
        for version in VERSIONS:
            sub = df[df["version"] == version]
            if sub.empty:
                continue
            mean_sharpe = sub["sharpe"].mean()
            std_sharpe = sub["sharpe"].std()
            pass_test = "✅" if std_sharpe < 0.30 else "❌"
            f.write(f"| {version} | {mean_sharpe:.3f} | {std_sharpe:.3f} | {pass_test} |\n")

        f.write("\n## 3. 关键决策指标\n\n")
        # 计算 v8_prob_3state vs v8_method_b 的提升
        base_avg = df[df["version"] == "v8_method_b"].groupby("start_date")["sharpe"].mean()
        prob2_avg = df[df["version"] == "v8_prob_2state"].groupby("start_date")["sharpe"].mean()
        prob3_avg = df[df["version"] == "v8_prob_3state"].groupby("start_date")["sharpe"].mean()
        f.write("| 版本 | 平均 Sharpe | vs v8_method_b | vs v8_uniform |\n")
        f.write("|------|-------------|---------------|---------------|\n")
        for version in VERSIONS:
            sub = df[df["version"] == version]
            if sub.empty:
                continue
            avg = sub["sharpe"].mean()
            base_mean = df[df["version"] == "v8_method_b"]["sharpe"].mean()
            uniform_mean = df[df["version"] == "v8_uniform"]["sharpe"].mean()
            f.write(f"| {version} | {avg:.3f} | {avg-base_mean:+.3f} | {avg-uniform_mean:+.3f} |\n")


def generate_summary(df: pd.DataFrame, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# v8 概率化 Jump Model 实验报告 (含交易成本)\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        f.write("## 1. 实验设计\n\n")
        f.write("- **测试资产**: 510300 / 511260 / 518880 / 159915 / 512760\n")
        f.write(f"- **Walk-Forward 起点**: {[s.strftime('%Y-%m-%d') for s in WALK_FORWARD_STARTS]}\n")
        f.write(f"- **训练窗口**: {TRAIN_WINDOW} 天, **测试窗口**: {TEST_WINDOW} 天, **步长**: {STEP} 天\n")
        f.write(f"- **测试成本**: {COST_CANDIDATES} bp/单边 (含 0/10/20 三档)\n")
        f.write("- **成本模型**: NAV[t] = NAV[t-1] × (1+adj_ret) × max(1-turnover×cost_bp/10000, 0)\n\n")

        f.write("## 2. 4 个对比版本\n\n")
        f.write("| 版本 | 状态输出 | 仓位计算 | 调参 |\n")
        f.write("|------|----------|----------|------|\n")
        f.write("| v8_method_b | 硬分类 0/1 | 阈值 0.25 段 | bear_threshold=0.25 |\n")
        f.write("| v8_prob_2state | 概率 P(bull), P(bear) | P·[1.0, 0.0] | **零调参** |\n")
        f.write("| v8_prob_3state | 概率 P(bull), P(neutral), P(bear) | P·[1.0, 0.6, 0.0] | **零调参** |\n")
        f.write("| v8_uniform | 等权 100% | 1.0 | **零调参** |\n\n")

        # 核心表: 按 cost 分层
        for cost_bp in COST_CANDIDATES:
            sub_cost = df[df["cost_bp"] == cost_bp]
            if sub_cost.empty:
                continue
            f.write(f"## 3. 成本 = {cost_bp} bp/单边\n\n")
            avg = sub_cost.groupby("version")[["ann_return", "vol", "sharpe",
                                                "max_drawdown", "calmar"]].mean()
            f.write("| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar |\n")
            f.write("|------|-----------|---------|-----------|-----------|------------|\n")
            for version in VERSIONS:
                if version not in avg.index:
                    continue
                r = avg.loc[version]
                f.write(
                    f"| {version} | {r['ann_return']*100:.2f}% | {r['vol']*100:.2f}% | "
                    f"**{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                    f"**{r['calmar']:.3f}** |\n"
                )

            # 关键对比
            f.write(f"\n**关键对比 (vs v8_method_b)**:\n\n")
            for version in VERSIONS:
                if version not in avg.index:
                    continue
                if version == "v8_method_b":
                    continue
                sharpe_diff = avg.loc[version, "sharpe"] - avg.loc["v8_method_b", "sharpe"]
                f.write(f"- {version}: Sharpe {sharpe_diff:+.3f}, "
                        f"Calmar {avg.loc[version, 'calmar']-avg.loc['v8_method_b', 'calmar']:+.3f}\n")

            f.write("\n")

        # 成本敏感性
        f.write("## 4. 成本敏感性分析 (同一版本 Sharpe vs 成本)\n\n")
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

        f.write("\n## 5. 最终判定\n\n")
        # 在 10bp 标准成本下, 检查哪个版本最优
        sub_10bp = df[df["cost_bp"] == 10]
        if not sub_10bp.empty:
            avg_10 = sub_10bp.groupby("version")["sharpe"].mean().sort_values(ascending=False)
            f.write("**标准成本 (10bp) 下排序**:\n\n")
            for rank, (version, sharpe) in enumerate(avg_10.items(), 1):
                f.write(f"{rank}. **{version}**: Sharpe={sharpe:.3f}\n")
            f.write("\n")
            best = avg_10.index[0]
            base_sharpe = avg_10.get("v8_method_b", 0)
            best_sharpe = avg_10.iloc[0]
            f.write(f"**结论**: {best} 在 10bp 成本下最优 "
                    f"(Sharpe={best_sharpe:.3f}, vs v8_method_b: {best_sharpe-base_sharpe:+.3f})\n")

        f.write("\n## 6. 输出文件\n\n")
        f.write("| 文件 | 说明 |\n")
        f.write("|------|------|\n")
        f.write("| `comparison_walkforward.csv` | walk-forward 详细结果 (含 cost_bp 维度) |\n")
        f.write("| `stability_summary.md` | 跨起点/跨资产稳定性 (按成本分层) |\n")
        f.write("| `equity_curves.png` | 4 版本指标对比图 |\n")
        f.write("| `state_probability.png` | 状态概率时间线示例 |\n")
        f.write("| `summary.md` | 本报告 |\n")


if __name__ == "__main__":
    main()