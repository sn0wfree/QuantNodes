#!/usr/bin/env python3
# coding=utf-8
"""v8 Jump Model 优化实验 — 3状态 + 宏观特征.

4 个对比版本:
  v8_base          : 2 状态 + 3 维内生特征 (DD_10, Sortino_20, Sortino_60)
  v8_3state        : 3 状态 + 3 维内生特征
  v8_3state_macro  : 3 状态 + 8 维特征 (3 内生 + 5 宏观)
  v8_2state_macro  : 2 状态 + 8 维特征 (3 内生 + 5 宏观)

5 个测试资产: 510300 / 511260 / 518880 / 159915 / 512760

输出:
  reports/momentum_etf_rotation/v8_3state_experiment/comparison.csv
  reports/momentum_etf_rotation/v8_3state_experiment/summary.md
  reports/momentum_etf_rotation/v8_3state_experiment/state_distribution.png
  reports/momentum_etf_rotation/v8_3state_experiment/equity_curves.png
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

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v8_3state_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OOS_START = pd.Timestamp("2022-02-17")
OOS_END = pd.Timestamp("2026-06-30")

TEST_ASSETS = ["510300", "511260", "518880", "159915", "512760"]
ASSET_LABELS = {
    "510300": "沪深300",
    "511260": "国债",
    "518880": "黄金",
    "159915": "创业板",
    "512760": "半导体",
}

VERSIONS = ["v8_base", "v8_3state", "v8_3state_macro", "v8_2state_macro"]
VERSION_LABELS = {
    "v8_base": "2-State\n(3 features)",
    "v8_3state": "3-State\n(3 features)",
    "v8_3state_macro": "3-State\n(+5 macro)",
    "v8_2state_macro": "2-State\n(+5 macro)",
}
VERSION_COLORS = {
    "v8_base": "#B71C1C",
    "v8_3state": "#0D47A1",
    "v8_3state_macro": "#1B5E20",
    "v8_2state_macro": "#E65100",
}

JUMP_PENALTY = 50.0
TRAIN_WINDOW = 1000
RETRAIN_EVERY = 30
N_RESTARTS = 10
N_ITER = 10
RANDOM_STATE = 42


# ============================================================
# 数据加载
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")


def load_macro_panel() -> pd.DataFrame:
    """加载 5 个宏观指标, 合并为日频 panel.

    包含: VIX, DXY, real_rate, cn_us_spread, gold_oil_corr
    """
    frames = []
    # VIX
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "macro_vix_daily.parquet")
    df = df.rename(columns={"vix": "VIX"})
    frames.append(df)
    # DXY
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "macro_dxy_daily_v2.parquet")
    df = df.rename(columns={"dxy": "DXY"})
    frames.append(df)
    # real_rate
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "macro_real_rate_daily.parquet")
    df = df.rename(columns={"real_rate": "REAL_RATE"})
    frames.append(df)
    # cn_us_spread
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "cn_us_spread_10y.parquet")
    df = df.set_index("date")[["cn_us_spread"]].rename(columns={"cn_us_spread": "CN_US_SPREAD"})
    frames.append(df)
    # gold_oil_corr
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "gold_oil_corr.parquet")
    df = df.set_index("date")[["gold_oil_corr"]].rename(columns={"gold_oil_corr": "GOLD_OIL_CORR"})
    frames.append(df)

    panel = frames[0]
    for f in frames[1:]:
        panel = panel.join(f, how="outer")
    panel = panel.sort_index()
    panel.index.name = "date"
    return panel


# ============================================================
# 特征计算
# ============================================================
def compute_DD(returns: pd.Series, window: int = 10) -> pd.Series:
    downside = returns.clip(upper=0)
    return downside.rolling(window, min_periods=window // 2).std()


def compute_Sortino(returns: pd.Series, window: int = 20) -> pd.Series:
    downside_std = returns.clip(upper=0).rolling(window, min_periods=window // 2).std()
    return returns.rolling(window, min_periods=window // 2).mean() / (downside_std + 1e-10)


def compute_macro_features(macro_panel: pd.DataFrame, reindex_to=None) -> pd.DataFrame:
    """5 维宏观特征 (VIX + DXY + real_rate + cn_us_spread + gold_oil_corr).

    每个特征先做 60 日滚动 z-score, 然后重索引到目标日期.
    """
    feats = pd.DataFrame(index=macro_panel.index)
    # VIX level + change
    feats["VIX"] = macro_panel["VIX"]
    feats["DXY"] = macro_panel["DXY"]
    feats["REAL_RATE"] = macro_panel["REAL_RATE"]
    feats["CN_US_SPREAD"] = macro_panel["CN_US_SPREAD"]
    feats["GOLD_OIL_CORR"] = macro_panel["GOLD_OIL_CORR"]
    # 滚动 z-score (60 日)
    feats_z = (feats - feats.rolling(60, min_periods=30).mean()) / (feats.rolling(60, min_periods=30).std() + 1e-10)
    if reindex_to is not None:
        feats_z = feats_z.reindex(reindex_to).ffill().bfill()
    return feats_z


def compute_features_base(returns: pd.Series) -> pd.DataFrame:
    """3 维内生特征: DD_10, Sortino_20, Sortino_60."""
    return pd.DataFrame({
        "DD_10": compute_DD(returns, 10),
        "Sortino_20": compute_Sortino(returns, 20),
        "Sortino_60": compute_Sortino(returns, 60),
    })


def compute_features_extended(returns: pd.Series, macro_panel: pd.DataFrame) -> pd.DataFrame:
    """8 维特征: 3 内生 + 5 宏观."""
    base = compute_features_base(returns)
    macro = compute_macro_features(macro_panel, reindex_to=returns.index)
    return base.join(macro, how="inner").dropna()


# ============================================================
# 通用 Jump Model 核心 (n_states 可配, 不修改主代码)
# ============================================================
def _compute_cost(features: np.ndarray, centroids: np.ndarray, state: int) -> float:
    diff = features - centroids[state]
    return float(np.sum(diff ** 2))


def _dp_insample(features: np.ndarray, centroids: np.ndarray, jump_penalty: float, n_states: int) -> np.ndarray:
    """DP 求解完整状态序列 (有未来函数, 仅在训练时内部用, 分类后只取 last_state)."""
    T = len(features)
    cost = np.zeros((T, n_states))
    for t in range(T):
        for s in range(n_states):
            cost[t, s] = _compute_cost(features[t], centroids, s)

    dp = np.full((T, n_states), np.inf)
    back = np.zeros((T, n_states), dtype=int)
    dp[0] = cost[0]

    for t in range(1, T):
        for s in range(n_states):
            dp[t, s] = dp[t - 1, s] + cost[t, s]
            back[t, s] = s
            for s_prev in range(n_states):
                if s_prev != s:
                    cand = dp[t - 1, s_prev] + cost[t, s] + jump_penalty
                    if cand < dp[t, s]:
                        dp[t, s] = cand
                        back[t, s] = s_prev

    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmin(dp[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = back[t + 1, states[t + 1]]
    return states


def _classify_n_states(states: np.ndarray, returns_window: pd.Series, n_states: int) -> dict[int, str]:
    """根据各状态累计收益排序, 映射为 Bull/Neutral/Bear.

    返回: {state_id: 'bull'/'neutral'/'bear'/...} 字典.
    对于 2 状态: 0=bull, 1=bear.
    对于 3 状态: 累计收益最高=bull, 中间=neutral, 最低=bear.
    """
    cum = {}
    for s in range(n_states):
        mask = states == s
        if mask.sum() > 0:
            cum[s] = float(returns_window.values[mask].sum())
        else:
            cum[s] = 0.0
    sorted_states = sorted(cum.keys(), key=lambda k: cum[k], reverse=True)
    if n_states == 2:
        return {sorted_states[0]: "bull", sorted_states[1]: "bear"}
    elif n_states == 3:
        return {sorted_states[0]: "bull", sorted_states[1]: "neutral", sorted_states[2]: "bear"}
    else:
        labels = ["bull", "neutral"] + [f"bear{i}" for i in range(n_states - 2)]
        return {s: labels[i] for i, s in enumerate(sorted_states)}


def _state_to_position(states: np.ndarray, state_labels: dict[int, str]) -> np.ndarray:
    """状态 → 仓位比例.

    2 状态: bull=1.0, bear=0.0
    3 状态: bull=1.0, neutral=0.6, bear=0.0
    """
    pos = np.zeros(len(states))
    for s, label in state_labels.items():
        if label == "bull":
            pos[states == s] = 1.0
        elif label == "neutral":
            pos[states == s] = 0.6
        elif label == "bear":
            pos[states == s] = 0.0
        else:
            pos[states == s] = 0.2
    return pos


def jump_model_experiment(
    returns: pd.Series,
    features_df: pd.DataFrame,
    n_states: int = 2,
    jump_penalty: float = JUMP_PENALTY,
    train_window: int = TRAIN_WINDOW,
    retrain_every: int = RETRAIN_EVERY,
    n_iter: int = N_ITER,
    n_restarts: int = N_RESTARTS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.Series, dict[int, str]]:
    """周期重估 Jump Model (无未来函数), 返回状态序列 + 状态标签.

    返回:
        all_states: pd.Series, 0/1/2 (n_states=3) 或 0/1 (n_states=2)
        state_labels: {state_id: 'bull'/'neutral'/'bear'/...}
    """
    T = len(features_df)
    if T < train_window:
        return pd.Series(np.zeros(T, dtype=int), index=features_df.index), {}

    np.random.seed(random_state)
    all_states = np.zeros(T, dtype=int)
    current_centroids = None
    state_labels = {}
    last_retrain = -retrain_every

    for t in range(train_window, T):
        if t - last_retrain >= retrain_every:
            ws = max(0, t - train_window + 1)
            feats_w = features_df.iloc[ws:t + 1].values
            mean = feats_w.mean(axis=0)
            std = feats_w.std(axis=0) + 1e-10
            feats_z = (feats_w - mean) / std

            best_cost = np.inf
            best_states = None
            best_centroids = None
            for _ in range(n_restarts):
                centroids = np.random.randn(n_states, feats_z.shape[1])
                for _ in range(n_iter):
                    st = _dp_insample(feats_z, centroids, jump_penalty, n_states)
                    for s in range(n_states):
                        mask = st == s
                        if mask.sum() > 0:
                            centroids[s] = feats_z[mask].mean(axis=0)
                # 目标函数值
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

            returns_w = returns.iloc[ws:t + 1]
            state_labels = _classify_n_states(best_states, returns_w, n_states)
            current_centroids = best_centroids
            last_retrain = t
            # 当前时刻的状态
            cur = best_states[-1]
            all_states[t] = cur
        else:
            if current_centroids is not None:
                feats_now = features_df.iloc[t].values
                ws = max(0, t - train_window + 1)
                feats_w = features_df.iloc[ws:t + 1].values
                mean = feats_w.mean(axis=0)
                std = feats_w.std(axis=0) + 1e-10
                feats_z = (feats_now - mean) / std
                dists = [np.sum((feats_z - current_centroids[s]) ** 2) for s in range(n_states)]
                all_states[t] = int(np.argmin(dists))

    return pd.Series(all_states, index=features_df.index, name="regime"), state_labels


# ============================================================
# 性能指标 (与主代码一致)
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
# 主实验
# ============================================================
def run_single(asset: str, daily_returns: pd.DataFrame, macro_panel: pd.DataFrame, version: str) -> dict:
    """对单个资产跑一个版本, 返回指标 + 状态分布."""
    returns = daily_returns[asset].dropna()
    # 准备特征
    if "macro" in version:
        feats = compute_features_extended(returns, macro_panel)
    else:
        feats = compute_features_base(returns).dropna()
    common_idx = returns.index.intersection(feats.index)
    returns = returns.loc[common_idx]
    feats = feats.loc[common_idx]

    # 确定 n_states
    n_states = 3 if "3state" in version else 2

    # 训练 Jump Model
    states, state_labels = jump_model_experiment(
        returns, feats, n_states=n_states, jump_penalty=JUMP_PENALTY,
        train_window=TRAIN_WINDOW, retrain_every=RETRAIN_EVERY,
        n_restarts=N_RESTARTS,
    )

    # 状态 → 仓位
    if n_states == 2:
        pos = _state_to_position(states.values, state_labels)
        # 应用方案 B 阈值 (基于 Bear%)
        bear_pct = pd.Series(states.values, index=states.index).rolling(60, min_periods=1).mean()
        # 阈值 0.25
        adjusted = pos.copy()
        for i in range(len(adjusted)):
            bp = bear_pct.iloc[i]
            if bp > 0.25:
                rf = 1.0 - (bp - 0.25) / 0.75
                adjusted[i] *= max(rf, 0.0)
    else:
        pos = _state_to_position(states.values, state_labels)
        adjusted = pos

    pos_series = pd.Series(adjusted, index=states.index).fillna(1.0)
    pos_series = pos_series.reindex(returns.index).ffill().fillna(1.0)

    # 构造 NAV
    adj_returns = returns * pos_series
    nav = (1 + adj_returns.fillna(0.0)).cumprod()
    nav = nav / nav.iloc[0]

    # 性能指标 (OOS)
    nav_oos = nav.loc[OOS_START:OOS_END]
    metrics = performance_metrics(nav_oos)

    # 状态分布 (OOS)
    states_oos = states.loc[OOS_START:OOS_END]
    state_dist = {}
    for s, label in state_labels.items():
        cnt = int((states_oos == s).sum())
        state_dist[label] = round(cnt / len(states_oos), 4) if len(states_oos) > 0 else 0.0
    bear_pct_oos = float((states_oos == max(state_labels.keys(), key=lambda k: state_labels[k].startswith("bear") if state_labels.get(k) else False)).sum() / len(states_oos)) if state_labels else 0.0
    if n_states == 2:
        # bear% = rolling 60 日均值
        bear_pct_60 = states_oos.rolling(60, min_periods=1).mean()
        mean_bear_pct = round(float(bear_pct_60.mean()), 4)
    else:
        # bear% = bear 状态占比 (瞬时)
        bear_state_ids = [s for s, lbl in state_labels.items() if lbl == "bear"]
        bear_inst = states_oos.isin(bear_state_ids).astype(float)
        bear_pct_60 = bear_inst.rolling(60, min_periods=1).mean()
        mean_bear_pct = round(float(bear_pct_60.mean()), 4)

    return {
        **metrics,
        "state_dist": state_dist,
        "mean_bear_pct": mean_bear_pct,
        "nav_oos": nav_oos,
    }


def main():
    logging.info("加载数据...")
    daily_returns = load_daily_returns()
    macro_panel = load_macro_panel()
    logging.info(f"ETF 收益: {daily_returns.shape}, 范围 {daily_returns.index[0]} ~ {daily_returns.index[-1]}")
    logging.info(f"宏观 panel: {macro_panel.shape}, 列: {macro_panel.columns.tolist()}")

    results = []
    nav_curves = {}
    state_dists = {}

    for asset in TEST_ASSETS:
        if asset not in daily_returns.columns:
            logging.warning(f"资产 {asset} 不在 daily_returns 中, 跳过")
            continue
        logging.info(f"\n=== {asset} ({ASSET_LABELS[asset]}) ===")
        for version in VERSIONS:
            logging.info(f"  {version}...")
            res = run_single(asset, daily_returns, macro_panel, version)
            results.append({
                "asset": asset,
                "asset_name": ASSET_LABELS[asset],
                "version": version,
                **res,
            })
            nav_curves[(asset, version)] = res["nav_oos"]
            state_dists[(asset, version)] = res["state_dist"]
            logging.info(
                f"    Sharpe={res['sharpe']:.3f}, Calmar={res['calmar']:.3f}, "
                f"AnnRet={res['ann_return']*100:.2f}%, MaxDD={res['max_drawdown']*100:.2f}%, "
                f"Bear%={res['mean_bear_pct']:.3f}, Dist={res['state_dist']}"
            )

    # 保存 CSV
    df_results = pd.DataFrame([{
        "asset": r["asset"],
        "asset_name": r["asset_name"],
        "version": r["version"],
        "AnnRet": r["ann_return"],
        "Vol": r["vol"],
        "Sharpe": r["sharpe"],
        "MaxDD": r["max_drawdown"],
        "Calmar": r["calmar"],
        "MeanBearPct": r["mean_bear_pct"],
    } for r in results])
    df_results.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    logging.info(f"\nCSV 已保存: {OUTPUT_DIR / 'comparison.csv'}")

    # 绘制状态分布图
    fig, axes = plt.subplots(len(TEST_ASSETS), 1, figsize=(14, 3 * len(TEST_ASSETS)))
    if len(TEST_ASSETS) == 1:
        axes = [axes]
    for i, asset in enumerate(TEST_ASSETS):
        if asset not in daily_returns.columns:
            continue
        ax = axes[i]
        width = 0.2
        x = np.arange(len(VERSIONS))
        for vi, version in enumerate(VERSIONS):
            dist = state_dists.get((asset, version), {})
            is_3state = "3state" in version
            labels = ["bull", "neutral", "bear"] if is_3state else ["bull", "bear"]
            vals = [dist.get(lbl, 0) for lbl in labels]
            colors = ["#4CAF50", "#FFC107", "#F44336"] if is_3state else ["#4CAF50", "#F44336"]
            offset = (vi - (len(VERSIONS) - 1) / 2) * width
            ax.bar(x[vi] + offset, vals, width, color=colors[:len(vals)],
                   edgecolor="black", linewidth=0.3)
        # 在第 0 个子图上画图例
        if i == 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#4CAF50", edgecolor="black", label="Bull"),
                Patch(facecolor="#FFC107", edgecolor="black", label="Neutral"),
                Patch(facecolor="#F44336", edgecolor="black", label="Bear"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(VERSIONS, rotation=15, fontsize=8)
        ax.set_title(f"{asset} ({ASSET_LABELS[asset]}) — 状态分布 (OOS)", fontsize=11)
        ax.set_ylabel("占比")
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "state_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"状态分布图已保存: {OUTPUT_DIR / 'state_distribution.png'}")

    # 绘制 NAV 曲线 (每个资产一张)
    fig, axes = plt.subplots(len(TEST_ASSETS), 1, figsize=(12, 3 * len(TEST_ASSETS)))
    if len(TEST_ASSETS) == 1:
        axes = [axes]
    for i, asset in enumerate(TEST_ASSETS):
        if asset not in daily_returns.columns:
            continue
        ax = axes[i]
        # 基准: buy-and-hold
        bh = daily_returns[asset].loc[OOS_START:OOS_END].fillna(0.0)
        bh_nav = (1 + bh).cumprod() / (1 + bh).cumprod().iloc[0]
        ax.plot(bh_nav.index, bh_nav.values, color="gray", linestyle="--", alpha=0.7, label="Buy & Hold", linewidth=1.5)
        for version in VERSIONS:
            nav = nav_curves.get((asset, version))
            if nav is not None and len(nav) > 0:
                ax.plot(nav.index, nav.values, color=VERSION_COLORS[version],
                        label=VERSION_LABELS[version].replace("\n", " "), linewidth=1.2)
        ax.set_title(f"{asset} ({ASSET_LABELS[asset]}) — OOS NAV")
        ax.set_ylabel("NAV")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    logging.info(f"NAV 曲线已保存: {OUTPUT_DIR / 'equity_curves.png'}")

    # 生成 summary.md
    generate_summary(df_results, OUTPUT_DIR / "summary.md")
    logging.info(f"\n总结报告已保存: {OUTPUT_DIR / 'summary.md'}")


def generate_summary(df: pd.DataFrame, output_path: Path):
    """生成 Markdown 总结报告."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# v8 Jump Model 优化实验报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**OOS 区间**: {OOS_START.date()} ~ {OOS_END.date()} ({(OOS_END - OOS_START).days} 天)\n\n")

        f.write("## 1. 实验设计\n\n")
        f.write("| 版本 | n_states | 特征 |\n")
        f.write("|------|---------|------|\n")
        f.write("| v8_base | 2 | DD_10, Sortino_20, Sortino_60 (3 维) |\n")
        f.write("| v8_3state | 3 | 同上 (3 维) |\n")
        f.write("| v8_3state_macro | 3 | 上 + VIX, DXY, real_rate, cn_us_spread, gold_oil_corr (8 维) |\n")
        f.write("| v8_2state_macro | 2 | 同上 (8 维) |\n\n")

        f.write("## 2. OOS 性能对比\n\n")
        f.write("| 资产 | 版本 | AnnRet | Vol | Sharpe | MaxDD | Calmar |\n")
        f.write("|------|------|--------|-----|--------|-------|--------|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['asset_name']} | {row['version']} | "
                f"{row['AnnRet']*100:.2f}% | {row['Vol']*100:.2f}% | "
                f"{row['Sharpe']:.3f} | {row['MaxDD']*100:.2f}% | {row['Calmar']:.3f} |\n"
            )

        f.write("\n## 3. 平均性能 (跨 5 资产)\n\n")
        avg = df.groupby("version")[["AnnRet", "Vol", "Sharpe", "MaxDD", "Calmar", "MeanBearPct"]].mean()
        f.write("| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar | Avg Bear% |\n")
        f.write("|------|-----------|---------|-----------|-----------|------------|----------|\n")
        for version in VERSIONS:
            r = avg.loc[version]
            f.write(
                f"| {version} | {r['AnnRet']*100:.2f}% | {r['Vol']*100:.2f}% | "
                f"{r['Sharpe']:.3f} | {r['MaxDD']*100:.2f}% | {r['Calmar']:.3f} | "
                f"{r['MeanBearPct']:.3f} |\n"
            )

        f.write("\n## 4. 决策建议\n\n")
        # 计算每个版本相对 base 的提升
        base_sharpe = avg.loc["v8_base", "Sharpe"]
        base_calmar = avg.loc["v8_base", "Calmar"]
        for version in VERSIONS:
            if version == "v8_base":
                continue
            sharpe_gain = avg.loc[version, "Sharpe"] - base_sharpe
            calmar_gain = avg.loc[version, "Calmar"] - base_calmar
            bear_reduction = avg.loc["v8_base", "MeanBearPct"] - avg.loc[version, "MeanBearPct"]
            f.write(f"- **{version}** vs base: Sharpe {sharpe_gain:+.3f}, "
                    f"Calmar {calmar_gain:+.3f}, Bear% 减少 {bear_reduction:+.3f}\n")

        f.write("\n## 5. 状态分布\n\n")
        f.write("见 `state_distribution.png`\n\n")

        f.write("## 6. NAV 曲线\n\n")
        f.write("见 `equity_curves.png`\n")


if __name__ == "__main__":
    main()