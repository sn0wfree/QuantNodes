#!/usr/bin/env python3
# coding=utf-8
"""v8 Jump Model 综合诊断 (Step 1-3 整合).

Step 1: 数据检查 (NaN/异常/真实市场状态)
Step 2: 代码路径追踪 (centroids/cost/状态分类)
Step 3: 参数评估矩阵 (n_restarts/train_window/random_state/jump_penalty)

输出:
  reports/momentum_etf_rotation/v8_diagnostic/
    ├── Step1_data_report.md
    ├── Step2_code_report.md
    ├── Step3_params_report.md
    ├── all_metrics.csv
    └── diagnostic_summary.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v8_diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_ASSETS = ["510300", "511260", "518880", "159915", "512760"]
ASSET_LABELS = {
    "510300": "沪深300", "511260": "国债", "518880": "黄金",
    "159915": "创业板", "512760": "半导体",
}
WALK_FORWARD_STARTS = [
    pd.Timestamp("2018-01-01"),
    pd.Timestamp("2019-01-01"),
    pd.Timestamp("2020-01-01"),
]

# 已知熊市期 (用于 Step 1.3)
KNOWN_BEAR_PERIODS = [
    ("2018贸易战", pd.Timestamp("2018-02-01"), pd.Timestamp("2018-12-31")),
    ("2020 COVID", pd.Timestamp("2020-01-15"), pd.Timestamp("2020-04-30")),
    ("2022熊市", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")),
]


# ============================================================
# 加载主代码的 Jump Model (与实验脚本对比)
# ============================================================
def load_jump_model_from_main_code():
    """从主代码加载 Jump Model 实现."""
    from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
        jump_model_rolling, compute_features, JUMP_PENALTY_MAP,
        TRAIN_WINDOW_MAP, RETRAIN_EVERY_MAP,
    )
    return {
        "jump_model_rolling": jump_model_rolling,
        "compute_features": compute_features,
        "JUMP_PENALTY_MAP": JUMP_PENALTY_MAP,
        "TRAIN_WINDOW_MAP": TRAIN_WINDOW_MAP,
        "RETRAIN_EVERY_MAP": RETRAIN_EVERY_MAP,
    }


def load_experiment_jump_model():
    """从实验脚本加载 Jump Model 实现."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v8_probabilistic_experiment",
        REPO / "scripts" / "v8_probabilistic_experiment.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Step 1: 数据检查
# ============================================================
def step1_data_check(daily_returns: pd.DataFrame) -> dict:
    """Step 1: 完整数据检查."""
    results = {"completeness": {}, "training_periods": {}, "bear_periods": {}}

    # 1.1 完整性检查
    for asset in TEST_ASSETS:
        rets = daily_returns[asset]
        n_total = len(rets)
        n_nan = rets.isna().sum()
        # 异常值
        valid_rets = rets.dropna()
        abs_max = valid_rets.abs().max() if len(valid_rets) > 0 else 0
        n_extreme = ((valid_rets.abs() > 0.10)).sum() if len(valid_rets) > 0 else 0
        results["completeness"][asset] = {
            "n_total": n_total,
            "n_nan": int(n_nan),
            "first_date": str(rets.index[0].date()),
            "last_date": str(rets.index[-1].date()),
            "abs_max_daily": float(abs_max),
            "n_extreme_10pct": int(n_extreme),
        }

    # 1.2 训练期真实市场状态
    for asset in TEST_ASSETS:
        rets = daily_returns[asset].dropna()
        results["training_periods"][asset] = {}
        for start_date in WALK_FORWARD_STARTS:
            period_rets = rets.loc[start_date:].iloc[:1000]
            if len(period_rets) < 950:
                continue
            cum_ret = (1 + period_rets).prod() - 1
            cum = (1 + period_rets).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            ann_vol = period_rets.std() * np.sqrt(252)
            last_60d = period_rets.iloc[-60:]
            last_60d_ret = (1 + last_60d).prod() - 1
            results["training_periods"][asset][start_date.strftime("%Y-%m-%d")] = {
                "n_days": len(period_rets),
                "cum_ret": round(cum_ret * 100, 2),
                "max_dd": round(max_dd * 100, 2),
                "ann_vol": round(ann_vol * 100, 2),
                "last_60d_ret": round(last_60d_ret * 100, 2),
            }

    # 1.3 已知熊市期实际表现
    for asset in TEST_ASSETS:
        rets = daily_returns[asset].dropna()
        results["bear_periods"][asset] = {}
        for period_name, period_start, period_end in KNOWN_BEAR_PERIODS:
            period_rets = rets.loc[period_start:period_end]
            if len(period_rets) < 5:
                continue
            cum_ret = (1 + period_rets).prod() - 1
            cum = (1 + period_rets).cumprod()
            max_dd = (cum / cum.cummax() - 1).min()
            results["bear_periods"][asset][period_name] = {
                "n_days": len(period_rets),
                "cum_ret": round(cum_ret * 100, 2),
                "max_dd": round(max_dd * 100, 2),
            }

    return results


# ============================================================
# Step 2: 代码路径追踪
# ============================================================
def _trace_jump_model_training(
    returns: pd.Series,
    feats: pd.DataFrame,
    n_restarts: int = 5,
    n_iter: int = 10,
    jump_penalty: float = 50.0,
    random_state: int = 42,
) -> dict:
    """追踪单次 Jump Model 训练的详细信息."""
    from scripts.v8_probabilistic_experiment import (
        _dp_with_probs, _classify_states,
    )

    np.random.seed(random_state)
    T = len(feats)
    n_states = 2
    mean = feats.values.mean(axis=0)
    std = feats.values.std(axis=0) + 1e-10
    feats_z = (feats.values - mean) / std

    # 多次重启
    all_costs = []
    all_states = []
    all_centroids = []
    for restart in range(n_restarts):
        centroids = np.random.randn(n_states, feats_z.shape[1])
        states = None
        for it in range(n_iter):
            st, _ = _dp_with_probs(feats_z, centroids, jump_penalty, n_states, 50.0)
            for s in range(n_states):
                mask = st == s
                if mask.sum() > 0:
                    centroids[s] = feats_z[mask].mean(axis=0)
            states = st

        cost = 0.0
        for tt in range(len(states)):
            cost += np.sum((feats_z[tt] - centroids[states[tt]]) ** 2)
        for tt in range(1, len(states)):
            if states[tt] != states[tt - 1]:
                cost += jump_penalty

        all_costs.append(cost)
        all_states.append(states)
        all_centroids.append(centroids.copy())

    # 选最优
    best_idx = int(np.argmin(all_costs))
    best_states = all_states[best_idx]
    best_centroids = all_centroids[best_idx]

    # Centroids 距离
    centroid_distance = float(np.linalg.norm(best_centroids[0] - best_centroids[1]))

    # 状态分类
    state_labels = _classify_states(best_states, returns, n_states)
    # state_labels 是 {raw_id: sem_id}, sem_id=0 是 bull (累计收益最高), sem_id=1 是 bear
    bull_raw = [k for k, v in state_labels.items() if v == 0]
    bear_raw = [k for k, v in state_labels.items() if v == 1]
    if len(bull_raw) == 0 or len(bear_raw) == 0:
        # 分类失败 (应该不会发生, 但防御性处理)
        bull_cum_ret = 0.0
        bear_cum_ret = 0.0
        bull_mask = np.zeros(len(best_states), dtype=bool)
        bear_mask = np.zeros(len(best_states), dtype=bool)
    else:
        bull_raw = bull_raw[0]
        bear_raw = bear_raw[0]

        # bull/bear 累计收益
        bull_mask = best_states == bull_raw
        bear_mask = best_states == bear_raw
        bull_cum_ret = float(returns.values[bull_mask].sum()) if bull_mask.sum() > 0 else 0.0
        bear_cum_ret = float(returns.values[bear_mask].sum()) if bear_mask.sum() > 0 else 0.0

    return {
        "best_cost": float(all_costs[best_idx]),
        "cost_min": float(np.min(all_costs)),
        "cost_max": float(np.max(all_costs)),
        "cost_std": float(np.std(all_costs)),
        "centroid_distance": round(centroid_distance, 4),
        "n_bull_days": int(bull_mask.sum()),
        "n_bear_days": int(bear_mask.sum()),
        "bull_cum_ret": round(bull_cum_ret * 100, 4),
        "bear_cum_ret": round(bear_cum_ret * 100, 4),
        "class_ret_diff": round((bull_cum_ret - bear_cum_ret) * 100, 4),
        "all_costs": [round(c, 2) for c in all_costs],
    }


def step2_code_trace(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Step 2: 对每个 (asset, start) 追踪 Jump Model 训练过程."""
    records = []
    for asset in TEST_ASSETS:
        rets = daily_returns[asset].dropna()
        for start_date in WALK_FORWARD_STARTS:
            period_rets = rets.loc[start_date:].iloc[:1000]
            if len(period_rets) < 950:
                continue
            from scripts.v8_probabilistic_experiment import compute_features
            feats = compute_features(period_rets).dropna()
            common = period_rets.index.intersection(feats.index)
            period_rets_aligned = period_rets.loc[common]
            feats = feats.loc[common]

            trace = _trace_jump_model_training(
                period_rets_aligned, feats,
                n_restarts=5, n_iter=10, jump_penalty=50.0, random_state=42,
            )
            records.append({
                "asset": asset,
                "asset_name": ASSET_LABELS[asset],
                "start_date": start_date.strftime("%Y-%m-%d"),
                **trace,
            })
            logging.info(
                f"  {asset} {start_date.date()}: centroid_dist={trace['centroid_distance']:.3f}, "
                f"n_bear={trace['n_bear_days']}, cost_std={trace['cost_std']:.2f}"
            )
    return pd.DataFrame(records)


# ============================================================
# Step 3: 参数评估矩阵
# ============================================================
def step3_params_eval(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Step 3: 参数敏感性矩阵."""
    records = []

    # 默认参数 (主代码)
    base_params = {
        "n_restarts": 5, "train_window": 1000, "random_state": 42,
        "jump_penalty": 50.0,
    }

    # 资产 510300 + 起点 2018-01-01 (代表性组合)
    asset = "510300"
    start_date = WALK_FORWARD_STARTS[0]
    rets = daily_returns[asset].dropna().loc[start_date:]
    period_rets = rets.iloc[:1000]  # 取前 1000 天

    from scripts.v8_probabilistic_experiment import compute_features
    feats_base = compute_features(period_rets).dropna()
    common = period_rets.index.intersection(feats_base.index)
    period_rets_aligned = period_rets.loc[common]
    feats_base = feats_base.loc[common]

    # 实验 A: n_restarts
    for nr in [5, 10, 20, 50]:
        trace = _trace_jump_model_training(
            period_rets_aligned, feats_base,
            n_restarts=nr, n_iter=10, jump_penalty=50.0, random_state=42,
        )
        records.append({
            "experiment": "A_n_restarts",
            "param_name": "n_restarts",
            "param_value": nr,
            "asset": asset,
            "start_date": start_date.strftime("%Y-%m-%d"),
            **{k: trace[k] for k in ["centroid_distance", "n_bull_days", "n_bear_days",
                                       "bear_cum_ret", "bull_cum_ret", "cost_std"]},
        })
        logging.info(f"  A_n_restarts={nr}: n_bear={trace['n_bear_days']}, "
                     f"centroid_dist={trace['centroid_distance']:.3f}")

    # 实验 B: train_window
    for tw in [500, 750, 1000, 1500]:
        if tw > len(rets):
            continue
        period_rets_b = rets.iloc[:tw]
        feats_b = compute_features(period_rets_b).dropna()
        common_b = period_rets_b.index.intersection(feats_b.index)
        period_rets_b_aligned = period_rets_b.loc[common_b]
        feats_b = feats_b.loc[common_b]

        trace = _trace_jump_model_training(
            period_rets_b_aligned, feats_b,
            n_restarts=5, n_iter=10, jump_penalty=50.0, random_state=42,
        )
        records.append({
            "experiment": "B_train_window",
            "param_name": "train_window",
            "param_value": tw,
            "asset": asset,
            "start_date": start_date.strftime("%Y-%m-%d"),
            **{k: trace[k] for k in ["centroid_distance", "n_bull_days", "n_bear_days",
                                       "bear_cum_ret", "bull_cum_ret", "cost_std"]},
        })
        logging.info(f"  B_train_window={tw}: n_bear={trace['n_bear_days']}, "
                     f"centroid_dist={trace['centroid_distance']:.3f}")

    # 实验 C: random_state
    for rs in [42, 0, 1, 2, 3]:
        trace = _trace_jump_model_training(
            period_rets_aligned, feats_base,
            n_restarts=5, n_iter=10, jump_penalty=50.0, random_state=rs,
        )
        records.append({
            "experiment": "C_random_state",
            "param_name": "random_state",
            "param_value": rs,
            "asset": asset,
            "start_date": start_date.strftime("%Y-%m-%d"),
            **{k: trace[k] for k in ["centroid_distance", "n_bull_days", "n_bear_days",
                                       "bear_cum_ret", "bull_cum_ret", "cost_std"]},
        })
        logging.info(f"  C_random_state={rs}: n_bear={trace['n_bear_days']}, "
                     f"centroid_dist={trace['centroid_distance']:.3f}")

    # 实验 D: jump_penalty
    for jp in [25, 50, 80, 120]:
        trace = _trace_jump_model_training(
            period_rets_aligned, feats_base,
            n_restarts=5, n_iter=10, jump_penalty=float(jp), random_state=42,
        )
        records.append({
            "experiment": "D_jump_penalty",
            "param_name": "jump_penalty",
            "param_value": jp,
            "asset": asset,
            "start_date": start_date.strftime("%Y-%m-%d"),
            **{k: trace[k] for k in ["centroid_distance", "n_bull_days", "n_bear_days",
                                       "bear_cum_ret", "bull_cum_ret", "cost_std"]},
        })
        logging.info(f"  D_jump_penalty={jp}: n_bear={trace['n_bear_days']}, "
                     f"centroid_dist={trace['centroid_distance']:.3f}")

    return pd.DataFrame(records)


# ============================================================
# 报告生成
# ============================================================
def write_step1_report(results: dict):
    with open(OUTPUT_DIR / "Step1_data_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 1: 数据检查报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        f.write("## 1.1 数据完整性\n\n")
        f.write("| 资产 | 总天数 | NaN数 | 起始日 | 结束日 | 最大单日收益 | 异常值(>10%) |\n")
        f.write("|------|--------|-------|--------|--------|------------|-------------|\n")
        for asset in TEST_ASSETS:
            d = results["completeness"][asset]
            f.write(
                f"| {ASSET_LABELS[asset]} ({asset}) | {d['n_total']} | {d['n_nan']} | "
                f"{d['first_date']} | {d['last_date']} | "
                f"{d['abs_max_daily']*100:.2f}% | {d['n_extreme_10pct']} |\n"
            )

        f.write("\n## 1.2 训练期真实市场状态 (1000天)\n\n")
        f.write("| 资产 | 起点 | 累计收益 | 最大回撤 | 年化波动 | 最后60天收益 |\n")
        f.write("|------|------|---------|---------|---------|-------------|\n")
        for asset in TEST_ASSETS:
            for start_date, d in results["training_periods"][asset].items():
                f.write(
                    f"| {ASSET_LABELS[asset]} ({asset}) | {start_date} | "
                    f"{d['cum_ret']:+.2f}% | {d['max_dd']:.2f}% | "
                    f"{d['ann_vol']:.2f}% | {d['last_60d_ret']:+.2f}% |\n"
                )

        f.write("\n## 1.3 已知熊市期实际表现\n\n")
        f.write("| 资产 | 时期 | 天数 | 累计收益 | 最大回撤 |\n")
        f.write("|------|------|------|---------|---------|\n")
        for asset in TEST_ASSETS:
            for period_name, d in results["bear_periods"][asset].items():
                f.write(
                    f"| {ASSET_LABELS[asset]} ({asset}) | {period_name} | "
                    f"{d['n_days']} | {d['cum_ret']:+.2f}% | {d['max_dd']:.2f}% |\n"
                )


def write_step2_report(df_trace: pd.DataFrame):
    with open(OUTPUT_DIR / "Step2_code_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 2: 代码路径追踪报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        f.write("## 2.1 Jump Model 训练详细参数\n\n")
        f.write("| 资产 | 起点 | centroids距离 | cost min-max | cost_std | bull/bear累计收益差 |\n")
        f.write("|------|------|---------------|--------------|----------|--------------------|\n")
        for _, r in df_trace.iterrows():
            f.write(
                f"| {r['asset_name']} ({r['asset']}) | {r['start_date']} | "
                f"{r['centroid_distance']:.4f} | "
                f"{r['cost_min']:.0f} ~ {r['cost_max']:.0f} | {r['cost_std']:.2f} | "
                f"{r['class_ret_diff']:+.4f}% |\n"
            )

        f.write("\n## 2.2 状态分布\n\n")
        f.write("| 资产 | 起点 | bull_days | bear_days | bull_ratio | bear_ratio |\n")
        f.write("|------|------|-----------|-----------|------------|------------|\n")
        for _, r in df_trace.iterrows():
            total = r["n_bull_days"] + r["n_bear_days"]
            bull_ratio = r["n_bull_days"] / total if total > 0 else 0
            bear_ratio = r["n_bear_days"] / total if total > 0 else 0
            f.write(
                f"| {r['asset_name']} ({r['asset']}) | {r['start_date']} | "
                f"{r['n_bull_days']} | {r['n_bear_days']} | "
                f"{bull_ratio*100:.1f}% | {bear_ratio*100:.1f}% |\n"
            )

        f.write("\n## 2.3 关键发现\n\n")
        # 判断 bear_days > 0 的比例
        n_with_bear = (df_trace["n_bear_days"] > 0).sum()
        n_total = len(df_trace)
        f.write(f"- **{n_with_bear}/{n_total}** 个训练组合检测到 bear 状态\n")
        # 平均 centroids 距离
        avg_dist = df_trace["centroid_distance"].mean()
        f.write(f"- 平均 centroids 距离: {avg_dist:.4f}\n")
        # 平均 cost_std
        avg_cost_std = df_trace["cost_std"].mean()
        f.write(f"- 平均 cost std (跨 restarts): {avg_cost_std:.2f}\n")
        # 分类稳定性
        avg_class_diff = df_trace["class_ret_diff"].mean()
        f.write(f"- 平均 bull/bear 累计收益差: {avg_class_diff:+.4f}%\n")

        # 诊断结论
        f.write("\n### 诊断结论\n\n")
        if avg_dist < 0.5:
            f.write("- ❌ **centroids 距离过小** (< 0.5): 两个聚类几乎重合\n")
        elif avg_dist < 1.0:
            f.write("- ⚠️ **centroids 距离较小** (< 1.0): 聚类分离度不够\n")
        else:
            f.write("- ✅ centroids 距离合理 (> 1.0)\n")

        if avg_cost_std / df_trace["best_cost"].mean() > 0.05:
            f.write("- ⚠️ **cost_std 较大** (> 5%): 模型可能不稳定\n")
        else:
            f.write("- ✅ cost_std 较小 (< 5%): 模型稳定\n")

        if n_with_bear < n_total / 2:
            f.write("- ❌ **大部分训练组合未能检测到 bear 状态**: 模型可能在训练期崩溃\n")
        else:
            f.write("- ✅ 大部分训练组合能检测到 bear 状态\n")


def write_step3_report(df_params: pd.DataFrame):
    with open(OUTPUT_DIR / "Step3_params_report.md", "w", encoding="utf-8") as f:
        f.write("# Step 3: 参数评估报告\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")
        f.write(f"**测试资产**: 510300 (沪深300)\n")
        f.write(f"**起点**: 2018-01-01 (训练期 1000 天)\n\n")

        for exp_name in ["A_n_restarts", "B_train_window", "C_random_state", "D_jump_penalty"]:
            sub = df_params[df_params["experiment"] == exp_name]
            if sub.empty:
                continue
            param_name = sub.iloc[0]["param_name"]
            f.write(f"## {exp_name}: {param_name}\n\n")
            f.write(f"| {param_name} | n_bull | n_bear | centroids距离 | cost_std | bull_cum_ret | bear_cum_ret |\n")
            f.write(f"|{'-'*15}|--------|--------|---------------|----------|-------------|-------------|\n")
            for _, r in sub.iterrows():
                f.write(
                    f"| {r['param_value']} | {r['n_bull_days']} | {r['n_bear_days']} | "
                    f"{r['centroid_distance']:.4f} | {r['cost_std']:.2f} | "
                    f"{r['bull_cum_ret']:.2f}% | {r['bear_cum_ret']:.2f}% |\n"
                )
            f.write("\n")


def write_diagnostic_summary(
    step1_results: dict,
    step2_df: pd.DataFrame,
    step3_df: pd.DataFrame,
):
    with open(OUTPUT_DIR / "diagnostic_summary.md", "w", encoding="utf-8") as f:
        f.write("# v8 Jump Model 综合诊断总结\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n\n")

        f.write("## 核心发现\n\n")

        # Step 1 关键发现
        f.write("### Step 1: 数据\n\n")
        # 检查训练期是否真有熊市
        has_bear_in_training = False
        for asset in TEST_ASSETS:
            for start_date, d in step1_results["training_periods"][asset].items():
                if d["max_dd"] < -15:  # 超过15%回撤
                    has_bear_in_training = True
                    break
        if has_bear_in_training:
            f.write("- ✅ 训练期内**存在重大回撤**（>15%），市场有真实的熊市期\n")
        else:
            f.write("- ⚠️ 训练期内回撤较小\n")

        # Step 2 关键发现
        f.write("\n### Step 2: 代码\n\n")
        n_with_bear = (step2_df["n_bear_days"] > 0).sum()
        n_total = len(step2_df)
        f.write(f"- **{n_with_bear}/{n_total}** 个训练组合成功检测到 bear 状态\n")
        avg_dist = step2_df["centroid_distance"].mean()
        f.write(f"- 平均 centroids 距离: {avg_dist:.4f}\n")
        if n_with_bear == 0:
            f.write("- ❌ **所有训练组合都输出 bear=0**: Jump Model 在 1000 天训练期未能成功区分两个状态\n")
            f.write("- 可能原因: n_restarts=5 不足, OR 1000 天数据本身聚类不清晰\n")

        # Step 3 关键发现
        f.write("\n### Step 3: 参数\n\n")
        for exp_name in ["A_n_restarts", "B_train_window", "C_random_state", "D_jump_penalty"]:
            sub = step3_df[step3_df["experiment"] == exp_name]
            if sub.empty:
                continue
            param_name = sub.iloc[0]["param_name"]
            max_bear = sub["n_bear_days"].max()
            min_bear = sub["n_bear_days"].min()
            f.write(f"- **{param_name}**: bear_days 范围 [{min_bear}, {max_bear}]\n")

        # 综合建议
        f.write("\n## Step 4 实验设计建议\n\n")
        if n_with_bear == 0:
            f.write("由于所有训练组合都输出 bear=0，**Walk-forward 在 1000 天训练窗口下不可靠**。\n\n")
            f.write("**建议方案**: 改用主代码全样本状态序列对比 v8_method_b vs v8_prob vs v8_uniform。\n\n")
            f.write("具体做法:\n")
            f.write("1. 用主代码的 `jump_model_periodic_retrain` 在 2018-2026 全样本上生成状态序列\n")
            f.write("2. 对每个状态序列应用 3 个策略 (method_b / prob / uniform)\n")
            f.write("3. 比较 OOS Sharpe / Calmar / MaxDD\n")
        else:
            f.write("部分训练组合能检测到 bear 状态，**Walk-forward 可用但需要优化参数**。\n\n")
            f.write("**建议**: 增加 n_restarts 到 10 重做 Walk-forward。\n")


# ============================================================
# Main
# ============================================================
def main():
    logging.info("=" * 70)
    logging.info("v8 Jump Model 综合诊断 (Step 1-3)")
    logging.info("=" * 70)

    daily_returns = pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")
    logging.info(f"ETF 数据: {daily_returns.shape}, 范围 {daily_returns.index[0]} ~ {daily_returns.index[-1]}")

    # Step 1
    logging.info("\n" + "=" * 70)
    logging.info("Step 1: 数据检查")
    logging.info("=" * 70)
    step1_results = step1_data_check(daily_returns)
    write_step1_report(step1_results)
    logging.info(f"Step1_data_report.md 已生成")

    # Step 2
    logging.info("\n" + "=" * 70)
    logging.info("Step 2: 代码路径追踪")
    logging.info("=" * 70)
    step2_df = step2_code_trace(daily_returns)
    step2_df.to_csv(OUTPUT_DIR / "code_trace.csv", index=False)
    write_step2_report(step2_df)
    logging.info(f"Step2_code_report.md 已生成")

    # Step 3
    logging.info("\n" + "=" * 70)
    logging.info("Step 3: 参数评估")
    logging.info("=" * 70)
    step3_df = step3_params_eval(daily_returns)
    step3_df.to_csv(OUTPUT_DIR / "params_eval.csv", index=False)
    write_step3_report(step3_df)
    logging.info(f"Step3_params_report.md 已生成")

    # 综合 summary
    write_diagnostic_summary(step1_results, step2_df, step3_df)
    logging.info(f"\ndiagnostic_summary.md 已生成")
    logging.info(f"\n所有报告已保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()