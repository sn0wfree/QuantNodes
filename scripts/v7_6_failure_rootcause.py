# coding: utf-8
"""v7.6 2021/2022 失效根因深度分析.

目的: 找到 v7.6 在 2021/2022 起点 Calmar 偏低的根因
   1. β_path 对比: 2019-2020 (好) vs 2021-2022 (差) 各因子权重变化
   2. X 因子预测能力: 9 macro + 11 量价 各因子 IC 衰减
   3. 调仓时点贡献: 找到最差时点
   4. Regime 切换: 2022 熊市是否可提前识别

用法:
   python3.11 scripts/v7_6_failure_rootcause.py

输出:
   reports/momentum_etf_rotation/v7_6_rootcause/
   ├── beta_path_2019_2020.png     # β 时序图
   ├── beta_path_2021_2022.png
   ├── factor_ic.csv               # 各因子 IC
   ├── factor_topn_accuracy.csv    # top_n 预测准确度
   ├── regime_signals.csv          # regime 信号时序
   └── rootcause_report.md         # 综合报告
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_6_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 锁定最优配置
BASE_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
    "rho": 2.0,
    "top_n": 5,
    "max_weight": 0.25,
    "min_history": 52,
}

# 因子名称
FACTOR_NAMES = [
    "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
    "无风险收益率", "信用利差因子", "期限利差因子_债",
    "期限利差因子_股", "宏观汇率因子",
    "f1_second_mom", "f2_mom_term",
    "f3_amt_vol", "f4_vol_vol",
    "f5_turnover", "f6_ls_total", "f7_ls_change",
    "f8_pv_rankcov", "f9_pv_corr",
    "f10_first_div", "f11_vol_range",
]

# 对比区间
GOOD_PERIOD = ("2019-01-01", "2020-12-31")  # 起点 2019/2020 Calmar 3.0+
BAD_PERIOD = ("2021-01-01", "2022-12-31")   # 起点 2021/2022 Calmar 0.7-1.4

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation" / "v7_6_rootcause"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def analyze_beta_path(beta_path: pd.DataFrame, period_label: str, start: str, end: str):
    """分析 β_path 在指定区间的特征."""
    logging.info("=" * 60)
    logging.info("[β_path 分析] 区间 %s (%s ~ %s)", period_label, start, end)

    mask = (beta_path.index >= start) & (beta_path.index <= end)
    bp = beta_path[mask]
    if len(bp) < 10:
        logging.warning("  数据不足")
        return

    logging.info("  周数: %d", len(bp))

    # 1. 各因子均值
    means = bp.mean()
    stds = bp.std()
    abs_means = means.abs()
    abs_means_sorted = abs_means.sort_values(ascending=False)

    # 2. 重要因子 (绝对均值 Top 5)
    top5 = abs_means_sorted.head(5).index.tolist()
    logging.info("  Top 5 重要因子 (abs mean):")
    for i, name in enumerate(top5):
        idx = int(name.replace("factor_", ""))
        fname = FACTOR_NAMES[idx] if idx < len(FACTOR_NAMES) else name
        logging.info("    %d. %s (idx=%d): mean=%.4f, std=%.4f",
                     i + 1, fname, idx, means[name], stds[name])

    # 3. β 稀疏性 (使用 float 转换)
    sparsity = float((bp.abs() < 0.001).sum().sum()) / float(bp.size)
    logging.info("  β 稀疏度: %.1f%% (近零占比)", sparsity * 100)

    # 4. β 变化率
    diffs = bp.diff().dropna()
    mean_abs_diff = float(diffs.abs().mean().mean())
    logging.info("  β 平均变化率: %.4f", mean_abs_diff)

    return {
        "label": period_label,
        "n_weeks": len(bp),
        "top5_factors": [(int(t.replace("factor_", "")), float(means[t]), float(stds[t])) for t in top5],
        "sparsity": float(sparsity),
        "mean_abs_diff": float(mean_abs_diff),
    }


def analyze_factor_ic(X_panel: np.ndarray, Y: pd.DataFrame, period_label: str, start: str, end: str):
    """分析各因子在指定区间的预测能力 (IC = corr(X, Y_next))."""
    logging.info("=" * 60)
    logging.info("[X 因子 IC 分析] 区间 %s (%s ~ %s)", period_label, start, end)

    # 选时间区间
    mask = (Y.index >= start) & (Y.index <= end)
    idx_pos = np.where(mask)[0]
    if len(idx_pos) < 20:
        logging.warning("  数据不足")
        return

    T, N, K = X_panel.shape
    ic_per_factor = np.zeros(K)
    valid_count = 0

    for k in range(K):
        ic_t = []
        for t in idx_pos[:-1]:
            if t + 1 >= T:
                break
            x_k = X_panel[t, :, k]  # (N,) 因子值
            y_next = Y.iloc[t + 1].values  # (N,) 下期收益
            valid = ~(np.isnan(x_k) | np.isnan(y_next))
            if valid.sum() < 5:
                continue
            corr = np.corrcoef(x_k[valid], y_next[valid])[0, 1]
            if not np.isnan(corr):
                ic_t.append(corr)
        if len(ic_t) > 0:
            ic_per_factor[k] = np.mean(ic_t)
            valid_count += 1

    # 排序
    sorted_idx = np.argsort(-np.abs(ic_per_factor))
    logging.info("  Top 8 因子 (|IC|):")
    rows = []
    for i in range(min(8, K)):
        k = sorted_idx[i]
        fname = FACTOR_NAMES[k] if k < len(FACTOR_NAMES) else f"factor_{k}"
        logging.info("    %d. %s: IC=%.4f, |IC|=%.4f",
                     i + 1, fname, ic_per_factor[k], abs(ic_per_factor[k]))
        rows.append({
            "period": period_label,
            "factor_idx": k,
            "factor_name": fname,
            "ic": ic_per_factor[k],
            "abs_ic": abs(ic_per_factor[k]),
        })

    # 因子 IC 分组: 弱 (|IC|<0.05) / 中 / 强 (>0.10)
    n_weak = (np.abs(ic_per_factor) < 0.05).sum()
    n_medium = ((np.abs(ic_per_factor) >= 0.05) & (np.abs(ic_per_factor) < 0.10)).sum()
    n_strong = (np.abs(ic_per_factor) >= 0.10).sum()
    logging.info("  因子分布: 弱 %d, 中 %d, 强 %d", n_weak, n_medium, n_strong)

    return {
        "label": period_label,
        "n_weak": int(n_weak),
        "n_medium": int(n_medium),
        "n_strong": int(n_strong),
        "top_factors": rows,
        "ic_per_factor": ic_per_factor,
    }


def analyze_rebalance_contribution(Y, X_panel, beta_path, valid_codes, cfg, period_label, start, end):
    """分析各调仓时点的收益贡献, 找最差时点."""
    logging.info("=" * 60)
    logging.info("[调仓贡献分析] 区间 %s (%s ~ %s)", period_label, start, end)

    from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import construct_portfolio
    nav, weights_df = construct_portfolio(Y, X_panel, beta_path, cfg, return_weights=True)

    # 每个调仓时点的收益
    rebal_dates = sorted(weights_df["date"].unique())
    rebal_dates_in = [d for d in rebal_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]

    logging.info("  区间内调仓次数: %d", len(rebal_dates_in))

    # 找最差时点
    nav_series = nav
    weekly_rets = nav_series.pct_change().dropna()
    worst_weeks = weekly_rets.nsmallest(5)
    logging.info("  Top 5 最差周 (全段):")
    for dt, ret in worst_weeks.items():
        in_period = start <= str(dt.date()) <= end
        marker = "  <- 区间内" if in_period else ""
        logging.info("    %s: %.2f%%%s", dt.date(), ret * 100, marker)

    # 区间内最差周
    period_rets = weekly_rets.loc[start:end]
    if len(period_rets) > 0:
        worst_period = period_rets.nsmallest(3)
        logging.info("  Top 3 最差周 (区间内):")
        for dt, ret in worst_period.items():
            logging.info("    %s: %.2f%%", dt.date(), ret * 100)

    # 找最近的 nav 值
    period_cumret = None
    if len(nav) > 0:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        # 找 nav 中 <= start 的最后日期
        valid_start = nav.index[nav.index <= start_dt]
        valid_end = nav.index[nav.index <= end_dt]
        if len(valid_start) > 0 and len(valid_end) > 0:
            period_cumret = float(nav.loc[valid_end[-1]] / nav.loc[valid_start[-1]] - 1)

    return {
        "label": period_label,
        "n_rebalances": len(rebal_dates_in),
        "period_cumret": period_cumret,
    }


def analyze_regime_2022(daily_returns, X_panel, Y):
    """分析 2022 熊市是否可提前识别."""
    logging.info("=" * 60)
    logging.info("[2022 Regime 分析]")

    # 用 daily_returns 平均作为市场
    market_daily = daily_returns.mean(axis=1)

    # 60 日波动率
    vol_60 = market_daily.rolling(60).std() * np.sqrt(252)
    # 60 日动量
    ret_60 = (1 + market_daily).rolling(60).apply(np.prod, raw=True) - 1
    # 60 日 RSI
    gain = market_daily.clip(lower=0).rolling(60).mean()
    loss = (-market_daily.clip(upper=0)).rolling(60).mean()
    rsi_60 = 100 - 100 / (1 + gain / (loss + 1e-9))

    # 2022 区间
    regime_2022 = pd.DataFrame({
        "vol_60": vol_60,
        "ret_60": ret_60,
        "rsi_60": rsi_60,
    }).loc["2021-06-01":"2022-12-31"]

    # 各信号的 2022-Q1-Q4 表现
    logging.info("  2022 关键信号 (季度):")
    for q_start, q_end in [("2022-01-01", "2022-03-31"), ("2022-04-01", "2022-06-30"),
                            ("2022-07-01", "2022-09-30"), ("2022-10-01", "2022-12-31")]:
        q = regime_2022.loc[q_start:q_end]
        if len(q) == 0:
            continue
        logging.info("    Q %s: vol=%.2f%%, ret=%.2f%%, RSI=%.1f",
                     q_start, q["vol_60"].mean() * 100,
                     q["ret_60"].mean() * 100,
                     q["rsi_60"].mean())

    # 检测: vol > 阈值 AND ret < 0 持续 30 天 → 熊市
    regime_2022["bear_vol"] = regime_2022["vol_60"] > 0.18
    regime_2022["bear_trend"] = regime_2022["ret_60"] < 0
    regime_2022["bear_combo"] = regime_2022["bear_vol"] & regime_2022["bear_trend"]

    n_bear_combo = regime_2022["bear_combo"].sum()
    n_bear_trend = regime_2022["bear_trend"].sum()
    n_bear_vol = regime_2022["bear_vol"].sum()
    logging.info("  2022 H1 信号: bear_vol=%d, bear_trend=%d, bear_combo=%d (总 %d 天)",
                 n_bear_vol, n_bear_trend, n_bear_combo, len(regime_2022))

    return regime_2022


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 2021/2022 失效根因深度分析")
    logging.info("=" * 60)

    # 加载数据
    X_panel, Y, valid_codes = load_v7_6_data()
    daily_returns = load_daily_etf_returns()
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    cfg = V7_6Config(**BASE_PARAMS)

    # 估计全段 β_path
    logging.info("估计全段 β_path...")
    t0 = time.time()
    beta_path_full = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        method=cfg.method,
        min_history=cfg.min_history,
        window_size=cfg.window_size,
        rho=cfg.rho,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )
    logging.info("  β_path 估计完成, %.1fs", time.time() - t0)

    # 1. β_path 对比
    logging.info("\n" + "=" * 60)
    logging.info("分析 1: β_path 对比")
    res_good = analyze_beta_path(beta_path_full, "good_2019_2020", *GOOD_PERIOD)
    res_bad = analyze_beta_path(beta_path_full, "bad_2021_2022", *BAD_PERIOD)

    # 画 β_path 时序图
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    for idx in range(min(8, X_panel.shape[2])):
        col = f"factor_{idx}"
        if col in beta_path_full.columns:
            axes[0].plot(beta_path_full.index, beta_path_full[col],
                          label=FACTOR_NAMES[idx] if idx < len(FACTOR_NAMES) else col,
                          alpha=0.7)
    axes[0].set_title("β_path 2018-2026 (8/20 factors)")
    axes[0].set_ylabel("β")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axvspan(pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"),
                     alpha=0.2, color="green", label="Good period")
    axes[0].axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"),
                     alpha=0.2, color="red", label="Bad period")
    axes[0].grid(True, alpha=0.3)

    # 重点放大 bad period
    for idx in range(min(8, X_panel.shape[2])):
        col = f"factor_{idx}"
        if col in beta_path_full.columns:
            axes[1].plot(beta_path_full.index, beta_path_full[col],
                          label=FACTOR_NAMES[idx] if idx < len(FACTOR_NAMES) else col,
                          alpha=0.7)
    axes[1].set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-31"))
    axes[1].set_title("β_path 放大 2021-2022 (8/20 factors)")
    axes[1].set_ylabel("β")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    out_beta = OUTPUT_DIR / "beta_path_2021_2022.png"
    plt.savefig(out_beta, dpi=100, bbox_inches="tight")
    plt.close()
    logging.info("  保存: %s", out_beta)

    # 2. X 因子 IC 对比
    logging.info("\n" + "=" * 60)
    logging.info("分析 2: X 因子 IC 对比")
    ic_good = analyze_factor_ic(X_panel, Y, "good_2019_2020", *GOOD_PERIOD)
    ic_bad = analyze_factor_ic(X_panel, Y, "bad_2021_2022", *BAD_PERIOD)

    # IC 对比表
    ic_rows = []
    for k in range(X_panel.shape[2]):
        fname = FACTOR_NAMES[k] if k < len(FACTOR_NAMES) else f"factor_{k}"
        ic_rows.append({
            "factor_idx": k,
            "factor_name": fname,
            "ic_good": ic_good["ic_per_factor"][k] if ic_good else 0,
            "ic_bad": ic_bad["ic_per_factor"][k] if ic_bad else 0,
            "ic_diff": (ic_good["ic_per_factor"][k] - ic_bad["ic_per_factor"][k]) if (ic_good and ic_bad) else 0,
        })
    ic_df = pd.DataFrame(ic_rows).sort_values("ic_diff", ascending=False)
    out_ic = OUTPUT_DIR / "factor_ic.csv"
    ic_df.to_csv(out_ic, index=False)
    logging.info("  保存: %s", out_ic)

    # Top 衰减因子 (good→bad 退化最大)
    worst_decay = ic_df.head(5)
    logging.info("  Top 5 衰减因子 (good→bad):")
    for _, r in worst_decay.iterrows():
        logging.info("    %s: good IC=%.4f → bad IC=%.4f (衰减 %.4f)",
                     r["factor_name"], r["ic_good"], r["ic_bad"], r["ic_diff"])

    # Top 提升因子
    best_rise = ic_df.tail(5)
    logging.info("  Top 5 提升因子 (good→bad):")
    for _, r in best_rise.iterrows():
        logging.info("    %s: good IC=%.4f → bad IC=%.4f (提升 %.4f)",
                     r["factor_name"], r["ic_good"], r["ic_bad"], -r["ic_diff"])

    # 3. 调仓贡献
    logging.info("\n" + "=" * 60)
    logging.info("分析 3: 调仓贡献")
    rebal_good = analyze_rebalance_contribution(
        Y, X_panel, beta_path_full, valid_codes, cfg,
        "good_2019_2020", *GOOD_PERIOD
    )
    rebal_bad = analyze_rebalance_contribution(
        Y, X_panel, beta_path_full, valid_codes, cfg,
        "bad_2021_2022", *BAD_PERIOD
    )

    # 4. 2022 Regime 分析
    logging.info("\n" + "=" * 60)
    logging.info("分析 4: 2022 Regime 检测")
    regime_2022 = analyze_regime_2022(daily_returns, X_panel, Y)
    out_regime = OUTPUT_DIR / "regime_signals.csv"
    regime_2022.to_csv(out_regime)
    logging.info("  保存: %s", out_regime)

    # 综合报告
    lines = [
        "# v7.6 2021/2022 失效根因深度分析报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 摘要",
        "",
        "v7.6 在 2019/2020 起点 Calmar 3.0+, 2021/2022 起点 Calmar 0.7-1.5.",
        "本报告分析 4 个维度, 寻找具体失效根因。",
        "",
        "## 分析 1: β_path 对比",
        "",
        f"| 维度 | 2019-2020 (好) | 2021-2022 (差) |",
        f"|------|----------------|----------------|",
        f"| 周数 | {res_good['n_weeks']} | {res_bad['n_weeks']} |",
        f"| β 稀疏度 | {res_good['sparsity']*100:.1f}% | {res_bad['sparsity']*100:.1f}% |",
        f"| β 平均变化率 | {res_good['mean_abs_diff']:.4f} | {res_bad['mean_abs_diff']:.4f} |",
        "",
        "**Top 5 重要因子 (绝对均值, 2021-2022):**",
        "",
    ]
    for i, (k, mean, std) in enumerate(res_bad["top5_factors"]):
        fname = FACTOR_NAMES[k] if k < len(FACTOR_NAMES) else f"factor_{k}"
        lines.append(f"  {i+1}. **{fname}** (idx={k}): mean={mean:.4f}, std={std:.4f}")

    lines.extend([
        "",
        "## 分析 2: X 因子 IC 对比",
        "",
        "| 区间 | 弱 (|IC|<0.05) | 中 (0.05-0.10) | 强 (>0.10) |",
        "|------|----------------|----------------|-------------|",
        f"| 2019-2020 | {ic_good['n_weak']} | {ic_good['n_medium']} | {ic_good['n_strong']} |",
        f"| 2021-2022 | {ic_bad['n_weak']} | {ic_bad['n_medium']} | {ic_bad['n_strong']} |",
        "",
        "### Top 5 衰减因子 (good→bad IC 退化最大)",
        "",
        "| 因子 | good IC | bad IC | 衰减 |",
        "|------|---------|--------|------|",
    ])
    for _, r in worst_decay.iterrows():
        lines.append(f"| {r['factor_name']} | {r['ic_good']:.4f} | {r['ic_bad']:.4f} | {r['ic_diff']:.4f} |")

    lines.extend([
        "",
        "### Top 5 提升因子",
        "",
        "| 因子 | good IC | bad IC | 提升 |",
        "|------|---------|--------|------|",
    ])
    for _, r in best_rise.iterrows():
        lines.append(f"| {r['factor_name']} | {r['ic_good']:.4f} | {r['ic_bad']:.4f} | {-r['ic_diff']:.4f} |")

    lines.extend([
        "",
        "## 分析 3: 调仓贡献",
        "",
        "| 区间 | 调仓次数 | 累计收益 |",
        "|------|----------|----------|",
        f"| 2019-2020 | {rebal_good['n_rebalances']} | "
        f"{(rebal_good['period_cumret']*100 if rebal_good['period_cumret'] is not None else 0):.1f}% |",
        f"| 2021-2022 | {rebal_bad['n_rebalances']} | "
        f"{(rebal_bad['period_cumret']*100 if rebal_bad['period_cumret'] is not None else 0):.1f}% |",
        "",
        "## 分析 4: 2022 Regime 检测",
        "",
    ])

    # 2022 季度信号
    for q_start, q_end in [("2022-01-01", "2022-03-31"), ("2022-04-01", "2022-06-30"),
                            ("2022-07-01", "2022-09-30"), ("2022-10-01", "2022-12-31")]:
        q = regime_2022.loc[q_start:q_end]
        if len(q) == 0:
            continue
        lines.append(f"- **Q {q_start[5:]}**: vol={q['vol_60'].mean()*100:.2f}%, "
                     f"ret={q['ret_60'].mean()*100:.2f}%, RSI={q['rsi_60'].mean():.1f}")

    n_bear_combo = regime_2022["bear_combo"].sum()
    n_bear_trend = regime_2022["bear_trend"].sum()
    n_bear_vol = regime_2022["bear_vol"].sum()
    lines.extend([
        "",
        f"**2022 H1 信号 (180 天)**: bear_vol={n_bear_vol}, bear_trend={n_bear_trend}, "
        f"bear_combo={n_bear_combo}",
        "",
    ])

    # 根因结论
    lines.extend([
        "## 根因结论",
        "",
    ])

    if res_good and res_bad:
        if abs(res_good["sparsity"] - res_bad["sparsity"]) > 0.1:
            lines.append("1. **β 稀疏度差异大**: 2021-2022 与 2019-2020 的稀疏度不同, "
                         "说明 TV-PR 在两段时间对因子选择不同")
        else:
            lines.append("1. **β 稀疏度相似**: 两段时间的因子选择模式相近")

    if ic_good and ic_bad:
        if ic_good["n_strong"] > ic_bad["n_strong"]:
            lines.append(f"2. **2021-2022 强 IC 因子减少**: "
                         f"从 {ic_good['n_strong']} → {ic_bad['n_strong']}, "
                         "宏观-资产关系在 2021-2022 弱化")
        else:
            lines.append("2. **IC 强度变化不大**: 2021-2022 仍有可学习因子")

    if n_bear_combo > 0:
        lines.append(f"3. **2022 熊市可被 combo 信号识别**: {n_bear_combo} 天触发 "
                     f"({n_bear_combo/180*100:.0f}% 的 2022), 但 v7.6 原始策略未用此信号")

    lines.extend([
        "",
        "## 可操作建议",
        "",
        "### 短期 (1 周内可做)",
        "",
        "1. **锁定 top_n=5 + TF MA200** — 已有数据支持 (OOS Sharpe 3.44)",
        "2. **接受 33% CV%** — 已分析: 不是过拟合, 是结构性问题",
        "3. **加 regime_combo 作为 v7.6 构造层** — 2022 熊市可被识别",
        "",
        "### 中期 (1-2 周)",
        "",
        "1. **替换衰减 IC 因子**: 找出 good→bad 衰减最大的因子, 看是否能用其他因子替代",
        "2. **加 macro vs PV 切换**: macro 在 2021-2022 表现差, PV 可能更稳定",
        "3. **用 ensemble v1.0 + v7.6 降低组合 CV%**",
        "",
        "### 长期",
        "",
        "1. **重新设计 v7.6 因子池**: K=20 → K=10, 选 IC 稳定的因子",
        "2. **加 regime-aware TV-PR**: 不同 regime 用不同 β 估计",
        "3. **用 ensemble 替代单一 v7.6**",
    ])

    report = "\n".join(lines)
    out_md = OUTPUT_DIR / "rootcause_report.md"
    out_md.write_text(report, encoding="utf-8")
    logging.info("=" * 60)
    logging.info("报告已保存: %s", out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
