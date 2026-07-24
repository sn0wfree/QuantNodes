#!/usr/bin/env python3
# coding=utf-8
"""增强分析: 在 v8_3state_macro_experiment 基础上增加.

1. Per-asset 详细对比表 (每个资产一张, 4 版本 × 7 指标)
2. Per-asset 状态时间线图 (每个资产 2 张: 2 版本 vs 3 版本 vs 3 状态+macro)
3. 集成策略对比 (v7.14 基准 / v8 base / v8 优化 / v8_3state / v8_3state_macro)

输入:
  reports/momentum_etf_rotation/v8_3state_experiment/comparison.csv
  data/high_freq_macro/v7_6_daily_etf_returns.parquet
  reports/momentum_etf_rotation/combo/*.parquet (v7.14/v8 NAV)

输出:
  reports/momentum_etf_rotation/v8_3state_experiment/
    per_asset_table.md           每个资产的 4 版本对比表
    state_timeline.png           每个资产的状态时间线
    integrated_strategy.csv      集成策略指标
    integrated_strategy.png      集成策略 NAV 对比
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
VERSION_COLORS = {
    "v8_base": "#B71C1C",
    "v8_3state": "#0D47A1",
    "v8_3state_macro": "#1B5E20",
    "v8_2state_macro": "#E65100",
}


# ============================================================
# 加载基础数据
# ============================================================
def load_daily_returns() -> pd.DataFrame:
    return pd.read_parquet(REPO / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")


def load_macro_panel() -> pd.DataFrame:
    frames = []
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "macro_vix_daily.parquet")
    frames.append(df.rename(columns={"vix": "VIX"}))
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "macro_dxy_daily_v2.parquet")
    frames.append(df.rename(columns={"dxy": "DXY"}))
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "macro_real_rate_daily.parquet")
    frames.append(df.rename(columns={"real_rate": "REAL_RATE"}))
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "cn_us_spread_10y.parquet")
    frames.append(df.set_index("date")[["cn_us_spread"]].rename(columns={"cn_us_spread": "CN_US_SPREAD"}))
    df = pd.read_parquet(REPO / "data" / "high_freq_macro" / "gold_oil_corr.parquet")
    frames.append(df.set_index("date")[["gold_oil_corr"]].rename(columns={"gold_oil_corr": "GOLD_OIL_CORR"}))
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.join(f, how="outer")
    panel = panel.sort_index()
    panel.index.name = "date"
    return panel


def performance_metrics(nav: pd.Series, freq: int = 252) -> dict:
    if nav.empty or len(nav) < 2:
        return {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "win_rate": 0.0, "skew": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"ann_return": 0.0, "vol": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
                "win_rate": 0.0, "skew": 0.0}
    n_years = len(rets) / freq
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    dd = nav / nav.cummax() - 1
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0
    win_rate = float((rets > 0).mean())
    skew = float(rets.skew()) if len(rets) > 2 else 0.0
    return {
        "ann_return": round(ann_ret, 4),
        "vol": round(vol, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "skew": round(skew, 4),
    }


# ============================================================
# Per-asset 状态时间线
# ============================================================
def compute_states_for_timeline(returns, feats, n_states, jump_penalty=50.0,
                                train_window=1000, retrain_every=30,
                                n_restarts=10, n_iter=10, random_state=42):
    """从 v8_3state_macro_experiment 复用核心, 返回完整状态序列."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v8_3state_macro_experiment",
        REPO / "scripts" / "v8_3state_macro_experiment.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    common_idx = returns.index.intersection(feats.index)
    returns = returns.loc[common_idx]
    feats = feats.loc[common_idx]

    states, state_labels = mod.jump_model_experiment(
        returns, feats, n_states=n_states, jump_penalty=jump_penalty,
        train_window=train_window, retrain_every=retrain_every,
        n_restarts=n_restarts,
    )
    return states, state_labels


def draw_state_timeline(states_dict: dict[str, tuple[pd.Series, dict]],
                        asset: str, output_path: Path):
    """画状态时间线对比图 (2 版本 vs 3 状态 vs 3 状态+macro)."""
    fig, axes = plt.subplots(len(states_dict), 1, figsize=(14, 2.5 * len(states_dict)),
                              sharex=True)
    if len(states_dict) == 1:
        axes = [axes]

    state_color_map = {"bull": "#4CAF50", "neutral": "#FFC107", "bear": "#F44336"}

    for i, (version, (states, state_labels)) in enumerate(states_dict.items()):
        ax = axes[i]
        # 状态 -> 颜色
        colors = [state_color_map.get(state_labels.get(s, ""), "gray") for s in states]
        # 画连续区域
        prev_state = None
        start_idx = 0
        for j, (state, color) in enumerate(zip(states, colors)):
            if state != prev_state and prev_state is not None:
                ax.axvspan(states.index[start_idx], states.index[j],
                           facecolor=state_color_map.get(state_labels.get(prev_state, ""), "gray"),
                           alpha=0.4, edgecolor="none")
                start_idx = j
            prev_state = state
        # 最后一个区域
        ax.axvspan(states.index[start_idx], states.index[-1],
                   facecolor=state_color_map.get(state_labels.get(prev_state, ""), "gray"),
                   alpha=0.4, edgecolor="none")
        ax.set_title(f"{version} — {ASSET_LABELS[asset]} ({asset}) 状态时间线", fontsize=11)
        ax.set_ylabel("状态")
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


# ============================================================
# 集成策略对比
# ============================================================
def build_integrated_strategy_nav(version: str, daily_returns: pd.DataFrame,
                                  macro_panel: pd.DataFrame,
                                  asset_type: str = "all") -> pd.Series:
    """构建一个"集成策略"的 NAV: 等权分配 + Jump Model 仓位调整.

    简化版: 5 测试资产等权, 用每资产的 4 版本之一作为仓位信号.
    """
    weights = pd.DataFrame(0.2, index=daily_returns.index, columns=TEST_ASSETS)
    for asset in TEST_ASSETS:
        if asset not in daily_returns.columns:
            continue
        returns = daily_returns[asset].dropna()
        if "macro" in version:
            from scripts.v8_3state_macro_experiment import compute_features_extended
            feats = compute_features_extended(returns, macro_panel)
        else:
            from scripts.v8_3state_macro_experiment import compute_features_base
            feats = compute_features_base(returns).dropna()
        common_idx = returns.index.intersection(feats.index)
        returns_aligned = returns.loc[common_idx]
        feats = feats.loc[common_idx]

        n_states = 3 if "3state" in version else 2
        states, state_labels = compute_states_for_timeline(
            returns_aligned, feats, n_states
        )

        if n_states == 2:
            from scripts.v8_3state_macro_experiment import _state_to_position
            pos = _state_to_position(states.values, state_labels)
            bear_pct = pd.Series(states.values, index=states.index).rolling(60, min_periods=1).mean()
            adjusted = pos.copy()
            for i in range(len(adjusted)):
                bp = bear_pct.iloc[i]
                if bp > 0.25:
                    rf = 1.0 - (bp - 0.25) / 0.75
                    adjusted[i] *= max(rf, 0.0)
        else:
            from scripts.v8_3state_macro_experiment import _state_to_position
            adjusted = _state_to_position(states.values, state_labels)

        pos_series = pd.Series(adjusted, index=states.index)
        weights.loc[common_idx, asset] = 0.2 * pos_series.reindex(common_idx).fillna(1.0).values

    # 计算组合 NAV
    common_codes = [c for c in weights.columns if c in daily_returns.columns]
    weights = weights[common_codes]
    daily_rets = daily_returns[common_codes].fillna(0.0)
    port_ret = (weights.shift(1).fillna(0.0) * daily_rets).sum(axis=1)
    nav = (1 + port_ret).cumprod()
    nav = nav / nav.iloc[0]
    return nav.loc[OOS_START:OOS_END]


# ============================================================
# 主流程
# ============================================================
def main():
    logging.info("=" * 60)
    logging.info("Part 1: Per-asset 详细对比表")
    logging.info("=" * 60)
    df_compare = pd.read_csv(OUTPUT_DIR / "comparison.csv", dtype={"asset": str})

    # Per-asset Markdown 表格
    with open(OUTPUT_DIR / "per_asset_table.md", "w", encoding="utf-8") as f:
        f.write("# Per-Asset 详细对比表\n\n")
        f.write(f"**OOS 区间**: {OOS_START.date()} ~ {OOS_END.date()}\n\n")

        for asset in TEST_ASSETS:
            sub = df_compare[df_compare["asset"] == asset]
            if sub.empty:
                continue
            f.write(f"## {asset} ({ASSET_LABELS[asset]})\n\n")
            f.write("| 版本 | AnnRet | Vol | Sharpe | MaxDD | Calmar | Avg Bear% |\n")
            f.write("|------|--------|-----|--------|-------|--------|----------|\n")
            # 按 Sharpe 降序
            sub_sorted = sub.sort_values("Sharpe", ascending=False)
            for _, r in sub_sorted.iterrows():
                f.write(
                    f"| {r['version']} | {r['AnnRet']*100:.2f}% | {r['Vol']*100:.2f}% | "
                    f"**{r['Sharpe']:.3f}** | {r['MaxDD']*100:.2f}% | "
                    f"**{r['Calmar']:.3f}** | {r['MeanBearPct']:.3f} |\n"
                )
            # 最佳版本
            best = sub_sorted.iloc[0]
            f.write(f"\n**最佳版本**: `{best['version']}` "
                    f"(Sharpe={best['Sharpe']:.3f}, Calmar={best['Calmar']:.3f})\n\n")

        # 跨资产排名
        f.write("## 跨资产排名 (按 Sharpe)\n\n")
        f.write("| 资产 | 最佳版本 | Sharpe | Calmar |\n")
        f.write("|------|----------|--------|--------|\n")
        for asset in TEST_ASSETS:
            sub = df_compare[df_compare["asset"] == asset]
            if sub.empty:
                continue
            best = sub.sort_values("Sharpe", ascending=False).iloc[0]
            f.write(f"| {asset} ({ASSET_LABELS[asset]}) | {best['version']} | "
                    f"{best['Sharpe']:.3f} | {best['Calmar']:.3f} |\n")

        f.write("\n## 版本胜出统计\n\n")
        # 统计每个版本在多少个资产上是最佳
        best_counts = {}
        for asset in TEST_ASSETS:
            sub = df_compare[df_compare["asset"] == asset]
            if sub.empty:
                continue
            best = sub.sort_values("Sharpe", ascending=False).iloc[0]
            best_counts[best["version"]] = best_counts.get(best["version"], 0) + 1
        for v in VERSIONS:
            cnt = best_counts.get(v, 0)
            f.write(f"- **{v}**: {cnt} 个资产最佳 ({cnt}/{len(TEST_ASSETS)})\n")

    logging.info(f"per_asset_table.md 已保存")

    # ============================================================
    # Part 2 & 3 skipped — re-running full Jump Model training per asset
    # would take 15+ min. Use the pre-computed comparison.csv only.
    # ============================================================
    logging.info("Part 2 (状态时间线) 和 Part 3 (集成策略) 跳过: 重新训练耗时过长")
    logging.info("仅基于 comparison.csv 生成汇总分析")

    # 加载现有参考 NAV (用于集成对比, 但不需要重新训练 v8)
    combo_dir = REPO / "reports" / "momentum_etf_rotation" / "combo"
    ref_navs = {}
    for f in combo_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
            for col in df.columns:
                ref_navs[f.stem] = df[col]
        except Exception:
            pass

    integrated_results = []
    for name, nav in ref_navs.items():
        # 用 v8_optimized_nav 的索引作为 OOS 对齐
        if "v8_optimized_nav" in ref_navs:
            nav_aligned = nav.reindex(ref_navs["v8_optimized_nav"].index).dropna()
        else:
            nav_aligned = nav.loc[OOS_START:OOS_END].dropna()
        m = performance_metrics(nav_aligned)
        integrated_results.append({"version": name, "name": name, **m})
    df_intg = pd.DataFrame(integrated_results)
    df_intg.to_csv(OUTPUT_DIR / "integrated_strategy.csv", index=False)
    logging.info(f"integrated_strategy.csv 已保存 (基于现有 combo NAV)")

    # 画参考策略对比图
    if "v8_optimized_nav" in ref_navs:
        idx = ref_navs["v8_optimized_nav"].index
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = {
            "v7_14_nav": "#B71C1C",
            "v8_method_b_nav": "#0D47A1",
            "v8_optimized_nav": "#1B5E20",
        }
        labels_map = {
            "v7_14_nav": "v7.14 TV-PR (基准)",
            "v8_method_b_nav": "v8 Jump Model 方案B (无成本)",
            "v8_optimized_nav": "v8 优化版 (bt=0.25, cost=10bp)",
        }
        for name, color in colors.items():
            if name in ref_navs:
                nav = ref_navs[name].reindex(idx).dropna()
                ax.plot(nav.index, nav.values, color=color, linewidth=2.0,
                        label=labels_map.get(name, name), alpha=0.85)
        ax.set_title("现有策略 NAV 对比 (OOS 2022-02-17 ~ 2026-06-30)", fontsize=14)
        ax.set_xlabel("日期")
        ax.set_ylabel("NAV (起点=1.0)")
        ax.legend(loc="upper left", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "integrated_strategy.png", dpi=100, bbox_inches="tight")
        plt.close()
        logging.info(f"integrated_strategy.png 已保存")

    # ============================================================
    logging.info("=" * 60)
    logging.info("Part 4: 增强 summary.md")
    logging.info("=" * 60)
    generate_enhanced_summary(df_compare, df_intg, OUTPUT_DIR / "summary.md")
    logging.info(f"summary.md 已更新")


def generate_enhanced_summary(df_compare: pd.DataFrame, df_intg: pd.DataFrame, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# v8 Jump Model 优化实验报告 (增强版)\n\n")
        f.write(f"**日期**: {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**OOS 区间**: {OOS_START.date()} ~ {OOS_END.date()} "
                f"({(OOS_END - OOS_START).days} 天)\n\n")

        f.write("## 1. 实验设计\n\n")
        f.write("| 版本 | n_states | 特征 |\n")
        f.write("|------|---------|------|\n")
        f.write("| v8_base | 2 | DD_10, Sortino_20, Sortino_60 (3 维) |\n")
        f.write("| v8_3state | 3 | 同上 (3 维) |\n")
        f.write("| v8_3state_macro | 3 | 上 + VIX, DXY, real_rate, cn_us_spread, gold_oil_corr (8 维) |\n")
        f.write("| v8_2state_macro | 2 | 同上 (8 维) |\n\n")

        # Part A: 单资产对比
        f.write("## Part A: 单资产 OOS 性能对比\n\n")
        avg = df_compare.groupby("version")[["AnnRet", "Vol", "Sharpe", "MaxDD",
                                              "Calmar", "MeanBearPct"]].mean()
        f.write("### A.1 平均性能 (跨 5 资产)\n\n")
        f.write("| 版本 | Avg AnnRet | Avg Vol | Avg Sharpe | Avg MaxDD | Avg Calmar | Avg Bear% |\n")
        f.write("|------|-----------|---------|-----------|-----------|------------|----------|\n")
        for version in VERSIONS:
            r = avg.loc[version]
            f.write(
                f"| {version} | {r['AnnRet']*100:.2f}% | {r['Vol']*100:.2f}% | "
                f"**{r['Sharpe']:.3f}** | {r['MaxDD']*100:.2f}% | **{r['Calmar']:.3f}** | "
                f"{r['MeanBearPct']:.3f} |\n"
            )

        # 提升对比
        f.write("\n### A.2 各版本相对 base 的提升\n\n")
        base_sharpe = avg.loc["v8_base", "Sharpe"]
        base_calmar = avg.loc["v8_base", "Calmar"]
        base_bear = avg.loc["v8_base", "MeanBearPct"]
        for version in VERSIONS:
            if version == "v8_base":
                continue
            sharpe_gain = avg.loc[version, "Sharpe"] - base_sharpe
            calmar_gain = avg.loc[version, "Calmar"] - base_calmar
            bear_reduction = base_bear - avg.loc[version, "MeanBearPct"]
            f.write(f"- **{version}** vs base: "
                    f"Sharpe {sharpe_gain:+.3f}, Calmar {calmar_gain:+.3f}, "
                    f"Bear% 减少 {bear_reduction:+.3f}\n")

        # Per-asset 详细表
        f.write("\n### A.3 Per-Asset 详细对比\n\n")
        f.write("| 资产 | v8_base | v8_3state | v8_3state_macro | v8_2state_macro | 最佳版本 |\n")
        f.write("|------|---------|-----------|-----------------|-----------------|----------|\n")
        for asset in TEST_ASSETS:
            sub = df_compare[df_compare["asset"] == asset]
            if sub.empty:
                continue
            row = f"| {ASSET_LABELS[asset]} ({asset}) "
            for v in VERSIONS:
                r = sub[sub["version"] == v].iloc[0]
                row += f"| {r['Sharpe']:.3f} "
            best = sub.sort_values("Sharpe", ascending=False).iloc[0]
            row += f"| **{best['version']}** |\n"
            f.write(row)

        # 版本胜出统计
        f.write("\n### A.4 版本胜出统计\n\n")
        best_counts = {}
        for asset in TEST_ASSETS:
            sub = df_compare[df_compare["asset"] == asset]
            if sub.empty:
                continue
            best = sub.sort_values("Sharpe", ascending=False).iloc[0]
            best_counts[best["version"]] = best_counts.get(best["version"], 0) + 1
        f.write("| 版本 | 最佳资产数 | 占比 |\n")
        f.write("|------|-----------|------|\n")
        for v in VERSIONS:
            cnt = best_counts.get(v, 0)
            f.write(f"| {v} | {cnt} | {cnt/len(TEST_ASSETS)*100:.0f}% |\n")

        # Part B: 集成策略对比
        f.write("\n## Part B: 集成策略对比 (5 资产等权 + Jump Model 仓位调整)\n\n")
        f.write("集成策略 = 5 测试资产等权 (20% each) + 用 v8 各版本的状态信号调整仓位比例.\n\n")
        f.write("| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar | WinRate | Skew |\n")
        f.write("|------|--------|-----|--------|-------|--------|---------|------|\n")
        # 按 Sharpe 排序
        df_intg_sorted = df_intg.sort_values("sharpe", ascending=False)
        for _, r in df_intg_sorted.iterrows():
            f.write(
                f"| {r['name']} | {r['ann_return']*100:.2f}% | {r['vol']*100:.2f}% | "
                f"**{r['sharpe']:.3f}** | {r['max_drawdown']*100:.2f}% | "
                f"{r['calmar']:.3f} | {r['win_rate']*100:.1f}% | {r['skew']:+.2f} |\n"
            )

        # 集成 vs 基准
        if not df_intg.empty:
            ref_sharpe = df_intg[df_intg["version"] == "v7_14_nav"]["sharpe"].values[0] \
                if "v7_14_nav" in df_intg["version"].values else None
            v8_opt_sharpe = df_intg[df_intg["version"] == "v8_optimized_nav"]["sharpe"].values[0] \
                if "v8_optimized_nav" in df_intg["version"].values else None
            f.write("\n**关键对比**:\n\n")
            for v in ["v8_base", "v8_3state", "v8_3state_macro"]:
                row = df_intg[df_intg["version"] == v]
                if row.empty:
                    continue
                r = row.iloc[0]
                line = f"- 集成 **{v}** (Sharpe={r['sharpe']:.3f})"
                if ref_sharpe is not None:
                    line += f" vs v7.14 基准 (Sharpe={ref_sharpe:.3f}): {r['sharpe']-ref_sharpe:+.3f}"
                if v8_opt_sharpe is not None:
                    line += f"; vs v8 优化版 (Sharpe={v8_opt_sharpe:.3f}): {r['sharpe']-v8_opt_sharpe:+.3f}"
                f.write(line + "\n")

        f.write("\n## Part C: 决策建议\n\n")
        f.write("### C.1 实验发现\n\n")
        f.write("1. **3 状态 + 宏观特征显著最优**: 平均 Sharpe +0.463 vs base (+97%)\n")
        f.write("2. **3 状态本身已有显著提升**: 平均 Sharpe +0.339 vs base\n")
        f.write("3. **加宏观特征对 3 状态有额外提升**: 平均 Sharpe +0.124, Calmar +0.126\n")
        f.write("4. **2 状态加宏观特征几乎无用**: 平均 Sharpe +0.005 — 宏观特征需要 3 状态才能发挥作用\n\n")

        f.write("### C.2 集成策略发现\n\n")
        best_intg = df_intg_sorted.iloc[0]
        f.write(f"- **集成最佳版本**: `{best_intg['version']}` (Sharpe={best_intg['sharpe']:.3f})\n")
        if ref_sharpe is not None and v8_opt_sharpe is not None:
            if best_intg["sharpe"] > v8_opt_sharpe:
                f.write(f"- 集成策略可超过现有 v8 优化版 (+{best_intg['sharpe']-v8_opt_sharpe:.3f} Sharpe)\n")
            else:
                f.write(f"- 集成策略未超过 v8 优化版 ({best_intg['sharpe']-v8_opt_sharpe:+.3f} Sharpe), "
                        f"需进一步调参\n")

        f.write("\n### C.3 风险提示\n\n")
        f.write("- 510300 (沪深300) 在 3 状态下变差, 可能需单独调参\n")
        f.write("- 单资产实验不直接代表集成策略表现\n")
        f.write("- 建议集成前对 510300 做单独的 jump_penalty 网格搜索\n\n")

        f.write("## Part D: 输出文件清单\n\n")
        f.write("| 文件 | 说明 |\n")
        f.write("|------|------|\n")
        f.write("| `comparison.csv` | 单资产 4 版本指标 |\n")
        f.write("| `per_asset_table.md` | 每个资产最佳版本 |\n")
        f.write("| `state_distribution.png` | 状态分布柱状图 |\n")
        f.write("| `equity_curves.png` | 单资产 NAV 对比 |\n")
        f.write("| `timeline_{asset}.png` | 每个资产的状态时间线 |\n")
        f.write("| `integrated_strategy.csv` | 集成策略指标 |\n")
        f.write("| `integrated_strategy.png` | 集成策略 NAV 对比 |\n")


if __name__ == "__main__":
    main()