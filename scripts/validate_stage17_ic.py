# coding=utf-8
"""IC 因子择时回测验证 (Stage 17, v4.0).

跑全周期 IC 时序, 输出:
1. 6 因子 IC 均值/标准差/ICIR
2. 因子权重时序
3. 因子 vs 等权对比 (用 5 只风格组 + 7 只 Smart β ETF)
4. 月度调仓的简单 backtest: 等权 vs IC 加权 vs 子策略等权

注: 这是 IC 验证, 不是完整 v4 回测. v4 multi_strategy 留后续.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from QuantNodes.strategy.momentum_etf_rotation.v4 import (
    ALL_V4_CODES,
    FACTOR_NAMES,
    FactorTimingConfig,
    backtest_factor_timing,
    backtest_factor_weights_history,
    compute_factor_weights,
    compute_strategy_weights,
    load_smartbeta_panel,
)


def compute_ic_metrics(ic: pd.DataFrame) -> pd.DataFrame:
    """计算每个因子的 IC 指标."""
    rows = []
    for name in ic.columns:
        s = ic[name].dropna()
        if len(s) == 0:
            continue
        rows.append({
            "factor": name,
            "n": len(s),
            "mean": s.mean(),
            "std": s.std(),
            "icir": s.mean() / s.std() if s.std() > 0 else 0.0,
            "hit_rate": (s > 0).mean(),
            "min": s.min(),
            "max": s.max(),
            "skew": s.skew(),
        })
    return pd.DataFrame(rows).set_index("factor")


def simulate_simple_backtest(
    nav_df: pd.DataFrame,
    ic_history: pd.DataFrame,
    method: str = "equal",  # "equal" | "ic"
    all_codes: list[str] | None = None,
    cfg: FactorTimingConfig | None = None,
    start: str = "2020-01-01",
    end: str = "2026-06-30",
) -> pd.Series:
    """简单回测: 月度调仓, 等权 vs IC 加权 12 只 Smart β ETF.

    Args:
        nav_df: 价格面板
        ic_history: 滚动 IC 历史 (index=date, columns=factor)
        method: "equal" 等权 | "ic" 用最新 IC 加权
        all_codes: 候选 ETF codes
        cfg: 因子择时配置
        start/end: 回测范围

    Returns:
        pd.Series: NAV 序列
    """
    cfg = cfg or FactorTimingConfig()
    all_codes = all_codes or list(ALL_V4_CODES)
    valid_codes = [c for c in all_codes if c in nav_df.columns]

    panel = nav_df.loc[start:end, valid_codes]
    if panel.empty:
        return pd.Series(dtype=float)

    # 月度调仓日
    rebal_dates = (
        pd.Series(panel.index)
        .groupby(panel.index.to_period("M"))
        .max()
        .tolist()
    )

    nav = np.ones(len(panel))
    weights = None

    for i, date in enumerate(panel.index):
        if date in rebal_dates:
            if method == "equal":
                weights = {c: 1.0 / len(valid_codes) for c in valid_codes}
            elif method == "ic":
                # 取最近 IC
                if not ic_history.empty and date in ic_history.index:
                    ic_dict = ic_history.loc[date].to_dict()
                else:
                    # 用最近一期
                    if not ic_history.empty:
                        idx = ic_history.index.get_indexer([date], method="ffill")[0]
                        if idx >= 0:
                            ic_dict = ic_history.iloc[idx].to_dict()
                        else:
                            ic_dict = {n: 0.0 for n in FACTOR_NAMES}
                    else:
                        ic_dict = {n: 0.0 for n in FACTOR_NAMES}

                w = compute_factor_weights(
                    pd.DataFrame([ic_dict], index=[date]), cfg,
                )
                strat_w = compute_strategy_weights(w, cfg.factor_to_strategy)
                # 等分到子策略内 ETF
                weights = {}
                # style_rotation: 等分到 5 只风格组
                style_codes = ["510300", "510500", "159915", "588000", "510880"]
                for c in style_codes:
                    if c in valid_codes:
                        weights[c] = strat_w.get("style_rotation", 0.0) / len(style_codes)
                # smart_beta: 等分到 7 只 Smart β
                sb_codes = ["512890", "512260", "515900", "512040", "159786", "515080", "515100"]
                for c in sb_codes:
                    if c in valid_codes:
                        weights[c] = strat_w.get("smart_beta", 0.0) / len(sb_codes)

        # 计算日收益
        if weights and i > 0:
            daily_ret = 0.0
            for code, w in weights.items():
                if code in panel.columns:
                    a, b = panel[code].iloc[i], panel[code].iloc[i - 1]
                    if not pd.isna(a) and not pd.isna(b) and b != 0:
                        daily_ret += w * (a / b - 1)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        elif i == 0:
            nav[i] = 1.0
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=panel.index, name=method)


def main():
    print("=" * 60)
    print("IC 因子择时回测验证 (Stage 17, v4.0) — 多参数对比")
    print("=" * 60)

    panel = load_smartbeta_panel()
    print(f"\n数据: {panel.shape[0]} 天 × {panel.shape[1]} ETF")
    print(f"范围: {panel.index[0].date()} ~ {panel.index[-1].date()}")
    print(f"ETF codes: {list(ALL_V4_CODES)}")

    # 1. 跑滚动 IC
    print("\n[1/4] 计算滚动 IC ...")
    cfg = FactorTimingConfig(
        ic_window=60, forward_window=20, ic_step=5,
        lookback=60, smooth_window=12,
    )
    ic_history = backtest_factor_timing(
        panel, list(ALL_V4_CODES), cfg,
        start="2020-01-01", end="2026-06-30",
    )
    print(f"  IC shape: {ic_history.shape}")
    print(f"  IC 范围: {ic_history.index[0].date()} ~ {ic_history.index[-1].date()}")

    # 2. IC 指标
    print("\n[2/4] 因子 IC 统计:")
    ic_metrics = compute_ic_metrics(ic_history)
    print(ic_metrics.round(4).to_string())

    # 3. 因子权重时序
    print("\n[3/4] 因子权重时序 (采样):")
    weights_hist = backtest_factor_weights_history(
        panel, list(ALL_V4_CODES), cfg,
        start="2020-01-01", end="2026-06-30",
    )
    print(f"  权重 shape: {weights_hist.shape}")
    if not weights_hist.empty:
        print(f"  起始权重 ({weights_hist.index[0].date()}):")
        print("    " + ", ".join(f"{k}={v:.3f}" for k, v in weights_hist.iloc[0].round(3).to_dict().items()))
        print(f"  末尾权重 ({weights_hist.index[-1].date()}):")
        print("    " + ", ".join(f"{k}={v:.3f}" for k, v in weights_hist.iloc[-1].round(3).to_dict().items()))
        print("  权重均值:")
        print("    " + ", ".join(f"{k}={v:.3f}" for k, v in weights_hist.mean().round(3).to_dict().items()))

    # 4. 多参数对比
    print("\n[4/4] 多参数回测对比 (12 Smart β ETF, 2020-2026):")
    print()

    # 等权 baseline
    nav_equal = simulate_simple_backtest(
        panel, ic_history, method="equal",
        start="2020-01-01", end="2026-06-30",
    )

    # 不同 IC 加权参数
    test_configs = [
        ("IC_aggressive",  {"base": 0.0,  "power": 3.0, "min_weight": 0.0}),
        ("IC_default",     {"base": 0.05, "power": 2.0, "min_weight": 0.05}),
        ("IC_soft",        {"base": 0.10, "power": 1.5, "min_weight": 0.10}),
        ("IC_long_window", {"base": 0.05, "power": 2.0, "min_weight": 0.05, "ic_window": 120, "smooth_window": 24}),
        ("IC_short_fwd",   {"base": 0.05, "power": 2.0, "min_weight": 0.05, "forward_window": 10}),
        ("IC_long_fwd",    {"base": 0.05, "power": 2.0, "min_weight": 0.05, "forward_window": 40}),
    ]

    results = {"equal": metrics(nav_equal)}

    for name, params in test_configs:
        # 用新参数重算 IC
        cfg_test = FactorTimingConfig(
            ic_window=params.get("ic_window", 60),
            forward_window=params.get("forward_window", 20),
            ic_step=5, lookback=60,
            smooth_window=params.get("smooth_window", 12),
            base=params.get("base", 0.05),
            power=params.get("power", 2.0),
            min_weight=params.get("min_weight", 0.05),
        )
        ic_test = backtest_factor_timing(
            panel, list(ALL_V4_CODES), cfg_test,
            start="2020-01-01", end="2026-06-30",
        )
        nav = simulate_simple_backtest(
            panel, ic_test, method="ic", cfg=cfg_test,
            start="2020-01-01", end="2026-06-30",
        )
        results[name] = metrics(nav)

    # 打印对比表
    print(f"  {'配置':<18} {'Sharpe':>8} {'Calmar':>8} {'AnnRet':>8} {'DD':>8} {'Nav':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for name, m in results.items():
        print(f"  {name:<18} {m['sharpe']:>8.3f} {m['calmar']:>8.3f} {m['ann_return']:>8.3%} {m['max_dd']:>8.3%} {m['final_nav']:>8.3f}")

    # 找出最佳
    best_name = max(results, key=lambda k: results[k]['calmar'])
    print(f"\n  最佳 (按 Calmar): {best_name}, Calmar={results[best_name]['calmar']:.3f}")

    # 落盘
    out_dir = Path("reports/momentum_etf_rotation/v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_history.to_parquet(out_dir / "ic_history.parquet")
    weights_hist.to_parquet(out_dir / "factor_weights.parquet")
    ic_metrics.to_csv(out_dir / "ic_metrics.csv")

    # 多参数结果
    pd.DataFrame(results).T.to_csv(out_dir / "ic_param_comparison.csv")

    # 全部 NAV
    nav_all = {"equal": nav_equal}
    for name, params in test_configs:
        cfg_test = FactorTimingConfig(
            ic_window=params.get("ic_window", 60),
            forward_window=params.get("forward_window", 20),
            ic_step=5, lookback=60,
            smooth_window=params.get("smooth_window", 12),
            base=params.get("base", 0.05),
            power=params.get("power", 2.0),
            min_weight=params.get("min_weight", 0.05),
        )
        ic_test = backtest_factor_timing(
            panel, list(ALL_V4_CODES), cfg_test,
            start="2020-01-01", end="2026-06-30",
        )
        nav_all[name] = simulate_simple_backtest(
            panel, ic_test, method="ic", cfg=cfg_test,
            start="2020-01-01", end="2026-06-30",
        )
    pd.DataFrame(nav_all).to_parquet(out_dir / "ic_simple_backtest.parquet")

    summary = {
        "config": {
            "ic_window": cfg.ic_window,
            "forward_window": cfg.forward_window,
            "ic_step": cfg.ic_step,
            "lookback": cfg.lookback,
            "smooth_window": cfg.smooth_window,
            "base": cfg.base,
            "power": cfg.power,
        },
        "ic_metrics": ic_metrics.round(4).to_dict(orient="index"),
        "multi_param_comparison": results,
        "best_config": best_name,
    }
    with open(out_dir / "ic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n落盘: {out_dir}/ic_*.{{parquet,csv,json}}")
    print("\n完成!")


def metrics(nav: pd.Series) -> dict:
    """计算 NAV 关键指标."""
    n = len(nav)
    if n < 2:
        return {}
    ann_ret = nav.iloc[-1] ** (252 / n) - 1
    daily_ret = nav.pct_change().dropna()
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    return {
        "final_nav": float(nav.iloc[-1]),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
    }


if __name__ == "__main__":
    main()
