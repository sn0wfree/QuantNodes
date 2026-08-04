"""v10.3 K方案 DD参数优化 — Optuna搜索最优DD-mapper参数."""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
    load_navs, compute_nav, scheme_i_blended,
    V10_3G_WEEKLY_MIN_CHANGE, _apply_constraints, _compute_target_weights,
    _get_risk_model, _get_all_risk_probs, _apply_risk_layers,
    blended_min_defensive, V10_3G_TREND_WINDOW, V10_3G_TREND_THRESH, V10_3G_HYSTERESIS,
    BASE_WEIGHTS)
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

log = print


def dd_mult_param(dd: float, bp: list[float], vals: list[float]) -> float:
    """参数化DD-mapper: piecewise linear interpolation.

    Args:
        dd: 当前回撤
        bp: 断点 [0, bp1, bp2, bp3, bp4]
        vals: 对应值 [1.0, v1, v2, v3, v4]
    """
    if dd <= bp[0]:
        return vals[0]
    for i in range(1, len(bp)):
        if dd <= bp[i]:
            t = (dd - bp[i-1]) / (bp[i] - bp[i-1])
            return vals[i-1] + t * (vals[i] - vals[i-1])
    return vals[-1]


def scheme_k_optimized(prices: pd.DataFrame, bp: list[float], vals: list[float],
                        hyst: float = 0.05) -> pd.DataFrame:
    """K方案: 参数化DD-mapper + hysteresis."""
    i_w = scheme_i_blended(prices)
    weights = i_w.copy()
    i_nav = compute_nav(prices, i_w, cost_bp=10)
    weekly_dates = set(prices.index[4::5])
    prev_mult = 1.0

    for i in range(252, len(prices)):
        date = prices.index[i]
        if date in weekly_dates:
            nav_val = i_nav.iloc[i]
            peak = i_nav.iloc[:i+1].max()
            dd = (peak - nav_val) / peak if peak > 0 else 0
            new_mult = dd_mult_param(dd, bp, vals)
            new_mult = max(vals[-1], min(vals[0], new_mult))
            if abs(new_mult - prev_mult) > hyst:
                prev_mult = new_mult
            weights.loc[date] = i_w.loc[date] * prev_mult
    return weights


def objective(trial: optuna.Trial, prices: pd.DataFrame, train_slice: slice) -> float:
    """Optuna目标: 最大化训练集Sharpe (含成本 + 换手惩罚)."""
    # 优化断点 (确保单调递增)
    bp1 = trial.suggest_float("bp1", 0.02, 0.05)
    bp2 = trial.suggest_float("bp2", bp1 + 0.01, 0.07)
    bp3 = trial.suggest_float("bp3", bp2 + 0.01, 0.10)
    bp4 = trial.suggest_float("bp4", bp3 + 0.01, 0.15)
    bp = [0, bp1, bp2, bp3, bp4]

    # 优化值 (单调递减)
    v1 = trial.suggest_float("v1", 0.85, 1.0)
    v2 = trial.suggest_float("v2", 0.60, v1 - 0.05)
    v3 = trial.suggest_float("v3", 0.30, v2 - 0.05)
    v4 = trial.suggest_float("v4", 0.10, v3 - 0.05)
    vals = [1.0, v1, v2, v3, v4]

    hyst = trial.suggest_float("hyst", 0.01, 0.10)

    try:
        w = scheme_k_optimized(prices, bp, vals, hyst)
        w_train = w.iloc[train_slice]
        # 含成本的 NAV
        nav_train = compute_nav(prices.iloc[train_slice], w_train, cost_bp=10)
        m = compute_metrics(nav_train)
        sharpe = m.get("Sharpe", 0)

        # 换手惩罚: 每100%换手惩罚0.05 Sharpe
        turnover = w_train.diff().abs().sum(axis=1).mean() * 252
        cost_penalty = turnover * 0.0005
        sharpe -= cost_penalty

        # MaxDD 惩罚
        maxdd = abs(m.get("MaxDD", -1))
        if maxdd > 0.10:
            sharpe *= 0.5

        return sharpe
    except Exception:
        return -10.0


def main():
    log("=" * 80)
    log("  v10.3 K方案 — DD参数优化 (Optuna)")
    log("=" * 80)

    prices = load_navs()
    log(f"Loaded {len(prices)} days")

    # Walk-forward: 训练集=前70%, 测试集=后30%
    n = len(prices)
    train_end = int(n * 0.7)
    train_slice = slice(0, train_end)

    log(f"\n训练集: {prices.index[0].date()} ~ {prices.index[train_end].date()} ({train_end}天)")
    log(f"测试集: {prices.index[train_end].date()} ~ {prices.index[-1].date()} ({n - train_end}天)")

    # Optuna优化
    log("\n[1] Optuna搜索 (50 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, prices, train_slice), n_trials=50, show_progress_bar=False)

    best = study.best_params
    bp1, bp2, bp3, bp4 = best["bp1"], best["bp2"], best["bp3"], best["bp4"]
    v1, v2, v3, v4 = best["v1"], best["v2"], best["v3"], best["v4"]
    hyst = best["hyst"]

    bp = [0, bp1, bp2, bp3, bp4]
    vals = [1.0, v1, v2, v3, v4]

    log(f"\n最优参数:")
    log(f"  断点: {[f'{x:.2%}' for x in bp]}")
    log(f"  值: {[f'{x:.2f}' for x in vals]}")
    log(f"  Hysteresis: {hyst:.2%}")
    log(f"  训练集Sharpe: {study.best_value:.3f}")

    # 全周期回测
    log("\n[2] 全周期回测...")
    from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import scheme_g_trend_adaptive
    for cost_bp in [0, 5, 10]:
        log(f"\n  成本={cost_bp}bp:")
        log(f"  {'Scheme':<12} {'AnnRet':<10} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<10} {'Turnover':<12}")
        log(f"  {'-'*70}")

        for name, func, args in [
            ("G", scheme_g_trend_adaptive, ()),
            ("K_current", lambda p: scheme_k_optimized(p, [0,0.04,0.06,0.07,0.10], [1.0,0.9,0.8,0.7,0.3]), ()),
            ("K_optimized", scheme_k_optimized, (bp, vals, hyst)),
        ]:
            try:
                if args:
                    w = func(prices, *args)
                else:
                    w = func(prices)
                nav = compute_nav(prices, w, cost_bp=cost_bp)
                m = compute_metrics(nav)
                turnover = w.diff().abs().sum(axis=1).mean() * 252
                log(f"  {name:<12} {m['AnnRet']:<10.2%} {m['Sharpe']:<10.3f} {m['Sortino']:<10.3f} {m['MaxDD']:<10.2%} {turnover:<12.0%}")
            except Exception as e:
                log(f"  {name:<12} Error: {e}")

    # 分年度
    log(f"\n[3] 分年度 (成本=10bp):")
    log(f"  {'Year':<8}", end="")
    for name in ["G", "K_current", "K_optimized"]:
        log(f"{name:<16}", end="")
    log()
    log(f"  {'-'*56}")

    w_opt = scheme_k_optimized(prices, bp, vals, hyst)
    w_cur = scheme_k_optimized(prices, [0,0.04,0.06,0.07,0.10], [1.0,0.9,0.8,0.7,0.3])
    nav_opt = compute_nav(prices, w_opt, cost_bp=10)
    nav_cur = compute_nav(prices, w_cur, cost_bp=10)
    nav_g = compute_nav(prices, scheme_g_trend_adaptive(prices), cost_bp=10)

    for year in [2022, 2023, 2024, 2025, 2026]:
        row = f"  {year:<8}"
        for nav in [nav_g, nav_cur, nav_opt]:
            yearly = nav.loc[f'{year}']
            if len(yearly) < 2:
                row += f"{'N/A':<16}"
                continue
            m = compute_metrics(yearly)
            row += f"S={m['Sharpe']:.3f} DD={m['MaxDD']:.2%}  "
        log(row)

    # 保存最优参数
    out = pd.DataFrame([{"bp": bp, "vals": vals, "hyst": hyst,
                          "train_sharpe": study.best_value}])
    out_path = REPO / "reports/momentum_etf_rotation/v10_3/k_optimized_params.csv"
    out.to_csv(out_path, index=False)
    log(f"\n最优参数已保存: {out_path}")


if __name__ == "__main__":
    main()
