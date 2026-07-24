# coding=utf-8
"""Nagel 风格 LW 因子择时回测 — 对比 v4 当前.

测试:
1. v4 当前 (IC^2, 无收缩) - baseline
2. LW with 固定 λ (扫描 0.01-100)
3. LW with 滚动验证 λ
4. LW with regime + 滚动 λ
5. OOS walk-forward (2018-2021 train, 2022-2026 test)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v4.lw_factor_timing_integration import (
    LWFactorTimingConfig,
    compute_lw_factor_weights,
    compute_lw_weights,
    select_lambda_for_date,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.factor_timing_v4 import (
    FactorTimingConfig,
    aggregate_factor_to_etf,
    backtest_factor_timing,
    compute_factor_weights,
    get_active_factors,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import (
    ALL_V4_CODES,
    load_smartbeta_panel,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.style_rotation_v4 import classify_regime

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"


def ann_return(nav):
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav):
    pk = nav.cummax()
    return float((nav / pk - 1.0).min())


def sharpe(rets):
    if rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(252))


def metrics(nav):
    rets = nav.pct_change().dropna()
    ar = ann_return(nav)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "sharpe": sharpe(rets),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
    }


def backtest_lw_factor_timing(
    panel: pd.DataFrame,
    cfg: LWFactorTimingConfig,
    fixed_lambda: float | None = None,
    use_regime: bool = True,
    use_rolling_lambda: bool = True,
    freq: str = "M",
) -> tuple[pd.Series, list[dict]]:
    """LW 因子择时回测.

    Args:
        fixed_lambda: 如果指定, 用固定 λ (不滚动); 否则滚动选
        use_regime: 是否 regime-conditioned
        use_rolling_lambda: 是否滚动选 λ
    """
    dates = panel.index
    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    base_cfg = FactorTimingConfig(
        use_low_vol=cfg.use_low_vol,
        factor_fw=cfg.factor_fw,
        lookback=cfg.factor_lookback,
    )
    ic_hist = backtest_factor_timing(panel, list(ALL_V4_CODES), base_cfg, START, END)
    if ic_hist.empty:
        return pd.Series(np.ones(len(dates)), index=dates), []

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    log_rows = []

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > cfg.min_history:
            if date not in ic_hist.index:
                idx = ic_hist.index.get_indexer([date], method="ffill")[0]
                if idx < 0:
                    continue
                date_ic = ic_hist.index[idx]
            else:
                date_ic = date
                idx = ic_hist.index.get_loc(date)

            ic_so_far = ic_hist.iloc[:idx + 1]
            if len(ic_so_far) < 12:
                continue

            regime = "sideways"
            if use_regime:
                regime = classify_regime(panel, date)

            if fixed_lambda is not None:
                lam = fixed_lambda
            elif use_rolling_lambda:
                lam = select_lambda_for_date(ic_so_far, date_ic, cfg)
            else:
                lam = 1.0

            etf_w, f_w, _ = compute_lw_factor_weights(
                ic_so_far[list(cfg.regime_factors.get(regime, ic_so_far.columns))] if use_regime else ic_so_far,
                cfg, regime=regime, as_of=date_ic,
            )

            if fixed_lambda is not None:
                active = [f for f in cfg.regime_factors.get(regime, ()) if f in ic_so_far.columns]
                if not active:
                    active = list(ic_so_far.columns)
                if len(active) >= 2:
                    f_w_local, _ = compute_lw_weights(ic_so_far[active], cfg, lam=fixed_lambda)
                    l1 = sum(abs(w) for w in f_w_local.values())
                    if l1 > 1e-12:
                        f_w_local = {k: v / l1 * cfg.l1_norm for k, v in f_w_local.items()}
                    etf_w = _aggregate_lw(f_w_local, cfg)

            if etf_w:
                last_weights = etf_w
                log_rows.append({
                    "date": date,
                    "weights": dict(etf_w),
                    "regime": regime,
                    "lambda": lam,
                })

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in panel.columns:
                    r = panel[code].iloc[i] / panel[code].iloc[i - 1] - 1.0
                    if not np.isnan(r):
                        daily_ret += w * r
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="lw_nav"), log_rows


def _aggregate_lw(factor_weights, cfg):
    out = {}
    for fac, w in factor_weights.items():
        if w <= 0:
            continue
        codes = cfg.factor_to_etf.get(fac, ())
        if not codes:
            continue
        per_etf = w / len(codes)
        for c in codes:
            out[c] = out.get(c, 0.0) + per_etf
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def backtest_v4_baseline(panel, freq="M"):
    """v4 当前因子择时 baseline (IC^2 无收缩)."""
    cfg = FactorTimingConfig()
    dates = panel.index
    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    ic_hist = backtest_factor_timing(panel, list(ALL_V4_CODES), cfg, START, END)
    if ic_hist.empty:
        return pd.Series(np.ones(len(dates)), index=dates)

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            idx = ic_hist.index.get_indexer([date], method="ffill")[0]
            if idx < 0:
                continue
            row = ic_hist.iloc[idx]
            regime = classify_regime(panel, date)
            f_w = compute_factor_weights(pd.DataFrame([row.to_dict()], index=[date]), cfg, regime=regime)
            etf_w = aggregate_factor_to_etf(f_w, cfg)
            if etf_w:
                last_weights = etf_w

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in panel.columns:
                    r = panel[code].iloc[i] / panel[code].iloc[i - 1] - 1.0
                    if not np.isnan(r):
                        daily_ret += w * r
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="v4_baseline")


def main():
    panel = load_smartbeta_panel()
    print(f"[data] {panel.shape[0]} days × {panel.shape[1]} codes")

    print("\n========= 1. v4 baseline (IC^2, no shrinkage) =========")
    nav_v4 = backtest_v4_baseline(panel, "M")
    m = metrics(nav_v4)
    print(f"  v4_baseline: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 2. LW with 固定 λ (扫描) =========")
    cfg = LWFactorTimingConfig()
    lambda_results = {}
    for lam in [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 100.0]:
        nav_lw, _ = backtest_lw_factor_timing(
            panel, cfg, fixed_lambda=lam, use_regime=True, use_rolling_lambda=False,
        )
        m = metrics(nav_lw)
        lambda_results[lam] = m
        print(f"  λ={lam:6.2f}: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 3. LW with 滚动验证 λ =========")
    nav_roll, log_roll = backtest_lw_factor_timing(
        panel, cfg, fixed_lambda=None, use_regime=True, use_rolling_lambda=True,
    )
    m = metrics(nav_roll)
    print(f"  滚动 λ: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")
    if log_roll:
        df_log = pd.DataFrame(log_roll)
        print(f"  平均 λ = {df_log['lambda'].mean():.2f},  中位 λ = {df_log['lambda'].median():.2f}")
        print("  λ 分布:")
        for lam, cnt in df_log["lambda"].value_counts().sort_index().items():
            print(f"    λ={lam:6.2f}: {cnt:3d} 次 ({cnt/len(df_log)*100:.1f}%)")

    print("\n========= 4. LW with 滚动 λ + 静态等权(λ=∞) =========")
    cfg_eq = LWFactorTimingConfig(candidate_lambdas=(1000.0,))
    nav_eq, _ = backtest_lw_factor_timing(
        panel, cfg_eq, fixed_lambda=None, use_regime=True, use_rolling_lambda=True,
    )
    m = metrics(nav_eq)
    print(f"  等权 (regime): Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 5. OOS Walk-Forward 验证 =========")
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    train_idx = panel.index <= train_end
    test_idx = (panel.index >= test_start) & (panel.index <= END)
    print(f"  Train: {panel.index[0].date()} to {train_end}  "
          f"(n={train_idx.sum()})")
    print(f"  Test:  {test_start} to {END}  "
          f"(n={test_idx.sum()})")

    print("\n  --- OOS 性能对比 ---")
    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]

    test_v4 = nav_v4.loc[test_start:]
    m = metrics(test_v4)
    print(f"  v4 baseline:    Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    test_roll = nav_roll.loc[test_start:]
    m = metrics(test_roll)
    print(f"  LW 滚动 λ:     Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    for lam in [0.0, 1.0, 10.0, 30.0, 100.0]:
        nav_lw, _ = backtest_lw_factor_timing(
            panel, cfg, fixed_lambda=lam, use_regime=True, use_rolling_lambda=False,
        )
        test_lw = nav_lw.loc[test_start:]
        m = metrics(test_lw)
        print(f"  LW λ={lam:6.2f}:     Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 6. 关键对比: v3 + LW 组合 =========")
    test_v3 = n_v3.loc[test_start:]
    for w_v3, w_lw in [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]:
        nav_mix = w_v3 * test_v3 + w_lw * test_roll
        m = metrics(nav_mix)
        print(f"  v3 {w_v3:.0%} + LW滚动 {w_lw:.0%}:  "
              f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "v4_baseline": nav_v4,
        "lw_rolling": nav_roll,
    })
    out_df.to_parquet(out_dir / "lw_factor_timing_navs.parquet")
    print(f"\n[save] {out_dir / 'lw_factor_timing_navs.parquet'}")

    if log_roll:
        df_log = pd.DataFrame(log_roll)
        df_log.to_csv(out_dir / "lw_rolling_lambda_log.csv", index=False)
        print(f"[save] {out_dir / 'lw_rolling_lambda_log.csv'}")


if __name__ == "__main__":
    main()
