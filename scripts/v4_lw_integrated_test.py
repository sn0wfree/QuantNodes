# coding=utf-8
"""v4 LW 集成测试 — 验证 LW 模式作为 v4 可选 config.

测试:
1. v4 IC^2 默认 (lw_enabled=False) - baseline
2. v4 LW 固定 λ=10 (lw_enabled=True, lw_lambda_mode="fixed")
3. v4 LW 滚动 λ (lw_enabled=True, lw_lambda_mode="rolling")
4. v3 + 各 v4 因子择时 组合
5. OOS walk-forward 2022-2026
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v4.factor_timing_v4 import (
    FactorTimingConfig,
    aggregate_factor_to_etf,
    backtest_factor_timing,
    compute_factor_weights,
    compute_factor_weights_lw,
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


def backtest_factor(panel, cfg, freq="M"):
    dates = panel.index
    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    ic_hist = backtest_factor_timing(panel, list(ALL_V4_CODES), cfg, START, END)
    if ic_hist.empty:
        return pd.Series(np.ones(len(dates)), index=dates), []

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    log_rows = []

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            idx = ic_hist.index.get_indexer([date], method="ffill")[0]
            if idx < 0:
                continue
            row = ic_hist.iloc[idx]

            regime = classify_regime(panel, date)

            if cfg.lw_enabled:
                ic_window = ic_hist.iloc[:idx + 1]
                f_w = compute_factor_weights_lw(ic_window, cfg, regime=regime)
            else:
                f_w = compute_factor_weights(
                    pd.DataFrame([row.to_dict()], index=[date]), cfg, regime=regime,
                )

            etf_w = aggregate_factor_to_etf(f_w, cfg)
            if etf_w:
                last_weights = etf_w
                log_rows.append({
                    "date": date,
                    "regime": regime,
                    "factor_weights": dict(f_w),
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

    return pd.Series(nav, index=dates, name="v4_factor"), log_rows


def main():
    panel = load_smartbeta_panel()
    print(f"[data] {panel.shape[0]} days × {panel.shape[1]} codes")

    print("\n========= 1. v4 IC^2 默认 (lw_enabled=False) =========")
    cfg1 = FactorTimingConfig()
    nav1, _ = backtest_factor(panel, cfg1)
    m = metrics(nav1)
    print(f"  v4 IC^2:    Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 2. v4 LW 固定 λ 扫描 =========")
    for lam in [0.0, 1.0, 5.0, 10.0, 30.0, 100.0]:
        cfg_lw = FactorTimingConfig(
            lw_enabled=True, lw_lambda_mode="fixed", lw_lambda_fixed=lam,
        )
        nav_lw, _ = backtest_factor(panel, cfg_lw)
        m = metrics(nav_lw)
        print(f"  LW λ={lam:6.2f}: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 3. v4 LW 滚动 λ =========")
    cfg_roll = FactorTimingConfig(lw_enabled=True, lw_lambda_mode="rolling")
    nav_roll, log_roll = backtest_factor(panel, cfg_roll)
    m = metrics(nav_roll)
    print(f"  LW 滚动 λ:  Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 4. v3 + v4 因子 组合 (不同模式) =========")
    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]
    for w_v3 in [0.5, 0.6, 0.7, 0.8]:
        w_v4 = 1 - w_v3
        for name, nav_v4 in [("IC^2", nav1), ("LW λ=10", None), ("LW 滚动", nav_roll)]:
            if name == "LW λ=10":
                cfg_lam = FactorTimingConfig(
                    lw_enabled=True, lw_lambda_mode="fixed", lw_lambda_fixed=10.0,
                )
                nav_v4, _ = backtest_factor(panel, cfg_lam)
            nav_mix = w_v3 * n_v3 + w_v4 * nav_v4
            m = metrics(nav_mix)
            print(f"  v3 {w_v3:.0%} + {name} {w_v4:.0%}:  "
                  f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
                  f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")
        print()

    print("\n========= 5. OOS Walk-Forward (2022-2026) =========")
    test_start = "2022-01-01"
    print(f"  Train: 2018-01 to 2021-12  (4y)")
    print(f"  Test:  {test_start} to {END}  (4.5y)")
    print()

    for name, cfg, nav_v4 in [
        ("IC^2", FactorTimingConfig(), nav1),
        ("LW λ=10", FactorTimingConfig(lw_enabled=True, lw_lambda_mode="fixed", lw_lambda_fixed=10.0), None),
        ("LW 滚动", FactorTimingConfig(lw_enabled=True, lw_lambda_mode="rolling"), nav_roll),
    ]:
        if nav_v4 is None:
            nav_v4, _ = backtest_factor(panel, cfg)
        test_v4 = nav_v4.loc[test_start:]
        m = metrics(test_v4)
        print(f"  {name} OOS: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n  v3 + v4 OOS:")
    test_v3 = n_v3.loc[test_start:]
    for w_v3 in [0.6, 0.7, 0.8]:
        w_v4 = 1 - w_v3
        for name, nav_v4 in [("IC^2", nav1), ("LW 滚动", nav_roll)]:
            nav_mix = w_v3 * test_v3 + w_v4 * nav_v4.loc[test_start:]
            m = metrics(nav_mix)
            print(f"    v3 {w_v3:.0%} + {name} {w_v4:.0%}:  "
                  f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
                  f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "v4_ic2": nav1,
        "v4_lw_rolling": nav_roll,
    })
    out_df.to_parquet(out_dir / "v4_lw_integrated_navs.parquet")
    print(f"\n[save] {out_dir / 'v4_lw_integrated_navs.parquet'}")


if __name__ == "__main__":
    main()
