# coding=utf-8
"""v5 行业量价因子行业轮动回测 — 与 v3 / v4 组合验证.

测试:
1. v5 单独 (5 因子 vs 11 因子 vs 全部)
2. v3 + v5 组合
3. v3 + v4 因子 + v5 三策略
4. v4 风格 + v5 双策略
5. OOS walk-forward
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v5 import (
    IndustryRotationV5Config,
    IndustryRotationV5SubStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import load_smartbeta_panel

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


def backtest_v5(panel, cfg, freq="M"):
    dates = panel.index
    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    print(f"[v5] 预计算 11 因子 (44 codes × 11 factors × 2200 days) ...")
    from QuantNodes.strategy.momentum_etf_rotation.v5 import compute_all_factors_panel
    factor_panel = compute_all_factors_panel(panel, cfg.factor_cfg)
    print(f"[v5] {len(factor_panel)} codes 因子 panel 准备好")

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    log_rows = []

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            from QuantNodes.strategy.momentum_etf_rotation.v5 import compute_composite_factor
            composite = compute_composite_factor(
                factor_panel, cfg.factor_cfg, date, cfg.factor_weights,
            )
            if len(composite) >= cfg.top_n:
                top = composite.nlargest(cfg.top_n)
                last_weights = {code: 1.0 / cfg.top_n for code in top.index}
                log_rows.append({
                    "date": date,
                    "weights": dict(last_weights),
                    "chosen": list(top.index),
                })

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in panel.columns.get_level_values(0):
                    p_t = panel[code]["close"].iloc[i]
                    p_prev = panel[code]["close"].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        r = p_t / p_prev - 1.0
                        daily_ret += w * r
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="v5_industry"), log_rows


def main():
    print(f"[data] 加载 OHLCV 面板 ...")
    panel = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet")
    panel = panel.loc[START:END]
    print(f"[data] {panel.shape[0]} 天 × {panel.shape[1]} 列 ({len(panel.columns.get_level_values(0).unique())} codes)")

    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]
    n_v4_style = pd.read_parquet("reports/momentum_etf_rotation/v4/v4_merged_navs.parquet")["v4_style_merged"]
    n_v4_factor = pd.read_parquet("reports/momentum_etf_rotation/v4/v4_merged_navs.parquet")["v4_factor_merged"]

    print("\n========= 1. v5 单独回测 =========")
    cfg = IndustryRotationV5Config(top_n=5)
    nav_v5, log_v5 = backtest_v5(panel, cfg)
    m = metrics(nav_v5)
    print(f"  v5 行业量价 Top-5: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n  Year-by-year:")
    yearly = nav_v5.resample("YE").last() / nav_v5.resample("YE").first() - 1
    for year, ret in yearly.items():
        print(f"    {year.year}: {ret*100:+6.2f}%")

    print("\n  Top-N 扫描:")
    for top_n in [3, 5, 7, 10, 15, 20]:
        cfg_n = IndustryRotationV5Config(top_n=top_n)
        nav_n, _ = backtest_v5(panel, cfg_n)
        m = metrics(nav_n)
        print(f"    Top-{top_n:2d}: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 2. v3 + v5 组合 (8y) =========")
    for w_v3 in [0.5, 0.6, 0.7, 0.8]:
        w_v5 = 1 - w_v3
        nav_mix = w_v3 * n_v3 + w_v5 * nav_v5
        m = metrics(nav_mix)
        print(f"  v3 {w_v3:.0%} + v5 {w_v5:.0%}: Ann={m['ann_return']*100:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 3. v3 + v4 因子 + v5 三策略 =========")
    for w_v3, w_v4, w_v5 in [
        (0.6, 0.0, 0.4), (0.5, 0.2, 0.3), (0.5, 0.25, 0.25),
        (0.4, 0.3, 0.3), (0.33, 0.33, 0.34), (0.4, 0.2, 0.4),
    ]:
        nav_mix = w_v3 * n_v3 + w_v4 * n_v4_factor + w_v5 * nav_v5
        m = metrics(nav_mix)
        print(f"  v3 {w_v3:.0%} + v4f {w_v4:.0%} + v5 {w_v5:.0%}:  "
              f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 4. v4 风格 + v5 双策略 =========")
    for w_v4s, w_v5 in [(0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]:
        nav_mix = w_v4s * n_v4_style + w_v5 * nav_v5
        m = metrics(nav_mix)
        print(f"  v4s {w_v4s:.0%} + v5 {w_v5:.0%}: Ann={m['ann_return']*100:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 5. 相关性分析 =========")
    navs = pd.DataFrame({
        "v3": n_v3,
        "v4_style": n_v4_style,
        "v4_factor": n_v4_factor,
        "v5_industry": nav_v5,
    }).dropna()
    rets = navs.pct_change().dropna()
    print("  日收益相关:")
    print(rets.corr().round(2).to_string())

    print("\n========= 6. OOS Walk-Forward (2022-2026) =========")
    test_start = "2022-01-01"
    print(f"  Train: 2018-01 to {test_start}  (4y)")
    print(f"  Test:  {test_start} to {END}  (4.5y)")

    test_v3 = n_v3.loc[test_start:]
    test_v4f = n_v4_factor.loc[test_start:]
    test_v5 = nav_v5.loc[test_start:]

    print("\n  v5 单独 OOS:")
    m = metrics(test_v5)
    print(f"    量价 Top-5 OOS: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n  v3 + v5 OOS:")
    for w_v3 in [0.5, 0.6, 0.7, 0.8]:
        w_v5 = 1 - w_v3
        nav_mix = w_v3 * test_v3 + w_v5 * test_v5
        m = metrics(nav_mix)
        print(f"    v3 {w_v3:.0%} + v5 {w_v5:.0%}: Ann={m['ann_return']*100:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n  v3 + v4f + v5 OOS:")
    for w_v3, w_v4, w_v5 in [
        (0.6, 0.0, 0.4), (0.5, 0.2, 0.3), (0.5, 0.25, 0.25),
        (0.4, 0.3, 0.3), (0.33, 0.33, 0.34), (0.4, 0.2, 0.4),
    ]:
        nav_mix = w_v3 * test_v3 + w_v4 * test_v4f + w_v5 * test_v5
        m = metrics(nav_mix)
        print(f"    v3 {w_v3:.0%} + v4f {w_v4:.0%} + v5 {w_v5:.0%}:  "
              f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    out_dir = REPO / "reports/momentum_etf_rotation/v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "v3_baseline": n_v3,
        "v4_style": n_v4_style,
        "v4_factor": n_v4_factor,
        "v5_industry": nav_v5,
    })
    out_df.to_parquet(out_dir / "v5_navs.parquet")
    print(f"\n[save] {out_dir / 'v5_navs.parquet'}")


if __name__ == "__main__":
    main()
