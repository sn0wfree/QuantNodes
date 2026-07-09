# coding=utf-8
"""v5 子策略独立回测 + 组合回测验证.

目标: 验证 Stage 18 v5 4+5 改进的实际效果
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v5.style_rotation_v5 import (
    StyleRotationV5Config,
    StyleRotationV5SubStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v5.factor_timing_v5 import (
    FactorTimingV5Config,
    FactorTimingV5SubStrategy,
    classify_regime_v5,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import (
    ALL_V4_CODES,
    load_smartbeta_panel,
    SMART_BETA_CODES,
    STYLE_GROUP_CODES,
)

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


def backtest_sub_strategy(strat, panel, freq="M"):
    """通用子策略回测."""
    dates = panel.index
    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    elif freq == "W-FRI":
        rebal_dates = dates.to_series().resample("W-FRI").last().index
    elif freq == "W-MON":
        rebal_dates = dates.to_series().resample("W-MON").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_dates_set = set(d for d in rebal_dates if d in dates)

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    log_rows = []

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_dates_set and i > 252:
            result = strat.run_step(panel, date)
            if result.weights:
                last_weights = result.weights
                log_rows.append({
                    "date": date,
                    "weights": dict(result.weights),
                    "regime": result.meta.get("regime", "?"),
                    "signal": result.signal_strength,
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

    return pd.Series(nav, index=dates, name="nav"), pd.DataFrame(log_rows)


def main():
    panel = load_smartbeta_panel()
    print(f"[data] {panel.shape[0]} days × {panel.shape[1]} codes")

    print("\n========= 1. v5 风格轮动独立回测 =========")
    cfg_a = StyleRotationV5Config(
        windows=(5, 20, 120, 180),
        window_weights=(0.10, 0.20, 0.30, 0.40),
        dividend_floor=0.20,
        top_n=2,
        top_n_per_style=1,
        sideways_style_exposure=0.50,
        rebalance_freq="M",
        max_weight=0.40,
    )
    strat_a = StyleRotationV5SubStrategy(cfg_a)
    nav_a, log_a = backtest_sub_strategy(strat_a, panel, "M")
    m_a = metrics(nav_a)
    print(f"  StyleRotationV5: Ann={m_a['ann_return']*100:.2f}%  "
          f"Sharpe={m_a['sharpe']:.2f}  DD={m_a['max_dd']*100:.2f}%  "
          f"Calmar={m_a['calmar']:.3f}  n_rebal={len(log_a)}")

    yearly_a = nav_a.resample("YE").last() / nav_a.resample("YE").first() - 1
    print("  Year-by-year:")
    for year, ret in yearly_a.items():
        print(f"    {year.year}: {ret*100:+6.2f}%")

    print("\n  Regime 分布 (rebal 日):")
    if "regime" in log_a.columns and len(log_a) > 0:
        regime_counts = log_a["regime"].value_counts()
        for r, c in regime_counts.items():
            print(f"    {r}: {c} ({c/len(log_a)*100:.1f}%)")

    print("\n========= 2. v5 因子择时独立回测 =========")
    cfg_f = FactorTimingV5Config(
        rebalance_freq="M",
        max_weight=0.50,
    )
    strat_f = FactorTimingV5SubStrategy(cfg_f)
    nav_f, log_f = backtest_sub_strategy(strat_f, panel, "M")
    m_f = metrics(nav_f)
    print(f"  FactorTimingV5: Ann={m_f['ann_return']*100:.2f}%  "
          f"Sharpe={m_f['sharpe']:.2f}  DD={m_f['max_dd']*100:.2f}%  "
          f"Calmar={m_f['calmar']:.3f}  n_rebal={len(log_f)}")

    yearly_f = nav_f.resample("YE").last() / nav_f.resample("YE").first() - 1
    print("  Year-by-year:")
    for year, ret in yearly_f.items():
        print(f"    {year.year}: {ret*100:+6.2f}%")

    print("\n  Regime 分布 (rebal 日):")
    if "regime" in log_f.columns and len(log_f) > 0:
        regime_counts = log_f["regime"].value_counts()
        for r, c in regime_counts.items():
            print(f"    {r}: {c} ({c/len(log_f)*100:.1f}%)")

    print("\n========= 3. 基准对比 =========")
    sb_codes = list(SMART_BETA_CODES.values())
    sb_rets = panel[sb_codes].pct_change().fillna(0)
    eq_nav = (1 + sb_rets.mean(axis=1)).cumprod()
    eq_nav.iloc[0] = 1.0
    m_eq = metrics(eq_nav)
    print(f"  等权 7 Smart β: Ann={m_eq['ann_return']*100:.2f}%  "
          f"Sharpe={m_eq['sharpe']:.2f}  DD={m_eq['max_dd']*100:.2f}%  "
          f"Calmar={m_eq['calmar']:.3f}")

    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]
    m_v3 = metrics(n_v3)
    print(f"  v3 baseline:    Ann={m_v3['ann_return']*100:.2f}%  "
          f"Sharpe={m_v3['sharpe']:.2f}  DD={m_v3['max_dd']*100:.2f}%  "
          f"Calmar={m_v3['calmar']:.3f}")

    print("\n========= 4. 组合: 50% v5 风格 + 50% v5 因子 =========")
    nav_combo = 0.5 * nav_a + 0.5 * nav_f
    m_combo = metrics(nav_combo)
    print(f"  50/50:          Ann={m_combo['ann_return']*100:.2f}%  "
          f"Sharpe={m_combo['sharpe']:.2f}  DD={m_combo['max_dd']*100:.2f}%  "
          f"Calmar={m_combo['calmar']:.3f}")

    print("\n========= 5. 相关性分析 =========")
    navs = pd.DataFrame({
        "v3": n_v3,
        "v5_style": nav_a,
        "v5_factor": nav_f,
        "combo_50_50": nav_combo,
    }).dropna()
    rets_df = navs.pct_change().dropna()
    print("  日收益相关:")
    print(rets_df.corr().round(2).to_string())

    print("\n  60d 滚动相关 (v5_style vs v5_factor):")
    rc = rets_df["v5_style"].rolling(60).corr(rets_df["v5_factor"])
    print(f"    mean={rc.mean():+.3f}  min={rc.min():+.3f}  max={rc.max():+.3f}")

    print("\n========= 6. v5 风格 + v3 =========")
    for w_a in [0.2, 0.3, 0.4, 0.5, 0.6]:
        nav_mix = w_a * nav_a + (1 - w_a) * n_v3
        m_mix = metrics(nav_mix)
        print(f"  v5风格 {w_a:.0%} + v3 {1-w_a:.0%}:  "
              f"Ann={m_mix['ann_return']*100:.2f}%  Sharpe={m_mix['sharpe']:.2f}  "
              f"DD={m_mix['max_dd']*100:.2f}%  Calmar={m_mix['calmar']:.3f}")

    print("\n  v5 因子 + v3:")
    for w_f in [0.2, 0.3, 0.4, 0.5, 0.6]:
        nav_mix = w_f * nav_f + (1 - w_f) * n_v3
        m_mix = metrics(nav_mix)
        print(f"  v5因子 {w_f:.0%} + v3 {1-w_f:.0%}:  "
              f"Ann={m_mix['ann_return']*100:.2f}%  Sharpe={m_mix['sharpe']:.2f}  "
              f"DD={m_mix['max_dd']*100:.2f}%  Calmar={m_mix['calmar']:.3f}")

    print("\n  三策略组合 (v3 + v5_style + v5_factor):")
    for w_v3, w_a, w_f in [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.4, 0.3, 0.3), (0.33, 0.33, 0.34)]:
        nav_mix = w_v3 * n_v3 + w_a * nav_a + w_f * nav_f
        m_mix = metrics(nav_mix)
        print(f"  v3 {w_v3:.0%} + v5s {w_a:.0%} + v5f {w_f:.0%}:  "
              f"Ann={m_mix['ann_return']*100:.2f}%  Sharpe={m_mix['sharpe']:.2f}  "
              f"DD={m_mix['max_dd']*100:.2f}%  Calmar={m_mix['calmar']:.3f}")

    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "v3_baseline": n_v3,
        "v5_style": nav_a,
        "v5_factor": nav_f,
        "combo_50_50": nav_combo,
    })
    out_df.to_parquet(out_dir / "v5_navs.parquet")
    print(f"\n[save] {out_dir / 'v5_navs.parquet'}")


if __name__ == "__main__":
    main()
