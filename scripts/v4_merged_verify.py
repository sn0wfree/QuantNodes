# coding=utf-8
"""验证 v4 升级后 (Stage 18 合并版) 与 v5 表现一致.

测试:
1. v4A (仅风格轮动) 应 ≈ v5 风格 Calmar 0.439
2. v4D (含因子择时) 应 ≈ v5 因子 Calmar 0.712
3. v4A + v4D 等权应 ≈ v5 组合 Calmar 0.736
4. v3 33% + v4A 33% + v4D 34% 应 ≈ 三策略 Calmar 0.763
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v4.style_rotation_v4 import (
    StyleRotationConfig,
    StyleRotationSubStrategy,
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
from QuantNodes.strategy.momentum_etf_rotation.v4.sub_strategy_v4 import SubStrategyResult

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


def backtest_style_rotation(panel, freq="M"):
    """v4 升级版风格轮动回测 (使用新默认配置)."""
    cfg = StyleRotationConfig()
    cfg.rebalance_freq = freq
    strat = StyleRotationSubStrategy(cfg)
    dates = panel.index
    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    last_cash = 0.0
    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            result = strat.run_step(panel, date)
            if result.weights or result.meta.get("cash_weight", 0) > 0:
                last_weights = result.weights
                last_cash = result.meta.get("cash_weight", 0.0)

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

    return pd.Series(nav, index=dates, name="v4_style")


def backtest_factor_timing_v4(panel, freq="M"):
    """v4 升级版因子择时回测 (Stage 18 5 改进)."""
    cfg = FactorTimingConfig()
    dates = panel.index
    panel_s = panel.loc[START:END]
    ic_hist = backtest_factor_timing(panel_s, list(ALL_V4_CODES), cfg, START, END)
    if ic_hist.empty:
        return pd.Series(np.ones(len(dates)), index=dates)

    if freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    else:
        rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            idx = ic_hist.index.get_indexer([date], method="ffill")[0]
            if idx >= 0:
                row = ic_hist.iloc[idx]
                regime = "sideways"
                if "510300" in panel.columns:
                    sub = panel.loc[:date, "510300"]
                    if len(sub) > 252:
                        mom60 = float(sub.iloc[-1] / sub.iloc[-61] - 1.0)
                        mom252 = float(sub.iloc[-1] / sub.iloc[-253] - 1.0)
                        if mom60 > 0.05 and mom252 > 0.10:
                            regime = "bull"
                        elif mom60 < -0.05 and mom252 < -0.10:
                            regime = "bear"
                f_w = compute_factor_weights(
                    pd.DataFrame([row.to_dict()], index=[date]),
                    cfg, regime=regime,
                )
                etf_w = aggregate_factor_to_etf(f_w, cfg)
                if etf_w:
                    last_weights = etf_w
                elif "value" in cfg.factor_to_etf:
                    last_weights = {cfg.factor_to_etf["value"][0]: 1.0}
                else:
                    last_weights = {}

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

    return pd.Series(nav, index=dates, name="v4_factor")


def main():
    panel = load_smartbeta_panel()
    print(f"[data] {panel.shape[0]} days × {panel.shape[1]} codes")

    print("\n========= 1. v4 升级版风格轮动 =========")
    nav_style = backtest_style_rotation(panel, "M")
    m = metrics(nav_style)
    print(f"  v4_style: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")
    print(f"  预期: Calmar ~0.439 (v5 一致)")

    print("\n========= 2. v4 升级版因子择时 =========")
    nav_factor = backtest_factor_timing_v4(panel, "M")
    m = metrics(nav_factor)
    print(f"  v4_factor: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")
    print(f"  预期: Calmar ~0.712 (v5 一致)")
    yearly_f = nav_factor.resample("YE").last() / nav_factor.resample("YE").first() - 1
    for year, ret in yearly_f.items():
        print(f"    {year.year}: {ret*100:+6.2f}%")

    print("\n========= 3. 50/50 v4_style + v4_factor =========")
    nav_combo = 0.5 * nav_style + 0.5 * nav_factor
    m = metrics(nav_combo)
    print(f"  50/50: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")
    print(f"  预期: Calmar ~0.736")

    print("\n========= 4. 三策略组合 (v3 + v4_style + v4_factor) =========")
    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]
    for w_v3, w_s, w_f in [(0.6, 0.2, 0.2), (0.5, 0.25, 0.25), (0.4, 0.3, 0.3), (0.33, 0.33, 0.34)]:
        nav_mix = w_v3 * n_v3 + w_s * nav_style + w_f * nav_factor
        m = metrics(nav_mix)
        print(f"  v3 {w_v3:.0%} + v4_style {w_s:.0%} + v4_factor {w_f:.0%}:  "
              f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")
    print(f"  预期最优: Calmar ~0.763 (33/33/34)")

    print("\n========= 5. 相关性 =========")
    navs = pd.DataFrame({
        "v3": n_v3,
        "v4_style": nav_style,
        "v4_factor": nav_factor,
    }).dropna()
    rets = navs.pct_change().dropna()
    print("  日收益相关:")
    print(rets.corr().round(2).to_string())

    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    out_df = pd.DataFrame({
        "v3_baseline": n_v3,
        "v4_style_merged": nav_style,
        "v4_factor_merged": nav_factor,
    })
    out_df.to_parquet(out_dir / "v4_merged_navs.parquet")
    print(f"\n[save] {out_dir / 'v4_merged_navs.parquet'}")


if __name__ == "__main__":
    main()
