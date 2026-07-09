# coding=utf-8
"""行业量价因子行业轮动回测 — 实施华西证券策略.

策略:
  1. 每月末计算 11 个量价因子
  2. 各因子 z-score 标准化 (截面)
  3. 复合因子 = 11 因子 z-score 等权加总 (论文用 IC 加权, 我们用等权)
  4. 选复合因子值最高的 5 个 ETF, 等权持仓
  5. 月末调仓

实施注意:
  - 行业代理: 用 44 只 ETF (含 Smart β + 行业 ETF) 作为行业代理
  - 数据: 2018-01 到 2026-06 OHLCV (Sina API)
  - 论文用 28 个中信一级行业 + 2010-2022.07, 我们用 ETF 池 + 8 年

参考:
  - 华西证券《行业有效量价因子与行业轮动策略》
  - reports/momentum_etf_rotation/v4/papers/huaxi_industry_rotation.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v4.industry_factors import (
    FactorEngineConfig,
    compute_all_factors_panel,
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


def cross_section_zscore(
    factor_panel: dict[str, pd.DataFrame],
    factor: str,
    as_of: pd.Timestamp,
) -> pd.Series:
    """截面 z-score: 在 as_of 日, 各 code 的 factor 值, 去均值/std."""
    values = {}
    for code, df in factor_panel.items():
        if factor not in df.columns:
            continue
        if as_of not in df.index:
            idx = df.index.get_indexer([as_of], method="ffill")[0]
            if idx < 0:
                continue
            val = df[factor].iloc[idx]
        else:
            val = df[factor].loc[as_of]
        if pd.isna(val):
            continue
        values[code] = float(val)
    s = pd.Series(values)
    if s.empty:
        return s
    return (s - s.mean()) / s.std() if s.std() > 0 else s * 0


def compute_composite_factor(
    factor_panel: dict[str, pd.DataFrame],
    cfg: FactorEngineConfig,
    as_of: pd.Timestamp,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """计算 as_of 日各 code 的复合因子 = Σ w_i × zscore(f_i)."""
    factors = list(cfg.name_map.keys())
    if weights is None:
        weights = {f: 1.0 / len(factors) for f in factors}

    composite = pd.Series(dtype=float)
    for fac in factors:
        w = weights.get(fac, 0.0) if weights else 1.0 / len(factors)
        if w == 0:
            continue
        z = cross_section_zscore(factor_panel, fac, as_of)
        composite = composite.add(z * w, fill_value=0.0)
    return composite


def backtest_industry_rotation(
    ohlcv_panel: pd.DataFrame,
    cfg: FactorEngineConfig | None = None,
    top_n: int = 5,
    freq: str = "ME",
    min_history: int = 252,
    weights: dict[str, float] | None = None,
    verbose: bool = True,
) -> tuple[pd.Series, list[dict]]:
    """行业量价因子行业轮动回测.

    Args:
        ohlcv_panel: 多级 columns (code, field) panel
        cfg: 因子配置
        top_n: 选 Top-N (论文: 5)
        freq: 调仓频率 (论文: M)
        min_history: 最少历史天数
        weights: 复合因子权重 (None = 等权)
        verbose: 打印进度

    Returns:
        nav: NAV 时序
        log: 调仓日志
    """
    cfg = cfg or FactorEngineConfig()

    if verbose:
        print(f"[compute] 计算 11 因子 ...")
    factor_panel = compute_all_factors_panel(ohlcv_panel, cfg)
    if verbose:
        print(f"[compute] {len(factor_panel)} 个 ETF 因子计算完成")

    panel = ohlcv_panel.loc[START:END]
    dates = panel.index
    rebal_dates = dates.to_series().resample(freq).last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}
    log_rows = []

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > min_history:
            composite = compute_composite_factor(factor_panel, cfg, date, weights)
            if len(composite) >= top_n:
                top = composite.nlargest(top_n)
                new_weights = {code: 1.0 / top_n for code in top.index}
                last_weights = new_weights
                log_rows.append({
                    "date": date,
                    "weights": dict(new_weights),
                    "composite": composite.to_dict(),
                })

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                if code in panel.columns.get_level_values(0):
                    sub = panel[code]
                    if not sub["close"].iloc[i: i + 1].empty and not sub["close"].iloc[i - 1: i].empty:
                        p_t = sub["close"].iloc[i]
                        p_prev = sub["close"].iloc[i - 1]
                        if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                            r = p_t / p_prev - 1.0
                            daily_ret += w * r
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="industry_rotation"), log_rows


def backtest_equal_weight_benchmark(
    ohlcv_panel: pd.DataFrame,
    cfg: FactorEngineConfig | None = None,
) -> pd.Series:
    """等权 ETF 基准 (对应论文的 28 中信一级行业等权)."""
    panel = ohlcv_panel.loc[START:END]
    codes = sorted(panel.columns.get_level_values(0).unique())
    codes = [c for c in codes if panel[c]["close"].notna().sum() > 252]
    n = len(codes)
    if n == 0:
        return pd.Series(np.ones(len(panel)), index=panel.index)

    nav = np.ones(len(panel))
    for i in range(1, len(panel)):
        daily_ret = 0.0
        for code in codes:
            sub = panel[code]
            p_t = sub["close"].iloc[i]
            p_prev = sub["close"].iloc[i - 1]
            if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                daily_ret += (1.0 / n) * (p_t / p_prev - 1.0)
        nav[i] = nav[i - 1] * (1 + daily_ret)
    return pd.Series(nav, index=panel.index, name="equal_weight")


def backtest_top_n_benchmark(
    ohlcv_panel: pd.DataFrame,
    cfg: FactorEngineConfig | None = None,
    top_n: int = 5,
) -> pd.Series:
    """每月选动量 Top-N 等权 (动量基准)."""
    panel = ohlcv_panel.loc[START:END]
    dates = panel.index
    rebal_dates = dates.to_series().resample("ME").last().index
    rebal_set = set(d for d in rebal_dates if d in dates)

    codes = sorted(panel.columns.get_level_values(0).unique())
    nav = np.ones(len(dates))
    last_weights: dict[str, float] = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue
        if date in rebal_set and i > 252:
            rets_60 = {}
            for code in codes:
                sub = panel[code]["close"]
                if i >= 60 and pd.notna(sub.iloc[i]) and pd.notna(sub.iloc[i - 60]):
                    rets_60[code] = float(sub.iloc[i] / sub.iloc[i - 60] - 1.0)
            if len(rets_60) >= top_n:
                sorted_codes = sorted(rets_60, key=rets_60.get, reverse=True)[:top_n]
                last_weights = {c: 1.0 / top_n for c in sorted_codes}

        if last_weights:
            daily_ret = 0.0
            for code, w in last_weights.items():
                sub = panel[code]
                p_t = sub["close"].iloc[i]
                p_prev = sub["close"].iloc[i - 1]
                if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                    daily_ret += w * (p_t / p_prev - 1.0)
            nav[i] = nav[i - 1] * (1 + daily_ret)
        else:
            nav[i] = nav[i - 1]

    return pd.Series(nav, index=dates, name="momentum_top5")


def main():
    print(f"[data] 加载 OHLCV 面板 ...")
    panel = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel = panel.loc[START:END]
    print(f"[data] {panel.shape[0]} 天 × {panel.shape[1]} 列 ({len(panel.columns.get_level_values(0).unique())} codes)")

    print("\n========= 1. 等权 ETF 基准 =========")
    nav_eq = backtest_equal_weight_benchmark(panel)
    m = metrics(nav_eq)
    print(f"  等权 {len(panel.columns.get_level_values(0).unique())} ETF: "
          f"Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 2. 动量 Top-5 基准 =========")
    nav_mom = backtest_top_n_benchmark(panel, top_n=5)
    m = metrics(nav_mom)
    print(f"  60d 动量 Top-5: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 3. 量价因子 Top-5 行业轮动 =========")
    cfg = FactorEngineConfig()
    nav_industry, log_industry = backtest_industry_rotation(
        panel, cfg, top_n=5, freq="ME", min_history=252,
    )
    m = metrics(nav_industry)
    print(f"  量价因子 Top-5: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
          f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n  Year-by-year:")
    yearly = nav_industry.resample("YE").last() / nav_industry.resample("YE").first() - 1
    for year, ret in yearly.items():
        print(f"    {year.year}: {ret*100:+6.2f}%")

    print("\n  月胜率:")
    monthly = nav_industry.resample("ME").last().pct_change().dropna()
    win_rate = (monthly > 0).mean()
    print(f"    月胜率: {win_rate*100:.1f}%  ({monthly.gt(0).sum()}/{len(monthly)})")

    print("\n========= 4. 调仓换手率分析 =========")
    if log_industry:
        prev_weights = {}
        turnovers = []
        for row in log_industry:
            if prev_weights:
                w_prev = set(prev_weights.keys())
                w_new = set(row["weights"].keys())
                n_change = len(w_new - w_prev) + len(w_prev - w_new)
                turnovers.append(n_change / 5)
            prev_weights = row["weights"]
        if turnovers:
            print(f"  平均换手率: {np.mean(turnovers)*100:.2f}%/月")
            print(f"  中位换手率: {np.median(turnovers)*100:.2f}%/月")

    print("\n========= 5. Top-N 扫描 =========")
    for top_n in [3, 5, 7, 10, 15, 20]:
        nav_n, _ = backtest_industry_rotation(panel, cfg, top_n=top_n, freq="ME", min_history=252, verbose=False)
        m = metrics(nav_n)
        print(f"  Top-{top_n:2d}: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 6. 因子权重扫描 =========")
    factors = list(cfg.name_map.keys())
    for label, weights in [
        ("等权 11 因子", None),
        ("仅动量类 (f1+f2)", {"f1_second_mom": 0.5, "f2_mom_term": 0.5}),
        ("仅量价背离 (f8+f9+f10)", {"f8_pv_rankcov": 1/3, "f9_pv_corr": 1/3, "f10_first_div": 1/3}),
        ("仅反转 (f3+f4+f6+f8+f9+f10)", {"f3_amt_vol": 1/6, "f4_vol_vol": 1/6, "f6_ls_total": 1/6,
                                            "f8_pv_rankcov": 1/6, "f9_pv_corr": 1/6, "f10_first_div": 1/6}),
    ]:
        nav_w, _ = backtest_industry_rotation(panel, cfg, top_n=5, weights=weights, verbose=False)
        m = metrics(nav_w)
        print(f"  {label}: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 7. v3 + 量价因子 组合 =========")
    n_v3 = pd.read_parquet("reports/momentum_etf_rotation/v4/stage17_navs.parquet")["v3_baseline"]
    for w_v3 in [0.5, 0.6, 0.7, 0.8]:
        w_industry = 1 - w_v3
        nav_mix = w_v3 * n_v3 + w_industry * nav_industry
        m = metrics(nav_mix)
        print(f"  v3 {w_v3:.0%} + 量价 {w_industry:.0%}: Ann={m['ann_return']*100:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 8. 相关性分析 =========")
    navs = pd.DataFrame({
        "v3": n_v3,
        "industry_rotation": nav_industry,
        "equal_weight": nav_eq,
        "momentum_top5": nav_mom,
    }).dropna()
    print("  日收益相关:")
    print(navs.pct_change().dropna().corr().round(2).to_string())

    out_dir = REPO / "reports/momentum_etf_rotation/v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "v3_baseline": n_v3,
        "industry_rotation": nav_industry,
        "equal_weight": nav_eq,
        "momentum_top5": nav_mom,
    })
    out_df.to_parquet(out_dir / "industry_rotation_navs.parquet")
    print(f"\n[save] {out_dir / 'industry_rotation_navs.parquet'}")

    print("\n========= 9. OOS Walk-Forward (2022-2026) =========")
    test_start = "2022-01-01"
    print(f"  Train: 2018-01 to {test_start}  (4y)")
    print(f"  Test:  {test_start} to {END}  (4.5y)")

    test_v3 = n_v3.loc[test_start:]
    test_eq = nav_eq.loc[test_start:]
    test_industry = nav_industry.loc[test_start:]

    for name, nav in [("v3 baseline", test_v3),
                       ("等权 44 ETF", test_eq),
                       ("量价 Top-5", test_industry)]:
        m = metrics(nav)
        print(f"  {name} OOS: Ann={m['ann_return']*100:.2f}%  Sharpe={m['sharpe']:.2f}  "
              f"DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n  v3 + 量价 OOS:")
    for w_v3 in [0.5, 0.6, 0.7, 0.8]:
        w_industry = 1 - w_v3
        nav_mix = w_v3 * test_v3 + w_industry * test_industry
        m = metrics(nav_mix)
        print(f"    v3 {w_v3:.0%} + 量价 {w_industry:.0%}: Ann={m['ann_return']*100:.2f}%  "
              f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%  Calmar={m['calmar']:.3f}")


if __name__ == "__main__":
    main()
