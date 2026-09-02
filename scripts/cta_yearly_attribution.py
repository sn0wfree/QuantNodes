# coding=utf-8
"""cta_yearly_attribution.py — 定版配置年度+品种级归因.

定版配置: ema=0.15, mc=0.30(默认), tvpr×inv_vol, 10bp滑点基准.

分析:
    1. 年度指标表 (AnnRet/Vol/Sharpe/MaxDD/换手)
    2. 品种级PnL贡献 (每品种全年贡献, 识别拖后腿品种)
    3. 最差年份的品种分解

用法:
    python3 scripts/cta_yearly_attribution.py
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

REPORT_DIR = Path("reports/cta")

FINAL_CFG = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True, position_ema_alpha=0.15,
    slippage_bps=10.0,
)


def run_final() -> tuple[FactorModuleComboStrategy, object, pd.DatetimeIndex]:
    cfg_kw = dict(FINAL_CFG)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    s = FactorModuleComboStrategy(config=cfg)
    r = s.run()
    return s, r, pd.DatetimeIndex(r.dates)


def yearly_table(nav: pd.Series, pos: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """年度指标: 收益/波动/Sharpe/MaxDD/日均换手."""
    ret = nav.pct_change().dropna()
    rows = []
    for year, g in ret.groupby(ret.index.year):
        yr_ret = (1 + g).prod() - 1
        vol = g.std() * np.sqrt(252)
        sharpe = yr_ret / vol if vol > 0 else 0
        nav_y = (1 + g).cumprod()
        dd = (nav_y / nav_y.cummax() - 1).min()
        pos_y = pos[(pos.index.year == year)]
        n_pos = (pos_y != 0).sum(axis=1).mean()
        turnover_y = pos_y.diff().abs().sum(axis=1).mean() * 252 / max(n_pos, 1)
        rows.append({
            "year": year, "ann_ret": yr_ret, "vol": vol, "sharpe": sharpe,
            "max_dd": dd, "avg_n_pos": (pos_y != 0).sum(axis=1).mean(),
            "daily_turnover": turnover_y,
        })
    return pd.DataFrame(rows)


def product_pnl(s: FactorModuleComboStrategy, pos: pd.DataFrame,
                dates: pd.DatetimeIndex) -> pd.DataFrame:
    """品种级PnL贡献 (名义收益, 含成本近似)."""
    prices = s._price_panel.reindex(dates)
    rets = prices.pct_change(fill_method=None)

    rows = []
    for p in prices.columns:
        if p not in pos.columns:
            continue
        pnl_gross = pos[p].shift(1) * rets[p]
        trade = pos[p].diff().abs()
        cost = trade * 10 / 10000  # 10bp 单边近似
        pnl_net = pnl_gross - cost
        pnl_net_y = pnl_net.groupby(pnl_net.index.year).sum()
        rows.append({
            "product": p,
            "pnl_total": pnl_net.sum(),
            "pnl_gross_total": pnl_gross.sum(),
            "turnover_total": trade.sum(),
            **{f"y{y}": pnl_net_y.get(y, 0.0) for y in sorted(pnl_net_y.index)},
        })
    df = pd.DataFrame(rows)
    return df.set_index("product").sort_values("pnl_total")


def main():
    s, r, dates = run_final()
    nav = pd.Series(r.nav.values, index=dates)
    pos = pd.DataFrame(r.positions, index=dates)
    m = r.metrics
    print(f"定版配置 @10bp: Sharpe={m['Sharpe']:.3f} AnnRet={m['AnnRet']:.2%} "
          f"MaxDD={m['MaxDD']:.2%} Calmar={m['Calmar']:.3f}")

    # 1. 年度表
    yt = yearly_table(nav, pos, dates)
    print("\n=== 年度指标 ===")
    print(yt.to_string(index=False,
                       float_format=lambda x: f"{x:.3f}"))
    yt.to_csv(REPORT_DIR / "final_yearly.csv", index=False)

    # 2. 品种贡献
    pp = product_pnl(s, pos, dates)
    print("\n=== 品种贡献 TOP10 正/负 (净值收益, 全期) ===")
    cols = ["pnl_total", "pnl_gross_total", "turnover_total"]
    print("最差10品种:")
    print(pp[cols].head(10).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n最佳10品种:")
    print(pp[cols].tail(10).to_string(float_format=lambda x: f"{x:.4f}"))
    pp.to_csv(REPORT_DIR / "final_product_pnl.csv")

    # 3. 最差年份品种分解
    worst_year = int(yt.loc[yt["sharpe"].idxmin(), "year"])
    print(f"\n=== 最差年份 {worst_year} 品种分解 ===")
    ycol = f"y{worst_year}"
    print(pp.nsmallest(10, ycol)[[ycol, "pnl_total"]].to_string(
        float_format=lambda x: f"{x:.4f}"))
    print(f"\n{worst_year}年 正贡献品种数: {(pp[ycol] > 0).sum()} / "
          f"负贡献品种数: {(pp[ycol] <= 0).sum()}")

    # 4. NAV曲线
    pd.DataFrame({"nav": nav}).to_csv(REPORT_DIR / "final_nav.csv")
    print(f"\n已保存: {REPORT_DIR}/final_{{yearly,product_pnl,nav}}.csv")


if __name__ == "__main__":
    main()
