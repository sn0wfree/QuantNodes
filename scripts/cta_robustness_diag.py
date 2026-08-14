# coding=utf-8
"""CTA 稳健性诊断: 换手率 / 杠杆 / 尾部风险 (当前最优配置 tvpr×inv_vol)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

CFG = ComboConfig(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True,
)


def main():
    s = FactorModuleComboStrategy(config=CFG)
    r = s.run()
    nav = r.nav
    equity = pd.Series(r.equity, index=pd.DatetimeIndex(r.dates))
    ret = nav.pct_change().dropna()

    print("\n=== 基础指标 ===")
    m = r.metrics
    print(f"Sharpe={m['Sharpe']:.3f} AnnRet={m['AnnRet']:.3f} Vol={m['Vol']:.3f} "
          f"MaxDD={m['MaxDD']:.3f} Calmar={m['Calmar']:.3f} WinRate={m['WinRate']:.2f}")

    # ── 换手率 ──────────────────────────────────────────────
    print("\n=== 换手率 ===")
    prices = pd.DataFrame(
        {p: s._price_panel[p] for p in s._price_panel.columns}, index=pd.DatetimeIndex(r.dates),
    )
    pos = pd.DataFrame(r.positions, index=pd.DatetimeIndex(r.dates))
    spec = {p: s._registry.get_or_default(p) for p in prices.columns}
    notional = pd.DataFrame(
        {p: pos[p].abs() * prices[p] * spec[p].contract_multiplier
         for p in prices.columns if p in pos.columns},
    )
    gross = notional.sum(axis=1)
    turnover = (notional.diff().abs().sum(axis=1)) / (gross.replace(0, np.nan).mean())
    ann_turnover = turnover.mean() * 252
    print(f"日均换手率 (名义): {turnover.mean():.4f}  年化换手: {ann_turnover:.2f} 倍")

    # 成本影响估算 (双边 1bp~5bp)
    for bp in [0.0001, 0.0005, 0.001, 0.002]:
        cost = turnover.mean() * bp * 252
        print(f"  成本 {bp*100:.1f}bp: 年化拖累 {cost:.4f} "
              f"({cost/m['AnnRet']*100:.0f}% 收益)")

    # ── 杠杆率 ──
    print("\n=== 杠杆率 ===")
    lev = gross / equity
    print(f"日均杠杆: {lev.mean():.3f}  中位: {lev.median():.3f}  "
          f"p95: {lev.quantile(0.95):.3f}  最大: {lev.max():.3f}")
    print(f"杠杆 > 2: {(lev > 2).mean()*100:.1f}%  "
              f"杠杆 > 1: {(lev > 1).mean()*100:.1f}%")

    # ── 尾部风险 ──
    print("\n=== 尾部风险 ===")
    var95 = ret.quantile(0.05)
    cvar95 = ret[ret <= var95].mean()
    worst5 = ret.nsmallest(5)
    print(f"日收益 VaR95: {var95:.4f} ({var95*100:.2f}%)")
    print(f"日收益 CVaR95: {cvar95:.4f} ({cvar95*100:.2f}%)")
    print("最差 5 日: " + ", ".join(f"{d.date()}: {v*100:.2f}%" for d, v in worst5.items()))
    print(f"最长回撤: {m['MaxDDDays']} 天")

    # 极端窗口回撤
    dd = nav / nav.cummax() - 1
    for name, start, end in [("2020 疫情", "2020-02-01", "2020-04-30"),
                             ("2022 商品", "2022-03-01", "2022-07-31"),
                             ("2024 商品", "2024-01-01", "2024-12-31")]:
        sub = dd[start:end]
        if len(sub):
            print(f"{name} 窗口回撤: {sub.min()*100:.2f}%")


if __name__ == "__main__":
    main()
