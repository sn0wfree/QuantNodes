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

# tv=1.5 定版 (2026-09-02, Sharpe 2.30/Calmar 3.29 @10bp)
TV150_CFG = dict(FINAL_CFG, target_vol=1.50)


def run_final(cfg_dict: dict | None = None,
              tag: str = "final",
              exclude: list | None = None,
              ) -> tuple[FactorModuleComboStrategy, object, pd.DatetimeIndex]:
    cfg_kw = dict(cfg_dict or FINAL_CFG)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    if exclude:
        from QuantNodes.strategy.cta.data_loader import CtaDataLoader
        dl = CtaDataLoader()
        all_prods = dl.list_active_products()
        cfg_kw_products = [p for p in all_prods if p not in exclude]
        s = FactorModuleComboStrategy(config=cfg, products=cfg_kw_products)
    else:
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tv150", action="store_true",
                        help="用tv=1.5定版配置 (默认tv=0.30)")
    parser.add_argument("--calibrated", action="store_true",
                        help="用按品种校准成本 (cost_model.json)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="剔除品种列表 (如 --exclude EC)")
    args = parser.parse_args()

    cfg = dict(TV150_CFG if args.tv150 else FINAL_CFG)
    if args.calibrated:
        import json
        table = json.loads((REPORT_DIR / "cost_model.json").read_text())
        cfg["commission_rate"] = {p: v["commission_bps"] / 10000 for p, v in table.items()}
        cfg["slippage_bps"] = {p: v["slippage_bps"] for p, v in table.items()}
    tag = "tv150" if args.tv150 else "final"
    if args.calibrated:
        tag += "_calibrated"
    if args.exclude:
        tag += "_ex" + "-".join(args.exclude)
    s, r, dates = run_final(cfg, tag, exclude=args.exclude or None)
    nav = pd.Series(r.nav.values, index=dates)
    pos = pd.DataFrame(r.positions, index=dates)
    m = r.metrics
    print(f"[{tag}] @10bp: Sharpe={m['Sharpe']:.3f} AnnRet={m['AnnRet']:.2%} "
          f"MaxDD={m['MaxDD']:.2%} Calmar={m['Calmar']:.3f}")

    # 1. 年度表
    yt = yearly_table(nav, pos, dates)
    print("\n=== 年度指标 ===")
    print(yt.to_string(index=False,
                       float_format=lambda x: f"{x:.3f}"))
    yt.to_csv(REPORT_DIR / f"{tag}_yearly.csv", index=False)

    # 2. 品种贡献
    pp = product_pnl(s, pos, dates)
    print("\n=== 品种贡献 TOP10 正/负 (净值收益, 全期) ===")
    cols = ["pnl_total", "pnl_gross_total", "turnover_total"]
    print("最差10品种:")
    print(pp[cols].head(10).to_string(float_format=lambda x: f"{x:.4f}"))
    print("\n最佳10品种:")
    print(pp[cols].tail(10).to_string(float_format=lambda x: f"{x:.4f}"))
    pp.to_csv(REPORT_DIR / f"{tag}_product_pnl.csv")

    # 2b. 收益集中度: 品种/年度
    total = pp["pnl_total"].sum()
    pos_pnl = pp["pnl_total"][pp["pnl_total"] > 0].sort_values(ascending=False)
    print(f"\n=== 收益集中度 (全期品种PnL合计={total:+.2f}) ===")
    print(f"正贡献品种: {len(pos_pnl)}个, 合计{pos_pnl.sum():+.2f}; "
          f"负贡献: {(pp['pnl_total'] <= 0).sum()}个, 合计{total - pos_pnl.sum():+.2f}")
    cum = 0.0
    for i, (p, v) in enumerate(pos_pnl.items(), 1):
        cum += v
        if cum / total >= 0.5 and i <= 15:
            top_str = ", ".join(pos_pnl.head(i).index)
            print(f"→ 前{i}个品种贡献全期净收益的{cum/total:.0%} (TOP{i}: {top_str})")
            break
    # 年度贡献占比
    ycols = [c for c in pp.columns if c.startswith("y")]
    ysum = pp[ycols].sum()
    print("\n年度净贡献 (全部品种合计):")
    for c in ycols:
        print(f"  {c}: {ysum[c]:+.2f} ({ysum[c]/ysum.sum():+.0%})")

    # 3. 最差年份品种分解
    yt_real = yt[yt["avg_n_pos"] > 0]
    worst_year = int(yt_real.loc[yt_real["sharpe"].idxmin(), "year"])
    print(f"\n=== 最差年份 {worst_year} 品种分解 ===")
    ycol = f"y{worst_year}"
    print(pp.nsmallest(10, ycol)[[ycol, "pnl_total"]].to_string(
        float_format=lambda x: f"{x:.4f}"))
    print(f"\n{worst_year}年 正贡献品种数: {(pp[ycol] > 0).sum()} / "
          f"负贡献品种数: {(pp[ycol] <= 0).sum()}")

    # 4. NAV曲线
    pd.DataFrame({"nav": nav}).to_csv(REPORT_DIR / f"{tag}_nav.csv")
    print(f"\n已保存: {REPORT_DIR}/{tag}_{{yearly,product_pnl,nav}}.csv")


if __name__ == "__main__":
    main()
