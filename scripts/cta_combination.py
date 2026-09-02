#!/usr/bin/env python3
"""cta_combination.py — 主CTA + 子策略合并实验.

子策略:
  - ZN库存 (cta_zn_inventory.py) — OOS Sharpe 0.47 (已PIT验证)
  - CU溢价+TC (cta_cu_premium.py) — OOS Sharpe 1.10 (已PIT验证)
  - M经纪持仓 — 止损 (PIT修正后Sharpe -0.26, 之前0.58是PIT bug)

合并原则:
  - 各子策略日频pnl序列 (vol_target_backtest返回) × 资金比例 = 组合pnl
  - 主CTA不与子策略同品种 (主CTA含CU/M/ZN但子策略是单品种特例, 信号独立)
  - 相关性估计: 子策略与主CTA天然低相关 (基本面 vs 技术面)

比例实验:
  100/0/0  90/5/5  85/10/5  85/7.5/7.5  80/10/10
  80/15/5  80/5/15  85/5/10  85/10/10 (均衡)

输出:
  - 每比例的 Sharpe / AnnRet / MaxDD / Calmar (全期+IS+OOS)
  - 推荐上实盘的资金分配
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from cta_shared.factor_research import REPORT_DIR  # noqa: E402

IS_END = "2022-01-01"
PORTFOLIOS = [
    # (主CTA%, ZN%, CU%, 描述)
    (1.00, 0.00, 0.00, "主CTA单独"),
    (0.95, 0.05, 0.00, "主CTA+ZN5%"),
    (0.95, 0.00, 0.05, "主CTA+CU5%"),
    (0.90, 0.05, 0.05, "主CTA+ZN5+CU5"),
    (0.90, 0.10, 0.00, "主CTA+ZN10%"),
    (0.90, 0.00, 0.10, "主CTA+CU10%"),
    (0.85, 0.10, 0.05, "主CTA+ZN10+CU5"),
    (0.85, 0.05, 0.10, "主CTA+ZN5+CU10"),
    (0.85, 0.075, 0.075, "主CTA+ZN7.5+CU7.5"),
    (0.85, 0.15, 0.00, "主CTA+ZN15%"),
    (0.85, 0.00, 0.15, "主CTA+CU15%"),
    (0.80, 0.10, 0.10, "主CTA+ZN10+CU10"),
    (0.80, 0.15, 0.05, "主CTA+ZN15+CU5"),
    (0.80, 0.05, 0.15, "主CTA+ZN5+CU15"),
]


def load_returns() -> dict[str, pd.Series]:
    """加载各策略日收益率 (从NAV计算)."""
    sources = {
        "主CTA": REPORT_DIR / "final_nav_lump.csv",
        "ZN库存": REPORT_DIR / "zn_inventory_nav.csv",
        "CU溢价TC": REPORT_DIR / "cu_premium_nav.csv",
    }
    out = {}
    common = None
    for name, path in sources.items():
        nav = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        ret = nav.pct_change(fill_method=None).fillna(0)
        out[name] = ret
        common = ret.index if common is None else common.intersection(ret.index)
    return {k: v.reindex(common) for k, v in out.items()}


def stats(r: pd.Series, is_end: str = IS_END) -> dict:
    """分期Sharpe/AnnRet/MaxDD/Calmar."""
    def _one(s: pd.Series) -> dict:
        if len(s) < 50:
            return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "calmar": 0.0}
        nav = (1 + s).cumprod()
        ann = nav.iloc[-1] ** (252 / len(s)) - 1
        dd = (nav / nav.cummax() - 1).min()
        return {
            "sharpe": ann / (s.std() * np.sqrt(252)) if s.std() > 0 else 0.0,
            "ann_ret": ann, "max_dd": dd,
            "calmar": ann / abs(dd) if dd != 0 else 0.0,
        }
    return {
        "full": _one(r), "is": _one(r[r.index < is_end]),
        "oos": _one(r[r.index >= is_end]),
    }


def main():
    rets = load_returns()
    main_ret = rets["主CTA"]
    zn_ret = rets["ZN库存"]
    cu_ret = rets["CU溢价TC"]

    # 相关性
    print("=== 策略间相关性 (日收益) ===")
    corr = pd.DataFrame(rets).corr()
    print(corr.round(3).to_string())
    print()

    print("=== 合并实验结果 ===")
    print(f"{'组合':<28} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>7} {'Calmar':>7} | "
          f"{'IS-Shp':>6} {'OOS-Shp':>7} {'OOS-AR':>7} {'OOS-DD':>7}")
    print("-" * 120)

    rows = []
    for w_main, w_zn, w_cu, label in PORTFOLIOS:
        if abs(w_main + w_zn + w_cu - 1.0) > 0.01:
            continue
        combo = w_main * main_ret + w_zn * zn_ret + w_cu * cu_ret
        st = stats(combo)
        rows.append({"组合": label, "w_主CTA": w_main, "w_ZN": w_zn, "w_CU": w_cu, **st["full"]})
        print(f"{label:<28} {st['full']['sharpe']:>7.3f} {st['full']['ann_ret']:>7.2%} "
              f"{st['full']['max_dd']:>7.2%} {st['full']['calmar']:>7.2f} | "
              f"{st['is']['sharpe']:>6.3f} {st['oos']['sharpe']:>7.3f} "
              f"{st['oos']['ann_ret']:>7.2%} {st['oos']['max_dd']:>7.2%}")

    # 推荐: OOS Sharpe最高的均衡组合
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.to_csv(REPORT_DIR / "cta_combination_results.csv", index=False)
        best = df.loc[df["sharpe"].idxmax()]
        print(f"\n★ 推荐组合 (全期Sharpe最高): {best['组合']}")
        print(f"  主CTA={best['w_主CTA']}, ZN={best['w_ZN']}, CU={best['w_CU']}")

        oos_best = df.loc[df["ann_ret"].idxmax()]
        print(f"\n★ 推荐组合 (OOS AnnRet最高): {oos_best['组合']}")


if __name__ == "__main__":
    main()
