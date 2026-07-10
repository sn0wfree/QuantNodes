# coding=utf-8
"""v6 单策略回测: 7 种风控组合对比.

Stage 26: v6 = v1.0 风控框架 + v5.1.1 选股 + v5.1.1 加权.

跑 52 ETF 池 (口径 A 含 5bp 成本), 给出:
- 全期业绩 (2018-2026)
- OOS 业绩 (2022-2026)
- 与 v5 / v5.1.1 / v3 / v1.0 对比
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QuantNodes.strategy.momentum_etf_rotation.v6 import run_v6_backtest, V6Config

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"
OOS_START = "2022-01-01"


def ann_return(nav):
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def sharpe(nav):
    rets = nav.pct_change().dropna()
    return float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0


def max_dd(nav):
    return float((nav / nav.cummax() - 1.0).min())


def metrics(nav):
    ar = ann_return(nav)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "sharpe": sharpe(nav),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
    }


def main():
    print(f"[data] 加载 OHLCV 面板 ({START} ~ {END}) ...")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_ohlcv = panel_ohlcv.loc[START:END]
    panel_close = panel_ohlcv.xs("close", axis=1, level=1)
    print(f"[data] {panel_close.shape[0]} 天 × {panel_close.shape[1]} codes")

    print("\n========= v6 三档回测 (口径 A 含 5bp 成本) =========")
    cases = [
        ("v6 无风控 (纯选股+加权)", False, False, False),
        ("v6 只 TF",              False, True,  False),
        ("v6 只 Cost",            False, False, True),
        ("v6 只 VT",              True,  False, False),
        ("v6 TF + Cost",          False, True,  True),
        ("v6 VT + Cost",          True,  False, True),
        ("v6 全风控 (VT+TF+Cost)", True, True, True),
    ]

    results = {}
    for name, vt, tf, cost in cases:
        nav = run_v6_backtest(
            panel_close, panel_ohlcv, V6Config(),
            apply_vol_targeting=vt, apply_trend_filter=tf, apply_cost_model=cost,
        )
        full = metrics(nav)
        oos = metrics(nav.loc[OOS_START:])
        results[name] = (nav, full, oos)
        print(f"  {name}: 全期 Ann={full['ann_return']*100:.2f}% Sharpe={full['sharpe']:.2f} "
              f"DD={full['max_dd']*100:.2f}% Calmar={full['calmar']:.3f} | "
              f"OOS Ann={oos['ann_return']*100:.2f}% Sharpe={oos['sharpe']:.2f} "
              f"DD={oos['max_dd']*100:.2f}% Calmar={oos['calmar']:.3f}")

    # 推荐 v6: TF+Cost 写入 parquet
    recommended_nav = results["v6 TF + Cost"][0]
    recommended_name = "v6 行业量价 (TF+Cost)"

    # 与 v5 / v5.1.1 对比
    print("\n========= v6 (TF+Cost) vs v5 / v5.1.1 对比 =========")
    try:
        v51_nav = pd.read_parquet(REPO / "reports/momentum_etf_rotation/combo/unified_v1v5_navs_calA.parquet")["v5.1 量价 (逆波动)"]
        m51 = metrics(v51_nav)
        m51_oos = metrics(v51_nav.loc[OOS_START:])
        print(f"  v5.1.1 baseline: 全期 Calmar={m51['calmar']:.3f}  OOS Calmar={m51_oos['calmar']:.3f}")
    except Exception as e:
        print(f"  v5.1.1 加载失败: {e}")

    # 保存推荐 v6 NAV
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_path = out_dir / "v6_navs.parquet"

    save_df = {"v6 TF+Cost": recommended_nav}
    # 其他风控档都保存用于消融
    save_df["v6 无风控"] = results["v6 无风控 (纯选股+加权)"][0]
    save_df["v6 只 TF"] = results["v6 只 TF"][0]
    save_df["v6 只 Cost"] = results["v6 只 Cost"][0]
    save_df["v6 只 VT"] = results["v6 只 VT"][0]
    save_df["v6 VT+Cost"] = results["v6 VT + Cost"][0]
    save_df["v6 全风控"] = results["v6 全风控 (VT+TF+Cost)"][0]
    pd.DataFrame(save_df).to_parquet(out_path)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
