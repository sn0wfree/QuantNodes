#!/usr/bin/env python3
"""cta_m_save_nav.py — 保存M经纪持仓子策略NAV (合并实验输入).

复用 cta_broker_holding 的 build_factors + vol_target_backtest, 不改原脚本.
只保存 F3+F4+F5 组合 (IS/OOS稳健的子集) 的NAV到 reports/cta/m_broker_holding_nav.csv.
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "scripts")
from pathlib import Path
from cta_shared.factor_research import REPORT_DIR, vol_target_backtest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cta_broker_holding import build_factors, load_cn_close  # noqa: E402

IS_END = "2022-01-01"
Z_EXTREME = 1.0


def main():
    close = load_cn_close(["M"])
    factors = build_factors("M")
    factors = {k: v.reindex(close.index).ffill().shift(1) for k, v in factors.items()}  # PIT+1

    # F3+F4+F5组合, IS方向均正
    COST = 5.5  # M: 滑点3.3+手续费1.5
    sig = pd.Series(0.0, index=close.index)
    for col in ["F3_cr5_share", "F4_hedge_ratio", "F5_tail_head"]:
        sig += (np.sign(factors[col]) * (factors[col].abs() > Z_EXTREME).astype(float)).fillna(0)
    sig = np.sign(sig)

    res = vol_target_backtest(sig.to_frame("M"), close,
                              cost_bps={"M": COST}, target_vol=0.08)
    res["nav"].to_frame("nav").to_csv(REPORT_DIR / "m_broker_holding_nav.csv")
    print(f"M NAV saved: {res['nav'].index.min().date()}~{res['nav'].index.max().date()}, "
          f"全期Sharpe={res['sharpe']:.3f}")


import numpy as np  # noqa: E402

if __name__ == "__main__":
    main()
