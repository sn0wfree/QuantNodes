"""scripts/cmf/test_macro_regime_verify.py — 生产集成回归验证.

验证 gold_trend_bnh.compute_macro_regime_signal 与
scripts/cmf/test_p23_macro_gate.py 中内存计算结果一致 (双窗口+四子周期).

期望值 (rt w21/p2.5/0.15, gvz lb90/t18/p0.8):
    OOS 2010-2019: 年化~6.14% MaxDD~-29.40% Calmar~0.21
    IS  2020-2026: 年化~17.73% MaxDD~-8.59% Calmar~2.06
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

import pandas as pd

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
    compute_trend_bnh_signal,
)

WINDOWS = [
    ("OOS 2010-2019", "2010-01-01", "2019-12-31"),
    ("IS  2020-2026", "2020-01-01", "2026-08-25"),
    ("子周期 10-14", "2010-01-01", "2014-12-31"),
    ("子周期 15-19", "2015-01-01", "2019-12-31"),
    ("子周期 20-23", "2020-01-01", "2023-12-31"),
    ("子周期 24-26YTD", "2024-01-01", "2026-08-25"),
]


def main():
    sig = compute_macro_regime_signal()
    close = sig["close"]
    pos = sig["position"]

    # 基线对照 (旧策略)
    base_old = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")
    pos_old = base_old["position"].reindex(close.index).fillna(0)

    print("=" * 96)
    print("compute_macro_regime_signal() 生产函数验证")
    print("=" * 96)
    print(f'{"窗口":<16} {"策略年化":>9} {"Sharpe":>7} {"MaxDD":>9} {"Calmar":>7} | '
          f'{"旧基线年化":>10} {"MaxDD":>9} {"Calmar":>7}')
    print("-" * 96)
    for name, s, e in WINDOWS:
        r = calc_metrics(pos, close, s, e)
        r_old = calc_metrics(pos_old, close, s, e)
        print(f'{name:<16} {r["ann_ret"]*100:>8.2f}% {r["sharpe"]:>7.2f} '
              f'{r["max_dd"]*100:>8.2f}% {r["calmar"]:>7.2f} | '
              f'{r_old["ann_ret"]*100:>9.2f}% {r_old["max_dd"]*100:>8.2f}% '
              f'{r_old["calmar"]:>7.2f}')

    # 回归锚点检查 (与 test_p23_macro_gate.py 强门控组合的偏差容差)
    anchors = {
        "OOS 2010-2019": (0.0614, -0.2940),
        "IS  2020-2026": (0.1773, -0.0859),
    }
    print("\n回归锚点 (容差 ±1.5%):")
    ok = True
    for name, s, e in WINDOWS[:2]:
        r = calc_metrics(pos, close, s, e)
        exp_ann, exp_dd = anchors[name]
        d_ann = abs(r["ann_ret"] - exp_ann)
        d_dd = abs(r["max_dd"] - exp_dd)
        passed = d_ann < 0.015 and d_dd < 0.015
        ok &= passed
        status = "PASS" if passed else f"FAIL (Δann={d_ann*100:.2f}%, Δdd={d_dd*100:.2f}%)"
        print(f"  {name}: 实际 ann={r['ann_ret']*100:.2f}% dd={r['max_dd']*100:.2f}% → {status}")

    print(f"\n{'全部通过' if ok else '存在失败项'}")


if __name__ == "__main__":
    main()
