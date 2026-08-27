"""scripts/cmf/test_oos_validation.py — 阶段5: out-of-sample验证.

对test_vix_refine.py中Top 20配置,在2010-2019数据上验证是否过拟合.
对比 2010-2019 (OOS) vs 2020-2025 (IS).

回测范围:
- OOS: 2010-01-01 → 2019-12-31
- IS:  2020-01-01 → 2025-12-31
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")

OOS_START = "2010-01-01"
OOS_END = "2019-12-31"
IS_START = "2020-01-01"
IS_END = "2025-12-31"


def load_base():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    vix_df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].reindex(close.index).ffill()
    return close, vix


def get_raw_signal(close, vix):
    vix_pct = vix.rolling(252).rank(pct=True)
    trend = compute_trend_bnh_signal(
        trailing_stop=0.08, freq="daily"
    )["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    return final_base.clip(0, 1.2)


def scale_zscore(vix, lookback, target, power, floor, ceil):
    ma = vix.rolling(lookback).mean()
    std = vix.rolling(lookback).std()
    zscore = (vix - target) / std.replace(0, np.nan)
    factor = np.exp(-zscore * power).clip(floor, ceil)
    return factor.fillna(1.0)


def scale_smooth(vix, vix_low, vix_high, min_scale, max_scale):
    factor = max_scale - (vix - vix_low) / (vix_high - vix_low) * (max_scale - min_scale)
    return factor.clip(min_scale, max_scale)


def scale_ratio(vix, target_vix, floor, ceil):
    factor = (target_vix / vix).clip(floor, ceil)
    return factor


# Top 20 配置 (来自test_vix_refine.py)
TOP_CONFIGS = [
    ("zscore", dict(lookback=63, target=16, power=0.6, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=63, target=16, power=0.6, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=90, target=16, power=0.9, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=0.8, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=63, target=16, power=0.7, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=0.8, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=90, target=16, power=1.0, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=0.9, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=63, target=16, power=0.7, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=63, target=16, power=0.8, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=1.1, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=1.0, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=63, target=16, power=0.9, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=63, target=16, power=0.8, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=90, target=16, power=1.2, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=1.1, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=63, target=16, power=1.0, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=90, target=16, power=0.7, floor=0.3, ceil=1.5)),
    ("zscore", dict(lookback=63, target=16, power=0.9, floor=0.3, ceil=1.3)),
    ("zscore", dict(lookback=90, target=17, power=1.2, floor=0.3, ceil=1.5)),
    # 加几个非zscore的对照 (虽然没进Top)
    ("smooth", dict(vix_low=10, vix_high=25, min_scale=0.3, max_scale=1.5)),
    ("smooth", dict(vix_low=12, vix_high=28, min_scale=0.3, max_scale=1.5)),
    ("ratio", dict(target_vix=17, floor=0.5, ceil=1.5)),
    ("ratio", dict(target_vix=19, floor=0.5, ceil=1.5)),
]


def evaluate(pos, close, start, end):
    r = calc_metrics(pos, close, start, end)
    return {
        "ann_ret": r["ann_ret"],
        "sharpe": r["sharpe"],
        "max_dd": r["max_dd"],
        "calmar": r["calmar"],
        "turnover": r["turnover"],
    }


def config_to_str(method, p):
    if method == "zscore":
        return f"zscore_lb{p['lookback']}_p{p['power']}_t{p['target']}_[{p['floor']},{p['ceil']}]"
    elif method == "smooth":
        return f"smooth_[{p['vix_low']},{p['vix_high']}]/[{p['min_scale']},{p['max_scale']}]"
    elif method == "ratio":
        return f"ratio_t{p['target_vix']}_[{p['floor']},{p['ceil']}]"


def run_oos_validation():
    close, vix = load_base()
    raw = get_raw_signal(close, vix)

    # 基础策略OOS vs IS
    base_oos = evaluate(raw, close, OOS_START, OOS_END)
    base_is = evaluate(raw, close, IS_START, IS_END)

    print("=" * 110)
    print("基础策略对比 (无VIX自适应)")
    print("=" * 110)
    print(f'{"周期":<20} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')
    print(f'{"OOS 2010-2019":<20} {base_oos["ann_ret"]*100:>7.2f}% '
          f'{base_oos["sharpe"]:>6.2f} {base_oos["max_dd"]*100:>7.2f}% '
          f'{base_oos["calmar"]:>6.2f} {base_oos["turnover"]:>6.1f}')
    print(f'{"IS 2020-2025":<20} {base_is["ann_ret"]*100:>7.2f}% '
          f'{base_is["sharpe"]:>6.2f} {base_is["max_dd"]*100:>7.2f}% '
          f'{base_is["calmar"]:>6.2f} {base_is["turnover"]:>6.1f}')

    print("\n" + "=" * 140)
    print("Top 24配置 OOS验证")
    print("=" * 140)
    print(f'{"配置":<55} '
          f'{"OOS年化":>8} {"OOSSharpe":>9} {"OOSMaxDD":>9} {"OOSCalmar":>10} | '
          f'{"IS年化":>8} {"ISSharpe":>8} {"ISMaxDD":>8} {"ISCalmar":>9}')
    print("-" * 140)

    results = []
    for method, p in TOP_CONFIGS:
        if method == "zscore":
            factor = scale_zscore(vix, **p)
        elif method == "smooth":
            factor = scale_smooth(vix, **p)
        elif method == "ratio":
            factor = scale_ratio(vix, **p)
        pos = (raw * factor).clip(0, 1.2)

        oos = evaluate(pos, close, OOS_START, OOS_END)
        is_ = evaluate(pos, close, IS_START, IS_END)

        cs = config_to_str(method, p)
        results.append((cs, oos, is_))
        print(f'{cs:<55} '
              f'{oos["ann_ret"]*100:>7.2f}% {oos["sharpe"]:>8.2f} '
              f'{oos["max_dd"]*100:>8.2f}% {oos["calmar"]:>9.2f} | '
              f'{is_["ann_ret"]*100:>7.2f}% {is_["sharpe"]:>7.2f} '
              f'{is_["max_dd"]*100:>7.2f}% {is_["calmar"]:>8.2f}')

    print("\n" + "=" * 110)
    print("结论判定")
    print("=" * 110)

    overfit_count = 0
    robust_count = 0
    for cs, oos, is_ in results:
        oos_ok = (oos["max_dd"] > base_oos["max_dd"]) or (oos["calmar"] > base_oos["calmar"])
        is_ok = (is_["max_dd"] > base_is["max_dd"]) or (is_["calmar"] > base_is["calmar"])
        if oos_ok and is_ok:
            robust_count += 1
        elif not oos_ok and is_ok:
            overfit_count += 1

    print(f"稳健配置 (OOS+IS均改善): {robust_count}/{len(results)}")
    print(f"过拟合配置 (IS改善但OOS恶化): {overfit_count}/{len(results)}")

    # 综合排序: OOS+IS平均Calmar
    print("\n" + "=" * 110)
    print("综合排序 (OOS+IS Calmar均值):")
    print("=" * 110)
    results.sort(key=lambda x: (x[1]["calmar"] + x[2]["calmar"]) / 2, reverse=True)
    print(f'{"配置":<55} {"OOSCalmar":>9} {"ISCalmar":>9} {"均值":>7}')
    print("-" * 110)
    for cs, oos, is_ in results[:10]:
        avg = (oos["calmar"] + is_["calmar"]) / 2
        print(f'{cs:<55} {oos["calmar"]:>8.2f} {is_["calmar"]:>8.2f} {avg:>6.2f}')


if __name__ == "__main__":
    run_oos_validation()
