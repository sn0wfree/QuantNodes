"""scripts/cmf/test_vix_adaptive.py — 阶段3: VIX自适应仓位策略.

策略: 根据VIX水平动态调整基础仓位规模
- VIX高 → 减仓 (市场波动大,降低风险敞口)
- VIX低 → 加仓 (市场平静,放大敞口)

多种VIX→仓位映射函数:
1. 比率缩放: position = base * (target_vix / vix_now).clip(min, max)
2. Z-score缩放: position = base * exp(-zscore * power).clip(min, max)
3. 百分位缩放: position = base * f(vix_percentile)
4. 平滑缩放: position = base * smooth_step(vix_low, vix_high, vix_now)

回测范围: 2020-2025
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


def load_base():
    """加载基础数据 (金价, VIX)."""
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
    """计算原始信号 (未经VIX缩放)."""
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


# ====== VIX → 仓位缩放因子映射函数 ======

def scale_ratio(vix, target_vix=15.0, floor=0.3, ceil=1.3):
    """比率缩放: position = base * clip(target_vix / vix, floor, ceil).

    VIX↑ → 缩放因子↓
    VIX↓ → 缩放因子↑
    """
    factor = (target_vix / vix).clip(floor, ceil)
    return factor


def scale_zscore(vix, lookback=252, target=15.0, power=1.0, floor=0.3, ceil=1.3):
    """Z-score缩放: position = base * exp(-zscore * power).clip(floor, ceil).

    使用滚动均值/标准差计算Z-score。
    """
    ma = vix.rolling(lookback).mean()
    std = vix.rolling(lookback).std()
    zscore = (vix - target) / std.replace(0, np.nan)
    factor = np.exp(-zscore * power).clip(floor, ceil)
    return factor.fillna(1.0)


def scale_percentile(vix, lookback=252, low_pct=0.25, high_pct=1.3):
    """百分位缩放: VIX百分位低→加仓,百分位高→减仓."""
    pct = vix.rolling(lookback).rank(pct=True)
    factor = (low_pct + (high_pct - low_pct) * (1 - pct)).clip(0.3, 1.5)
    return factor.fillna(1.0)


def scale_smooth(vix, vix_low=12.0, vix_high=30.0, max_scale=1.3, min_scale=0.3):
    """平滑线性缩放: VIX<vix_low→max_scale, VIX>vix_high→min_scale, 中间线性插值."""
    factor = max_scale - (vix - vix_low) / (vix_high - vix_low) * (max_scale - min_scale)
    return factor.clip(min_scale, max_scale)


SCALERS = {
    "ratio": scale_ratio,
    "zscore": scale_zscore,
    "percentile": scale_percentile,
    "smooth": scale_smooth,
}


def run_vix_adaptive_tests():
    close, vix = load_base()
    raw = get_raw_signal(close, vix)

    r_base = calc_metrics(raw, close, "2020-01-01", "2025-12-31")
    print(f"=== 基础策略 (无VIX自适应) ===")
    print(f"  年化={r_base['ann_ret']*100:.2f}%, Sharpe={r_base['sharpe']:.2f}, "
          f"MaxDD={r_base['max_dd']*100:.2f}%, Calmar={r_base['calmar']:.2f}")
    print(f"  换手={(raw.diff()!=0).sum()}\n")

    print(f"=== VIX自适应仓位策略 (2020-2025) ===")
    print(f'{"配置":<55} {"年化":>8} {"Sharpe":>7} {"MaxDD":>8} {"Calmar":>7} {"换手":>7}')

    # ====== 测试比率缩放 ======
    for target_vix in [13, 15, 17, 19, 21]:
        for floor in [0.3, 0.5]:
            for ceil in [1.2, 1.5, 2.0]:
                factor = scale_ratio(vix, target_vix=target_vix, floor=floor, ceil=ceil)
                pos = (raw * factor).clip(0, 1.2)
                r = calc_metrics(pos, close, "2020-01-01", "2025-12-31")
                print(f'  ratio target={target_vix}/[{floor},{ceil}]: '
                      f'年化={r["ann_ret"]*100:>6.2f}%, '
                      f'Sharpe={r["sharpe"]:>5.2f}, '
                      f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                      f'Calmar={r["calmar"]:>5.2f}, '
                      f'换手={r["turnover"]:>4.1f}')

    print()
    # ====== 测试Z-score缩放 ======
    for lookback in [126, 252]:
        for power in [0.3, 0.5, 0.8, 1.0]:
            for floor, ceil in [(0.3, 1.3), (0.5, 1.5)]:
                factor = scale_zscore(vix, lookback=lookback, target=15.0, power=power, floor=floor, ceil=ceil)
                pos = (raw * factor).clip(0, 1.2)
                r = calc_metrics(pos, close, "2020-01-01", "2025-12-31")
                print(f'  zscore lb={lookback}/p={power}/[{floor},{ceil}]: '
                      f'年化={r["ann_ret"]*100:>6.2f}%, '
                      f'Sharpe={r["sharpe"]:>5.2f}, '
                      f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                      f'Calmar={r["calmar"]:>5.2f}, '
                      f'换手={r["turnover"]:>4.1f}')

    print()
    # ====== 测试百分位缩放 ======
    for lookback in [126, 252]:
        for low_pct, high_pct in [(0.5, 1.0), (0.3, 1.3), (0.3, 1.5)]:
            factor = scale_percentile(vix, lookback=lookback, low_pct=low_pct, high_pct=high_pct)
            pos = (raw * factor).clip(0, 1.2)
            r = calc_metrics(pos, close, "2020-01-01", "2025-12-31")
            print(f'  pctile lb={lookback}/[{low_pct},{high_pct}]: '
                  f'年化={r["ann_ret"]*100:>6.2f}%, '
                  f'Sharpe={r["sharpe"]:>5.2f}, '
                  f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                  f'Calmar={r["calmar"]:>5.2f}, '
                  f'换手={r["turnover"]:>4.1f}')

    print()
    # ====== 测试平滑线性缩放 ======
    for vix_low, vix_high in [(10, 25), (12, 28), (15, 30), (12, 35)]:
        for max_scale in [1.2, 1.5]:
            for min_scale in [0.3, 0.5]:
                factor = scale_smooth(vix, vix_low=vix_low, vix_high=vix_high,
                                      max_scale=max_scale, min_scale=min_scale)
                pos = (raw * factor).clip(0, 1.2)
                r = calc_metrics(pos, close, "2020-01-01", "2025-12-31")
                print(f'  smooth [{vix_low},{vix_high}]/[{min_scale},{max_scale}]: '
                      f'年化={r["ann_ret"]*100:>6.2f}%, '
                      f'Sharpe={r["sharpe"]:>5.2f}, '
                      f'MaxDD={r["max_dd"]*100:>6.2f}%, '
                      f'Calmar={r["calmar"]:>5.2f}, '
                      f'换手={r["turnover"]:>4.1f}')


if __name__ == "__main__":
    run_vix_adaptive_tests()
