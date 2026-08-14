#!/usr/bin/env python3
"""Regime-Adaptive 期权策略回测脚本.

用法:
    cd QuantNodes && python scripts/regime_adaptive_bt.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.options.iv_panel import load_iv_panel
from QuantNodes.strategy.cta.options.vol_signals import VolSignalGenerator
from QuantNodes.strategy.cta.options.strategies import get_strategy, list_strategies
# 触发 register 装饰器
import QuantNodes.strategy.cta.options.strategies.short_strangle
import QuantNodes.strategy.cta.options.strategies.iron_condor
import QuantNodes.strategy.cta.options.strategies.calendar_spread
import QuantNodes.strategy.cta.options.strategies.covered_call
import QuantNodes.strategy.cta.options.strategies.vertical_spread
import QuantNodes.strategy.cta.options.strategies.regime_adaptive
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap


def run_regime_backtest(product="RB", start="2023-01-01", end="2025-12-31"):
    """运行 Regime-Adaptive 回测."""
    print(f"Regime-Adaptive 回测 — {product} — {start} ~ {end}")
    print("=" * 60)

    t0 = time.time()

    # 1. 加载数据
    loader = CtaDataLoader()
    um = UnderlyingMap(loader)
    oe = OptionExpiry(loader)

    # 标的价格
    px = loader.load_continuous_main(product)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.set_index("trade_date")
    price_series = px["close"]

    # IV 面板
    iv_panel = load_iv_panel(start=start, end=end)

    # HV 面板 (从价格计算)
    log_ret = np.log(price_series / price_series.shift(1))
    hv_series = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    hv_panel = pd.DataFrame({product: hv_series}).dropna()

    # 2. 创建 VolSignalGenerator
    vol_gen = VolSignalGenerator(iv_panel, hv_panel, iv_pct_window=20)

    # 3. 创建策略
    strategy = get_strategy("regime_adaptive")

    # 4. 逐日模拟
    holdings = {}  # product → {"symbol", "qty", "expiry", "type"}
    dates = price_series.index[(price_series.index >= start) & (price_series.index <= end)]

    trades = []
    daily_pnl = []
    current_regime = None

    for date in dates:
        spot = price_series.loc[date]
        if np.isnan(spot) or spot <= 0:
            continue

        # 获取当前 regime
        regime = vol_gen.compute_regime(product, date)

        # 检查现有持仓: regime 变化或到期 → 平仓
        if product in holdings:
            h = holdings[product]
            should_exit = False
            # regime 变化时平仓
            if current_regime is not None and regime != current_regime:
                should_exit = True
            # 持仓超过20天平仓
            hold_days = (date - h["entry_date"]).days
            if hold_days >= 20:
                should_exit = True

            if should_exit:
                trades.append({
                    "date": date, "symbol": h["symbol"], "side": "close",
                    "qty": -h["qty"], "product": product,
                })
                holdings.pop(product)

        current_regime = regime

        # 入场: 无持仓时
        if product not in holdings:
            targets = strategy.select(
                product, date, spot, None,
                loader=loader, um=um, oe=oe, vol_signal_gen=vol_gen,
            )
            for sym, qty in targets.items():
                holdings[product] = {
                    "symbol": sym, "qty": qty, "type": "option",
                    "entry_date": date,
                }
                trades.append({
                    "date": date, "symbol": sym, "side": "open",
                    "qty": qty, "product": product, "regime": regime.name,
                })

        daily_pnl.append({
            "date": date, "spot": spot,
            "n_holdings": len(holdings),
            "regime": regime.name if regime is not None else "NONE",
        })

    elapsed = time.time() - t0

    # 5. 输出结果
    pnl_df = pd.DataFrame(daily_pnl).set_index("date")
    pnl_df["ret"] = pnl_df["spot"].pct_change()

    print(f"\n  数据点: {len(dates)}")
    print(f"  交易次数: {len(trades)}")
    print(f"  时间: {elapsed:.0f}s")

    # 信号分布
    signal_counts = {}
    for d in dates[:100]:  # 前100天采样
        regime = vol_gen.compute_regime(product, d)
        signal_counts[regime.name] = signal_counts.get(regime.name, 0) + 1
    print(f"\n  Regime 分布 (前100天):")
    for name, count in sorted(signal_counts.items()):
        print(f"    {name}: {count}")

    return pnl_df, trades


if __name__ == "__main__":
    for prod in ["RB", "I", "AG"]:
        try:
            run_regime_backtest(product=prod)
            print()
        except Exception as e:
            print(f"  ERROR: {e}\n")
