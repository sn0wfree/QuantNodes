#!/usr/bin/env python3
"""Gamma Scalper 回测脚本.

用法:
    cd QuantNodes && python scripts/gamma_scalper_bt.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.gamma_scalper import (
    GammaScalper,
    GammaScalperConfig,
)


def main():
    cfg = GammaScalperConfig(
        products=["RB"],
        initial_cash=1_000_000.0,
        commission_rate=0.0001,
        margin_call=False,
        min_days=30,
        max_days=60,
        roll_days=7,
    )

    scalper = GammaScalper(cfg)

    t0 = time.time()
    result = scalper.run(start="2023-01-01", end="2025-12-31")
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Gamma Scalper — RB — 2023~2025")
    print(f"{'='*50}")
    print(f"  Sharpe:    {result.metrics.get('sharpe', 0):.3f}")
    print(f"  Ann Ret:   {result.metrics.get('ann_return', 0):.3f}")
    print(f"  Max DD:    {result.metrics.get('max_drawdown', 0):.3f}")
    print(f"  Trades:    {len(result.trades)}")
    print(f"  Time:      {elapsed:.0f}s")

    # P&L 分解
    pnl = scalper.pnl_summary()
    if pnl:
        print(f"\n--- P&L 分解 ---")
        for comp in ["theta", "gamma", "vega", "delta_hedge", "total"]:
            if comp in pnl:
                print(f"  {comp:12s}: {pnl[comp]['sum']:+.2f}")

    # Hedge 记录
    hr = scalper.hedge_records()
    if not hr.empty:
        print(f"\n--- Delta Hedge ---")
        print(f"  调仓次数: {len(hr)}")
        print(f"  总成本:   {hr['cost'].sum():.2f}")

    # Roll 记录
    rr = scalper.roll_records()
    if not rr.empty:
        print(f"\n--- Roll ---")
        print(f"  展期次数: {len(rr)}")

    return result


if __name__ == "__main__":
    main()
