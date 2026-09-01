#!/usr/bin/env python3
"""refresh_and_track.py — 一键执行: 数据刷新 + 生产仓位查询.

数据刷新:
  Step 1: AU期货数据增量拉取 (fetch_gold_contracts.py)
  Step 2: FRED数据刷新 (fetch_fred_supplements.py, 失败跳过)
  Step 3: 重建主力合约表 (detect_gold_main.py)
  Step 4: 重建adj价格 (build_gold_smooth.py)

仓位查询:
  Step 5: 生产仓位计算 + 输出状态

用法:
    python3 scripts/cmf/refresh_and_track.py            # 一键刷新+查询
    python3 scripts/cmf/refresh_and_track.py --no-refresh  # 仅查询(跳过数据拉取)
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts/cmf")

import argparse

import numpy as np
import pandas as pd


def step1_fetch_au() -> bool:
    """增量拉取AU期货数据."""
    from fetch_gold_contracts import main as fetch_main
    fetch_main()
    return True


def step2_fetch_fred() -> bool:
    """拉取FRED数据 (失败跳过)."""
    try:
        from fetch_fred_supplements import main as fetch_main
        fetch_main()
        return True
    except Exception as e:
        print(f"  ⚠️ FRED刷新失败, 跳过: {e}")
        return False


def step3_rebuild_main() -> bool:
    """重建主力合约表."""
    from detect_gold_main import main as rebuild_main
    rebuild_main()
    return True


def step4_rebuild_smooth() -> bool:
    """重建adj价格."""
    from build_gold_smooth import main as rebuild_main
    rebuild_main()
    return True


def step5_compute_position() -> dict:
    """计算生产仓位 + 输出状态."""
    from QuantNodes.strategy.commodity_fundamental.gold import (
        build_production_position, position_state, load_au_adj_ohlcv
    )

    close, open_price = load_au_adj_ohlcv()
    pos = build_production_position()
    st = position_state(pos, close)

    # 历史业绩 (简化: 链路与gold_track.py一致)
    r = (pos.shift(1) * open_price.pct_change().shift(-1)).dropna() - pos.diff().abs() * 0.0005
    r = r.dropna()
    nav = (1 + r).cumprod()
    n = len(r)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    sh = cagr / (r.std() * np.sqrt(252))
    dd = (nav / nav.cummax() - 1).min()

    return {
        "date": st["date"],
        "position": st["position"],
        "pos_change": st["pos_change"],
        "state": st["state"],
        "daily_ret": st["daily_ret"],
        "cagr": cagr,
        "sharpe": sh,
        "maxdd": dd,
    }


def main():
    parser = argparse.ArgumentParser(description="黄金生产策略一键执行")
    parser.add_argument("--no-refresh", action="store_true",
                        help="跳过数据刷新, 仅查询仓位")
    args = parser.parse_args()

    print("=" * 70)
    print("黄金生产策略一键执行")
    print("=" * 70)

    if not args.no_refresh:
        # Step 1: AU期货
        print("\n[1/5] 拉取AU期货数据...")
        try:
            step1_fetch_au()
            print("  ✓ 完成")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            print("  停止: 无AU数据无法计算信号")
            return

        # Step 2: FRED (失败跳过)
        print("\n[2/5] 拉取FRED数据...")
        fred_ok = step2_fetch_fred()
        if fred_ok:
            print("  ✓ 完成")
        # else: 已在step2内打印跳过信息

        # Step 3: 重建主力合约
        print("\n[3/5] 重建主力合约表...")
        try:
            step3_rebuild_main()
            print("  ✓ 完成")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            print("  停止: 无主力合约表无法计算信号")
            return

        # Step 4: 重建adj价格
        print("\n[4/5] 重建adj价格...")
        try:
            step4_rebuild_smooth()
            print("  ✓ 完成")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            print("  停止: 无adj价格无法计算信号")
            return
    else:
        print("\n[跳过数据刷新]")

    # Step 5: 计算仓位
    print("\n[5/5] 计算仓位...")
    try:
        result = step5_compute_position()
    except Exception as e:
        print(f"  ✗ 计算失败: {e}")
        return

    # 输出
    print("\n" + "=" * 70)
    print("生产策略 (B3) 最新持仓")
    print("=" * 70)
    print(f"数据截止: {result['date'].date()}")
    print(f"当前仓位: {result['position']:+.4f} ({result['position']*100:+.2f}%)")
    print(f"风控状态: {result['state']}")
    print(f"今日动作: {result['pos_change']:+.4f} ({'+' if result['pos_change']>0 else ''}{result['pos_change']*100:+.2f}%)")
    print(f"日收益: {result['daily_ret']:+.2%}")
    print()
    print(f"历史业绩 (2018-2026, O2O+5bps):")
    print(f"  CAGR={result['cagr']:+.2%}  Sharpe={result['sharpe']:.2f}  MaxDD={result['maxdd']:+.2%}")


if __name__ == "__main__":
    main()
