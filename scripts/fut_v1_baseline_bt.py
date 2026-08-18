#!/usr/bin/env python3
"""fut_v1 基线回测 — 输出指标 + 净值曲线"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time, os

from scripts.common import (
    compute_metrics, run_strategy, print_metrics, print_yearly,
    plot_equity_curve, parse_args, get_products, DEFAULT_PRODUCTS,
)

OUT_DIR = "/home/ll/Public/QuantNodes/output"


def main():
    args = parse_args("fut_v1 Baseline Backtest")
    os.makedirs(args.output, exist_ok=True)

    products = get_products(args.quick)
    n = len(products)

    t0 = time.time()
    print(f"fut_v1 Baseline Backtest ({n} Products)")
    print("=" * 60)

    result, metrics = run_strategy(products=products, start=args.start, end=args.end)
    print(f"  Done in {time.time()-t0:.0f}s")

    # 打印指标
    print_metrics(metrics, f"fut_v1 Baseline ({n} Products)")

    # 打印年度分析
    print_yearly(result.nav)

    # 保存NAV
    nav_path = os.path.join(args.output, f"fut_v1_{n}p_nav.parquet")
    result.nav.to_frame("nav").to_parquet(nav_path)
    print(f"\n  NAV saved: {nav_path}")

    # 生成图表
    chart_path = os.path.join(args.output, f"fut_v1_{n}p_equity_curve.png")
    plot_equity_curve(result.nav, metrics, f"fut_v1 Baseline ({n} Products)", chart_path)

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
