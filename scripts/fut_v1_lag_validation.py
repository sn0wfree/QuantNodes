#!/usr/bin/env python3
"""前视偏差实证验证: execution_lag=1 vs execution_lag=2

逻辑:
- 若策略因果正确 (无前视), 多滞后一天不应显著改变 Sharpe
- 若存在前视, 增加滞后会显著降低 Sharpe (信号依赖未来信息)

用法: python scripts/fut_v1_lag_validation.py [--quick]
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time

from scripts.common import (
    compute_metrics, run_strategy, print_comparison, parse_args, get_products,
)


def main():
    args = parse_args("前视偏差实证验证")
    products = get_products(args.quick)
    n = len(products)

    print("=" * 72)
    print(f"前视偏差实证验证 ({n}品种)")
    print(f"区间: {args.start} ~ {args.end}")
    print("=" * 72)
    print("逻辑: 若因果正确, lag=2 的 Sharpe 不应显著低于 lag=1")
    print("      若存在前视, lag=2 的 Sharpe 会显著下降")
    print()

    results = {}
    navs = {}

    for lag in [1, 2]:
        t0 = time.time()
        print(f"运行 execution_lag={lag} ...", flush=True)
        result, metrics = run_strategy(
            products=products, start=args.start, end=args.end,
            execution_lag=lag,
        )
        results[f"lag={lag}"] = metrics
        navs[lag] = result.nav
        print(f"  Done ({time.time()-t0:.0f}s)")
        print(f"  Sharpe={metrics['Sharpe']:.2f}  MaxDD={metrics['MaxDD']:.1%}")

    # 对比
    print_comparison(results, title="Lag Comparison")

    s1, s2 = results["lag=1"]["Sharpe"], results["lag=2"]["Sharpe"]
    print()
    if s2 >= s1 * 0.9:
        print(f"✅ 判定: lag=2 Sharpe({s2:.2f}) ≥ lag=1 Sharpe({s1:.2f}) 的 90%")
        print("   → 策略因果正确, shift(-1) 无前视偏差")
    else:
        print(f"❌ 判定: lag=2 Sharpe({s2:.2f}) < lag=1 Sharpe({s1:.2f}) 的 90%")
        print("   → 信号依赖未来信息, 存在前视偏差!")

    # 保存NAV
    import pandas as pd
    out = pd.DataFrame({f"lag1": navs[1], f"lag2": navs[2]})
    path = f"{args.output}/fut_v1_lag_validation.parquet"
    out.to_parquet(path)
    print(f"\nNAV saved: {path}")


if __name__ == "__main__":
    main()
