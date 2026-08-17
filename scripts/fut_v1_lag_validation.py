#!/usr/bin/env python3
"""前视偏差实证验证: execution_lag=1 vs execution_lag=2

逻辑:
- 若策略因果正确 (无前视), 多滞后一天不应显著改变 Sharpe
- 若存在前视, 增加滞后会显著降低 Sharpe (信号依赖未来信息)

用法: python scripts/fut_v1_lag_validation.py [--quick]
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time, argparse
import numpy as np, pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS,
)


def compute_metrics(nav):
    r = nav.pct_change().dropna()
    if len(r) == 0:
        return {}
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann_ret = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 0 else 0
    return {"AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe,
            "MaxDD": dd, "Calmar": calmar}


def run_lag(products, start, end, lag, agg):
    cfg = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=lag, aggregation=agg,
    )
    strat = FactorModuleComboStrategy(products=products, config=cfg, start=start, end=end)
    result = strat.run()
    return compute_metrics(result.nav), result.nav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--agg", default="tvpr", choices=["tvpr", "regression", "inv_vol"])
    ap.add_argument("--quick", action="store_true", help="只用10品种快速验证")
    args = ap.parse_args()

    products = DEFAULT_PRODUCTS if not args.quick else ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    n = len(products)

    print("=" * 72)
    print(f"前视偏差实证验证 (aggregation={args.agg}, {n}品种)")
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
        m, nav = run_lag(products, args.start, args.end, lag, args.agg)
        results[lag] = m
        navs[lag] = nav
        print(f"  Done ({time.time()-t0:.0f}s)")
        print(f"  AnnRet={m['AnnRet']:>6.1%}  Sharpe={m['Sharpe']:>6.2f}  "
              f"MaxDD={m['MaxDD']:>6.1%}  Calmar={m['Calmar']:>5.2f}")

    print()
    print("-" * 72)
    print("对比结果:")
    print(f"  {'':>12} {'lag=1':>10} {'lag=2':>10} {'变化':>10}")
    print("  " + "-" * 44)
    for key in ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar"]:
        v1 = results[1].get(key, 0)
        v2 = results[2].get(key, 0)
        if key in ("AnnRet", "AnnVol", "MaxDD"):
            print(f"  {key:>12}: {v1:>7.1%} {v2:>7.1%} {v2-v1:+7.1%}")
        else:
            print(f"  {key:>12}: {v1:>7.2f} {v2:>7.2f} {v2-v1:+7.2f}")

    s1, s2 = results[1]["Sharpe"], results[2]["Sharpe"]
    print()
    if s2 >= s1 * 0.9:
        print(f"✅ 判定: lag=2 Sharpe({s2:.2f}) ≥ lag=1 Sharpe({s1:.2f}) 的 90%")
        print("   → 策略因果正确, shift(-1) 无前视偏差")
    else:
        print(f"❌ 判定: lag=2 Sharpe({s2:.2f}) < lag=1 Sharpe({s1:.2f}) 的 90%")
        print("   → 信号依赖未来信息, 存在前视偏差!")

    # 保存NAV供对比
    out = pd.DataFrame({f"lag1_{args.agg}": navs[1], f"lag2_{args.agg}": navs[2]})
    path = f"output/fut_v1_lag_validation_{args.agg}.parquet"
    out.to_parquet(path)
    print(f"\nNAV saved: {path}")


if __name__ == "__main__":
    main()
