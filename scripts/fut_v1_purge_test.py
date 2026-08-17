#!/usr/bin/env python3
"""前视偏差决定性测试: 清洗版beta vs 当前版beta

原理:
- 当前版: Y = pct_change().shift(-1)
    → beta[τ] 拟合用 Y[0:τ-1], 含 Y[τ-1]=r[τ] (今天收盘 p[τ])
- 清洗版: Y = pct_change() (无shift)
    → beta[τ] 拟合用 r[0:τ-1], 末项 r[τ-1] (昨天收盘 p[τ-1])
    → 完全排除今天收益对 beta 的影响

判定:
- 若清洗版 Sharpe ≈ 当前版 → shift(-1) 无前视 (今天收益可用, 信号 T+1 执行)
- 若清洗版 Sharpe 显著更低 → 原版 beta 依赖今天收益 = 前视

用法: python scripts/fut_v1_purge_test.py [--quick]
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time, argparse
import numpy as np, pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator


class PurgedTVPR(FactorModuleComboStrategy):
    """清洗版 TVPR: beta 拟合完全排除当天收益"""

    def _agg_tvpr(self, panels):
        cfg = self._config
        names = list(panels)
        cols = list(self._price_panel.columns)
        X = np.stack([
            panels[n].reindex(columns=cols).values for n in names
        ], axis=-1)
        # 清洗版: 无 shift → beta[τ] 只用到 r[τ-1] (昨天收盘)
        Y = self._price_panel.pct_change()
        beta = tvpr_estimator(
            Y, X,
            lambda_tv=cfg.tvpr_lambda_tv,
            lambda_l1=cfg.tvpr_lambda_l1,
            method="expanding",
            min_history=cfg.tvpr_min_history,
            step=cfg.agg_update,
        )
        combo = np.einsum("tni,ti->tn", X, beta.values)
        return pd.DataFrame(combo, index=self._price_panel.index, columns=cols)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    products = DEFAULT_PRODUCTS if not args.quick else ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    n = len(products)

    def make_cfg():
        return ComboConfig(
            initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
            momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
            module_weighting="inv_vol", continuous=True, cont_z_window=252,
            cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
        )

    print("=" * 72)
    print(f"清洗版beta测试 ({n}品种)")
    print(f"区间: {args.start} ~ {args.end}")
    print("=" * 72)
    print("当前版: Y=shift(-1)  → beta[τ] 用今天收益 r[τ] (p[τ])")
    print("清洗版: Y=shift(0)   → beta[τ] 只用昨天收益 r[τ-1] (p[τ-1])")
    print()

    # 当前版
    t0 = time.time()
    print("运行当前版 (shift -1)...", flush=True)
    s1 = FactorModuleComboStrategy(products=products, config=make_cfg(), start=args.start, end=args.end)
    r1 = s1.run()
    m1 = compute_metrics(r1.nav)
    print(f"  Done ({time.time()-t0:.0f}s)  Sharpe={m1['Sharpe']:.2f} AnnRet={m1['AnnRet']:.1%}")

    # 清洗版
    t0 = time.time()
    print("运行清洗版 (shift 0)...", flush=True)
    s2 = PurgedTVPR(products=products, config=make_cfg(), start=args.start, end=args.end)
    r2 = s2.run()
    m2 = compute_metrics(r2.nav)
    print(f"  Done ({time.time()-t0:.0f}s)  Sharpe={m2['Sharpe']:.2f} AnnRet={m2['AnnRet']:.1%}")

    print()
    print("-" * 60)
    print("对比:")
    print(f"  {'':>12} {'当前版(shift-1)':>16} {'清洗版(shift0)':>16} {'变化':>10}")
    print("  " + "-" * 54)
    for key in ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar"]:
        v1 = m1.get(key, 0)
        v2 = m2.get(key, 0)
        if key in ("AnnRet", "AnnVol", "MaxDD"):
            print(f"  {key:>12}: {v1:>13.1%} {v2:>13.1%} {v2-v1:+8.1%}")
        else:
            print(f"  {key:>12}: {v1:>13.2f} {v2:>13.2f} {v2-v1:+8.2f}")

    s_cur, s_pur = m1["Sharpe"], m2["Sharpe"]
    print()
    if s_pur >= s_cur * 0.9:
        print(f"✅ 判定: 清洗版 Sharpe({s_pur:.2f}) ≥ 当前版({s_cur:.2f}) 的 90%")
        print("   → beta 用到今天收益 r[τ] 对业绩影响很小, shift(-1) 无实质前视")
        print("   → 结论: 原代码正确, 保持 shift(-1)")
    else:
        print(f"❌ 判定: 清洗版 Sharpe({s_pur:.2f}) < 当前版({s_cur:.2f}) 的 90%")
        print("   → beta 依赖今天收益, 存在前视偏差, 需修复 shift(-1)")

    pd.DataFrame({"current": r1.nav, "purged": r2.nav}).to_parquet(
        f"output/fut_v1_purge_test_{'quick' if args.quick else 'full'}.parquet")


if __name__ == "__main__":
    main()
