#!/usr/bin/env python3
"""前视偏差决定性测试 v3: 排除最近观测点的干净清洗

⚠️ v1 教训 (2026-08-17):
   组合策略的 _agg_tvpr 由各 module 实例调用, 组合层子类 override
   不会被调用 → v1 测试两组跑的都是原代码, 结果0差异是假象.
   必须 patch _ModuleStrategy._agg_tvpr (所有 module 的基类).

v3 正确清洗:
- 当前版: Y=shift(-1), beta[τ] 拟合用全部 Y[0:τ-1] (含 (X[τ-1], r[τ]))
- 清洗版: Y=shift(-1) 不变, 但 beta[τ] 拟合用 Y[0:τ-2] (排除 (X[τ-1], r[τ]))
  → beta[τ] 只用 ≤ τ-2 的数据 (截至昨天收盘 p[τ-1])
  → 目标语义不变(仍预测下一日收益), 只是不用今天收益 r[τ]

用法: python scripts/fut_v1_purge_test.py [--quick]
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time
import numpy as np, pandas as pd

import QuantNodes.strategy.cta.strategies.factor_modules as fm
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS, _ModuleStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    tvpr_estimator, tvpr_admm,
)

from scripts.common import (
    compute_metrics, print_comparison, parse_args, get_products, DEFAULT_CONFIG,
)


def expanding_window_tvpr_purged(
    Y, X_panel, lambda_tv, lambda_l1,
    min_history=52, rho=1.0, max_iter=200, tol=1e-5, step=1,
    warm_start_max_iter=50,
):
    """同 expanding_window_tvpr, 但每个 t 排除最后一个观测点 (X[t-1], Y[t-1])."""
    T, N, K = X_panel.shape
    beta_oos = np.zeros((T, K))
    beta_warm = None
    z1_warm = u1_warm = z2_warm = u2_warm = None
    prev_t = 0

    for t in range(min_history, T, step):
        Y_train = Y.iloc[: t - 1]
        X_train = X_panel[: t - 1]

        beta_init = None
        z1_init = u1_init = z2_init = u2_init = None
        if beta_warm is not None:
            tt = t - 1
            beta_init = np.zeros((tt, K))
            beta_init[:prev_t - 1] = beta_warm[:prev_t - 1]
            z1_init = np.zeros((tt - 1, K))
            z1_init[:prev_t - 2] = z1_warm[:prev_t - 2]
            u1_init = np.zeros((tt - 1, K))
            u1_init[:prev_t - 2] = u1_warm[:prev_t - 2]
            z2_init = np.zeros((tt, K))
            z2_init[:prev_t - 1] = z2_warm[:prev_t - 1]
            u2_init = np.zeros((tt, K))
            u2_init[:prev_t - 1] = u2_warm[:prev_t - 1]

        effective_max_iter = warm_start_max_iter if beta_warm is not None else max_iter

        beta_path, z1_new, z2_new, u1_new, u2_new = tvpr_admm(
            Y_train.values, X_train,
            lambda_tv, lambda_l1,
            rho=rho, max_iter=effective_max_iter, tol=tol,
            beta_init=beta_init,
            z1_init=z1_init, u1_init=u1_init,
            z2_init=z2_init, u2_init=u2_init,
            return_aux=True,
        )
        last_beta = beta_path[-1]

        beta_warm = beta_path
        z1_warm = z1_new
        u1_warm = u1_new
        z2_warm = z2_new
        u2_warm = u2_new
        prev_t = t

        for s in range(max(t - step, min_history), t + 1):
            if s < T:
                beta_oos[s] = last_beta

    return pd.DataFrame(beta_oos, index=Y.index,
                        columns=[f"factor_{i}" for i in range(K)])


def make_agg_tvpr(mode: str):
    """构造 _agg_tvpr. mode: 'current'=shift(-1)原版, 'purged'=排除最近观测."""
    def _agg(self, panels):
        cfg = self._config
        names = list(panels)
        cols = list(self._price_panel.columns)
        X = np.stack([
            panels[n].reindex(columns=cols).values for n in names
        ], axis=-1)
        Y = self._price_panel.pct_change().shift(-1)
        if mode == "current":
            beta = tvpr_estimator(
                Y, X, lambda_tv=cfg.tvpr_lambda_tv, lambda_l1=cfg.tvpr_lambda_l1,
                method="expanding", min_history=cfg.tvpr_min_history, step=cfg.agg_update,
            )
        else:
            beta = expanding_window_tvpr_purged(
                Y, X, lambda_tv=cfg.tvpr_lambda_tv, lambda_l1=cfg.tvpr_lambda_l1,
                min_history=cfg.tvpr_min_history, step=cfg.agg_update,
            )
        combo = np.einsum("tni,ti->tn", X, beta.values)
        return pd.DataFrame(combo, index=self._price_panel.index, columns=cols)
    return _agg


_orig_agg_tvpr = _ModuleStrategy._agg_tvpr


def main():
    args = parse_args("前视偏差决定性测试 v3")
    products = get_products(args.quick)
    n = len(products)

    def make_cfg():
        return ComboConfig(**DEFAULT_CONFIG)

    print("=" * 72)
    print(f"前视偏差决定性测试 v3 ({n}品种) — 排除最近观测点")
    print(f"区间: {args.start} ~ {args.end}")
    print("=" * 72)
    print("当前版: beta[τ] 用全部 Y[0:τ-1] (含今天收益 r[τ])")
    print("清洗版: beta[τ] 用 Y[0:τ-2] (不含今天收益 r[τ], 目标语义不变)")
    print()

    results = {}
    navs = {}

    # 当前版
    _ModuleStrategy._agg_tvpr = make_agg_tvpr("current")
    t0 = time.time()
    print("运行当前版...", flush=True)
    s1 = FactorModuleComboStrategy(products=products, config=make_cfg(), start=args.start, end=args.end)
    r1 = s1.run()
    results["current"] = compute_metrics(r1.nav)
    navs["current"] = r1.nav
    print(f"  Done ({time.time()-t0:.0f}s)  Sharpe={results['current']['Sharpe']:.4f}")

    # 清洗版
    _ModuleStrategy._agg_tvpr = make_agg_tvpr("purged")
    t0 = time.time()
    print("运行清洗版 (排除最近观测)...", flush=True)
    s2 = FactorModuleComboStrategy(products=products, config=make_cfg(), start=args.start, end=args.end)
    r2 = s2.run()
    results["purged"] = compute_metrics(r2.nav)
    navs["purged"] = r2.nav
    print(f"  Done ({time.time()-t0:.0f}s)  Sharpe={results['purged']['Sharpe']:.4f}")

    # 恢复原函数
    _ModuleStrategy._agg_tvpr = _orig_agg_tvpr

    # 对比
    print_comparison(results, title="Purge Test Comparison (Current vs Purged)")

    # NAV差异
    aligned = pd.concat([navs["current"], navs["purged"]], axis=1, keys=["cur", "pur"]).dropna()
    nav_diff = (aligned["cur"] - aligned["pur"]).abs()
    print(f"\nNAV差异: max={nav_diff.max():.6f}, mean={nav_diff.mean():.6f}, "
          f"不同天数={int((nav_diff > 1e-9).sum())}/{len(nav_diff)}")

    # 判定
    s_cur, s_pur = results["current"]["Sharpe"], results["purged"]["Sharpe"]
    print()
    if s_pur >= s_cur * 0.9:
        print(f"✅ 判定: 清洗版 Sharpe({s_pur:.4f}) ≥ 当前版({s_cur:.4f}) 的 90%")
        print("   → beta 用到今天收益 r[τ] 对业绩影响很小, shift(-1) 无实质前视")
    else:
        print(f"❌ 判定: 清洗版 Sharpe({s_pur:.4f}) < 当前版({s_cur:.4f}) 的 90%")
        print("   → beta 依赖今天收益 r[τ], 存在前视偏差!")

    # 保存
    pd.DataFrame({"current": navs["current"], "purged": navs["purged"]}).to_parquet(
        f"{args.output}/fut_v1_purge_test_v3.parquet")


if __name__ == "__main__":
    main()
