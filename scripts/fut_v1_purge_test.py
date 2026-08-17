#!/usr/bin/env python3
"""前视偏差决定性测试 v3: 排除最近观测点的干净清洗

⚠️ v1 教训 (2026-08-17):
   组合策略的 _agg_tvpr 由各 module 实例调用 (FactorModuleComboStrategy.
   _signals_raw 遍历 self._modules 调 m._signals_raw), 组合层子类 override
   不会被调用 → v1 测试两组跑的都是原代码, 结果0差异是假象.
   必须 patch _ModuleStrategy._agg_tvpr (所有 module 的基类).

⚠️ v2 教训:
   shift(0) 改变了预测目标 (Y[s]=r[s] 当日收益 vs Y[s]=r[s+1] 下一日收益),
   不是干净清洗. NAV 从5.07崩到2.74 是因为目标语义变了, 不能证明前视.

v3 正确清洗:
- 当前版: Y=shift(-1), beta[τ] 拟合用全部 Y[0:τ-1] (含 (X[τ-1], r[τ]))
- 清洗版: Y=shift(-1) 不变, 但 beta[τ] 拟合用 Y[0:τ-2] (排除 (X[τ-1], r[τ]))
  → beta[τ] 只用 ≤ τ-2 的数据 (截至昨天收盘 p[τ-1])
  → 目标语义不变(仍预测下一日收益), 只是不用今天收益 r[τ]

判定 (实测):
- 10品种: 清洗版2.109 vs 当前2.091 (清洗版略高, <1%)
- 72品种: 清洗版1.470 vs 当前1.618 (当前高9%)
- 解读: 当前版用今天收益 r[τ] 是合法信息 (EOD信号: 收盘后算信号, T+1执行),
  清洗版假设收盘前算信号, 过于保守. 真正的非法前视是 shift(-2) 用明天价格.
  → shift(-1) 无前视, 保持现状.

用法: python scripts/fut_v1_purge_test.py [--quick]
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time, argparse
import numpy as np, pandas as pd

import QuantNodes.strategy.cta.strategies.factor_modules as fm
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS, _ModuleStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    tvpr_estimator, tvpr_admm,
)


def expanding_window_tvpr_purged(
    Y, X_panel, lambda_tv, lambda_l1,
    min_history=52, rho=1.0, max_iter=200, tol=1e-5, step=1,
    warm_start_max_iter=50,
):
    """同 expanding_window_tvpr, 但每个 t 排除最后一个观测点 (X[t-1], Y[t-1]).

    Y 仍为 shift(-1) 的下一日收益 → 目标语义不变.
    beta[t] 只用 [0, t-2] → 完全不用今天收益 r[t].
    """
    T, N, K = X_panel.shape
    beta_oos = np.zeros((T, K))
    beta_warm = None
    z1_warm = u1_warm = z2_warm = u2_warm = None
    prev_t = 0

    for t in range(min_history, T, step):
        # 清洗: 排除最后一个观测点
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
    print(f"前视偏差决定性测试 v3 ({n}品种) — 排除最近观测点")
    print(f"区间: {args.start} ~ {args.end}")
    print("=" * 72)
    print("当前版: beta[τ] 用全部 Y[0:τ-1] (含今天收益 r[τ])")
    print("清洗版: beta[τ] 用 Y[0:τ-2] (不含今天收益 r[τ], 目标语义不变)")
    print()

    results = {}
    navs = {}

    _ModuleStrategy._agg_tvpr = make_agg_tvpr("current")
    t0 = time.time()
    print("运行当前版...", flush=True)
    s1 = FactorModuleComboStrategy(products=products, config=make_cfg(), start=args.start, end=args.end)
    r1 = s1.run()
    results["current"] = compute_metrics(r1.nav)
    navs["current"] = r1.nav
    print(f"  Done ({time.time()-t0:.0f}s)  Sharpe={results['current']['Sharpe']:.4f}")

    _ModuleStrategy._agg_tvpr = make_agg_tvpr("purged")
    t0 = time.time()
    print("运行清洗版 (排除最近观测)...", flush=True)
    s2 = FactorModuleComboStrategy(products=products, config=make_cfg(), start=args.start, end=args.end)
    r2 = s2.run()
    results["purged"] = compute_metrics(r2.nav)
    navs["purged"] = r2.nav
    print(f"  Done ({time.time()-t0:.0f}s)  Sharpe={results['purged']['Sharpe']:.4f}")

    _ModuleStrategy._agg_tvpr = _orig_agg_tvpr

    print()
    print("-" * 68)
    print("对比:")
    print(f"  {'':>12} {'当前版':>16} {'清洗版(排除最近)':>18} {'变化':>12}")
    print("  " + "-" * 60)
    for key in ["AnnRet", "AnnVol", "Sharpe", "MaxDD", "Calmar"]:
        v1 = results["current"].get(key, 0)
        v2 = results["purged"].get(key, 0)
        if key in ("AnnRet", "AnnVol", "MaxDD"):
            print(f"  {key:>12}: {v1:>13.3%} {v2:>15.3%} {v2-v1:+10.3%}")
        else:
            print(f"  {key:>12}: {v1:>13.4f} {v2:>15.4f} {v2-v1:+10.4f}")

    nav1, nav2 = navs["current"], navs["purged"]
    aligned = pd.concat([nav1, nav2], axis=1, keys=["cur", "pur"]).dropna()
    nav_diff = (aligned["cur"] - aligned["pur"]).abs()
    print(f"\nNAV差异: max={nav_diff.max():.6f}, mean={nav_diff.mean():.6f}, "
          f"不同天数={int((nav_diff > 1e-9).sum())}/{len(nav_diff)}")

    s_cur, s_pur = results["current"]["Sharpe"], results["purged"]["Sharpe"]
    print()
    if s_pur >= s_cur * 0.9:
        print(f"✅ 判定: 清洗版 Sharpe({s_pur:.4f}) ≥ 当前版({s_cur:.4f}) 的 90%")
        print("   → beta 用到今天收益 r[τ] 对业绩影响很小, shift(-1) 无实质前视")
    else:
        print(f"❌ 判定: 清洗版 Sharpe({s_pur:.4f}) < 当前版({s_cur:.4f}) 的 90%")
        print("   → beta 依赖今天收益 r[τ], 存在前视偏差!")

    pd.DataFrame({"current": navs["current"], "purged": navs["purged"]}).to_parquet(
        f"output/fut_v1_purge_test_v3_{'quick' if args.quick else 'full'}.parquet")


if __name__ == "__main__":
    main()
