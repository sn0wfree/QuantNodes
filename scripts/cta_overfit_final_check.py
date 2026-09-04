# coding=utf-8
"""cta_overfit_final_check.py — 定版配置 (FINAL_LUMP_CFG) 过拟合三件套重跑.

背景: docs/cta/95 §七 的三件套 (严格OOS/Bootstrap/起点依赖) 是 2026-08-12
旧配置时代 (tv=0.30) 的结果, 不能沿用到现行定版 (tv=1.5, lump, 按品种校准成本).
本脚本用定版配置重跑, 方法论与旧版对齐 (网格/块大小/起点完全一致, 保证可比):
    1. Bootstrap CV%:   52天块 × 200次, Sharpe 分布 CV (SOP阈值 50%)
    2. 起点依赖:        2018/2019/2020/2021/2022 五起点全量回测
    3. 严格OOS:         IS(2018-2021) 网格选参 λ_tv{0.15,0.5}×momentum{40,60},
                        OOS(2022-2025) 验证一次, 核对 IS最优 是否=定版参数

 Sharpe 口径: 与 95 §7.5 对齐 = 252计数CAGR / 年化Vol;
 同时附 compute_metrics 官方口径 (日历年CAGR) 便于文档引用.

注意: 独立新脚本 — 不修改 scripts/cta_overfitting_check.py (他人工作).
输出: reports/cta/overfit_final_check.csv  (三件套逐行)
      reports/cta/overfit_final_bootstrap.csv
      reports/cta/overfit_final_summary.json

用法: python3 scripts/cta_overfit_final_check.py [--workers 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.combo import (  # noqa: E402
    ComboConfig, FactorModuleComboStrategy,
)
from QuantNodes.strategy.cta.strategies.base import DEFAULT_PRODUCTS  # noqa: E402

REPORT_DIR = Path("reports/cta")

# 定版配置 (与 cta_final_baseline.py FINAL_LUMP_CFG 逐字段一致 + 按品种校准成本)
FINAL_LUMP_CFG = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=1.5, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True, position_ema_alpha=0.15,
    roll_cost_bps=10.0, roll_model="lump",
)
CALIBRATED_COSTS = {  # cost_model.json 解析 (定版口径)
    "commission_rate": {p: v["commission_bps"] / 10000
                        for p, v in json.loads(
                            (REPORT_DIR / "cost_model.json").read_text()).items()},
    "slippage_bps": {p: v["slippage_bps"]
                     for p, v in json.loads(
                         (REPORT_DIR / "cost_model.json").read_text()).items()},
}

IS_END = "2022-01-01"          # 与 95 §7.2 / cta_rebuild_portfolio.py 一致
BOOT_BLOCK = 52                # 与 95 §7.5 一致
BOOT_N = 200
BOOT_SEED = 42
STARTS = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]
GRID = [(0.15, 40), (0.15, 60), (0.5, 40), (0.5, 60)]  # 与 95 §7.2 网格一致
BASE_LAMBDA_TV = 0.5           # ModuleConfig 默认, 定版隐含值


def _metrics_252(nav: pd.Series) -> dict:
    """95 §7.5 口径: 252计数CAGR + 年化Vol Sharpe.

    切段 (IS/OOS) 时 nav 必须重定基, 否则 CAGR 会把切点前累计收益算进来.
    """
    nav = nav / nav.iloc[0]
    rets = nav.pct_change().dropna()
    if len(rets) < 50:
        return {"sharpe": np.nan, "ann_ret": np.nan, "vol": np.nan,
                "max_dd": np.nan, "calmar": np.nan}
    years = len(rets) / 252
    ann = nav.iloc[-1] ** (1 / years) - 1
    vol = float(rets.std() * np.sqrt(252))
    dd = float((nav / nav.cummax() - 1).min())
    return {"sharpe": ann / vol if vol > 0 else np.nan, "ann_ret": ann, "vol": vol,
            "max_dd": dd, "calmar": ann / abs(dd) if dd < 0 else np.nan}


def _metrics_official(nav: pd.Series) -> dict:
    """compute_metrics 官方口径 (Sharpe=日历年CAGR/Vol)."""
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import (
        compute_metrics,
    )
    m = compute_metrics(nav)
    return {k: float(m[k]) for k in ("Sharpe", "AnnRet", "Vol", "MaxDD", "Calmar")
            if k in m}


def _run_one(task: tuple) -> tuple:
    """worker: (label, start, lambda_tv, momentum) -> (task, nav序列)."""
    label, start, lam_tv, mom_win = task
    cfg_kw = dict(FINAL_LUMP_CFG)
    cfg_kw["tvpr_lambda_tv"] = lam_tv
    cfg_kw["momentum_filter_window"] = mom_win
    cfg_kw.update(CALIBRATED_COSTS)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    s = FactorModuleComboStrategy(
        config=cfg, products=list(DEFAULT_PRODUCTS),
        start=start, end="2025-12-31",
    )
    r = s.run()
    return (label, start, lam_tv, mom_win), r.nav


def run_bootstrap(nav: pd.Series) -> dict:
    """52天块 × 200次 块自助法 → Sharpe 分布 CV (与 95 §7.5 同法)."""
    rng = np.random.default_rng(BOOT_SEED)
    r = nav.pct_change().dropna().values
    n = len(r)
    n_blocks = int(np.ceil(n / BOOT_BLOCK))
    out = np.empty(BOOT_N)
    for i in range(BOOT_N):
        idx = rng.integers(0, n - BOOT_BLOCK + 1, size=n_blocks)
        sample = np.concatenate([r[j:j + BOOT_BLOCK] for j in idx])[:n]
        eq = np.cumprod(1.0 + sample)
        ann = eq[-1] ** (252 / n) - 1
        vol = sample.std(ddof=1) * np.sqrt(252)
        out[i] = ann / vol if vol > 0 else 0.0
    mean, std = float(out.mean()), float(out.std(ddof=1))
    return {
        "mean": mean, "std": std,
        "cv_pct": std / mean * 100 if mean > 0 else float("nan"),
        "p5": float(np.percentile(out, 5)), "p95": float(np.percentile(out, 95)),
        "min": float(out.min()), "max": float(out.max()),
        "n_boot": BOOT_N, "block_days": BOOT_BLOCK,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    # 任务: 起点2018定版(基线, 复用于严格OOS的(0.5,60)与起点依赖2018) + 网格其余3 + 起点其余4
    tasks = [("基线/严格OOS(0.5,60)", "2018-01-01", BASE_LAMBDA_TV, 60)]
    for lam, mom in GRID:
        if (lam, mom) != (BASE_LAMBDA_TV, 60):
            tasks.append((f"严格OOS({lam},{mom})", "2018-01-01", lam, mom))
    for st in STARTS[1:]:
        tasks.append((f"起点{st[:4]}", st, BASE_LAMBDA_TV, 60))
    print(f"共 {len(tasks)} 次全量回测 (workers={args.workers})")

    results: dict[tuple, pd.Series] = {}
    nav_dir = REPORT_DIR / "overfit_final_navs"
    nav_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            t, nav = fut.result()
            results[t] = nav
            safe = t[0].replace("/", "_").replace("(", "").replace(")", "")
            nav.to_frame("nav").to_parquet(nav_dir / f"{safe}_{t[1][:4]}.parquet")
            m = _metrics_252(nav)
            print(f"  done {t}: Sharpe {m['sharpe']:.3f} AnnRet {m['ann_ret']:+.1%} "
                  f"MaxDD {m['max_dd']:+.1%}", flush=True)

    # ── 1. 基线 ──────────────────────────────────────────────────────
    base_nav = results[("基线/严格OOS(0.5,60)", "2018-01-01", BASE_LAMBDA_TV, 60)]
    rows = []
    m25, mof = _metrics_252(base_nav), _metrics_official(base_nav)
    rows.append({"test": "基线(起点2018,定版)", "param": "λ_tv=0.5,mom=60",
                 "start": "2018-01-01", "sharpe": m25["sharpe"],
                 "ann_ret": m25["ann_ret"], "max_dd": m25["max_dd"],
                 "calmar": m25["calmar"],
                 "official_sharpe": mof.get("Sharpe"),
                 "official_ann_ret": mof.get("AnnRet")})

    # ── 2. Bootstrap CV ─────────────────────────────────────────────
    bs = run_bootstrap(base_nav)
    pd.DataFrame([bs]).to_csv(REPORT_DIR / "overfit_final_bootstrap.csv", index=False)
    print(f"Bootstrap CV = {bs['cv_pct']:.1f}%  (p5 {bs['p5']:.3f} / p95 {bs['p95']:.3f})")

    # ── 3. 起点依赖 ─────────────────────────────────────────────────
    for st in STARTS:
        nav = results[("基线/严格OOS(0.5,60)", "2018-01-01", BASE_LAMBDA_TV, 60)] \
            if st == "2018-01-01" else results[(f"起点{st[:4]}", st, BASE_LAMBDA_TV, 60)]
        m = _metrics_252(nav)
        rows.append({"test": "起点依赖", "param": "λ_tv=0.5,mom=60", "start": st,
                     "sharpe": m["sharpe"], "ann_ret": m["ann_ret"],
                     "max_dd": m["max_dd"], "calmar": m["calmar"],
                     "official_sharpe": None, "official_ann_ret": None})

    # ── 4. 严格OOS: IS段选参 → OOS段验证一次 ─────────────────────────
    grid_rows = []
    for lam, mom in GRID:
        nav = results[("基线/严格OOS(0.5,60)", "2018-01-01", BASE_LAMBDA_TV, 60)] \
            if (lam, mom) == (BASE_LAMBDA_TV, 60) \
            else results[(f"严格OOS({lam},{mom})", "2018-01-01", lam, mom)]
        is_m = _metrics_252(nav[nav.index < IS_END])
        oos_m = _metrics_252(nav[nav.index >= IS_END])
        grid_rows.append({"lambda_tv": lam, "momentum": mom,
                          "is_sharpe": is_m["sharpe"], "is_ann_ret": is_m["ann_ret"],
                          "oos_sharpe": oos_m["sharpe"], "oos_ann_ret": oos_m["ann_ret"],
                          "oos_max_dd": oos_m["max_dd"], "oos_calmar": oos_m["calmar"]})
        rows.append({"test": f"严格OOS λ_tv={lam},mom={mom}", "param": f"({lam},{mom})",
                     "start": "2018-01-01(IS选参)/OOS验证",
                     "sharpe": is_m["sharpe"], "ann_ret": is_m["ann_ret"],
                     "max_dd": is_m["max_dd"], "calmar": is_m["calmar"],
                     "official_sharpe": oos_m["sharpe"],
                     "official_ann_ret": oos_m["ann_ret"]})
    grid_df = pd.DataFrame(grid_rows).sort_values("is_sharpe", ascending=False)
    best = grid_df.iloc[0]
    print("\n严格OOS (IS=2018-2021选参, OOS=2022-2025验证):")
    print(grid_df.to_string(index=False))
    flag = "是" if (best["lambda_tv"], best["momentum"]) == (0.5, 60) else "否"
    print(f"IS最优: λ_tv={best['lambda_tv']}, mom={best['momentum']} (定版=(0.5,60)? {flag})")
    print(f"IS最优 OOS Sharpe = {best['oos_sharpe']:.3f}")

    pd.DataFrame(rows).to_csv(REPORT_DIR / "overfit_final_check.csv", index=False)

    # ── 5. 汇总判定 ─────────────────────────────────────────────────
    starts_sharpe = [r["sharpe"] for r in rows if r["test"] == "起点依赖"]
    checks = {
        "bootstrap_cv_lt_50": bs["cv_pct"] < 50.0,
        "start_all_positive": all(s > 0 for s in starts_sharpe),
        "strict_oos_sharpe_ge_1": float(best["oos_sharpe"]) >= 1.0,
        "is_best_is_default": (best["lambda_tv"], best["momentum"]) == (0.5, 60),
    }
    verdict = "PASS" if all(checks.values()) else (
        "WARN" if sum(checks.values()) >= 3 else "FAIL")
    summary = {
        "config": "FINAL_LUMP_CFG (tv=1.5, lump, 按品种校准成本)",
        "baseline": {k: round(v, 4) for k, v in m25.items()},
        "bootstrap": {k: round(v, 4) for k, v in bs.items()},
        "start_dependence": {st[:4]: round(s, 3) for st, s in zip(STARTS, starts_sharpe)},
        "strict_oos": grid_df.to_dict(orient="records"),
        "is_best": {"lambda_tv": float(best["lambda_tv"]),
                    "momentum": int(best["momentum"]),
                    "oos_sharpe": float(best["oos_sharpe"])},
        "checks": checks, "verdict": verdict,
    }
    (REPORT_DIR / "overfit_final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n判定: {verdict}  {checks}")
    print("已保存: overfit_final_check.csv / overfit_final_bootstrap.csv / "
          "overfit_final_summary.json")


if __name__ == "__main__":
    main()
