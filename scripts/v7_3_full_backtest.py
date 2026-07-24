# coding=utf-8
"""v7.3 完整版端到端回测 — 忠实于 source notebook.

[设计原则]
- 直接复用 9 因子 + 13 indices 数据 (用户决策: 不写代码, 不重新实现因子)
- 端到端 run_v7_3_backtest
- 与 v6.2 ir_expanding 对比 + 50/50 组合
- 输出到 reports/momentum_etf_rotation/v7/

[方法论] Faithful to source cell 102:
    sample = pd.concat([main_idx.resample('W').pct_change(), factor_pct], axis=1).dropna()
    quarter_window = 8
    bootstrap_lasso: times=500, resample=78-104 weeks
    Symmetry: 窗口全样本 (Klein 2013)
    FRP: sum_lower=0.9, sum_upper=1.0, max_weight=0.5

[执行命令]
    python3.11 scripts/v7_3_full_backtest.py --bootstrap 500        # 标准 (~30min)
    python3.11 scripts/v7_3_full_backtest.py --bootstrap 100        # 快速测试 (~5min)
    python3.11 scripts/v7_3_full_backtest.py --bootstrap 2000       # 极致 (~2h)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config,
    load_factor_returns,
    load_index_panel,
    run_v7_3_backtest,
)


def metrics(s: pd.Series, label: str = "") -> dict:
    """Calmar / Sharpe / MaxDD / Ann."""
    s = s.dropna()
    r = s.pct_change().dropna()
    if len(r) < 2:
        return {"name": label, "ann": 0.0, "vol": 0.0, "sharpe": 0.0,
                "dd": 0.0, "calmar": 0.0}
    n = len(r)
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0.0
    dd = (s / s.cummax() - 1).min()
    calmar = ann / abs(dd) if abs(dd) > 0.001 else 0.0
    return {"name": label, "ann": ann, "vol": vol, "sharpe": sharpe,
            "dd": dd, "calmar": calmar}


def print_metrics(m: dict) -> None:
    print(
        f"  {m['name']:48s} ann={m['ann']*100:7.2f}%  vol={m['vol']*100:7.2f}%"
        f"  sharpe={m['sharpe']:6.3f}  dd={m['dd']*100:7.2f}%  calmar={m['calmar']:6.3f}"
    )


def main(bootstrap_times: int = 500) -> None:
    print("=" * 70)
    print(f"v7.3 完整版端到端 (Symmetry + Bootstrap-Lasso × {bootstrap_times} + FRP)")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    factor_ret = load_factor_returns()
    idx_ret = load_index_panel(start="2008-01-01")
    print(f"  9 周频宏观因子:  {factor_ret.shape}")
    print(f"  13 指数日频收益: {idx_ret.shape}")

    print(f"\n[2/5] v7.3 v2 OOS 回测 (bootstrap_times={bootstrap_times})...")
    print("  季度调仓 + 8 quarter 窗口 + Symmetry 窗口白化 + Bootstrap-Lasso + FRP")
    cfg = V7_3Config(
        bootstrap_times=bootstrap_times,
        bootstrap_resample_min=78,
        bootstrap_resample_max=104,
        quarter_window=8,
    )
    import time
    t0 = time.time()
    nav_v73 = run_v7_3_backtest(idx_ret, factor_ret, cfg)
    print(f"  elapsed: {time.time()-t0:.1f}s")
    print(f"  v7.3 NAV:  {len(nav_v73)} rows, "
          f"{nav_v73.index[0].date()} -> {nav_v73.index[-1].date()}")

    print("\n[3/5] 加载对照组...")
    try:
        v62 = pd.read_parquet(
            REPO / "reports" / "momentum_etf_rotation" / "combo"
            / "v6_1_v6_2_combined_navs.parquet"
        )["v6.2 ir_expanding"]
        print(f"  v6.2 NAV range: {v62.index[0].date()} -> {v62.index[-1].date()}")
    except FileNotFoundError:
        print("  WARN: v6.2 ir_expanding NAVs not found, 跳过对照")
        v62 = None

    try:
        v10 = pd.read_parquet(
            REPO / "reports" / "momentum_etf_rotation" / "combo"
            / "unified_v1v5_navs_calA.parquet"
        )["v1.0 locked"]
    except FileNotFoundError:
        v10 = None

    # 对齐
    aligned_idx = nav_v73.index
    nav_v73_a = nav_v73
    v62_a = v62.reindex(aligned_idx) if v62 is not None else None
    v10_a = v10.reindex(aligned_idx) if v10 is not None else None

    # 5 ETF 等权 (idx_rets 等权)
    nav_eq = (
        1
        + idx_ret[list(cfg.index_pool)]
        .reindex(aligned_idx)
        .fillna(0.0)
        .mean(axis=1)
    ).cumprod()

    # --- 4. 评估 ---
    print("\n[4/5] 评估...")
    results = []
    print(f"\n--- 全期 ({aligned_idx.min().date()} -> {aligned_idx.max().date()}) ---")
    m = metrics(nav_v73_a, "v7.3 完整版 (Bootstrap-Lasso + FRP)")
    results.append(m)
    print_metrics(m)
    m = metrics(nav_eq, "13 指数等权")
    results.append(m)
    print_metrics(m)
    if v62_a is not None:
        m = metrics(v62_a.dropna(), "v6.2 ir_expanding (current best)")
        results.append(m)
        print_metrics(m)
        # combo 50/50
        combo = (v62_a.fillna(1.0) + nav_v73_a.fillna(1.0)) / 2.0
        combo = combo / combo.iloc[0]
        m = metrics(combo, "combo 50/50 (v6.2 + v7.3)")
        results.append(m)
        print_metrics(m)
    if v10_a is not None:
        m = metrics(v10_a.dropna(), "v1.0 locked (低风险基准)")
        results.append(m)
        print_metrics(m)

    # OOS 2022-2026 子集
    if "2022-01-01" < aligned_idx.max().strftime("%Y-%m-%d"):
        print(f"\n--- OOS 2022-01-01 -> {aligned_idx.max().date()} ---")
        sub = nav_v73_a.loc["2022-01-01":]
        m = metrics(sub, "v7.3 完整版 (OOS 2022+)")
        results.append(m)
        print_metrics(m)
        sub_eq = nav_eq.reindex(aligned_idx).loc["2022-01-01":]
        m = metrics(sub_eq, "13 指数等权 (OOS 2022+)")
        results.append(m)
        print_metrics(m)
        if v62_a is not None:
            sub62 = v62_a.loc["2022-01-01":].dropna()
            m = metrics(sub62, "v6.2 ir_expanding (OOS 2022+)")
            results.append(m)
            print_metrics(m)
            # combo: 同全期公式 (简单平均 NAV 起点归 1)
            v62_oos = v62_a.loc["2022-01-01":]
            v73_oos = nav_v73_a.loc["2022-01-01":]
            common_oos = v62_oos.dropna().index.intersection(v73_oos.dropna().index)
            combo22 = (v62_oos.reindex(common_oos) + v73_oos.reindex(common_oos)) / 2.0
            combo22 = combo22 / combo22.iloc[0] if combo22.iloc[0] != 0 else combo22
            m = metrics(combo22, "combo 50/50 (OOS 2022+)")
            results.append(m)
            print_metrics(m)

    # 相关性
    if v62_a is not None:
        print(f"\n--- 相关性 (OOS 2022+) ---")
        v73_ret = nav_v73_a.loc["2022-01-01":].pct_change().dropna()
        v62_ret = v62_a.loc["2022-01-01":].pct_change().dropna()
        common = v73_ret.index.intersection(v62_ret.index)
        corr = v73_ret.loc[common].corr(v62_ret.loc[common])
        print(f"  v7.3 vs v6.2 ir_expanding: corr = {corr:.3f}")

    # --- 5. 保存 ---
    print("\n[5/5] 保存结果...")
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v7"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    out_csv = out_dir / "v7_3_full_oos_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"  {out_csv}")

    navs = pd.DataFrame({"v7.3 完整版 (Bootstrap-Lasso + FRP)": nav_v73_a})
    if v62_a is not None:
        navs["v6.2 ir_expanding"] = v62_a
    if v10_a is not None:
        navs["v1.0 locked"] = v10_a
    navs["13 指数等权"] = nav_eq
    if v62_a is not None:
        navs["combo 50/50 (v6.2 + v7.3)"] = (v62_a.fillna(1.0) + nav_v73_a.fillna(1.0)) / 2.0
        navs["combo 50/50"] = navs["combo 50/50 (v6.2 + v7.3)"] / navs["combo 50/50 (v6.2 + v7.3)"].iloc[0]
    out_pq = out_dir / "v7_3_full_oos_navs.parquet"
    navs.to_parquet(out_pq)
    print(f"  {out_pq}")

    print(f"\n{'=' * 70}")
    print("完成.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap", type=int, default=500,
        help="Bootstrap-Lasso 次数 (默认 500, source 一致)",
    )
    args = parser.parse_args()
    main(bootstrap_times=args.bootstrap)
