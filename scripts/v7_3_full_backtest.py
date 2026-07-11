# coding=utf-8
"""v7.3 完整版端到端: Symmetry + Bootstrap-Lasso + FactorRiskParity.

[设计原则]
- 直接复用 Excel 数据 (用户决策: 不写代码)
- 端到端 run_v7_3_backtest
- 与 v6.2 ir_expanding 对比 + 50/50 组合
- 输出到 reports/momentum_etf_rotation/v7/

[计算预估]
- 287 weekly rebal × bootstrap_times=200 × 5 assets × LassoCV(5)
- ≈ 144 万次 LassoCV fit
- 在 4 core 上, 估算 8-15 分钟

[用户决策]
- bootstrap_times = 200 (用户决策 "全面 2000 次" 但 CI 需可行, 默认 200)
  生产运行用 2000
- 不并行修 6 BME Panel bug
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
    load_etf_panel,
    load_factor_returns,
    run_v7_3_backtest,
)


def metrics(s: pd.Series, label: str = "") -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    if len(r) < 2:
        return {"name": label, "ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    n = len(r)
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = ann / vol if vol > 0 else 0.0
    dd = (s / s.cummax() - 1).min()
    calmar = ann / abs(dd) if abs(dd) > 0.001 else 0.0
    return {"name": label, "ann": ann, "vol": vol, "sharpe": sharpe, "dd": dd, "calmar": calmar}


def main(bootstrap_times: int = 200) -> None:
    print("=" * 70)
    print(f"v7.3 完整版端到端 (Symmetry + Bootstrap-Lasso × {bootstrap_times} + FRP)")
    print("=" * 70)

    # --- 1. 加载 ---
    print("\n[1/5] 加载数据...")
    factor_ret = load_factor_returns()
    etf_nav = load_etf_panel(start="2018-01-01")
    print(f"  9 周频宏观因子:  {factor_ret.shape}")
    print(f"  5 ETF 日频净值: {etf_nav.shape}")

    # --- 2. v7.3 OOS 回测 ---
    print(f"\n[2/5] v7.3 OOS 回测 (bootstrap_times={bootstrap_times})...")
    print("  预计 8-15 分钟, 请耐心等待...")
    cfg = V7_3Config(
        bootstrap_times=bootstrap_times,
        bootstrap_min_weeks=104,
        bootstrap_max_weeks=156,
        min_history_weeks=52 * 3,
    )
    nav_v73 = run_v7_3_backtest(factor_ret, etf_nav, cfg)
    print(f"  v7.3 NAV:  {len(nav_v73)} rows, "
          f"{nav_v73.index.min().date()} -> {nav_v73.index.max().date()}")

    # --- 3. 对照组 ---
    print("\n[3/5] 加载对照组...")
    v62 = pd.read_parquet(
        REPO / "reports" / "momentum_etf_rotation" / "combo"
        / "v6_1_v6_2_combined_navs.parquet"
    )["v6.2 ir_expanding"]

    # v6.1 IC12 (老 baseline)
    v61 = pd.read_parquet(
        REPO / "reports" / "momentum_etf_rotation" / "combo"
        / "v6_1_v6_2_combined_navs.parquet"
    )["v6.1 IC12"]

    # v1.0 locked
    v10 = pd.read_parquet(
        REPO / "reports" / "momentum_etf_rotation" / "combo"
        / "unified_v1v5_navs_calA.parquet"
    )["v1.0 locked"]

    # 对齐
    aligned_idx = nav_v73.index
    nav_v73_a = nav_v73.reindex(aligned_idx)
    v62_a = v62.reindex(aligned_idx)
    v61_a = v61.reindex(aligned_idx)
    v10_a = v10.reindex(aligned_idx)

    # 5 ETF 等权
    nav_eq = (1 + etf_nav.pct_change().fillna(0).loc[aligned_idx]
              [list(cfg.etf_pool)].mean(axis=1)).cumprod()

    # --- 4. 评估 ---
    print("\n[4/5] 评估...")
    results = []
    print(f"\n--- 全期 ({aligned_idx.min().date()} -> {aligned_idx.max().date()}) ---")
    for name, s in [
        ("v7.3 完整版 (Bootstrap-Lasso + FRP)", nav_v73_a),
        ("v6.2 ir_expanding (current best)", v62_a),
        ("v6.1 IC12", v61_a),
        ("v1.0 locked", v10_a),
        ("5 ETF 等权", nav_eq.reindex(aligned_idx)),
    ]:
        m = metrics(s.dropna(), name)
        results.append(m)
        print(f"  {m['name']:48s} ann={m['ann']*100:7.2f}%  vol={m['vol']*100:7.2f}%"
              f"  sharpe={m['sharpe']:6.3f}  dd={m['dd']*100:7.2f}%  calmar={m['calmar']:6.3f}")

    # 50/50 组合
    combo_50_50 = (v62_a.fillna(1.0) + nav_v73_a.fillna(1.0)) / 2
    combo_50_50 = combo_50_50 / combo_50_50.iloc[0]  # 起点归 1
    m = metrics(combo_50_50, "combo 50/50 (v6.2 + v7.3)")
    results.append(m)
    print(f"  {m['name']:48s} ann={m['ann']*100:7.2f}%  vol={m['vol']*100:7.2f}%"
          f"  sharpe={m['sharpe']:6.3f}  dd={m['dd']*100:7.2f}%  calmar={m['calmar']:6.3f}")

    # OOS 2022-2026
    if "2022-01-01" < aligned_idx.max().strftime("%Y-%m-%d"):
        print(f"\n--- OOS 2022-01-01 -> {aligned_idx.max().date()} ---")
        for name, s in [
            ("v7.3 完整版 (OOS 2022+)", nav_v73_a.loc["2022-01-01":]),
            ("v6.2 ir_expanding (OOS 2022+)", v62_a.loc["2022-01-01":]),
            ("combo 50/50 (OOS 2022+)", combo_50_50.loc["2022-01-01":]),
        ]:
            m = metrics(s.dropna(), name)
            results.append(m)
            print(f"  {m['name']:48s} ann={m['ann']*100:7.2f}%  vol={m['vol']*100:7.2f}%"
                  f"  sharpe={m['sharpe']:6.3f}  dd={m['dd']*100:7.2f}%  calmar={m['calmar']:6.3f}")

    # 相关性
    print(f"\n--- 相关性 (OOS 2022+) ---")
    v73_ret = nav_v73_a.loc["2022-01-01":].pct_change().dropna()
    v62_ret = v62_a.loc["2022-01-01":].pct_change().dropna()
    common = v73_ret.index.intersection(v62_ret.index)
    corr = v73_ret.loc[common].corr(v62_ret.loc[common])
    print(f"  v7.3 vs v6.2 ir_expanding: corr = {corr:.3f}")
    print(f"  (用户判据 < 0.5)")

    # --- 5. 保存 ---
    print("\n[5/5] 保存结果...")
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v7"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    df_results = pd.DataFrame(results)
    out_csv = out_dir / "v7_3_full_oos_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"  {out_csv}")

    # NAV parquet
    navs = pd.DataFrame({
        "v7.3 完整版 (Bootstrap-Lasso + FRP)": nav_v73,
        "v6.2 ir_expanding": v62.reindex(nav_v73.index),
        "v6.1 IC12": v61.reindex(nav_v73.index),
        "v1.0 locked": v10.reindex(nav_v73.index),
        "5 ETF 等权": nav_eq.reindex(nav_v73.index),
        "combo 50/50 (v6.2 + v7.3)": combo_50_50.reindex(nav_v73.index),
    })
    out_pq = out_dir / "v7_3_full_oos_navs.parquet"
    navs.to_parquet(out_pq)
    print(f"  {out_pq}")

    print(f"\n{'=' * 70}")
    print("完成.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap", type=int, default=200,
        help="Bootstrap-Lasso 次数 (默认 200, 用户决策推荐 2000, CI 默认 200)",
    )
    args = parser.parse_args()
    main(bootstrap_times=args.bootstrap)
