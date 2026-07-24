# coding=utf-8
"""因子择时 (FactorTiming) 诊断研究 — 什么让它赢/输.

分析维度:
1. 每个因子的 IC 分布: 均值/标准差/命中率/分位
2. 不同 forward_window 的 IC 衰减 (1w, 2w, 4w, 13w, 26w, 52w)
3. IC 的自相关 (持续性 / 半衰期)
4. IC 跨 regime: 牛市/熊市/震荡市 的因子表现
5. IC vs 实际收益: 高 IC 是否真的预测高收益
6. 哪些因子最稳定 (持续正 IC), 哪些 regime-dependent
7. IC 排名时序: 每月 #1 因子是哪个
8. 因子择时 vs 等权 Smart β: 真实 alpha 多少
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v4.factor_ic import (
    FACTOR_NAMES,
    factor_ic_at,
    compute_factor_scores,
    compute_forward_return,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import (
    ALL_V4_CODES,
    SMART_BETA_CODES,
    STYLE_GROUP_CODES,
    load_smartbeta_panel,
)

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"


def ann_return(nav: pd.Series) -> float:
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav: pd.Series) -> float:
    pk = nav.cummax()
    return float((nav / pk - 1.0).min())


def sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))


def metrics(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    ar = ann_return(nav)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "sharpe": sharpe(rets),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
    }


def classify_regime(panel: pd.DataFrame) -> pd.Series:
    px = panel["510300"]
    mom60 = px.pct_change(60)
    mom252 = px.pct_change(252)
    out = pd.Series("sideways", index=panel.index)
    out[(mom60 > 0.05) & (mom252 > 0.10)] = "bull"
    out[(mom60 < -0.05) & (mom252 < -0.10)] = "bear"
    return out


def main():
    panel = load_smartbeta_panel()
    all_codes = list(ALL_V4_CODES)
    print(f"[data] {panel.shape[0]} days × {panel.shape[1]} codes")

    rebal_dates = panel.index.to_series().resample("W-FRI").last().index
    rebal_dates = [d for d in rebal_dates if d in panel.index and panel.index.get_loc(d) >= 252]
    print(f"[sample] {len(rebal_dates)} 调仓日 (weekly)")

    print("\n========= 1. 多 forward_window 的 IC 衰减 =========")
    fwd_windows = [5, 10, 20, 40, 60, 120, 180, 252]
    ic_decay = {fw: {n: [] for n in FACTOR_NAMES} for fw in fwd_windows}
    for date in rebal_dates:
        for fw in fwd_windows:
            ic = factor_ic_at(panel, date, all_codes,
                              forward_window=fw, lookback=60)
            for n in FACTOR_NAMES:
                ic_decay[fw][n].append(ic[n])

    print(f"{'Factor':<10s} | " + " | ".join(f"FW{fw:3d}" for fw in fwd_windows))
    for n in FACTOR_NAMES:
        means = [np.mean(ic_decay[fw][n]) for fw in fwd_windows]
        print(f"{n:<10s} | " + " | ".join(f"{m:+5.3f}" for m in means))

    print("\n========= 2. IC 分布 (forward_window=20, 默认) =========")
    fw_default = 20
    ic_data = {n: [] for n in FACTOR_NAMES}
    for date in rebal_dates:
        ic = factor_ic_at(panel, date, all_codes,
                          forward_window=fw_default, lookback=60)
        for n in FACTOR_NAMES:
            ic_data[n].append(ic[n])
    for n in FACTOR_NAMES:
        arr = np.array(ic_data[n])
        pos_rate = (arr > 0).mean()
        print(f"  {n:<10s}  mean={arr.mean():+5.3f}  std={arr.std():5.3f}  "
              f"hit_rate={pos_rate*100:5.1f}%  abs_mean={np.abs(arr).mean():5.3f}  "
              f"|IC|>0.05 频率={100*(np.abs(arr) > 0.05).mean():5.1f}%")

    print("\n========= 3. IC 自相关 (持续性 / 半衰期) =========")
    for n in FACTOR_NAMES:
        arr = pd.Series(ic_data[n])
        if arr.std() == 0:
            print(f"  {n:<10s}  IC 全为 0 (no signal)")
            continue
        acs = []
        for lag in [1, 4, 13, 26, 52]:
            ac = arr.autocorr(lag=lag)
            acs.append(f"lag{lag}={ac:+.2f}")
        print(f"  {n:<10s}  " + "  ".join(acs))

    print("\n========= 4. IC × Regime 分解 =========")
    regime = classify_regime(panel)
    regime_aligned = regime.reindex(rebal_dates, method="ffill")
    for r in ["bull", "bear", "sideways"]:
        mask = regime_aligned == r
        if mask.sum() < 5:
            continue
        print(f"\n  --- {r} (n={mask.sum()}) ---")
        for n in FACTOR_NAMES:
            arr = np.array(ic_data[n])[mask.values]
            print(f"    {n:<10s}  mean={arr.mean():+5.3f}  std={arr.std():5.3f}  "
                  f"hit_rate={(arr > 0).mean()*100:5.1f}%")

    print("\n========= 5. IC → 实际收益 的关系 (forward=20) =========")
    actual_records = {}
    for date in rebal_dates:
        idx = panel.index.get_loc(date)
        if idx >= len(panel) - 21:
            continue
        fwd = panel.iloc[idx + 1: idx + 21]
        if len(fwd) < 20:
            continue
        rets = fwd.iloc[-1] / fwd.iloc[0] - 1.0
        actual_records[date] = rets
    actual_df = pd.DataFrame.from_dict(actual_records, orient="index")
    actual_df = actual_df.reindex(columns=all_codes).dropna(how="all")

    for n in FACTOR_NAMES:
        ics = np.array(ic_data[n])[:len(actual_df)]
        mean_actual = actual_df.mean(axis=1).values
        mask = ~np.isnan(ics) & ~np.isnan(mean_actual)
        if mask.sum() > 10:
            corr = np.corrcoef(ics[mask], mean_actual[mask])[0, 1]
            print(f"  {n:<10s}  IC vs 平均 forward 收益 相关 = {corr:+.3f}  (n={mask.sum()})")

    print("\n========= 6. 每月最佳因子 (按 IC 排名) =========")
    monthly_dates = panel.index.to_series().resample("ME").last().index
    monthly_dates = [d for d in monthly_dates if d in panel.index and panel.index.get_loc(d) >= 252]
    best_factor_log = []
    for date in monthly_dates:
        ic = factor_ic_at(panel, date, all_codes,
                          forward_window=20, lookback=60)
        if not ic:
            continue
        best = max(ic, key=ic.get)
        best_factor_log.append({"date": date, "best_factor": best, "best_ic": ic[best]})

    df_best = pd.DataFrame(best_factor_log)
    counts = df_best["best_factor"].value_counts()
    print("  每月 #1 因子出现次数:")
    for f, c in counts.items():
        print(f"    {f:<10s}  {c:3d} ({c/len(df_best)*100:.1f}%)")

    print("\n  各因子作为 #1 时的平均 IC:")
    for f in FACTOR_NAMES:
        sub = df_best[df_best["best_factor"] == f]
        if len(sub) > 0:
            print(f"    {f:<10s}  n={len(sub):3d}  mean_IC={sub['best_ic'].mean():+.3f}")

    print("\n========= 7. 等权 Smart β 基准 vs IC 择时 =========")
    sb_codes = list(SMART_BETA_CODES.values())
    sb_rets = panel[sb_codes].pct_change().fillna(0)
    eq_ret = sb_rets.mean(axis=1)
    eq_nav = (1 + eq_ret).cumprod()
    eq_nav.iloc[0] = 1.0
    eq_m = metrics(eq_nav)
    print(f"  等权 7 Smart β: Ann={eq_m['ann_return']*100:.2f}%  "
          f"Sharpe={eq_m['sharpe']:.2f}  DD={eq_m['max_dd']*100:.2f}%  Calmar={eq_m['calmar']:.3f}")
    for c in sb_codes:
        if c in panel.columns:
            n = panel[c].dropna()
            if len(n) > 0:
                m_single = metrics(n / n.iloc[0])
                print(f"    单独 {c}: Ann={m_single['ann_return']*100:.2f}%  "
                      f"Sharpe={m_single['sharpe']:.2f}  DD={m_single['max_dd']*100:.2f}%")

    print("\n  单因子 Smart β 静态持仓 (max IC 因子 → 等权持仓其代表 ETF):")
    for n in FACTOR_NAMES:
        if n in ("momentum", "reversal"):
            codes = [c[0] for c in STYLE_GROUP_CODES.values()]
        elif n == "value":
            codes = ["512040"]
        elif n == "low_vol":
            codes = ["512260"]
        elif n == "dividend":
            codes = ["510880", "512890", "515080", "515100"]
        elif n == "quality":
            codes = ["515900"]
        else:
            continue
        codes = [c for c in codes if c in panel.columns]
        if not codes:
            continue
        sub = panel[codes].pct_change().fillna(0)
        w_ret = sub.mean(axis=1)
        w_nav = (1 + w_ret).cumprod()
        w_nav.iloc[0] = 1.0
        w_m = metrics(w_nav)
        print(f"    {n:<10s}  n_etf={len(codes)}  Ann={w_m['ann_return']*100:.2f}%  "
              f"Sharpe={w_m['sharpe']:.2f}  DD={w_m['max_dd']*100:.2f}%  Calmar={w_m['calmar']:.3f}")

    out = {
        "ic_decay_means": {
            n: {str(fw): float(np.mean(ic_decay[fw][n])) for fw in fwd_windows}
            for n in FACTOR_NAMES
        },
        "ic_distribution": {
            n: {
                "mean": float(np.mean(ic_data[n])),
                "std": float(np.std(ic_data[n])),
                "hit_rate": float(np.mean(np.array(ic_data[n]) > 0)),
            } for n in FACTOR_NAMES
        },
        "best_factor_counts": counts.to_dict(),
    }
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "factor_timing_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[save] {out_dir / 'factor_timing_diagnostic.json'}")


if __name__ == "__main__":
    main()
