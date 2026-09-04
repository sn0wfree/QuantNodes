# coding=utf-8
"""cta_gold_eval.py — 第2步: 黄金策略评估 (IS重调参 + OOS验证).

背景: gold_combined 注释自认"参数优化 2018-2025 目标" = 全样本调优,
OOS(2022-25) 已被污染. 本脚本严格执行:
    1. IS(2018-2021) 网格重调: trend_bnh_weight × trend_stop (12组)
    2. OOS(2022-2025) 纯验证: 仅用 IS 最优参数跑 OOS
口径: 周频信号 → 日频 ffill → vol_target_backtest(smoothed_au, cost 2bp,
      tv 8%, pit_shift=1) — 与子策略体系一致.

PIT 快审: tips/gvz/vix/dxy/etf_tonnes 均为日度EOD ✓;
position_flow 底层 CFTC (weight 0.05, 且方向已证伪) — 审计记录, 不修.

门槛: OOS Sharpe≥0.15 且 r(主CTA)<0.3.
输出: reports/cta/gold_eval.csv + gold_eval_navs.parquet
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.factor_research import vol_target_backtest  # noqa: E402

warnings.filterwarnings("ignore")

from QuantNodes.strategy.commodity_fundamental.gold.gold_combined import (  # noqa: E402
    compute_combined_signal,
)

REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
COST = 2.0
TV = 0.08

GRID = [(tw, ts) for tw in (0.40, 0.50, 0.60, 0.70) for ts in (0.10, 0.12, 0.15)]


def to_daily_ret(pos_w: pd.Series, close: pd.Series) -> pd.Series:
    sig = pos_w.reindex(close.index, method="ffill").fillna(0).to_frame("AU")
    res = vol_target_backtest(sig, close.to_frame("AU"), cost_bps={"AU": COST},
                              target_vol=TV, pit_shift=1)
    return res["nav"].pct_change(fill_method=None).fillna(0)


def seg_sharpe(r: pd.Series, lo: str | None, hi: str | None) -> float:
    r = r.dropna()
    if lo:
        r = r[r.index >= lo]
    if hi:
        r = r[r.index < hi]
    if len(r) < 50 or r.std() == 0:
        return float("nan")
    return r.mean() / r.std() * np.sqrt(252)


def main() -> None:
    import duckdb
    con_cta = duckdb.connect("/home/ll/Public/DataCache/futures_options_daily.duckdb",
                             read_only=True)
    df = con_cta.execute(
        "SELECT trade_date, close FROM smoothed_au_daily "
        "WHERE close IS NOT NULL ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
    close = close[close.index <= "2025-12-31"]

    print("=" * 96)
    print("第2步: 黄金 — IS(2018-21)网格重调 → OOS(2022-25)纯验证")
    print("=" * 96)

    rows = []
    rets = {}
    for tw, ts in GRID:
        sig = compute_combined_signal(trend_bnh_weight=tw, trend_stop=ts)
        r = to_daily_ret(sig["position"], close)
        rets[(tw, ts)] = r
        is_s = seg_sharpe(r, None, IS_END)
        oos_s = seg_sharpe(r, IS_END, None)
        rows.append({"trend_w": tw, "trend_stop": ts,
                     "is_sharpe": round(is_s, 3), "oos_sharpe": round(oos_s, 3)})
        print(f"  trend_w={tw:.2f} stop={ts:.2f}: IS {is_s:+.3f} | OOS {oos_s:+.3f}")

    df_g = pd.DataFrame(rows)
    best = df_g.loc[df_g["is_sharpe"].idxmax()]
    print(f"\n★ IS最优: trend_w={best['trend_w']}, stop={best['trend_stop']} "
          f"(IS {best['is_sharpe']:+.3f}) → 其OOS为纯样本外: {best['oos_sharpe']:+.3f}")

    # OOS 验证 + 相关性
    best_r = rets[(best["trend_w"], best["trend_stop"])]
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_r = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)
    common = best_r.index.intersection(main_r.index)
    r_main = best_r[common].corr(main_r[common])
    oos_ok = best["oos_sharpe"] >= 0.15
    corr_ok = abs(r_main) < 0.3
    print(f"\n门槛: OOS≥0.15 ({'PASS' if oos_ok else 'FAIL'}) | "
          f"r(主CTA)<0.3: {r_main:+.3f} ({'PASS' if corr_ok else 'FAIL'})")
    print(f"裁决: {'PASS — 可进入组合接入测试' if oos_ok and corr_ok else 'FAIL — 不接入'}")

    best_r.to_frame("ret").to_parquet(f"{REPORT_DIR}/gold_eval_navs.parquet")
    df_g.to_csv(f"{REPORT_DIR}/gold_eval.csv", index=False)
    print("\n落盘: gold_eval.csv + gold_eval_navs.parquet")


if __name__ == "__main__":
    main()
