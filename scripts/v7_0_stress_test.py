"""v7.0 极端市压测 (Stage 30.5 Phase A3).

[动机] 单次回测未涵盖极端市, 必须单独测 3 关键事件:
    1. 2020-02-03 ~ 2020-03-23: 疫情暴跌 + 反弹
    2. 2022-04: 港股暴跌 (中概互联 -15%)
    3. 2024-09-24 ~ 2024-10-08: 政策反转 (A 股 +20%)

[测试]
    在每个事件窗口, 跑 5 方案 + baseline, 输出:
    - 窗口累计收益
    - 窗口最大 DD
    - 窗口 Sharpe (日频)
    - 窗口最大单日回撤

[输出] reports/.../v7_0_stress_test.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    build_regime_timeline,
    run_topk_v7_backtest,
    run_bl_v7_backtest,
    run_beta_v7_backtest,
    run_momentum_v7_backtest,
    run_iv_v7_backtest,
)

warnings.filterwarnings("ignore")

ETFS = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']

STRESS_EVENTS = [
    ("2020-02-03", "2020-03-23", "疫情暴跌+反弹 (国内春节后+全球熔断)"),
    ("2022-04-01", "2022-04-29", "2022 港股暴跌 + A 股深跌"),
    ("2024-09-24", "2024-10-08", "2024 政策反转 (A 股急涨)"),
]

STRATEGIES = {
    "A_topk": ("Top-K (K=5)", lambda p, t: run_topk_v7_backtest(p, t, k=5)),
    "B_bl": ("Black-Litterman", lambda p, t: run_bl_v7_backtest(p, t, tau=0.05, max_weight=0.30)),
    "C_beta": ("Macro Beta (K=5)", lambda p, t: run_beta_v7_backtest(p, t, lookback=252, k=5)),
    "D_momentum": ("Momentum (63d)", lambda p, t: run_momentum_v7_backtest(p, t, lookback=63, k=5)),
    "E_iv": ("Inverse Vol", lambda p, t: run_iv_v7_backtest(p, t, lookback=252, max_weight=0.30)),
}


def event_metrics(nav: pd.Series, event_start: str, event_end: str) -> dict:
    """事件窗口 metrics.

    因 5 方案是月度 rebal, 用 rebal 前最近 NAV 与 rebal 后最近 NAV 计算.
    """
    s = nav.dropna()
    if len(s) < 2:
        return {"event_total": 0.0, "event_max_dd": 0.0, "event_daily_sharpe": 0.0, "event_max_daily_drop": 0.0}
    es = pd.Timestamp(event_start)
    ee = pd.Timestamp(event_end)
    before_idx = s.index[s.index <= es]
    after_idx = s.index[s.index >= ee]
    if len(before_idx) == 0 or len(after_idx) == 0:
        return {"event_total": 0.0, "event_max_dd": 0.0, "event_daily_sharpe": 0.0, "event_max_daily_drop": 0.0}
    nav_before = s.loc[before_idx[-1]]
    nav_after = s.loc[after_idx[0]]
    total = nav_after / nav_before - 1
    window = s.loc[before_idx[-1]:after_idx[0]]
    dd = (window / window.cummax() - 1).min() if len(window) > 1 else 0.0
    daily_ret = window.pct_change().dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if (len(daily_ret) > 1 and daily_ret.std() > 0) else 0.0
    max_drop = daily_ret.min() if len(daily_ret) > 0 else 0.0
    return {
        "event_total": total,
        "event_max_dd": dd,
        "event_daily_sharpe": sharpe,
        "event_max_daily_drop": max_drop,
    }


def main() -> None:
    print("[v7.0 极端市压测] 加载数据...")
    nav_main = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    sb = pd.read_parquet(REPO / "data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    panel_full = pd.DataFrame()
    for c in ETFS:
        if c in nav_main.columns:
            s = nav_main[c].dropna()
        elif c in sb.columns:
            s = sb[c].dropna()
        else:
            continue
        panel_full[c] = s
    panel_full = panel_full.dropna(how='all').ffill().dropna()
    print(f"  panel: {panel_full.shape}, range: {panel_full.index[0].date()} - {panel_full.index[-1].date()}")

    tl_df = build_regime_timeline()
    tl_df['date'] = pd.to_datetime(tl_df['date'])
    tl_df = tl_df.set_index('date')

    eq_nav_full = (1 + panel_full.pct_change().mean(axis=1)).cumprod()
    eq_nav_full.name = "nav_cum"

    rows = []
    for event_start, event_end, event_desc in STRESS_EVENTS:
        print(f"\n=== 事件: {event_desc} ===")
        print(f"  区间: {event_start} ~ {event_end}")

        baseline = event_metrics(eq_nav_full, event_start, event_end)
        print(f"  baseline 等权 7 ETF: total={baseline['event_total']*100:+.2f}% "
              f"max_dd={baseline['event_max_dd']*100:.2f}% "
              f"max_daily_drop={baseline['event_max_daily_drop']*100:.2f}%")

        rows.append({
            "event": event_desc,
            "event_start": event_start,
            "event_end": event_end,
            "strategy": "baseline",
            "event_total": baseline["event_total"],
            "event_max_dd": baseline["event_max_dd"],
            "event_daily_sharpe": baseline["event_daily_sharpe"],
            "event_max_daily_drop": baseline["event_max_daily_drop"],
        })

        for strat_key, (strat_name, strat_fn) in STRATEGIES.items():
            try:
                nav_df, weights_df, _ = strat_fn(panel_full, tl_df)
                m = event_metrics(nav_df["nav_cum"], event_start, event_end)
                print(f"  {strat_name:25s}  total={m['event_total']*100:+.2f}% "
                      f"max_dd={m['event_max_dd']*100:.2f}% "
                      f"max_daily_drop={m['event_max_daily_drop']*100:.2f}%")
                rows.append({
                    "event": event_desc,
                    "event_start": event_start,
                    "event_end": event_end,
                    "strategy": strat_key,
                    "event_total": m["event_total"],
                    "event_max_dd": m["event_max_dd"],
                    "event_daily_sharpe": m["event_daily_sharpe"],
                    "event_max_daily_drop": m["event_max_daily_drop"],
                })
            except Exception as e:
                print(f"  {strat_name:25s}  ERROR: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    csv_path = out_dir / "v7_0_stress_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    print("\n=== 压测汇总 (按事件) ===")
    for event in df["event"].unique():
        sub = df[df["event"] == event].sort_values("event_total", ascending=False)
        print(f"\n{event}:")
        print(sub[["strategy", "event_total", "event_max_dd", "event_max_daily_drop"]].to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
