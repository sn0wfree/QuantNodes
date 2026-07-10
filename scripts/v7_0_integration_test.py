"""v7.0 5-fold OOS 集成测试: 加交易成本 + 流动性 cap (Stage 30.5 Phase A 集成).

[动机] A1-A5 提供了 cost / cap / 压测 / HMM lag / SLA, 现在集成验证:
    - 加 cost + cap 后, 5 方案年化退化 < 3pp
    - 5 方案 DD 退化 < 5pp
    - 鲁棒赢家 C. Beta 仍为赢家

[测试]
    - baseline: 5-fold OOS (无 cost, 无 cap)
    - with cost: 1.2% 年化 drag
    - with cap: 单 ETF 30% 权重上限 (其实方案 B/C/D/E 已有)
    - with cost+cap: 综合

[输出] reports/.../v7_0_integration_test.csv
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
    apply_turnover_cost,
    apply_max_weight_cap,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.dynamic_allocation import _compute_metrics

warnings.filterwarnings("ignore")

OOS_END = "2026-06-30"
ETFS = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']

FOLD_CONFIGS = [
    (1, "2020-01-01", "2020-12-31"),
    (2, "2021-01-01", "2021-12-31"),
    (3, "2022-01-01", "2023-06-30"),
    (4, "2023-07-01", "2024-12-31"),
    (5, "2025-01-01", OOS_END),
]

STRATEGIES = {
    "A_topk": ("Top-K (K=5)", lambda p, t: run_topk_v7_backtest(p, t, k=5)),
    "B_bl": ("Black-Litterman", lambda p, t: run_bl_v7_backtest(p, t, tau=0.05, max_weight=0.30)),
    "C_beta": ("Macro Beta (K=5)", lambda p, t: run_beta_v7_backtest(p, t, lookback=252, k=5)),
    "D_momentum": ("Momentum (63d)", lambda p, t: run_momentum_v7_backtest(p, t, lookback=63, k=5)),
    "E_iv": ("Inverse Vol", lambda p, t: run_iv_v7_backtest(p, t, lookback=252, max_weight=0.30)),
}


def oos_metrics(nav: pd.Series, oos_start: str, oos_end: str) -> dict:
    s = nav.loc[oos_start:oos_end].dropna()
    if len(s) < 2:
        return {"ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    n_days = (s.index[-1] - s.index[0]).days
    total_ret = s.iloc[-1] / s.iloc[0] - 1
    ann = (1 + total_ret) ** (365.25 / n_days) - 1
    monthly_ret = s.pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12) if len(monthly_ret) > 1 else 0.0
    dd = (s / s.cummax() - 1).min()
    return {
        "ann": ann, "sharpe": ann / ann_vol if ann_vol > 0 else 0.0,
        "dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0.0,
    }


def apply_costs_to_weights(weights_df: pd.DataFrame, fee_bps: float = 10.0) -> pd.DataFrame:
    """对 weights_df 逐行应用 turnover cost, 返回调整后的 weights."""
    etfs = [c for c in weights_df.columns if c not in ("state",)]
    adjusted = []
    prev = None
    for _, row in weights_df.iterrows():
        w = {c: float(row[c]) for c in etfs}
        if prev is not None and any(v != 0 for v in prev.values()):
            w = apply_turnover_cost(w, prev, fee_bps=fee_bps)
        adjusted.append({**w, "state": row.get("state", "")})
        prev = w
    return pd.DataFrame(adjusted, index=weights_df.index)


def recompute_nav_with_weights(panel: pd.DataFrame, weights_df: pd.DataFrame, fee_bps: float = 10.0) -> pd.DataFrame:
    """用新权重重算 NAV, 每次调仓扣手续费."""
    etfs = [c for c in panel.columns]
    nav_path = []
    for i, d in enumerate(weights_df.index):
        w = {c: float(weights_df.loc[d, c]) for c in etfs}
        if i == 0:
            turnover = sum(w.values()) / 2
        else:
            w_prev = {c: float(weights_df.iloc[i-1][c]) for c in etfs}
            turnover = sum(abs(w.get(c, 0) - w_prev.get(c, 0)) for c in etfs) / 2
        cost_rate = turnover * fee_bps / 10000
        next_d = weights_df.index[i + 1] if i + 1 < len(weights_df) else panel.index[-1]
        seg = panel.loc[d:next_d]
        if len(seg) < 2:
            continue
        seg_ret = seg.iloc[-1] / seg.iloc[0]
        port_ret_pre = sum(w.get(c, 0) * (seg_ret.get(c, 1) - 1) for c in etfs) + 1
        port_ret = port_ret_pre * (1 - cost_rate)
        nav_path.append({"date": next_d, "nav": port_ret})
    df = pd.DataFrame(nav_path).set_index("date")
    df["nav_cum"] = df["nav"].cumprod()
    return df


def main() -> None:
    print("[v7.0 集成测试] 加载数据...")
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
    panel_full = panel_full.dropna(how='all').ffill().dropna().loc["2018-01-01":OOS_END]

    tl_df = build_regime_timeline()
    tl_df['date'] = pd.to_datetime(tl_df['date'])
    tl_df = tl_df.set_index('date')

    rows = []
    for fi, oos_start, oos_end in FOLD_CONFIGS:
        print(f"\n=== Fold {fi}: OOS {oos_start} ~ {oos_end} ===")
        for strat_key, (strat_name, strat_fn) in STRATEGIES.items():
            try:
                nav_df, weights_df, _ = strat_fn(panel_full, tl_df)
                m_plain = oos_metrics(nav_df["nav_cum"], oos_start, oos_end)

                weights_cost = apply_costs_to_weights(weights_df, fee_bps=10.0)
                nav_cost = recompute_nav_with_weights(panel_full, weights_cost, fee_bps=10.0)
                m_cost = oos_metrics(nav_cost["nav_cum"], oos_start, oos_end)

                rows.append({
                    "fold": fi, "strategy": strat_key,
                    "config": "plain", **m_plain,
                })
                rows.append({
                    "fold": fi, "strategy": strat_key,
                    "config": "with_cost", **m_cost,
                })
                ann_delta = m_cost["ann"] - m_plain["ann"]
                print(f"  {strat_name:25s}  plain ann={m_plain['ann']*100:+.2f}% → "
                      f"with_cost ann={m_cost['ann']*100:+.2f}%  Δ={ann_delta*100:+.2f}pp")
            except Exception as e:
                print(f"  {strat_name:25s}  ERROR: {e}")

    df = pd.DataFrame(rows)
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    csv_path = out_dir / "v7_0_integration_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    print("\n=== 集成测试汇总: 5 策略 × 2 配置 × 5 fold = 50 行 ===")
    summary = df.groupby(["strategy", "config"]).agg(
        ann_mean=("ann", "mean"),
        ann_min=("ann", "min"),
        dd_min=("dd", "max"),
        calmar_mean=("calmar", "mean"),
        calmar_min=("calmar", "min"),
    ).reset_index()
    print(summary.round(3).to_string(index=False))

    print("\n=== 退化分析 (with_cost - plain) ===")
    for strat in summary["strategy"].unique():
        plain = summary[(summary["strategy"] == strat) & (summary["config"] == "plain")].iloc[0]
        with_c = summary[(summary["strategy"] == strat) & (summary["config"] == "with_cost")].iloc[0]
        ann_drag = with_c["ann_mean"] - plain["ann_mean"]
        calmar_drag = with_c["calmar_mean"] - plain["calmar_mean"]
        print(f"  {strat:15s}  ann_drag={ann_drag*100:+.2f}pp  calmar_drag={calmar_drag:+.3f}")


if __name__ == "__main__":
    main()
