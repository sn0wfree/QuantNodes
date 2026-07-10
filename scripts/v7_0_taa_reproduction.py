"""
v7.0 TAA 复现版回测 (Stage 30.4 reproduction).

[目的] 验证 state-based TAA 能否复现业界 19% 年化.
[对比基准]
- 沪深300 (510300) 静态持有: ~4% 年化
- 中证500 (510500) 静态持有: ~6% 年化
- 创业板 (159915) 静态持有: ~12% 年化
- 黄金 (518880) 静态持有: ~14% 年化
- 等权组合 (5 ETF): ~8% 年化
- v7.0 TAA (5 状态切换): 目标 ≥15% 年化
- 业界 macro 19% / 国泰海通 22.67% / 26.84%
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
    V7Config, run_v7_taa_backtest, build_regime_timeline,
    state_history_to_df, STATE_ALLOCATIONS,
)

warnings.filterwarnings("ignore")


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    return {
        "ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else 0,
        "dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0,
    }


def main():
    print("[v7.0 TAA reproduction] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    print(f"  panel: {panel_close.shape}")

    # 1. 单 ETF 静态持有 (基准)
    print("\n=== 单 ETF 静态持有 (2018-2026, 8.5 年) ===")
    etfs = {
        "510300": "沪深300",
        "510500": "中证500",
        "159915": "创业板",
        "518880": "黄金",
    }
    for code, name in etfs.items():
        s = panel_close[code].dropna()
        if len(s) < 252:
            continue
        sub = s.loc[:"2026-06-30"]
        m = metrics(sub)
        print(f"  {code} {name:6s}: 年化 {m['ann']*100:6.2f}%, Sharpe {m['sharpe']:5.2f}, DD {m['dd']*100:6.2f}%, Calmar {m['calmar']:5.2f}")

    # 2. 等权 4 ETF 组合 (基准)
    print("\n=== 等权 4 ETF 组合 (沪深300/中证500/创业板/黄金) ===")
    nav_eq = pd.Series(1.0, index=panel_close.index)
    for i, date in enumerate(panel_close.index):
        if i == 0:
            continue
        daily_ret = 0.0
        n_active = 0
        for code in etfs.keys():
            if code in panel_close.columns:
                p_t = panel_close[code].iloc[i]
                p_prev = panel_close[code].iloc[i - 1]
                if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                    daily_ret += 0.25 * (p_t / p_prev - 1.0)
                    n_active += 1
        if n_active > 0:
            # 重新归一化 (因半导体未上市早期只 4 ETF)
            scale = 4 / n_active
            daily_ret *= scale
        nav_eq.iloc[i] = nav_eq.iloc[i - 1] * (1 + daily_ret)
    m_eq = metrics(nav_eq)
    print(f"  等权 4 ETF: 年化 {m_eq['ann']*100:.2f}%, Sharpe {m_eq['sharpe']:.2f}, DD {m_eq['dd']*100:.2f}%, Calmar {m_eq['calmar']:.2f}")

    # 3. v7.0 TAA (5 状态切换)
    print("\n=== v7.0 TAA (5 状态 HMM 驱动) ===")
    cfg = V7Config()
    print("  5 状态权重表:")
    for state, weights in STATE_ALLOCATIONS.items():
        w_str = " | ".join([f"{c} {w*100:>4.0f}%" for c, w in weights.items() if w > 0])
        print(f"    {state:12s}: {w_str}")

    print("\n  预计算 5 状态时间线 (PIT 调整)...")
    timeline = build_regime_timeline(start="2018-06-01", end="2026-06-30")

    print("  跑 v7.0 TAA 回测...")
    nav7, history = run_v7_taa_backtest(panel_close, cfg, regime_timeline=timeline)
    m7 = metrics(nav7)
    print(f"  v7.0 TAA: 年化 {m7['ann']*100:.2f}%, Sharpe {m7['sharpe']:.2f}, DD {m7['dd']*100:.2f}%, Calmar {m7['calmar']:.2f}")

    # 4. 状态切换统计
    history_df = state_history_to_df(history)
    print(f"\n=== 5 状态切换统计 (共 {len(history)} 次调仓) ===")
    regime_count = history_df.groupby("regime")["date"].nunique()
    print(regime_count.to_string())

    # 5. 5 状态月度数
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline["month"] = timeline["date"].dt.to_period("M")
    monthly = timeline.groupby(["month", "regime"]).size().unstack(fill_value=0)
    monthly["dominant"] = monthly.idxmax(axis=1)
    print(f"\n=== 5 状态月数分布 (2018-2026) ===")
    print(monthly["dominant"].value_counts().to_string())

    # 6. 关键对比
    print(f"\n=== 复现目标 vs 实际 ===")
    print(f"  业界 19% 复现目标:  ≥ 19.00%")
    print(f"  实际 v7.0 TAA:        {m7['ann']*100:.2f}% (Calmar {m7['calmar']:.2f})")
    if m7["ann"] >= 0.19:
        print(f"  ✓ 复现成功! 超过业界 19% 基准")
    elif m7["ann"] >= 0.15:
        print(f"  ⚠ 接近 19% (≥15%), 需微调权重")
    else:
        print(f"  ✗ 差距较大, 需重新设计权重")

    # 7. 保存 NAV
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v7_0_taa_reproduction.csv"
    nav_df = pd.DataFrame({
        "date": panel_close.index,
        "v7_0_taa": nav7.values,
    })
    nav_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")

    # 8. 保存 state history
    hist_path = out_dir / "v7_0_taa_state_history.csv"
    history_df.to_csv(hist_path, index=False)
    print(f"[saved] {hist_path}")


if __name__ == "__main__":
    main()
