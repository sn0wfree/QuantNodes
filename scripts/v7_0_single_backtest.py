"""
v7.0 单次回测验证 (Stage 30.3.4).

[目的] 验证 run_v7_0_backtest:
1. 跑通全流程, 算出 NAV
2. 与 v6.2 baseline 对比 Calmar
3. 输出 regime 切换历史
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
    V7Config, run_v7_0_backtest, build_regime_timeline,
)
from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest

warnings.filterwarnings("ignore")

OOS_END = "2026-06-30"


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
    print("[v7.0 single backtest] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc[:"2026-06-30"]
    panel_ohlcv = panel_ohlcv.loc[:"2026-06-30"]
    print(f"  panel: {panel_close.shape}")

    # 1. v6.2 baseline (Stage 29 PROMISING, 5-fold mean 1.512)
    print("\n[1/3] v6.2 ir_expanding (Stage 29 baseline)...")
    cfg62 = V6_2Config()
    cfg62.sort_method = "ir_expanding"
    nav62 = run_v6_2_backtest(panel_close, panel_ohlcv, cfg62)
    m62 = metrics(nav62.loc["2022-01-01":])
    print(f"  OOS (2022-2026) Calmar: {m62['calmar']:.3f}")

    # 2. v6.1 IC12 baseline (Stage 27 RECOMMENDED)
    print("\n[2/3] v6.1 IC12 (Stage 27 baseline)...")
    cfg61 = V6_1Config()
    cfg61.ic_min_months = 12
    nav61 = run_v6_1_backtest(panel_close, panel_ohlcv, cfg61)
    m61 = metrics(nav61.loc["2022-01-01":])
    print(f"  OOS (2022-2026) Calmar: {m61['calmar']:.3f}")

    # 3. v7.0 (Stage 30.3 新)
    print("\n[3/3] v7.0 (Stage 30.3, 5 状态 vol_target)...")
    cfg7 = V7Config()
    cfg7.sort_method = "ir_expanding"   # 沿用 v6.2 Stage 29 默认
    cfg7.use_regime = True             # 启用 5 状态

    # 预计算 regime timeline (PIT 调整)
    print("  预计算 5 状态时间线 (PIT 调整)...")
    timeline = build_regime_timeline(start="2018-06-01", end="2026-06-30")
    print(f"  timeline 行数: {len(timeline)}")
    print(f"  状态分布: {timeline['regime'].value_counts().to_dict()}")

    nav7 = run_v7_0_backtest(panel_close, panel_ohlcv, cfg7, regime_timeline=timeline)
    m7 = metrics(nav7.loc["2022-01-01":])
    print(f"  OOS (2022-2026) Calmar: {m7['calmar']:.3f}")

    # 4. 对比
    print("\n=== 3 策略 OOS Calmar 对比 (2022-2026) ===")
    print(f"  v6.1 IC12:        {m61['calmar']:.3f}  (ann_ret={m61['ann']*100:.2f}%, DD={m61['dd']*100:.2f}%)")
    print(f"  v6.2 ir_expanding:{m62['calmar']:.3f}  (ann_ret={m62['ann']*100:.2f}%, DD={m62['dd']*100:.2f}%)")
    print(f"  v7.0 (5 状态 VT): {m7['calmar']:.3f}  (ann_ret={m7['ann']*100:.2f}%, DD={m7['dd']*100:.2f}%)")

    # 5. 状态切换统计
    timeline["date"] = pd.to_datetime(timeline["date"])
    timeline["month"] = timeline["date"].dt.to_period("M")
    monthly_dominant = timeline.groupby("month")["regime"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "neutral")
    print(f"\n  5 状态月度数: {monthly_dominant.value_counts().to_dict()}")

    # 6. 保存 NAV
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v7_0_single_backtest.csv"
    nav_df = pd.DataFrame({
        "date": nav62.index,
        "v6_1_IC12": nav61.values,
        "v6_2_ir_expanding": nav62.values,
        "v7_0": nav7.values,
    })
    nav_df.to_csv(out_path, index=False)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
