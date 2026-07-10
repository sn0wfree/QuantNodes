# coding=utf-8
"""v6.1 + v6.2 整合脚本 (Stage 27):

跑 2 个消融 (v6.1 七组 + v6.2 六组), 合并成单一 NAV parquet,
便于集成到 combo/ HTML 图表中.

新增 2 列:
- v6.1 IC12 (Stage 27 推荐 v6.1)
- v6.2 orth_IC36 (Stage 27 推荐 v6.2, OOS Calmar 0.901 新冠军)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    r = s.pct_change().dropna()
    n = len(r)
    if n < 2:
        return {"calmar": 0.0, "dd": 0.0, "ann": 0.0, "sharpe": 0.0}
    ann = (1 + r).prod() ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (s / s.cummax() - 1).min()
    return {"calmar": ann / abs(dd) if dd != 0 else 0, "dd": dd, "ann": ann, "sharpe": ann / vol if vol > 0 else 0}


OOS_START = "2022-01-01"
OOS_END = "2026-06-30"


def main() -> None:
    print("[v6_combined] 加载数据...")
    panel_close = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    panel_ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    panel_close = panel_close.loc["2018-01-01":OOS_END]
    panel_ohlcv = panel_ohlcv.loc["2018-01-01":OOS_END]

    out = pd.DataFrame(index=panel_close.index)

    # v6.1 IC12 (推荐配置)
    print("\n[v6.1 IC12]")
    cfg = V6_1Config(ic_min_months=12)
    nav = run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    out["v6.1 IC12"] = nav
    om = metrics(nav.loc[OOS_START:OOS_END])
    print(f"  OOS Calmar {om['calmar']:.3f}, DD {om['dd']:.2%}, ann {om['ann']:+.2%}")

    # v6.2 orth_IC36 (推荐配置)
    print("\n[v6.2 orth_IC36]")
    cfg = V6_2Config(ic_min_months=36, use_orthogonal=True)
    nav = run_v6_2_backtest(panel_close, panel_ohlcv, cfg)
    out["v6.2 (正交+IC36)"] = nav
    om = metrics(nav.loc[OOS_START:OOS_END])
    print(f"  OOS Calmar {om['calmar']:.3f}, DD {om['dd']:.2%}, ann {om['ann']:+.2%}")

    # 保存
    out_dir = REPO / "reports/momentum_etf_rotation/combo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v6_1_v6_2_combined_navs.parquet"
    out.to_parquet(out_path)
    print(f"\n[save] {out_path} ({out.shape[1]} cols, {out.shape[0]} rows)")


if __name__ == "__main__":
    main()
