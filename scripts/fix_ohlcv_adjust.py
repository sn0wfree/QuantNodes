# coding=utf-8
"""OHLCV 前复权修复: 检测并平滑拆合股导致的跳变.

检测日收益 >|50%| 的分红/拆股/缩股事件,
对事件之前的价格做前复权乘数调整.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data/real" / "etf_ohlcv_2018-01-01_2026-06-30.parquet"
DST = REPO / "data/real" / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"

THRESHOLD = 0.50  # 日收益超过此阈值视为公司行为


def detect_corp_actions(panel: pd.DataFrame) -> list[dict]:
    """检测所有 >50% 价格跳变."""
    actions = []
    for code in panel.columns.levels[0]:
        c = panel[code]["close"]
        ret = c / c.shift(1) - 1
        spikes = ret[ret.abs() > THRESHOLD]
        for date in spikes.index:
            prev = c.shift(1).loc[date]
            curr = c.loc[date]
            actions.append({
                "code": code,
                "date": date,
                "ratio": curr / prev,
                "prev_close": prev,
                "curr_close": curr,
            })
    return pd.DataFrame(actions).sort_values("date").to_dict("records")


def forward_adjust(panel: pd.DataFrame, actions: list[dict]) -> pd.DataFrame:
    """对 OHLCV 做前复权: 事件前价格 × ratio, 使时序连续.

    注意: 调整乘数累乘, 多个事件时从最早到最晚依次处理.
    """
    adj = panel.copy()
    for act in actions:
        code = act["code"]
        ratio = act["ratio"]
        cut_date = act["date"]
        for field in ["open", "high", "low", "close", "volume"]:
            col = adj[code][field]
            mask = col.index < cut_date
            adj.loc[mask, (code, field)] = col[mask] * ratio
        print(f"  [adj] {code} @ {cut_date:%Y-%m-%d}: prev={act['prev_close']:.4f} → curr={act['curr_close']:.4f}  ratio={ratio:.4f}")
    return adj


def main():
    print("[load] 读取 OHLCV ...")
    panel = pd.read_parquet(SRC)
    print(f"  ETF 数: {len(panel.columns.levels[0])}, 日期: {len(panel.index)}")

    print("[detect] 扫描公司行为 ...")
    actions = detect_corp_actions(panel)
    print(f"  发现 {len(actions)} 个事件:")
    for a in actions:
        print(f"    {a['code']} @ {a['date']:%Y-%m-%d}: {a['prev_close']:.4f} → {a['curr_close']:.4f} ({a['ratio']-1:.2%})")

    print("[adjust] 前复权 ...")
    adj = forward_adjust(panel, actions)

    print(f"[save] → {DST}")
    DST.parent.mkdir(parents=True, exist_ok=True)
    adj.to_parquet(DST, index=True)

    # 验证: 检查调整后的事件日期波动
    print("[verify] 调整后 >5% 日收益次数:")
    for code in set(a["code"] for a in actions):
        c = adj[code]["close"]
        ret = c / c.shift(1) - 1
        spikes = ret[ret.abs() > 0.05]
        print(f"  {code}: {len(spikes)} 天 >5% (原含 1 天 >50%)")


if __name__ == "__main__":
    main()
