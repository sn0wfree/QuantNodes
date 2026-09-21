#!/usr/bin/env python3
# coding=utf-8
"""构建 panel_v5: panel_v4 + 候选 ETF 扩充 (A+B 路线图 §阶段2).

数据源: data/real/etf_nav_2018-01-01_2026-06-30.parquet
       data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet
新增 14 只 ETF (宽基/海外/红利/商品/债券), 用于池对照实验 (B 阶段).

注: 本地 ETF 数据止 2026-06-30 (panel_v4 止 2026-09-03),
    池对照窗口取 2018-01 → 2026-06-30 (公共区间).

运行: python -m QuantNodes.strategy.allweather.data_etl.build_panel_v5
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# 新增候选 ETF (code → 名称备注); 均已确认本地有数据
NEW_ETFS: dict[str, str] = {
    "510500": "中证500",
    "510050": "上证50",
    "512400": "有色金属ETF",
    "159928": "消费ETF",
    "513050": "中概互联",
    "159920": "恒生ETF",
    "510880": "红利ETF",
    "518880": "黄金ETF",
    "511260": "10年国债ETF",
    "512880": "证券ETF",
    "510900": "H股ETF",
    "588000": "科创50ETF",
    "159985": "豆粕ETF",
    "513520": "日经225ETF",
}


def build_panel_v5(
    panel_v4_path: str = "data/allweather/panel_v4.parquet",
    out_path: str = "data/allweather/panel_v5.parquet",
) -> pd.DataFrame:
    """panel_v4 + 14 只候选 ETF → panel_v5."""
    panel = pd.read_parquet(panel_v4_path)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()
    print(f"panel_v4: {panel.shape} ({panel.index[0].date()} → {panel.index[-1].date()})")

    nav = pd.read_parquet(
        "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    nav.index = pd.to_datetime(nav.index)
    sb = pd.read_parquet(
        "data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    sb.index = pd.to_datetime(sb.index)
    combined = pd.concat([nav, sb], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]

    added, skipped = [], []
    for code, label in NEW_ETFS.items():
        col = f"etf_{code}"
        if col in panel.columns:
            skipped.append(col)
            continue
        if code not in combined.columns:
            skipped.append(f"{col}(无数据)")
            continue
        s = combined[code].dropna()
        if len(s) < 250:
            skipped.append(f"{col}(仅{len(s)}行)")
            continue
        panel[col] = s.reindex(panel.index)
        added.append(f"{col}({label}, {s.index[0].date()}+)")

    print(f"\n新增 {len(added)} 只:")
    for a in added:
        print(f"  + {a}")
    if skipped:
        print(f"跳过: {skipped}")

    print(f"\npanel_v5: {panel.shape} "
          f"({panel.index[0].date()} → {panel.index[-1].date()})")
    nonnull = panel.notna().mean()
    low = nonnull[nonnull < 0.9].sort_values()
    print(f"非空率 <90% 的列 ({len(low)}):")
    for c, v in low.items():
        last = panel[c].dropna().index[-1].date()
        print(f"  {c}: {v*100:.1f}% (末 {last})")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    print(f"\n已保存: {out_path}")
    return panel


if __name__ == "__main__":
    build_panel_v5()
