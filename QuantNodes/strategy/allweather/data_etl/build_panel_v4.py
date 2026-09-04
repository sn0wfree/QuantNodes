"""panel_v4 重建 (§5.34): 合并 panel_v3 + akshare 增量数据 → 输出 panel_v4.

使用 panel_v3 (DuckDB + 原 NAV parquet) 作为基础, 覆盖 2026-07-08 起的增量
(akshare 拉的 ETF/期货) → 输出 panel_v4.parquet 截止 2026-09-03 (当前可达).

运行:
    python -m QuantNodes.strategy.allweather.data_etl.build_panel_v4
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd



def build_panel_v4(
    panel_v3_path: str = "data/allweather/panel_v3.parquet",
    inc_dir: str = "data/allweather/akshare_inc",
    panel_v4_path: str = "data/allweather/panel_v4.parquet",
) -> pd.DataFrame:
    """合并 panel_v3 + akshare 增量 → panel_v4.

    panel_v4 = panel_v3.loc[:<增量起点>].combine_first(增量面板).
    """
    panel_v3 = pd.read_parquet(panel_v3_path)
    print(f"panel_v3: {panel_v3.shape} ({panel_v3.index.min().date()} → "
          f"{panel_v3.index.max().date()})")

    etf_inc = pd.read_parquet(Path(inc_dir) / "etf_inc_2026-07-08_2026-09-03.parquet")
    fut_inc = pd.read_parquet(Path(inc_dir) / "futures_inc_2026-07-08_2026-09-03.parquet")
    print(f"akshare ETF 增量: {etf_inc.shape}, 期货增量: {fut_inc.shape}")

    # 对齐索引: 增量面板索引是 RangeIndex (pos 0=起点), 需要转换为 datetime
    start_date = panel_v3.index.max()  # panel_v3 截止日 = 增量起点
    for inc_df in (etf_inc, fut_inc):
        if not isinstance(inc_df.index, pd.DatetimeIndex):
            inc_df.index = pd.bdate_range(start=start_date, periods=len(inc_df))
    print(f"对齐索引后 ETF 增量首日: {etf_inc.index[0].date()}")

    # 合并
    panel_v4 = panel_v3.combine_first(etf_inc).combine_first(fut_inc)
    panel_v4 = panel_v4.sort_index()
    print(f"\npanel_v4: {panel_v4.shape} ({panel_v4.index.min().date()} → "
          f"{panel_v4.index.max().date()})")

    # 保存
    Path(panel_v4_path).parent.mkdir(parents=True, exist_ok=True)
    panel_v4.to_parquet(panel_v4_path)
    print(f"已保存: {panel_v4_path}")

    # 覆盖率
    coverage = panel_v4.notna().mean()
    print(f"\n非空率: 平均 {coverage.mean():.2%}")
    for col in panel_v4.columns:
        rate = coverage[col]
        if rate < 0.95:
            last = panel_v4[col].dropna().index.max() if rate > 0 else None
            print(f"  ⚠ {col}: {rate:.2%} (末 {last.date() if last else 'N/A'})")

    return panel_v4


if __name__ == "__main__":
    build_panel_v4()
