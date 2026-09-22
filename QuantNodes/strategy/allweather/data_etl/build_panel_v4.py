"""panel_v4 重建 (§5.34 P0 修复): 合并 panel_v3 + akshare 增量数据 → 输出 panel_v4.

修复要点 (P0-3/P0-4):
- 不再用 panel_v3.index.max() 作为增量起点 (会丢 2026-07-08→07-20 的 9 天)
- 不再用 bdate_range 推算末日期 (会创建幽灵日期到 2026-09-16)
- 用 parquet 文件名解析真实 [start, end], 直接按日期窗口拼接

合并逻辑:
- panel_v3 只保留 < inc_start 的日期
- 增量取 inc_start → inc_end 全部 (覆盖 panel_v3 在该窗口的旧值)
- 末日期严格 = max(etf_inc.index.max(), fut_inc.index.max())

运行:
    python -m QuantNodes.strategy.allweather.data_etl.build_panel_v4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _splice_levels(hist: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    """增量段按列水平比例缩放至历史段末端水平 (消除拼接断层).

    历史段 (复权价) 与增量段 (原始价) 水平基准不同; 常数缩放保留增量收益,
    使拼接日收益连续 (避免 512690 -63.7% 类伪跳变).
    """
    out = inc.copy()
    for col in out.columns:
        if col not in hist.columns:
            continue
        h = hist[col].dropna()
        i = out[col].dropna()
        if len(h) and len(i):
            base = i.iloc[0]
            if np.isfinite(base) and abs(base) > 1e-12:
                out[col] = out[col] * (h.iloc[-1] / base)
    return out


def _parse_inc_window(inc_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    """从 etf_inc_{start}_{end}.parquet 文件名解析 [inc_start, inc_end]."""
    etf_files = sorted(inc_dir.glob("etf_inc_*.parquet"))
    if not etf_files:
        raise FileNotFoundError(f"找不到 ETF 增量 parquet: {inc_dir}/etf_inc_*.parquet")
    name = etf_files[-1].stem
    parts = name.split("_")
    start, end = pd.Timestamp(parts[2]), pd.Timestamp(parts[3])
    return start, end


def build_panel_v4(
    panel_v3_path: str = "data/allweather/panel_v3.parquet",
    inc_dir: str = "data/allweather/akshare_inc",
    panel_v4_path: str = "data/allweather/panel_v4.parquet",
) -> pd.DataFrame:
    """合并 panel_v3 + akshare 增量 → panel_v4 (无幽灵日期)."""
    inc_path = Path(inc_dir)
    inc_start, inc_end = _parse_inc_window(inc_path)
    print(f"增量窗口 (真实): {inc_start.date()} → {inc_end.date()}")

    panel_v3 = pd.read_parquet(panel_v3_path)
    print(f"panel_v3: {panel_v3.shape} ({panel_v3.index.min().date()} → "
          f"{panel_v3.index.max().date()})")

    etf_inc = pd.read_parquet(inc_path / f"etf_inc_{inc_start.date()}_{inc_end.date()}.parquet")
    fut_inc = pd.read_parquet(inc_path / f"futures_inc_{inc_start.date()}_{inc_end.date()}.parquet")
    print(f"akshare ETF 增量: {etf_inc.shape} ({etf_inc.index.min().date()} → "
          f"{etf_inc.index.max().date()})")
    print(f"akshare 期货增量: {fut_inc.shape} ({fut_inc.index.min().date()} → "
          f"{fut_inc.index.max().date()})")

    panel_v3_pre = panel_v3.loc[panel_v3.index < inc_start]
    print(f"\npanel_v3 增量前段: {panel_v3_pre.shape} ({panel_v3_pre.index.min().date()} → "
          f"{panel_v3_pre.index.max().date() if len(panel_v3_pre) > 0 else 'N/A'})")

    inc_all = pd.concat([etf_inc, fut_inc], axis=1).sort_index()
    print(f"合并后增量: {inc_all.shape}")

    # 拼接水平对齐 (§5.44 修复): 历史段为复权价, akshare 增量为未复权原始价,
    # 裸 concat 会产生伪跳变 (512690 -63.7%). 按列比例缩放增量段至历史末端水平
    # (常数缩放保留增量收益, 消除拼接断层).
    inc_all = _splice_levels(panel_v3_pre, inc_all)

    panel_v4 = pd.concat([panel_v3_pre, inc_all]).sort_index()
    panel_v4 = panel_v4[~panel_v4.index.duplicated(keep="last")]
    print(f"\npanel_v4: {panel_v4.shape} ({panel_v4.index.min().date()} → "
          f"{panel_v4.index.max().date()})")

    Path(panel_v4_path).parent.mkdir(parents=True, exist_ok=True)
    panel_v4.to_parquet(panel_v4_path)
    print(f"已保存: {panel_v4_path}")

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
