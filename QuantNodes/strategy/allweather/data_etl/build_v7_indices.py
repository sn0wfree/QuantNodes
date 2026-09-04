"""v7 指数增量合并 (§5.35 P1): v7_index_prices.parquet + akshare 增量 → v7_index_v2.

仅 4 个指数可补 (沪深300/中证500/中证1000/中证10年期国债).
未补指数 (恒生/中债3-5年/中债1-3年财富/国开行/企业债/南华工业品/南华农产品/
布伦特原油连续/沪金指数) 保留 v7_index_prices 原值.

运行:
    python -m QuantNodes.strategy.allweather.data_etl.build_v7_indices
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_v7_indices(
    v7_path: str = "data/high_freq_macro/v7_index_prices.parquet",
    inc_dir: str = "data/allweather/akshare_inc",
    v7_v2_path: str = "data/high_freq_macro/v7_index_v2.parquet",
) -> pd.DataFrame:
    """合并 v7_index_prices + akshare 4 列增量 → v7_index_v2."""
    v7 = pd.read_parquet(v7_path)
    print(f"v7_index_prices: {v7.shape} ({v7.index[0].date()} → {v7.index[-1].date()})")

    inc_files = sorted(Path(inc_dir).glob("v7_indices_inc_*.parquet"))
    if not inc_files:
        print(f"⚠ {inc_dir}/v7_indices_inc_*.parquet 不存在, 仅返回 v7 副本")
        v7.to_parquet(v7_v2_path)
        return v7

    inc = pd.read_parquet(inc_files[-1])
    print(f"akshare v7 增量: {inc.shape} ({inc.index[0].date()} → {inc.index[-1].date()})")

    v7_pre = v7.loc[v7.index < inc.index[0]]
    v7_v2 = pd.concat([v7_pre, inc]).sort_index()
    v7_v2 = v7_v2[~v7_v2.index.duplicated(keep="last")]
    print(f"\nv7_index_v2: {v7_v2.shape} ({v7_v2.index[0].date()} → {v7_v2.index[-1].date()})")

    Path(v7_v2_path).parent.mkdir(parents=True, exist_ok=True)
    v7_v2.to_parquet(v7_v2_path)
    print(f"已保存: {v7_v2_path}")

    for col in inc.columns:
        last_v7 = v7[col].dropna().index[-1] if v7[col].notna().any() else None
        last_v2 = v7_v2[col].dropna().index[-1] if v7_v2[col].notna().any() else None
        delta = (last_v2 - last_v7).days if last_v7 and last_v2 else None
        print(f"  {col}: {last_v7.date() if last_v7 else 'N/A'} → "
              f"{last_v2.date() if last_v2 else 'N/A'} (+{delta} 天)")

    return v7_v2


if __name__ == "__main__":
    build_v7_indices()
