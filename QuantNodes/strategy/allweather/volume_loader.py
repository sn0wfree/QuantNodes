"""量价面板加载器 (D.1, PLAN §17.12): 期货 volume+OI / ETF volume.

数据源:
- 期货: DuckDB continuous_main_{p}_daily (volume, open_interest), 与 panel 同源
- ETF: data/real/ohlcv/{code}.parquet (volume 列)

输出: (T×N) volume 面板 + (T×N_fut) OI 面板, 对齐 panel_v2 索引.
缓存: data/allweather/volume_oi_cache.parquet (gitignore).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import DUCKDB_PATH


def _prod_from_col(col: str) -> str:
    return col.removeprefix("fut_").upper()


def _code_from_col(col: str) -> str:
    return col.removeprefix("etf_")


def load_volume_oi_panel(
    panel: pd.DataFrame,
    duckdb_path: Path = DUCKDB_PATH,
    start_date: str = "2017-01-01",
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载量价面板, 对齐 panel 索引.

    Returns:
        (volume, oi): 各 (T×N) 面板; ETF 的 OI 列为 NaN (无持仓量).
        volume: 成交量 (手); oi: 持仓量 (仅期货).
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = pd.read_parquet(cache_path)
        vol = cached[[c for c in cached.columns if c.startswith("vol_")]]
        vol.columns = [c.replace("vol_", "") for c in vol.columns]
        oi = cached[[c for c in cached.columns if c.startswith("oi_")]]
        oi.columns = [c.replace("oi_", "") for c in oi.columns]
        return vol.reindex(panel.index), oi.reindex(panel.index)

    fut_cols = [c for c in panel.columns if c.startswith("fut_")]
    etf_cols = [c for c in panel.columns if c.startswith("etf_")]

    vol_frames: dict[str, pd.Series] = {}
    oi_frames: dict[str, pd.Series] = {}

    # ── 期货: DuckDB ──
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        for col in fut_cols:
            p = _prod_from_col(col)
            table = f"continuous_main_{p.lower()}_daily"
            try:
                df = con.execute(
                    f"SELECT trade_date, volume, open_interest FROM {table} "
                    f"WHERE trade_date >= '{start_date}' ORDER BY trade_date"
                ).df()
            except Exception:
                df = pd.DataFrame()
            if df.empty:
                vol_frames[col] = pd.Series(dtype=float)
                oi_frames[col] = pd.Series(dtype=float)
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df = df.set_index("trade_date")
            vol_frames[col] = df["volume"].astype(float).rename(col)
            oi_frames[col] = df["open_interest"].astype(float).rename(col)
    finally:
        con.close()

    # ── ETF: ohlcv parquet ──
    ohlcv_dir = Path("data/real/ohlcv")
    for col in etf_cols:
        code = _code_from_col(col)
        f = ohlcv_dir / f"{code}.parquet"
        if not f.exists():
            vol_frames[col] = pd.Series(dtype=float)
            continue
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        if "volume" in df.columns:
            vol_frames[col] = df["volume"].astype(float).rename(col)
        else:
            vol_frames[col] = pd.Series(dtype=float)

    volume = pd.DataFrame(vol_frames).sort_index()
    oi = pd.DataFrame(oi_frames).sort_index()

    # 对齐 panel 索引 (ffill 量价数据缺失日, 前向填充因果)
    volume = volume.reindex(panel.index).ffill()
    oi = oi.reindex(panel.index).ffill()

    if cache_path is not None:
        cache = pd.concat(
            [volume.add_prefix("vol_"), oi.add_prefix("oi_")], axis=1)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path)

    return volume, oi
