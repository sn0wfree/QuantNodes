"""akshare 数据拉取 ETL (§5.34 panel_v4 重建).

拉取期货主力连续 (futures_main_sina) + ETF 净值 (stock_zh_index_daily),
标准化到 panel 格式.
不复用现有 DuckDB/NAV parquet — 独立增量拉取 → 写入新文件 → rebuild panel 时合并.

用法:
    python -m QuantNodes.strategy.allweather.data_etl.akshare_pull
    # 或:
    from data_etl.akshare_pull import pull_etf_panel, pull_futures_panel
"""
from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd

# ETF code 映射: etf_xxxxxx → stock_zh_index_daily 用的 sh/sz 前缀
ETF_CODE_MAP: dict[str, str] = {
    "etf_513100": "sh513100",
    "etf_513500": "sh513500",
    "etf_159915": "sz159915",
    "etf_510300": "sh510300",
    "etf_512890": "sh512890",
    "etf_512800": "sh512800",
    "etf_159740": "sz159740",
    # 不可用 code (stock_zh_index_daily 返回 'date' 单列):
    # etf_512010 (医药), etf_512690 (酒), etf_515030 (新能源车),
    # etf_515050 (5G), etf_512200 (房地产) — 这些走原 NAV parquet
}

# 期货代码映射: fut_xx → sina 主连代码 (futures_main_sina 的 symbol)
FUT_PRODUCT_MAP: dict[str, str] = {
    "fut_au": "AU0", "fut_ag": "AG0", "fut_cu": "CU0", "fut_al": "AL0",
    "fut_zn": "ZN0", "fut_ni": "NI0", "fut_rb": "RB0", "fut_hc": "HC0",
    "fut_i": "I0", "fut_j": "J0", "fut_jm": "JM0",
    "fut_ta": "TA0", "fut_ma": "MA0", "fut_pp": "PP0", "fut_l": "L0",
    "fut_bu": "BU0", "fut_sc": "SC0", "fut_m": "M0", "fut_y": "Y0",
    "fut_p": "P0", "fut_cf": "CF0", "fut_sr": "SR0", "fut_c": "C0",
    # 已 OK (t/tf/ts/if/ic/ih) 不需拉
}


def pull_etf_daily(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """单只 ETF 历史日线 (akshare stock_zh_index_daily).

    Returns:
        DataFrame with columns [date, open, high, low, close, volume]
    """
    df = ak.stock_zh_index_daily(symbol=symbol)
    if df is None or len(df) == 0:
        raise RuntimeError(f"akshare 返回空数据: {symbol}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[start_date:end_date]
    return df.reset_index()


def pull_etf_panel(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 7 个可用 ETF 的增量面板 (列名对齐 panel, 索引为 date)."""
    frames = {}
    for etf_col, sina_code in ETF_CODE_MAP.items():
        try:
            df = pull_etf_daily(sina_code, start_date, end_date)
            frames[etf_col] = df.set_index("date")["close"].rename(etf_col)
            print(f"  ✓ {etf_col} ({sina_code}): {len(df)} 行")
        except Exception as e:
            print(f"  ✗ {etf_col} ({sina_code}): {e}")
    return pd.DataFrame(frames).sort_index()


def pull_futures_panel(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 23 个商品的连续主力面板 (akshare futures_main_sina).

    连续主力接口 (data 已是主力切换后的), 无需 roll-adjust.
    Returns:
        DataFrame (T x 23) with columns close (其他列保留供审计)
    """
    frames = {}
    for fut_col, sina_code in FUT_PRODUCT_MAP.items():
        try:
            df = ak.futures_main_sina(symbol=sina_code)
            if df is None or len(df) == 0:
                print(f"  ✗ {fut_col} ({sina_code}): 空")
                continue
            # 列名标准化
            df = df.rename(columns={
                "日期": "date", "开盘价": "open", "最高价": "high",
                "最低价": "low", "收盘价": "close", "成交量": "volume",
                "持仓量": "open_interest",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.loc[start_date:end_date]
            if len(df) == 0:
                print(f"  ✗ {fut_col} ({sina_code}): 窗口内空")
                continue
            frames[fut_col] = df["close"].rename(fut_col)
            print(f"  ✓ {fut_col} ({sina_code}): {len(df)} 行")
        except Exception as e:
            print(f"  ✗ {fut_col} ({sina_code}): {type(e).__name__}: {e}")
    return pd.DataFrame(frames).sort_index()


if __name__ == "__main__":
    START = "2026-07-08"  # panel_v3 截止日
    END = "2026-09-03"     # 当前可达

    print("=" * 60)
    print(f"akshare 增量数据拉取 (§5.34 panel_v4, {START} → {END})")
    print("=" * 60)

    print("\n=== ETF (7 个可用 code) ===")
    etf_df = pull_etf_panel(START, END)
    if len(etf_df) > 0:
        print(f"\nETF 增量: {etf_df.shape}, 列: {list(etf_df.columns)}")

    print("\n=== 期货 (23 个商品) ===")
    fut_df = pull_futures_panel(START, END)
    if len(fut_df) > 0:
        print(f"\n期货增量: {fut_df.shape}, 列 (前10): {list(fut_df.columns)[:10]}")

    out_dir = Path("data/allweather/akshare_inc")
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(etf_df) > 0:
        etf_path = out_dir / f"etf_inc_{START}_{END}.parquet"
        etf_df.to_parquet(etf_path)
        print(f"\nETF 增量已保存: {etf_path}")
    if len(fut_df) > 0:
        fut_path = out_dir / f"futures_inc_{START}_{END}.parquet"
        fut_df.to_parquet(fut_path)
        print(f"期货增量已保存: {fut_path}")
