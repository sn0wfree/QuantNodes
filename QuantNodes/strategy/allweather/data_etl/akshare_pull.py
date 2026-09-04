"""akshare 数据拉取 ETL (§5.34 panel_v4 + §5.35 P1 完整覆盖).

拉取 4 类数据:
1. ETF 12 个 (stock_zh_index_daily) — 7 已有 + 5 补全
2. 期货 29 个 (futures_main_sina) — 23 已有 + 6 股指期货
3. v7 指数 5 个 (stock_zh_index_daily) — 沪深300/中证500/中证1000/上证50/中证10年期国债

v7 指数单独保存 (与 v7_index_prices.parquet 合并), 不进入 panel_v4 (panel 没有 v7 列).
中证红利/申万行业 已停止更新 (返回 0 行), 跳过.
恒生指数 hkHSI 接口格式不同, 暂不补.

运行:
    python -m QuantNodes.strategy.allweather.data_etl.akshare_pull
"""
from __future__ import annotations

from pathlib import Path

import akshare as ak
import pandas as pd

ETF_CODE_MAP: dict[str, str] = {
    # 已有 (P0)
    "etf_513100": "sh513100",   # 纳指ETF
    "etf_513500": "sh513500",   # 标普500
    "etf_159915": "sz159915",   # 创业板
    "etf_510300": "sh510300",   # 沪深300
    "etf_512890": "sh512890",   # 红利低波
    "etf_512800": "sh512800",   # 银行
    "etf_159740": "sz159740",   # 大数据
    # P1 补全
    "etf_512010": "sh512010",   # 医药
    "etf_512690": "sh512690",   # 酒
    "etf_515030": "sh515030",   # 新能源车
    "etf_515050": "sh515050",   # 5G
    "etf_512200": "sh512200",   # 房地产
}

FUT_PRODUCT_MAP: dict[str, str] = {
    # 已有 (P0)
    "fut_au": "AU0", "fut_ag": "AG0", "fut_cu": "CU0", "fut_al": "AL0",
    "fut_zn": "ZN0", "fut_ni": "NI0", "fut_rb": "RB0", "fut_hc": "HC0",
    "fut_i": "I0", "fut_j": "J0", "fut_jm": "JM0",
    "fut_ta": "TA0", "fut_ma": "MA0", "fut_pp": "PP0", "fut_l": "L0",
    "fut_bu": "BU0", "fut_sc": "SC0", "fut_m": "M0", "fut_y": "Y0",
    "fut_p": "P0", "fut_cf": "CF0", "fut_sr": "SR0", "fut_c": "C0",
    # P1 补全 (股指 + 国债期货)
    "fut_ic": "IC0",   # 中证500期货
    "fut_if": "IF0",   # 沪深300期货
    "fut_ih": "IH0",   # 上证50期货
    "fut_t": "T0",     # 10年国债期货
    "fut_tf": "TF0",   # 5年国债期货
    "fut_ts": "TS0",   # 2年国债期货
}

# v7 指数补全 (与 v7_index_prices.parquet 列名对齐)
# panel 不含这些列, 单独保存用于实验 (csi_1000 / cash_proxy)
V7_INDEX_MAP: dict[str, str] = {
    "沪深300指数": "sh000300",
    "中证500指数": "sh000905",
    "中证1000": "sh000852",
    "中债10年期国债指数": "sh000012",
}
# v7 中以下列 akshare 无对应或死指数, 跳过:
#   恒生指数 (hkHSI 接口格式不同), 中债3-5年期/1-3年财富/国开行/企业债 (无 akshare 直接对应),
#   南华工业品/农产品指数 (无对应), 布伦特原油连续, 沪金指数 (akshare 无指数版)


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
    """拉取 12 个 ETF 增量面板 (列名对齐 panel, DatetimeIndex)."""
    series = []
    for etf_col, sina_code in ETF_CODE_MAP.items():
        try:
            df = pull_etf_daily(sina_code, start_date, end_date)
            s = df.set_index("date")["close"].rename(etf_col)
            series.append(s)
            print(f"  ✓ {etf_col} ({sina_code}): {len(df)} 行")
        except Exception as e:
            print(f"  ✗ {etf_col} ({sina_code}): {e}")
    return pd.concat(series, axis=1).sort_index()


def pull_futures_panel(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 29 个期货增量面板 (akshare futures_main_sina 连续主力).

    连续主力接口, 无需 roll-adjust.
    Returns:
        DataFrame (T x N) with columns close.
    """
    series = []
    for fut_col, sina_code in FUT_PRODUCT_MAP.items():
        try:
            df = ak.futures_main_sina(symbol=sina_code)
            if df is None or len(df) == 0:
                print(f"  ✗ {fut_col} ({sina_code}): 空")
                continue
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
            series.append(df["close"].rename(fut_col))
            print(f"  ✓ {fut_col} ({sina_code}): {len(df)} 行")
        except Exception as e:
            print(f"  ✗ {fut_col} ({sina_code}): {type(e).__name__}: {e}")
    return pd.concat(series, axis=1).sort_index()


def pull_v7_indices(start_date: str, end_date: str) -> pd.DataFrame:
    """拉取 v7 指数 5 列增量 (与 v7_index_prices.parquet 列名对齐).

    保存为单独 parquet (panel_v4 不含 v7 列), 用于实验脚本注入 csi_1000/cash_proxy 等.
    """
    series = []
    for v7_col, sina_code in V7_INDEX_MAP.items():
        try:
            df = ak.stock_zh_index_daily(symbol=sina_code)
            if df is None or len(df) == 0:
                print(f"  ✗ {v7_col} ({sina_code}): 空")
                continue
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            df = df.loc[start_date:end_date]
            if len(df) == 0:
                print(f"  ✗ {v7_col} ({sina_code}): 窗口内空")
                continue
            series.append(df["close"].rename(v7_col))
            print(f"  ✓ {v7_col} ({sina_code}): {len(df)} 行")
        except Exception as e:
            print(f"  ✗ {v7_col} ({sina_code}): {type(e).__name__}: {e}")
    return pd.concat(series, axis=1).sort_index()


if __name__ == "__main__":
    START = "2026-07-08"
    END = "2026-09-03"

    print("=" * 60)
    print(f"akshare 增量数据拉取 (§5.34 + §5.35 P1, {START} → {END})")
    print("=" * 60)

    out_dir = Path("data/allweather/akshare_inc")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== ETF (12 个: 7 已有 + 5 补全) ===")
    etf_df = pull_etf_panel(START, END)
    if len(etf_df) > 0:
        etf_path = out_dir / f"etf_inc_{START}_{END}.parquet"
        etf_df.to_parquet(etf_path)
        print(f"\nETF 增量已保存: {etf_path} ({etf_df.shape})")

    print("\n=== 期货 (29 个: 23 商品 + 6 股指/国债) ===")
    fut_df = pull_futures_panel(START, END)
    if len(fut_df) > 0:
        fut_path = out_dir / f"futures_inc_{START}_{END}.parquet"
        fut_df.to_parquet(fut_path)
        print(f"\n期货增量已保存: {fut_path} ({fut_df.shape})")

    print("\n=== v7 指数 (5 个: 沪深300/中证500/中证1000/上证50/中债10年) ===")
    v7_df = pull_v7_indices(START, END)
    if len(v7_df) > 0:
        v7_path = out_dir / f"v7_indices_inc_{START}_{END}.parquet"
        v7_df.to_parquet(v7_path)
        print(f"\nv7 指数增量已保存: {v7_path} ({v7_df.shape})")
