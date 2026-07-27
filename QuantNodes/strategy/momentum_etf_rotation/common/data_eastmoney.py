# coding=utf-8
"""Eastmoney 行情 API 拉取 (Smart β ETF 用).

API endpoint:
    https://push2his.eastmoney.com/api/qt/stock/kline/get

secid 格式:
    1.510880  → sh510880 (沪市)
    0.159915  → sz159915 (深市)

速率: 1 req/ETF, 无严格限速, 加 100ms 留余量即可.

主要函数:
    fetch_one_etf_eastmoney() → 拉取单只 ETF 日线, 返回 pd.Series
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
REQUEST_TIMEOUT = 30


def _market_of(code: str) -> int:
    """根据 ETF code 前缀判断市场: 1=沪, 0=深."""
    if code.startswith("5"):
        return 1
    elif code.startswith("1"):
        return 0
    else:
        raise ValueError(f"无法识别 ETF code 市场: {code}")


def _http_get(url: str, params: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """GET 请求, 失败抛 IOError."""
    try:
        r = requests.get(
            url, params=params, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise IOError(f"Eastmoney API request failed: {e}") from e


def fetch_one_etf_eastmoney(
    code: str,
    start: str = "2018-01-01",
    end: str = "2026-06-30",
    sleep_ms: int = 100,
) -> pd.Series:
    """从 Eastmoney 行情 API 拉取单只 ETF 日线 (前复权).

    Args:
        code:    ETF code (如 "510300" / "159915")
        start:   开始日期 YYYY-MM-DD (含)
        end:     结束日期 YYYY-MM-DD (含)
        sleep_ms: 速率限制 (默认 100ms)

    Returns:
        pd.Series: index=DatetimeIndex, values=close prices (前复权)
        空 Series 表示该 code 无数据
    """
    market = _market_of(code)
    secid = f"{market}.{code}"
    params = {
        "secid": secid,
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "1",
        "beg": "0",
        "end": "20991231",
        "lmt": "10000",
    }

    try:
        data = _http_get(API_URL, params)
    except IOError as e:
        logger.warning("[%s] API request 失败: %s", code, e)
        return pd.Series(dtype=float, name=code)

    if data.get("rc") != 0:
        logger.warning("[%s] API rc=%s, msg=%s", code, data.get("rc"), data.get("rt"))
        return pd.Series(dtype=float, name=code)

    klines = data.get("data", {}).get("klines", [])
    if not klines:
        logger.info("[%s] 无 K 线数据", code)
        return pd.Series(dtype=float, name=code)

    # 解析: 日期, 开, 收, 高, 低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
    records = []
    for row in klines:
        parts = row.split(",")
        if len(parts) < 6:
            continue
        try:
            records.append({
                "date": pd.to_datetime(parts[0]),
                "close": float(parts[2]),
            })
        except (ValueError, TypeError):
            continue

    if not records:
        return pd.Series(dtype=float, name=code)

    df = pd.DataFrame(records).set_index("date").sort_index()
    series = df["close"].rename(code)
    series.index.name = None

    # 日期过滤
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    series = series.loc[(series.index >= start_dt) & (series.index <= end_dt)]

    # 速率限制
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    return series


if __name__ == "__main__":
    # 简单测试
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--code", default="510880")
    args = p.parse_args()
    s = fetch_one_etf_eastmoney(args.code, "2018-01-01", "2026-06-30", 0)
    print(f"{args.code}: {len(s)} rows, first={s.index[0].date() if len(s) else 'N/A'}, last={s.index[-1].date() if len(s) else 'N/A'}")
    if len(s) > 0:
        print(f"  price: {s.iloc[0]:.3f} -> {s.iloc[-1]:.3f} ({(s.iloc[-1]/s.iloc[0]-1)*100:.1f}%)")
