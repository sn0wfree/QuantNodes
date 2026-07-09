# coding=utf-8
"""Sina 行情 API 拉取 OHLCV 数据 (Tencent 失效时备用).

API endpoint:
    https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData

请求格式:
    ?symbol=sh510300 & scale=240 (daily) & ma=no & datalen=N

返回: JSON 数组, 每元素:
    {"day": "2024-01-02", "open": "3.478", "high": "3.500", "low": "3.450",
     "close": "3.490", "volume": 12345678}

速率: ~10 req/s 实测, 加 100ms sleep 留余量.

主要函数:
    fetch_one_etf_sina() → 拉取单只 ETF 日线 OHLCV, 返回 pd.DataFrame
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
REQUEST_TIMEOUT = 15


def _market_of(code: str) -> str:
    """根据 ETF code 前缀判断市场 (新浪格式: sh510300 / sz159915)."""
    if code.startswith("5"):
        return "sh"
    elif code.startswith("1"):
        return "sz"
    else:
        raise ValueError(f"无法识别 ETF code 市场: {code}")


def _http_get(url: str, params: dict, timeout: int = REQUEST_TIMEOUT) -> list:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise IOError(f"Sina API request failed: {e}") from e


def fetch_one_etf_sina(
    code: str,
    datalen: int = 2200,
    scale: int = 240,
    sleep_ms: int = 100,
) -> pd.DataFrame:
    """从 Sina 行情 API 拉取单只 ETF 日线 OHLCV (前复权).

    Args:
        code: ETF code (如 "510300" / "159915")
        datalen: 返回数据条数 (默认 2200, 覆盖 8 年)
        scale: 周期 (240=daily, 60=hourly, 1=1min)
        sleep_ms: 速率限制

    Returns:
        pd.DataFrame: index=date, columns=[open, high, low, close, volume]
    """
    market = _market_of(code)
    symbol = f"{market}{code}"
    params = {
        "symbol": symbol,
        "scale": scale,
        "ma": "no",
        "datalen": datalen,
    }

    try:
        raw = _http_get(API_URL, params)
    except IOError as e:
        logger.warning("[%s] API request 失败: %s", code, e)
        return pd.DataFrame()

    if not raw or not isinstance(raw, list):
        logger.info("[%s] 无数据", code)
        return pd.DataFrame()

    records = []
    for row in raw:
        try:
            records.append({
                "date": pd.to_datetime(row["day"]),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        except (ValueError, TypeError, KeyError) as e:
            logger.debug("[%s] 跳过一行: %s", code, e)
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date").sort_index()
    df.index.name = None

    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    return df


def fetch_panel_sina(
    codes: list[str],
    datalen: int = 2200,
    out_dir: str | Path = "data/real/ohlcv",
) -> dict[str, pd.DataFrame]:
    """批量拉取多只 ETF OHLCV.

    Returns:
        dict, code → DataFrame
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []
    for code in codes:
        cache = out_dir / f"{code}.parquet"
        if cache.exists():
            try:
                df = pd.read_parquet(cache)
                if len(df) >= 200:
                    results[code] = df
                    continue
            except Exception:
                pass

        df = fetch_one_etf_sina(code, datalen=datalen)
        if len(df) > 0:
            df.to_parquet(cache)
            results[code] = df
        else:
            failed.append(code)
        logger.info("[%s] %d 行 (失败 %d 个)", code, len(df), len(failed))

    logger.info("成功 %d/%d, 失败: %s", len(results), len(codes), failed)
    return results


def build_ohlcv_panel(
    codes: list[str],
    out_path: str | Path = "data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet",
) -> pd.DataFrame:
    """构建 OHLCV 面板 parquet (多级 columns: code × {open,high,low,close,volume})."""
    raw = fetch_panel_sina(codes)
    if not raw:
        return pd.DataFrame()

    all_data = {}
    for code, df in raw.items():
        all_data[code] = df

    panel = pd.concat(all_data, axis=1)
    panel.columns.names = ["code", "field"]
    panel = panel.sort_index(axis=1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path)
    logger.info("保存到 %s, shape=%s", out_path, panel.shape)
    return panel


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    codes_path = Path("data/real/fetch_log.json")
    if not codes_path.exists():
        print(f"无法找到 {codes_path}")
        sys.exit(1)
    with open(codes_path) as f:
        log = json.load(f)
    codes = list(log.get("fetched", {}).keys())
    print(f"将拉取 {len(codes)} 只 ETF OHLCV")

    panel = build_ohlcv_panel(codes)
    print(f"面板 shape: {panel.shape}")
    print(f"前 2 行: \n{panel.iloc[:2]}")
