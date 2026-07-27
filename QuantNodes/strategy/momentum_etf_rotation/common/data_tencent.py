# coding=utf-8
"""Tencent 行情 API 拉取 (从 scripts/fetch_real_etf_panel.py 提取).

API endpoint:
    https://web.ifzq.gtimg.cn/appstock/app/fqkline/get

请求格式:
    param=<market>,<code>,day,<start>,<end>,320,qfq

速率: 7 req/s 实测, 加 150ms sleep 留余量.

主要函数:
    fetch_one_etf_tencent() → 拉取单只 ETF 日线, 返回 pd.Series
    write_fetch_log()        → 写 fetch_log.json

设计原则:
    - 与 scripts/fetch_real_etf_panel.py 完全一致 (确保兼容性)
    - 可独立运行 (不依赖包安装)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
REQUEST_TIMEOUT = 15


def _market_of(code: str) -> str:
    """根据 ETF code 前缀判断市场.
    ETF 代码约定:
      5xxxxx → sh (上海)
      1xxxxx → sz (深圳)
    """
    if code.startswith("5"):
        return "sh"
    elif code.startswith("1"):
        return "sz"
    else:
        raise ValueError(f"无法识别 ETF code 市场: {code}")


def _http_get(url: str, params: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """GET 请求, 失败抛 IOError."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise IOError(f"Tencent API request failed: {e}") from e


def fetch_one_etf_tencent(
    code: str,
    start: str = "2018-01-01",
    end: str = "2025-07-06",
    sleep_ms: int = 150,
) -> pd.Series:
    """从 Tencent 行情 API 拉取单只 ETF 日线 (前复权).

    Args:
        code:    ETF code (如 "510300" / "159915")
        start:   开始日期 YYYY-MM-DD
        end:     结束日期 YYYY-MM-DD
        sleep_ms: 速率限制 (默认 150ms, 实测 7 req/s)

    Returns:
        pd.Series: index=DatetimeIndex (DatetimeIndex), values=close prices
        空 Series 表示该 code 无数据

    Notes:
        - 如果 API 返回空, 返回空 Series (不抛错)
        - 失败时记录日志, 调用方可继续
    """
    market = _market_of(code)
    params = {"param": f"{market},{code},day,{start},{end},320,qfq"}

    try:
        data = _http_get(API_URL, params)
    except IOError as e:
        logger.warning("[%s] API request 失败: %s", code, e)
        return pd.Series(dtype=float, name=code)

    try:
        stock_info = data["data"][market][code]
    except (KeyError, TypeError):
        logger.info("[%s] 无数据 (可能 %s 后上市)", code, start)
        return pd.Series(dtype=float, name=code)

    raw_klines = stock_info.get("qfqday") or stock_info.get("day") or []
    if not raw_klines:
        logger.info("[%s] 无 K 线数据", code)
        return pd.Series(dtype=float, name=code)

    records = []
    for row in raw_klines:
        if len(row) < 6:
            continue
        date_str, _open, _close, _high, _low, _vol = row[:6]
        try:
            records.append({
                "date": pd.to_datetime(date_str),
                "close": float(_close),
            })
        except (ValueError, TypeError):
            continue

    if not records:
        return pd.Series(dtype=float, name=code)

    df = pd.DataFrame(records).set_index("date").sort_index()
    series = df["close"].rename(code)
    series.index.name = None

    # 速率限制
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    return series


def write_fetch_log(
    fetched: dict[str, int],
    failed: list[str],
    log_path: Path | str,
) -> None:
    """写 fetch_log.json (与 data/real/fetch_log.json schema 一致).

    Args:
        fetched: {code: row_count} 成功拉取的 ETF 及其行数
        failed: 失败/缺失的 code 列表
        log_path: 输出 JSON 路径

    Schema:
        {
            "fetched": {"510300": 2058, ...},
            "failed": [],
            "fetched_count": N,
            "failed_count": M
        }
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched": dict(fetched),
        "failed": list(failed),
        "fetched_count": len(fetched),
        "failed_count": len(failed),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Fetch log written: %s (fetched=%d, failed=%d)",
                log_path, len(fetched), len(failed))


__all__ = [
    "fetch_one_etf_tencent",
    "write_fetch_log",
    "API_URL",
]
