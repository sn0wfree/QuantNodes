# coding=utf-8
"""
clickhouse_data_loader.py - Stage 2 ClickHouse 数据加载器

从 ClickHouse 加载全 A 股日线数据，支持本地 parquet 缓存。
复用 database_node/clickhouse_node.py 的 CHBase 客户端。

ClickHouse 表 schema (quote.stock_quote):
    ts_code     LowCardinality(String)  → code
    trade_date  DateTime                → date (cast Date)
    open        Float64                 → open
    high        Float64                 → high
    low         Float64                 → low
    close       Float64                 → close
    vol         Float64                 → vol
    amount      Float64                 → amount

Stage 2 替代 MockDataLoader，接口完全兼容 DataLoader ABC。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from .contracts import DataLoader

logger = logging.getLogger(__name__)

__all__ = ["ClickHouseDataLoader"]


class ClickHouseDataLoader(DataLoader):
    """Stage 2 ClickHouse 数据加载器

    从 ClickHouse 加载全 A 股日线数据。
    支持本地 parquet 缓存（首次查询后自动缓存）。

    用法::

        loader = ClickHouseDataLoader(
            table="quote.stock_quote",
            start_date="2019-01-01",
            end_date="2024-12-31",
        )
        df = loader.load()  # polars.DataFrame
    """

    # ClickHouse → polars 字段映射
    FIELD_MAP = {
        "ts_code": "code",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "vol": "vol",
        "amount": "amount",
    }

    def __init__(
        self,
        table: str = "quote.stock_quote",
        host: str = "localhost",
        port: int = 8123,
        user: str = "data",
        password: str = "123456",
        database: str = "quote",
        start_date: str = "2019-01-01",
        end_date: str = "2024-12-31",
        min_amount_percentile: float = 0.0,
        cache_parquet: Optional[str] = "data/cache/full_a_2019_2024.parquet",
    ) -> None:
        self.table = table
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.start_date = start_date
        self.end_date = end_date
        self.min_amount_percentile = min_amount_percentile
        self.cache_parquet = Path(cache_parquet) if cache_parquet else None

    def load(self) -> pl.DataFrame:
        """加载数据：优先读缓存 parquet，否则从 ClickHouse 查询并缓存。"""
        if self.cache_parquet and self.cache_parquet.exists():
            logger.info("[ClickHouseDataLoader] 读取缓存: %s", self.cache_parquet)
            df = pl.read_parquet(self.cache_parquet)
            logger.info("[ClickHouseDataLoader] 缓存加载完成: %s rows", df.height)
            return df

        logger.info(
            "[ClickHouseDataLoader] 从 ClickHouse 查询: %s (%s ~ %s)",
            self.table, self.start_date, self.end_date,
        )
        df = self._query_clickhouse()
        df = self._clean(df)

        if self.cache_parquet:
            self.cache_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(self.cache_parquet)
            logger.info(
                "[ClickHouseDataLoader] 缓存已保存: %s (%s rows)",
                self.cache_parquet, df.height,
            )

        return df

    def _query_clickhouse(self) -> pl.DataFrame:
        """从 ClickHouse 查询数据，返回 polars DataFrame。"""
        import http.client
        import json

        fields = ", ".join(
            f"{ch_name} AS {pl_name}" for ch_name, pl_name in self.FIELD_MAP.items()
        )
        sql = (
            f"SELECT {fields}, CAST(trade_date AS Date) AS date "
            f"FROM {self.table} "
            f"WHERE trade_date >= '{self.start_date}' "
            f"AND trade_date <= '{self.end_date}' "
            f"ORDER BY date, code"
        )

        logger.info("[ClickHouseDataLoader] SQL: %s", sql[:200])

        conn = http.client.HTTPConnection(self.host, port=self.port)
        auth_params = f"?user={self.user}&password={self.password}"

        try:
            conn.request(
                "POST",
                "/" + auth_params,
                body=sql + " FORMAT JSONEachRow",
                headers={"Content-Type": "text/plain"},
            )
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")

            if resp.status != 200:
                raise RuntimeError(f"ClickHouse query failed: {resp.status} {raw[:200]}")

            rows = [json.loads(line) for line in raw.strip().split("\n") if line]
            if not rows:
                raise RuntimeError("ClickHouse returned empty result")

            df = pl.DataFrame(rows)
            logger.info("[ClickHouseDataLoader] 查询完成: %s rows", df.height)
            return df

        finally:
            conn.close()

    def _clean(self, df: pl.DataFrame) -> pl.DataFrame:
        """数据清洗：类型转换 + 过滤。"""
        # 确保 date 列为 Date 类型
        if df["date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("date").str.to_date())
        elif df["date"].dtype == pl.Datetime:
            df = df.with_columns(pl.col("date").cast(pl.Date))

        # 确保数值列为 Float64
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

        # 过滤停牌（vol == 0）
        before = df.height
        df = df.filter(pl.col("vol") > 0)
        if df.height < before:
            logger.info(
                "[ClickHouseDataLoader] 过滤停牌: %d → %d rows",
                before, df.height,
            )

        # 过滤低流动性（可选）
        if self.min_amount_percentile > 0:
            threshold = df["amount"].quantile(self.min_amount_percentile)
            before = df.height
            df = df.filter(pl.col("amount") >= threshold)
            logger.info(
                "[ClickHouseDataLoader] 过滤低流动性 (< %.0f): %d → %d rows",
                threshold, before, df.height,
            )

        return df

    def load_summary(self) -> dict:
        """返回数据摘要（不加载全量数据）。"""
        import http.client
        import json

        sql = (
            f"SELECT min(trade_date) as min_date, max(trade_date) as max_date, "
            f"count() as total_rows, count(distinct ts_code) as n_stocks "
            f"FROM {self.table} "
            f"WHERE trade_date >= '{self.start_date}' "
            f"AND trade_date <= '{self.end_date}'"
        )

        conn = http.client.HTTPConnection(self.host, port=self.port)
        auth_params = f"?user={self.user}&password={self.password}"

        try:
            conn.request("POST", "/" + auth_params, body=sql + " FORMAT JSONEachRow",
                         headers={"Content-Type": "text/plain"})
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            if resp.status == 200 and raw.strip():
                return json.loads(raw.strip().split("\n")[0])
        finally:
            conn.close()

        return {"error": "query failed"}
