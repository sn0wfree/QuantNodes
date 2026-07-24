# coding=utf-8
"""合约参数表 — contract_specs 编程访问.

DuckDB 中 contract_specs 表结构:
    exchange            LowCardinality(String)   # CFFEX / SHFE / DCE / CZCE / INE / GFEX
    product             LowCardinality(String)   # IF / IC / AU / RB / ...
    contract_multiplier Float64                  # 元/手 (IF=300, T=10000, AU=1000)
    price_unit          LowCardinality(String)   # 元/吨 / 元/手 / 元/克
    tick_size           Float64                  # 最小变动价位 (IF=0.2, AU=0.02)

公共 API:
    load_contract_specs()           → pd.DataFrame (91 行, 含全部字段)
    contract_spec_dataframe(...)    与上等价, 允许指定路径
    contract_multiplier(product)    → float         # 快速查"IF" → 300.0
    tick_size(product)              → float         # 快速查"AU" → 0.02
    ContractSpec                    dataclass, 单产品合约元数据
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .data_loader import DEFAULT_DUCKDB_PATH


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """单产品合约参数 (取自 DuckDB contract_specs)."""

    exchange: str
    product: str
    contract_multiplier: float
    price_unit: str
    tick_size: float

    @property
    def key(self) -> str:
        """'IF' (取 product, 因为同一 product 多交易所不冲突)."""
        return self.product

    @property
    def exchange_key(self) -> str:
        """'CFFEX.IF' (exchange.product, 跨所时有歧义时用这个)."""
        return f"{self.exchange}.{self.product}"


@lru_cache(maxsize=4)
def _cached_specs(path: str) -> tuple[ContractSpec, ...]:
    """内部 LRU 缓存, 同一路径只读一次 DuckDB."""
    import duckdb

    con = duckdb.connect(path, read_only=True)
    try:
        rows = con.execute(
            "SELECT exchange, product, contract_multiplier, price_unit, tick_size "
            "FROM contract_specs ORDER BY exchange, product"
        ).fetchall()
    finally:
        con.close()
    return tuple(
        ContractSpec(
            exchange=r[0],
            product=r[1],
            contract_multiplier=float(r[2]) if r[2] is not None else 0.0,
            price_unit=str(r[3]) if r[3] is not None else "",
            tick_size=float(r[4]) if r[4] is not None else 0.0,
        )
        for r in rows
    )


def _to_path(duckdb_path: Path | str | None) -> Path:
    """规范化 DuckDB 路径."""
    if duckdb_path is None:
        return DEFAULT_DUCKDB_PATH
    return Path(duckdb_path)


def contract_spec_dataframe(duckdb_path: Path | str | None = None) -> pd.DataFrame:
    """读 contract_specs 全表为 DataFrame.

    Returns:
        pd.DataFrame, 91 行, 列:
        exchange / product / contract_multiplier / price_unit / tick_size
    """
    import duckdb

    con = duckdb.connect(str(_to_path(duckdb_path)), read_only=True)
    try:
        return con.execute(
            "SELECT exchange, product, contract_multiplier, price_unit, tick_size "
            "FROM contract_specs ORDER BY exchange, product"
        ).df()
    finally:
        con.close()


def load_contract_specs(
    duckdb_path: Path | str | None = None,
) -> tuple[ContractSpec, ...]:
    """读 contract_specs 为 ContractSpec 元组.

    Returns:
        不可变 tuple[ContractSpec, ...], 长度 ≈ 91
    """
    return _cached_specs(str(_to_path(duckdb_path)))


def _spec_for(product: str, duckdb_path: Path | str | None) -> ContractSpec:
    """查单个产品的 ContractSpec, 不存在则 KeyError."""
    for s in load_contract_specs(duckdb_path):
        if s.product == product:
            return s
    raise KeyError(
        f"product '{product}' not in contract_specs; "
        f"available products (first 10): "
        f"{[s.product for s in load_contract_specs(duckdb_path)[:10]]}"
    )


def contract_multiplier(product: str, duckdb_path: Path | str | None = None) -> float:
    """查单个产品的合约乘数 (元/手).

    Examples:
        >>> contract_multiplier("IF")    # 300.0
        >>> contract_multiplier("AU")    # 1000.0
        >>> contract_multiplier("T")     # 10000.0
    """
    return _spec_for(product, duckdb_path).contract_multiplier


def tick_size(product: str, duckdb_path: Path | str | None = None) -> float:
    """查单个产品的最小变动价位.

    Examples:
        >>> tick_size("IF")  # 0.2
        >>> tick_size("AU")  # 0.02
        >>> tick_size("T")   # 0.005
    """
    return _spec_for(product, duckdb_path).tick_size
