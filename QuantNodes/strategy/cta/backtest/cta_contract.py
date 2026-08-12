# coding=utf-8
"""期货/期权合约规格.

包装 contract_specs (期货乘数/tick) 与 futures_rule_snapshot (保证金比例),
为 CTA 回测提供合约级别参数:
    - contract_multiplier: 乘数 (元/手)
    - margin_ratio: 保证金比例 (decimal)
    - tick_size: 最小变动价位
    - price_unit: 计价单位
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

DEFAULT_DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()


@dataclass(frozen=True, slots=True)
class CtaContractSpec:
    """单个期货产品的合约参数 (CTA 回测用)."""

    exchange: str
    product: str
    contract_multiplier: float   # 元/手
    price_unit: str
    tick_size: float
    margin_ratio: float = 0.1    # 默认 10%, 可由规则快照覆盖


@dataclass(frozen=True, slots=True)
class CtaContract:
    """单张期货合约 (含到期月份, 用于换月识别)."""

    product: str
    symbol: str          # 'IF2608' / 'au2602'
    expiry_year: int
    expiry_month: int
    spec: CtaContractSpec

    @property
    def key(self) -> str:
        return self.symbol


@lru_cache(maxsize=8)
def _load_specs_cached(path: str) -> tuple[tuple, ...]:
    """缓存读取 contract_specs (DuckDB)."""
    import duckdb
    con = duckdb.connect(path, read_only=True)
    try:
        rows = con.execute(
            "SELECT exchange, product, contract_multiplier, price_unit, tick_size "
            "FROM contract_specs ORDER BY exchange, product"
        ).fetchall()
    finally:
        con.close()
    return tuple(rows)


@lru_cache(maxsize=8)
def _load_margin_cached(path: str) -> dict[str, float]:
    """缓存读取最新保证金快照 {产品代码: margin_ratio}.

    注意: futures_rule_snapshot 的 product 列是中文名, code 列才是产品代码
    (如 'AU' / 'IF' / 'M'), 故用 code 列映射.
    """
    import duckdb
    con = duckdb.connect(path, read_only=True)
    try:
        rows = con.execute(
            "SELECT code, margin_ratio FROM futures_rule_snapshot "
            "WHERE as_of_date = (SELECT MAX(as_of_date) FROM futures_rule_snapshot) "
            "AND code IS NOT NULL"
        ).fetchall()
    except Exception:
        rows = []
    finally:
        con.close()
    return {r[0]: float(r[1]) / 100.0 for r in rows if r[1] is not None}


class CtaContractRegistry:
    """合约规格注册表: 从 DuckDB 加载 product → CtaContractSpec."""

    def __init__(self, duckdb_path: Path | str | None = None):
        self._path = str(Path(duckdb_path or DEFAULT_DUCKDB_PATH))
        self._specs: dict[str, CtaContractSpec] = {}
        self._margin: dict[str, float] = {}

        for (exchange, product, mult, unit, tick) in _load_specs_cached(self._path):
            self._specs[product] = CtaContractSpec(
                exchange=exchange,
                product=product,
                contract_multiplier=float(mult) if mult is not None else 0.0,
                price_unit=str(unit) if unit is not None else "",
                tick_size=float(tick) if tick is not None else 0.0,
            )
        self._margin = _load_margin_cached(self._path)
        # 应用保证金覆盖
        for product, spec in self._specs.items():
            if product in self._margin and self._margin[product] > 0:
                object.__setattr__(spec, "margin_ratio", self._margin[product])

    def get(self, product: str) -> Optional[CtaContractSpec]:
        return self._specs.get(product.upper())

    def get_or_default(self, product: str, default_margin: float = 0.1) -> CtaContractSpec:
        spec = self.get(product)
        if spec is not None:
            return spec
        return CtaContractSpec(
            exchange="", product=product.upper(), contract_multiplier=1.0,
            price_unit="", tick_size=0.0, margin_ratio=default_margin,
        )

    @property
    def products(self) -> list[str]:
        return sorted(self._specs.keys())
