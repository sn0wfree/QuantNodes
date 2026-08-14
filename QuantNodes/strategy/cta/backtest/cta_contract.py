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
class OptionContractSpec:
    """单只期权合约参数 (买方 only, MVP 简化)."""

    exchange: str
    product: str          # 标的品种 (e.g. 'RB' / 'IO')
    underlying: str       # 标的合约代码 (e.g. 'rb2605' / 'IF2605')
    option_type: str      # 'C' (call) / 'P' (put)
    strike: float         # 行权价
    expiry_date: str      # 到期日 'YYYY-MM-DD'
    contract_multiplier: float   # 复用标的期货乘数
    premium_unit: float = 1.0    # 权利金单位 (元/手), 实际 = price × multiplier


@dataclass(frozen=True, slots=True)
class CtaContract:
    """单张期货/期权合约 (含到期月份, 用于换月识别).

    期权时 option_type 非 None; 期货时 None.
    """

    product: str
    symbol: str          # 'IF2608' / 'au2602' / 'rb2605C600-P-5000'
    expiry_year: int
    expiry_month: int
    spec: CtaContractSpec
    option_spec: OptionContractSpec | None = None  # 期权时填入

    @property
    def key(self) -> str:
        return self.symbol

    @property
    def is_option(self) -> bool:
        return self.option_spec is not None


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

    # ── 期权合约解析 (买方 only, MVP) ─────────────────────────────────

    def get_option(
        self, symbol: str, trade_date: str | None = None,
    ) -> CtaContract:
        """解析期权合约 → CtaContract (option_spec 填入).

        Args:
            symbol: 期权合约代码 (如 'rb2605C600-P-5000' / 'IO2502-C-3700')
            trade_date: 交易日 (用于从 DuckDB 查实际到期日, 默认 None)

        Returns:
            CtaContract 含 OptionContractSpec; 期货代码或未知时 OptionContractSpec 为 None

        期权合约乘数继承 product 的期货乘数 (RB 与 rb 同乘数).
        """
        from ..options.symbol_parser import parse_symbol

        # 先用 parse_symbol 推断产品码 (它处理各交易所格式差异)
        exchange_guess = ""
        fut_spec = None
        product = ""
        # 从已知规格反查可能的交易所, 逐个尝试解析
        for spec in self._specs.values():
            candidate = parse_symbol(symbol, spec.exchange)
            if candidate is not None:
                product = candidate.product.upper()
                fut_spec = self._specs.get(product)
                exchange_guess = spec.exchange
                break
        if fut_spec is None and not product:
            return CtaContract(
                product=symbol[:2].upper(), symbol=symbol,
                expiry_year=0, expiry_month=0,
                spec=self.get_or_default(symbol[:2].upper()),
            )
        exchange = exchange_guess
        mult = fut_spec.contract_multiplier if fut_spec else 1.0
        parsed = parse_symbol(symbol, exchange)
        if parsed is None:
            return CtaContract(
                product=product, symbol=symbol, expiry_year=0, expiry_month=0,
                spec=fut_spec or self.get_or_default(product),
            )
        year = parsed.expiry_year
        month = parsed.expiry_month
        # 到期日: 当月最后一天 (简化, 精确到期日需交易日历)
        expiry = f"{year:04d}-{month:02d}-01" if year else ""
        opt_spec = OptionContractSpec(
            exchange=exchange,
            product=product,
            underlying=parsed.underlying or "",
            option_type=parsed.option_type or "",
            strike=float(parsed.strike_price or 0),
            expiry_date=expiry or "",
            contract_multiplier=mult,
        )
        return CtaContract(
            product=product, symbol=symbol,
            expiry_year=year, expiry_month=month,
            spec=fut_spec or self.get_or_default(product),
            option_spec=opt_spec,
        )
