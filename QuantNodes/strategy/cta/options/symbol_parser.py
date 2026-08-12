# coding=utf-8
"""期权 symbol 统一解析器.

国内 6 家交易所期权合约代码格式不一, 需统一解析为结构化字段:
    expiry  (year, month)
    option_type  ('C' / 'P')
    strike_price (float)
    underlying   (标的合约代码, 如 IF2609 / au2602)

支持的格式 (实测 DuckDB v_all_options):
    CFFEX:  IO2609-P-5400          → 标的前缀2字符, YYMM, 类型大写
    DCE:    m2608-P-2750           → 标的前缀1字符, YYMM, 类型大写
    GFEX:   si2604-P-8500          → 标的前缀2字符, YYMM, 类型大写
    SHFE:   au2602P800             → 无分隔符, 类型接在 YYMM 后
    CZCE:   TA703C5500             → 标的前缀2字符, 年份1位, 类型无分隔
    INE:    sc2605P510             → 无分隔符 (与 SHFE 同)

返回 OptionSymbol 结构, 或直接用 OptionSymbol.parse(symbol, exchange).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

# CZCE 产品前缀均 2 字符 (如 TA/MA/CF/SR), SHFE/DCE/INE/GFEX 前缀 1-2 字符
_CZCE_PRODUCTS = {
    "WH", "PM", "CF", "SR", "TA", "OI", "RI", "MA", "FG", "RS",
    "RM", "ZC", "JR", "LR", "SF", "SM", "CY", "AP", "UR", "CJ",
    "SA", "PK", "PF", "PX", "SH", "PR", "PL",
}


@dataclass(frozen=True, slots=True)
class OptionSymbol:
    """结构化期权合约代码."""

    exchange: str
    product: str          # 'IO' / 'AU' / 'M' / 'TA' ...
    underlying: str       # 标的合约代码 'IO2609' / 'au2602' / 'TA703'
    option_type: str      # 'C' / 'P'
    expiry_year: int
    expiry_month: int
    strike_price: float

    @property
    def key(self) -> str:
        """规范 key: 'IO-2609-C-5400' (统一大小写+分隔符)."""
        return (
            f"{self.underlying.upper()}-{self.option_type}-"
            f"{self.strike_price:g}"
        )


@lru_cache(maxsize=4096)
def parse_symbol(symbol: str, exchange: str) -> Optional[OptionSymbol]:
    """解析单只期权合约代码.

    Args:
        symbol:   原始合约代码, 如 'IO2609-P-5400' / 'au2602P800' / 'TA703C5500'
        exchange: 交易所代码 'CFFEX' / 'SHFE' / 'DCE' / 'CZCE' / 'INE' / 'GFEX'

    Returns:
        OptionSymbol, 无法解析时 None
    """
    if not symbol or not exchange:
        return None
    exchange = exchange.upper()

    try:
        if exchange == "CZCE":
            return _parse_czce(symbol)
        # 其他交易所存在 dashed / undashed 两种历史格式混用,
        # 按 "先 dashed 后 undashed" 或反之尝试, 直到成功.
        # 优先 dashed (新格式): 'm2608-P-2750' / 'bc2603-C-900'
        if "-" in symbol:
            r = _parse_dashed(symbol, exchange)
            if r is not None:
                return r
        return _parse_undashed(symbol, exchange)
    except (ValueError, IndexError):
        return None


def _parse_dashed(symbol: str, exchange: str) -> OptionSymbol:
    """CFFEX/DCE/GFEX/INE 的 'IO2609-P-5400' / 'm2608-P-2750' 格式."""
    parts = symbol.split("-")
    if len(parts) != 3:
        return None
    underlying = parts[0]
    option_type = parts[1].upper()
    strike = float(parts[2])

    if option_type not in ("C", "P"):
        return None

    product, yymm = _split_product_expiry(underlying)
    year, month = _yymm_to_ym(yymm, exchange=exchange)
    return OptionSymbol(
        exchange=exchange, product=product, underlying=underlying,
        option_type=option_type, expiry_year=year, expiry_month=month,
        strike_price=strike,
    )


def _parse_undashed(symbol: str, exchange: str) -> OptionSymbol:
    """无分隔格式 'au2602P800': 找到 [CP] 字符切分.

    类型字符必须紧跟到期数字 (前一个字符是数字), 避免把产品前缀里的
    'c' 误判 (如 sc/bc/lc/ec 等带 c 的产品).
    """
    for i in range(len(symbol) - 1, 0, -1):
        ch = symbol[i]
        if ch in ("C", "P", "c", "p") and symbol[i - 1].isdigit():
            underlying = symbol[:i]
            option_type = ch.upper()
            strike = float(symbol[i + 1:])
            product, yymm = _split_product_expiry(underlying)
            year, month = _yymm_to_ym(yymm, exchange=exchange)
            return OptionSymbol(
                exchange=exchange, product=product, underlying=underlying,
                option_type=option_type, expiry_year=year,
                expiry_month=month, strike_price=strike,
            )
    return None


def _parse_czce(symbol: str) -> OptionSymbol:
    """CZCE 'TA703C5500' 格式: 2字符产品 + 年(1位)月(2位) + [CP] + strike.

    年份编码: 0 → 2020, 1 → 2021, ..., 9 → 2029.
    """
    for i in range(len(symbol) - 1, 0, -1):
        ch = symbol[i]
        if ch in ("C", "P", "c", "p") and symbol[i - 1].isdigit():
            underlying = symbol[:i]
            option_type = ch.upper()
            strike = float(symbol[i + 1:])

            # CZCE 标的 = 2 字符产品 + 3 位年月 (y 1位 + mm 2位)
            product = underlying[:2].upper()
            ymm = underlying[2:]
            if len(ymm) != 3:
                return None
            year = 2020 + int(ymm[0])
            month = int(ymm[1:])
            if not (1 <= month <= 12):
                return None
            return OptionSymbol(
                exchange="CZCE", product=product, underlying=underlying,
                option_type=option_type, expiry_year=year,
                expiry_month=month, strike_price=strike,
            )
    return None


def _split_product_expiry(underlying: str) -> tuple[str, str]:
    """切分标的 'IO2609' → ('IO', '2609'); 'm2608' → ('m', '2608')."""
    if not underlying:
        raise ValueError("empty underlying")
    # 找第一个数字, 数字前为产品前缀
    for i, ch in enumerate(underlying):
        if ch.isdigit():
            return underlying[:i], underlying[i:]
    raise ValueError(f"no expiry in underlying: {underlying}")


def _yymm_to_ym(yymm: str, exchange: str) -> tuple[int, int]:
    """'2609' → (2026, 9). SHFE/DCE/INE/GFEX 为 YYMM; CZCE 单独处理."""
    if len(yymm) != 4:
        raise ValueError(f"bad yymm: {yymm}")
    year = 2000 + int(yymm[:2])
    month = int(yymm[2:])
    if not (1 <= month <= 12):
        raise ValueError(f"bad month: {month}")
    return year, month


def is_czce_product(product: str) -> bool:
    """判断产品是否在 CZCE (用于年份解码)."""
    return product.upper() in _CZCE_PRODUCTS
