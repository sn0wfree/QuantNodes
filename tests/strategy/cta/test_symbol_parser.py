# coding=utf-8
"""期权 symbol 解析器测试."""
import pytest

from QuantNodes.strategy.cta.options.symbol_parser import parse_symbol


@pytest.mark.parametrize(
    "symbol,exchange,product,ey,em,otype,strike",
    [
        # CFFEX (dashed)
        ("IO2609-P-5400", "CFFEX", "IO", 2026, 9, "P", 5400.0),
        ("IO2610-C-4100", "CFFEX", "IO", 2026, 10, "C", 4100.0),
        # SHFE (undashed)
        ("au2602P800", "SHFE", "au", 2026, 2, "P", 800.0),
        ("au2604C624", "SHFE", "au", 2026, 4, "C", 624.0),
        # DCE (dashed)
        ("m2608-P-2750", "DCE", "m", 2026, 8, "P", 2750.0),
        ("m2607-P-2950", "DCE", "m", 2026, 7, "P", 2950.0),
        # CZCE (1位年份)
        ("TA703C5500", "CZCE", "TA", 2027, 3, "C", 5500.0),
        ("CJ612P11000", "CZCE", "CJ", 2026, 12, "P", 11000.0),
        # INE (undashed + dashed)
        ("sc2605P510", "INE", "sc", 2026, 5, "P", 510.0),
        ("bc2603-C-900", "INE", "bc", 2026, 3, "C", 900.0),
        # GFEX (dashed + undashed)
        ("si2604-P-8500", "GFEX", "si", 2026, 4, "P", 8500.0),
        ("lc2609C15000", "GFEX", "lc", 2026, 9, "C", 15000.0),
    ],
)
def test_parse_symbol_ok(symbol, exchange, product, ey, em, otype, strike):
    r = parse_symbol(symbol, exchange)
    assert r is not None
    assert r.product == product
    assert r.expiry_year == ey
    assert r.expiry_month == em
    assert r.option_type == otype
    assert abs(r.strike_price - strike) < 1e-6


@pytest.mark.parametrize(
    "symbol,exchange",
    [
        ("", "SHFE"),
        (None, "SHFE"),
        ("IO2609", "CFFEX"),          # 缺类型/strike
        ("TA70C5500", "CZCE"),        # 年份位数错误
        ("garbage", "SHFE"),
    ],
)
def test_parse_symbol_invalid(symbol, exchange):
    assert parse_symbol(symbol, exchange) is None


def test_parse_symbol_key_normalization():
    r = parse_symbol("au2602P800", "SHFE")
    assert r.key == "AU2602-P-800"
    r2 = parse_symbol("IO2609-P-5400", "CFFEX")
    assert r2.key == "IO2609-P-5400"
