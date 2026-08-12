# coding=utf-8
"""期权 greeks 计算测试."""
import pandas as pd

from QuantNodes.strategy.cta.options.greeks import (
    GreeksResult,
    compute_greeks_for_option,
    compute_options_greeks,
    compute_straddle_greeks,
    days_to_expiry,
    map_cffex_underlying,
)


class TestDaysToExpiry:
    def test_normal(self):
        assert abs(days_to_expiry("2025-01-06", "2025-02-15") - 40 / 365) < 1e-9

    def test_past_expiry(self):
        # 已到期 → 极小正数 (防除零), 但不为 0
        assert days_to_expiry("2025-03-01", "2025-02-15") > 0


class TestComputeGreeksForOption:
    def test_atm_call(self):
        r = compute_greeks_for_option(
            symbol="au2502C632", option_type="C", strike_price=632.0,
            underlying_price=624.92, market_price=7.0, time_to_maturity=0.1,
        )
        assert r.valid
        assert 0 < r.delta < 1
        assert r.gamma > 0
        assert r.theta < 0
        assert 0.01 < r.iv < 2.0

    def test_atm_put(self):
        r = compute_greeks_for_option(
            symbol="au2502P624", option_type="P", strike_price=624.0,
            underlying_price=624.92, market_price=6.0, time_to_maturity=0.1,
        )
        assert r.valid
        assert -1 < r.delta < 0
        assert r.gamma > 0

    def test_deep_otm_invalid(self):
        # 价格 0 → IV 无解 (非正价格)
        r = compute_greeks_for_option(
            symbol="m2505-C-3500", option_type="C", strike_price=3500.0,
            underlying_price=2697.0, market_price=0.0, time_to_maturity=0.1,
        )
        assert not r.valid

    def test_invalid_inputs(self):
        r = compute_greeks_for_option(
            symbol="x", option_type="C", strike_price=0,
            underlying_price=100, market_price=0, time_to_maturity=0,
        )
        assert not r.valid


class TestStraddle:
    def test_straddle_greeks(self):
        call = compute_greeks_for_option(
            symbol="au2502C624", option_type="C", strike_price=624.0,
            underlying_price=624.92, market_price=8.0, time_to_maturity=0.1,
        )
        put = compute_greeks_for_option(
            symbol="au2502P624", option_type="P", strike_price=624.0,
            underlying_price=624.92, market_price=8.0, time_to_maturity=0.1,
        )
        assert call.valid and put.valid
        s = compute_straddle_greeks(call, put)
        assert s is not None
        # long straddle: gamma > 0, vega > 0, theta < 0, delta 接近 0
        assert s.gamma > 0
        assert s.vega > 0
        assert s.theta < 0
        assert abs(s.delta) < 1.0
        assert s.cost > 0


class TestMapCffexUnderlying:
    def test_io_to_if(self):
        assert map_cffex_underlying("IO2502", "IO") == "IF2502"
        assert map_cffex_underlying("MO2503", "MO") == "IM2503"
        assert map_cffex_underlying("HO2506", "HO") == "IH2506"

    def test_non_cffex_unchanged(self):
        assert map_cffex_underlying("au2602", "au") == "au2602"


class TestComputeOptionsGreeks:
    def test_batch(self):
        df = pd.DataFrame([
            {"symbol": "au2502C624", "underlying": "au2602", "product": "AU",
             "option_type": "C", "strike_price": 624.0, "close": 8.0,
             "trade_date": "2025-01-06"},
            {"symbol": "au2502P624", "underlying": "au2602", "product": "AU",
             "option_type": "P", "strike_price": 624.0, "close": 8.0,
             "trade_date": "2025-01-06"},
        ])
        underlying_map = {"au2602": 624.92}
        expiry_map = {"au2602": pd.Timestamp("2025-02-15")}
        results = compute_options_greeks(df, underlying_map, expiry_map)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, GreeksResult)
            assert r.valid
