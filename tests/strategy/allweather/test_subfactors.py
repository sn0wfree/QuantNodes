# coding=utf-8
"""AllWeather V6 — subfactors.py 单元测试 (PLAN §13.3.1)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.subfactors import (
    SubFactorConfig,
    compute_all_subfactors,
    compute_subfactor_a,
    compute_subfactor_b,
    compute_subfactor_c,
)


def _panel(n_assets: int = 10, t: int = 300, seed: int = 0) -> pd.DataFrame:
    """合成价格面板 (T × N, bdate)."""
    r = np.random.default_rng(seed)
    rets = r.normal(0.0003, 0.015, size=(t, n_assets))
    prices = np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(
        prices,
        index=pd.bdate_range("2020-01-01", periods=t),
        columns=[f"A{i}" for i in range(n_assets)],
    )


def _volumes(prices: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """合成 volume 面板 (与 prices 同 shape, 全正)."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        r.uniform(1e5, 1e6, size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )


def _oi(prices: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """合成 open_interest 面板."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        r.uniform(1e4, 1e5, size=prices.shape),
        index=prices.index,
        columns=prices.columns,
    )


class TestSubFactorA:
    def test_output_range(self):
        """Sub-A 输出值域 [-1, +1]."""
        prices = _panel()
        cfg = SubFactorConfig()
        sig = compute_subfactor_a(prices, cfg)
        assert sig.shape == prices.shape
        assert sig.min().min() >= -1.0
        assert sig.max().max() <= 1.0
        assert sig.notna().any().any()


class TestSubFactorB:
    def test_combines_trend_quality_and_momentum_slope(self):
        """Sub-B = (TrendQuality + MomentumSlope) / 2, shape 对齐."""
        prices = _panel()
        cfg = SubFactorConfig()
        sig = compute_subfactor_b(prices, cfg)
        assert sig.shape == prices.shape
        assert sig.min().min() >= -1.0
        assert sig.max().max() <= 1.0


class TestSubFactorC:
    def test_etf_path_without_oi(self):
        """ETF 无 OI: Sub-C 仅用 VolumeTrendFactor."""
        prices = _panel(n_assets=10)
        volumes = _volumes(prices)
        cfg = SubFactorConfig()
        sig = compute_subfactor_c(prices, volumes, open_interest=None, cfg=cfg)
        assert sig.shape == prices.shape
        assert sig.min().min() >= -1.0
        assert sig.max().max() <= 1.0

    def test_futures_path_with_oi(self):
        """期货含 OI: Sub-C 叠加 OIPriceDivergence."""
        prices = _panel(n_assets=10)
        volumes = _volumes(prices)
        oi = _oi(prices)
        cfg = SubFactorConfig()
        sig = compute_subfactor_c(prices, volumes, open_interest=oi, cfg=cfg)
        assert sig.shape == prices.shape
        assert sig.min().min() >= -1.0
        assert sig.max().max() <= 1.0


class TestComputeAllSubfactors:
    def test_returns_dict_with_three_keys(self):
        """compute_all_subfactors 返回 {a, b, c}."""
        prices = _panel(n_assets=10)
        volumes = _volumes(prices)
        cfg = SubFactorConfig()
        out = compute_all_subfactors(prices, volumes, open_interest=None, cfg=cfg)
        assert set(out.keys()) == {"a", "b", "c"}
        for k, v in out.items():
            assert v.shape == prices.shape
            assert v.min().min() >= -1.0
            assert v.max().max() <= 1.0
