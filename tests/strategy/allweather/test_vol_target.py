# coding=utf-8
"""AllWeather — vol_target 单元测试 (Step 4, PLAN §十二)."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.allweather.vol_target import apply_vol_target


class TestApplyVolTarget:
    def _setup(self, daily_vol: float, t: int = 300, n: int = 5, seed: int = 0):
        r = np.random.default_rng(seed)
        rets = pd.DataFrame(r.normal(0, daily_vol, size=(t, n)),
                            columns=[f"a{i}" for i in range(n)])
        weights = pd.DataFrame(1.0 / n, index=rets.index, columns=rets.columns)
        return weights, rets

    def test_high_vol_capped_at_max_leverage(self):
        weights, rets = self._setup(daily_vol=0.05)  # 年化 ~79%
        scaled = apply_vol_target(weights, rets, target_vol=0.10,
                                  lookback=20, max_leverage=3.0)
        leverage = scaled.sum(axis=1)
        assert (leverage.dropna() <= 3.0 + 1e-10).all()

    def test_low_vol_scales_up(self):
        weights, rets = self._setup(daily_vol=0.0005)  # 年化 ~0.8%
        scaled = apply_vol_target(weights, rets, target_vol=0.10,
                                  lookback=20, max_leverage=3.0)
        leverage = scaled.sum(axis=1)
        assert (leverage.dropna().iloc[30:] > 1.0).all()

    def test_realized_vol_near_target(self):
        weights, rets = self._setup(daily_vol=0.01, t=1000, seed=7)
        scaled = apply_vol_target(weights, rets, target_vol=0.10,
                                  lookback=60, max_leverage=5.0)
        port = (scaled * rets).sum(axis=1).dropna()
        realized_annual = port.iloc[200:].std() * np.sqrt(252)
        assert realized_annual == pytest.approx(0.10, rel=0.3)

    def test_no_lookahead_truncation_consistent(self):
        weights, rets = self._setup(daily_vol=0.01, t=400, seed=3)
        full = apply_vol_target(weights, rets, lookback=20)
        for t in (100, 250, 399):
            trunc = apply_vol_target(weights.iloc[: t + 1], rets.iloc[: t + 1], lookback=20)
            np.testing.assert_allclose(
                trunc.iloc[-1].values, full.iloc[t].values, rtol=1e-10
            )

    def test_warmup_leverage_is_one(self):
        weights, rets = self._setup(daily_vol=0.01, t=50)
        scaled = apply_vol_target(weights, rets, lookback=20)
        # 前 lookback+1 日 (shift 后) 无估计 → 杠杆 1
        leverage = scaled.sum(axis=1)
        assert (leverage.iloc[:20] == 1.0).all()

    def test_zero_vol_data_no_crash(self):
        rets = pd.DataFrame(0.0, index=pd.date_range("2020-01-01", periods=100),
                            columns=list("ab"))
        weights = pd.DataFrame(0.5, index=rets.index, columns=rets.columns)
        scaled = apply_vol_target(weights, rets, lookback=20)
        # realized = 0 → target/0 = inf → clip 到 max_leverage
        assert np.isfinite(scaled.values).all()
        assert (scaled.sum(axis=1) <= 3.0 + 1e-10).all()
