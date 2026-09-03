# coding=utf-8
"""AllWeather — experiments 脚本集冒烟测试 (小面板, 快速)."""
import pandas as pd
import numpy as np

from QuantNodes.strategy.allweather.backtest import AllWeatherConfig
from QuantNodes.strategy.allweather.ensemble import cta_sleeve_returns
from QuantNodes.strategy.allweather.experiments import (
    diag_d1,
    diag_d2,
    run_lever_sweep,
    run_segments,
)


def _panel(t: int = 300, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    cols = {}
    for j in range(5):
        common = r.normal(0, 0.008, size=(t, 1))
        idio = r.normal(0, 0.010, size=(t, 1))
        cols[f"fut_f{j}"] = np.cumprod(1 + common + idio, axis=0).ravel()
    cols["etf_e0"] = np.cumprod(1 + r.normal(0.0003, 0.011, size=t), axis=0)
    return pd.DataFrame(cols, index=pd.bdate_range("2020-01-01", periods=t))


class TestExperiments:
    def test_lever_sweep(self):
        df = run_lever_sweep(_panel(), caps=[1, 2], max_weight=0.5)
        assert list(df["config"]) == ["lev1", "lev2"]
        assert df["Sharpe"].notna().all()

    def test_diag_d1(self):
        panel = _panel()
        d = diag_d1(panel, AllWeatherConfig(window=126, trend_mode="none"))
        assert 0 < d["struct_hhi_mean"] <= 1.0
        assert d["last_top1_weight"] > 0

    def test_diag_d2_small(self):
        df = diag_d2(_panel(), AllWeatherConfig(window=126, trend_mode="none"),
                     n=1, k=2)
        assert len(df) == 2  # full_pool + 1 次 drop
        assert df["Sharpe"].notna().all()

    def test_segments(self):
        df = run_segments(_panel(), AllWeatherConfig(window=126), "smoke")
        assert set(df["segment"]) >= {"full"}

    def test_cta_min_hold_days(self):
        panel = _panel()
        ret1, dw1 = cta_sleeve_returns(panel, min_hold_days=1)
        ret5, dw5 = cta_sleeve_returns(panel, min_hold_days=5)
        # 持有5天换手应 < 持有1天
        assert dw5.sum() < dw1.sum()
        assert len(ret5) == len(ret1)
