# coding=utf-8
"""AllWeather V6 — backtest subfactor 接入单元测试 (PLAN §13.3.3)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.backtest import (
    AllWeatherConfig,
    _apply_subfactor,
    run_backtest,
    subfactor_weights,
)


def _panel(t: int = 400, n: int = 8, seed: int = 0) -> pd.DataFrame:
    """合成价格面板."""
    r = np.random.default_rng(seed)
    rets = r.normal(0.0003, 0.012, size=(t, n))
    prices = np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(
        prices,
        index=pd.bdate_range("2020-01-01", periods=t),
        columns=[f"fut_{i}" for i in range(n)],
    )


class TestSubfactorWeights:
    def test_neutral_score_leaves_base_unchanged(self):
        """中性分数 (z=0) 不动基础权重."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        score = pd.Series({"a": 0.0, "b": 0.0})
        out = subfactor_weights(base, score, alpha=0.3)
        assert abs(out["a"] - 0.5) < 1e-9
        assert abs(out["b"] - 0.5) < 1e-9

    def test_positive_negative_symmetric(self):
        """正负分数对称调整 (z=+2 加权, z=-2 减权)."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        score = pd.Series({"a": 2.0, "b": -2.0})
        out = subfactor_weights(base, score, alpha=0.3)
        # 0.5*(1+0.6)=0.8, 0.5*(1-0.6)=0.2
        assert abs(out["a"] - 0.8) < 1e-9
        assert abs(out["b"] - 0.2) < 1e-9

    def test_extreme_negative_clips_to_zero(self):
        """极负分数 clip 到 0 (不允许负权重)."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        score = pd.Series({"a": -5.0, "b": -5.0})
        out = subfactor_weights(base, score, alpha=0.3)
        assert out["a"] >= 0
        assert out["b"] >= 0

    def test_alpha_zero_returns_base(self):
        """alpha=0 时直接返回 base (零开销)."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        score = pd.Series({"a": 10.0, "b": -10.0})
        out = subfactor_weights(base, score, alpha=0.0)
        assert (out == base).all()

    def test_asymmetric_alpha(self):
        """alpha_pos/alpha_neg 非对称叠加."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        score = pd.Series({"a": 2.0, "b": -2.0})
        out = subfactor_weights(base, score, alpha=0.3,
                                alpha_pos=0.1, alpha_neg=0.5)
        # a 正向 → 0.5*(1+0.1*2)=0.6
        # b 负向 → 0.5*(1+0.5*(-2))=0.5*(1-1)=0.0 (clip)
        assert abs(out["a"] - 0.6) < 1e-9
        assert abs(out["b"]) < 1e-9


class TestApplySubfactor:
    def test_none_mode_returns_base(self):
        """mode=none 或 score=None 时直接返回 base."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        date = pd.Timestamp("2020-06-01")
        cfg = AllWeatherConfig(subfactor_mode="none")
        out = _apply_subfactor(base, date, cfg)
        assert (out == base).all()

    def test_missing_date_returns_base(self):
        """date 不在 score_panel.index 时返回 base."""
        base = pd.Series({"a": 0.5, "b": 0.5})
        date = pd.Timestamp("2020-06-01")
        score_panel = pd.DataFrame(
            {"a": [1.0], "b": [1.0]},
            index=pd.bdate_range("2021-01-01", periods=1),
        )
        cfg = AllWeatherConfig(
            subfactor_mode="abc",
            subfactor_score=score_panel,
        )
        out = _apply_subfactor(base, date, cfg)
        assert (out == base).all()


class TestSubfactorInBacktest:
    def test_run_backtest_with_subfactor_no_crash(self):
        """run_backtest + subfactor_score 不崩溃, NAV 长度正确."""
        panel = _panel()
        dates = panel.index
        score_idx = dates[dates.dayofweek == 4]  # W-FRI
        rng = np.random.default_rng(42)
        score_panel = pd.DataFrame(
            rng.uniform(-1, 1, size=(len(score_idx), panel.shape[1])),
            index=score_idx,
            columns=panel.columns,
        )
        cfg = AllWeatherConfig(
            window=126,
            subfactor_mode="abc",
            subfactor_alpha=0.3,
            subfactor_score=score_panel,
        )
        res = run_backtest(panel, cfg)
        assert len(res.nav) == len(panel)
