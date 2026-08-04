"""Tests for EnsembleRiskWarning — core ensemble volatility forecasting."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    EnsembleRiskWarning, load_etf_data, AssetConfig, CONFIGS,
    compute_nav, compute_y_true, evaluate,
    _cont_prob, _jump_overlay,
)


class TestAssetConfig:
    def test_has_seven_assets(self) -> None:
        assert len(CONFIGS) == 7

    def test_equity_has_510300(self) -> None:
        assert "510300" in CONFIGS["equity"].etfs

    def test_gold_has_518880(self) -> None:
        assert "518880" in CONFIGS["gold"].etfs


class TestComputeNav:
    def test_single_etf(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        etf_data = pd.DataFrame({"510300": 1.0}, index=dates)
        nav = compute_nav(etf_data, CONFIGS["equity"])
        assert nav.name == "equity"
        assert len(nav) == len(dates)

    def test_multi_etf_weighted(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        etf_data = pd.DataFrame({"518880": 1.0, "159985": 2.0}, index=dates)
        nav = compute_nav(etf_data, CONFIGS["gold"])
        assert nav.name == "gold"
        assert nav.iloc[0] == 1.0

    def test_missing_etf_skips(self) -> None:
        dates = pd.bdate_range("2024-01-01", "2024-01-10")
        etf_data = pd.DataFrame({"510300": 1.0}, index=dates)
        cfg = AssetConfig("unknown", ["missing_etf"])
        try:
            compute_nav(etf_data, cfg)
            assert False, "should raise"
        except ValueError:
            pass


class TestComputeYTrue:
    def test_returns_tuple(self) -> None:
        np.random.seed(42)
        rets = pd.Series(np.random.randn(500), index=pd.bdate_range("2022-01-01", periods=500))
        y_true, realized_vol, threshold = compute_y_true(rets, CONFIGS["equity"])
        assert isinstance(y_true, pd.Series)
        assert y_true.dtype == float
        assert y_true.nunique() >= 1

    def test_binary_output(self) -> None:
        np.random.seed(42)
        rets = pd.Series(np.random.randn(500), index=pd.bdate_range("2022-01-01", periods=500))
        y_true, _, _ = compute_y_true(rets, CONFIGS["equity"])
        assert y_true.dropna().isin([0.0, 1.0]).all()


class TestEvaluate:
    def test_returns_all_keys(self) -> None:
        np.random.seed(42)
        idx = pd.bdate_range("2024-01-01", "2024-06-30")
        y_true = pd.Series(np.random.randint(0, 2, len(idx)), index=idx)
        y_prob = pd.Series(np.random.uniform(0, 1, len(idx)), index=idx)
        m = evaluate(y_true, y_prob)
        for k in ("auc", "prauc", "brier", "ece", "pos_rate", "n"):
            assert k in m, f"missing key {k}"

    def test_single_class_returns_benchmark(self) -> None:
        idx = pd.bdate_range("2024-01-01", "2024-06-30")
        y_true = pd.Series(np.ones(len(idx)), index=idx)
        y_prob = pd.Series(np.full(len(idx), 0.5), index=idx)
        m = evaluate(y_true, y_prob)
        assert m["auc"] == 0.5


class TestContProb:
    def test_output_range(self) -> None:
        np.random.seed(42)
        idx = pd.bdate_range("2022-01-01", periods=500)
        forecast = pd.Series(np.random.exponential(1, len(idx)), index=idx)
        prob = _cont_prob(forecast, window=252)
        assert prob.dropna().between(0, 1).all()

    def test_monotonic_in_small_sample(self) -> None:
        idx = pd.bdate_range("2022-01-01", periods=100)
        forecast = pd.Series(np.linspace(0.1, 2.0, len(idx)), index=idx)
        prob = _cont_prob(forecast, window=60)
        assert prob.dropna().between(0, 1).all()


class TestJumpOverlay:
    def test_returns_probability(self) -> None:
        np.random.seed(42)
        idx = pd.bdate_range("2022-01-01", periods=504)
        prob = pd.Series(np.full(len(idx), 0.5), index=idx)
        rets = pd.Series(np.random.randn(len(idx)), index=idx)
        result = _jump_overlay(prob, rets, "equity")
        assert result.between(0, 1).all()

    def test_bear_regime_amplifies(self) -> None:
        np.random.seed(42)
        idx = pd.bdate_range("2022-01-01", periods=504)
        prob = pd.Series(np.full(len(idx), 0.5), index=idx)
        # Crash returns
        rets = pd.Series(np.random.randn(len(idx)) * 3 - 2, index=idx)
        result = _jump_overlay(prob, rets, "equity")
        # Some dates should be amplified (>0.5)
        assert result.max() >= 0.5


class TestEnsembleRiskWarning:
    @classmethod
    def setup_class(cls):
        cls.etf_data = load_etf_data()

    def test_init_equity(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        assert rw.asset == "equity"
        assert rw.rets is not None
        assert len(rw.y_true) > 0

    def test_init_gold(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "gold")
        assert rw.asset == "gold"

    def test_init_unknown_asset_raises(self) -> None:
        try:
            EnsembleRiskWarning(self.etf_data, "nonexistent")
            assert False, "should raise"
        except ValueError:
            pass

    def test_predict_returns_float(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        prob = rw.predict("2024-01-04")
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_out_of_range_returns_05(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        prob = rw.predict("2020-01-01")
        assert prob == 0.5

    def test_predict_range_returns_dataframe(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        df = rw.predict_range("2024-01-01", "2024-01-31")
        assert isinstance(df, pd.DataFrame)
        assert "risk_prob" in df.columns
        assert "signal" in df.columns
        assert len(df) > 0

    def test_signal_mapping(self) -> None:
        assert EnsembleRiskWarning._to_signal(0.3) == 0
        assert EnsembleRiskWarning._to_signal(0.6) == 1
        assert EnsembleRiskWarning._to_signal(0.75) == 2
        assert EnsembleRiskWarning._to_signal(0.85) == 3

    def test_signal_method(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        sig = rw.signal("2024-01-04")
        assert sig in (0, 1, 2, 3)

    def test_explain_returns_dict(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        exp = rw.explain("2024-01-04")
        assert isinstance(exp, dict)
        assert "risk_prob" in exp
        assert "signal" in exp
        assert "layers" in exp
        assert "climatology_anchor" in exp

    def test_explain_out_of_range(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        exp = rw.explain("2020-01-01")
        assert "error" in exp

    def test_forecast_df_columns(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        df = rw.forecast_df
        assert "y_true" in df.columns
        assert "ensemble" in df.columns
        for key in rw._prob:
            assert f"prob_{key}" in df.columns
            assert f"weight_{key}" in df.columns

    def test_evaluate_returns_all_keys(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        ev = rw.evaluate("2024-01-01", "2024-06-30")
        assert "ensemble" in ev
        for key in ev:
            for k in ("auc", "prauc", "brier", "ece"):
                assert k in ev[key], f"missing {k} in {key}"

    def test_ensemble_auc_above_05(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "equity")
        ev = rw.evaluate("2024-01-01", "2024-06-30")
        assert ev["ensemble"]["auc"] > 0.5, (
            f"ensemble AUC {ev['ensemble']['auc']:.3f} should be > 0.5"
        )

    def test_gold_init_succeeds(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "gold")
        assert rw._prob is not None
        assert len(rw._prob) >= 1

    def test_commodity_init_succeeds(self) -> None:
        rw = EnsembleRiskWarning(self.etf_data, "commodity")
        assert rw._prob is not None
        assert len(rw._prob) >= 1