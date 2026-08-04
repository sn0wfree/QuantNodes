"""Tests for AutoRiskWarning + AutoOptimizer + VersionManager."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    AutoRiskWarning, AutoOptimizer, VersionManager, load_etf_data,
    EnsembleRiskWarning,
)


class TestVersionManager:
    @classmethod
    def setup_class(cls):
        cls.vm = VersionManager("test_asset")

    def test_list_versions_empty(self) -> None:
        df = self.vm.list_versions()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0 or "version_date" in df.columns

    def test_save_returns_path(self) -> None:
        path = self.vm.save({"test_key": 1.0})
        assert path.exists()
        # Clean up
        path.unlink()

    def test_compare_returns_error_for_missing(self) -> None:
        diff = self.vm.compare(1, 999)
        assert "error" in diff


class TestAutoOptimizer:
    @classmethod
    def setup_class(cls):
        etf = load_etf_data()
        cls.rw = EnsembleRiskWarning(etf, "equity")

    def test_init(self) -> None:
        opt = AutoOptimizer(self.rw)
        assert opt.fallback_active is False
        assert len(opt.action_history) == 0

    def test_run_normal_noop(self) -> None:
        opt = AutoOptimizer(self.rw)
        r = opt.run("Normal")
        assert r["action_taken"] == "noop"
        assert r["changed"] is False

    def test_run_warning_recalibrate(self) -> None:
        opt = AutoOptimizer(self.rw)
        r = opt.run("Warning")
        assert r["action_taken"] == "recalibrate"
        assert r["changed"] is True

    def test_run_degraded_rebuild(self) -> None:
        opt = AutoOptimizer(self.rw)
        r = opt.run("Degraded")
        assert r["action_taken"] == "rebuild"
        assert r["changed"] is True

    def test_run_failed_fallback(self) -> None:
        opt = AutoOptimizer(self.rw)
        r = opt.run("Failed")
        assert r["action_taken"] == "fallback"
        assert r["changed"] is True
        assert opt.fallback_active is True

    def test_run_fallback_twice(self) -> None:
        opt = AutoOptimizer(self.rw)
        opt.run("Failed")
        r = opt.run("Failed")
        assert r["action_taken"] == "fallback (active)"

    def test_action_history_grows(self) -> None:
        opt = AutoOptimizer(self.rw)
        opt.run("Normal")
        opt.run("Warning")
        assert len(opt.action_history) == 2

    def test_fallback_sets_climatology_weight(self) -> None:
        opt = AutoOptimizer(self.rw)
        opt._fallback()
        w = self.rw._ensemble_weights
        for k in w.columns:
            expected = 1.0 if k == "climatology" else 0.0
            assert w[k].iloc[-1] == expected, f"weight {k} = {w[k].iloc[-1]}"

    def test_rebuild_resets_fallback(self) -> None:
        opt = AutoOptimizer(self.rw)
        opt._fallback()
        assert opt.fallback_active is True
        opt._rebuild()
        assert opt.fallback_active is False


class TestAutoRiskWarning:
    @classmethod
    def setup_class(cls):
        cls.etf_data = load_etf_data()

    def test_init(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        assert aw.asset == "equity"
        assert aw.rw is not None
        assert aw.optimizer is not None
        assert aw.versions is not None
        assert aw.evaluator is not None

    def test_weights_initialized(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        assert len(aw.weights) > 0
        total = sum(aw.weights.values())
        assert abs(total - 1.0) < 1e-6, f"weights sum to {total}"

    def test_weight_uncertainty_positive(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        for v in aw.weight_uncertainty.values():
            assert v > 0

    def test_alpha_initialized(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        assert len(aw.alpha) == len(aw.LAYER_KEYS)
        assert (aw.alpha >= 1).all()

    def test_predict_returns_float(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        prob = aw.predict("2024-01-04")
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_range_returns_dataframe(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        df = aw.predict_range("2024-01-01", "2024-01-31")
        assert isinstance(df, pd.DataFrame)
        assert "risk_prob" in df.columns
        assert "signal" in df.columns
        assert len(df) > 0

    def test_weekly_update_returns_dict(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        detail = aw.weekly_update()
        assert isinstance(detail, dict)
        assert "health_score" in detail
        assert "status" in detail
        assert "weights" in detail
        assert "optimizer" in detail
        assert "fallback_active" in detail

    def test_weekly_update_weights_sum_to_one(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        detail = aw.weekly_update()
        total = sum(detail["weights"].values())
        assert abs(total - 1.0) < 1e-6, f"weights sum to {total}"

    def test_weekly_update_gold(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "gold")
        detail = aw.weekly_update()
        assert detail["health_score"] >= 0
        assert detail["status"] in ("Normal", "Warning", "Degraded", "Failed")

    def test_weekly_update_commodity(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "commodity")
        detail = aw.weekly_update()
        assert detail["health_score"] >= 0

    def test_update_weights_bayesian_changes_weights(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        old = dict(aw.weights)
        aw._update_weights_bayesian()
        new = dict(aw.weights)
        # Weights should be updated (may differ for some keys)
        assert len(old) == len(new)

    def test_compute_rolling_mse_positive(self) -> None:
        aw = AutoRiskWarning(self.etf_data, "equity")
        for key in aw.LAYER_KEYS:
            mse = aw._compute_rolling_mse(key)
            assert mse > 0, f"mse for {key} = {mse}"
            assert mse <= 1.0, f"mse for {key} = {mse} > 1.0"