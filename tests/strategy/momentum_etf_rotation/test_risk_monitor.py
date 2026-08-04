"""Tests for ModelMonitor + SelfEvaluator."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    EnsembleRiskWarning, load_etf_data,
)
from QuantNodes.strategy.momentum_etf_rotation.risk.monitor import (
    ModelMonitor, SelfEvaluator, ALERT_RULES,
)


class TestModelMonitor:
    @classmethod
    def setup_class(cls):
        etf = load_etf_data()
        cls.rw = EnsembleRiskWarning(etf, "equity")

    def test_init_creates_rolling(self) -> None:
        m = ModelMonitor(self.rw)
        assert m._rolling is not None

    def test_rolling_has_expected_columns(self) -> None:
        m = ModelMonitor(self.rw)
        if len(m._rolling) > 0:
            for col in ("rolling_auc", "rolling_brier", "rolling_ece", "auc_gap"):
                assert col in m._rolling.columns, f"missing {col}"

    def test_check_alerts_returns_list(self) -> None:
        m = ModelMonitor(self.rw)
        if len(m._rolling) > 0:
            alerts = m.check_alerts(m._rolling.index[-1])
            assert isinstance(alerts, list)

    def test_drift_detected_returns_bool(self) -> None:
        m = ModelMonitor(self.rw)
        result = m.drift_detected()
        assert isinstance(result, (bool, np.bool_))

    def test_current_status_is_string(self) -> None:
        m = ModelMonitor(self.rw)
        status = m.current_status
        assert status in ("normal", "warning", "critical")

    def test_report_returns_string(self) -> None:
        m = ModelMonitor(self.rw)
        if len(m._rolling) > 0:
            r = m.report()
            assert isinstance(r, str)
            assert len(r) > 0

    def test_report_no_data(self) -> None:
        m = ModelMonitor(self.rw)
        # Empty rolling should produce "No data" message
        m._rolling = pd.DataFrame()
        r = m.report()
        assert "No data" in r


class TestAlertRules:
    def test_auc_degraded(self) -> None:
        assert ALERT_RULES["auc_degraded"]({"rolling_auc": 0.50}) is True
        assert ALERT_RULES["auc_degraded"]({"rolling_auc": 0.60}) is False

    def test_brier_degraded(self) -> None:
        assert ALERT_RULES["brier_degraded"]({"rolling_brier": 0.35}) is True
        assert ALERT_RULES["brier_degraded"]({"rolling_brier": 0.25}) is False

    def test_ece_degraded(self) -> None:
        assert ALERT_RULES["ece_degraded"]({"rolling_ece": 0.25}) is True
        assert ALERT_RULES["ece_degraded"]({"rolling_ece": 0.15}) is False

    def test_outperformed_by_baseline(self) -> None:
        assert ALERT_RULES["outperformed_by_baseline"]({"auc_gap": -0.10}) is True
        assert ALERT_RULES["outperformed_by_baseline"]({"auc_gap": 0.0}) is False

    def test_too_many_alerts(self) -> None:
        assert ALERT_RULES["too_many_alerts"]({"high_signal_ratio": 0.50}) is True
        assert ALERT_RULES["too_many_alerts"]({"high_signal_ratio": 0.30}) is False

    def test_low_precision(self) -> None:
        assert ALERT_RULES["low_precision"]({"high_signal_precision": 0.10}) is True
        assert ALERT_RULES["low_precision"]({"high_signal_precision": 0.30}) is False


class TestSelfEvaluator:
    @classmethod
    def setup_class(cls):
        etf = load_etf_data()
        cls.rw = EnsembleRiskWarning(etf, "equity")

    def test_health_score_in_range(self) -> None:
        s = SelfEvaluator(self.rw)
        score = s.health_score()
        assert 0 <= score <= 100, f"health_score {score} out of [0, 100]"

    def test_status_is_known(self) -> None:
        s = SelfEvaluator(self.rw)
        st = s.status()
        assert st in ("Normal", "Warning", "Degraded", "Failed"), f"unknown status {st}"

    def test_action_maps_to_status(self) -> None:
        s = SelfEvaluator(self.rw)
        action = s.action()
        assert action in ("continue", "recalibrate", "rebuild", "fallback")

    def test_detail_returns_dict(self) -> None:
        s = SelfEvaluator(self.rw)
        d = s.detail()
        assert isinstance(d, dict)
        assert "health_score" in d
        assert "status" in d
        assert "action" in d

    def test_refresh_creates_new_monitor(self) -> None:
        s = SelfEvaluator(self.rw)
        old_id = id(s._monitor)
        s.refresh()
        assert id(s._monitor) != old_id

    def test_report_returns_string(self) -> None:
        s = SelfEvaluator(self.rw)
        r = s.report()
        assert isinstance(r, str)
        assert len(r) > 0