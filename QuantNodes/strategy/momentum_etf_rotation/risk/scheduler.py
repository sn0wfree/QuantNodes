"""Risk scheduler — daily/weekly operational infrastructure."""
from __future__ import annotations
import pandas as pd

from .ensemble import EnsembleRiskWarning, load_etf_data, MON_DIR
from .auto import AutoRiskWarning


# ── Operational Infrastructure ──────────────────────────────────

class RiskScheduler:
    """Daily/weekly risk model update scheduler.

    Daily:
      - Check data freshness (last NAV date)
      - Predict risk probability for today
      - Log to status file

    Weekly:
      - Run AutoRiskWarning.weekly_update() (self-eval + Bayesian weights)
      - Save version snapshot
      - Check alert rules, write to alert log

    Usage:
        scheduler = RiskScheduler()
        scheduler.daily()
        scheduler.weekly()  # run once per week
    """

    STATUS_DIR = MON_DIR / "status"
    ALERT_LOG = MON_DIR / "alerts.csv"

    def __init__(self):
        self.STATUS_DIR.mkdir(exist_ok=True, parents=True)
        self.models: dict[str, AutoRiskWarning] = {}

    def _ensure_model(self, asset: str) -> AutoRiskWarning:
        if asset not in self.models:
            etf_data = load_etf_data()
            self.models[asset] = AutoRiskWarning(etf_data, asset)
        return self.models[asset]

    def daily(self) -> dict:
        today = pd.Timestamp.now().normalize()
        result = {"date": str(today.date()), "assets": {}}
        for asset in ["equity", "commodity", "gold"]:
            aw = self._ensure_model(asset)
            prob = aw.predict(today)
            signal = EnsembleRiskWarning._to_signal(prob)
            result["assets"][asset] = {
                "risk_prob": prob,
                "signal": signal,
                "status": aw.evaluator.status(),
            }
        pd.DataFrame([result["assets"]]).to_parquet(
            self.STATUS_DIR / f"daily_{today.strftime('%Y-%m-%d')}.parquet")
        return result

    def weekly(self) -> dict:
        today = pd.Timestamp.now().normalize()
        result = {"date": str(today.date()), "assets": {}}
        for asset in ["equity", "commodity", "gold"]:
            aw = self._ensure_model(asset)
            detail = aw.weekly_update()
            result["assets"][asset] = {
                "health_score": detail["health_score"],
                "status": detail["status"],
                "action": detail["action"],
                "fallback_active": detail.get("fallback_active", False),
                "weights": detail["weights"],
            }
            alerts = aw.evaluator._monitor.check_alerts(today)
            if alerts:
                for a in alerts:
                    a["asset"] = asset
                    a["date"] = str(today.date())
                    pd.DataFrame([a]).to_csv(
                        self.ALERT_LOG, mode="a", header=not self.ALERT_LOG.exists(),
                        index=False)
        return result

    def data_freshness(self) -> dict:
        etf_data = load_etf_data()
        last_date = etf_data.index.max()
        days_since = (pd.Timestamp.now() - last_date).days
        return {"last_data_date": str(last_date.date()), "days_since_update": days_since,
                "stale": days_since > 5}
