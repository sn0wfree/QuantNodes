"""Model monitor — rolling metric tracking, drift detection, alerting."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .ensemble import EnsembleRiskWarning, evaluate


# ── ModelMonitor ──────────────────────────────────────────────────

ALERT_RULES = {
    "auc_degraded": lambda m: m.get("rolling_auc", 1) < 0.55,
    "brier_degraded": lambda m: m.get("rolling_brier", 0) > 0.30,
    "ece_degraded": lambda m: m.get("rolling_ece", 0) > 0.20,
    "outperformed_by_baseline": lambda m: m.get("auc_gap", 0) < -0.05,
    "too_many_alerts": lambda m: m.get("high_signal_ratio", 0) > 0.40,
    "low_precision": lambda m: m.get("high_signal_precision", 1) < 0.20,
}


class ModelMonitor:
    """Track model performance over time and detect degradation.

    Computes rolling metrics (60d window), drift detection,
    benchmark comparison, and generates reports.
    """

    def __init__(self, rw: EnsembleRiskWarning, window: int = 60):
        self.rw = rw
        self.window = window
        self._rolling = self._compute_rolling_metrics()
        self._alerts = []

    def _compute_rolling_metrics(self) -> pd.DataFrame:
        """Compute rolling 60d AUC, Brier, ECE for ensemble and climatology."""
        prob = self.rw._ensemble_prob
        clim = self.rw._prob["climatology"]
        yt = self.rw.y_true.reindex(prob.index).dropna()

        dates = prob.index
        if len(dates) < self.window + 10:
            return pd.DataFrame()

        rows = []
        for i in range(self.window, len(dates)):
            s = dates[i - self.window]
            e = dates[i]
            p_slice = prob.loc[s:e]
            c_slice = clim.loc[s:e]
            y_slice = yt.loc[s:e]
            common = y_slice.dropna().index.intersection(p_slice.dropna().index)
            if len(common) < 10:
                continue
            m = evaluate(y_slice.loc[common], p_slice.loc[common])
            mc = evaluate(y_slice.loc[common], c_slice.loc[common])
            high_sig = (p_slice.loc[common] > 0.8).sum()
            high_actual = (y_slice.loc[common] == 1).sum()
            high_sig_actual = ((p_slice.loc[common] > 0.8) & (y_slice.loc[common] == 1)).sum()

            rows.append({
                "date": e,
                "rolling_auc": m["auc"],
                "rolling_brier": m["brier"],
                "rolling_ece": m["ece"],
                "rolling_prauc": m["prauc"],
                "climatology_auc": mc["auc"],
                "auc_gap": m["auc"] - mc["auc"],
                "high_signal_ratio": high_sig / max(len(common), 1),
                "high_signal_precision": high_sig_actual / max(high_sig, 1),
                "n_samples": len(common),
            })

        return pd.DataFrame(rows).set_index("date")

    def check_alerts(self, date: pd.Timestamp | str | None = None) -> list[dict]:
        """Check all alert rules. Returns list of triggered alerts."""
        d = pd.Timestamp(date) if date else self._rolling.index[-1]
        if d not in self._rolling.index:
            return []
        row = self._rolling.loc[d].to_dict()
        triggered = []
        for rule_name, rule_fn in ALERT_RULES.items():
            if rule_fn(row):
                triggered.append({"rule": rule_name, "date": d, "metrics": row})
        self._alerts = triggered
        return triggered

    def drift_detected(self, lookback: int = 126) -> bool:
        """Check if AUC has been steadily declining (6 months)."""
        if len(self._rolling) < lookback:
            return False
        recent = self._rolling.iloc[-lookback:]
        aucs = recent["rolling_auc"].values
        # Linear regression slope
        x = np.arange(len(aucs))
        slope = np.polyfit(x, aucs, 1)[0]
        # Significant negative slope
        return slope < -0.0005  # -0.05 per 100 days

    @property
    def current_status(self) -> str:
        if not self._alerts:
            return "normal"
        critical = [a for a in self._alerts if a["rule"] in
                    ("auc_degraded", "outperformed_by_baseline")]
        if critical:
            return "critical"
        return "warning"

    def report(self, start: str | None = None, end: str | None = None) -> str:
        """Generate a human-readable monitoring report."""
        s = pd.Timestamp(start) if start else self._rolling.index[0]
        e = pd.Timestamp(end) if end else self._rolling.index[-1]
        rolling = self._rolling.loc[s:e]
        if len(rolling) == 0:
            return "No data for report period."

        latest = rolling.iloc[-1]
        prev = rolling.iloc[-min(30, len(rolling))]

        lines = []
        asset = self.rw.asset
        lines.append(f"=== {asset.upper()} Risk Warning - Monitoring Report ===")
        lines.append(f"Period: {s.date()} ~ {e.date()}")
        lines.append(f"Evaluated: {len(rolling)} days\n")

        lines.append("Model Performance:")
        rules = [
            ("Rolling AUC (60d)", "rolling_auc", 0.55, "normal"),
            ("Rolling Brier (60d)", "rolling_brier", 0.30, "max"),
            ("ECE", "rolling_ece", 0.20, "max"),
            ("PR-AUC", "rolling_prauc", None, None),
            ("Climatology Gap", "auc_gap", -0.05, "min"),
        ]
        for label, col, threshold, direction in rules:
            if col not in latest.index:
                continue
            val = latest[col]
            if val is None:
                continue
            status = "○"
            if threshold is not None:
                if direction == "min" and val < threshold:
                    status = "⚠"
                elif direction == "max" and val > threshold:
                    status = "⚠"
                elif direction == "normal" and val < threshold:
                    status = "⚠"
            trend = "" if len(rolling) < 2 else (
                "↑" if val > prev[col] else "↓"
            )
            lines.append(f"  {status} {label}: {val:.4f} {trend}")

        lines.append("\nSignal Distribution:")
        sig = self.rw._ensemble_prob.loc[s:e]
        levels = [(0, 0.5, "Normal"), (0.5, 0.7, "Alert"),
                  (0.7, 0.9, "Warning"), (0.9, 1.0, "Danger")]
        for lo, hi, lbl in levels:
            cnt = ((sig >= lo) & (sig < hi)).sum()
            pct = cnt / max(len(sig), 1) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            lines.append(f"  {lbl:<10} {cnt:>4}d ({pct:>5.1f}%) {bar} {sig.mean():.0%}")

        lines.append(f"\nHigh Signal Precision: {latest['high_signal_precision']:.1%}")
        lines.append(f"High Signal Ratio: {latest['high_signal_ratio']:.1%}")
        lines.append(f"\nDrift Detection:")
        if self.drift_detected():
            lines.append("  ⚠ AUC declining (6-month trend negative)")
        else:
            lines.append("  ○ AUC stable")

        self.check_alerts(e)
        if self._alerts:
            lines.append(f"\nActive Alerts: {len(self._alerts)}")
            for a in self._alerts:
                lines.append(f"  ⚠ {a['rule']}")
        else:
            lines.append("\nActive Alerts: None")

        lines.append(f"\nStatus: {self.current_status.upper()}")
        return "\n".join(lines)


# ── VersionManager ──────────────────────────────────────────────

class VersionManager:
    """Track weekly model version snapshots for comparison."""

    def __init__(self, asset: str):
        self.asset = asset
        self.version_dir = MON_DIR / "versions"
        self.version_dir.mkdir(exist_ok=True, parents=True)
        self._seq = 0

    def save(self, snapshot: dict) -> Path:
        now = pd.Timestamp.now()
        self._seq += 1
        date = now.strftime("%Y-%m-%d")
        path = self.version_dir / f"{self.asset}_v{date}_s{self._seq:03d}.parquet"
        snapshot["version_date"] = str(now)
        snapshot["version_seq"] = self._seq
        pd.DataFrame([snapshot]).to_parquet(path)
        return path

    def list_versions(self) -> pd.DataFrame:
        files = sorted(self.version_dir.glob(f"{self.asset}_v*.parquet"))
        if not files:
            return pd.DataFrame()
        records = []
        for f in files:
            df = pd.read_parquet(f)
            records.append(df.iloc[0].to_dict())
        df = pd.DataFrame(records)
        if "version_seq" in df.columns:
            df = df.sort_values("version_seq")
        return df

    def compare(self, seq1: int, seq2: int) -> dict:
        files = sorted(self.version_dir.glob(f"{self.asset}_v*_s{seq1:03d}.parquet"))
        f1 = files[0] if files else None
        files = sorted(self.version_dir.glob(f"{self.asset}_v*_s{seq2:03d}.parquet"))
        f2 = files[0] if files else None
        if f1 is None or f2 is None:
            return {"error": f"Version not found: {seq1} or {seq2}"}
        d1 = pd.read_parquet(f1).iloc[0]
        d2 = pd.read_parquet(f2).iloc[0]
        diff = {}
        for col in d1.index:
            if col in d2.index:
                v1, v2 = d1[col], d2[col]
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    diff[col] = float(v2 - v1)
        return diff


# ── SelfEvaluator ────────────────────────────────────────────────

class SelfEvaluator:
    """Health scoring + status + action routing for AutoRiskWarning."""

    def __init__(self, rw: EnsembleRiskWarning, window: int = 60):
        self.rw = rw
        self.window = window
        self._monitor = ModelMonitor(rw, window)
        self._rolling = self._monitor._rolling

    def refresh(self):
        self._monitor = ModelMonitor(self.rw, self.window)
        self._rolling = self._monitor._rolling

    def health_score(self) -> int:
        roll = self._rolling.tail(60)
        if len(roll) < 10:
            return 50
        auc = roll["rolling_auc"].mean()
        brier = roll["rolling_brier"].mean()
        gap = roll["auc_gap"].mean()
        precision = roll["high_signal_precision"].mean()

        auc_score = min(auc / 0.55, 1.0) * 40
        brier_score = max(1.0 - brier / 0.30, 0) * 30
        gap_score = max(1.0 + (gap / 0.05), 0) * 20
        signal_score = max(1.0 - (1.0 - precision) / 0.50, 0) * 10
        return int(auc_score + brier_score + gap_score + signal_score)

    def status(self) -> str:
        s = self.health_score()
        if s >= 80:
            return "Normal"
        if s >= 60:
            return "Warning"
        if s >= 40:
            return "Degraded"
        return "Failed"

    def action(self) -> str:
        return {
            "Normal": "continue",
            "Warning": "recalibrate",
            "Degraded": "rebuild",
            "Failed": "fallback",
        }[self.status()]

    def detail(self) -> dict:
        roll = self._rolling.tail(60)
        if len(roll) == 0:
            return {"health_score": 50, "status": "Unknown", "action": "continue"}
        return {
            "health_score": self.health_score(),
            "status": self.status(),
            "action": self.action(),
            "rolling_auc": float(roll["rolling_auc"].iloc[-1]),
            "rolling_brier": float(roll["rolling_brier"].iloc[-1]),
            "rolling_ece": float(roll["rolling_ece"].iloc[-1]),
            "auc_gap": float(roll["auc_gap"].iloc[-1]),
            "high_signal_precision": float(roll["high_signal_precision"].iloc[-1]),
            "high_signal_ratio": float(roll["high_signal_ratio"].iloc[-1]),
        }

    def report(self) -> str:
        return self._monitor.report()

