"""Auto risk warning — self-evaluation, Bayesian weight update, versioning, auto-optimizer."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .ensemble import EnsembleRiskWarning, MON_DIR
from .monitor import ModelMonitor, SelfEvaluator
from .calibrate import PlattCalibrator


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


# ── AutoRiskWarning ──────────────────────────────────────────────

class AutoRiskWarning:
    """Automatic learning framework: Bayesian weights + self-evaluation + versioning.

    Weekly workflow:
      1. Evaluate all layers
      2. Bayesian weight update (Dirichlet prior)
      3. SelfEvaluator health score
      4. AutoOptimizer route action based on status
      5. Save version snapshot

    Usage:
        aw = AutoRiskWarning(etf_data, "equity")
        report = aw.weekly_update()
        prob = aw.predict("2024-01-04")
    """

    def __init__(self, etf_data: pd.DataFrame, asset: str):
        self.rw = EnsembleRiskWarning(etf_data, asset)
        self.asset = asset
        self.evaluator = SelfEvaluator(self.rw)
        self.versions = VersionManager(asset)
        self.optimizer = AutoOptimizer(self.rw)
        self._init_weights()

    def _init_weights(self):
        self.LAYER_KEYS = list(self.rw._prob.keys())
        n = len(self.LAYER_KEYS)
        self.weights = {k: 1.0 / n for k in self.LAYER_KEYS}
        self.weight_uncertainty = {k: 0.5 for k in self.LAYER_KEYS}
        self.alpha = np.ones(n)

    def _compute_rolling_mse(self, key: str, window: int = 60) -> float:
        prob = self.rw._prob[key]
        yt = self.rw.y_true.reindex(prob.index).dropna()
        common = yt.index.intersection(prob.dropna().index)
        if len(common) >= window:
            common = common[-window:]
        if len(common) < 10:
            return 0.25
        err = prob.loc[common] - yt.loc[common]
        return float((err ** 2).mean())

    def _update_weights_bayesian(self, window: int = 60):
        alpha = np.ones(len(self.LAYER_KEYS))
        for i, k in enumerate(self.LAYER_KEYS):
            mse = self._compute_rolling_mse(k, window)
            likelihood = 1.0 / max(mse, 1e-10)
            alpha[i] += likelihood * min(window, 60)

        self.alpha = alpha
        total = float(alpha.sum())
        for i, k in enumerate(self.LAYER_KEYS):
            self.weights[k] = alpha[i] / total
            self.weight_uncertainty[k] = 1.0 / np.sqrt(alpha[i])

    def predict(self, date: pd.Timestamp | str) -> float:
        d = pd.Timestamp(date)
        prob = 0.0
        for k in self.LAYER_KEYS:
            if d in self.rw._prob[k].index:
                prob += self.weights[k] * float(self.rw._prob[k].loc[d])
            else:
                prob += self.weights[k] * 0.5
        return min(max(prob, 0.0), 1.0)

    def predict_range(self, start: str, end: str) -> pd.DataFrame:
        dates = pd.bdate_range(start, end)
        dates = [d for d in dates if d in self.rw._prob[self.LAYER_KEYS[0]].index]
        rows = []
        for d in dates:
            p = self.predict(d)
            rows.append({"date": d, "risk_prob": p,
                         "signal": EnsembleRiskWarning._to_signal(p)})
        df = pd.DataFrame(rows).set_index("date")
        return df

    def weekly_update(self) -> dict:
        self.evaluator.refresh()
        self._update_weights_bayesian()
        status = self.evaluator.status()
        opt_result = self.optimizer.run(status)
        detail = self.evaluator.detail()
        detail["weights"] = dict(self.weights)
        detail["weight_uncertainty"] = dict(self.weight_uncertainty)
        detail["alpha"] = self.alpha.tolist()
        detail["optimizer"] = opt_result
        detail["fallback_active"] = self.optimizer.fallback_active
        self.versions.save(detail)
        return detail


# ── AutoOptimizer ───────────────────────────────────────────────

class AutoOptimizer:
    """Executes self-healing actions based on SelfEvaluator status.

    Actions:
      Normal:    no-op
      Warning:   Platt calibrate probability outputs
      Degraded:  rebuild all layer models
      Failed:    switch to Climatology fallback (single model)
    """

    def __init__(self, rw: EnsembleRiskWarning):
        self.rw = rw
        self.calibrator = PlattCalibrator()
        self.action_history: list[dict] = []
        self.fallback_active = False

    def run(self, status: str) -> dict:
        result = {"status": status, "action_taken": None, "changed": False}
        if status == "Normal":
            result["action_taken"] = "noop"
        elif status == "Warning":
            result["action_taken"] = "recalibrate"
            result["changed"] = self._recalibrate()
        elif status == "Degraded":
            result["action_taken"] = "rebuild"
            result["changed"] = self._rebuild()
        elif status == "Failed":
            if not self.fallback_active:
                result["action_taken"] = "fallback"
                result["changed"] = self._fallback()
            else:
                result["action_taken"] = "fallback (active)"
        self.action_history.append(result)
        return result

    def _recalibrate(self) -> bool:
        prob = self.rw._ensemble_prob
        yt = self.rw.y_true.reindex(prob.index).dropna()
        cal = self.calibrator.fit_transform(prob, yt)
        self.rw._ensemble_prob = cal
        return True

    def _rebuild(self) -> bool:
        self.rw._fit_all()
        self.calibrator = PlattCalibrator()
        self.fallback_active = False
        return True

    def _fallback(self) -> bool:
        w = self.rw._ensemble_weights
        for k in w.columns:
            w[k] = 1.0 if k == "climatology" else 0.0
        self.rw._ensemble_prob = self.rw._prob["climatology"].copy()
        self.fallback_active = True
        return True
