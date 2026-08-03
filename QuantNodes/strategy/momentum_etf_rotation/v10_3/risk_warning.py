"""v10.3 Risk Warning — Ensemble Volatility Forecasting + Model Monitor.

Production-ready risk warning system for each asset class.

EnsembleRiskWarning:
  Layer 1: HAR-RV (OLS) — baseline vol prediction
  Layer 2: Climatology (DOY mean) — calendar effect
  Layer 3: EWMA (RiskMetrics, 0.94) — fast jump response
  Layer 4: Current Vol (20d trailing) — current state
  Layer 5: Macro (VIZ/DXY/CN-US) — macro factor regression
  Post: Jump Detection Overlay — bear regime boost

ModelMonitor:
  Rolling 60d AUC/Brier/ECE drift detection
  Baseline comparison vs Climatology
  Alert generation for model degradation
"""
from __future__ import annotations
import sys, warnings, time, json
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

# ── Config ──────────────────────────────────────────────────────
OOS_START = "2022-01-04"
OOS_END = "2026-05-29"
log = print
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v10_3"
MON_DIR = OUT_DIR / "monitoring"
HF_DIR = REPO / "data" / "high_freq_macro"


@dataclass
class AssetConfig:
    name: str
    etfs: list[str]
    weights: list[float] | None = None
    vol_window: int = 20
    vol_percentile: float = 0.80
    lookback: int = 252


CONFIGS = {
    "equity": AssetConfig("equity", ["510300"]),
    "commodity": AssetConfig("commodity", ["518880", "159985", "161226", "159981"],
                              weights=[0.50, 0.20, 0.15, 0.15]),
    "gold": AssetConfig("gold", ["518880"]),
    "soybean": AssetConfig("soybean", ["159985"]),
    "silver": AssetConfig("silver", ["161226"]),
    "energy_chem": AssetConfig("energy_chem", ["159981"]),
    "commodity_ex_gold": AssetConfig("commodity_ex_gold", ["159985", "161226", "159981"],
                                      weights=[0.40, 0.30, 0.30]),
}


# ── Data helpers ─────────────────────────────────────────────────

def load_etf_data() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def compute_nav(etf_data: pd.DataFrame, cfg: AssetConfig) -> pd.Series:
    avail = [c for c in cfg.etfs if c in etf_data.columns]
    if not avail:
        raise ValueError(f"No ETFs for {cfg.name}: {cfg.etfs}")
    if cfg.weights:
        w = dict(zip(cfg.etfs, cfg.weights))
        total = sum(w[e] for e in avail)
        nav = sum(etf_data[e] * w[e] / total for e in avail)
    else:
        nav = etf_data[avail[0]]
    nav.name = cfg.name
    return nav


def compute_y_true(rets: pd.Series, cfg: AssetConfig) -> tuple:
    realized_vol = rets.rolling(cfg.vol_window).std().shift(-cfg.vol_window)
    threshold = realized_vol.rolling(cfg.lookback, min_periods=60).quantile(cfg.vol_percentile)
    y_true = (realized_vol > threshold).astype(float)
    return y_true, realized_vol, threshold


# ── Metrics ──────────────────────────────────────────────────────

def evaluate(y_true: pd.Series, y_prob: pd.Series) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    common = y_true.dropna().index.intersection(y_prob.dropna().index)
    yt, yp = y_true.loc[common], y_prob.loc[common]
    if yt.nunique() < 2:
        return {"auc": 0.5, "prauc": yt.mean(), "brier": 0.25, "ece": 0.5,
                "pos_rate": yt.mean(), "n": len(yt)}
    auc = roc_auc_score(yt, yp)
    prauc = average_precision_score(yt, yp)
    brier = brier_score_loss(yt, yp)
    # ECE: Expected Calibration Error
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(yp, bins) - 1
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() > 0:
            ece += abs(yp[mask].mean() - yt[mask].mean()) * mask.sum()
    ece /= len(yt)
    return {"auc": auc, "prauc": prauc, "brier": brier, "ece": ece,
            "pos_rate": yt.mean(), "n": len(yt)}

# ── Layer models ──────────────────────────────────────────────────

def _model_harrv(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    from sklearn.linear_model import LinearRegression
    abs_ret = rets.abs() / 100
    vol_5d = rets.rolling(5).std() / 100
    vol_22d = rets.rolling(22).std() / 100
    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    refit_dates = oos_idx[::20]
    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = pd.DataFrame({"v1": abs_ret.iloc[:ri], "v5": vol_5d.iloc[:ri],
                          "v22": vol_22d.iloc[:ri], "c": 1.0})
        y = fwd_vol.iloc[:ri]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        try:
            m = LinearRegression(fit_intercept=False).fit(Xc, yc)
            fs = rd + pd.Timedelta(days=1)
            for fd in pd.bdate_range(fs, periods=20):
                if fd not in forecasts.index:
                    continue
                fi = rets.index.get_indexer([fd], method="ffill")[0]
                feat = np.array([abs_ret.iloc[fi-1], vol_5d.iloc[fi-1],
                                 vol_22d.iloc[fi-1], 1.0]).reshape(1, -1)
                if not np.any(np.isnan(feat)):
                    forecasts[fd] = m.predict(feat)[0]
        except Exception:
            continue
    return forecasts.fillna(method="ffill")


def _model_climatology(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    for d in oos_idx:
        hist = fwd_vol.loc[:d]
        doy_vals = hist[hist.index.dayofyear == d.dayofyear]
        if len(doy_vals) >= 3:
            forecasts[d] = doy_vals.mean()
        else:
            forecasts[d] = hist.dropna().mean()
    return forecasts.fillna(method="ffill")


def _model_ewma(rets: pd.Series, oos_idx: pd.Index, lam: float = 0.94) -> pd.Series:
    forecasts = pd.Series(index=oos_idx, dtype=float)
    r = rets / 100
    init_window = 60
    var = r.iloc[:init_window].var()
    var_series = pd.Series(index=r.index, dtype=float)
    var_series.iloc[init_window-1] = var
    for i in range(init_window, len(r)):
        var_series.iloc[i] = lam * var_series.iloc[i-1] + (1 - lam) * r.iloc[i-1]**2
    for d in oos_idx:
        if d in var_series.index:
            forecasts[d] = np.sqrt(var_series.loc[d]) * np.sqrt(252)
    return forecasts.fillna(method="ffill")


def _model_current_vol(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    vol = rets.rolling(20).std() / 100 * np.sqrt(252)
    return vol.reindex(oos_idx).fillna(method="ffill")


def _model_macro_equity(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    from sklearn.linear_model import LinearRegression
    vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]
    vix.index = pd.to_datetime(vix.index)
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
    dxy.index = pd.to_datetime(dxy.index)
    cs = pd.read_parquet(HF_DIR / "cn_us_spread_10y.parquet")
    cs["date"] = pd.to_datetime(cs["date"])
    cs = cs.set_index("date")["cn_us_spread"]

    idx = rets.index
    vix = vix.reindex(idx).ffill()
    dxy = dxy.reindex(idx).ffill()
    cs = cs.reindex(idx).ffill()

    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    dxy_z = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    cs_z = (cs - cs.rolling(252).mean()) / cs.rolling(252).std()

    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    refit_dates = oos_idx[::20]
    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = pd.DataFrame({"vix_z": vix_z.iloc[:ri], "dxy_z": dxy_z.iloc[:ri],
                          "cs_z": cs_z.iloc[:ri], "c": 1.0})
        y = fwd_vol.iloc[:ri]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        try:
            m = LinearRegression(fit_intercept=False).fit(Xc, yc)
            fs = rd + pd.Timedelta(days=1)
            for fd in pd.bdate_range(fs, periods=20):
                if fd not in forecasts.index:
                    continue
                fi = rets.index.get_indexer([fd], method="ffill")[0]
                feat = np.array([vix_z.iloc[fi-1], dxy_z.iloc[fi-1],
                                 cs_z.iloc[fi-1], 1.0]).reshape(1, -1)
                if not np.any(np.isnan(feat)):
                    forecasts[fd] = m.predict(feat)[0]
        except Exception:
            continue
    return forecasts.fillna(method="ffill")


def _model_macro_commodity(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    from sklearn.linear_model import LinearRegression
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
    dxy.index = pd.to_datetime(dxy.index)
    go = pd.read_parquet(HF_DIR / "gold_oil_corr.parquet")
    go["date"] = pd.to_datetime(go["date"])
    go = go.set_index("date")["gold_oil_corr"]
    rr = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")["real_rate"]
    rr.index = pd.to_datetime(rr.index)

    idx = rets.index
    dxy = dxy.reindex(idx).ffill()
    go = go.reindex(idx).ffill()
    rr = rr.reindex(idx).ffill()

    dxy_z = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    go_z = (go - go.rolling(252).mean()) / go.rolling(252).std()
    rr_z = (rr - rr.rolling(252).mean()) / rr.rolling(252).std()

    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    refit_dates = oos_idx[::20]
    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = pd.DataFrame({"dxy_z": dxy_z.iloc[:ri], "go_z": go_z.iloc[:ri],
                          "rr_z": rr_z.iloc[:ri], "c": 1.0})
        y = fwd_vol.iloc[:ri]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        try:
            m = LinearRegression(fit_intercept=False).fit(Xc, yc)
            fs = rd + pd.Timedelta(days=1)
            for fd in pd.bdate_range(fs, periods=20):
                if fd not in forecasts.index:
                    continue
                fi = rets.index.get_indexer([fd], method="ffill")[0]
                feat = np.array([dxy_z.iloc[fi-1], go_z.iloc[fi-1],
                                 rr_z.iloc[fi-1], 1.0]).reshape(1, -1)
                if not np.any(np.isnan(feat)):
                    forecasts[fd] = m.predict(feat)[0]
        except Exception:
            continue
    return forecasts.fillna(method="ffill")


def _model_macro_gold(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    from sklearn.linear_model import LinearRegression
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
    dxy.index = pd.to_datetime(dxy.index)
    rr = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")["real_rate"]
    rr.index = pd.to_datetime(rr.index)
    vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]
    vix.index = pd.to_datetime(vix.index)

    idx = rets.index
    dxy = dxy.reindex(idx).ffill()
    rr = rr.reindex(idx).ffill()
    vix = vix.reindex(idx).ffill()

    dxy_z = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    rr_z = (rr - rr.rolling(252).mean()) / rr.rolling(252).std()
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()

    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    refit_dates = oos_idx[::20]
    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = pd.DataFrame({"dxy_z": dxy_z.iloc[:ri], "rr_z": rr_z.iloc[:ri],
                          "vix_z": vix_z.iloc[:ri], "c": 1.0})
        y = fwd_vol.iloc[:ri]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        try:
            m = LinearRegression(fit_intercept=False).fit(Xc, yc)
            fs = rd + pd.Timedelta(days=1)
            for fd in pd.bdate_range(fs, periods=20):
                if fd not in forecasts.index:
                    continue
                fi = rets.index.get_indexer([fd], method="ffill")[0]
                feat = np.array([dxy_z.iloc[fi-1], rr_z.iloc[fi-1],
                                 vix_z.iloc[fi-1], 1.0]).reshape(1, -1)
                if not np.any(np.isnan(feat)):
                    forecasts[fd] = m.predict(feat)[0]
        except Exception:
            continue
    return forecasts.fillna(method="ffill")


MACRO_MODELS = {
    "equity": _model_macro_equity,
    "commodity": _model_macro_commodity,
    "gold": _model_macro_gold,
}


def _jump_overlay(prob: pd.Series, rets: pd.Series,
                  asset_type: str = "equity") -> pd.Series:
    from QuantNodes.strategy.momentum_etf_rotation.v8.jump_model import (
        jump_model_periodic_retrain)
    regime = jump_model_periodic_retrain(rets, asset_type=asset_type,
                                         show_progress=False)
    common = prob.index.intersection(regime.index)
    result = prob.loc[common].copy()
    bear_dates = regime.loc[common][regime.loc[common] == 1].index
    if len(bear_dates) > 0:
        result.loc[bear_dates] = np.minimum(result.loc[bear_dates] * 1.3, 1.0)
    return result


def _cont_prob(forecast: pd.Series, window: int = 252) -> pd.Series:
    prob = pd.Series(index=forecast.index, dtype=float)
    for i in range(len(forecast)):
        d = forecast.index[i]
        hist = forecast.loc[:d].rolling(window, min_periods=60).rank(pct=True)
        prob.iloc[i] = hist.loc[d]
    return prob

# ── EnsembleRiskWarning ──────────────────────────────────────────

class EnsembleRiskWarning:
    """5-layer ensemble volatility forecasting model.

    Layers:
      HAR-RV → Climatology → EWMA(0.94) → Current 20d Vol → Macro (VIX/DXY/CN-US)
      Post-processed by Jump Detection Overlay (v8/jump_model)
      Ensembled via dynamic MSE weights (60d rolling window)

    Output: risk_prob [0, 1] for any date.

    Usage:
        etf_data = load_etf_data()
        rw = EnsembleRiskWarning(etf_data, "equity")
        prob = rw.predict(pd.Timestamp("2024-01-04"))
        df = rw.predict_range("2024-01-01", "2024-06-30")
        rw.evaluate()  # full OOS metrics
    """

    def __init__(self, etf_data: pd.DataFrame, asset: str,
                 oos_start: str = OOS_START, oos_end: str = OOS_END):
        if asset not in CONFIGS:
            raise ValueError(f"Unknown asset {asset}, choose from {list(CONFIGS.keys())}")
        self.cfg = CONFIGS[asset]
        self.asset = asset
        self.oos_start = pd.Timestamp(oos_start)
        self.oos_end = pd.Timestamp(oos_end)

        nav = compute_nav(etf_data, self.cfg)
        self.rets = nav.pct_change().dropna() * 100
        self.y_true, self._realized_vol, self._threshold = compute_y_true(self.rets / 100, self.cfg)

        self.oos_idx = self.rets.index[(self.rets.index >= self.oos_start) &
                                        (self.rets.index <= self.oos_end)]
        self.oos_idx = self.oos_idx.intersection(self.y_true.dropna().index)

        self._fit_all()

    def _fit_all(self):
        self._raw = {}
        self._prob = {}

        models = {
            "harrv": _model_harrv,
            "climatology": _model_climatology,
            "ewma": _model_ewma,
            "current_vol": _model_current_vol,
        }
        if self.asset in MACRO_MODELS:
            models["macro"] = MACRO_MODELS[self.asset]

        for key, fn in models.items():
            self._raw[key] = fn(self.rets, self.oos_idx)
            common = self._raw[key].dropna().index.intersection(self.y_true.dropna().index)
            self._prob[key] = _cont_prob(self._raw[key].loc[common])

        self._ensemble_prob, self._ensemble_weights = self._compute_ensemble_weights()
        self._ensemble_prob = _jump_overlay(self._ensemble_prob, self.rets, self.asset)
        if self.asset == "equity":
            self._ensemble_prob = _jump_overlay(self._ensemble_prob, self.rets, self.asset)
        self._metrics = {}

    def _compute_ensemble_weights(self, window: int = 60):
        keys = list(self._prob.keys())
        probs = {k: self._prob[k] for k in keys}
        yt = self.y_true.dropna()
        common = yt.index.intersection(probs[keys[0]].dropna().index)
        weights = pd.DataFrame(index=common, columns=keys, dtype=float)
        ensemble = pd.Series(index=common, dtype=float)

        for i, d in enumerate(common):
            if i < window:
                for k in keys:
                    weights.loc[d, k] = 1.0 / len(keys)
                ensemble.loc[d] = sum(probs[k].loc[d] * weights.loc[d, k] for k in keys)
                continue
            hist_idx = yt.loc[:d].index[-window:]
            mse = {}
            for k in keys:
                err = probs[k].reindex(hist_idx) - yt.reindex(hist_idx)
                mse[k] = (err ** 2).mean()
            inv = {k: 1.0 / max(v, 1e-10) for k, v in mse.items()}
            total = sum(inv.values())
            for k in keys:
                weights.loc[d, k] = inv[k] / total
            ensemble.loc[d] = sum(probs[k].loc[d] * weights.loc[d, k] for k in keys)

        return ensemble, weights

    def predict(self, date: pd.Timestamp | str) -> float:
        """Return risk probability [0, 1] for a single date."""
        d = pd.Timestamp(date)
        if d in self._ensemble_prob.index:
            return float(self._ensemble_prob.loc[d])
        return 0.5

    def predict_range(self, start: str, end: str) -> pd.DataFrame:
        """Return risk probabilities and signal levels for a date range."""
        sr = self._ensemble_prob.loc[start:end].copy()
        df = pd.DataFrame({
            "risk_prob": sr,
            "signal": sr.apply(self._to_signal),
        })
        df.index.name = "date"
        return df

    def signal(self, date: pd.Timestamp | str, threshold: float = 0.8) -> int:
        """0=normal, 1=alert, 2=warning, 3=danger."""
        prob = self.predict(date)
        return self._to_signal(prob, threshold)

    @staticmethod
    def _to_signal(prob: float, threshold: float = 0.8) -> int:
        if prob >= threshold:
            return 3
        if prob >= threshold - 0.15:
            return 2
        if prob >= threshold - 0.30:
            return 1
        return 0

    def evaluate(self, start: str | None = None, end: str | None = None) -> dict:
        """Evaluate model performance over a period.

        Returns: {auc, prauc, brier, ece, pos_rate, n} for ensemble and each layer.
        """
        s = pd.Timestamp(start) if start else self.oos_start
        e = pd.Timestamp(end) if end else self.oos_end
        result = {}
        prob = self._ensemble_prob.loc[s:e]
        yt = self.y_true.loc[s:e].reindex(prob.index).dropna()
        result["ensemble"] = evaluate(yt, prob.reindex(yt.index))
        for key in self._prob:
            p = self._prob[key].loc[s:e]
            result[key] = evaluate(yt, p.reindex(yt.index))
        self._metrics = result
        return result

    @property
    def forecast_df(self) -> pd.DataFrame:
        """Full forecast DataFrame with all layers and weights."""
        df = pd.DataFrame({"y_true": self.y_true, "ensemble": self._ensemble_prob})
        for key in self._prob:
            df[f"prob_{key}"] = self._prob[key]
        for i, key in enumerate(self._prob):
            df[f"weight_{key}"] = self._ensemble_weights.iloc[:, i]
        return df

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


# ── AutoRiskWarning ──────────────────────────────────────────────

class AutoRiskWarning:
    """Automatic learning framework: Bayesian weights + self-evaluation + versioning.

    Weekly workflow:
      1. Evaluate all 4 layers
      2. Bayesian weight update (Dirichlet prior)
      3. SelfEvaluator health score
      4. Route action based on status
      5. Save version snapshot

    Usage:
        aw = AutoRiskWarning(etf_data, "equity")
        report = aw.weekly_update()
        prob = aw.predict("2024-01-04")
    """

    LAYER_KEYS = ["harrv", "climatology", "ewma", "current_vol"]

    def __init__(self, etf_data: pd.DataFrame, asset: str):
        self.rw = EnsembleRiskWarning(etf_data, asset)
        self.asset = asset
        self.evaluator = SelfEvaluator(self.rw)
        self.versions = VersionManager(asset)
        self._init_weights()

    def _init_weights(self):
        self.weights = {k: 0.25 for k in self.LAYER_KEYS}
        self.weight_uncertainty = {k: 0.5 for k in self.LAYER_KEYS}
        self.alpha = np.ones(len(self.LAYER_KEYS))

    def _compute_rolling_mse(self, key: str, window: int = 60) -> float:
        prob = self.rw._prob[key]
        yt = self.rw.y_true.reindex(prob.index).dropna()
        common = yt.index.intersection(prob.dropna().index)
        if len(common) < window:
            common = common
        else:
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
        detail = self.evaluator.detail()
        detail["weights"] = dict(self.weights)
        detail["weight_uncertainty"] = dict(self.weight_uncertainty)
        detail["alpha"] = self.alpha.tolist()
        self.versions.save(detail)
        return detail


# ── PlattCalibrator ─────────────────────────────────────────────

class PlattCalibrator:
    """Probability calibration via Platt scaling (logistic regression).

    Fits: P(y=1|prob) = 1 / (1 + exp(A * logit(prob) + B))
    where A, B are estimated from recent 60d predictions vs actuals.
    """

    def __init__(self, window: int = 60, min_samples: int = 20):
        self.window = window
        self.min_samples = min_samples
        self.A = 1.0
        self.B = 0.0
        self._fitted = False

    def fit(self, prob: pd.Series, y_true: pd.Series):
        common = prob.dropna().index.intersection(y_true.dropna().index)
        if len(common) < self.min_samples:
            return
        recent = common[-self.window:]
        x = prob.loc[recent].values
        y = y_true.loc[recent].values
        if len(np.unique(y)) < 2:
            return
        from sklearn.linear_model import LogisticRegression
        # Clip to avoid log(0) / log(1)
        eps = 1e-6
        x_clip = np.clip(x, eps, 1 - eps)
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x_clip.reshape(-1, 1), y)
        self.A = float(lr.coef_[0][0])
        self.B = float(lr.intercept_[0])
        self._fitted = True

    def transform(self, prob: pd.Series) -> pd.Series:
        if not self._fitted:
            return prob
        eps = 1e-6
        x = np.clip(prob.values, eps, 1 - eps)
        # logit → linear → sigmoid
        logit = np.log(x / (1 - x))
        calibrated = 1.0 / (1.0 + np.exp(-(self.A * logit + self.B)))
        return pd.Series(calibrated, index=prob.index)

    def fit_transform(self, prob: pd.Series, y_true: pd.Series) -> pd.Series:
        self.fit(prob, y_true)
        return self.transform(prob)


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


# ── Integrated AutoRiskWarning.weekly_update ────────────────────

# Patch: override weekly_update in AutoRiskWarning to include AutoOptimizer
def _auto_weekly_update(self) -> dict:
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

# Apply the patch
AutoRiskWarning.weekly_update = _auto_weekly_update

# Update __init__ to include optimizer
_orig_init = AutoRiskWarning.__init__
def _auto_init(self, etf_data: pd.DataFrame, asset: str):
    _orig_init(self, etf_data, asset)
    self.optimizer = AutoOptimizer(self.rw)
AutoRiskWarning.__init__ = _auto_init


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
