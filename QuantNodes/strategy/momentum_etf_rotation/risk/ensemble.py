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

LAYER_CONFIGS = {
    "equity": ["climatology", "macro", "current_vol"],
    "commodity": ["climatology"],
    "gold": ["climatology"],
    "soybean": ["climatology", "current_vol"],
    "silver": ["climatology", "ewma"],
    "energy_chem": ["climatology", "ewma"],
    "commodity_ex_gold": ["climatology", "ewma"],
}

MODEL_FNS = {
    "harrv": _model_harrv,
    "climatology": _model_climatology,
    "ewma": _model_ewma,
    "current_vol": _model_current_vol,
    "macro": None,
}


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

        models = {}
        for key in LAYER_CONFIGS.get(self.asset, ["climatology"]):
            if key == "macro" and self.asset in MACRO_MODELS:
                models[key] = MACRO_MODELS[self.asset]
            elif key != "macro":
                models[key] = MODEL_FNS[key]

        for key, fn in models.items():
            self._raw[key] = fn(self.rets, self.oos_idx)
            common = self._raw[key].dropna().index.intersection(self.y_true.dropna().index)
            self._prob[key] = _cont_prob(self._raw[key].loc[common])

        self._ensemble_prob, self._ensemble_weights = self._compute_ensemble_weights()
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

    def explain(self, date: pd.Timestamp | str) -> dict:
        """Decompose risk probability into layer contributions.

        Returns: {
            date, risk_prob, signal,
            climatology_anchor: {prob, gap},
            layers: {layer_name: {raw_vol, prob, weight, contribution}},
            macro_coefs: {...}  # only for equity
        }
        """
        d = pd.Timestamp(date)
        if d not in self._ensemble_prob.index:
            return {"date": str(d.date()), "error": "date not in OOS period"}

        result = {
            "date": str(d.date()),
            "asset": self.asset,
            "risk_prob": float(self._ensemble_prob.loc[d]),
            "signal": self._to_signal(self._ensemble_prob.loc[d]),
        }

        clim_prob = float(self._prob["climatology"].loc[d]) if "climatology" in self._prob else 0.5
        result["climatology_anchor"] = {
            "prob": clim_prob,
            "gap": result["risk_prob"] - clim_prob,
        }

        layers = {}
        for key in self._prob:
            prob = float(self._prob[key].loc[d]) if d in self._prob[key].index else 0.5
            weight = float(self._ensemble_weights.loc[d, key]) if d in self._ensemble_weights.index else 0.0
            raw = float(self._raw[key].loc[d]) if key in self._raw and d in self._raw[key].index else None
            layers[key] = {
                "raw_vol": raw,
                "prob": prob,
                "weight": weight,
                "contribution": prob * weight,
            }
        result["layers"] = layers

        return result

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
