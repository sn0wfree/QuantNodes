"""v10.3 Risk Warning — Ensemble Volatility Forecasting Model.

Industry-standard ensemble of 4 models:
  1. HAR-RV (Corsi 2009) — OLS, daily/weekly/monthly components
  2. Climatology (DOY mean) — calendar effect
  3. EWMA (RiskMetrics, lambda=0.94) — fast jump response
  4. Current Vol (20d trailing) — baseline

Ensemble: dynamic weights based on rolling 60d MSE (Bayesian averaging).
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v10_3"
OUT_DIR.mkdir(exist_ok=True)
OOS_START = "2022-01-04"
OOS_END = "2026-05-29"
log = print

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
}


def load_data() -> pd.DataFrame:
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


def evaluate(y_true: pd.Series, y_prob: pd.Series) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    common = y_true.dropna().index.intersection(y_prob.dropna().index)
    yt, yp = y_true.loc[common], y_prob.loc[common]
    if yt.nunique() < 2:
        return {"auc": 0.5, "prauc": yt.mean(), "brier": 0.25, "pos_rate": yt.mean(), "n": len(yt)}
    return {"auc": roc_auc_score(yt, yp), "prauc": average_precision_score(yt, yp),
            "brier": brier_score_loss(yt, yp), "pos_rate": yt.mean(), "n": len(yt)}


def cont_prob(forecast: pd.Series, window: int = 252) -> pd.Series:
    prob = pd.Series(index=forecast.index, dtype=float)
    for i in range(len(forecast)):
        d = forecast.index[i]
        hist = forecast.loc[:d].rolling(window, min_periods=60).rank(pct=True)
        prob.iloc[i] = hist.loc[d]
    return prob

# ===========================================================
# Model 1: HAR-RV
# ===========================================================

def model_harrv(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
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


# ===========================================================
# Model 2: Climatology (DOY)
# ===========================================================

def model_climatology(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
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


# ===========================================================
# Model 3: EWMA (RiskMetrics)
# ===========================================================

def model_ewma(rets: pd.Series, oos_idx: pd.Index, lam: float = 0.94) -> pd.Series:
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


# ===========================================================
# Model 4: Current Vol (20d trailing)
# ===========================================================

def model_current_vol(rets: pd.Series, oos_idx: pd.Index) -> pd.Series:
    vol = rets.rolling(20).std() / 100 * np.sqrt(252)
    forecasts = vol.reindex(oos_idx).fillna(method="ffill")
    return forecasts

# ===========================================================
# Ensemble: Dynamic Weighted Average
# ===========================================================

def compute_ensemble(probs: dict[str, pd.Series], y_true: pd.Series,
                     window: int = 60) -> tuple[pd.Series, pd.DataFrame]:
    """Dynamic weights based on rolling window MSE of probability forecasts.

    Weight = (1 / MSE_i) / sum(1 / MSE_j) for each model i.
    MSE computed over last `window` days against binary y_true.
    """
    keys = list(probs.keys())
    common = y_true.dropna().index.intersection(probs[keys[0]].dropna().index)
    weights = pd.DataFrame(index=common, columns=keys, dtype=float)
    ensemble = pd.Series(index=common, dtype=float)

    for i, d in enumerate(common):
        if i < window:
            for k in keys:
                weights.loc[d, k] = 1.0 / len(keys)
            ensemble.loc[d] = sum(probs[k].loc[d] * weights.loc[d, k] for k in keys)
            continue

        hist_idx = y_true.loc[:d].index[-window:]
        mse = {}
        for k in keys:
            err = probs[k].reindex(hist_idx) - y_true.reindex(hist_idx)
            mse[k] = (err**2).mean()

        inv_mse = {k: 1.0 / max(v, 1e-10) for k, v in mse.items()}
        total = sum(inv_mse.values())
        for k in keys:
            weights.loc[d, k] = inv_mse[k] / total
        ensemble.loc[d] = sum(probs[k].loc[d] * weights.loc[d, k] for k in keys)

    return ensemble, weights


# ===========================================================
# Run all
# ===========================================================

def run_asset(cfg: AssetConfig):
    log("=" * 60)
    log(f"Ensemble Model: {cfg.name}")
    log("=" * 60)

    log("\n[1] Loading data...")
    etf_data = load_data()
    nav = compute_nav(etf_data, cfg)
    rets = nav.pct_change().dropna() * 100
    log(f"  Returns: {len(rets)}d, {rets.index[0].date()} ~ {rets.index[-1].date()}")

    log("\n[2] Computing y_true...")
    y_true, realized_vol, threshold = compute_y_true(rets / 100, cfg)
    log(f"  y_true: {y_true.mean():.1%} positive ({int(y_true.sum())}/{len(y_true)})")

    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.dropna().index)

    log("\n[3] Running individual models...")
    models = {
        "harrv": ("HAR-RV", lambda: model_harrv(rets, oos_idx)),
        "climatology": ("DOY Climatology", lambda: model_climatology(rets, oos_idx)),
        "ewma": ("EWMA(0.94)", lambda: model_ewma(rets, oos_idx)),
        "current_vol": ("Current 20d Vol", lambda: model_current_vol(rets, oos_idx)),
    }

    forecasts = {}
    results = []
    models_idx = {}
    for idx, (key, (label, fn)) in enumerate(models.items()):
        t0 = time.time()
        fc = fn()
        elapsed = time.time() - t0
        forecasts[key] = fc
        common = fc.dropna().index.intersection(y_true.dropna().index)
        prob = cont_prob(fc.loc[common])
        ya = y_true.loc[common]
        th = threshold.loc[common]
        bin_pred = (fc.loc[common] > th).astype(float)
        m = evaluate(ya, prob)
        mb = evaluate(ya, bin_pred)
        log(f"  {label:<20} AUC={m['auc']:.4f}  PR-AUC={m['prauc']:.4f}  "
            f"Brier={m['brier']:.4f}  ({elapsed:.1f}s)")
        results.append({"key": key, "label": label, "forecast": fc.loc[common],
                        "prob": prob, "y_true": ya, "threshold": th,
                        "metrics": m, "metrics_bin": mb})
        models_idx[key] = idx

    log("\n[4] Computing ensemble...")
    t0 = time.time()
    # Use probabilities for ensemble weighting
    ensemble_probs, ensemble_weights = compute_ensemble(
        {k: results[models_idx[k]]["prob"]
         for k in ["harrv", "climatology", "ewma", "current_vol"]},
        y_true)
    elapsed = time.time() - t0
    common = ensemble_probs.dropna().index.intersection(y_true.dropna().index)
    prob = ensemble_probs.loc[common]
    ya = y_true.loc[common]
    th = threshold.loc[common]
    bin_pred = (prob > 0.5).astype(float)
    m = evaluate(ya, prob)
    mb = evaluate(ya, bin_pred)
    log(f"  Ensemble (dynamic MSE)        AUC={m['auc']:.4f}  PR-AUC={m['prauc']:.4f}  "
        f"Brier={m['brier']:.4f}  ({elapsed:.1f}s)")
    results.append({"key": "ensemble", "label": "Ensemble (MSE weights)",
                    "forecast": prob, "prob": prob,
                    "y_true": ya, "threshold": th,
                    "metrics": m, "metrics_bin": mb})

    # Comparison table
    log(f"\n{'=' * 80}")
    log(f"{'Model':<25} {'AUC(cont)':<10} {'PR-AUC':<10} {'Brier':<10} {'AUC(bin)':<10}")
    log(f"{'-' * 80}")
    for r in sorted(results, key=lambda x: x["metrics"]["auc"], reverse=True):
        log(f"{r['label']:<25} {r['metrics']['auc']:<10.4f} {r['metrics']['prauc']:<10.4f} "
            f"{r['metrics']['brier']:<10.4f} {r['metrics_bin']['auc']:<10.4f}")
    log(f"{'=' * 80}")

    # Save summary
    rows = [{"asset": cfg.name, "model": r["key"], "label": r["label"],
             "auc_cont": r["metrics"]["auc"], "prauc_cont": r["metrics"]["prauc"],
             "brier_cont": r["metrics"]["brier"], "auc_binary": r["metrics_bin"]["auc"]}
            for r in results]
    pd.DataFrame(rows).to_csv(OUT_DIR / f"ensemble_{cfg.name}_summary.csv", index=False)
    log(f"\nSaved: {OUT_DIR / f'ensemble_{cfg.name}_summary.csv'}")

    # Save forecasts
    zero = results[0]
    df = pd.DataFrame({"y_true": zero["y_true"], "threshold": zero["threshold"]})
    for r in results:
        df[f"vol_{r['key']}"] = r["forecast"]
        df[f"prob_{r['key']}"] = r["prob"]
    df["weight_harrv"] = ensemble_weights.iloc[:, 0]
    df["weight_climatology"] = ensemble_weights.iloc[:, 1]
    df["weight_ewma"] = ensemble_weights.iloc[:, 2]
    df["weight_vol"] = ensemble_weights.iloc[:, 3]
    df.to_parquet(OUT_DIR / f"ensemble_{cfg.name}_forecasts.parquet")
    log(f"Saved: {OUT_DIR / f'ensemble_{cfg.name}_forecasts.parquet'}")

    return results


if __name__ == "__main__":
    log("=" * 60)
    log("Ensemble Volatility Forecasting Model")
    log("=" * 60)
    for name in ["equity", "commodity", "gold"]:
        run_asset(CONFIGS[name])
        log("")
    log("[Done]")
