"""v10.3 Risk Warning — GARCH v3 addendum: HAR-RV Jump-Robust Model.

HAR-RV (Heterogeneous Autoregressive for Realized Volatility, Corsi 2009):
  - Daily, weekly, monthly volatility components
  - Captures long memory and jumps naturally
  - OLS estimation (simple, fast, no daily refit needed)
  - Known to outperform GARCH for multi-step volatility forecasting

Compared to GARCH:
  - Jump response: instantaneous (daily component), not gradual
  - Multi-step: direct, not via recursive forecast
  - Estimation: OLS vs MLE, much faster
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
    realized_vol = rets.rolling(cfg.vol_window).std().shift(-cfg.vol_window) * np.sqrt(252)
    threshold = realized_vol.rolling(cfg.lookback, min_periods=60).quantile(cfg.vol_percentile)
    y_true = (realized_vol > threshold).astype(float).dropna()
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
# HAR-RV Model
# ===========================================================

def run_harrv(rets: pd.Series, y_true: pd.Series, threshold: pd.Series,
              cfg: AssetConfig, min_train: int = 504) -> dict:
    """HAR-RV: forecast forward 20d vol using daily/weekly/monthly components.

    Features:
      - vol_1d: absolute return (daily vol proxy)
      - vol_5d: rolling 5d std (weekly)
      - vol_22d: rolling 22d std (monthly)
    Target: forward 20d realized vol (annualized)
    """
    from sklearn.linear_model import LinearRegression

    abs_ret = rets.abs() / 100  # daily vol proxy (decimal)
    vol_5d = rets.rolling(5).std() / 100  # weekly vol
    vol_22d = rets.rolling(22).std() / 100  # monthly vol
    fwd_vol = rets.rolling(cfg.vol_window).std().shift(-cfg.vol_window) / 100 * np.sqrt(252)

    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.index)

    # Refit every 20 days (HAR is stable, doesn't need daily refit)
    refit_dates = rets.index[rets.index.isin(oos_idx)][::20]

    forecasts = pd.Series(index=oos_idx, dtype=float)
    fit_times = []

    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < min_train:
            continue

        # Build training data: (target: fwd_vol, features: vol_1d, vol_5d, vol_22d)
        train_slice = slice(0, ri)
        X = pd.DataFrame({
            "vol_1d": abs_ret.iloc[train_slice],
            "vol_5d": vol_5d.iloc[train_slice],
            "vol_22d": vol_22d.iloc[train_slice],
            "const": 1.0,
        })
        y = fwd_vol.iloc[train_slice]

        # Drop NaN rows
        valid = X.notna().all(axis=1) & y.notna()
        X_clean = X.loc[valid].values
        y_clean = y.loc[valid].values

        try:
            model = LinearRegression(fit_intercept=False)
            model.fit(X_clean, y_clean)

            # Predict for next 20 days
            fs = rd + pd.Timedelta(days=1)
            fdates = pd.bdate_range(fs, periods=20)
            fdates = [d for d in fdates if d in forecasts.index]

            for fd in fdates:
                fi = rets.index.get_indexer([fd], method="ffill")[0]
                if fi < 1:
                    continue
                feat = np.array([abs_ret.iloc[fi - 1],
                                 vol_5d.iloc[fi - 1],
                                 vol_22d.iloc[fi - 1],
                                 1.0]).reshape(1, -1)
                if np.any(np.isnan(feat)):
                    continue
                forecasts[fd] = model.predict(feat)[0]

            fit_times.append(len(X_clean))

        except Exception as e:
            log(f"    HAR-RV fit failed at {rd.date()}: {e}")
            continue

    forecasts = forecasts.fillna(method="ffill")
    common = forecasts.dropna().index.intersection(y_true.index)
    fc = forecasts.loc[common]
    ya = y_true.loc[common]
    th = threshold.loc[common]
    prob = cont_prob(fc)
    bin_pred = (fc > th).astype(float)

    return {"key": "harrv", "label": "HAR-RV (jump-robust)",
            "forecast": fc, "prob": prob, "y_true": ya, "threshold": th,
            "binary": bin_pred, "n_fits": len(fit_times),
            "metrics_bin": evaluate(ya, bin_pred),
            "metrics_cont": evaluate(ya, prob),
            "n_samples": len(ya)}


# ===========================================================
# Run all
# ===========================================================

def run_asset(cfg: AssetConfig):
    log("=" * 60)
    log(f"HAR-RV: {cfg.name}")
    log("=" * 60)

    log("\n[1] Loading data...")
    etf_data = load_data()
    nav = compute_nav(etf_data, cfg)
    rets = nav.pct_change().dropna() * 100
    log(f"  Returns: {len(rets)}d, {rets.index[0].date()} ~ {rets.index[-1].date()}")

    log("\n[2] Computing y_true...")
    y_true, realized_vol, threshold = compute_y_true(rets / 100, cfg)
    log(f"  y_true: {y_true.mean():.1%} positive ({int(y_true.sum())}/{len(y_true)})")

    log("\n[3] Running HAR-RV...")
    t0 = time.time()
    r = run_harrv(rets, y_true, threshold, cfg)
    et = time.time() - t0
    log(f"  AUC(bin): {r['metrics_bin']['auc']:.4f}  AUC(cont): {r['metrics_cont']['auc']:.4f}")
    log(f"  PR-AUC(cont): {r['metrics_cont']['prauc']:.4f}  Brier(cont): {r['metrics_cont']['brier']:.4f}")
    log(f"  Fits: {r['n_fits']}  Time: {et:.0f}s  N: {r['n_samples']}")

    log(f"\n{'=' * 60}")
    log(f"{'Model':<35} {'AUC(cont)':<10} {'PR-AUC':<10} {'Brier':<10}")
    log(f"{'-' * 60}")
    log(f"{r['label']:<35} {r['metrics_cont']['auc']:<10.4f} "
        f"{r['metrics_cont']['prauc']:<10.4f} {r['metrics_cont']['brier']:<10.4f}")
    log(f"{'=' * 60}")

    # Save
    pd.DataFrame([{
        "asset": cfg.name, "model": r["key"],
        "auc_cont": r["metrics_cont"]["auc"],
        "prauc_cont": r["metrics_cont"]["prauc"],
        "brier_cont": r["metrics_cont"]["brier"],
        "n_fits": r["n_fits"],
    }]).to_csv(OUT_DIR / f"harrv_{cfg.name}_summary.csv", index=False)
    log(f"\nSaved: {OUT_DIR / f'harrv_{cfg.name}_summary.csv'}")

    df = pd.DataFrame({
        "y_true": r["y_true"], "threshold": r["threshold"],
        "forecast": r["forecast"], "prob": r["prob"],
    })
    df.to_parquet(OUT_DIR / f"harrv_{cfg.name}_forecasts.parquet")
    log(f"Saved: {OUT_DIR / f'harrv_{cfg.name}_forecasts.parquet'}")

    return r


if __name__ == "__main__":
    log("=" * 60)
    log("HAR-RV Jump-Robust Volatility Model")
    log("=" * 60)
    for name in ["equity", "commodity", "gold"]:
        run_asset(CONFIGS[name])
        log("")
    log("[Done]")
