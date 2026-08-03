"""v10.3 Risk Warning — GARCH v3: Gold standalone + Jump-Augmented RS-GARCH.

Two new models:
  1. Gold standalone: GJR-GARCH(1,1,1) Student-t on 518880
  2. RS-GARCH (Regime-Switching): HMM identifies low/high vol regimes,
     regime-specific GARCH + probability-weighted forecasts

Addresses: GARCH reacts slowly to jumps — regime switching lets the model
immediately shift to high-vol parameters when a jump occurs.
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
# Model 1: Gold standalone — GJR-GARCH(1,1,1) Student-t
# ===========================================================

def run_gold_garch(rets: pd.Series, y_true: pd.Series, threshold: pd.Series) -> dict:
    from arch import arch_model
    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.index)
    refit_dates = rets.index[rets.index.isin(oos_idx)]

    forecasts = pd.Series(index=oos_idx, dtype=float)
    fit_times = []

    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        train = rets.iloc[:ri]
        try:
            m = arch_model(train, vol="Garch", p=1, o=1, q=1, dist="studentst", rescale=False)
            res = m.fit(disp="off", show_warning=False)
            fwd = res.forecast(horizon=20)
            fwd_vol = np.sqrt(fwd.variance.iloc[-1].values) * np.sqrt(252) / 100
            fs = rd + pd.Timedelta(days=1)
            fdates = [d for d in pd.bdate_range(fs, periods=20) if d in forecasts.index]
            for j, fd in enumerate(fdates):
                if j < len(fwd_vol):
                    forecasts[fd] = fwd_vol[j]
            fit_times.append(len(train))
        except Exception:
            continue

    forecasts = forecasts.fillna(method="ffill")
    common = forecasts.dropna().index.intersection(y_true.index)
    fc = forecasts.loc[common]
    ya = y_true.loc[common]
    th = threshold.loc[common]
    prob = cont_prob(fc)
    bin_pred = (fc > th).astype(float)
    return {"key": "gold_gjrgarch_t", "label": "Gold GJR-GARCH(1,1,1) t",
            "forecast": fc, "prob": prob, "y_true": ya, "threshold": th,
            "binary": bin_pred, "n_fits": len(fit_times),
            "metrics_bin": evaluate(ya, bin_pred),
            "metrics_cont": evaluate(ya, prob)}


# ===========================================================
# Model 2: RS-GARCH — Regime-Switching via HMM
# ===========================================================

def run_rs_garch(rets: pd.Series, y_true: pd.Series, threshold: pd.Series) -> dict:
    from hmmlearn import hmm
    from arch import arch_model

    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.index)
    refit_dates = rets.index[rets.index.isin(oos_idx)]

    forecasts = pd.Series(index=oos_idx, dtype=float)
    regime_probs = pd.Series(index=oos_idx, dtype=float)
    fit_times = []

    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        train = rets.iloc[:ri]

        try:
            train_2d = train.values.reshape(-1, 1)
            hmm_model = hmm.GaussianHMM(n_components=2, covariance_type="diag",
                                         random_state=42, n_iter=100, tol=1e-3)
            hmm_model.fit(train_2d)
            states = hmm_model.predict(train_2d)
            state_probs = hmm_model.predict_proba(train_2d)

            means = hmm_model.means_.flatten()
            vol_state = np.argmax(np.abs(means))
            low_vol_state = 1 - vol_state
            high_vol_prob = state_probs[:, vol_state]

            high_mask = states == vol_state
            low_mask = states == low_vol_state

            garch_high = None
            garch_low = None

            if high_mask.sum() > 60:
                try:
                    mh = arch_model(train[high_mask].values, vol="Garch", p=1, o=1, q=1,
                                    dist="studentst", rescale=False)
                    garch_high = mh.fit(disp="off", show_warning=False)
                except Exception:
                    pass

            if low_mask.sum() > 60:
                try:
                    ml = arch_model(train[low_mask].values, vol="Garch", p=1, o=1, q=1,
                                    dist="studentst", rescale=False)
                    garch_low = ml.fit(disp="off", show_warning=False)
                except Exception:
                    pass

            if garch_high is not None and garch_low is not None:
                fwd_h = garch_high.forecast(horizon=20)
                fwd_l = garch_low.forecast(horizon=20)
                vol_h = np.sqrt(fwd_h.variance.iloc[-1].values) * np.sqrt(252) / 100
                vol_l = np.sqrt(fwd_l.variance.iloc[-1].values) * np.sqrt(252) / 100
                w_h = high_vol_prob[-1]
                fwd_vol = w_h * vol_h + (1 - w_h) * vol_l
            elif garch_high is not None:
                fwd = garch_high.forecast(horizon=20)
                fwd_vol = np.sqrt(fwd.variance.iloc[-1].values) * np.sqrt(252) / 100
            elif garch_low is not None:
                fwd = garch_low.forecast(horizon=20)
                fwd_vol = np.sqrt(fwd.variance.iloc[-1].values) * np.sqrt(252) / 100
            else:
                continue

            fs = rd + pd.Timedelta(days=1)
            fdates = [d for d in pd.bdate_range(fs, periods=20) if d in forecasts.index]
            for j, fd in enumerate(fdates):
                if j < len(fwd_vol):
                    forecasts[fd] = fwd_vol[j]
                    regime_probs[fd] = high_vol_prob[-1]

            fit_times.append(len(train))

        except Exception as e:
            log(f"    RS-GARCH fit failed at {rd.date()}: {e}")
            continue

    forecasts = forecasts.fillna(method="ffill")
    regime_probs = regime_probs.fillna(method="ffill")
    common = forecasts.dropna().index.intersection(y_true.index)
    fc = forecasts.loc[common]
    ya = y_true.loc[common]
    th = threshold.loc[common]
    prob = cont_prob(fc)
    bin_pred = (fc > th).astype(float)
    return {"key": "rs_gjrgarch_t", "label": "RS-GARCH(2) GJR-GARCH t",
            "forecast": fc, "prob": prob, "y_true": ya, "threshold": th,
            "binary": bin_pred, "regime_prob": regime_probs.loc[common],
            "n_fits": len(fit_times),
            "metrics_bin": evaluate(ya, bin_pred),
            "metrics_cont": evaluate(ya, prob)}


# ===========================================================
# Run all
# ===========================================================

def run_asset(cfg: AssetConfig, run_gold: bool = False, run_rs: bool = False):
    lab = f"{cfg.name} (gold standalone)" if run_gold else cfg.name
    log("=" * 60)
    log(f"GARCH v3: {lab}")
    log("=" * 60)

    log("\n[1] Loading data...")
    etf_data = load_data()
    nav = compute_nav(etf_data, cfg)
    rets = nav.pct_change().dropna() * 100
    log(f"  Returns: {len(rets)}d, {rets.index[0].date()} ~ {rets.index[-1].date()}")

    log("\n[2] Computing y_true...")
    y_true, realized_vol, threshold = compute_y_true(rets / 100, cfg)
    log(f"  y_true: {y_true.mean():.1%} positive ({int(y_true.sum())}/{len(y_true)})")

    results = []

    if run_gold:
        log("\n[3] Gold standalone GJR-GARCH...")
        t0 = time.time()
        r = run_gold_garch(rets, y_true, threshold)
        et = time.time() - t0
        log(f"  AUC(bin): {r['metrics_bin']['auc']:.4f}  AUC(cont): {r['metrics_cont']['auc']:.4f}")
        log(f"  PR-AUC(cont): {r['metrics_cont']['prauc']:.4f}  Brier(cont): {r['metrics_cont']['brier']:.4f}")
        log(f"  Fits: {r['n_fits']}  Time: {et:.0f}s")
        results.append(r)

    if run_rs:
        log("\n[4] RS-GARCH (Regime-Switching)...")
        t0 = time.time()
        r = run_rs_garch(rets, y_true, threshold)
        et = time.time() - t0
        log(f"  AUC(bin): {r['metrics_bin']['auc']:.4f}  AUC(cont): {r['metrics_cont']['auc']:.4f}")
        log(f"  PR-AUC(cont): {r['metrics_cont']['prauc']:.4f}  Brier(cont): {r['metrics_cont']['brier']:.4f}")
        log(f"  Fits: {r['n_fits']}  Time: {et:.0f}s")
        results.append(r)

    if results:
        log(f"\n{'=' * 70}")
        log(f"{'Model':<35} {'AUC(bin)':<10} {'AUC(cont)':<10} {'PR-AUC':<10} {'Brier':<10} {'Fits':<6}")
        log(f"{'-' * 70}")
        for r in results:
            log(f"{r['label']:<35} {r['metrics_bin']['auc']:<10.4f} {r['metrics_cont']['auc']:<10.4f} "
                f"{r['metrics_cont']['prauc']:<10.4f} {r['metrics_cont']['brier']:<10.4f} {r['n_fits']:<6}")
        log(f"{'=' * 70}")

        rows = []
        for r in results:
            rows.append({
                "asset": cfg.name, "model": r["key"],
                "auc_binary": r["metrics_bin"]["auc"],
                "auc_cont": r["metrics_cont"]["auc"],
                "prauc_cont": r["metrics_cont"]["prauc"],
                "brier_cont": r["metrics_cont"]["brier"],
                "n_fits": r["n_fits"],
            })
        pd.DataFrame(rows).to_csv(OUT_DIR / f"garch_{cfg.name}_v3_summary.csv", index=False)
        log(f"\nSaved: {OUT_DIR / f'garch_{cfg.name}_v3_summary.csv'}")

        df = pd.DataFrame({"y_true": results[0]["y_true"], "threshold": results[0]["threshold"]})
        for r in results:
            df[f"forecast_{r['key']}"] = r["forecast"]
            df[f"prob_{r['key']}"] = r["prob"]
        if run_rs and "regime_prob" in results[-1]:
            df["hmm_high_vol_prob"] = results[-1]["regime_prob"]
        df.to_parquet(OUT_DIR / f"garch_{cfg.name}_v3_forecasts.parquet")
        log(f"Saved: {OUT_DIR / f'garch_{cfg.name}_v3_forecasts.parquet'}")

    return results


if __name__ == "__main__":
    log("=" * 60)
    log("GARCH Volatility Forecasting Model v3")
    log("=" * 60)

    log("\n--- Gold standalone ---")
    run_asset(CONFIGS["gold"], run_gold=True)

    log("\n--- Equity RS-GARCH ---")
    run_asset(CONFIGS["equity"], run_rs=True)

    log("\n--- Commodity RS-GARCH ---")
    run_asset(CONFIGS["commodity"], run_rs=True)

    log("\n[Done]")
