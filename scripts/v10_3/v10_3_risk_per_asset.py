"""v10.3 Risk Warning — Per-Asset-Class Risk Model.

Predict forward 20d vol > 80th percentile for each asset class.
Model: LightGBM binary classifier. Validation: Walk-forward (5 folds, TSCV).
See v10_3_risk_per_asset_design.md for full design doc.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
HF_DIR = REPO / "data" / "high_freq_macro"
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v10_3"
OUT_DIR.mkdir(exist_ok=True)
OOS_START = "2022-01-04"
OOS_END = "2026-05-29"
log = print


@dataclass
class AssetClassConfig:
    name: str
    etfs: list[str]
    weights: list[float] | None = None
    vol_window: int = 20
    vol_percentile: float = 0.80
    lookback: int = 252


EQUITY_CFG = AssetClassConfig(
    name="equity", etfs=["510300"],
    vol_window=20, vol_percentile=0.80, lookback=252,
)

COMMODITY_CFG = AssetClassConfig(
    name="commodity", etfs=["518880", "159985", "161226", "159981"],
    weights=[0.50, 0.20, 0.15, 0.15],
    vol_window=20, vol_percentile=0.80, lookback=252,
)


def load_etf_data() -> pd.DataFrame:
    df = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_macro_data() -> dict:
    vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]
    vix.index = pd.to_datetime(vix.index)
    v9 = pd.read_parquet(HF_DIR / "v9_factors_weekly.parquet")
    v9.index = pd.to_datetime(v9.index)
    v9 = v9.resample("D").ffill()
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
    dxy.index = pd.to_datetime(dxy.index)
    rr = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")["real_rate"]
    rr.index = pd.to_datetime(rr.index)
    go = pd.read_parquet(HF_DIR / "gold_oil_corr.parquet")
    go["date"] = pd.to_datetime(go["date"])
    go = go.set_index("date")["gold_oil_corr"]
    cs = pd.read_parquet(HF_DIR / "cn_us_spread_10y.parquet")
    cs["date"] = pd.to_datetime(cs["date"])
    cs = cs.set_index("date")["cn_us_spread"]
    return {"vix": vix, "v9": v9, "dxy": dxy, "real_rate": rr,
            "gold_oil": go, "cn_us": cs}


def compute_asset_nav(etf_data: pd.DataFrame, cfg: AssetClassConfig) -> pd.Series:
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


def build_target(nav: pd.Series, cfg: AssetClassConfig) -> pd.Series:
    rets = nav.pct_change().dropna()
    fwd_vol = rets.rolling(cfg.vol_window).std().shift(-cfg.vol_window) * np.sqrt(252)
    thresh = fwd_vol.rolling(cfg.lookback, min_periods=60).quantile(cfg.vol_percentile)
    y = (fwd_vol > thresh).astype(float).dropna()
    log(f"  Target: fwd{cfg.vol_window}d vol > {cfg.vol_percentile:.0%}ile")
    log(f"  Pos rate: {y.mean():.1%} ({int(y.sum())}/{len(y)})")
    return y


def build_features(nav: pd.Series, macro: dict) -> pd.DataFrame:
    rets = nav.pct_change().dropna()
    idx = rets.index
    f = pd.DataFrame(index=idx)

    f["ret_1d"] = rets
    f["ret_5d"] = rets.rolling(5).sum()
    f["ret_20d"] = rets.rolling(20).sum()
    f["ret_60d"] = rets.rolling(60).sum()
    f["ret_12m"] = (1 + rets).rolling(252).apply(lambda x: x.prod() - 1, raw=True)
    f["vol_5d"] = rets.rolling(5).std() * np.sqrt(252)
    f["vol_20d"] = rets.rolling(20).std() * np.sqrt(252)
    f["vol_60d"] = rets.rolling(60).std() * np.sqrt(252)
    f["skew_60d"] = rets.rolling(60).skew()
    f["kurt_60d"] = rets.rolling(60).kurt()
    f["max_dd_60d"] = rets.rolling(60).apply(lambda x: (1+x).cumprod().min()-1, raw=True)
    f["drawdown"] = nav / nav.cummax() - 1
    f["dist_200ma"] = nav / nav.rolling(200).mean() - 1

    vix = macro["vix"].reindex(idx).ffill()
    f["vix"] = vix
    f["vix_zscore"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    f["vix_ratio_ma20"] = vix / vix.rolling(20).mean()
    f["vix_ratio_ma60"] = vix / vix.rolling(60).mean()
    for th in [20, 25, 30]:
        f[f"vix_above_{th}"] = (vix > th).astype(float)

    v9 = macro["v9"].reindex(idx).ffill()
    for col in v9.columns:
        s = v9[col]
        f[f"macro_{col}"] = s
        f[f"macro_{col}_zscore"] = ((s - s.rolling(65).mean()) / s.rolling(65).std())
        f[f"macro_{col}_chg"] = s.diff(5)

    dxy = macro["dxy"].reindex(idx).ffill()
    f["dxy"] = dxy
    f["dxy_zscore"] = (dxy - dxy.rolling(252).mean()) / dxy.rolling(252).std()
    f["dxy_logret"] = np.log(dxy / dxy.shift(1))

    rr = macro["real_rate"].reindex(idx).ffill()
    f["real_rate"] = rr
    f["real_rate_zscore"] = (rr - rr.rolling(252).mean()) / rr.rolling(252).std()

    go = macro["gold_oil"].reindex(idx).ffill()
    f["gold_oil_corr"] = go

    cs = macro["cn_us"].reindex(idx).ffill()
    f["cn_us_spread"] = cs
    f["cn_us_spread_zscore"] = (cs - cs.rolling(252).mean()) / cs.rolling(252).std()

    f["vix_x_ret_12m"] = f["vix"] * f["ret_12m"]
    f["vix_x_vol_20d"] = f["vix"] * f["vol_20d"]
    f["dxy_x_ret_20d"] = f["dxy"] * f["ret_20d"]

    return f


def select_features(X: pd.DataFrame, y: pd.Series, min_auc: float = 0.52) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    X_fill = X.fillna(X.median())
    aucs = []
    for col in X_fill.columns:
        try:
            auc = roc_auc_score(y, X_fill[col])
            if auc >= min_auc:
                aucs.append((col, auc))
        except Exception:
            pass
    aucs.sort(key=lambda x: x[1], reverse=True)
    sel = [c for c, _ in aucs]
    if len(sel) > 1:
        corr = X_fill[sel].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
        sel = [c for c in sel if c not in to_drop]
    log(f"  Feature selection: {len(X.columns)} -> {len(sel)} (AUC>{min_auc}, de-correlated)")
    return X_fill[sel]


def train_optuna_tscv(X, y, n_trials=20):
    import optuna, lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit

    scale = (len(y) - y.sum()) / y.sum()
    tscv = TimeSeriesSplit(n_splits=4)

    def objective(trial):
        params = {
            "objective": "binary", "metric": "auc", "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("lr", 0.01, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "scale_pos_weight": scale, "verbose": -1,
        }
        aucs = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            if y_val.nunique() < 2:
                continue
            try:
                model = lgb.LGBMClassifier(**params, n_estimators=200, n_jobs=-1)
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                y_prob = model.predict_proba(X_val)[:, 1]
                from sklearn.metrics import roc_auc_score
                aucs.append(roc_auc_score(y_val, y_prob))
            except Exception:
                continue
        return np.mean(aucs) if aucs else 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log(f"  Optuna TSCV best AUC: {study.best_value:.4f}")
    return study.best_params


def train_model(X, y, params):
    import lightgbm as lgb
    from sklearn.calibration import CalibratedClassifierCV
    p = params.copy()
    p.update({"verbose": -1, "n_estimators": 200, "n_jobs": -1})
    base = lgb.LGBMClassifier(**p)
    base.fit(X, y)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X, y)
    return model


def evaluate(y_true, y_prob):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "PosRate": y_true.mean(), "N": len(y_true),
    }


def walk_forward_split(X):
    dates = X.index
    ts = dates[0]
    ms = pd.Timestamp(OOS_START)
    splits = []
    for _ in range(5):
        te = ms + pd.Timedelta(days=365)
        if te > dates[-1]:
            te = dates[-1]
        tr_end = ms - pd.Timedelta(days=10)
        if tr_end <= ts:
            break
        tr = dates[(dates >= ts) & (dates <= tr_end)]
        te_idx = dates[(dates >= ms) & (dates <= te)]
        if len(tr) > 252 and len(te_idx) > 20:
            splits.append((tr, te_idx))
        ms = te + pd.Timedelta(days=1)
    return splits


def run_asset_model(cfg: AssetClassConfig):
    log("=" * 60)
    log(f"Asset Risk Model: {cfg.name}")
    log(f"  ETFs: {cfg.etfs}")
    if cfg.weights:
        log(f"  Weights: {dict(zip(cfg.etfs, cfg.weights))}")
    log("=" * 60)

    log("\n[1] Loading data...")
    etf_data = load_etf_data()
    macro = load_macro_data()
    nav = compute_asset_nav(etf_data, cfg)
    log(f"  NAV: {len(nav)}d, {nav.index[0].date()} ~ {nav.index[-1].date()}")

    log("\n[2] Building target (fwd 20d vol > 80%ile)...")
    y = build_target(nav, cfg)

    log("\n[3] Building features (55 cols)...")
    features = build_features(nav, macro)
    log(f"  Features: {features.shape[1]} cols")

    idx = features.index.intersection(y.index)
    X, y = features.loc[idx], y.loc[idx]
    log(f"  Aligned: {len(X)} samples")

    log("\n[4] Walk-forward training...")
    splits = walk_forward_split(X)
    log(f"  {len(splits)} folds:")
    for i, (tr, te) in enumerate(splits):
        log(f"    Fold {i+1}: train {len(tr)}d ({tr[0].date()}~{tr[-1].date()}), "
            f"test {len(te)}d ({te[0].date()}~{te[-1].date()}), "
            f"y_pos={y.loc[te].mean():.1%}")

    wf_metrics = []
    for i, (tr_idx, te_idx) in enumerate(splits):
        log(f"\n  === Fold {i+1} ===")

        X_tr_all = X.loc[tr_idx]
        y_tr = y.loc[tr_idx]
        X_sel = select_features(X_tr_all, y_tr)

        X_te = X.loc[te_idx]
        X_te = X_te[X_sel.columns].fillna(X_sel.median())
        y_te = y.loc[te_idx]

        if i == 0:
            best_params = train_optuna_tscv(X_sel, y_tr, n_trials=20)
            log(f"  Params: {best_params}")

        model = train_model(X_sel, y_tr, best_params)
        y_prob = model.predict_proba(X_te)[:, 1]
        m = evaluate(y_te, y_prob)
        m["Fold"] = i + 1
        m["N_features"] = len(X_sel.columns)
        wf_metrics.append(m)
        log(f"  Result: AUC={m['AUC']:.4f} PR-AUC={m['PR-AUC']:.4f} "
            f"Brier={m['Brier']:.4f} ({m['N_features']} features)")

    mf = pd.DataFrame(wf_metrics)
    log(f"\n  === Walk-forward summary ===")
    log(f"  AUC: {mf['AUC'].mean():.4f} +/- {mf['AUC'].std():.4f}")
    log(f"  PR-AUC: {mf['PR-AUC'].mean():.4f} +/- {mf['PR-AUC'].std():.4f}")
    log(f"  Brier: {mf['Brier'].mean():.4f} +/- {mf['Brier'].std():.4f}")
    mf.to_csv(OUT_DIR / f"risk_{cfg.name}_wf_metrics.csv", index=False)

    import joblib
    best_wf_fold = mf["AUC"].idxmax()
    best_model = train_model(X_sel, y_tr, best_params)
    joblib.dump({"model": best_model, "features": list(X_sel.columns)},
                OUT_DIR / f"risk_{cfg.name}_model.pkl")
    log(f"  Saved model: {OUT_DIR / f'risk_{cfg.name}_model.pkl'}")

    oos_pred = pd.DataFrame({"y_true": y_te, "y_prob": y_prob}, index=X_te.index)
    oos_pred.to_parquet(OUT_DIR / f"risk_{cfg.name}_oos_pred.parquet")
    log(f"  Saved predictions: {OUT_DIR / f'risk_{cfg.name}_oos_pred.parquet'}")


if __name__ == "__main__":
    log("=" * 60)
    log("Running per-asset-class risk models")
    log("=" * 60)
    for cfg in [EQUITY_CFG, COMMODITY_CFG]:
        run_asset_model(cfg)
        log("\n")
    log("[Done]")
