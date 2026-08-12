"""Gold Model 3-Way Comparison: BayesianRidge+Q vs Conformal+EN vs Bootstrap+EN

Three uncertainty-aware models compared:
1. BayesianRidge + Quantile Regression (current)
2. Conformal Prediction + ElasticNet
3. Bootstrap Aggregation + ElasticNet
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.quantile_regression import QuantReg

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk.ensemble import (
    _model_climatology, _cont_prob,
    load_etf_data, CONFIGS, compute_y_true, evaluate, OOS_START, OOS_END
)


# ── Factor Loading (shared) ──────────────────────────────────────
def _load_factors(rets):
    HF_DIR = REPO / 'data' / 'high_freq_macro'
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]; dxy.index = pd.to_datetime(dxy.index)
    rr = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")["real_rate"]; rr.index = pd.to_datetime(rr.index)
    vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]; vix.index = pd.to_datetime(vix.index)
    us10y = pd.read_parquet(HF_DIR / "us_10y_yield_fred.parquet")
    us10y = us10y.set_index("date")["us_10y_yield"] if "date" in us10y.columns else us10y
    us10y.index = pd.to_datetime(us10y.index)
    cs = pd.read_parquet(HF_DIR / "cn_us_spread_10y.parquet")
    cs = cs.set_index("date")["cn_us_spread"] if "date" in cs.columns else cs
    cs.index = pd.to_datetime(cs.index)
    go = pd.read_parquet(HF_DIR / "gold_oil_corr.parquet")
    go = go.set_index("date")["gold_oil_corr"] if "date" in go.columns else go
    go.index = pd.to_datetime(go.index)
    idx_prices = pd.read_parquet(HF_DIR / "v7_index_prices.parquet")
    gj = idx_prices["收盘价:沪金指数"]; gj.index = pd.to_datetime(gj.index)
    ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet")

    idx = rets.index
    dxy = dxy.reindex(idx).ffill(); rr = rr.reindex(idx).ffill()
    vix = vix.reindex(idx).ffill(); us10y = us10y.reindex(idx).ffill()
    cs = cs.reindex(idx).ffill(); go = go.reindex(idx).ffill()
    gj = gj.reindex(idx).ffill()
    o = ohlcv["518880"]["open"].reindex(idx).ffill()
    c = ohlcv["518880"]["close"].reindex(idx).ffill()
    h = ohlcv["518880"]["high"].reindex(idx).ffill()
    l = ohlcv["518880"]["low"].reindex(idx).ffill()
    v = ohlcv["518880"]["volume"].reindex(idx).ffill()

    gj_z = (gj - gj.rolling(252).mean()) / gj.rolling(252).std().clip(lower=1e-6)
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std().clip(lower=1e-6)
    go_z = (go - go.rolling(252).mean()) / go.rolling(252).std().clip(lower=1e-6)
    dxy_5d = dxy.pct_change(5)
    breakeven = us10y - rr
    ret_20d = rets.rolling(20).sum()
    vol_20d = rets.rolling(20).std()
    vol_ratio = v / v.rolling(20).mean()
    intra_vol = (h - l) / c
    parkinson = np.sqrt(1 / (4 * np.log(2)) * (np.log(h / l)) ** 2).rolling(20).mean()
    gap = ((o - c.shift(1)) / c.shift(1)).abs().rolling(20).mean()
    ret_vol_ratio = ret_20d * vol_ratio
    skew_20d_abs = rets.rolling(20).skew().abs()
    skew60_vol20 = rets.rolling(60).skew() * vol_20d
    kurt_252d = rets.rolling(252).kurt()
    rr_z = (rr - rr.rolling(252).mean()) / rr.rolling(252).std().clip(lower=1e-6)
    us10y_z = (us10y - us10y.rolling(252).mean()) / us10y.rolling(252).std().clip(lower=1e-6)
    diverge_rr = gj_z - rr_z
    diverge_us10y = gj_z - us10y_z
    vol_20d_pct = vol_20d.pct_change(5)

    feat_names = [
        "gj_z", "ret_20d", "vol_20d", "us10y", "rr", "vix_z",
        "breakeven", "go_z", "dxy_5d", "cs", "ret_vol_ratio",
        "intra_vol", "vol_ratio", "gap_20d", "parkinson",
        "diverge_rr", "diverge_us10y", "vol_20d_pct",
        "skew_20d_abs", "skew60_vol20", "kurt_252d",
    ]
    feat_arrays = [
        gj_z, ret_20d, vol_20d, us10y, rr, vix_z,
        breakeven, go_z, dxy_5d, cs, ret_vol_ratio,
        intra_vol, vol_ratio, gap, parkinson,
        diverge_rr, diverge_us10y, vol_20d_pct,
        skew_20d_abs, skew60_vol20, kurt_252d,
    ]
    return feat_names, feat_arrays, rets.index


def _build_X(feat_arrays, feat_names, rd):
    return pd.DataFrame({n: f.loc[:rd] for n, f in zip(feat_names, feat_arrays)})


def _build_feat(feat_arrays, fi):
    return np.array([f.iloc[fi - 1] for f in feat_arrays]).reshape(1, -1)


# ── Model 1: BayesianRidge + Quantile (current) ──────────────────
def run_bayesian_ridge_quantile(rets, oos_idx, feat_names, feat_arrays, all_idx):
    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    uncertainty = pd.Series(index=oos_idx, dtype=float)
    for rd in oos_idx[::20]:
        ri = all_idx.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = _build_X(feat_arrays, feat_names, rd)
        y = fwd_vol.loc[:rd]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        if len(Xc) < 100:
            continue
        scaler = StandardScaler()
        Xc_scaled = scaler.fit_transform(Xc)
        try:
            m_br = BayesianRidge(fit_intercept=False, compute_score=True).fit(Xc_scaled, yc)
            qr = QuantReg(yc, Xc_scaled).fit(q=0.9, max_iter=5000)
            for fd in pd.bdate_range(rd + pd.Timedelta(days=1), periods=20):
                if fd not in forecasts.index:
                    continue
                fi = all_idx.get_indexer([fd], method="ffill")[0]
                if fi < 1:
                    continue
                feat = _build_feat(feat_arrays, fi)
                if not np.any(np.isnan(feat)):
                    feat_scaled = scaler.transform(feat)
                    pred, std = m_br.predict(feat_scaled, return_std=True)
                    pred_q90 = qr.predict(feat_scaled)
                    inv_std = 1.0 / (std[0] + 1e-6)
                    w = inv_std / (inv_std + 1)
                    forecasts[fd] = w * pred[0] + (1 - w) * pred_q90[0]
                    uncertainty[fd] = std[0]
        except Exception:
            continue
    return forecasts, uncertainty


# ── Model 2: Conformal Prediction + ElasticNet ───────────────────
def run_conformal_elasticnet(rets, oos_idx, feat_names, feat_arrays, all_idx):
    """Split Conformal Prediction: use calibration set to estimate residuals."""
    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    uncertainty = pd.Series(index=oos_idx, dtype=float)
    alpha = 0.1  # 90% coverage

    for rd in oos_idx[::20]:
        ri = all_idx.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = _build_X(feat_arrays, feat_names, rd)
        y = fwd_vol.loc[:rd]
        valid = X.notna().all(axis=1) & y.notna()
        Xv, yv = X.loc[valid], y.loc[valid]
        if len(Xv) < 200:
            continue

        # Split: last 20% as calibration set
        n = len(Xv)
        n_cal = max(int(n * 0.2), 50)
        X_train, X_cal = Xv.iloc[:-n_cal].values, Xv.iloc[-n_cal:].values
        y_train, y_cal = yv.iloc[:-n_cal].values, yv.iloc[-n_cal:].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_cal_s = scaler.transform(X_cal)

        try:
            m = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=False,
                           max_iter=10000, random_state=42).fit(X_train_s, y_train)

            # Conformal: compute calibration residuals
            cal_pred = m.predict(X_cal_s)
            cal_residuals = np.abs(y_cal - cal_pred)
            q_level = np.ceil((1 - alpha) * (len(cal_residuals) + 1)) / len(cal_residuals)
            q_hat = np.quantile(cal_residuals, min(q_level, 1.0))

            for fd in pd.bdate_range(rd + pd.Timedelta(days=1), periods=20):
                if fd not in forecasts.index:
                    continue
                fi = all_idx.get_indexer([fd], method="ffill")[0]
                if fi < 1:
                    continue
                feat = _build_feat(feat_arrays, fi)
                if not np.any(np.isnan(feat)):
                    feat_scaled = scaler.transform(feat)
                    pred = m.predict(feat_scaled)[0]
                    forecasts[fd] = pred
                    uncertainty[fd] = q_hat  # conformal interval half-width
        except Exception:
            continue
    return forecasts, uncertainty


# ── Model 3: Bootstrap Aggregation + ElasticNet ──────────────────
def run_bootstrap_elasticnet(rets, oos_idx, feat_names, feat_arrays, all_idx, n_boot=50):
    """Bootstrap: fit multiple ElasticNet on resampled data, average predictions."""
    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    uncertainty = pd.Series(index=oos_idx, dtype=float)

    for rd in oos_idx[::20]:
        ri = all_idx.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = _build_X(feat_arrays, feat_names, rd)
        y = fwd_vol.loc[:rd]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        if len(Xc) < 100:
            continue
        scaler = StandardScaler()
        Xc_scaled = scaler.fit_transform(Xc)

        try:
            rng = np.random.RandomState(42)
            models = []
            for _ in range(n_boot):
                idx = rng.choice(len(Xc), size=len(Xc), replace=True)
                m = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=False,
                               max_iter=10000, random_state=42)
                m.fit(Xc_scaled[idx], yc[idx])
                models.append(m)

            for fd in pd.bdate_range(rd + pd.Timedelta(days=1), periods=20):
                if fd not in forecasts.index:
                    continue
                fi = all_idx.get_indexer([fd], method="ffill")[0]
                if fi < 1:
                    continue
                feat = _build_feat(feat_arrays, fi)
                if not np.any(np.isnan(feat)):
                    feat_scaled = scaler.transform(feat)
                    preds = np.array([m.predict(feat_scaled)[0] for m in models])
                    forecasts[fd] = preds.mean()
                    uncertainty[fd] = preds.std()
        except Exception:
            continue
    return forecasts, uncertainty


# ── Evaluation Helpers ────────────────────────────────────────────
def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000):
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            aucs.append(roc_auc_score(y_true.iloc[idx], y_prob.iloc[idx]))
        except:
            continue
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def delong_test(y_true, y_prob1, y_prob2, n_bootstrap=1000):
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            diffs.append(roc_auc_score(y_true.iloc[idx], y_prob1.iloc[idx])
                        - roc_auc_score(y_true.iloc[idx], y_prob2.iloc[idx]))
        except:
            continue
    diffs = np.array(diffs)
    z = (roc_auc_score(y_true, y_prob1) - roc_auc_score(y_true, y_prob2)) / (diffs.std() + 1e-10)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def calibration_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            ece += abs(y_prob[mask].mean() - y_true[mask].mean()) * mask.sum() / len(y_true)
    return ece


# ── Main Comparison ──────────────────────────────────────────────
def run():
    print("=" * 80)
    print("Gold Model 3-Way Comparison")
    print("=" * 80)

    etf = load_etf_data()
    cfg = CONFIGS['gold']
    nav = etf['518880']
    rets = nav.pct_change().dropna() * 100
    y_true, _, _ = compute_y_true(rets / 100, cfg)
    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.dropna().index)

    print(f"Data: {oos_idx[0].date()} ~ {oos_idx[-1].date()} ({len(oos_idx)} days)")
    print(f"Positive rate: {y_true.loc[oos_idx].mean():.3f}")

    feat_names, feat_arrays, all_idx = _load_factors(rets)

    # Run 3 models
    print("\nRunning models...")
    raw_bq, unc_bq = run_bayesian_ridge_quantile(rets, oos_idx, feat_names, feat_arrays, all_idx)
    raw_cp, unc_cp = run_conformal_elasticnet(rets, oos_idx, feat_names, feat_arrays, all_idx)
    raw_bt, unc_bt = run_bootstrap_elasticnet(rets, oos_idx, feat_names, feat_arrays, all_idx)
    print("  1. BayesianRidge+Quantile: done")
    print("  2. Conformal+ElasticNet: done")
    print("  3. Bootstrap+ElasticNet: done")

    # Convert to ensemble probabilities
    prob_bq = _cont_prob(raw_bq.dropna())
    prob_cp = _cont_prob(raw_cp.dropna())
    prob_bt = _cont_prob(raw_bt.dropna())
    raw_clim = _model_climatology(rets, oos_idx)
    prob_clim = _cont_prob(raw_clim.dropna())

    yt = y_true.loc[oos_idx].dropna()
    common = yt.index.intersection(prob_bq.dropna().index) \
                  .intersection(prob_cp.dropna().index) \
                  .intersection(prob_bt.dropna().index) \
                  .intersection(prob_clim.dropna().index)
    yt = yt.loc[common]

    # Equal-weight ensemble for all
    ens_bq = 0.5 * prob_bq.loc[common] + 0.5 * prob_clim.loc[common]
    ens_cp = 0.5 * prob_cp.loc[common] + 0.5 * prob_clim.loc[common]
    ens_bt = 0.5 * prob_bt.loc[common] + 0.5 * prob_clim.loc[common]

    # ── Core Metrics ──
    print("\n" + "-" * 80)
    print("Core Metrics:")
    models = {
        'BayesianRidge+Q': (ens_bq, unc_bq),
        'Conformal+EN':    (ens_cp, unc_cp),
        'Bootstrap+EN':    (ens_bt, unc_bt),
    }
    metrics = {}
    for name, (ens, unc) in models.items():
        m = evaluate(yt, ens)
        metrics[name] = m
        unc_common = unc.loc[common].dropna()
        print(f"\n  {name}:")
        print(f"    AUC:     {m['auc']:.4f}")
        print(f"    PR-AUC:  {m['prauc']:.4f}")
        print(f"    Brier:   {m['brier']:.4f}")
        print(f"    ECE:     {m['ece']:.4f}")
        print(f"    Unc mean: {unc_common.mean():.4f}  std: {unc_common.std():.4f}")

    # ── Bootstrap CI ──
    print("\n" + "-" * 80)
    print("Bootstrap AUC 95% CI:")
    for name, (ens, _) in models.items():
        mean, lo, hi = bootstrap_auc_ci(yt, ens)
        print(f"  {name:20s}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # ── Pairwise DeLong ──
    print("\n" + "-" * 80)
    print("Pairwise DeLong Tests:")
    pairs = [('BayesianRidge+Q', 'Conformal+EN'),
             ('BayesianRidge+Q', 'Bootstrap+EN'),
             ('Conformal+EN', 'Bootstrap+EN')]
    for n1, n2 in pairs:
        z, p = delong_test(yt, models[n1][0], models[n2][0])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        diff = metrics[n1]['auc'] - metrics[n2]['auc']
        print(f"  {n1} vs {n2}: diff={diff:+.4f}  z={z:.3f}  p={p:.4f} {sig}")

    # ── Time Period Stability ──
    print("\n" + "-" * 80)
    print("Time Period Stability (4 periods):")
    n = len(common)
    ps = n // 4
    period_aucs = {name: [] for name in models}
    for i in range(4):
        dates = common[i * ps:(i + 1) * ps]
        yt_p = yt.loc[dates]
        if len(yt_p) > 10 and yt_p.nunique() > 1:
            parts = []
            for name, (ens, _) in models.items():
                auc_p = roc_auc_score(yt_p, ens.loc[dates])
                period_aucs[name].append(auc_p)
                parts.append(f"{name[:8]}={auc_p:.3f}")
            print(f"  Period {i+1}: {'  '.join(parts)}")
    print("  AUC std:")
    for name in models:
        print(f"    {name:20s}: {np.std(period_aucs[name]):.4f}")

    # ── Uncertainty Quality ──
    print("\n" + "-" * 80)
    print("Uncertainty Quality (low-unc vs high-unc AUC):")
    for name, (ens, unc) in models.items():
        uc = unc.loc[common].dropna()
        if len(uc) > 0:
            med = uc.median()
            lo_mask = uc <= med
            hi_mask = uc > med
            if yt.loc[lo_mask.index].nunique() > 1 and yt.loc[hi_mask.index].nunique() > 1:
                auc_lo = roc_auc_score(yt.loc[lo_mask.index], ens.loc[lo_mask.index])
                auc_hi = roc_auc_score(yt.loc[hi_mask.index], ens.loc[hi_mask.index])
                print(f"  {name:20s}: low-unc AUC={auc_lo:.4f}  high-unc AUC={auc_hi:.4f}  diff={auc_lo - auc_hi:+.4f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    best = max(metrics, key=lambda k: metrics[k]['auc'])
    print(f"\n  Best AUC: {best} ({metrics[best]['auc']:.4f})")
    print(f"\n  {'Model':<22} {'AUC':>8} {'PR-AUC':>8} {'Brier':>8} {'ECE':>8}")
    print("  " + "-" * 54)
    for name in models:
        m = metrics[name]
        marker = " <--" if name == best else ""
        print(f"  {name:<22} {m['auc']:>8.4f} {m['prauc']:>8.4f} {m['brier']:>8.4f} {m['ece']:>8.4f}{marker}")


if __name__ == "__main__":
    run()
