"""Gold Model A/B Comparison: BayesianRidge+Quantile vs ElasticNet

Runs both models on same data and compares:
1. Ensemble AUC/PR-AUC/Brier
2. Statistical significance (bootstrap)
3. Calibration
4. Time-period stability
5. Signal quality
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk.ensemble import (
    _model_macro_gold, _model_climatology, _cont_prob,
    load_etf_data, CONFIGS, compute_y_true, evaluate, OOS_START, OOS_END
)


def run_gold_macro(rets, oos_idx, version='bayesian'):
    """Run gold macro model with specified version."""
    from sklearn.preprocessing import StandardScaler
    
    if version == 'bayesian':
        from sklearn.linear_model import BayesianRidge
        from statsmodels.regression.quantile_regression import QuantReg
    else:
        from sklearn.linear_model import ElasticNet
    
    # Load factors (same as _model_macro_gold)
    HF_DIR = REPO / 'data' / 'high_freq_macro'
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
    dxy.index = pd.to_datetime(dxy.index)
    rr = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")["real_rate"]
    rr.index = pd.to_datetime(rr.index)
    vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]
    vix.index = pd.to_datetime(vix.index)
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
    gj = idx_prices["收盘价:沪金指数"]
    gj.index = pd.to_datetime(gj.index)
    ohlcv = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet")

    idx = rets.index
    dxy = dxy.reindex(idx).ffill()
    rr = rr.reindex(idx).ffill()
    vix = vix.reindex(idx).ffill()
    us10y = us10y.reindex(idx).ffill()
    cs = cs.reindex(idx).ffill()
    go = go.reindex(idx).ffill()
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

    fwd_vol = rets.rolling(20).std().shift(-20) / 100 * np.sqrt(252)
    forecasts = pd.Series(index=oos_idx, dtype=float)
    refit_dates = oos_idx[::20]
    
    for rd in refit_dates:
        ri = rets.index.get_indexer([rd], method="ffill")[0]
        if ri < 504:
            continue
        X = pd.DataFrame({
            "gj_z": gj_z.loc[:rd], "ret_20d": ret_20d.loc[:rd],
            "vol_20d": vol_20d.loc[:rd], "us10y": us10y.loc[:rd],
            "rr": rr.loc[:rd], "vix_z": vix_z.loc[:rd],
            "breakeven": breakeven.loc[:rd], "go_z": go_z.loc[:rd],
            "dxy_5d": dxy_5d.loc[:rd], "cs": cs.loc[:rd],
            "ret_vol_ratio": ret_vol_ratio.loc[:rd],
            "intra_vol": intra_vol.loc[:rd],
            "vol_ratio": vol_ratio.loc[:rd],
            "gap_20d": gap.loc[:rd],
            "parkinson": parkinson.loc[:rd],
            "diverge_rr": diverge_rr.loc[:rd],
            "diverge_us10y": diverge_us10y.loc[:rd],
            "vol_20d_pct": vol_20d_pct.loc[:rd],
            "skew_20d_abs": skew_20d_abs.loc[:rd],
            "skew60_vol20": skew60_vol20.loc[:rd],
            "kurt_252d": kurt_252d.loc[:rd],
        })
        y = fwd_vol.loc[:rd]
        valid = X.notna().all(axis=1) & y.notna()
        Xc, yc = X.loc[valid].values, y.loc[valid].values
        if len(Xc) < 100:
            continue
        scaler = StandardScaler()
        Xc_scaled = scaler.fit_transform(Xc)
        try:
            if version == 'bayesian':
                m_br = BayesianRidge(fit_intercept=False, compute_score=True)
                m_br.fit(Xc_scaled, yc)
                qr = QuantReg(yc, Xc_scaled).fit(q=0.9, max_iter=5000)
            else:
                m = ElasticNet(alpha=0.01, l1_ratio=0.5, fit_intercept=False,
                               max_iter=10000, random_state=42).fit(Xc_scaled, yc)
            
            fs = rd + pd.Timedelta(days=1)
            for fd in pd.bdate_range(fs, periods=20):
                if fd not in forecasts.index:
                    continue
                fi = rets.index.get_indexer([fd], method="ffill")[0]
                if fi < 1:
                    continue
                feat = np.array([
                    gj_z.iloc[fi-1], ret_20d.iloc[fi-1],
                    vol_20d.iloc[fi-1], us10y.iloc[fi-1],
                    rr.iloc[fi-1], vix_z.iloc[fi-1],
                    breakeven.iloc[fi-1], go_z.iloc[fi-1],
                    dxy_5d.iloc[fi-1], cs.iloc[fi-1],
                    ret_vol_ratio.iloc[fi-1], intra_vol.iloc[fi-1],
                    vol_ratio.iloc[fi-1], gap.iloc[fi-1],
                    parkinson.iloc[fi-1], diverge_rr.iloc[fi-1],
                    diverge_us10y.iloc[fi-1], vol_20d_pct.iloc[fi-1],
                    skew_20d_abs.iloc[fi-1], skew60_vol20.iloc[fi-1],
                    kurt_252d.iloc[fi-1],
                ]).reshape(1, -1)
                if not np.any(np.isnan(feat)):
                    feat_scaled = scaler.transform(feat)
                    if version == 'bayesian':
                        pred_br, std_br = m_br.predict(feat_scaled, return_std=True)
                        pred_q90 = qr.predict(feat_scaled)
                        inv_std = 1.0 / (std_br[0] + 1e-6)
                        w_br = inv_std / (inv_std + 1)
                        forecasts[fd] = w_br * pred_br[0] + (1 - w_br) * pred_q90[0]
                    else:
                        forecasts[fd] = m.predict(feat_scaled)[0]
        except Exception:
            continue
    return forecasts


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000, ci=0.95):
    """Compute AUC with confidence interval via bootstrap."""
    rng = np.random.RandomState(42)
    aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            auc = roc_auc_score(y_true.iloc[idx], y_prob.iloc[idx])
            aucs.append(auc)
        except:
            continue
    aucs = np.array(aucs)
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return aucs.mean(), lower, upper


def delong_test(y_true, y_prob1, y_prob2, n_bootstrap=1000):
    """DeLong test for comparing two AUCs."""
    rng = np.random.RandomState(42)
    auc_diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            a1 = roc_auc_score(y_true.iloc[idx], y_prob1.iloc[idx])
            a2 = roc_auc_score(y_true.iloc[idx], y_prob2.iloc[idx])
            auc_diffs.append(a1 - a2)
        except:
            continue
    auc_diffs = np.array(auc_diffs)
    z_stat = (roc_auc_score(y_true, y_prob1) - roc_auc_score(y_true, y_prob2)) / (auc_diffs.std() + 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    return z_stat, p_value


def calibration_analysis(y_true, y_prob, n_bins=10):
    """Compute calibration metrics."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_means, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_means.append(y_true[mask].mean())
            bin_counts.append(mask.sum())
    ece = sum(abs(c - m) * n / len(y_true) for c, m, n in zip(bin_centers, bin_means, bin_counts))
    return ece, bin_centers, bin_means


def run_comparison():
    """Run full A/B comparison."""
    print("=" * 80)
    print("Gold Model A/B Comparison: BayesianRidge+Quantile vs ElasticNet")
    print("=" * 80)
    
    etf = load_etf_data()
    cfg = CONFIGS['gold']
    nav = etf['518880']
    rets = nav.pct_change().dropna() * 100
    y_true, realized_vol, threshold = compute_y_true(rets / 100, cfg)
    
    oos_idx = rets.index[(rets.index >= OOS_START) & (rets.index <= OOS_END)]
    oos_idx = oos_idx.intersection(y_true.dropna().index)
    
    print(f"\nData: {oos_idx[0].date()} ~ {oos_idx[-1].date()} ({len(oos_idx)} days)")
    print(f"Positive rate: {y_true.loc[oos_idx].mean():.3f}")
    
    # Run both models
    print("\n" + "-" * 80)
    print("Running models...")
    
    raw_bq = run_gold_macro(rets, oos_idx, version='bayesian')
    raw_en = run_gold_macro(rets, oos_idx, version='elasticnet')
    
    print("  BayesianRidge+Quantile: done")
    print("  ElasticNet: done")
    
    # Compute ensemble probabilities
    prob_bq = _cont_prob(raw_bq.dropna())
    prob_en = _cont_prob(raw_en.dropna())
    
    # Get climatology for ensemble
    raw_clim = _model_climatology(rets, oos_idx)
    prob_clim = _cont_prob(raw_clim.dropna())
    
    # Compute ensemble with equal weights for fair comparison
    yt = y_true.loc[oos_idx].dropna()
    
    # Find common indices
    common_bq = yt.index.intersection(prob_bq.dropna().index).intersection(prob_clim.dropna().index)
    common_en = yt.index.intersection(prob_en.dropna().index).intersection(prob_clim.dropna().index)
    
    yt_bq = yt.loc[common_bq]
    yt_en = yt.loc[common_en]
    
    # Simple equal-weight ensemble
    ens_bq = 0.5 * prob_bq.loc[common_bq] + 0.5 * prob_clim.loc[common_bq]
    ens_en = 0.5 * prob_en.loc[common_en] + 0.5 * prob_clim.loc[common_en]
    
    # Evaluate
    print("\n" + "-" * 80)
    print("Evaluation Results:")
    
    m_bq = evaluate(yt_bq, ens_bq)
    m_en = evaluate(yt_en, ens_en)
    
    print(f"\n{'Metric':<12} {'BayesianRidge+Q':>16} {'ElasticNet':>16} {'Diff':>10}")
    print("-" * 56)
    for k in ['auc', 'prauc', 'brier', 'ece', 'n']:
        v1 = m_bq[k]
        v2 = m_en[k]
        diff = v1 - v2 if k != 'n' else v1 - v2
        print(f"{k:<12} {v1:>16.4f} {v2:>16.4f} {diff:>+10.4f}")
    
    # Bootstrap CI
    print("\n" + "-" * 80)
    print("Bootstrap AUC Confidence Intervals (95%):")
    
    auc_bq, lo_bq, hi_bq = bootstrap_auc_ci(yt_bq, ens_bq)
    auc_en, lo_en, hi_en = bootstrap_auc_ci(yt_en, ens_en)
    
    print(f"  BayesianRidge+Q: {auc_bq:.4f} [{lo_bq:.4f}, {hi_bq:.4f}]")
    print(f"  ElasticNet:      {auc_en:.4f} [{lo_en:.4f}, {hi_en:.4f}]")
    
    # DeLong test
    print("\n" + "-" * 80)
    print("Statistical Significance (DeLong Test):")
    
    # Use common dates for both models
    common_all = common_bq.intersection(common_en)
    yt_all = yt.loc[common_all]
    ens_bq_all = ens_bq.loc[common_all]
    ens_en_all = ens_en.loc[common_all]
    
    z_stat, p_value = delong_test(yt_all, ens_bq_all, ens_en_all)
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    
    print(f"  AUC difference: {roc_auc_score(yt_all, ens_bq_all) - roc_auc_score(yt_all, ens_en_all):+.4f}")
    print(f"  z-statistic: {z_stat:.4f}")
    print(f"  p-value: {p_value:.4f} {sig}")
    
    # Calibration
    print("\n" + "-" * 80)
    print("Calibration (ECE):")
    
    ece_bq, bc_bq, bm_bq = calibration_analysis(yt_bq, ens_bq)
    ece_en, bc_en, bm_en = calibration_analysis(yt_en, ens_en)
    
    print(f"  BayesianRidge+Q: {ece_bq:.4f}")
    print(f"  ElasticNet:      {ece_en:.4f}")
    
    # Time period stability
    print("\n" + "-" * 80)
    print("Time Period Stability (4 periods):")
    
    n = len(common_all)
    period_size = n // 4
    periods_bq, periods_en = [], []
    
    for i in range(4):
        start = i * period_size
        end = min((i + 1) * period_size, n)
        dates = common_all[start:end]
        
        yt_p = yt_all.loc[dates]
        if len(yt_p) > 10 and yt_p.nunique() > 1:
            auc_p_bq = roc_auc_score(yt_p, ens_bq_all.loc[dates])
            auc_p_en = roc_auc_score(yt_p, ens_en_all.loc[dates])
            periods_bq.append(auc_p_bq)
            periods_en.append(auc_p_en)
            print(f"  Period {i+1}: BQ={auc_p_bq:.3f}  EN={auc_p_en:.3f}  diff={auc_p_bq-auc_p_en:+.3f}")
    
    print(f"  AUC std: BQ={np.std(periods_bq):.4f}  EN={np.std(periods_en):.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    better = "BayesianRidge+Quantile" if m_bq['auc'] > m_en['auc'] else "ElasticNet"
    improvement = abs(m_bq['auc'] - m_en['auc']) / m_en['auc'] * 100
    
    print(f"""
  Winner: {better}
  
  AUC Improvement: {improvement:.2f}%
  
  Statistical Significance: {sig} (p={p_value:.4f})
  
  Key Findings:
    1. AUC: {"BQ" if m_bq['auc'] > m_en['auc'] else "EN"} wins ({m_bq['auc']:.4f} vs {m_en['auc']:.4f})
    2. PR-AUC: {"BQ" if m_bq['prauc'] > m_en['prauc'] else "EN"} wins ({m_bq['prauc']:.4f} vs {m_en['prauc']:.4f})
    3. Calibration: {"BQ" if ece_bq < ece_en else "EN"} better (ECE={min(ece_bq, ece_en):.4f})
    4. Stability: {"BQ" if np.std(periods_bq) < np.std(periods_en) else "EN"} more stable (std={min(np.std(periods_bq), np.std(periods_en)):.4f})
  
  Recommendation: {"Use BayesianRidge+Quantile" if better == "BayesianRidge+Quantile" else "Keep ElasticNet"}
""")


if __name__ == "__main__":
    run_comparison()
