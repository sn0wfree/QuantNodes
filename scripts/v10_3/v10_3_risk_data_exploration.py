#!/usr/bin/env python3
"""v10.3 Risk Warning — Phase 1: Data Exploration.

Loads ETF daily returns, y_E target, and external signals (VIX, DXY, macro factors).
Computes:
  1. Single-signal AUC ranking
  2. Signal correlation matrix
  3. y_E distribution (OOS)
  4. Extreme event backtest (2022/2024/2025)
  5. Ensemble layer vs external signal comparison

Output: reports/momentum_etf_rotation/v10_3/risk_exploration/
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    EnsembleRiskWarning, load_etf_data, CONFIGS,
)

OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v10_3" / "risk_exploration"
OUT_DIR.mkdir(exist_ok=True, parents=True)
HF_DIR = REPO / "data" / "high_freq_macro"

OOS_START = "2022-01-04"
OOS_END = "2026-05-29"

log = print


# ── 1. Data loading ──────────────────────────────────────────────

def load_macro_signals() -> pd.DataFrame:
    signals = {}
    vix = pd.read_parquet(HF_DIR / "macro_vix_daily.parquet")["vix"]
    vix.index = pd.to_datetime(vix.index)
    signals["vix"] = vix
    dxy = pd.read_parquet(HF_DIR / "macro_dxy_daily_v2.parquet")["dxy"]
    dxy.index = pd.to_datetime(dxy.index)
    signals["dxy"] = dxy
    rr = pd.read_parquet(HF_DIR / "macro_real_rate_daily.parquet")["real_rate"]
    rr.index = pd.to_datetime(rr.index)
    signals["real_rate"] = rr
    cs = pd.read_parquet(HF_DIR / "cn_us_spread_10y.parquet")
    cs["date"] = pd.to_datetime(cs["date"])
    cs = cs.set_index("date")["cn_us_spread"]
    signals["cn_us_spread"] = cs
    go = pd.read_parquet(HF_DIR / "gold_oil_corr.parquet")
    go["date"] = pd.to_datetime(go["date"])
    go = go.set_index("date")["gold_oil_corr"]
    signals["gold_oil_corr"] = go
    us10y = pd.read_parquet(HF_DIR / "us_10y_yield_fred.parquet")
    us10y["date"] = pd.to_datetime(us10y["date"])
    us10y = us10y.set_index("date")["us_10y_yield"]
    signals["us_10y_yield"] = us10y
    df = pd.DataFrame(signals)
    df.index.name = "date"
    return df


def load_v9_factors() -> pd.DataFrame:
    df = pd.read_parquet(HF_DIR / "v9_factors_weekly.parquet")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def compute_equity_returns() -> pd.Series:
    etf = load_etf_data()
    nav = etf["510300"]
    return nav.pct_change().dropna() * 100


def compute_asset_returns(asset: str) -> pd.Series:
    etf = load_etf_data()
    cfg = CONFIGS[asset]
    avail = [c for c in cfg.etfs if c in etf.columns]
    if cfg.weights:
        w = dict(zip(cfg.etfs, cfg.weights))
        total = sum(w[e] for e in avail)
        nav = sum(etf[e] * w[e] / total for e in avail)
    else:
        nav = etf[avail[0]]
    return nav.pct_change().dropna() * 100

# ── 2. Target: y_E ──────────────────────────────────────────────

def build_y_E(rets: pd.Series,
              horizon: int = 10,
              dd_threshold: float = -2.5,
              ret_percentile: float = 0.20) -> pd.Series:
    max_dd = rets.rolling(horizon).min().shift(-horizon)
    fwd_ret = rets.shift(-horizon).rolling(horizon).sum()
    ret_q = fwd_ret.rolling(252, min_periods=60).rank(pct=True)
    y_E = ((max_dd < dd_threshold) | (ret_q < ret_percentile)).astype(float)
    return y_E


# ── 3. Feature engineering ───────────────────────────────────────

def build_features(rets: pd.Series, macro: pd.DataFrame,
                   factors: pd.DataFrame) -> pd.DataFrame:
    idx = rets.index
    features = {}

    # Self technical
    features["vol_5d"] = rets.rolling(5).std()
    features["vol_20d"] = rets.rolling(20).std()
    features["vol_60d"] = rets.rolling(60).std()
    features["ret_1d"] = rets
    features["ret_5d"] = rets.rolling(5).sum()
    features["ret_20d"] = rets.rolling(20).sum()
    features["ret_60d"] = rets.rolling(60).sum()
    features["ret_12m"] = rets.rolling(252).sum()
    features["skew_60d"] = rets.rolling(60).skew()
    features["kurt_60d"] = rets.rolling(60).kurt()
    features["max_dd_60d"] = rets.rolling(60).min()
    cum = (1 + rets / 100).cumprod()
    rolling_max = cum.rolling(252, min_periods=60).max()
    features["drawdown"] = (cum / rolling_max - 1) * 100
    ma200 = rets.rolling(200).mean()
    features["dist_200ma"] = (rets - ma200) / ma200.abs().clip(lower=1e-6)

    # VIX derived
    vix = macro["vix"].reindex(idx).ffill()
    features["vix"] = vix
    features["vix_zscore"] = ((vix - vix.rolling(252).mean())
                              / vix.rolling(252).std().clip(lower=1e-6))
    features["vix_ratio_ma20"] = vix / vix.rolling(20).mean().clip(lower=1e-6)
    features["vix_ratio_ma60"] = vix / vix.rolling(60).mean().clip(lower=1e-6)
    features["vix_above_20"] = (vix > 20).astype(float)
    features["vix_above_25"] = (vix > 25).astype(float)
    features["vix_above_30"] = (vix > 30).astype(float)

    # DXY derived
    dxy = macro["dxy"].reindex(idx).ffill()
    features["dxy"] = dxy
    features["dxy_zscore"] = ((dxy - dxy.rolling(252).mean())
                              / dxy.rolling(252).std().clip(lower=1e-6))
    features["dxy_logret"] = np.log(dxy / dxy.shift(1)).fillna(0)

    # Real rate
    rr = macro["real_rate"].reindex(idx).ffill()
    features["real_rate"] = rr
    features["real_rate_zscore"] = ((rr - rr.rolling(252).mean())
                                    / rr.rolling(252).std().clip(lower=1e-6))

    # CN-US spread
    cs = macro["cn_us_spread"].reindex(idx).ffill()
    features["cn_us_spread"] = cs
    features["cn_us_spread_zscore"] = ((cs - cs.rolling(252).mean())
                                       / cs.rolling(252).std().clip(lower=1e-6))

    # Gold-oil correlation
    features["gold_oil_corr"] = macro["gold_oil_corr"].reindex(idx).ffill()

    # US 10Y yield
    features["us_10y_yield"] = macro["us_10y_yield"].reindex(idx).ffill()

    # Macro factors (weekly -> daily ffill)
    for col in factors.columns:
        f = factors[col].reindex(idx).ffill()
        features[f"factor_{col}"] = f
        features[f"factor_{col}_zscore"] = (
            (f - f.rolling(252).mean()) / f.rolling(252).std().clip(lower=1e-6)
        )

    # Interaction features
    features["vix_x_ret_12m"] = features["vix_zscore"] * features["ret_12m"]
    features["vix_x_vol_20d"] = features["vix_zscore"] * features["vol_20d"]
    features["dxy_x_ret_20d"] = features["dxy_zscore"] * features["ret_20d"]

    df = pd.DataFrame(features, index=idx)
    df.index.name = "date"
    return df

# ── 4. Evaluation ────────────────────────────────────────────────

def single_signal_auc(y_true: pd.Series, signal: pd.Series,
                      name: str) -> dict:
    from sklearn.metrics import roc_auc_score
    common = y_true.dropna().index.intersection(signal.dropna().index)
    yt = y_true.loc[common]
    yp = signal.loc[common]
    if yt.nunique() < 2 or len(common) < 10:
        return {"name": name, "auc": 0.5, "n": len(common)}
    auc = roc_auc_score(yt, yp)
    return {"name": name, "auc": float(auc), "n": len(common)}


def signal_auc_ranking(y_true: pd.Series, features: pd.DataFrame,
                       label: str = "") -> pd.DataFrame:
    rows = []
    for col in features.columns:
        s = features[col]
        r = single_signal_auc(y_true, s, col)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("auc", ascending=False)
    df["label"] = label
    return df


def correlation_matrix(features: pd.DataFrame, out_path: Path):
    corr = features.corr(method="spearman")
    corr.to_csv(out_path)
    log(f"  Correlation matrix saved: {out_path.name}  ({len(corr.columns)}x{len(corr.columns)})")
    return corr


def extreme_events() -> list[dict]:
    return [
        {"name": "2022 Crash", "start": "2022-01-04", "end": "2022-10-31",
         "desc": "Fed hawkish hike + Russia-Ukraine"},
        {"name": "2024 Selloff", "start": "2024-01-02", "end": "2024-02-29",
         "desc": "Pre-924 policy trough"},
        {"name": "2025 Volatility", "start": "2025-03-01", "end": "2025-06-30",
         "desc": "Tariff shock + China-US friction"},
        {"name": "2026 Liquidity", "start": "2026-01-02", "end": "2026-03-31",
         "desc": "Liquidity tightening expectation"},
    ]


# ── 5. Ensemble layer comparison ────────────────────────────────

def load_ensemble_layers(asset: str = "equity") -> dict[str, pd.Series]:
    etf = load_etf_data()
    rw = EnsembleRiskWarning(etf, asset)
    layers = {}
    for key in rw._prob:
        layers[f"ensemble_{key}"] = rw._prob[key]
    layers["ensemble"] = rw._ensemble_prob
    return layers

# ── 6. Main ──────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("v10.3 Risk Warning - Phase 1: Data Exploration")
    log("=" * 60)

    # [1/7] Load data
    log("\n[1/7] Loading data...")
    rets = compute_equity_returns()
    macro = load_macro_signals()
    factors = load_v9_factors()
    log(f"  Equity returns: {len(rets)} days ({rets.index.min().date()} ~ {rets.index.max().date()})")
    log(f"  Macro signals: {len(macro.columns)} cols")
    log(f"  V9 factors: {len(factors.columns)} cols")

    # [2/7] Build y_E
    log("\n[2/7] Building y_E target...")
    y_E = build_y_E(rets)
    oos_mask = (y_E.index >= OOS_START) & (y_E.index <= OOS_END)
    y_E_oos = y_E.loc[oos_mask].dropna()
    log(f"  y_E OOS: {len(y_E_oos)} days, pos_rate={y_E_oos.mean():.1%}")

    # Save y_E
    y_E_oos.name = "y_E"
    y_E_oos.to_csv(OUT_DIR / "y_E_oos.csv")
    y_E.name = "y_E"
    y_E.to_csv(OUT_DIR / "y_E_full.csv")

    # [3/7] Build features
    log("\n[3/7] Building features...")
    features = build_features(rets, macro, factors)
    features_oos = features.loc[y_E_oos.index]
    n_self = sum("ret_" in c or "vol_" in c or "skew_" in c or "kurt_" in c
                 or "max_dd_" in c or "drawdown" in c or "dist_" in c for c in features.columns)
    n_vix = sum("vix" in c for c in features.columns)
    n_dxy = sum("dxy" in c for c in features.columns)
    n_macro = sum("factor_" in c for c in features.columns)
    n_other = len(features.columns) - n_self - n_vix - n_dxy - n_macro
    log(f"  Features: {len(features.columns)} cols, {len(features_oos)} days")
    log(f"    self_tech={n_self}  vix={n_vix}  dxy={n_dxy}  macro={n_macro}  other={n_other}")

    features_oos.to_csv(OUT_DIR / "features_oos.csv")
    log("  Features saved to features_oos.csv")

    # [4/7] Single-signal AUC ranking
    log("\n[4/7] Computing single-signal AUC ranking...")
    ranking = signal_auc_ranking(y_E_oos, features_oos, label="equity")
    ranking.to_csv(OUT_DIR / "signal_auc_ranking.csv", index=False)
    log(f"  Top 10 signals by AUC:")
    for _, row in ranking.head(10).iterrows():
        log(f"    {row['name']:30s}  AUC={row['auc']:.4f}  n={int(row['n'])}")

    # Group AUC by feature category
    ranking["group"] = "other"
    ranking.loc[ranking["name"].str.contains("ret_|vol_|skew_|kurt_|max_dd_|drawdown|dist_"), "group"] = "self_tech"
    ranking.loc[ranking["name"].str.contains("vix"), "group"] = "vix"
    ranking.loc[ranking["name"].str.contains("dxy"), "group"] = "dxy"
    ranking.loc[ranking["name"].str.contains("factor_"), "group"] = "macro"
    ranking.loc[ranking["name"].isin(["real_rate","real_rate_zscore","cn_us_spread","cn_us_spread_zscore","gold_oil_corr","us_10y_yield"]), "group"] = "other_macro"
    top_by_group = ranking.loc[ranking.groupby("group")["auc"].idxmax()].sort_values("auc", ascending=False)
    log(f"\n  Best signal per group:")
    for _, row in top_by_group.iterrows():
        log(f"    {row['group']:15s}  {row['name']:30s}  AUC={row['auc']:.4f}")

    # [5/7] Correlation matrix
    log("\n[5/7] Computing correlation matrix...")
    corr = correlation_matrix(features_oos, OUT_DIR / "feature_correlation_spearman.csv")

    # Find highly correlated pairs (|rho| > 0.85)
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = abs(corr.iloc[i, j])
            if val > 0.85:
                high_corr.append({"col1": corr.columns[i], "col2": corr.columns[j],
                                  "rho": f"{corr.iloc[i, j]:.3f}"})
    if high_corr:
        hc = pd.DataFrame(high_corr).sort_values("rho", ascending=False)
        hc.to_csv(OUT_DIR / "high_corr_pairs.csv", index=False)
        log(f"  High-correlation pairs (|rho|>0.85): {len(hc)}")
        for _, row in hc.head(10).iterrows():
            log(f"    {row['col1']:30s}  {row['col2']:30s}  rho={row['rho']}")
    else:
        log("  No high-correlation pairs found (|rho|>0.85)")

    # [6/7] y_E distribution
    log("\n[6/7] y_E distribution analysis...")
    y_E_full = y_E.dropna()
    y_E_annual = y_E_full.groupby(y_E_full.index.year).agg(["mean", "count"])
    y_E_annual.columns = ["pos_rate", "n_days"]
    y_E_annual.to_csv(OUT_DIR / "y_E_annual.csv")
    log(f"  y_E by year:")
    for yr, row in y_E_annual.iterrows():
        log(f"    {int(yr)}:  pos_rate={row['pos_rate']:.1%}  ({int(row['n_days'])} days)")

    # Monthly distribution
    y_E_monthly = y_E_full.groupby(y_E_full.index.month).mean()
    y_E_monthly.to_csv(OUT_DIR / "y_E_monthly.csv")
    log(f"  y_E by month:")
    for m, v in y_E_monthly.items():
        log(f"    Month {int(m):02d}:  pos_rate={v:.1%}")

    # [7/7] Extreme event backtest
    log("\n[7/7] Extreme event backtest...")
    events = extreme_events()
    event_rows = []
    for ev in events:
        mask = (y_E_oos.index >= ev["start"]) & (y_E_oos.index <= ev["end"])
        yt = y_E_oos.loc[mask]
        feats = features_oos.loc[mask]
        row = {"event": ev["name"], "period": ev["desc"],
               "y_E_pos_rate": f"{yt.mean():.1%}" if len(yt) > 0 else "N/A",
               "n_days": len(yt)}
        # Top 5 AUC signals
        top5 = ranking.head(5)["name"].values
        for col in top5:
            if col in feats.columns:
                row[col] = f"{feats[col].mean():.3f}"
        event_rows.append(row)
    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(OUT_DIR / "extreme_events.csv", index=False)
    log(f"  Events saved to extreme_events.csv")
    for _, r in event_df.iterrows():
        log(f"    {r['event']:15s}  y_E={r['y_E_pos_rate']:>6s}  ({r['n_days']}d)  {r['period']}")

    # [BONUS] Ensemble layer vs external signal comparison
    log("\n[BONUS] Ensemble layer vs external signal comparison...")
    layers = load_ensemble_layers("equity")
    layer_results = []
    for name, prob in layers.items():
        common = y_E_oos.index.intersection(prob.dropna().index)
        yt = y_E_oos.loc[common]
        yp = prob.loc[common]
        if yt.nunique() >= 2 and len(common) >= 10:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(yt, yp)
            layer_results.append({"name": name, "auc": f"{auc:.4f}", "n": len(common)})
    layer_df = pd.DataFrame(layer_results).sort_values("auc", ascending=False)
    layer_df.to_csv(OUT_DIR / "ensemble_vs_external.csv", index=False)
    log(f"  Ensemble layer AUC vs y_E:")
    for _, r in layer_df.iterrows():
        log(f"    {r['name']:20s}  AUC={r['auc']}  n={r['n']}")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  y_E OOS pos_rate: {y_E_oos.mean():.1%}")
    log(f"  Top signal: {ranking.iloc[0]['name']} (AUC={ranking.iloc[0]['auc']:.4f})")
    log(f"  Signals with AUC > 0.60: {(ranking['auc'] > 0.60).sum()} / {len(ranking)}")
    log(f"  Ensemble AUC vs y_E: {layer_df.iloc[0]['auc'] if len(layer_df) > 0 else 'N/A'}")
    log(f"  Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
