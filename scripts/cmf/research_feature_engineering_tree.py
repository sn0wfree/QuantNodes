#!/usr/bin/env python3
"""research_feature_engineering_tree.py — 扩展特征工程 + 树模型.

构造更多变换特征喂RF/GBR:
  - 基础特征 (12个指标的52周Z-score)
  - 滞后特征 (lag 1, 2, 4, 8周)
  - 差分/变化率 (1w, 4w)
  - 滚动统计 (z-score, percentile, skewness, kurtosis)
  - 交互特征 (top相关对乘积/差)

对比: 12特征 vs 扩展后~50特征
评估: walk-forward OOS IC, R², 方向准确率
"""
from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from itertools import combinations

OUTPUT_DIR = "reports/gold"

INDICATORS = {
    "tips10y":("gold_fred","tips10y"),"tips5y":("gold_fred","tips"),
    "rir":("gold_fred","rir"),"us_10y":("gold_macro","us_10y"),
    "us_2y":("gold_macro","us_2y"),"fed_total_assets":("gold_fred","fed_total_assets"),
    "m2":("gold_fred","m2_money_supply"),"comex_inv":("gold_position","comex_inv"),
    "cftc_oi":("gold_position","cftc_oi"),"brent":("fred_dcoil","brent"),
    "wti":("fred_dcoil","wti"),"us_cpi_yoy":("美国CPI","us_cpi_yoy"),
}


def load_data():
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    def load(src, vc):
        df = con.execute("SELECT trade_date, value FROM fundamentals_daily "
                        "WHERE source=? AND value_col=? ORDER BY trade_date", [src, vc]).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")["value"].astype(float)
    gold = load("gold_macro", "gold_spot")
    data = {name: load(s, v) for name, (s, v) in INDICATORS.items()}
    con.close()
    return gold, data


def build_extended_features(gold, data, week_idx):
    """构造扩展特征集 (基线+滞后+差分+rolling+交互)."""
    features = pd.DataFrame(index=week_idx)

    # === 1. 基线: 52周Z-score (PIT shift(1)) ===
    base_z = {}
    for name, s in data.items():
        s_w = s.reindex(week_idx, method="ffill")
        m52 = s_w.rolling(52, min_periods=12).mean()
        sd52 = s_w.rolling(52, min_periods=12).std() + 1e-8
        z = ((s_w - m52) / sd52).shift(1)  # PIT
        features[f"{name}_z"] = z
        base_z[name] = z

    # === 2. 滞后特征 (lag 1, 2, 4, 8) ===
    for name, z in list(base_z.items())[:6]:  # 仅top6指标(减少特征爆炸)
        for lag in [1, 2, 4, 8]:
            features[f"{name}_lag{lag}"] = z.shift(lag)

    # === 3. 差分/变化率 ===
    for name, s in list(data.items())[:6]:
        s_w = s.reindex(week_idx, method="ffill")
        chg1 = s_w.pct_change(1).shift(1)
        chg4 = s_w.pct_change(4).shift(1)
        features[f"{name}_chg1"] = chg1
        features[f"{name}_chg4"] = chg4

    # === 4. 滚动统计 (skewness, kurtosis, percentile) ===
    for name in ["tips10y", "us_10y", "fed_total_assets", "m2", "vix_close"]:
        s_w = gold.reindex(week_idx, method="ffill") if name == "vix_close" else \
              data[name].reindex(week_idx, method="ffill")
        # vix_close: 自身波动率作为特征
        if name == "vix_close":
            s_w = gold.rolling(20).std() * np.sqrt(52)
            s_w = s_w.reindex(week_idx).ffill()
        if len(s_w.dropna()) > 52:
            features[f"{name}_skew20"] = s_w.rolling(20).skew().shift(1)
            features[f"{name}_kurt20"] = s_w.rolling(20).kurt().shift(1)
            features[f"{name}_pctile52"] = s_w.rolling(52, min_periods=12).rank(pct=True).shift(1)

    # === 5. 交互特征 (top相关对乘积) ===
    # tips10y × us_10y (利率层交互)
    if "tips10y_z" in features.columns and "us_10y_z" in features.columns:
        features["tips10y_x_us10y"] = features["tips10y_z"] * features["us_10y_z"]
    # fed × m2 (货币层交互)
    if "fed_total_assets_z" in features.columns and "m2_z" in features.columns:
        features["fed_x_m2"] = features["fed_total_assets_z"] * features["m2_z"]
    # rir × tips5y (实际利率层)
    if "rir_z" in features.columns and "tips5y_z" in features.columns:
        features["rir_x_tips5y"] = features["rir_z"] * features["tips5y_z"]
    # 油价×TIPS
    if "wti_chg1" in features.columns and "tips10y_z" in features.columns:
        features["wti_x_tips10y"] = features["wti_chg1"] * features["tips10y_z"]
    # 跨资产: brent vs wti 价差
    if "brent_chg1" in features.columns and "wti_chg1" in features.columns:
        features["brent_wti_spread"] = features["brent_chg1"] - features["wti_chg1"]

    return features.dropna()


def build_baseline_features(gold, data, week_idx):
    """12特征基线 (与之前research_tree_model一致)."""
    features = pd.DataFrame(index=week_idx)
    for name, s in data.items():
        s_w = s.reindex(week_idx, method="ffill")
        m52 = s_w.rolling(52, min_periods=12).mean()
        sd52 = s_w.rolling(52, min_periods=12).std() + 1e-8
        features[f"{name}_z"] = ((s_w - m52) / sd52).shift(1)
    return features.dropna()


def walk_forward(X, y, model_fn, min_train=104, step=52):
    preds = pd.Series(np.nan, index=y.index)
    common = X.dropna().index.intersection(y.dropna().index)
    X = X.reindex(common)
    y = y.reindex(common)
    for i in range(min_train, len(common)):
        X_tr, y_tr = X.iloc[:i], y.iloc[:i]
        X_te, y_te = X.iloc[i:i+step], y.iloc[i:i+step]
        if len(X_tr) < 52 or len(X_te) == 0:
            continue
        try:
            m = model_fn()
            m.fit(X_tr.values, y_tr.values)
            preds.iloc[i:i+step] = m.predict(X_te.values)
        except Exception:
            pass
    return preds


def evaluate(y_true, y_pred):
    valid = y_pred.dropna()
    if len(valid) < 10:
        return {"ic": np.nan, "r2": np.nan, "hit": np.nan}
    y = y_true.reindex(valid.index)
    ss_res = ((y - valid) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    ic = valid.corr(y, method="spearman")
    hit = ((valid > 0) == (y > 0)).mean()
    return {"ic": ic, "r2": r2, "hit": hit}


def main():
    print("=" * 70)
    print("扩展特征工程 + 树模型 — 12特征基线 vs 扩展后~50特征")
    print("=" * 70)

    gold, data = load_data()
    w = gold.resample("W-FRI").last().dropna()
    fwd1 = w.pct_change().shift(-1)
    fwd2 = w.pct_change().shift(-2)

    # 构造两套特征
    X_base = build_baseline_features(gold, data, w.index)
    X_ext = build_extended_features(gold, data, w.index)
    print(f"\n  基线特征: {X_base.shape[1]}列")
    print(f"  扩展特征: {X_ext.shape[1]}列")

    models = {
        "Ridge (线性基线)": lambda: Ridge(alpha=1.0),
        "RandomForest(n=50)": lambda: RandomForestRegressor(
            n_estimators=50, max_depth=3, min_samples_leaf=40,
            random_state=42, n_jobs=-1),
        "GBR(n=30, d=2)": lambda: GradientBoostingRegressor(
            n_estimators=30, max_depth=2, learning_rate=0.05,
            min_samples_leaf=40, random_state=42),
    }

    print("\n[评估] Walk-forward (train=104周, step=52周)")
    print("=" * 80)
    print(f"{'配置':<35} {'FWD1 IC':>10} {'FWD1 R²':>10} {'FWD1 Hit':>9} {'FWD2 IC':>10} {'FWD2 R²':>10}")
    print("-" * 80)

    results = {}
    for X_name, X in [("12特征(基线)", X_base), (f"扩展{X_ext.shape[1]}特征", X_ext)]:
        for mname, mfn in models.items():
            for hname, y in [("FWD1", fwd1), ("FWD2", fwd2)]:
                key = f"{X_name}|{mname}|{hname}"
                preds = walk_forward(X, y, mfn, min_train=104, step=52)
                m = evaluate(y, preds)
                results[key] = m
                print(f"  {key:<35} {m['ic']:>+10.4f} {m['r2']:>+10.4f} {m['hit']:>9.1%} "
                      f"{results.get(f'{X_name}|{mname}|FWD2', {}).get('ic', np.nan):>+10.4f} "
                      f"{results.get(f'{X_name}|{mname}|FWD2', {}).get('r2', np.nan):>+10.4f}")

    # 输出CSV
    rows = []
    for k, v in results.items():
        X_name, mname, hname = k.split("|")
        rows.append({
            "feature_set": X_name, "model": mname, "horizon": hname,
            "ic": v["ic"], "r2": v["r2"], "hit_rate": v["hit"],
        })
    pd.DataFrame(rows).to_csv(f"{OUTPUT_DIR}/feature_eng_tree.csv", index=False, float_format="%.4f")
    print(f"\n✓ 已输出: {OUTPUT_DIR}/feature_eng_tree.csv")

    # 关键对比
    print("\n" + "=" * 70)
    print("关键对比 (FWD1 IC)")
    print("=" * 70)
    for mname in models:
        b = results[f"12特征(基线)|{mname}|FWD1"]["ic"]
        e = results[f"扩展{X_ext.shape[1]}特征|{mname}|FWD1"]["ic"]
        diff = e - b
        better = "扩展胜" if abs(e) > abs(b) else "基线胜"
        print(f"  {mname:<22} 基线={b:+.4f}  扩展={e:+.4f}  Δ={diff:+.4f}  {better}")


if __name__ == "__main__":
    main()
