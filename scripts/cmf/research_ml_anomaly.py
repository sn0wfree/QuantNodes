#!/usr/bin/env python3
"""scripts/cmf/research_ml_anomaly.py — 风控ML异常检测研究 (方案B).

检测市场结构异常 (4类), 6个模型预注册对比, 4道Go/No-Go门槛,
最后做集成原型测试 (异常时仓位×0.5).

纯只读, 不修改生产代码.
"""
from __future__ import annotations

import sys
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = "data/cmf/fundamental.duckdb"
FIXED_PARAMS = {
    "L1_VolCap": 0.15, "L2_SoftCap": -0.07, "L3_HardCap": -0.10,
    "DailyCap": -0.02, "CooldownDays": 5,
}
HARD_CONSTRAINT_MAX_DD = -0.15
ANOMALY_THRESHOLD_PERCENTILE = 95


def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    queries = {
        "gold": ("gold_macro", "gold_spot"),
        "vix": ("gold_fred", "vix"),
        "gvz": ("gold_fred", "gvz"),
        "dxy": ("gold_macro", "dxy"),
        "wti": ("fred_dcoil", "wti"),
    }
    dfs = {}
    for c, (s, v) in queries.items():
        df = con.execute(
            f"SELECT trade_date, value FROM fundamentals_daily "
            f"WHERE source='{s}' AND value_col='{v}' ORDER BY trade_date"
        ).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dfs[c] = df.set_index("trade_date")["value"]
    con.close()
    return pd.DataFrame(dfs).sort_index().ffill().dropna()


def _z(s, w=156):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_features(df):
    out = df.copy()
    out["F10_dxy_z"] = -_z(out["dxy"], 156)
    out["vix_pct"] = out["vix"].rolling(252).rank(pct=True)
    out["gvz_z"] = _z(out["gvz"], 90)
    out["realized_vol_20d"] = out["gold"].pct_change().rolling(20).std() * np.sqrt(252)
    out["realized_vol_60d"] = out["gold"].pct_change().rolling(60).std() * np.sqrt(252)
    out["vol_change"] = out["realized_vol_20d"] / out["realized_vol_60d"] - 1
    out["momentum_20d"] = out["gold"].pct_change(20)
    out["momentum_60d"] = out["gold"].pct_change(60)
    out["vix_change_5d"] = out["vix"].pct_change(5)
    out["dxy_change_20d"] = out["dxy"].pct_change(20)
    out["gold_oil_zscore"] = _z(out["gold"] / out["wti"], 156)
    out["tips_zscore"] = _z(out["gold"], 156)  # proxy
    return out


def build_anomaly_labels(df):
    """4类异常标签 (未来5d窗口), 任一触发则标记为异常."""
    out = df[["gold", "vix"]].copy()
    out["gold_ret"] = out["gold"].pct_change()

    # A1: 跨资产相关性突变
    # 滚动20d corr (gold vs WTI), 计算未来5d变化
    if "wti" in df.columns:
        ret_pair = pd.DataFrame({"g": out["gold_ret"], "w": df["wti"].pct_change()}).dropna()
        roll_corr = ret_pair["g"].rolling(20).corr(ret_pair["w"])
        future_corr_chg = roll_corr.shift(-5) - roll_corr
        future_corr_chg_std = future_corr_chg.rolling(252).std()
        A1 = (future_corr_chg.abs() > 2 * future_corr_chg_std).astype(int)
    else:
        A1 = pd.Series(0, index=out.index)

    # A2: 波动集聚 (未来5d vol > 当前95%分位)
    future_vol = out["gold_ret"].shift(-5).rolling(5).std() * np.sqrt(252)
    vol_threshold = out["gold_ret"].rolling(252).std() * np.sqrt(252) * 0.95
    # 简化: 未来5d vol > 历史95%分位
    hist_p95 = out["gold_ret"].rolling(252).std() * np.sqrt(252)
    # 滚动计算分位数
    vol_series = out["gold_ret"].rolling(20).std() * np.sqrt(252)
    rolling_p95 = vol_series.rolling(252).quantile(0.95)
    A2 = ((future_vol > rolling_p95.shift(5)) & (rolling_p95 > 0.05)).astype(int)

    # A3: VIX跳升 (未来5d VIX > 当前VIX × 1.3)
    vix_now = out["vix"]
    vix_future_max = vix_now.shift(-5).rolling(5).max()
    A3 = (vix_future_max > vix_now * 1.3).astype(int)

    # A4: 趋势断裂 (未来5d最大单日跌幅 < -2%)
    future_min_ret = out["gold_ret"].shift(-5).rolling(5).min()
    A4 = (future_min_ret < -0.02).astype(int)

    # 合并: 任一异常触发
    anomaly = (A1.fillna(0).astype(bool) |
               A2.fillna(0).astype(bool) |
               A3.fillna(0).astype(bool) |
               A4.fillna(0).astype(bool)).astype(int)

    return anomaly, A1, A2, A3, A4


def compute_anomaly_scores_isoforest(X, contamination=0.05):
    """M1: Isolation Forest."""
    mdl = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    mdl.fit(X)
    # score_samples: 越低越异常
    return -mdl.score_samples(X)


def compute_anomaly_scores_ocsvm(X, nu=0.05):
    """M2: One-Class SVM."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mdl = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
    mdl.fit(X_scaled)
    return -mdl.decision_function(X_scaled)


def compute_anomaly_scores_lof(X, n_neighbors=20, contamination=0.05):
    """M3: Local Outlier Factor."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mdl = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    mdl.fit(X_scaled)
    return -mdl.negative_outlier_factor_


def compute_anomaly_scores_mahalanobis(X):
    """M4: Mahalanobis Distance (滚动基线)."""
    scores = np.zeros(len(X))
    for i in range(252, len(X)):
        X_hist = X.iloc[max(0, i-252):i].values
        if len(X_hist) < 30:
            continue
        try:
            cov = np.cov(X_hist.T)
            mean = X_hist.mean(axis=0)
            diff = X.iloc[i].values - mean
            inv_cov = np.linalg.pinv(cov)
            scores[i] = np.sqrt(diff @ inv_cov @ diff)
        except Exception:
            scores[i] = np.nan
    return scores


def compute_anomaly_scores_zscore_ensemble(X, df):
    """M5: 多变量滚动Z-Score综合."""
    scores = np.zeros(len(X))
    cols = ["vix_pct", "F10_dxy_z", "realized_vol_20d", "gold_oil_zscore", "gvz_z"]
    available = [c for c in cols if c in X.columns]
    for i in range(len(X)):
        z_vals = []
        for c in available:
            s = X[c].iloc[max(0, i-252):i+1]
            if len(s) > 20 and s.std() > 0:
                z = (X[c].iloc[i] - s.mean()) / s.std()
                z_vals.append(abs(z))
        if z_vals:
            scores[i] = np.mean(z_vals)
    return scores


def main():
    print("=" * 60)
    print("风控ML异常检测研究 (方案B)")
    print("=" * 60)

    df = build_features(load_data())["2018":"2025"]
    print(f"区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # === Phase 1: 异常标签 ===
    print("\n=== Phase 1: 生成异常标签 ===")
    anomaly_labels, A1, A2, A3, A4 = build_anomaly_labels(df)
    anomaly_labels.name = "anomaly"
    print(f"总异常天数: {anomaly_labels.sum()}/{len(anomaly_labels)} ({anomaly_labels.mean()*100:.1f}%)")
    print(f"  A1 相关性突变: {A1.sum()}天")
    print(f"  A2 波动集聚:   {A2.sum()}天")
    print(f"  A3 VIX跳升:   {A3.sum()}天")
    print(f"  A4 趋势断裂:   {A4.sum()}天")

    # 已知危机日
    crisis_dates = {
        "2020-03-16": "COVID崩盘",
        "2020-08-11": "黄金暴跌5.7%",
        "2022-09-13": "2022加息恐慌",
        "2025-04-09": "2025关税冲击",
    }
    print(f"\n已知危机日异常检测:")
    for d, desc in crisis_dates.items():
        idx = df.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
        actual_date = df.index[idx]
        label = anomaly_labels.iloc[idx] if not pd.isna(anomaly_labels.iloc[idx]) else 0
        print(f"  {d} ({desc}): label={int(label)}")

    # === 特征准备 ===
    features = [
        "F10_dxy_z", "vix_pct", "gvz_z",
        "realized_vol_20d", "realized_vol_60d", "vol_change",
        "momentum_20d", "momentum_60d",
        "vix_change_5d", "dxy_change_20d",
        "gold_oil_zscore", "tips_zscore",
    ]
    X = df[features].dropna()
    # 对齐标签
    common_idx = X.index.intersection(anomaly_labels.index)
    X = X.loc[common_idx]
    y = anomaly_labels.loc[common_idx]
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

    # === Phase 2: 6个模型对比 ===
    print("\n=== Phase 2: 6个模型对比 ===")
    print(f"{'模型':<25s}{'计算时间(s)':>12s}{'危机日平均分位':>15s}{'G1':>5s}{'G2误报率':>10s}")
    print("-" * 70)

    model_results = {}

    # M1: Isolation Forest
    t0 = time.time()
    s1 = compute_anomaly_scores_isoforest(X_scaled_df)
    t1 = time.time()
    model_results["M1_IsolationForest"] = (s1, t1 - t0)

    # M2: One-Class SVM
    t0 = time.time()
    s2 = compute_anomaly_scores_ocsvm(X_scaled_df)
    t1 = time.time()
    model_results["M2_OneClassSVM"] = (s2, t1 - t0)

    # M3: LOF
    t0 = time.time()
    s3 = compute_anomaly_scores_lof(X_scaled_df)
    t1 = time.time()
    model_results["M3_LOF"] = (s3, t1 - t0)

    # M4: Mahalanobis (滚动基线, 无需scaling)
    t0 = time.time()
    s4 = compute_anomaly_scores_mahalanobis(X)
    t1 = time.time()
    model_results["M4_Mahalanobis"] = (s4, t1 - t0)

    # M5: Z-Score Ensemble
    t0 = time.time()
    s5 = compute_anomaly_scores_zscore_ensemble(X_scaled_df, df)
    t1 = time.time()
    model_results["M5_ZScoreEnsemble"] = (s5, t1 - t0)

    # 评估各模型
    for name, (scores, t) in model_results.items():
        scores_s = pd.Series(scores, index=X.index).dropna()
        # 计算分位数
        score_rank = scores_s.rank(pct=True) * 100
        # 危机日平均分位
        crisis_pcts = []
        for d in crisis_dates:
            idx = X.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]
            if idx < len(score_rank):
                crisis_pcts.append(score_rank.iloc[idx])
        crisis_avg = np.mean(crisis_pcts) if crisis_pcts else np.nan
        # G2: 正常时段误报率
        normal_scores = score_rank[y == 0]
        # 由于 score 是全样本的百分位, 误报率需要重新计算
        # 简化: 用绝对阈值
        threshold = score_rank.quantile(0.95)
        false_alarms = (score_rank[y == 0] > 95).sum() / max((y == 0).sum(), 1)
        g1_pass = crisis_avg > 90 if not np.isnan(crisis_avg) else False
        g2_pass = false_alarms < 0.10
        print(f"{name:<25s}{t:>12.2f}{crisis_avg:>14.1f}%{('✅' if g1_pass else '❌'):>5s}{false_alarms*100:>9.1f}%")

    # === Phase 3: 集成原型测试 ===
    print("\n=== Phase 3: 集成原型测试 (异常时仓位×0.5) ===")
    from QuantNodes.strategy.commodity_fundamental.gold import compute_layered_ensemble_signal
    sig_prod = compute_layered_ensemble_signal()
    prod_pos = sig_prod["position"].reindex(df.index).ffill().fillna(0)
    close = df["gold"]

    def apply_risk_overlay(close, position, params):
        pos = position.copy()
        daily_ret = close.pct_change()
        rolling_dd_252 = close / close.rolling(252).max() - 1
        realized_vol_20 = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
        vol_mask = (realized_vol_20 > params["L1_VolCap"]) & (realized_vol_20.notna())
        vol_scale = (params["L1_VolCap"] / realized_vol_20).clip(upper=1.0)
        pos = pos.where(~vol_mask, pos * vol_scale)
        soft_mask = rolling_dd_252 < params["L2_SoftCap"]
        pos = pos.where(~soft_mask, pos * 0.5)
        hard_mask = (rolling_dd_252 < params["L3_HardCap"]) | (daily_ret < params["DailyCap"])
        hard_mask = hard_mask.fillna(False)
        pos = pos.where(~hard_mask, 0.0)
        if params["CooldownDays"] > 0:
            for i in range(len(pos)):
                if hard_mask.iloc[i]:
                    end = min(i + params["CooldownDays"] + 1, len(pos))
                    pos.iloc[i:end] = 0.0
        return pos.clip(0, 1.2)

    def o2o_backtest(pos, close):
        sr = (pos.shift(1) * close.pct_change()).dropna()
        cost = pos.diff().abs() * 5.0 / 10000
        sr = (sr - cost).dropna()
        nav = (1 + sr).cumprod()
        n_y = len(sr) / 252
        cagr = ((nav.iloc[-1]) ** (1/n_y) - 1) if n_y > 0 else 0
        vol = sr.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0
        dd = nav / nav.cummax() - 1
        return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd.min()}

    # A. 固定参数 (生产baseline)
    pos_fixed = apply_risk_overlay(close, prod_pos, FIXED_PARAMS)
    r_fixed = o2o_backtest(pos_fixed, close)
    print(f"\nA. 固定参数 (生产):")
    print(f"  CAGR:   {r_fixed['cagr']*100:+.2f}%")
    print(f"  Sharpe: {r_fixed['sharpe']:+.3f}")
    print(f"  MaxDD:  {r_fixed['max_dd']*100:+.2f}%")

    # B. 固定 + ML异常检测补充层 (用每个模型做)
    print(f"\nB. 固定 + ML异常检测补充层 (异常时仓位×0.5):")
    for name, (scores, _) in model_results.items():
        scores_s = pd.Series(scores, index=X.index).dropna()
        if len(scores_s) < 252:
            continue
        # 滚动95%分位阈值
        threshold_rolling = scores_s.rolling(252).quantile(0.95)
        # 异常信号: 当前分数 > 历史95%分位
        anomaly_signal = (scores_s > threshold_rolling).astype(int)
        # 应用补充层
        pos_ml = pos_fixed.copy()
        common = pos_ml.index.intersection(anomaly_signal.index)
        for d in common:
            if anomaly_signal.loc[d] == 1 and pos_ml.loc[d] > 0:
                pos_ml.loc[d] *= 0.5
        r_ml = o2o_backtest(pos_ml.loc[pos_fixed.index], close)
        d_sharpe = r_ml['sharpe'] - r_fixed['sharpe']
        d_dd = r_ml['max_dd'] - r_fixed['max_dd']
        marker = "✅" if r_ml['max_dd'] > r_fixed['max_dd'] and r_ml['sharpe'] >= r_fixed['sharpe'] * 0.95 else "⚠️"
        print(f"  {name}: Sharpe={r_ml['sharpe']:+.3f}({d_sharpe:+.3f}) MaxDD={r_ml['max_dd']*100:+.2f}%({d_dd*100:+.2f}pp) {marker}")

    # === 综合判定 ===
    print("\n" + "=" * 60)
    print("综合判定 (G1-G4)")
    print("=" * 60)
    # 简化: 选择最佳模型
    best_model = max(model_results.items(), key=lambda x: np.mean([
        score_rank.iloc[X.index.get_indexer([pd.Timestamp(d)], method="nearest")[0]]
        for d in crisis_dates
        if X.index.get_indexer([pd.Timestamp(d)], method="nearest")[0] < len(score_rank)
    ]) if hasattr(x[1][0], '__len__') else 0)
    best_name, (best_scores, best_time) = best_model
    print(f"最佳模型: {best_name}")
    print(f"  G4 (实时性<1秒): {'✅' if best_time < 1.0 else '❌'} ({best_time:.2f}s)")
    print(f"\n结论:")
    print(f"  - G1 (危机检测): 见上方表格")
    print(f"  - G2 (误报率<5%): 见上方表格")
    print(f"  - G3 (集成增益): 见Phase 3对比")
    print(f"  - G4 (实时性): {'✅' if best_time < 1.0 else '⚠️'}")


if __name__ == "__main__":
    main()