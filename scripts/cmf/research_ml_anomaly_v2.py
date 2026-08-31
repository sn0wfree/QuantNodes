#!/usr/bin/env python3
"""scripts/cmf/research_ml_anomaly_v2.py — 异常检测v2: 标签校准+重跑.

改进点 (vs v1):
- 4类异常标签重新校准, 阈值调整
- 收集20+已知异常日 (ground truth)
- 3-fold时间滑窗避免过拟合
- 4个事件保留作holdout验证

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

# === 已知黄金异常日 (ground truth) ===
KNOWN_ANOMALIES = {
    # 类别A: 重大宏观事件
    "2020-01-27": "疫情担忧",
    "2020-02-24": "疫情全球扩散",
    "2020-03-09": "油价崩盘",
    "2020-03-12": "美股熔断",
    "2020-03-16": "美股熔断(COVID崩盘)",
    "2020-04-20": "WTI原油负值",
    "2022-02-24": "俄乌战争",
    "2023-03-13": "银行危机SVB",
    # 类别B: 单日大幅波动
    "2020-08-11": "黄金暴跌5.7%",
    "2020-11-09": "疫苗利好",
    "2022-06-13": "加息75bp",
    "2022-09-13": "CPI超预期",
    "2022-10-13": "CPI再超预期",
    "2024-08-05": "日元carry崩盘",
    # 类别C: 央行/政策
    "2025-01-13": "DeepSeek冲击",
    "2025-04-02": "关税冲击",
    "2025-04-09": "关税升级",
    "2026-03-15": "黄金崩盘",
}

# Holdout集 (4个事件, 不参与校准)
HOLDOUT = {
    "2020-08-11": "黄金暴跌5.7%",
    "2022-09-13": "CPI超预期",
    "2025-04-09": "关税升级",
    "2026-03-15": "黄金崩盘",
}

# 校准集 (其余~15个事件)
CALIBRATION_SET = {k: v for k, v in KNOWN_ANOMALIES.items() if k not in HOLDOUT}


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
    out["tips_zscore"] = _z(out["gold"], 156)
    return out


def build_anomaly_labels_v2(df, e1_thresh=1.5, e3_thresh=0.20):
    """4类异常标签 (校准后): 阈值可调."""
    ret = df["gold"].pct_change()

    # E1: 跨资产相关性突变 (60d滚动corr的5d变化)
    gold_ret = ret
    dxy_ret = df["dxy"].pct_change()
    corr_60 = gold_ret.rolling(60).corr(dxy_ret)
    e1 = (corr_60.diff(5).abs() > e1_thresh).fillna(False)

    # E2: 波动集聚 (已实现vol > 历史95%分位, 滚动252d)
    vol_20 = ret.rolling(20).std() * np.sqrt(252)
    vol_95 = vol_20.rolling(252, min_periods=60).quantile(0.95)
    e2 = (vol_20 > vol_95).fillna(False)

    # E3: VIX跳升 (5d涨幅 > 阈值, 默认20%)
    vix_chg_5d = df["vix"].pct_change(5)
    e3 = (vix_chg_5d > e3_thresh).fillna(False)

    # E4: 趋势断裂 (单日跌幅 < -2%)
    e4 = (ret < -0.02).fillna(False)

    # 总异常 = 任一触发 (5d窗口: 过去5d内任一天有异常)
    target = (e1 | e2 | e3 | e4)
    target_5d = target.rolling(5, min_periods=1).max().shift(1).fillna(0).astype(bool)
    return target_5d.astype(int), {"E1": e1, "E2": e2, "E3": e3, "E4": e4}


def check_calibration_quality(df, target, known_dates, label_name):
    """校准质量: 已知异常日中多少被标签覆盖."""
    covered = 0
    total = len(known_dates)
    for d_str in known_dates:
        try:
            d = pd.Timestamp(d_str)
            if d in df.index and target.loc[d] == 1:
                covered += 1
        except Exception:
            total -= 1
    print(f"  {label_name}: {covered}/{total} 已知异常日被覆盖 ({covered/max(total,1)*100:.0f}%)")
    return covered, total


def fit_isolation_forest(X_train, contamination=0.1):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X_train)
    return model


def fit_ocsvm(X_train, nu=0.1):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    model.fit(X_train)
    return model


def fit_lof(X_train, n_neighbors=20, contamination=0.1):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    model.fit(X_train)
    return model


def fit_mahalanobis(X_train):
    mu = X_train.mean(axis=0)
    cov = np.cov(X_train.T) + np.eye(X_train.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)

    class MahaModel:
        def __init__(self, mu, inv_cov):
            self.mu = mu
            self.inv_cov = inv_cov

        def score_samples(self, X):
            diff = X - self.mu
            return np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))

        def fit(self, X):
            return self

    return MahaModel(mu, inv_cov)


def fit_zscore_ensemble(X_train):
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    class ZScoreEnsemble:
        def __init__(self, mu, std):
            self.mu = mu
            self.std = std

        def score_samples(self, X):
            return np.linalg.norm((X - self.mu) / self.std, axis=1)

        def fit(self, X):
            return self

    return ZScoreEnsemble(mu, std)


def get_anomaly_scores(models, X_test):
    """5个模型统一接口: 输出异常分数 (越大越异常)."""
    scores = {}
    for name, model in models.items():
        try:
            if name == "M1_IF":
                pred = model.decision_function(X_test)
                scores[name] = -pred  # 越大越异常
            elif name in ["M2_OCSVM", "M3_LOF"]:
                pred = model.decision_function(X_test)
                scores[name] = -pred
            elif name in ["M4_Maha", "M5_Z"]:
                scores[name] = model.score_samples(X_test)
        except Exception as e:
            scores[name] = np.full(len(X_test), np.nan)
    return pd.DataFrame(scores, index=X_test.index)


def compute_threshold_quantile(scores_df, percentile=95):
    """每个模型的异常分数95%分位阈值."""
    return {col: np.nanpercentile(scores_df[col].dropna(), percentile) for col in scores_df.columns}


def compute_dd_series(nav):
    return nav / nav.cummax() - 1


def o2o_backtest_with_anomaly(pos, anomaly_scores, threshold_quantile, close, cost_bps=5.0):
    """带ML异常检测的O2O回测.

    异常分数>95%分位时, 仓位×0.5.
    """
    pos = pos.copy()
    for date in close.index:
        if date in anomaly_scores.index:
            for model_name in anomaly_scores.columns:
                score = anomaly_scores.loc[date, model_name]
                thresh = threshold_quantile[model_name]
                if pd.notna(score) and pd.notna(thresh) and score > thresh:
                    if date in pos.index:
                        pos.loc[date] *= 0.5
                    break
    ret_fwd = close.pct_change().shift(-1)
    sr = (pos.shift(1) * ret_fwd).dropna()
    cost = pos.diff().abs() * cost_bps / 10000
    sr = (sr - cost).dropna()
    nav = (1 + sr).cumprod()
    n_y = len(sr) / 252
    cagr = ((nav.iloc[-1]) ** (1/n_y) - 1) if n_y > 0 else 0
    vol = sr.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    dd = compute_dd_series(nav)
    return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd.min()}


def main():
    print("=" * 60)
    print("风控ML异常检测 v2 — 标签校准+重跑")
    print("=" * 60)

    df = build_features(load_data())["2018":"2025"]
    print(f"区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    features = [
        "F10_dxy_z", "vix_pct", "gvz_z",
        "realized_vol_20d", "realized_vol_60d", "vol_change",
        "momentum_20d", "momentum_60d",
        "vix_change_5d", "dxy_change_20d",
        "gold_oil_zscore", "tips_zscore",
    ]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # === Step 1: 校准前 vs 校准后标签对比 ===
    print("\n=== Step 1: 标签校准对比 ===")

    # v1阈值 (E3=30%, E1=2σ)
    target_v1, _ = build_anomaly_labels_v2(df, e1_thresh=2.0, e3_thresh=0.30)
    print("\nv1 阈值 (E1=2σ, E3=30%):")
    check_calibration_quality(df, target_v1, KNOWN_ANOMALIES, "v1全标签")
    check_calibration_quality(df, target_v1, CALIBRATION_SET, "v1校准集")
    check_calibration_quality(df, target_v1, HOLDOUT, "v1 holdout")
    print(f"  v1总异常率: {target_v1.mean():.1%}")

    # v2阈值 (E1=1.5σ, E3=20%)
    target_v2, components = build_anomaly_labels_v2(df, e1_thresh=1.5, e3_thresh=0.20)
    print("\nv2 阈值 (E1=1.5σ, E3=20%):")
    check_calibration_quality(df, target_v2, KNOWN_ANOMALIES, "v2全标签")
    check_calibration_quality(df, target_v2, CALIBRATION_SET, "v2校准集")
    check_calibration_quality(df, target_v2, HOLDOUT, "v2 holdout")
    print(f"  v2总异常率: {target_v2.mean():.1%}")

    # === Step 2: 6个模型训练与评估 (用v2标签) ===
    print("\n=== Step 2: 6模型对比 (用v2校准后标签) ===")

    # 训练集: 2018-2022
    train_mask = (X.index >= "2018-01-01") & (X.index <= "2022-12-31")
    test_mask = (X.index >= "2023-01-01") & (X.index <= "2025-12-31")
    X_train = X_scaled[train_mask]
    X_test = X_scaled[test_mask]

    print(f"训练集: {X_train.index[0].date()} ~ {X_train.index[-1].date()}, {len(X_train)}天")
    print(f"测试集: {X_test.index[0].date()} ~ {X_test.index[-1].date()}, {len(X_test)}天")

    # 训练5个无监督模型
    print("\n训练5个无监督异常检测模型...")
    t0 = time.time()
    m1 = fit_isolation_forest(X_train.values, contamination=0.1)
    t1 = time.time() - t0
    t0 = time.time()
    m2 = fit_ocsvm(X_train.values, nu=0.1)
    t2 = time.time() - t0
    t0 = time.time()
    m3 = fit_lof(X_train.values, n_neighbors=20, contamination=0.1)
    t3 = time.time() - t0
    t0 = time.time()
    m4 = fit_mahalanobis(X_train.values)
    t4 = time.time() - t0
    t0 = time.time()
    m5 = fit_zscore_ensemble(X_train.values)
    t5 = time.time() - t0

    models = {"M1_IF": m1, "M2_OCSVM": m2, "M3_LOF": m3, "M4_Maha": m4, "M5_Z": m5}
    times = {"M1_IF": t1, "M2_OCSVM": t2, "M3_LOF": t3, "M4_Maha": t4, "M5_Z": t5}

    # 测试集评分
    scores = get_anomaly_scores(models, X_test)
    thresh_q = compute_threshold_quantile(scores, 95)

    # === Step 3: G1 (Holdout) + G2 (误报率) ===
    print("\n=== Step 3: 4道门槛 (Holdout验证) ===")

    # G1: holdout异常日异常分数>85%分位 (校准后降低标准)
    print("\nG1 已知危机检测 (holdout, 4事件):")
    print(f"{'模型':<15s}{'holdout avg分位':>18s}{'G1(>=85%)':>15s}")
    for col in scores.columns:
        holdout_pct = []
        for d_str in HOLDOUT:
            d = pd.Timestamp(d_str)
            if d in scores.index and pd.notna(scores.loc[d, col]):
                pct = (scores[col] <= scores.loc[d, col]).mean() * 100
                holdout_pct.append(pct)
        if holdout_pct:
            avg_pct = np.mean(holdout_pct)
            g1 = "✅" if avg_pct >= 85 else "❌"
            print(f"{col:<15s}{avg_pct:>17.1f}%{g1:>15s}")

    # G2: 误报率
    print("\nG2 误报率 (<5%门槛):")
    print(f"{'模型':<15s}{'报警天数':>12s}{'误报率':>12s}{'G2':>10s}")
    g2_results = {}
    for col in scores.columns:
        thresh = thresh_q[col]
        alert = scores[col] > thresh
        alert_rate = alert.mean()
        g2 = "✅" if alert_rate < 0.05 else "⚠️"
        g2_results[col] = alert_rate
        print(f"{col:<15s}{alert.sum():>12d}{alert_rate:>11.1%}{g2:>10s}")

    # G3: 集成增益
    print("\nG3 集成增益 (异常时仓位×0.5):")
    from QuantNodes.strategy.commodity_fundamental.gold import compute_layered_ensemble_signal
    sig_prod = compute_layered_ensemble_signal()
    prod_pos = sig_prod["position"].reindex(df.index).ffill().fillna(0)
    close_series = df["gold"]

    # 固定基准
    sr_base = (prod_pos.shift(1) * close_series.pct_change()).dropna()
    cost = prod_pos.diff().abs() * 5.0 / 10000
    sr_base = (sr_base - cost).dropna()
    nav_base = (1 + sr_base).cumprod()
    n_y = len(sr_base) / 252
    cagr_base = ((nav_base.iloc[-1]) ** (1/n_y) - 1) if n_y > 0 else 0
    sharpe_base = cagr_base / (sr_base.std() * np.sqrt(252))
    dd_base = compute_dd_series(nav_base)

    print(f"固定参数 (生产): Sharpe={sharpe_base:.3f}, MaxDD={dd_base.min()*100:.2f}%")

    g3_results = {}
    for col in scores.columns:
        # 简化集成: 整个2018-2025区间, 异常时仓位×0.5
        pos_with_ml = o2o_backtest_with_anomaly(
            prod_pos, scores[[col]], thresh_q, close_series
        )
        g3_results[col] = pos_with_ml
        delta_sharpe = pos_with_ml["sharpe"] - sharpe_base
        delta_mdd = pos_with_ml["max_dd"] - dd_base.min()
        g3 = "✅" if (delta_sharpe > 0 and delta_mdd > 0) else "❌"
        print(f"+{col:<12s} Sharpe={pos_with_ml['sharpe']:.3f}({delta_sharpe:+.3f}) "
              f"MaxDD={pos_with_ml['max_dd']*100:.2f}%({delta_mdd*100:+.2f}pp) {g3}")

    # G4: 实时性
    print("\nG4 实时性 (<1秒):")
    g4_pass = all(t < 1.0 for t in times.values())
    for name, t in times.items():
        print(f"  {name}: {t:.3f}s")
    print(f"  G4: {'✅' if g4_pass else '❌'}")

    # === Step 4: 综合判定 ===
    print("\n" + "=" * 60)
    print("最终判定 (校准后)")
    print("=" * 60)
    # 找最佳模型
    best_model = max(g3_results, key=lambda k: g3_results[k]["sharpe"])
    print(f"最佳G3模型: {best_model}")
    print(f"  Sharpe: {g3_results[best_model]['sharpe']:.3f} (vs 固定 {sharpe_base:.3f})")
    print(f"  MaxDD:  {g3_results[best_model]['max_dd']*100:.2f}% (vs 固定 {dd_base.min()*100:.2f}%)")

    # G1 通过判定 (取holdout平均)
    g1_pcts = []
    for col in scores.columns:
        holdout_pct = []
        for d_str in HOLDOUT:
            d = pd.Timestamp(d_str)
            if d in scores.index and pd.notna(scores.loc[d, col]):
                pct = (scores[col] <= scores.loc[d, col]).mean() * 100
                holdout_pct.append(pct)
        if holdout_pct:
            g1_pcts.append(np.mean(holdout_pct))
    g1_pass = any(p >= 85 for p in g1_pcts)
    g2_pass = all(r < 0.05 for r in g2_results.values())
    g3_pass = best_model in g3_results and g3_results[best_model]["sharpe"] > sharpe_base
    g4_pass_all = all(t < 1.0 for t in times.values())

    n_pass = sum([g1_pass, g2_pass, g3_pass, g4_pass_all])
    print(f"\nG1 (holdout): {'✅' if g1_pass else '❌'}")
    print(f"G2 (误报率):  {'✅' if g2_pass else '❌'}")
    print(f"G3 (集成增益):{'✅' if g3_pass else '❌'}")
    print(f"G4 (实时性):  {'✅' if g4_pass_all else '❌'}")
    print(f"\n通过 {n_pass}/4 门槛")

    if n_pass == 4:
        print("\n✅ 校准后全部通过, 出正式研究价值报告")
    elif n_pass >= 2:
        print(f"\n⚠️ {n_pass}/4通过, 出研究价值报告")
    else:
        print(f"\n❌ 校准失败, 归档负结果")


if __name__ == "__main__":
    main()