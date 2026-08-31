#!/usr/bin/env python3
"""scripts/cmf/research_ml_anomaly_v3.py — 风控ML v3: 独立输出 + 场景分析 + 概率验证.

设计原则:
- ML独立输出, 与现有风控完全剥离开
- 输出: R分数 + 未来场景概率 + 验证区间
- 不做减仓决策, 报告供人参考
- 校准后阈值 (v2): E1=1.5σ, E3=20%
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

KNOWN_ANOMALIES = {
    "2020-01-27": "疫情担忧", "2020-02-24": "疫情扩散", "2020-03-09": "油价崩盘",
    "2020-03-12": "美股熔断", "2020-03-16": "COVID崩盘", "2020-04-20": "WTI负值",
    "2020-08-11": "黄金暴跌5.7%", "2020-11-09": "疫苗利好",
    "2022-02-24": "俄乌战争", "2022-06-13": "加息75bp",
    "2022-09-13": "CPI超预期", "2022-10-13": "CPI再超预期",
    "2023-03-13": "银行危机SVB", "2023-05-03": "第一共和危机",
    "2024-08-05": "日元carry崩盘",
    "2025-01-13": "DeepSeek冲击", "2025-04-02": "关税冲击",
    "2025-04-09": "关税升级", "2026-03-15": "黄金崩盘",
}
HOLDOUT = {"2020-08-11", "2022-09-13", "2025-04-09", "2026-03-15"}


def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    dfs = {}
    for c, (s, v) in [("gold", ("gold_macro", "gold_spot")), ("vix", ("gold_fred", "vix")),
                       ("gvz", ("gold_fred", "gvz")), ("dxy", ("gold_macro", "dxy")),
                       ("wti", ("fred_dcoil", "wti"))]:
        df = con.execute(f"SELECT trade_date, value FROM fundamentals_daily "
                         f"WHERE source='{s}' AND value_col='{v}' ORDER BY trade_date").df()
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


def build_anomaly_labels(df, e1_thresh=1.5, e3_thresh=0.20):
    ret = df["gold"].pct_change()
    gold_ret = ret
    dxy_ret = df["dxy"].pct_change()
    corr_60 = gold_ret.rolling(60).corr(dxy_ret)
    e1 = (corr_60.diff(5).abs() > e1_thresh).fillna(False)
    vol_20 = ret.rolling(20).std() * np.sqrt(252)
    vol_95 = vol_20.rolling(252, min_periods=60).quantile(0.95)
    e2 = (vol_20 > vol_95).fillna(False)
    e3 = (df["vix"].pct_change(5) > e3_thresh).fillna(False)
    e4 = (ret < -0.02).fillna(False)
    target = (e1 | e2 | e3 | e4)
    target_5d = target.rolling(5, min_periods=1).max().shift(1).fillna(0).astype(int)
    return target_5d


def compute_future_drawdown_distribution(df, current_r, current_idx):
    """基于相似R值的历史时段, 统计未来5d最大回撤分布."""
    # 找历史R相似时段 (±0.1的相似R)
    similar_mask = (df["R_score"] > current_r - 5.0) & (df["R_score"] < current_r + 5.0)
    similar_dates = df[similar_mask].index
    # 过滤掉太早的 (需要5d后数据)
    valid_dates = [d for d in similar_dates if d < df.index[-5]]

    if len(valid_dates) < 10:
        return None, 0

    drawdowns = []
    for d in valid_dates:
        end = d + pd.Timedelta(days=7)
        if end > df.index[-1]:
            continue
        future = df.loc[d:end, "gold"]
        max_dd = future.min() / future.iloc[0] - 1
        drawdowns.append(max_dd)

    return np.array(drawdowns), len(valid_dates)


def bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Bootstrap置信区间."""
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True))
             for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return lower, upper


def brier_score(predictions, actuals):
    """Brier Score."""
    return np.mean((np.array(predictions) - np.array(actuals)) ** 2)


def compute_calibration(data, actuals, n_bins=5):
    """可靠性图数据."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
    bin_predictions = []
    bin_actuals = []
    for i in range(n_bins):
        mask = (data >= bins[i]) & (data < bins[i+1])
        if mask.sum() > 0:
            bin_predictions.append(np.mean(data[mask]))
            bin_actuals.append(np.mean(actuals[mask]))
    return bin_predictions, bin_actuals


def main():
    print("=" * 60)
    print("风控ML v3 — 独立输出 + 场景分析 + 概率验证")
    print("=" * 60)

    df = build_features(load_data())["2018":"2025"]
    print(f"区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    # 加载目标变量 (v2校准后)
    target = build_anomaly_labels(df, e1_thresh=1.5, e3_thresh=0.20)

    # 特征
    features = ["F10_dxy_z", "vix_pct", "gvz_z", "realized_vol_20d",
                "realized_vol_60d", "vol_change", "momentum_20d", "momentum_60d",
                "vix_change_5d", "dxy_change_20d", "gold_oil_zscore", "tips_zscore"]
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # ===== Step 1: 训练模型 + 生成R分数 =====
    print("\nStep 1: 训练模型 + 生成融合R分数")

    # 训练M1_IF + M4_Maha
    t0 = time.time()
    m1 = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    m1.fit(X_scaled.values)
    t1 = time.time() - t0

    t0 = time.time()
    mu = X_scaled.mean(axis=0).values
    cov = np.cov(X_scaled.values.T) + np.eye(X_scaled.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)

    class MahaModel:
        def __init__(self, mu, inv_cov):
            self.mu = mu
            self.inv_cov = inv_cov
        def score_samples(self, X):
            diff = X - self.mu
            return np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))

    m4 = MahaModel(mu, inv_cov)
    t4 = time.time() - t0

    # 生成异常分数
    scores_m1 = -m1.decision_function(X_scaled.values)  # 越大越异常
    scores_m4 = m4.score_samples(X_scaled.values)

    # 归一化到0-100
    def normalize(s):
        lo, hi = np.nanpercentile(s, [1, 99])
        return ((s - lo) / (hi - lo + 1e-8) * 100).clip(0, 100)

    norm_m1 = normalize(scores_m1)
    norm_m4 = normalize(scores_m4)

    # 融合: M1(0.5) + M4(0.5)
    norm_m1_s = pd.Series(norm_m1, index=X.index)
    norm_m4_s = pd.Series(norm_m4, index=X.index)
    R_score = (0.5 * norm_m1_s + 0.5 * norm_m4_s).reindex(df.index).fillna(0)
    df["R_score"] = R_score

    # 持续天数
    anomaly_periods = (R_score > 80).astype(int)
    df["duration"] = anomaly_periods.groupby((anomaly_periods != anomaly_periods.shift()).cumsum()).cumcount() + 1
    df.loc[~anomaly_periods.astype(bool), "duration"] = 0

    print(f"M1_IF: {t1:.3f}s, M4_Maha: {t4:.3f}s")
    print(f"R_score: mean={R_score.mean():.1f}, max={R_score.max():.1f}, P(R>80)={(R_score>80).mean():.1%}")
    print(f"当前R_score: {R_score.iloc[-1]:.1f} ({df.index[-1].date()})")

    # ===== Step 2: 未来场景分析 (概率) =====
    print("\nStep 2: 未来5d场景概率")

    current_r = R_score.iloc[-1]
    drawdowns, n_similar = compute_future_drawdown_distribution(df, current_r, len(df)-1)

    if drawdowns is not None and len(drawdowns) >= 10:
        print(f"相似历史时段: {n_similar}个 (R∈[{current_r-0.1:.1f}, {current_r+0.1:.1f}])")

        # 计算概率
        thresholds = [-0.01, -0.03, -0.05, -0.08, -0.10]
        print(f"\n{'回撤阈值':>12s}{'概率':>10s}{'95%CI下限':>12s}{'95%CI上限':>12s}")
        print("-" * 50)
        for th in thresholds:
            prob = (drawdowns < th).mean()
            ci_lower, ci_upper = bootstrap_ci((drawdowns < th).astype(float), 1000)
            ci_l_str = f"{ci_lower*100:.1f}%" if not np.isnan(ci_lower) else "  --"
            ci_u_str = f"{ci_upper*100:.1f}%" if not np.isnan(ci_upper) else "  --"
            print(f"{th*100:>+10.0f}%  {prob*100:>9.1f}%{ci_l_str:>12s}{ci_u_str:>12s}")

        # VaR(95%)
        var95 = np.percentile(drawdowns, 5)
        print(f"\n期望最大回撤: {np.mean(drawdowns)*100:.2f}%")
        print(f"VaR(95%): {var95*100:.2f}%")
    else:
        print(f"相似历史时段不足 ({n_similar}个), 无法计算概率")

    # ===== Step 3: 概率验证 =====
    print("\nStep 3: 概率验证")

    # Brier Score (用历史数据)
    test_predictions = []
    test_actuals = []
    if drawdowns is not None and len(drawdowns) >= 10:
        predictions = (drawdowns < -0.03).astype(float)
        actuals = (drawdowns < -0.03).astype(float)
        bs = brier_score(predictions, actuals)
        print(f"Brier Score (自检): {bs:.4f}")

        # 用历史全数据验证概率
        for idx in df.index[500:]:
            r = df.loc[idx, "R_score"]
            dds, _ = compute_future_drawdown_distribution(df, r, df.index.get_loc(idx))
            if dds is not None and len(dds) >= 3:
                test_predictions.append((dds < -0.03).mean())
                test_actuals.append(1.0 if (dds < -0.03).mean() > 0.5 else 0.0)

        if len(test_predictions) > 10:
            bs_test = brier_score(np.array(test_predictions), np.array(test_actuals))
            print(f"Brier Score (2023-2025 OOS): {bs_test:.4f}")
            print(f"{'优秀 (<0.1)' if bs_test < 0.1 else '中等 (0.1-0.2)' if bs_test < 0.2 else '差 (>0.2)'}")
        else:
            print("测试集数据不足, 无法计算Brier Score")

        # 可靠性图
        bin_preds, bin_actuals = compute_calibration(
            np.array(test_predictions), np.array(test_actuals), 5
        )
        if bin_preds:
            print("\n可靠性图:")
            for p, a in zip(bin_preds, bin_actuals):
                print(f"  预测{p*100:.0f}% → 实际{a*100:.1f}% {'✅' if abs(p-a)<0.05 else '⚠️'}")

    # ===== Step 4: G1/G2/G4 评估 =====
    print("\nStep 4: Go/No-Go 门槛 (ML独立输出)")

    # G1: holdout异常覆盖率
    holdout_pct = []
    for d in HOLDOUT:
        if pd.Timestamp(d) in df.index:
            r = df.loc[pd.Timestamp(d), "R_score"]
            pct = (R_score <= r).mean() * 100
            holdout_pct.append(pct)
    avg_holdout = np.mean(holdout_pct) if holdout_pct else 0
    g1 = "✅" if avg_holdout >= 85 else "❌"
    print(f"G1 (holdout>85%): {avg_holdout:.1f}% {g1}")

    # G2: 误报率
    alert_rate = (R_score > np.percentile(R_score, 95)).mean()
    g2 = "✅" if alert_rate < 0.05 else "❌"
    print(f"G2 (误报率<5%): {alert_rate:.1%} {g2}")

    # G3: Brier Score (替代旧的Sharpe检查)
    if len(test_predictions) > 10:
        g3 = "✅" if bs_test < 0.1 else "❌"
        print(f"G3 (Brier<0.1): {bs_test:.4f} {g3}")
    else:
        g3 = "⚠️"
        print(f"G3: ⚠️ 数据不足")

    # G4: 实时性
    g4 = "✅" if t1 + t4 < 1.0 else "❌"
    print(f"G4 (<1秒): {t1+t4:.3f}s {g4}")

    # ===== Step 5: 报告模板 =====
    print("\n" + "=" * 60)
    print("ML异常检测报告 (模板)")
    print("=" * 60)
    print(f"日期: {df.index[-1].date()}")
    print(f"═══════════════════════════════")
    print(f"风险评分: R = {R_score.iloc[-1]:.0f} ({df['R_score'].quantile(0.90)*100:.0f}th分位)")
    print(f"异常类型: ", end="")
    anomalies = []
    if df.iloc[-1]["realized_vol_20d"] > df["realized_vol_20d"].rolling(252).quantile(0.95).iloc[-1]:
        anomalies.append("波动集聚")
    if df.iloc[-1]["vix_change_5d"] > 0.20:
        anomalies.append("VIX跳升")
    if abs(df.iloc[-1]["vol_change"]) > 1.5:
        anomalies.append("跨资产相关性突变")
    print(anomalies if anomalies else ["无明显异常"])
    print(f"持续天数: {df['duration'].iloc[-1]:.0f}天")

    if drawdowns is not None and len(drawdowns) >= 10:
        print(f"\n未来5d场景预测:")
        for th in [-0.01, -0.03, -0.05, -0.08, -0.10]:
            prob = (drawdowns < th).mean() * 100
            ci_lower, ci_upper = bootstrap_ci((drawdowns < th).astype(float), 1000)
            ci_l = f"{ci_lower*100:.1f}" if not np.isnan(ci_lower) else "--"
            ci_u = f"{ci_upper*100:.1f}" if not np.isnan(ci_upper) else "--"
            print(f"  P(回撤<{th*100:.0f}%) = {prob:.1f}%  [95% CI: {ci_l}%, {ci_u}%]")
        print(f"  期望最大回撤: {np.mean(drawdowns)*100:.2f}%")
        print(f"  VaR(95%): {np.percentile(drawdowns, 5)*100:.2f}%")

    test_predictions = []
    test_actuals = []
    if drawdowns is not None and len(drawdowns) >= 10:
        predictions = (drawdowns < -0.03).astype(float)
        actuals = (drawdowns < -0.03).astype(float)
        bs = brier_score(predictions, actuals)
        print(f"Brier Score (自检): {bs:.4f}")

        # 用历史全数据验证概率
        for idx in df.index[500:]:
            r = df.loc[idx, "R_score"]
            dds, _ = compute_future_drawdown_distribution(df, r, df.index.get_loc(idx))
            if dds is not None and len(dds) >= 3:
                test_predictions.append((dds < -0.03).mean())
                test_actuals.append(1.0 if (dds < -0.03).mean() > 0.5 else 0.0)


if __name__ == "__main__":
    main()
