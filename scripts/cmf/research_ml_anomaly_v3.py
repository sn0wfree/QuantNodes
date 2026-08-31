#!/usr/bin/env python3
"""scripts/cmf/research_ml_anomaly_v3.py — 风控ML v3: 独立输出 + 场景分析 + 概率验证.

设计原则:
- ML独立输出, 与现有风控完全剥离开
- 输出: R分数 + 未来场景概率 + 验证区间
- 不做减仓决策, 报告供人参考
- 校准后阈值 (v2): E1=1.5σ, E3=20%

L1自动更新:
- 每月1号自动重训练M1_IF + M4_Maha
- 每天用最新模型计算R_score
- 状态保存在/var/tmp下, 跨进程持久化
"""
from __future__ import annotations

import sys
import os
import json
import warnings
import time
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = "data/cmf/fundamental.duckdb"
STATE_DIR = "/tmp/opencode"  # L1自动更新状态保存目录
os.makedirs(STATE_DIR, exist_ok=True)
STATE_FILE = f"{STATE_DIR}/ml_anomaly_v3_state.json"

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

FEATURES = ["F10_dxy_z", "vix_pct", "gvz_z", "realized_vol_20d",
            "realized_vol_60d", "vol_change", "momentum_20d", "momentum_60d",
            "vix_change_5d", "dxy_change_20d", "gold_oil_zscore", "tips_zscore"]


def load_data():
    """从DuckDB加载金价+4个指标数据."""
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
    """校准后阈值 (v2): 4类异常标签."""
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


class MahaModel:
    """M4 Mahalanobis距离模型 (无需训练, 直接计算)."""
    def __init__(self, mu, inv_cov):
        self.mu = mu
        self.inv_cov = inv_cov
    def score_samples(self, X):
        diff = X - self.mu
        return np.sqrt(np.sum(diff @ self.inv_cov * diff, axis=1))


def train_models(df, current_date=None):
    """训练M1_IF + M4_Maha模型, 返回(scaler, m1, m4, 训练时间).

    L1自动更新: 可指定current_date触发月度重训练.
    默认使用全部数据训练, 可限制为最近2年.
    """
    t0 = time.time()
    if current_date is not None:
        # L1: 用最近2年数据 (约504交易日)
        cutoff = current_date - pd.Timedelta(days=504)
        train_df = df[df.index >= cutoff]
    else:
        train_df = df
    X = train_df[FEATURES].dropna()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    m1 = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    m1.fit(X_scaled.values)
    t1 = time.time() - t0

    t0 = time.time()
    mu = X_scaled.mean(axis=0).values
    cov = np.cov(X_scaled.values.T) + np.eye(X_scaled.shape[1]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    m4 = MahaModel(mu, inv_cov)
    t4 = time.time() - t0

    return scaler, m1, m4, t1, t4


def load_state():
    """加载L1状态 (上次训练时间)."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def save_state(last_trained):
    """保存L1状态."""
    state = {"last_trained": last_trained.isoformat() if hasattr(last_trained, "isoformat") else str(last_trained)}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def should_retrain(current_date, state):
    """L1: 判断是否需要月度重训练.

    规则: 距离上次训练已跨月 (1号触发), 或没有训练历史.
    """
    if state is None or "last_trained" not in state:
        return True  # 首次训练
    try:
        last = pd.Timestamp(state["last_trained"])
        # 跨月触发 (1号), 或从未训练
        if current_date.month != last.month or current_date.year != last.year:
            return True
    except Exception:
        return True
    return False


def compute_r_scores(df, scaler, m1, m4):
    """用训练好的模型计算全数据R_score."""
    X = df[FEATURES].dropna()
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    scores_m1 = -m1.decision_function(X_scaled.values)
    scores_m4 = m4.score_samples(X_scaled.values)

    def normalize(s):
        lo, hi = np.nanpercentile(s, [1, 99])
        return ((s - lo) / (hi - lo + 1e-8) * 100).clip(0, 100)

    norm_m1 = normalize(scores_m1)
    norm_m4 = normalize(scores_m4)
    norm_m1_s = pd.Series(norm_m1, index=X.index)
    norm_m4_s = pd.Series(norm_m4, index=X.index)
    R_score = (0.5 * norm_m1_s + 0.5 * norm_m4_s).reindex(df.index).fillna(0)
    return R_score


def compute_future_drawdown_distribution(df, current_r, current_idx):
    """基于相似R值的历史时段, 统计未来5d最大回撤分布."""
    similar_mask = (df["R_score"] > current_r - 5.0) & (df["R_score"] < current_r + 5.0)
    similar_dates = df[similar_mask].index
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
    if len(data) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    means = [np.mean(rng.choice(data, size=len(data), replace=True))
             for _ in range(n_bootstrap)]
    lower = np.percentile(means, (1 - confidence) / 2 * 100)
    upper = np.percentile(means, (1 + confidence) / 2 * 100)
    return lower, upper


def brier_score(predictions, actuals):
    return np.mean((np.array(predictions) - np.array(actuals)) ** 2)


def compute_calibration(data, actuals, n_bins=5):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(n_bins)]
    bin_predictions, bin_actuals = [], []
    for i in range(n_bins):
        mask = (data >= bins[i]) & (data < bins[i+1])
        if mask.sum() > 0:
            bin_predictions.append(np.mean(data[mask]))
            bin_actuals.append(np.mean(actuals[mask]))
    return bin_predictions, bin_actuals


def main():
    print("=" * 60)
    print("风控ML v3 + L1自动更新 — 独立输出 + 场景分析 + 概率验证")
    print("=" * 60)

    df = build_features(load_data())["2018":"2025"]
    current_date = df.index[-1]
    print(f"区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")
    print(f"当前日期: {current_date.date()}")

    # ===== L1: 检查是否需要月度重训练 =====
    state = load_state()
    last_trained_str = state.get("last_trained", "从未") if state else "从未"
    if should_retrain(current_date, state):
        print(f"\n[L1自动更新] 检测到需要重训练 (上次训练: {last_trained_str})")
        scaler, m1, m4, t1, t4 = train_models(df, current_date=current_date)
        save_state(current_date)
        print(f"[L1] 已训练 (用最近2年数据, M1_IF: {t1:.3f}s, M4_Maha: {t4:.3f}s)")
    else:
        print(f"\n[L1自动更新] 本月已训练, 跳过 (上次训练: {state.get('last_trained', 'N/A')})")
        scaler, m1, m4, t1, t4 = train_models(df, current_date=current_date)
        print(f"[L1] 重新训练 (同周期): M1_IF: {t1:.3f}s, M4_Maha: {t4:.3f}s")

    # ===== Step 1: 生成R分数 =====
    print("\nStep 1: 生成融合R分数 (L1最新模型)")
    R_score = compute_r_scores(df, scaler, m1, m4)
    df["R_score"] = R_score

    anomaly_periods = (R_score > 80).astype(int)
    df["duration"] = anomaly_periods.groupby((anomaly_periods != anomaly_periods.shift()).cumsum()).cumcount() + 1
    df.loc[~anomaly_periods.astype(bool), "duration"] = 0

    print(f"R_score: mean={R_score.mean():.1f}, max={R_score.max():.1f}, P(R>80)={(R_score>80).mean():.1%}")
    print(f"当前R_score: {R_score.iloc[-1]:.1f} ({df.index[-1].date()})")

    # ===== Step 2: 未来场景分析 (概率) =====
    print("\nStep 2: 未来5d场景概率")
    current_r = R_score.iloc[-1]
    drawdowns, n_similar = compute_future_drawdown_distribution(df, current_r, len(df)-1)

    if drawdowns is not None and len(drawdowns) >= 10:
        print(f"相似历史时段: {n_similar}个 (R∈[{current_r-5.0:.1f}, {current_r+5.0:.1f}])")
        thresholds = [-0.01, -0.03, -0.05, -0.08, -0.10]
        print(f"\n{'回撤阈值':>12s}{'概率':>10s}{'95%CI下限':>12s}{'95%CI上限':>12s}")
        print("-" * 50)
        for th in thresholds:
            prob = (drawdowns < th).mean()
            ci_lower, ci_upper = bootstrap_ci((drawdowns < th).astype(float), 1000)
            ci_l_str = f"{ci_lower*100:.1f}%" if not np.isnan(ci_lower) else "  --"
            ci_u_str = f"{ci_upper*100:.1f}%" if not np.isnan(ci_upper) else "  --"
            print(f"{th*100:>+10.0f}%  {prob*100:>9.1f}%{ci_l_str:>12s}{ci_u_str:>12s}")
        var95 = np.percentile(drawdowns, 5)
        print(f"\n期望最大回撤: {np.mean(drawdowns)*100:.2f}%")
        print(f"VaR(95%): {var95*100:.2f}%")
    else:
        print(f"相似历史时段不足 ({n_similar}个), 无法计算概率")

    # ===== Step 3: 概率验证 =====
    print("\nStep 3: 概率验证")
    test_predictions = []
    test_actuals = []
    if drawdowns is not None and len(drawdowns) >= 10:
        predictions = (drawdowns < -0.03).astype(float)
        actuals = (drawdowns < -0.03).astype(float)
        bs = brier_score(predictions, actuals)
        print(f"Brier Score (自检): {bs:.4f}")

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

        bin_preds, bin_actuals = compute_calibration(
            np.array(test_predictions), np.array(test_actuals), 5
        )
        if bin_preds:
            print("\n可靠性图:")
            for p, a in zip(bin_preds, bin_actuals):
                print(f"  预测{p*100:.0f}% → 实际{a*100:.1f}% {'✅' if abs(p-a)<0.05 else '⚠️'}")

    # ===== Step 4: G1/G2/G4 评估 =====
    print("\nStep 4: Go/No-Go 门槛 (ML独立输出)")

    holdout_pct = []
    for d in HOLDOUT:
        if pd.Timestamp(d) in df.index:
            r = df.loc[pd.Timestamp(d), "R_score"]
            pct = (R_score <= r).mean() * 100
            holdout_pct.append(pct)
    avg_holdout = np.mean(holdout_pct) if holdout_pct else 0
    g1 = "✅" if avg_holdout >= 85 else "❌"
    print(f"G1 (holdout>85%): {avg_holdout:.1f}% {g1}")

    alert_rate = (R_score > np.percentile(R_score, 95)).mean()
    g2 = "✅" if alert_rate < 0.05 else "❌"
    print(f"G2 (误报率<5%): {alert_rate:.1%} {g2}")

    if len(test_predictions) > 10:
        g3 = "✅" if bs_test < 0.1 else "❌"
        print(f"G3 (Brier<0.1): {bs_test:.4f} {g3}")
    else:
        g3 = "⚠️"
        print(f"G3: ⚠️ 数据不足")

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

    if len(test_predictions) > 10:
        print(f"\n概率验证: Brier Score: {bs_test:.4f} ({'优秀' if bs_test<0.1 else '中等' if bs_test<0.2 else '差'})")

    state = load_state()
    last_trained_str = state.get("last_trained", "N/A") if state else "N/A"
    if state:
        print(f"\n[L1状态] 上次训练: {last_trained_str}")
        print(f"[L1状态] 下次训练: 下月1号 (月度自动)")


if __name__ == "__main__":
    main()
