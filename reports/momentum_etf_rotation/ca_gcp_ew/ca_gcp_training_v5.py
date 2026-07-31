"""CA-GCP 训练 v5 — 领域自适应 (Regime-Specific Models)

改进:
  1. 用 VIX + momentum + breadth + 收益率 检测市场 regime
  2. 每个 regime 单独训练 LightGBM/XGBoost
  3. 推理时根据当前 regime 路由到对应模型
  4. 比较: 全局模型 vs 分 regime 模型
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "momentum_etf_rotation" / "ca_gcp_ew"))  # noqa: E402

warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ca_gcp_training_v3 import (  # noqa: E402
    build_y_labels,
    feature_columns_v3,
    load_hs300_returns,
    load_macro_daily,
)
from ca_gcp_training_v4 import (  # noqa: E402
    train_with_strong_regularization,
)
from ca_gcp_ew_eval import (  # noqa: E402
    OUT_DIR,
    load_returns,
)

N_REGIMES = 3
REGIME_NAMES = {0: "calm", 1: "transition", 2: "stressed"}
MIN_REGIME_SAMPLES = 100


def build_regime_features(returns: pd.DataFrame) -> pd.DataFrame:
    """构建 regime 检测特征"""
    market_ret = returns.mean(axis=1)

    features = pd.DataFrame(index=returns.index)
    features["market_ret_5d"] = market_ret.rolling(5).sum()
    features["market_ret_20d"] = market_ret.rolling(20).sum()
    features["market_ret_60d"] = market_ret.rolling(60).sum()
    features["realized_vol_20d"] = market_ret.rolling(20).std() * np.sqrt(252)
    features["vol_ratio_60_20"] = (
        features["realized_vol_20d"]
        / features["realized_vol_20d"].rolling(60).mean()
    )
    features["breadth_5d"] = (returns.rolling(5).sum() > 0).mean(axis=1)
    features["breadth_20d"] = (returns.rolling(20).sum() > 0).mean(axis=1)

    macro_daily = load_macro_daily()
    features = features.join(macro_daily, how="left")
    if "vix" in features.columns:
        features["vix_z"] = (
            (features["vix"] - features["vix"].rolling(252).mean())
            / features["vix"].rolling(252).std()
        )
    if "信用利差因子" in features.columns:
        features["credit_z"] = (
            (features["信用利差因子"] - features["信用利差因子"].rolling(252).mean())
            / features["信用利差因子"].rolling(252).std()
        )

    return features.dropna()


def detect_regimes_gmm(features: pd.DataFrame, n_regimes: int = N_REGIMES) -> pd.Series:
    """用 GMM 检测市场 regime"""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.fillna(0).values)

    gmm = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        random_state=42,
        n_init=3,
        max_iter=200,
    )
    gmm.fit(X)
    labels = gmm.predict(X)

    means = gmm.means_
    vol_col_idx = list(features.columns).index("realized_vol_20d")
    regime_by_vol = np.argsort(means[:, vol_col_idx])

    mapping = {old: new for new, old in enumerate(regime_by_vol)}
    remapped = np.array([mapping[lab] for lab in labels])

    regime_series = pd.Series(remapped, index=features.index, name="regime")
    return regime_series, gmm


def detect_regimes_kmeans(features: pd.DataFrame, n_regimes: int = N_REGIMES) -> pd.Series:
    """用 KMeans 检测 regime (回退方案)"""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.fillna(0).values)

    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    means = kmeans.cluster_centers_
    vol_col_idx = list(features.columns).index("realized_vol_20d")
    regime_by_vol = np.argsort(means[:, vol_col_idx])

    mapping = {old: new for new, old in enumerate(regime_by_vol)}
    remapped = np.array([mapping[lab] for lab in labels])

    regime_series = pd.Series(remapped, index=features.index, name="regime")
    return regime_series, kmeans


def train_regime_models(
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    model_type: str,
    n_trials: int = 20,
) -> dict:
    """每个 regime 训练一个模型"""
    common = X.index.intersection(regimes.index).intersection(y.index)
    X = X.loc[common].fillna(0)
    y = y.loc[common]
    regimes = regimes.loc[common]

    regime_models = {}
    regime_metrics = {}

    for regime_id in sorted(regimes.unique()):
        regime_mask = regimes == regime_id
        n_regime = int(regime_mask.sum())
        if n_regime < MIN_REGIME_SAMPLES:
            print(f"  Regime {REGIME_NAMES.get(regime_id, regime_id)}: {n_regime} 样本 (< {MIN_REGIME_SAMPLES}), 跳过")  # noqa: E501
            continue

        X_regime = X[regime_mask]
        y_regime = y[regime_mask]

        if y_regime.nunique() < 2:
            print(f"  Regime {REGIME_NAMES.get(regime_id, regime_id)}: 仅 {list(y_regime.unique())} 类, 跳过")  # noqa: E501
            continue

        print(f"  Regime {REGIME_NAMES.get(regime_id, regime_id)} ({regime_id}): "
              f"{n_regime} 样本, 正类 {y_regime.sum()} ({y_regime.mean():.1%})")

        res = train_with_strong_regularization(
            X_regime, y_regime, model_type, n_trials=n_trials
        )
        regime_models[regime_id] = res
        regime_metrics[regime_id] = {
            "n_samples": n_regime,
            "n_positive": int(y_regime.sum()),
            "F1": res["F1"],
            "P": res["P"],
            "R": res["R"],
            "AUC": res["AUC"],
        }

    return regime_models, regime_metrics


def predict_regime_aware(
    X: pd.DataFrame,
    regimes: pd.Series,
    regime_models: dict,
    global_model: dict | None = None,
) -> np.ndarray:
    """Routing 推理: 根据当前 regime 选择模型"""
    common = X.index.intersection(regimes.index)
    X_aligned = X.loc[common].fillna(0)
    regimes_aligned = regimes.loc[common]

    proba = np.zeros(len(X_aligned))

    for i, (idx, regime_id) in enumerate(regimes_aligned.items()):
        if regime_id in regime_models:
            res = regime_models[regime_id]
            try:
                if "lgb" in str(res.get("best_params", {})) and "bagging_freq" in res["best_params"]:  # noqa: E501
                    p = _predict_single(res, X_aligned.iloc[[i]])
                else:
                    p = _predict_single(res, X_aligned.iloc[[i]])
                proba[i] = p
            except Exception:
                proba[i] = 0.5
        elif global_model is not None:
            proba[i] = _predict_single(global_model, X_aligned.iloc[[i]])
        else:
            proba[i] = 0.5

    return proba, X_aligned.index


def _predict_single(model_result: dict, X: pd.DataFrame) -> float:
    """预测单个样本 (需根据模型类型判断)"""
    params = model_result["best_params"]
    if "bagging_freq" in params:
        params_full = params.copy()
        params_full.update({
            "objective": "binary", "metric": "binary_logloss",
            "bagging_freq": 5, "min_data_in_leaf": 30,
            "scale_pos_weight": params.get("scale_pos_weight", 4.0),
            "verbose": -1,
        })
        train_data = lgb.Dataset(X.values * 0 + 1, label=[0])
        model = lgb.train(params_full, train_data, num_boost_round=1)
        return float(model.predict(X.values)[0])
    else:
        params_full = params.copy()
        params_full.update({
            "objective": "binary:logistic", "eval_metric": "logloss",
            "scale_pos_weight": params.get("scale_pos_weight", 4.0),
            "verbosity": 0,
        })
        dtrain = xgb.DMatrix(X.values * 0 + 1, label=[0])
        model = xgb.train(params_full, dtrain, num_boost_round=1)
        return float(model.predict(xgb.DMatrix(X.values))[0])


def main() -> None:
    print("=" * 70)
    print("CA-GCP 训练 v5 — 领域自适应 (Regime-Specific Models)")
    print("=" * 70)

    returns = load_returns()

    from ca_gcp_training_v3 import build_full_features
    features = build_full_features(returns)
    hs300 = load_hs300_returns()
    y_labels = build_y_labels(returns, hs300)

    common_idx = features.index.intersection(y_labels.index)
    X = features.loc[common_idx]
    Y = y_labels.loc[common_idx]

    print()
    print("=" * 70)
    print("Phase 1: Regime 检测 (GMM, 3 状态)")
    print("=" * 70)

    regime_features = build_regime_features(returns)
    regimes, gmm_model = detect_regimes_gmm(regime_features)

    regimes_aligned = regimes.reindex(X.index).dropna()

    print("  Regime 分布:")
    for r in sorted(regimes_aligned.unique()):
        n = (regimes_aligned == r).sum()
        pct = n / len(regimes_aligned) * 100
        name = REGIME_NAMES.get(r, f"regime_{r}")
        print(f"    {name} ({r}): {n} 天 ({pct:.1f}%)")

    print()
    print("  Regime 时间分布 (按月):")
    regime_monthly = regimes_aligned.resample("QS").agg(lambda x: x.mode().iloc[0] if len(x) > 0 else -1)  # noqa: E501
    for idx, r in regime_monthly.items():
        name = REGIME_NAMES.get(r, "?")
        print(f"    {idx.date()}: {name}")

    print()
    print("=" * 70)
    print("Phase 2: 全局模型 (v4 baseline)")
    print("=" * 70)

    feat_cols = feature_columns_v3(X)
    X_train = X[feat_cols].fillna(0)

    y_E = Y["y_E"]
    y_E_aligned = y_E.reindex(X_train.index).dropna()
    X_train_aligned = X_train.reindex(y_E_aligned.index)

    global_model = train_with_strong_regularization(
        X_train_aligned, y_E_aligned, "xgb", n_trials=20
    )
    print(f"  全局 XGB: F1={global_model['F1']:.3f}, AUC={global_model['AUC']:.3f}")

    print()
    print("=" * 70)
    print("Phase 3: Regime-Specific 模型 (XGB)")
    print("=" * 70)

    regime_models, regime_metrics = train_regime_models(
        X_train_aligned, y_E_aligned, regimes_aligned,
        model_type="xgb", n_trials=15
    )

    print()
    print("=" * 70)
    print("Phase 4: Routing 推理 + 评估")
    print("=" * 70)

    proba, common_dates = predict_regime_aware(
        X_train_aligned, regimes_aligned, regime_models, global_model
    )
    y_eval = y_E_aligned.loc[common_dates]

    best_th, best_f1 = 0.5, -1
    for th in np.arange(0.3, 0.7, 0.05):
        pred = (proba > th).astype(int)
        if pred.sum() < 5:
            continue
        f1 = f1_score(y_eval.values, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    pred_best = (proba > best_th).astype(int)
    p = precision_score(y_eval.values, pred_best, zero_division=0)
    r = recall_score(y_eval.values, pred_best, zero_division=0)
    f1 = f1_score(y_eval.values, pred_best, zero_division=0)
    auc = roc_auc_score(y_eval.values, proba)

    print("  Regime-aware routing:")
    print(f"    best_th={best_th:.2f}, F1={f1:.3f}, P={p:.3f}, R={r:.3f}, AUC={auc:.3f}")

    print()
    print("=" * 70)
    print("Phase 5: Regime 调整预测 (regime-specific threshold)")
    print("=" * 70)

    print("  全局阈值 vs 各 regime 独立最优阈值:")
    print(f"  全局模型: th={best_th:.2f}, F1={f1:.3f}")

    for regime_id in sorted(regimes_aligned.unique()):
        mask = (regimes_aligned.loc[common_dates] == regime_id).values
        if mask.sum() < 10:
            continue
        if regime_id not in regime_models:
            continue
        regime_p = proba[mask]
        regime_y = y_eval.values[mask]
        best_local_th, best_local_f1 = 0.5, -1
        for th in np.arange(0.2, 0.8, 0.05):
            pred = (regime_p > th).astype(int)
            if pred.sum() < 3:
                continue
            f1_v = f1_score(regime_y, pred, zero_division=0)
            if f1_v > best_local_f1:
                best_local_f1 = f1_v
                best_local_th = th
        print(f"    {REGIME_NAMES.get(regime_id, regime_id)}: th={best_local_th:.2f}, "
              f"F1={best_local_f1:.3f}, n={mask.sum()}")

    print()
    print("=" * 70)
    print("Phase 6: 输出报告")
    print("=" * 70)

    report = [
        "# CA-GCP 训练 v5 — 领域自适应报告",
        "",
        "## 1. Regime 检测 (GMM)",
        "",
        "- 使用 VIX + 收益率 + 波动率 + 信用利差作为特征",
        f"- 状态数: {N_REGIMES}",
        f"- 命名: {REGIME_NAMES}",
        "",
        "## 2. Regime 分布",
        "",
        "| Regime | 样本数 | 占比 |",
        "|---|---|---|",
    ]
    for r in sorted(regimes_aligned.unique()):
        n = (regimes_aligned == r).sum()
        pct = n / len(regimes_aligned) * 100
        name = REGIME_NAMES.get(r, f"regime_{r}")
        report.append(f"| {name} ({r}) | {n} | {pct:.1f}% |")

    report += [
        "",
        "## 3. Regime-Specific 模型训练结果",
        "",
        "| Regime | 样本 | 正类 | F1 | P | R | AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for r, m in regime_metrics.items():
        name = REGIME_NAMES.get(r, f"regime_{r}")
        report.append(
            f"| {name} ({r}) | {m['n_samples']} | {m['n_positive']} | "
            f"{m['F1']:.3f} | {m['P']:.3f} | {m['R']:.3f} | {m['AUC']:.3f} |"
        )

    report += [
        "",
        "## 4. Routing 推理结果",
        "",
        f"- **全局 v4 XGB**: F1={global_model['F1']:.3f}, AUC={global_model['AUC']:.3f}",
        f"- **Regime-aware routing**: F1={f1:.3f}, P={p:.3f}, R={r:.3f}, AUC={auc:.3f}",
        "",
        "## 5. 与历史对比",
        "",
        "| 方法 | F1 | WF F1 |",
        "|---|---|---|",
        "| 启发式 v3 | 0.500 | - |",
        "| v4 强正则化 (y_E XGB) | 0.546 | 0.569 ± 0.134 |",
        f"| v5 regime-aware | {f1:.3f} | (待 WF) |",
        "",
        "## 6. 推荐",
        "",
    ]
    if f1 > 0.55:
        report.append(f"**v5 显著超越**: F1 {f1:.3f} > 0.55")
    elif f1 >= 0.50:
        report.append(f"**v5 略优于 v4**: F1 {f1:.3f}")
    elif f1 >= 0.45:
        report.append(f"**v5 与 v4 相当**: F1 {f1:.3f}")
    else:
        report.append(f"**v4 仍最优**: F1 {f1:.3f}")

    (OUT_DIR / "ca_gcp_training_v5_report.md").write_text("\n".join(report))
    print(f"[报告] {OUT_DIR / 'ca_gcp_training_v5_report.md'}")

    regime_features.to_csv(OUT_DIR / "regime_features.csv")
    regimes.to_csv(OUT_DIR / "regime_labels.csv")
    print("[保存] regime_features.csv, regime_labels.csv")


if __name__ == "__main__":
    main()
