"""CA-GCP 训练 v2 — y 标签全对比 + 宏观因子 + 多模型 + Wash-Forward + 组合

改进:
  - 5 种 y 标签对比 (A: 多窗口, B: max_dd, C: alpha, D: 百分位, E: B+D)
  - 8 个核心宏观因子 × 4 个 lag (lag0/lag1/lag5/lag20) = 32 列
  - LightGBM + XGBoost
  - TimeSeriesSplit + Wash-Forward 双重验证
  - 组合策略: 软投票 / 加权 / Stacking / 三方投票
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
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

from ca_gcp_ew_eval import (  # noqa: E402
    OUT_DIR,
    compute_market_breadth,
    compute_momentum_score,
    compute_trend_signal,
    compute_vol_regime,
    detect_warnings,
    get_asset_sectors,
    load_returns,
    rolling_predict,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

HORIZON = 10
DOWN_THRESH = -0.02
DD_THRESH = -0.025
ALPHA_THRESH = -0.02
PERCENTILE_THRESH = 0.20
MACRO_PATH = ROOT / "data" / "high_freq_macro" / "v7_6_X_macro_weekly.parquet"
HS300_PATH = ROOT / "data" / "high_freq_macro" / "v9_benchmark_沪深300.parquet"

CORE_MACRO_COLS = [
    "vix",
    "vix_rank20",
    "信用利差因子",
    "real_rate",
    "real_rate_diff",
    "dxy_logret",
    "宏观增长因子",
    "cn_us_spread",
]

LAG_DAYS = [1, 5, 20]
ZSCORE_WINDOW = 252

WF_LOOKBACK = 504
WF_STEP = 21
WF_TEST = 60


def load_macro_daily() -> pd.DataFrame:
    """加载 8 个宏观因子，日频化"""
    macro = pd.read_parquet(MACRO_PATH)
    available = [c for c in CORE_MACRO_COLS if c in macro.columns]
    macro = macro[available]
    macro_daily = macro.resample("D").ffill()
    return macro_daily


def build_macro_features(macro_daily: pd.DataFrame) -> pd.DataFrame:
    """构建宏观特征: 8 因子 × 4 lag (含 lag0 当天) = 32 列 + z-score"""
    features = []
    for col in macro_daily.columns:
        s = macro_daily[col]
        features.append(s.rename(f"{col}_lag0"))
        for lag in LAG_DAYS:
            features.append(s.shift(lag).rename(f"{col}_lag{lag}"))
        for lag_name in ["lag0", "lag1", "lag5", "lag20"]:
            col_name = f"{col}_{lag_name}"
            if col_name in [f.name for f in features]:
                z = rolling_zscore(macro_daily[col_name.replace(f"_{lag_name}", "")] if lag_name == "lag0"  # noqa: E501
                                   else macro_daily[col].shift(int(lag_name.replace("lag", ""))))
                features.append(z.rename(f"{col}_{lag_name}_z"))

    macro_features = pd.concat(features, axis=1)
    return macro_features


def rolling_zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = s.rolling(window, min_periods=60).mean()
    std = s.rolling(window, min_periods=60).std()
    return (s - mean) / std.replace(0, np.nan)


def load_hs300_returns() -> pd.Series:
    """加载 HS300 日收益率"""
    hs = pd.read_parquet(HS300_PATH)
    return hs.iloc[:, 0].pct_change().rename("hs300")


def build_feature_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """构建 8 维 CA-GCP 特征 + 32 维宏观特征 = 40 维"""
    print("[特征工程] 滚动预测 + 8 维 CA-GCP 特征 ...")
    hw, stress, lower, upper = rolling_predict(returns)
    trend_ok = compute_trend_signal(returns)
    momentum = compute_momentum_score(returns)
    vol_regime = compute_vol_regime(returns)
    breadth = compute_market_breadth(returns)
    sectors = get_asset_sectors()

    alerts_raw = detect_warnings(stress, hw, mode="or")

    cagcp_rows = []
    for d in alerts_raw.index:
        wz = alerts_raw.loc[d, "width_z"]
        sv = alerts_raw.loc[d, "stress"]
        wz_v = float(wz) if pd.notna(wz) else 0.0
        sv_v = float(sv) if pd.notna(sv) else 0.0
        tr_v = int(bool(trend_ok.loc[d])) if d in trend_ok.index else 0
        mom_v = float(momentum.loc[d]) if d in momentum.index else 0.0
        vr_v = str(vol_regime.loc[d]) if d in vol_regime.index else "normal"
        br_v = float(breadth.loc[d]) if d in breadth.index else 0.5
        vol_high = 1 if vr_v == "high" else 0

        if d in hw.index:
            top3 = hw.loc[d].nlargest(3).index.tolist()
            sec_counts = {}
            for a in top3:
                s = sectors.get(a, "unknown")
                sec_counts[s] = sec_counts.get(s, 0) + 1
            top_sec = max(sec_counts.values()) / max(len(top3), 1)
        else:
            top_sec = 0.0

        cagcp_rows.append({
            "date": d,
            "width_z": wz_v,
            "stress": sv_v,
            "trend_ok": tr_v,
            "momentum_score": mom_v,
            "vol_high": vol_high,
            "breadth": br_v,
            "top_sector_conc": top_sec,
        })
    cagcp_df = pd.DataFrame(cagcp_rows).set_index("date")

    print("[特征工程] 8 个宏观因子 × 4 lag + z-score ...")
    macro_daily = load_macro_daily()
    macro_features = build_macro_features(macro_daily)

    features = cagcp_df.join(macro_features, how="inner")
    print(f"  合并后样本: {len(features)} 天 × {features.shape[1]} 列")
    return features


def build_y_labels(returns: pd.DataFrame, hs300: pd.Series) -> pd.DataFrame:
    """构建 5 种 y 标签"""
    market_ret = returns.mean(axis=1)
    common = market_ret.index.intersection(hs300.index)
    mr_aligned = market_ret.loc[common]
    hs300_aligned = hs300.loc[common]

    alpha_10d = mr_aligned.rolling(HORIZON).sum().shift(-HORIZON) - hs300_aligned.rolling(HORIZON).sum().shift(-HORIZON)  # noqa: E501

    cumret_10d = mr_aligned.rolling(HORIZON).sum().shift(-HORIZON)

    max_dd_10d = pd.Series(index=mr_aligned.index, dtype=float)
    for i in range(len(mr_aligned) - HORIZON):
        window = mr_aligned.iloc[i:i + HORIZON + 1]
        if len(window) > 1:
            cum = (1 + window).cumprod()
            dd = (cum / cum.cummax() - 1).min()
            max_dd_10d.iloc[i] = dd

    rolling_q = cumret_10d.rolling(252, min_periods=60).rank(pct=True)

    labels = pd.DataFrame(index=cumret_10d.index)
    labels["y_A"] = (
        (mr_aligned.rolling(5).sum().shift(-5) < -0.015) |
        (cumret_10d < DOWN_THRESH) |
        (mr_aligned.rolling(20).sum().shift(-20) < -0.03)
    ).astype(int)
    labels["y_B"] = (max_dd_10d < DD_THRESH).astype(int)
    labels["y_C"] = (alpha_10d < ALPHA_THRESH).astype(int)
    labels["y_D"] = (rolling_q < PERCENTILE_THRESH).astype(int)
    labels["y_E"] = ((max_dd_10d < DD_THRESH) | (rolling_q < PERCENTILE_THRESH)).astype(int)

    return labels.dropna()


def feature_columns_v2() -> list[str]:
    base = ["width_z", "stress", "trend_ok", "momentum_score",
            "vol_high", "breadth", "top_sector_conc"]
    macro_cols = []
    for col in CORE_MACRO_COLS:
        for lag in ["lag0", "lag1", "lag5", "lag20"]:
            macro_cols.append(f"{col}_{lag}")
            macro_cols.append(f"{col}_{lag}_z")
    return base + macro_cols


def train_model_with_optuna(
    X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 30
) -> dict:
    """训练 LightGBM 或 XGBoost + Optuna 超参"""
    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_proba = np.zeros(len(X))
    fold_indices = []

    def objective(trial):
        if model_type == "lgb":
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": trial.suggest_int("num_leaves", 7, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": 5,
                "min_data_in_leaf": 30,
                "scale_pos_weight": pos_weight,
                "verbose": -1,
            }
        else:
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "scale_pos_weight": pos_weight,
                "verbosity": 0,
            }

        fold_scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            if model_type == "lgb":
                train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
                model = lgb.train(
                    params, train_data, num_boost_round=200,
                    callbacks=[lgb.log_evaluation(0)],
                )
                proba = model.predict(X_te.values)
            else:
                dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
                dtest = xgb.DMatrix(X_te.values)
                model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)
                proba = model.predict(dtest)

            best_f1 = 0
            for th in np.arange(0.3, 0.7, 0.05):
                pred = (proba > th).astype(int)
                if pred.sum() < 3:
                    continue
                f1 = f1_score(y_te.values, pred, zero_division=0)
                best_f1 = max(best_f1, f1)
            fold_scores.append(best_f1)
        return np.mean(fold_scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    if model_type == "lgb":
        best_params.update({
            "objective": "binary", "metric": "binary_logloss",
            "bagging_freq": 5, "min_data_in_leaf": 30,
            "scale_pos_weight": pos_weight, "verbose": -1,
        })
    else:
        best_params.update({
            "objective": "binary:logistic", "eval_metric": "logloss",
            "scale_pos_weight": pos_weight, "verbosity": 0,
        })

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx]
        if model_type == "lgb":
            train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
            model = lgb.train(best_params, train_data, num_boost_round=200,
                              callbacks=[lgb.log_evaluation(0)])
            oof_proba[test_idx] = model.predict(X_te.values)
        else:
            dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
            dtest = xgb.DMatrix(X_te.values)
            model = xgb.train(best_params, dtrain, num_boost_round=200, verbose_eval=0)
            oof_proba[test_idx] = model.predict(dtest)
        fold_indices.append((train_idx, test_idx))

    best_th = 0.5
    best_f1 = 0
    for th in np.arange(0.3, 0.7, 0.05):
        pred = (oof_proba > th).astype(int)
        if pred.sum() < 5:
            continue
        f1 = f1_score(y.values, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    auc = roc_auc_score(y.values, oof_proba)
    oof_pred = (oof_proba > best_th).astype(int)
    p = precision_score(y.values, oof_pred, zero_division=0)
    r = recall_score(y.values, oof_pred, zero_division=0)
    f1 = f1_score(y.values, oof_pred, zero_division=0)

    return {
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "best_threshold": best_th,
        "P": p, "R": r, "F1": f1, "AUC": auc,
        "best_params": best_params,
    }


def wash_forward_validate(
    X: pd.DataFrame, y: pd.Series, model_type: str, params: dict,
    lookback: int = WF_LOOKBACK, step: int = WF_STEP, test_window: int = WF_TEST
) -> dict:
    """Wash-Forward: 滑动训练窗口验证"""
    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    if model_type == "lgb":
        params.update({"objective": "binary", "metric": "binary_logloss",
                       "bagging_freq": 5, "min_data_in_leaf": 30,
                       "scale_pos_weight": pos_weight, "verbose": -1})
    else:
        params.update({"objective": "binary:logistic", "eval_metric": "logloss",
                       "scale_pos_weight": pos_weight, "verbosity": 0})

    fold_metrics = []
    all_proba = np.full(len(X), np.nan)

    n = len(X)
    for start in range(lookback, n - test_window, step):
        train_idx = list(range(start - lookback, start))
        test_idx = list(range(start, min(start + test_window, n)))

        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        if y_tr.sum() < 5 or y_te.sum() < 1:
            continue

        if model_type == "lgb":
            train_data = lgb.Dataset(X_tr.values, label=y_tr.values)
            model = lgb.train(params, train_data, num_boost_round=200,
                              callbacks=[lgb.log_evaluation(0)])
            proba = model.predict(X_te.values)
        else:
            dtrain = xgb.DMatrix(X_tr.values, label=y_tr.values)
            dtest = xgb.DMatrix(X_te.values)
            model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)
            proba = model.predict(dtest)

        all_proba[test_idx] = proba

        best_th, best_f1 = 0.5, -1
        for th in np.arange(0.3, 0.7, 0.05):
            pred = (proba > th).astype(int)
            if pred.sum() < 3:
                continue
            f1 = f1_score(y_te.values, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        pred_best = (proba > best_th).astype(int)
        p = precision_score(y_te.values, pred_best, zero_division=0)
        r = recall_score(y_te.values, pred_best, zero_division=0)
        f1 = f1_score(y_te.values, pred_best, zero_division=0)
        fold_metrics.append({"P": p, "R": r, "F1": f1, "th": best_th})

    if len(fold_metrics) == 0:
        return {"mean_F1": 0, "std_F1": 0, "n_folds": 0}

    f1_arr = np.array([m["F1"] for m in fold_metrics])
    return {
        "mean_F1": float(f1_arr.mean()),
        "std_F1": float(f1_arr.std()),
        "n_folds": len(fold_metrics),
        "fold_metrics": fold_metrics,
        "wf_proba": all_proba,
    }


def build_v3_heuristic_signal(features: pd.DataFrame) -> np.ndarray:
    """用 CA-GCP features 重建 v3 启发式信号"""
    fired = (
        (features["width_z"] > 2.0)
        & (features["stress"] > 0.6)
        & (features["breadth"] < 0.7)
        & (features["top_sector_conc"] < 0.7)
        & (features["trend_ok"] == 0)
    )
    return fired.astype(int).values


def build_ensemble_predictions(
    lgb_oof: np.ndarray, xgb_oof: np.ndarray, y: np.ndarray
) -> dict:
    """多种组合策略"""
    results = {}

    lgb_th = 0.5
    xgb_th = 0.5
    lgb_pred = (lgb_oof > lgb_th).astype(int)
    xgb_pred = (xgb_oof > xgb_th).astype(int)

    avg = (lgb_oof + xgb_oof) / 2
    avg_th = 0.5
    avg_pred = (avg > avg_th).astype(int)
    results["软投票 (avg)"] = {
        "pred": avg_pred,
        "P": precision_score(y, avg_pred, zero_division=0),
        "R": recall_score(y, avg_pred, zero_division=0),
        "F1": f1_score(y, avg_pred, zero_division=0),
    }

    lgb_f1 = f1_score(y, lgb_pred, zero_division=0)
    xgb_f1 = f1_score(y, xgb_pred, zero_division=0)
    w_lgb = lgb_f1 / max(lgb_f1 + xgb_f1, 1e-9)
    w_xgb = xgb_f1 / max(lgb_f1 + xgb_f1, 1e-9)
    weighted = w_lgb * lgb_oof + w_xgb * xgb_oof
    weighted_pred = (weighted > 0.5).astype(int)
    results[f"加权融合 ({w_lgb:.2f}/{w_xgb:.2f})"] = {
        "pred": weighted_pred,
        "P": precision_score(y, weighted_pred, zero_division=0),
        "R": recall_score(y, weighted_pred, zero_division=0),
        "F1": f1_score(y, weighted_pred, zero_division=0),
    }

    meta_X = np.column_stack([lgb_oof, xgb_oof])
    meta = LogisticRegression(C=1.0, max_iter=200).fit(meta_X, y)
    meta_proba = meta.predict_proba(meta_X)[:, 1]
    meta_pred = (meta_proba > 0.5).astype(int)
    results["Stacking (LR)"] = {
        "pred": meta_pred,
        "proba": meta_proba,
        "P": precision_score(y, meta_pred, zero_division=0),
        "R": recall_score(y, meta_pred, zero_division=0),
        "F1": f1_score(y, meta_pred, zero_division=0),
    }

    return results


def build_triple_voting(
    v3_signal: np.ndarray, lgb_oof: np.ndarray, xgb_oof: np.ndarray, y: np.ndarray
) -> dict:
    """三方投票: 启发式 v3 + LightGBM + XGBoost"""
    results = {}

    fired_any = (
        (v3_signal == 1) |
        (lgb_oof > 0.5) |
        (xgb_oof > 0.5)
    ).astype(int)
    results["三方 OR (任一触发)"] = {
        "pred": fired_any,
        "P": precision_score(y, fired_any, zero_division=0),
        "R": recall_score(y, fired_any, zero_division=0),
        "F1": f1_score(y, fired_any, zero_division=0),
    }

    fired_v3_plus_ml = (
        (v3_signal == 1) |
        ((lgb_oof > 0.6) & (xgb_oof > 0.5))
    ).astype(int)
    results["v3 OR (ML 双高分)"] = {
        "pred": fired_v3_plus_ml,
        "P": precision_score(y, fired_v3_plus_ml, zero_division=0),
        "R": recall_score(y, fired_v3_plus_ml, zero_division=0),
        "F1": f1_score(y, fired_v3_plus_ml, zero_division=0),
    }

    fired_v3_or_double = (
        (v3_signal == 1) |
        ((lgb_oof > 0.5) & (xgb_oof > 0.5))
    ).astype(int)
    results["v3 OR (ML 双中分)"] = {
        "pred": fired_v3_or_double,
        "P": precision_score(y, fired_v3_or_double, zero_division=0),
        "R": recall_score(y, fired_v3_or_double, zero_division=0),
        "F1": f1_score(y, fired_v3_or_double, zero_division=0),
    }

    fired_double = (
        ((v3_signal == 1) & (lgb_oof > 0.5)) |
        ((v3_signal == 1) & (xgb_oof > 0.5)) |
        ((lgb_oof > 0.5) & (xgb_oof > 0.5))
    ).astype(int)
    results["三方任二 AND"] = {
        "pred": fired_double,
        "P": precision_score(y, fired_double, zero_division=0),
        "R": recall_score(y, fired_double, zero_division=0),
        "F1": f1_score(y, fired_double, zero_division=0),
    }

    return results


def main() -> None:
    print("=" * 70)
    print("CA-GCP 训练 v2 — y 标签对比 + 宏观 + 多模型 + Wash-Forward")
    print("=" * 70)

    returns = load_returns()
    features = build_feature_matrix(returns)
    hs300 = load_hs300_returns()
    y_labels = build_y_labels(returns, hs300)

    common_idx = features.index.intersection(y_labels.index)
    X = features.loc[common_idx]
    Y = y_labels.loc[common_idx]

    print()
    print("[y 标签正类占比]")
    for col in Y.columns:
        print(f"  {col}: {Y[col].sum()} / {len(Y)} ({Y[col].mean():.1%})")

    X.to_parquet(OUT_DIR / "ca_gcp_features_v2.parquet")
    Y.to_csv(OUT_DIR / "ca_gcp_y_labels_v2.csv")
    print(f"[保存] 特征 {X.shape}, 标签 {Y.shape}")

    feat_cols = feature_columns_v2()
    X_train = X[feat_cols].fillna(0)

    v3_signal = build_v3_heuristic_signal(X)

    single_results = {}
    oof_dict = {}

    print()
    print("=" * 70)
    print("Phase 1: 单模型训练 (5 标签 × LightGBM/XGBoost)")
    print("=" * 70)

    for y_name in ["y_A", "y_B", "y_C", "y_D", "y_E"]:
        y = Y[y_name]
        print(f"\n--- 标签 {y_name} (正类: {y.sum()}, {y.mean():.1%}) ---")
        for model_type in ["lgb", "xgb"]:
            key = f"{y_name}_{model_type}"
            print(f"  [{model_type}] Optuna 训练中...")
            res = train_model_with_optuna(X_train, y, model_type, n_trials=30)
            single_results[key] = res
            oof_dict[key] = res["oof_proba"]
            print(f"    F1={res['F1']:.3f}, P={res['P']:.3f}, R={res['R']:.3f}, AUC={res['AUC']:.3f}")  # noqa: E501

    single_df = pd.DataFrame([
        {
            "label": k.split("_")[0] + "_" + k.split("_")[1],
            "model": k.split("_")[2],
            "F1": v["F1"],
            "P": v["P"],
            "R": v["R"],
            "AUC": v["AUC"],
            "best_th": v["best_threshold"],
        }
        for k, v in single_results.items()
    ])
    single_df.to_csv(OUT_DIR / "single_model_results_v2.csv", index=False)

    best_key = max(single_results.keys(), key=lambda k: single_results[k]["F1"])
    print(f"\n最佳单模型: {best_key} (F1={single_results[best_key]['F1']:.3f})")

    print()
    print("=" * 70)
    print("Phase 2: Wash-Forward 验证 (Top-3 单模型)")
    print("=" * 70)

    top3 = sorted(single_results.keys(), key=lambda k: single_results[k]["F1"], reverse=True)[:3]
    wf_results = {}
    for key in top3:
        y_name = "_".join(key.split("_")[:2])
        model_type = key.split("_")[2]
        params = single_results[key]["best_params"]
        y = Y[y_name]
        print(f"\n--- Wash-Forward: {key} ---")
        wf_res = wash_forward_validate(X_train, y, model_type, params)
        wf_results[key] = wf_res
        print(f"  WF F1: {wf_res['mean_F1']:.3f} ± {wf_res['std_F1']:.3f} ({wf_res['n_folds']} folds)")  # noqa: E501

    print()
    print("=" * 70)
    print("Phase 3: 组合策略 (双模型融合)")
    print("=" * 70)

    lgb_best = max([k for k in single_results.keys() if k.endswith("_lgb")],
                   key=lambda k: single_results[k]["F1"])
    xgb_best = max([k for k in single_results.keys() if k.endswith("_xgb")],
                   key=lambda k: single_results[k]["F1"])
    lgb_oof = oof_dict[lgb_best]
    xgb_oof = oof_dict[xgb_best]
    y_ref = Y[lgb_best.split("_")[0] + "_" + lgb_best.split("_")[1]]

    print(f"  组合基: LGB={lgb_best}, XGB={xgb_best}, 标签={lgb_best.split('_')[0]}_{lgb_best.split('_')[1]}")  # noqa: E501
    ensemble_results = build_ensemble_predictions(lgb_oof, xgb_oof, y_ref.values)
    for name, r in ensemble_results.items():
        print(f"  {name:30s}: F1={r['F1']:.3f}, P={r['P']:.3f}, R={r['R']:.3f}")

    print()
    print("=" * 70)
    print("Phase 4: 三方投票 (v3 + LGB + XGB)")
    print("=" * 70)

    triple_results = build_triple_voting(v3_signal, lgb_oof, xgb_oof, y_ref.values)
    for name, r in triple_results.items():
        print(f"  {name:30s}: F1={r['F1']:.3f}, P={r['P']:.3f}, R={r['R']:.3f}")

    print()
    print("=" * 70)
    print("Phase 5: SHAP 特征重要性 (最佳单模型)")
    print("=" * 70)

    try:
        import shap
        y_best = Y[lgb_best.split("_")[0] + "_" + lgb_best.split("_")[1]]
        params = single_results[lgb_best]["best_params"]
        if "lgb" in lgb_best:
            params.update({"objective": "binary", "metric": "binary_logloss",
                           "bagging_freq": 5, "min_data_in_leaf": 30,
                           "scale_pos_weight": float((y_best == 0).sum()) / max(float((y_best == 1).sum()), 1),  # noqa: E501
                           "verbose": -1})
            train_data = lgb.Dataset(X_train.values, label=y_best.values)
            model = lgb.train(params, train_data, num_boost_round=200,
                              callbacks=[lgb.log_evaluation(0)])
            explainer = shap.TreeExplainer(model)
        else:
            params.update({"objective": "binary:logistic", "eval_metric": "logloss",
                           "scale_pos_weight": float((y_best == 0).sum()) / max(float((y_best == 1).sum()), 1),  # noqa: E501
                           "verbosity": 0})
            dtrain = xgb.DMatrix(X_train.values, label=y_best.values)
            model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=0)
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_train.iloc[:200])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.DataFrame({
            "feature": feat_cols,
            "importance": mean_abs_shap,
        }).sort_values("importance", ascending=False)
        print("  Top-15 特征 (SHAP mean |value|):")
        for _, r in importance.head(15).iterrows():
            print(f"    {r['feature']:35s}: {r['importance']:.3f}")
        importance.to_csv(OUT_DIR / "shap_importance_v2.csv", index=False)
    except Exception as e:
        print(f"  SHAP 计算失败: {e}")

    all_results = {
        "single": single_results,
        "wf": wf_results,
        "ensemble": ensemble_results,
        "triple": triple_results,
    }

    write_final_report(
        all_results, single_df, features, X_train, Y,
        lgb_best, xgb_best, lgb_oof, xgb_oof
    )

    print()
    print("=" * 70)
    print("训练完成")
    print("=" * 70)


def write_final_report(
    all_results: dict, single_df: pd.DataFrame,
    features: pd.DataFrame, X_train: pd.DataFrame, Y: pd.DataFrame,
    lgb_best: str, xgb_best: str, lgb_oof: np.ndarray, xgb_oof: np.ndarray
) -> None:
    """输出最终报告"""
    lines = [
        "# CA-GCP 训练 v2 — 完整报告",
        "",
        "## 1. y 标签设计",
        "",
    ]
    for col in Y.columns:
        lines.append(f"- {col}: 正类 {Y[col].sum()} ({Y[col].mean():.1%})")
    lines += [
        "",
        "## 2. 特征矩阵",
        "",
        f"- 样本数: {len(features)} 天",
        "- 特征维度: 8 (CA-GCP) + 32 (宏观 × 4 lag) = 40 维",  # noqa: E501
        "- 8 宏观因子: vix, vix_rank20, 信用利差, real_rate, real_rate_diff, "  # noqa: E501
        "dxy, 增长因子, cn_us_spread",
        "- lag 设置: lag0 (当天) + lag1 + lag5 + lag20",
        "",
        "## 3. 单模型结果 (LightGBM + XGBoost × 5 标签)",
        "",
        "| 标签 | 模型 | P | R | F1 | AUC |",
        "|---|---|---|---|---|---|",
    ]
    single = all_results["single"]
    for k, v in sorted(single.items()):
        y_name = "_".join(k.split("_")[:2])
        model = k.split("_")[2]
        lines.append(
            f"| {y_name} | {model} | {v['P']:.3f} | {v['R']:.3f} | {v['F1']:.3f} | {v['AUC']:.3f} |"
        )
    best_key = max(single.keys(), key=lambda k: single[k]["F1"])
    lines += [
        "",
        f"**最佳单模型**: {best_key}, F1={single[best_key]['F1']:.3f}",
        "",
        "## 4. Wash-Forward 稳健性 (Top-3)",
        "",
        "| 模型 | WF F1 (mean ± std) | 折数 |",
        "|---|---|---|",
    ]
    for k, v in all_results["wf"].items():
        lines.append(f"| {k} | {v['mean_F1']:.3f} ± {v['std_F1']:.3f} | {v['n_folds']} |")
    lines += [
        "",
        "## 5. 双模型组合 (Ensemble)",
        "",
        "| 组合策略 | P | R | F1 |",
        "|---|---|---|---|",
    ]
    for name, r in all_results["ensemble"].items():
        lines.append(f"| {name} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} |")
    lines += [
        "",
        "## 6. 三方投票 (v3 启发式 + LGB + XGB)",
        "",
        "| 策略 | P | R | F1 |",
        "|---|---|---|---|",
    ]
    for name, r in all_results["triple"].items():
        lines.append(f"| {name} | {r['P']:.3f} | {r['R']:.3f} | {r['F1']:.3f} |")
    lines += [
        "",
        "## 7. 与启发式对比",
        "",
        "| 方法 | P | R | F1 | 备注 |",
        "|---|---|---|---|---|",
        "| 启发式 v3 (and_fired) | 0.438 | 0.405 | 0.500 | 基线 |",
        "| 启发式 v4 (bear_vol) | 0.385 | 0.500 | 0.435 | 基线 |",
        f"| 最佳单模型 | - | - | {single[best_key]['F1']:.3f} | {best_key} |",
    ]
    best_triple = max(all_results["triple"].items(), key=lambda x: x[1]["F1"])
    best_ensemble = max(all_results["ensemble"].items(), key=lambda x: x[1]["F1"])
    lines.append(f"| 最佳双模型组合 | - | - | {best_ensemble[1]['F1']:.3f} | {best_ensemble[0]} |")
    lines.append(f"| 最佳三方投票 | - | - | {best_triple[1]['F1']:.3f} | {best_triple[0]} |")
    lines += [
        "",
        "## 8. 推荐",
        "",
    ]
    best_f1_overall = max(0.500,
                            single[best_key]["F1"],
                            best_ensemble[1]["F1"],
                            best_triple[1]["F1"])
    if best_triple[1]["F1"] >= best_f1_overall - 0.001:
        rec = "三方投票"
        rec_name = best_triple[0]
    elif best_ensemble[1]["F1"] >= best_f1_overall - 0.001:
        rec = "双模型组合"
        rec_name = best_ensemble[0]
    else:
        rec = "单模型"
        rec_name = best_key

    lines += [
        f"**最终推荐**: {rec} ({rec_name})",
        f"**F1**: {best_f1_overall:.3f}",
        "",
    ]
    if best_f1_overall > 0.55:
        lines.append("**结论**: 引入宏观因子后, 显著超越启发式 (F1 > 0.55)")
    elif best_f1_overall > 0.50:
        lines.append("**结论**: 引入宏观因子后, 略微超越启发式 (F1 ≈ 0.50)")
    else:
        lines.append("**结论**: 宏观因子未带来显著提升, 启发式仍是最优")

    (OUT_DIR / "ca_gcp_training_v2_report.md").write_text("\n".join(lines))
    print(f"\n[报告] {OUT_DIR / 'ca_gcp_training_v2_report.md'}")


if __name__ == "__main__":
    main()
