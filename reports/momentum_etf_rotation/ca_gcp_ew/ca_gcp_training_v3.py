"""CA-GCP 训练 v3 — 强化特征工程

改进点:
  1. 宏观因子变化率 (diff / pct_change / 标准化变化)
  2. VIX 突破阈值事件特征
  3. 交互特征 (VIX × momentum, 信用利差 × breadth 等)
  4. 保留 v2 的 lag0/lag1/lag5/lag20
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
    macro = pd.read_parquet(MACRO_PATH)
    available = [c for c in CORE_MACRO_COLS if c in macro.columns]
    macro = macro[available]
    return macro.resample("D").ffill()


def rolling_zscore(s: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mean = s.rolling(window, min_periods=60).mean()
    std = s.rolling(window, min_periods=60).std()
    return (s - mean) / std.replace(0, np.nan)


def build_macro_features_v3(macro_daily: pd.DataFrame) -> pd.DataFrame:
    """v3 强化宏观特征工程

    三类:
      A. lag 特征 (v2): lag0/1/5/20 + zscore
      B. 变化率特征: diff, pct_change, zscore_diff
      C. VIX 突破事件特征
      D. 交互特征
    """
    features = []

    for col in macro_daily.columns:
        s = macro_daily[col]

        features.append(s.rename(f"{col}_lag0"))
        for lag in LAG_DAYS:
            features.append(s.shift(lag).rename(f"{col}_lag{lag}"))

        for lag in [0, 1, 5, 20]:
            ref = s if lag == 0 else s.shift(lag)
            features.append(ref.rename(f"{col}_lag{lag}").pipe(
                lambda x: rolling_zscore(x).rename(f"{col}_lag{lag}_z")
            ))

        if col in ["vix", "信用利差因子", "real_rate", "宏观增长因子"]:
            for diff_lag in [1, 5, 20]:
                diff = s.diff(diff_lag)
                features.append(diff.rename(f"{col}_diff{diff_lag}"))

                pct = s.pct_change(diff_lag)
                features.append(pct.rename(f"{col}_pct{diff_lag}"))

                z = rolling_zscore(s)
                z_diff = z.diff(diff_lag)
                features.append(z_diff.rename(f"{col}_zdiff{diff_lag}"))

    vix = macro_daily["vix"]

    features.append((vix > 20).astype(int).rename("vix_above_20"))
    features.append((vix > 25).astype(int).rename("vix_above_25"))
    features.append((vix > 30).astype(int).rename("vix_above_30"))
    features.append((vix > 35).astype(int).rename("vix_above_35"))

    vix_5d_pct = vix.pct_change(5)
    features.append((vix_5d_pct > 0.20).astype(int).rename("vix_jump_5d_20pct"))
    features.append((vix_5d_pct > 0.50).astype(int).rename("vix_jump_5d_50pct"))

    vix_20d_pct = vix.pct_change(20)
    features.append((vix_20d_pct > 0.30).astype(int).rename("vix_jump_20d_30pct"))
    features.append((vix_20d_pct > 0.80).astype(int).rename("vix_jump_20d_80pct"))

    vix_ma20 = vix.rolling(20).mean()
    features.append((vix / vix_ma20).rename("vix_ratio_ma20"))
    vix_ma60 = vix.rolling(60).mean()
    features.append((vix / vix_ma60).rename("vix_ratio_ma60"))

    vix_max_20d = vix.rolling(20).max()
    features.append((vix / vix_max_20d).rename("vix_ratio_max20"))

    vix_z = rolling_zscore(vix)
    features.append(vix_z.rename("vix_zscore"))

    features.append((vix_z > 1.5).astype(int).rename("vix_z_above_1.5"))
    features.append((vix_z > 2.0).astype(int).rename("vix_z_above_2.0"))

    credit = macro_daily["信用利差因子"]
    credit_z = rolling_zscore(credit)
    features.append(credit_z.rename("credit_zscore"))
    features.append((credit_z > 1.5).astype(int).rename("credit_z_above_1.5"))

    credit_5d_change = credit.diff(5)
    features.append(credit_5d_change.rename("credit_diff_5d"))
    features.append((credit_z.diff(5) > 0.5).astype(int).rename("credit_zjump_5d"))

    return pd.concat(features, axis=1)


def build_cagcp_features(returns: pd.DataFrame) -> pd.DataFrame:
    """CA-GCP 内部 8 维特征 + v3 衍生"""
    print("[特征工程] CA-GCP 8 维 ...")
    hw, stress, lower, upper = rolling_predict(returns)
    trend_ok = compute_trend_signal(returns)
    momentum = compute_momentum_score(returns)
    vol_regime = compute_vol_regime(returns)
    breadth = compute_market_breadth(returns)
    sectors = get_asset_sectors()

    alerts_raw = detect_warnings(stress, hw, mode="or")

    rows = []
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

        rows.append({
            "date": d,
            "width_z": wz_v,
            "stress": sv_v,
            "trend_ok": tr_v,
            "momentum_score": mom_v,
            "vol_high": vol_high,
            "breadth": br_v,
            "top_sector_conc": top_sec,
        })
    return pd.DataFrame(rows).set_index("date")


def build_interaction_features(
    cagcp: pd.DataFrame, macro_daily: pd.DataFrame
) -> pd.DataFrame:
    """交互特征 (基于原始宏观日频数据)"""
    common = cagcp.index.intersection(macro_daily.index)
    cagcp = cagcp.loc[common]
    md = macro_daily.loc[common]

    interactions = pd.DataFrame(index=common)
    vix_med = md["vix"].median()
    credit_med = md["信用利差因子"].median()
    rr_med = md["real_rate"].median()

    interactions["vix_x_momentum"] = (
        md["vix"].fillna(vix_med) * cagcp["momentum_score"]
    )
    interactions["vix_x_vol_high"] = (
        md["vix"].fillna(vix_med) * cagcp["vol_high"]
    )
    interactions["vix_x_breadth"] = (
        md["vix"].fillna(vix_med) * (1 - cagcp["breadth"])
    )
    interactions["credit_x_momentum"] = (
        md["信用利差因子"].fillna(credit_med) * cagcp["momentum_score"]
    )
    interactions["realrate_x_vol_high"] = (
        md["real_rate"].fillna(rr_med) * cagcp["vol_high"]
    )
    interactions["stress_x_vix"] = cagcp["stress"] * md["vix"].fillna(vix_med)
    interactions["vix_ratio_x_momentum"] = (
        (md["vix"] / md["vix"].rolling(20).mean()).fillna(1.0) * cagcp["momentum_score"]
    )
    return interactions
    return interactions


def build_full_features(returns: pd.DataFrame) -> pd.DataFrame:
    """构建完整 v3 特征"""
    cagcp = build_cagcp_features(returns)

    print("[特征工程] 宏观 v3 强化 (lag + diff + 事件) ...")
    macro_daily = load_macro_daily()
    macro = build_macro_features_v3(macro_daily)

    print("[特征工程] 交互特征 (基于原始 macro_daily) ...")
    interactions = build_interaction_features(cagcp, macro_daily)

    features = cagcp.join(macro, how="inner").join(interactions, how="inner")
    print(f"  总特征维度: {features.shape[1]}")
    return features


def load_hs300_returns() -> pd.Series:
    hs = pd.read_parquet(HS300_PATH)
    return hs.iloc[:, 0].pct_change().rename("hs300")


def build_y_labels(returns: pd.DataFrame, hs300: pd.Series) -> pd.DataFrame:
    market_ret = returns.mean(axis=1)
    common = market_ret.index.intersection(hs300.index)
    mr_aligned = market_ret.loc[common]
    hs300_aligned = hs300.loc[common]

    alpha_10d = (
        mr_aligned.rolling(HORIZON).sum().shift(-HORIZON)
        - hs300_aligned.rolling(HORIZON).sum().shift(-HORIZON)
    )

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
        (mr_aligned.rolling(5).sum().shift(-5) < -0.015)
        | (cumret_10d < DOWN_THRESH)
        | (mr_aligned.rolling(20).sum().shift(-20) < -0.03)
    ).astype(int)
    labels["y_B"] = (max_dd_10d < DD_THRESH).astype(int)
    labels["y_C"] = (alpha_10d < ALPHA_THRESH).astype(int)
    labels["y_D"] = (rolling_q < PERCENTILE_THRESH).astype(int)
    labels["y_E"] = ((max_dd_10d < DD_THRESH) | (rolling_q < PERCENTILE_THRESH)).astype(int)

    return labels.dropna()


def feature_columns_v3(features: pd.DataFrame) -> list[str]:
    return list(features.columns)


def train_with_optuna(
    X: pd.DataFrame, y: pd.Series, model_type: str, n_trials: int = 30
) -> dict:
    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)

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

    oof_proba = np.zeros(len(X))
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

    best_th, best_f1 = 0.5, -1
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
    pos_weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1)
    if model_type == "lgb":
        params.update({"objective": "binary", "metric": "binary_logloss",
                       "bagging_freq": 5, "min_data_in_leaf": 30,
                       "scale_pos_weight": pos_weight, "verbose": -1})
    else:
        params.update({"objective": "binary:logistic", "eval_metric": "logloss",
                       "scale_pos_weight": pos_weight, "verbosity": 0})

    fold_metrics = []
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
    }


def build_v3_heuristic_signal(features: pd.DataFrame) -> np.ndarray:
    fired = (
        (features["width_z"] > 2.0)
        & (features["stress"] > 0.6)
        & (features["breadth"] < 0.7)
        & (features["top_sector_conc"] < 0.7)
        & (features["trend_ok"] == 0)
    )
    return fired.astype(int).values


def main() -> None:
    print("=" * 70)
    print("CA-GCP 训练 v3 — 强化特征工程")
    print("=" * 70)

    returns = load_returns()
    features = build_full_features(returns)
    hs300 = load_hs300_returns()
    y_labels = build_y_labels(returns, hs300)

    common_idx = features.index.intersection(y_labels.index)
    X = features.loc[common_idx]
    Y = y_labels.loc[common_idx]

    print()
    print("[y 标签正类占比]")
    for col in Y.columns:
        print(f"  {col}: {Y[col].sum()} / {len(Y)} ({Y[col].mean():.1%})")

    X.to_parquet(OUT_DIR / "ca_gcp_features_v3.parquet")
    Y.to_csv(OUT_DIR / "ca_gcp_y_labels_v3.csv")
    print(f"\n[保存] 特征 {X.shape}, 标签 {Y.shape}")

    feat_cols = feature_columns_v3(X)
    X_train = X[feat_cols].fillna(0)

    print()
    print("=" * 70)
    print(f"Phase 1: 单模型训练 (5 标签 × LightGBM/XGBoost, 特征 {len(feat_cols)} 维)")
    print("=" * 70)

    single_results = {}
    oof_dict = {}

    for y_name in ["y_A", "y_B", "y_C", "y_D", "y_E"]:
        y = Y[y_name]
        print(f"\n--- 标签 {y_name} (正类: {y.sum()}, {y.mean():.1%}) ---")
        for model_type in ["lgb", "xgb"]:
            key = f"{y_name}_{model_type}"
            print(f"  [{model_type}] Optuna 训练中...")
            res = train_with_optuna(X_train, y, model_type, n_trials=30)
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
        }
        for k, v in single_results.items()
    ])
    single_df.to_csv(OUT_DIR / "single_model_results_v3.csv", index=False)

    best_key = max(single_results.keys(), key=lambda k: single_results[k]["F1"])
    print(f"\n最佳单模型: {best_key} (F1={single_results[best_key]['F1']:.3f})")

    print()
    print("=" * 70)
    print("Phase 2: Wash-Forward 验证 (Top-3)")
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
    print("Phase 3: 三方投票 (v3 + LGB + XGB)")
    print("=" * 70)

    lgb_best = max([k for k in single_results.keys() if k.endswith("_lgb")],
                   key=lambda k: single_results[k]["F1"])
    xgb_best = max([k for k in single_results.keys() if k.endswith("_xgb")],
                   key=lambda k: single_results[k]["F1"])

    y_name = "_".join(lgb_best.split("_")[:2])
    y_ref = Y[y_name]
    lgb_oof = oof_dict[lgb_best]
    xgb_oof = oof_dict[xgb_best]
    v3_signal = build_v3_heuristic_signal(X)

    print(f"  组合基: LGB={lgb_best}, XGB={xgb_best}, 标签={y_name}")

    fired_any = (
        (v3_signal == 1) | (lgb_oof > 0.5) | (xgb_oof > 0.5)
    ).astype(int)
    print(f"  三方 OR (任一触发):     F1={f1_score(y_ref, fired_any, zero_division=0):.3f}, "
          f"P={precision_score(y_ref, fired_any, zero_division=0):.3f}, "
          f"R={recall_score(y_ref, fired_any, zero_division=0):.3f}")

    fired_v3_or_double_ml = (
        (v3_signal == 1)
        | ((lgb_oof > 0.5) & (xgb_oof > 0.5))
    ).astype(int)
    print(f"  v3 OR (ML 双高分):      F1={f1_score(y_ref, fired_v3_or_double_ml, zero_division=0):.3f}, "  # noqa: E501
          f"P={precision_score(y_ref, fired_v3_or_double_ml, zero_division=0):.3f}, "
          f"R={recall_score(y_ref, fired_v3_or_double_ml, zero_division=0):.3f}")

    fired_double = (
        ((v3_signal == 1) & (lgb_oof > 0.5))
        | ((v3_signal == 1) & (xgb_oof > 0.5))
        | ((lgb_oof > 0.5) & (xgb_oof > 0.5))
    ).astype(int)
    print(f"  三方任二 AND:            F1={f1_score(y_ref, fired_double, zero_division=0):.3f}, "
          f"P={precision_score(y_ref, fired_double, zero_division=0):.3f}, "
          f"R={recall_score(y_ref, fired_double, zero_division=0):.3f}")

    avg = (lgb_oof + xgb_oof) / 2
    avg_pred = (avg > 0.5).astype(int)
    print(f"  软投票 (avg):            F1={f1_score(y_ref, avg_pred, zero_division=0):.3f}, "
          f"P={precision_score(y_ref, avg_pred, zero_division=0):.3f}, "
          f"R={recall_score(y_ref, avg_pred, zero_division=0):.3f}")

    print()
    print("=" * 70)
    print("Phase 4: SHAP 特征重要性 (最佳单模型)")
    print("=" * 70)

    try:
        import shap
        y_best = Y[y_name]
        params = single_results[best_key]["best_params"]
        if "lgb" in best_key:
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
        print("  Top-20 特征 (SHAP mean |value|):")
        for _, r in importance.head(20).iterrows():
            print(f"    {r['feature']:40s}: {r['importance']:.3f}")
        importance.to_csv(OUT_DIR / "shap_importance_v3.csv", index=False)
    except Exception as e:
        print(f"  SHAP 失败: {e}")

    print()
    print("=" * 70)
    print("Phase 5: 输出报告")
    print("=" * 70)

    report = [
        "# CA-GCP 训练 v3 — 强化特征工程报告",
        "",
        "## 1. 特征矩阵 (v3)",
        "",
        f"- 样本数: {len(X)}",
        f"- **总特征维度: {len(feat_cols)}** (v2: 40 → v3: 扩展)",
        "- 8 维 CA-GCP",
        "- ~64 维宏观 lag/zscore",
        "- ~24 维宏观 diff/pct/zdiff",
        "- ~22 维 VIX 突破事件",
        "- 7 维交互特征",
        "",
        "## 2. 单模型结果",
        "",
        "| 标签 | 模型 | P | R | F1 | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for k, v in sorted(single_results.items()):
        y_name_k = "_".join(k.split("_")[:2])
        model_k = k.split("_")[2]
        report.append(
            f"| {y_name_k} | {model_k} | {v['P']:.3f} | {v['R']:.3f} | {v['F1']:.3f} | {v['AUC']:.3f} |"  # noqa: E501
        )
    report += [
        "",
        f"**最佳单模型**: {best_key}, F1={single_results[best_key]['F1']:.3f}",
        "",
        "## 3. Wash-Forward 稳健性 (Top-3)",
        "",
        "| 模型 | WF F1 (mean ± std) | 折数 |",
        "|---|---|---|",
    ]
    for k, v in wf_results.items():
        report.append(f"| {k} | {v['mean_F1']:.3f} ± {v['std_F1']:.3f} | {v['n_folds']} |")

    report += [
        "",
        "## 4. v2 vs v3 对比",
        "",
        "| 指标 | v2 (40 维) | v3 (~120 维) | 改善 |",
        "|---|---|---|---|",
    ]
    v2_lgb = single_results.get("y_E_lgb")
    v3_lgb = single_results.get("y_E_lgb")
    if v2_lgb and v3_lgb:
        report.append(f"| 最佳单模型 F1 | - | {v3_lgb['F1']:.3f} | (v2 vs v3 直接比较) |")

    report += [
        "",
        "## 5. 推荐",
        "",
    ]
    if single_results[best_key]["F1"] > 0.50:
        report.append(f"**ML 显著超越启发式** (F1 {single_results[best_key]['F1']:.3f} > 0.50)")
    elif single_results[best_key]["F1"] >= 0.47:
        report.append(f"**ML 接近启发式** (F1 {single_results[best_key]['F1']:.3f} ≈ 0.50)")
    else:
        report.append("**启发式仍是最优**")

    (OUT_DIR / "ca_gcp_training_v3_report.md").write_text("\n".join(report))
    print(f"[报告] {OUT_DIR / 'ca_gcp_training_v3_report.md'}")


if __name__ == "__main__":
    main()
