# coding=utf-8
"""PyCaret 多模型估计器.

Phase 1: PyCaret compare_models 一次性筛选 top-K 模型
Phase 2: sklearn 原生 API 滚动训练（后续实现）
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
HF_DIR = REPO / "data" / "high_freq_macro"

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# 模型 ID
# ============================================================
CANDIDATE_MODELS = [
    "ridge", "lasso", "en", "huber",
    "rf", "et", "gbr",
    "lightgbm", "ada", "mlp",
]


# ============================================================
# 数据加载
# ============================================================
def load_v7_7_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """加载 v7.7 数据.

    Returns:
        X_panel: (T, N, K) 因子面板（保留 NaN）
        Y_raw: (T, N) 原始周收益
        Y_rank: (T, N) 截面 rank
        factor_names: 因子名列表
    """
    X = np.load(HF_DIR / "v7_7_X_panel.npy")
    Y_raw = np.load(HF_DIR / "v7_7_Y_raw.npy")
    Y_rank = np.load(HF_DIR / "v7_7_Y_rank.npy")
    names = pd.read_csv(HF_DIR / "v7_7_factor_names.csv")['0'].tolist()
    return X, Y_raw, Y_rank, names


def load_train_panel() -> pd.DataFrame:
    """加载预构造的训练面板."""
    return pd.read_parquet(HF_DIR / "v7_7_train_panel.parquet")


# ============================================================
# Phase 1: PyCaret compare_models
# ============================================================
def phase1_compare_models(
    target_col: str = "target_raw",
    model_ids: list[str] | None = None,
    n_select: int = 5,
    sample_size: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Phase 1: 用 PyCaret compare_models 一次性筛选 top-K 模型.

    Parameters:
        target_col: 目标列名 (target_raw 或 target_rank)
        model_ids: 候选模型列表 (None = CANDIDATE_MODELS)
        n_select: 选择前 K 个模型
        sample_size: 采样大小 (None = 全量)
        verbose: 打印结果

    Returns:
        results_df: PyCaret 评分表 (含 R2, MAE, RMSE 等)
    """
    from pycaret.regression import RegressionExperiment

    if model_ids is None:
        model_ids = CANDIDATE_MODELS

    # 加载数据
    panel = load_train_panel()
    feat_cols = [c for c in panel.columns if c.startswith('f')]
    df = panel[feat_cols + [target_col]].copy()
    df = df.rename(columns={target_col: 'target'})

    # 去除 target NaN
    df = df.dropna(subset=['target'])

    # 清理 inf/极大值
    X = df[feat_cols]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    # clip 极大值到 99.9 分位
    for col in feat_cols:
        p99 = X[col].abs().quantile(0.999)
        if p99 > 0:
            X[col] = X[col].clip(-p99, p99)
    df[feat_cols] = X

    # 采样
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    if verbose:
        print(f"训练集: {df.shape[0]} 样本, {len(feat_cols)} 因子")
        print(f"target: mean={df['target'].mean():.6f}, std={df['target'].std():.6f}")
        print(f"候选模型: {model_ids}")

    # PyCaret setup
    exp = RegressionExperiment()
    exp.setup(
        data=df,
        target='target',
        train_size=0.8,
        preprocess=False,
        session_id=42,
        verbose=False,
        html=False,
    )

    # compare_models
    best = exp.compare_models(
        include=model_ids,
        sort='R2',
        n_select=n_select,
        verbose=False,
    )

    # 获取评分表
    results_df = exp.pull()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Top-{n_select} 模型 (按 R2 排序)")
        print(f"{'='*60}")
        print(results_df.to_string())

    return results_df


def phase1_get_top_models(results_df: pd.DataFrame, n: int = 5) -> list[str]:
    """从 Phase 1 结果中提取 top-N 模型 ID.

    Parameters:
        results_df: phase1_compare_models 返回的评分表
        n: 提取前 n 个

    Returns:
        model_ids: PyCaret 模型 ID 列表
    """
    # PyCaret 的 compare_models 返回的 index 是模型名称
    # 需要映射回 PyCaret ID
    NAME_TO_ID = {
        "Linear Regression": "lr",
        "Lasso Regression": "lasso",
        "Ridge Regression": "ridge",
        "Elastic Net": "en",
        "Least Angle Regression": "lar",
        "Lasso Least Angle Regression": "llar",
        "Orthogonal Matching Pursuit": "omp",
        "Bayesian Ridge": "br",
        "Automatic Relevance Determination": "ard",
        "Passive Aggressive Regressor": "par",
        "Random Sample Consensus": "ransac",
        "TheilSen Regressor": "tr",
        "Huber Regressor": "huber",
        "Kernel Ridge": "kr",
        "Support Vector Regression": "svm",
        "K Neighbors Regressor": "knn",
        "Decision Tree Regressor": "dt",
        "Random Forest Regressor": "rf",
        "Extra Trees Regressor": "et",
        "AdaBoost Regressor": "ada",
        "Gradient Boosting Regressor": "gbr",
        "MLP Regressor": "mlp",
        "Extreme Gradient Boosting": "xgboost",
        "Light Gradient Boosting Machine": "lightgbm",
        "CatBoost Regressor": "catboost",
    }

    top_names = results_df.index.tolist()[:n]
    top_ids = []
    for name in top_names:
        if name in NAME_TO_ID:
            top_ids.append(NAME_TO_ID[name])
        else:
            # 尝试模糊匹配
            for k, v in NAME_TO_ID.items():
                if k.lower() in str(name).lower() or str(name).lower() in k.lower():
                    top_ids.append(v)
                    break

    return top_ids


# ============================================================
# Phase 2: sklearn 滚动估计（框架，后续完善）
# ============================================================
def phase2_sklearn_rolling(
    X_panel: np.ndarray,
    Y: np.ndarray,
    model_id: str,
    min_history: int = 52,
    verbose: bool = True,
) -> np.ndarray:
    """Phase 2: 用 sklearn 原生 API 滚动训练.

    Parameters:
        X_panel: (T, N, K) 因子面板
        Y: (T, N) 标签
        model_id: 模型 ID
        min_history: 最少训练期
        verbose: 打印进度

    Returns:
        scores: (T, N) 预测分数
    """
    T, N, K = X_panel.shape
    scores = np.full((T, N), np.nan)

    model = _create_sklearn_model(model_id)

    for t in range(min_history, T):
        # 构造训练集 [0, t)
        X_list, y_list = [], []
        for s in range(t):
            for i in range(N):
                y_val = Y[s, i]
                if np.isnan(y_val):
                    continue
                X_list.append(X_panel[s, i, :])
                y_list.append(y_val)

        if len(X_list) < 100:
            continue

        X_train = np.array(X_list)
        y_train = np.array(y_list)

        # 预测集: t 时刻
        X_pred = X_panel[t]  # (N, K)

        try:
            # 处理 NaN
            if model_id not in ("lightgbm",):
                train_valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
                X_tr = X_train[train_valid]
                y_tr = y_train[train_valid]
                X_pr = np.nan_to_num(X_pred, nan=0.0)
            else:
                train_valid = ~np.isnan(y_train)
                X_tr = X_train[train_valid]
                y_tr = y_train[train_valid]
                X_pr = X_pred

            model.fit(X_tr, y_tr)
            scores[t] = model.predict(X_pr)

        except Exception as e:
            if verbose:
                print(f"  t={t}: {e}")
            continue

        if verbose and (t - min_history) % 50 == 0:
            print(f"  [{model_id}] t={t}/{T}, train={len(X_list)}")

    return scores


def _create_sklearn_model(model_id: str):
    """创建 sklearn 模型对象."""
    if model_id == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.05, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1, n_jobs=-1,
        )
    elif model_id == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            max_depth=6, learning_rate=0.05, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0, n_jobs=-1,
        )
    elif model_id == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            depth=6, learning_rate=0.05, iterations=200,
            random_seed=42, verbose=0,
        )
    else:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
        from sklearn.ensemble import (
            RandomForestRegressor, ExtraTreesRegressor,
            GradientBoostingRegressor, AdaBoostRegressor,
        )
        MAP = {
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.01),
            "en": ElasticNet(alpha=0.01, l1_ratio=0.5),
            "huber": HuberRegressor(),
            "rf": RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42),
            "et": ExtraTreesRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42),
            "gbr": GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
            "ada": AdaBoostRegressor(n_estimators=100, random_state=42),
            "lr": Ridge(alpha=0.0),
        }
        if model_id in MAP:
            return MAP[model_id]
        raise ValueError(f"Unknown model_id: {model_id}")


__all__ = [
    "load_v7_7_data",
    "load_train_panel",
    "phase1_compare_models",
    "phase1_get_top_models",
    "phase2_sklearn_rolling",
    "CANDIDATE_MODELS",
]
