# coding: utf-8
"""L-En 5 线性模型等权集成 — P3 待实装.

论文 §3.3 关键发现:
  - OLS C 组 R²OOS = -101.68% (直接崩溃)
  - Ridge 2.80% / EN 3.00% (L2/L1 惩罚后稳健)
  - MLP 0.33% (非线性在小样本失败)
  - **L-En (5 线性模型等权集成) + C/M/S = R²OOS = 3.55%**

模型: OLS / PLS(n=5) / RidgeCV / LassoCV / ElasticNetCV
非线性 (RF/GBRT/XGBoost/MLP) 论文对照组, 本项目 P3 暂不实装.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                  RidgeCV)

from .config import (EN_L1_RATIOS, LASSO_ALPHAS, N_FOLDS_CV,
                      PLS_N_COMPONENTS, ROLL_WINDOW, TRAIN_END)

LOG = logging.getLogger(__name__)


def make_models() -> dict:
    """5 线性模型工厂 (沿论文 §3.2)."""
    return {
        "OLS": LinearRegression(),
        "PLS": PLSRegression(n_components=PLS_N_COMPONENTS),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "LASSO": LassoCV(alphas=LASSO_ALPHAS, cv=N_FOLDS_CV, max_iter=10000),
        "ElasticNet": ElasticNetCV(l1_ratio=EN_L1_RATIOS, cv=N_FOLDS_CV, max_iter=10000),
    }


def fit_predict_ts(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame) -> np.ndarray:
    """TS 模式 (预测 AVG 市场组合): 单 y, 多 X.

    Returns:
        y_test 预测数组 (1D).
    """
    # TODO P3: 训练 5 模型 → 5 预测 → 等权平均
    raise NotImplementedError("P3 待实装")


def fit_predict_cs(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame) -> np.ndarray:
    """CS 模式 (预测每商品预期收益): 多 y, 多 X.

    Returns:
        y_test 预测数组 (2D: n_test × n_products).
    """
    # TODO P3: 每个商品单独拟合 → 平均
    raise NotImplementedError("P3 待实装")


def rolling_train_predict(X: pd.DataFrame, y: pd.Series | pd.DataFrame,
                          train_end: str = TRAIN_END,
                          test_start: str = "2005-01-01",
                          roll_window: int = ROLL_WINDOW) -> pd.DataFrame:
    """120 月滚动训练 + 月度再预测 (论文 §3.2 测试期)."""
    # TODO P3: for t in test_dates: 取 [t-120, t) 训练, 预测 t
    raise NotImplementedError("P3 待实装")
