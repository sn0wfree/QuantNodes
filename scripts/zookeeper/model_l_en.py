# coding: utf-8
"""L-En 5 线性模型等权集成 — P3 实装.

论文 §3.3 关键发现:
  - OLS C 组 R²OOS = -101.68% (直接崩溃)
  - Ridge 2.80% / EN 3.00% (L2/L1 惩罚后稳健)
  - MLP 0.33% (非线性在小样本失败)
  - **L-En (5 线性模型等权集成) + C/M/S = R²OOS = 3.55%**

模型: OLS / PLS(n=5) / RidgeCV / LassoCV / ElasticNetCV
非线性 (RF/GBRT/XGBoost/MLP) 论文对照组, 本项目 P3 暂不实装.

⚠️ 本项目数据 2018-2025 (7 年), 论文 1988-2022 (34 年), 训练期不严格分.
   roll_window 默认为 36 (论文 120 的 1/3) 保证样本量.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import (ElasticNetCV, LassoCV, LinearRegression,
                                  RidgeCV)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .config import (EN_L1_RATIOS, LASSO_ALPHAS, N_FOLDS_CV,
                      PLS_N_COMPONENTS, ROLL_WINDOW, TRAIN_END)

LOG = logging.getLogger(__name__)


def make_models() -> dict:
    """5 线性模型工厂 (沿论文 §3.2)."""
    return {
        "OLS": LinearRegression(),
        "PLS": PLSRegression(n_components=PLS_N_COMPONENTS),
        "Ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "LASSO": LassoCV(alphas=LASSO_ALPHAS, cv=TimeSeriesSplit(n_splits=N_FOLDS_CV),
                         max_iter=10000),
        "ElasticNet": ElasticNetCV(l1_ratio=EN_L1_RATIOS,
                                   cv=TimeSeriesSplit(n_splits=N_FOLDS_CV),
                                   max_iter=10000),
    }


def _fit_predict_one(model, X_train: pd.DataFrame, y_train: pd.Series,
                     X_test: pd.DataFrame) -> np.ndarray:
    """单模型 fit+predict, 含标准化 (论文每估计期前零均值单位方差)."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train.fillna(0))
    X_te = scaler.transform(X_test.fillna(0))
    if hasattr(y_train, "values"):
        y_tr = y_train.values
    else:
        y_tr = np.asarray(y_train)
    # y 也标准化, 预测后还原
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-12
    y_tr_norm = (y_tr - y_mean) / y_std
    try:
        model.fit(X_tr, y_tr_norm)
        y_pred_norm = model.predict(X_te)
        if y_pred_norm.ndim > 1:
            y_pred_norm = y_pred_norm.ravel()
        return y_pred_norm * y_std + y_mean
    except Exception as e:
        LOG.warning(f"  模型失败 ({type(model).__name__}): {e}")
        return np.full(len(X_test), np.nan)


def fit_predict_ts(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, models: dict | None = None) -> np.ndarray:
    """TS 模式 (预测 AVG 市场组合): 单 y, 多 X.

    Returns:
        y_test 预测数组 (1D).
    """
    models = models or make_models()
    preds = []
    for name, model in models.items():
        p = _fit_predict_one(model, X_train, y_train, X_test)
        preds.append(p)
    return np.nanmean(np.vstack(preds), axis=0)


def fit_predict_cs(X_train: pd.DataFrame, y_train: pd.DataFrame,
                   X_test: pd.DataFrame, models: dict | None = None) -> pd.DataFrame:
    """CS 模式 (预测每商品预期收益): 多 y, 多 X.

    Args:
        X_train: MultiIndex columns (product, feature).
        y_train: 商品收益, columns=product.
        X_test: 同 X_train 结构.

    Returns:
        预测 DataFrame, columns=product, index=X_test row index.
    """
    models = models or make_models()
    out = {}
    for prod in y_train.columns:
        cols_train = X_train[prod] if prod in X_train.columns.get_level_values(0) else None
        cols_test = X_test[prod] if prod in X_test.columns.get_level_values(0) else None
        if cols_train is None or cols_test is None:
            out[prod] = pd.Series(np.nan, index=X_test.index)
            continue
        preds = []
        for name, model in models.items():
            p = _fit_predict_one(model, cols_train, y_train[prod], cols_test)
            preds.append(p)
        out[prod] = pd.Series(np.nanmean(np.vstack(preds), axis=0), index=X_test.index)
    return pd.DataFrame(out)


def rolling_train_predict_ts(features: pd.DataFrame, target: pd.Series,
                             train_end: str = TRAIN_END,
                             test_start: str | None = None,
                             roll_window: int = ROLL_WINDOW) -> pd.DataFrame:
    """TS 模式: 滚动训练 + 月度再预测.

    Args:
        features: Group C 截面均值 + Group M/S 面板, columns=feature.
        target: AVG 市场组合收益, index=trade_date.
        train_end: 训练期截止 (论文 2004-12-31).
        test_start: 测试期起点 (None=默认 train_end 之后).
        roll_window: 滚动训练窗口 (月数).

    Returns:
        DataFrame: index=trade_date, columns=['y_true', 'y_pred', 'y_baseline'].
    """
    test_start = test_start or train_end
    train_end_ts = pd.Timestamp(train_end)
    test_mask = features.index > train_end_ts
    test_dates = features.index[test_mask]
    if len(test_dates) == 0:
        LOG.warning(f"无测试期数据 (train_end={train_end})")
        return pd.DataFrame()

    LOG.info(f"TS 滚动训练: train_end={train_end}, test={test_dates[0]}~{test_dates[-1]}, "
             f"window={roll_window}")
    rows = []
    for t in test_dates:
        t_pos = features.index.get_loc(t)
        start_pos = max(0, t_pos - roll_window)
        X_tr = features.iloc[start_pos:t_pos]
        y_tr = target.iloc[start_pos:t_pos]
        X_te = features.iloc[[t_pos]]
        if len(X_tr) < 12:  # 至少 12 月训练
            continue
        y_pred = fit_predict_ts(X_tr, y_tr, X_te)[0]
        rows.append({
            "trade_date": t,
            "y_true": target.iloc[t_pos],
            "y_pred": y_pred,
            "y_baseline": target.iloc[:t_pos].mean() if t_pos > 0 else 0.0,
        })
    return pd.DataFrame(rows).set_index("trade_date")


def rolling_train_predict_cs(features_c: pd.DataFrame, features_m: pd.DataFrame,
                             returns: pd.DataFrame,
                             train_end: str = TRAIN_END,
                             test_start: str | None = None,
                             roll_window: int = ROLL_WINDOW,
                             beta_window: int | None = None) -> pd.DataFrame:
    """CS 模式: 滚动训练 + 月度再预测 r̂ 矩阵.

    Returns:
        DataFrame: index=trade_date, columns=product. 值 = L-En 预测下期收益.
    """
    from . import beta_rolling
    test_start = test_start or train_end
    train_end_ts = pd.Timestamp(train_end)
    test_mask = returns.index > train_end_ts
    test_dates = returns.index[test_mask]
    if len(test_dates) == 0:
        LOG.warning(f"无测试期数据 (train_end={train_end})")
        return pd.DataFrame()

    LOG.info(f"CS 滚动训练: train_end={train_end}, test={test_dates[0]}~{test_dates[-1]}, "
             f"window={roll_window}, beta_window={beta_window or min(35, roll_window)}")
    # Beta 窗口不能超过训练窗口, 否则全 NaN. pct_change 后首行 NaN, 有效数据 = roll_window-1.
    eff_beta_window = beta_window if beta_window is not None else min(35, roll_window)
    out = {}
    for t in test_dates:
        t_pos = returns.index.get_loc(t)
        start_pos = max(0, t_pos - roll_window)
        ret_tr = returns.iloc[start_pos:t_pos]
        if len(ret_tr) < 12:
            continue
        feat_c_tr = features_c.iloc[start_pos:t_pos]
        feat_c_te = features_c.iloc[[t_pos]]
        beta_full = beta_rolling.rolling_beta(ret_tr, features_m.iloc[start_pos:t_pos],
                                                window=eff_beta_window)
        # 构造 t 期商品面板
        if beta_full.empty:
            out[t] = pd.Series(np.nan, index=returns.columns)
            continue
        # 简化: 用 t-1 的 Beta 预测 t (避免前瞻)
        beta_t_minus_1 = beta_full.iloc[[-1]]
        if beta_t_minus_1.empty or beta_t_minus_1.isna().all().all():
            out[t] = pd.Series(np.nan, index=returns.columns)
            continue
        # 构造训练期商品面板: Group C 跨期 + (Group M × Beta)
        panels_tr, panels_te = [], []
        feat_c_tr_cols = (feat_c_tr.columns.get_level_values(0)
                          if isinstance(feat_c_tr.columns, pd.MultiIndex) else [])
        feat_c_te_cols = (feat_c_te.columns.get_level_values(0)
                          if isinstance(feat_c_te.columns, pd.MultiIndex) else [])
        beta_cols = (beta_full.columns.get_level_values(0)
                     if isinstance(beta_full.columns, pd.MultiIndex) else [])
        te_cols = (beta_t_minus_1.columns.get_level_values(0)
                   if isinstance(beta_t_minus_1.columns, pd.MultiIndex) else [])
        for prod in returns.columns:
            if prod not in feat_c_tr_cols or prod not in feat_c_te_cols:
                continue
            cols_tr_c = feat_c_tr[prod]
            cols_te_c = feat_c_te[prod]
            # 保留 MultiIndex (prod, feat), 避免 concat 错位
            cols_tr_c.columns = pd.MultiIndex.from_product(
                [[prod], cols_tr_c.columns], names=["product", "feature"])
            cols_te_c.columns = pd.MultiIndex.from_product(
                [[prod], cols_te_c.columns], names=["product", "feature"])
            beta_cols = beta_full.columns.levels[0]
            te_cols = beta_t_minus_1.columns.levels[0]
            macro_tr = (beta_full[prod].shift(1) if prod in beta_cols else pd.DataFrame())
            macro_te = (beta_t_minus_1[prod] if prod in te_cols else pd.DataFrame())
            full_tr = cols_tr_c.copy()
            full_te = cols_te_c.copy()
            if not macro_tr.empty:
                macro_tr.columns = pd.MultiIndex.from_product([[prod], macro_tr.columns],
                                                              names=["product", "feature"])
                full_tr = pd.concat([full_tr, macro_tr], axis=1)
            if not macro_te.empty:
                macro_te.columns = pd.MultiIndex.from_product([[prod], macro_te.columns],
                                                             names=["product", "feature"])
                full_te = pd.concat([full_te, macro_te], axis=1)
            panels_tr.append(full_tr)
            panels_te.append(full_te)
        if not panels_tr:
            out[t] = pd.Series(np.nan, index=returns.columns)
            continue
        X_tr_full = pd.concat(panels_tr, axis=1)
        X_te_full = pd.concat(panels_te, axis=1)
        preds = fit_predict_cs(X_tr_full, ret_tr, X_te_full)
        out[t] = preds.iloc[0]
    return pd.DataFrame(out).T
