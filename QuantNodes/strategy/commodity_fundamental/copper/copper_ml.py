# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_ml.py

[已归档 2026-08-21] 铜ML模型: RandomForest + XGBoost 捕捉非线性关系

归档原因 (10_copper_evaluation / 11_copper_ablation 评估结论):
    - 统一真实价口径 (2018+) Sharpe 0.62, 不优于 v4 (0.71) 且信号同族高相关
    - 结构 = v5 + ML预测器 (继承v5的经济状态层等负贡献层)
    - 复杂度高 (16特征+非线性项), 无增量价值
    - 另注意: build_signal_ml 返回 (signal, info) 元组, 调用方需解包

核心思想:
  - 使用机器学习捕捉非线性关系
  - 特征工程: 原始因子 + 交互项 + 非线性变换
  - Walk-forward回测: 避免过拟合
  - 集成学习: RandomForest + XGBoost

数据:
  - DXY (FRED DTWEXBGS)
  - TIPS (FRED DFII10)
  - LME库存 (天风API)
  - 钻机数 (akshare)
  - PMI (akshare)
  - 动量因子
  - 波动率状态
  - 宏观趋势
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..kalman_filter import weekly_kalman_center
from ..pipeline_v3 import _close, _dxy, _tips10y, _tfapi, _rig, _pmi
from ..sign_constrained_lasso import rolling_zscore, rolling_phy_center, compute_valuation_gap
from ..inventory_quadrant import inventory_state_for_product
from ..valuation_gap import industry_momentum, apply_valuation_layer
from ..portfolio import compute_metrics

DB = Path("data/cmf/fundamental.duckdb")


def build_features(close: pd.Series, dxy: pd.Series, tips: pd.Series,
                   lme: pd.Series, rig: pd.Series, pmi: pd.Series) -> pd.DataFrame:
    """构建特征矩阵 (原始因子 + 非线性变换 + 交互项)."""
    close_w = close.resample("W-FRI").last()
    ret_w = close_w.pct_change()

    # 原始因子z-score
    features = pd.DataFrame({
        "dxy_neg": -rolling_zscore(dxy, 104),
        "tips_neg": -rolling_zscore(tips, 104),
        "lme_neg": -rolling_zscore(lme, 52),
        "rig_neg": -rolling_zscore(rig, 52),
        "pmi": rolling_zscore(pmi, 52),
        "mom_2w": rolling_zscore(ret_w.rolling(2).sum(), 52),
        "momentum_13w": rolling_zscore(ret_w.rolling(13).sum(), 52),
    })

    # 波动率状态
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    vol_pct = vol_4w.rolling(52).rank(pct=True)
    features["vol_pct"] = vol_pct

    # 宏观趋势
    dxy_ma20 = dxy.rolling(20).mean()
    pmi_ma20 = pmi.rolling(20).mean()
    features["dxy_trend"] = (dxy < dxy_ma20).astype(float)
    features["pmi_trend"] = (pmi > pmi_ma20).astype(float)

    # 非线性变换
    features["dxy_neg_sq"] = features["dxy_neg"] ** 2
    features["mom_2w_sq"] = features["mom_2w"] ** 2
    features["vol_pct_sq"] = features["vol_pct"] ** 2

    # 交互项
    features["dxy_x_mom"] = features["dxy_neg"] * features["mom_2w"]
    features["vol_x_mom"] = features["vol_pct"] * features["mom_2w"]
    features["pmi_x_dxy"] = features["pmi"] * features["dxy_neg"]

    return features


def walk_forward_ml(X: pd.DataFrame, y: pd.Series,
                    refit_weeks: int = 26, min_train_weeks: int = 52,
                    model_type: str = "rf") -> tuple[pd.Series, dict]:
    """Walk-forward ML回测.

    返回:
        predictions: OOS预测序列
        metrics: 各重估点的指标
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error

    n = len(X)
    refit_points = list(range(min_train_weeks, n, refit_weeks))
    if n - 1 not in refit_points:
        refit_points.append(n - 1)

    predictions = pd.Series(index=X.index, dtype=float)
    feature_importance = []

    for i, T in enumerate(refit_points):
        X_hist = X.iloc[:T].dropna()
        y_hist = y.iloc[:T].reindex(X_hist.index).dropna()
        common = X_hist.index.intersection(y_hist.index)
        X_hist = X_hist.loc[common]
        y_hist = y_hist.loc[common]

        if len(X_hist) < min_train_weeks:
            continue

        # 训练模型
        if model_type == "rf":
            model = RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            )
        elif model_type == "gb":
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_samples_leaf=10, random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_hist, y_hist)

        # 记录特征重要性
        if hasattr(model, 'feature_importances_'):
            feature_importance.append({
                "refit_idx": T,
                "importance": dict(zip(X.columns, model.feature_importances_)),
            })

        # OOS预测: 用本次模型预测到下次重估之前的所有样本
        next_T = next((tp for tp in refit_points if tp > T), n)
        X_future = X.iloc[T:next_T].copy()
        X_future_clean = X_future.dropna()
        if len(X_future_clean) > 0:
            for fdate in X_future_clean.index:
                xv = X_future_clean.loc[fdate].values.reshape(1, -1)
                pred = model.predict(xv)[0]
                predictions.loc[fdate] = pred

    return predictions, {"feature_importance": feature_importance}


def build_signal_ml(symbol: str = "CU0", con=None,
                    model_type: str = "rf") -> pd.DataFrame:
    """铜ML模型: RandomForest/XGBoost.

    特征工程 + 非线性变换 + Walk-forward回测.
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    close = _close(symbol, con)
    close_w = close.resample("W-FRI").last()
    mu_market = weekly_kalman_center(close, q=0.015)

    dxy = _dxy(con).resample("W-FRI").last()
    tips = _tips10y(con).resample("W-FRI").last()
    lme = _tfapi(con, "LME库存总计_铜").resample("W-FRI").last()
    rig = _rig(con).resample("W-FRI").last()
    pmi = _pmi(con).resample("W-FRI").last().ffill()

    # 构建特征
    features = build_features(close, dxy, tips, lme, rig, pmi)
    features = features.dropna(how="any")

    # 目标变量
    y = pd.Series(np.log(mu_market).diff().reindex(features.index)).dropna()
    common = features.index.intersection(y.index)
    features = features.loc[common]
    y = y.loc[common]

    # Walk-forward ML
    predictions, ml_info = walk_forward_ml(features, y, refit_weeks=26,
                                           min_train_weeks=52, model_type=model_type)

    # 估值缺口
    mu_aligned = mu_market.reindex(common)
    p_phy = rolling_phy_center(mu_aligned, predictions, window=13)
    gap = compute_valuation_gap(mu_aligned, p_phy)

    # 库存象限
    inv_state = inventory_state_for_product(
        lme.reindex(common).ffill(),
        close_w.reindex(common),
        z_window=52, inv_dir=-1,
    )
    industry_dz = industry_momentum(rig.reindex(common).ffill(), z_window=52, delta_window=4)

    # 估值层
    base_pos = apply_valuation_layer(
        inv_state["quadrant_pos"],
        gap.reindex(common),
        industry_dz.reindex(common),
        gap_low_threshold=0.01, gap_high_threshold=-0.01,
    )

    # 波动率状态 + 宏观趋势
    ret_w = close_w.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    vol_pct = vol_4w.rolling(52).rank(pct=True)
    mom_2w = ret_w.rolling(2).sum()

    vol_aligned = vol_pct.reindex(common).fillna(0.5)
    mom_aligned = mom_2w.reindex(common).fillna(0)
    dxy_ma20 = dxy.rolling(20).mean()
    pmi_ma20 = pmi.rolling(20).mean()
    dxy_trend = (dxy < dxy_ma20).astype(float).reindex(common).fillna(0)
    pmi_trend = (pmi > pmi_ma20).astype(float).reindex(common).fillna(0)

    final_pos = base_pos.copy()
    low_vol_mask = vol_aligned < 0.3
    high_vol_mask = vol_aligned > 0.7
    final_pos[low_vol_mask & (mom_aligned > 0)] = 1.0
    final_pos[high_vol_mask] = 0.4

    # 宏观趋势加成
    macro_bull = (dxy_trend == 1) & (pmi_trend == 1)
    final_pos[macro_bull & (final_pos < 1.0)] = final_pos[macro_bull & (final_pos < 1.0)] + 0.3
    final_pos = final_pos.clip(upper=1.0)
    final_pos.iloc[0] = 0.0

    # 生成信号
    full_signal = pd.DataFrame({
        "M": close_w.reindex(common),
        "p_phy": p_phy.reindex(common),
        "gap": gap.reindex(common),
        "vol_pct": vol_pct.reindex(common),
        "mom_2w": mom_2w.reindex(common),
        "d_inv": inv_state["d_inv"].reindex(common),
        "d_price": inv_state["d_price"].reindex(common),
        "quadrant_pos": inv_state["quadrant_pos"].reindex(common),
        "industry_dz": industry_dz.reindex(common),
        "ml_pred": predictions.reindex(common),
        "position": final_pos.reindex(common),
    })

    return full_signal, ml_info


def backtest_ml(signal: pd.DataFrame, close: pd.Series,
                cost_bps: float = 5.0) -> dict:
    """回测铜ML模型."""
    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    strat_ret = pos.shift(1) * ret
    pos_change = pos.diff().abs()
    cost = pos_change * cost_bps / 10000
    strat_ret = strat_ret - cost
    strat_ret = strat_ret.dropna()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "total_ret": 0, "ann_ret": 0, "ann_vol": 0,
                "sharpe": 0, "max_dd": 0, "calmar": 0, "win_rate": 0,
                "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1 if len(nav) > 0 else 0
    n_weeks = len(strat_ret)
    n_years = n_weeks / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min() if len(nav) > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (strat_ret > 0).mean()

    return {"n_weeks": n_weeks, "total_ret": total_ret, "ann_ret": ann_ret,
            "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd,
            "calmar": calmar, "win_rate": win_rate, "nav": nav}


__all__ = ["build_features", "walk_forward_ml", "build_signal_ml", "backtest_ml"]
