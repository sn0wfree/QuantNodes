# coding=utf-8
"""v7.7 宏观子策略: PyCaret 多模型因子择时 [ARCHIVED: v7.7 ML 失败, R2 ≈ 0].

替代 v7.6 的 TV-PR 线性模型，用 PyCaret 对比多种 ML 模型。
复用 v7.6 的 construct_portfolio + calculate_daily_nav 逻辑。

回测流程:
  1. 加载数据: 39 因子 (17 macro + 22 量价)
  2. 滚动估计: 用 ML 模型预测每个资产的得分
  3. 构造组合: 按得分排序, 逆波动率加权
  4. 扣除成本: 5bp 佣金 + 5bp 滑点
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data_loader_v7_6 import (
    load_daily_etf_returns,
    load_v7_6_data,
)
from .data_loader import (
    EXPANDED_COLS,
    EQUITY_ETF_COLS,
    COMMODITY_ETF_COLS,
    EXPANDED_BOND_INDICES,
)
from .pycaret_estimator import (
    load_v7_7_data,
    phase2_sklearn_rolling,
)
from .macro_substrategy_v7_6 import (
    construct_portfolio,
    calculate_daily_nav,
)


# ============================================================
# Config
# ============================================================
@dataclass
class V7_7Config:
    """v7.7 PyCaret 多模型配置."""
    name: str = "v7_7_pycaret"

    # 资产池
    asset_pool: str = "expanded"
    index_pool: tuple[str, ...] = tuple(EXPANDED_COLS)
    equity_cols: tuple[str, ...] = tuple(EQUITY_ETF_COLS)
    commodity_cols: tuple[str, ...] = tuple(COMMODITY_ETF_COLS)
    bond_cols: tuple[str, ...] = tuple(EXPANDED_BOND_INDICES)

    # ML 模型
    model_id: str = "lightgbm"  # PyCaret 模型 ID
    target_type: Literal["raw", "rank"] = "raw"
    min_history: int = 52  # 最少训练期（周）

    # 调仓 (周频)
    rebalance_freq: str = "W"

    # 成本
    commission_bp: float = 5.0
    slippage_bp: float = 5.0
    cost_enabled: bool = True

    # 选股
    top_n: int = 10
    max_weight: float = 0.25
    vol_window: int = 26
    vol_floor: float = 0.01


# ============================================================
# 回测主函数
# ============================================================
def run_v7_7_backtest(
    X_panel: np.ndarray | None = None,
    Y: pd.DataFrame | None = None,
    valid_codes: list[str] | None = None,
    cfg: V7_7Config | None = None,
    return_weights: bool = False,
    return_daily: bool = False,
    return_scores: bool = False,
    verbose: bool = True,
) -> pd.Series | tuple:
    """v7.7 PyCaret 多模型端到端回测.

    Args:
        X_panel: (T, N, K) 因子面板 (None = 自动加载)
        Y: (T, N) 周频收益 DataFrame (None = 自动加载)
        valid_codes: 有效资产代码列表
        cfg: v7.7 配置
        return_weights: 是否返回持仓权重
        return_daily: 是否返回日频 NAV
        return_scores: 是否返回预测分数
        verbose: 打印进度

    Returns:
        nav: 周频 NAV
        weights_df: (可选) 持仓权重
        daily_nav: (可选) 日频 NAV
        scores: (可选) 预测分数
    """
    cfg = cfg or V7_7Config()

    # 1. 加载数据
    if X_panel is None or Y is None:
        # 尝试用 v7.7 数据
        try:
            X_panel_77, Y_raw, Y_rank, factor_names = load_v7_7_data()
        except FileNotFoundError:
            # 降级到 v7.6 数据
            X_panel_77, Y_df, codes = load_v7_6_data()
            Y_raw = Y_df.values
            Y_rank = None

        if X_panel is None:
            X_panel = X_panel_77
        if Y is None:
            Y = pd.DataFrame(
                Y_raw,
                columns=valid_codes or [str(i) for i in range(Y_raw.shape[1])],
            )

    T, N, K = X_panel.shape

    # 2. 准备标签
    if cfg.target_type == "rank":
        Y_label = np.full_like(Y.values if isinstance(Y, pd.DataFrame) else Y, np.nan)
        Y_vals = Y.values if isinstance(Y, pd.DataFrame) else Y
        for t in range(T):
            valid = ~np.isnan(Y_vals[t, :])
            n_valid = valid.sum()
            if n_valid > 1:
                ranks = pd.Series(Y_vals[t, valid]).rank().values
                Y_label[t, valid] = (ranks - 1) / (n_valid - 1)
    else:
        Y_label = Y.values if isinstance(Y, pd.DataFrame) else Y

    # 3. 滚动估计
    scores = phase2_sklearn_rolling(
        X_panel, Y_label,
        model_id=cfg.model_id,
        min_history=cfg.min_history,
        verbose=verbose,
    )

    # 4. 构造组合 (复用 v7.6 逻辑)
    scores_df = pd.DataFrame(scores, index=Y.index if isinstance(Y, pd.DataFrame) else None)
    nav, weights_df = construct_portfolio(Y, scores_df, cfg, return_weights=True)

    # 5. 返回结果
    result = [nav]
    if return_weights:
        result.append(weights_df)
    if return_daily:
        daily_returns = load_daily_etf_returns()
        daily_nav = calculate_daily_nav(weights_df, daily_returns, cfg)
        result.append(daily_nav)
    if return_scores:
        result.append(scores)

    return tuple(result) if len(result) > 1 else nav


def run_v7_7_batch(
    model_ids: list[str] | None = None,
    target_type: Literal["raw", "rank"] = "raw",
    verbose: bool = True,
) -> dict[str, pd.Series]:
    """批量运行多个模型，返回 NAV 字典.

    Returns:
        dict[model_id -> nav_series]
    """
    if model_ids is None:
        model_ids = ["ridge", "lightgbm", "rf"]

    X, Y_raw, Y_rank, names = load_v7_7_data()
    Y = pd.DataFrame(Y_raw)

    results = {}
    for mid in model_ids:
        if verbose:
            print(f"\n{'='*40}")
            print(f"Model: {mid}")
            print(f"{'='*40}")

        cfg = V7_7Config(model_id=mid, target_type=target_type)
        try:
            nav = run_v7_7_backtest(
                X_panel=X, Y=Y, cfg=cfg, verbose=verbose,
            )
            results[mid] = nav
            if verbose:
                print(f"  NAV end: {nav.iloc[-1]:.4f}")
        except Exception as e:
            if verbose:
                print(f"  FAILED: {e}")

    return results


# ============================================================
# 工厂函数
# ============================================================
def v7_7_lgbm(**overrides) -> V7_7Config:
    """v7.7 LightGBM 配置."""
    return V7_7Config(model_id="lightgbm", **overrides)


def v7_7_ridge(**overrides) -> V7_7Config:
    """v7.7 Ridge 线性基准."""
    return V7_7Config(model_id="ridge", **overrides)


def v7_7_rf(**overrides) -> V7_7Config:
    """v7.7 Random Forest."""
    return V7_7Config(model_id="rf", **overrides)


def v7_7_xgboost(**overrides) -> V7_7Config:
    """v7.7 XGBoost."""
    return V7_7Config(model_id="xgboost", **overrides)


def v7_7_catboost(**overrides) -> V7_7Config:
    """v7.7 CatBoost."""
    return V7_7Config(model_id="catboost", **overrides)


__all__ = [
    "V7_7Config",
    "run_v7_7_backtest",
    "run_v7_7_batch",
    "v7_7_lgbm",
    "v7_7_ridge",
    "v7_7_rf",
    "v7_7_xgboost",
    "v7_7_catboost",
]
