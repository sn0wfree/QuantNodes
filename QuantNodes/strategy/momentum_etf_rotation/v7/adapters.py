# coding=utf-8
"""v7 统一适配器层 — 所有 v7 策略用同一个 backtest_fn 接口.

设计:
  - 所有 v7 策略都暴露 backtest_fn(Y, X, **params) → (shares_df, prices_df, weights_df)
  - 输入: (T, N, K) 因子面板 + (T, N) 资产收益 (日期索引)
  - 输出:
    - shares: (T_daily, N) 日频份额 (按 NAV=1 基准: shares = weights / prices)
    - prices: (T_daily, N) 日频价格
    - weights: (T_native, N) 原频率目标权重 (周频/季频)
  - 不在适配器内部生成 NAV, 由 walk_forward.generate_nav_from_shares 统一累积

支持的策略:
  - v7.3 / v7.5 (Bootstrap-Lasso + FactorRiskParity, 季频调仓)
  - v7.6 / v7.10 (TV-PR + 逆波动率, 周频调仓)
  - v7.11 - v7.14 (TV-PR + 新因子, 周频调仓)
"""
from __future__ import annotations

from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[4]
HF_DIR = REPO / "data" / "high_freq_macro"
DATA_DIR = REPO / "data" / "real"


# ============================================================
# 数据加载 (统一格式, index.name='date')
# ============================================================
def _normalize_y(Y: pd.DataFrame) -> pd.DataFrame:
    """统一设置 Y.index.name='date'."""
    if isinstance(Y.index, pd.DatetimeIndex) and not Y.index.name:
        Y.index.name = 'date'
    return Y


def load_v7_6_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.6 数据 — (430, 43, 39) 因子面板."""
    from .data_loader_v7_6 import load_v7_6_data
    X, Y, codes = load_v7_6_data()
    return X, _normalize_y(Y), codes


def load_v7_10_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.10 数据 — (430, 43, 36) 因子面板."""
    from .data_loader_v7_6 import load_v7_10_data
    X, Y, codes = load_v7_10_data()
    return X, _normalize_y(Y), codes


def load_v7_11_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.11 数据 — (430, 43, 46) 因子面板."""
    X = np.load(HF_DIR / "v7_11_X_panel.npy")
    Y = pd.read_parquet(HF_DIR / "v7_11_Y_weekly.parquet")
    Y = _normalize_y(Y)
    codes = (HF_DIR / "v7_11_codes.csv").read_text().strip().split("\n")[1:]
    return X, Y, codes


def load_v7_12_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.12 数据 — (430, 43, 42) 因子面板."""
    X = np.load(HF_DIR / "v7_12_X_panel.npy")
    Y = pd.read_parquet(HF_DIR / "v7_12_Y_weekly.parquet")
    Y = _normalize_y(Y)
    codes = (HF_DIR / "v7_12_codes.csv").read_text().strip().split("\n")[1:]
    return X, Y, codes


def load_v7_13_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.13 数据 — (430, 43, 46) 因子面板."""
    X = np.load(HF_DIR / "v7_13_X_panel.npy")
    Y = pd.read_parquet(HF_DIR / "v7_13_Y_weekly.parquet")
    Y = _normalize_y(Y)
    codes = (HF_DIR / "v7_13_codes.csv").read_text().strip().split("\n")[1:]
    return X, Y, codes


def load_v7_14_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.14 数据 — (430, 43, 42) 因子面板, 复用 v7.10 Y."""
    X = np.load(HF_DIR / "v7_14_X_panel.npy")
    Y = pd.read_parquet(HF_DIR / "v7_10_Y_weekly.parquet")
    Y = _normalize_y(Y)
    codes = (HF_DIR / "v7_10_codes.csv").read_text().strip().split("\n")[1:]
    return X, Y, codes


def load_v7_3_data_uniform() -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """加载 v7.3/v7.5 数据 — 周频指数收益 (T, 13) + 因子面板.

    返回 (T_weekly, 13, K) 格式, 因子广播到所有资产.
    """
    from .data_loader import (
        load_factor_returns,
    )

    factor_returns = load_factor_returns()  # (T, 8) 周频因子对数收益
    from .data_loader import load_index_panel
    index_panel = load_index_panel()  # 日对数收益
    # 周频化: 用周末收盘价计算周收益 (pct_change)
    index_weekly = index_panel.resample("W").last().pct_change().dropna()

    # 对齐时间
    common_idx = index_weekly.index.intersection(factor_returns.index)
    index_weekly = index_weekly.loc[common_idx]
    factor_returns = factor_returns.loc[common_idx]

    Y = index_weekly
    T = len(common_idx)
    N = len(Y.columns)
    K = len(factor_returns.columns)

    X = np.zeros((T, N, K), dtype=np.float64)
    for k, col in enumerate(factor_returns.columns):
        X[:, :, k] = factor_returns[col].values[:, np.newaxis]

    codes = list(Y.columns)
    return X, _normalize_y(Y), codes


DATA_LOADERS = {
    "v7.3": load_v7_3_data_uniform,
    "v7.5": load_v7_3_data_uniform,
    "v7.6": load_v7_6_data_uniform,
    "v7.10": load_v7_10_data_uniform,
    "v7.11": load_v7_11_data_uniform,
    "v7.12": load_v7_12_data_uniform,
    "v7.13": load_v7_13_data_uniform,
    "v7.14": load_v7_14_data_uniform,
}


# ============================================================
# 适配器: 把 v7 内部函数包装成统一接口 backtest_fn
# ============================================================
# ============================================================
# Lambda CV 优化
# ============================================================
def _cv_single_fold(Y_arr, X_arr, lambda_tv, lambda_l1, train_end, val_end,
                    min_history=52, step=4, max_iter=50):
    """单折 CV: 用 train 估计 beta, 在 val 上评估."""
    from .tvpr_estimator import tvpr_admm

    # 估计 beta (expanding window)
    beta_warm = None
    beta_last = np.zeros(X_arr.shape[2])
    for t in range(min_history, train_end, step):
        if beta_warm is not None:
            beta_init = np.zeros((t, X_arr.shape[2]))
            beta_init[:beta_warm.shape[0]] = beta_warm
        else:
            beta_init = None
        beta_path = tvpr_admm(Y_arr[:t], X_arr[:t], lambda_tv, lambda_l1,
                              rho=1.0, max_iter=max_iter, tol=1e-4, beta_init=beta_init)
        beta_last = beta_path[-1]
        beta_warm = beta_path

    # 在 val 上评估
    Y_val = Y_arr[train_end:val_end]
    X_val = X_arr[train_end:val_end]
    nav = 1.0
    for t in range(1, len(Y_val)):
        scores = X_val[t] @ beta_last
        valid = ~np.isnan(scores) & ~np.isnan(Y_val[t])
        if valid.sum() == 0:
            continue
        sv = scores[valid]
        top_idx = np.argsort(sv)[-10:]
        chosen = np.where(valid)[0][top_idx]
        ret = float(np.nanmean(Y_val[t][chosen]))
        nav *= (1 + (ret if not np.isnan(ret) else 0.0))
    return nav


def select_lambda_cv(Y, X, min_history=52, n_splits=3, step=4, verbose=False):
    """两阶段 CV 选择最优 lambda.

    Stage 1: 粗搜 10 组合
    Stage 2: 在最优附近细搜 ~25 组合

    Parameters:
        Y: (T, N) 周频资产收益
        X: (T, N, K) 周频因子面板
        min_history: 最少历史期数
        n_splits: CV 折数
        step: beta 更新频率
        verbose: 是否打印进度

    Returns:
        best_lambda_tv, best_lambda_l1
    """
    Y_arr = Y.values if hasattr(Y, 'values') else Y
    X_arr = X if isinstance(X, np.ndarray) else X.values
    T = Y_arr.shape[0]

    # 粗搜网格
    LAMBDA_GRID_COARSE = [
        (0.01, 0.01), (0.01, 0.03), (0.01, 0.05),
        (0.03, 0.01), (0.03, 0.03), (0.03, 0.05),
        (0.05, 0.01), (0.05, 0.03), (0.05, 0.05), (0.05, 0.10),
    ]

    fold_size = (T - min_history) // (n_splits + 1)
    if fold_size < 3:
        return 0.05, 0.01

    # Stage 1: 粗搜
    if verbose:
        print(f"    Stage 1: 粗搜 {len(LAMBDA_GRID_COARSE)} 组合...")

    best_nav = -1
    best_lt, best_ll = 0.05, 0.01

    for lt, ll in LAMBDA_GRID_COARSE:
        navs = []
        for i in range(n_splits):
            train_end = min_history + (i + 1) * fold_size
            val_end = min(train_end + fold_size, T)
            nav = _cv_single_fold(Y_arr, X_arr, lt, ll, train_end, val_end,
                                  min_history=min_history, step=step)
            navs.append(nav)
        mean_nav = np.mean(navs)
        if mean_nav > best_nav:
            best_nav = mean_nav
            best_lt, best_ll = lt, ll

    if verbose:
        print(f"    Stage 1 最优: lambda_tv={best_lt:.3f}, lambda_l1={best_ll:.3f}, NAV={best_nav:.4f}")

    # Stage 2: 细搜
    tv_range = np.arange(max(0.005, best_lt - 0.01), best_lt + 0.015, 0.005)
    ll_range = np.arange(max(0.005, best_ll - 0.01), best_ll + 0.015, 0.005)
    fine_grid = [(round(lt, 4), round(ll, 4)) for lt in tv_range for ll in ll_range]

    if verbose:
        print(f"    Stage 2: 细搜 {len(fine_grid)} 组合...")

    for lt, ll in fine_grid:
        navs = []
        for i in range(n_splits):
            train_end = min_history + (i + 1) * fold_size
            val_end = min(train_end + fold_size, T)
            nav = _cv_single_fold(Y_arr, X_arr, lt, ll, train_end, val_end,
                                  min_history=min_history, step=step)
            navs.append(nav)
        mean_nav = np.mean(navs)
        if mean_nav > best_nav:
            best_nav = mean_nav
            best_lt, best_ll = lt, ll

    if verbose:
        print(f"    最终最优: lambda_tv={best_lt:.3f}, lambda_l1={best_ll:.3f}, NAV={best_nav:.4f}")

    return best_lt, best_ll


def make_v7_6_backtest_fn(version: str = "v7.6") -> Callable:
    """生成 v7.6/v7.10/v7.11-v7.14 的 backtest_fn.

    签名: backtest_fn(Y, X, **params) → (shares_df, prices_df, weights_df)
    """
    from .tvpr_estimator import expanding_window_tvpr
    from .macro_substrategy_v7_6 import (
        V7_6Config, construct_portfolio_components,
    )

    def backtest_fn(
        Y: pd.DataFrame,
        X: np.ndarray,
        **params,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """统一 v7.6+ backtest_fn.

        Parameters:
            Y: (T, N) 训练数据 (日期索引, NO LOOKAHEAD)
            X: (T, N, K) 训练因子面板 (NO LOOKAHEAD)
            **params: top_n, vol_window, max_weight, lambda_tv, lambda_l1, step,
                      use_cv (bool), cv_verbose (bool)

        Returns:
            shares: (T_daily, N) 日频份额 (NAV=1 基准)
            prices: (T_daily, N) 日频价格
            weights: (T_weekly, N) 周频目标权重
        """
        # CV 优化 lambda (如果启用)
        use_cv = params.get("use_cv", False)
        cv_verbose = params.get("cv_verbose", False)

        if use_cv:
            lambda_tv, lambda_l1 = select_lambda_cv(
                Y, X,
                min_history=52,
                n_splits=3,
                step=params.get("step", 4),
                verbose=cv_verbose,
            )
        else:
            lambda_tv = params.get("lambda_tv", 0.15)
            lambda_l1 = params.get("lambda_l1", 0.05)

        cfg = V7_6Config(
            top_n=params.get("top_n", 10),
            vol_window=params.get("vol_window", 26),
            max_weight=params.get("max_weight", 0.25),
            lambda_tv=lambda_tv,
            lambda_l1=lambda_l1,
            step=params.get("step", 13),
            method="expanding",
            cost_enabled=False,  # 成本由 walk_forward 统一扣
            stop_loss_threshold=params.get("stop_loss_threshold", None),
            stop_loss_cooldown=params.get("stop_loss_cooldown", 5),
            trend_filter_enabled=params.get("trend_filter_enabled", False),
        )

        # beta 估计 (只在 Y 上, 无未来函数)
        beta = expanding_window_tvpr(
            Y, X,
            cfg.lambda_tv, cfg.lambda_l1,
            min_history=cfg.min_history,
            step=cfg.step,
        )

        # 生成 shares, prices, weekly_weights
        shares, prices, weekly_weights = construct_portfolio_components(
            Y, X, beta, cfg,
        )
        # 统一设置 index.name='date'
        shares = _normalize_y(shares)
        prices = _normalize_y(prices)
        weekly_weights = _normalize_y(weekly_weights)
        return shares, prices, weekly_weights

    return backtest_fn


def make_v7_3_backtest_fn(version: str = "v7.3") -> Callable:
    """生成 v7.3/v7.5 的 backtest_fn (Bootstrap-Lasso, 季频调仓).

    签名: backtest_fn(Y, X, **params) → (shares_df, prices_df, weights_df)

    注意: v7.3 内部用 index_panel (T_daily, 13) 日对数收益 +
    factor_panel (T_weekly, 8) 周对数收益, 季度调仓.
    """
    from .data_loader import (
        load_index_panel, FACTOR_COLS,
    )
    from .macro_substrategy_v7_3 import (
        V7_3Config, run_v7_3_backtest,
    )

    def backtest_fn(
        Y: pd.DataFrame,
        X: np.ndarray,
        **params,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """v7.3/v7.5 backtest_fn — 返回日频 shares + prices + 季频 weights.

        Parameters:
            Y: (T_weekly, N) 周频训练数据 (日期索引)
            X: (T_weekly, N, K) 周频训练因子面板
            **params: quarter_window, max_weight, bootstrap_times, ...

        Returns:
            shares: (T_daily, N) 日频份额 (NAV=1 基准)
            prices: (T_daily, N) 日频价格
            weights: (T_quarter, N) 季频目标权重
        """
        cfg = V7_3Config(
            quarter_window=params.get("quarter_window", 8),
            max_weight=params.get("max_weight", 0.5),
            bootstrap_times=params.get("bootstrap_times", 500),
            bootstrap_random_state=params.get("bootstrap_random_state", 42),
            stop_loss_enabled=False,
            trend_filter_enabled=False,
        )

        # 从 X 提取 factor_returns
        K = X.shape[2]
        factor_cols_to_use = FACTOR_COLS[:K] if K <= len(FACTOR_COLS) else FACTOR_COLS
        if X.shape[2] > 0:
            factor_returns = pd.DataFrame(
                X[:, 0, :],
                index=Y.index,
                columns=factor_cols_to_use,
            )
        else:
            factor_returns = pd.DataFrame(index=Y.index)

        # 加载日频指数收益 (与 factor_returns 时间对齐)
        start_date = (Y.index[0] - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        index_panel = load_index_panel(start=start_date)
        # 只保留 cfg.index_pool 中的列
        available_cols = [c for c in cfg.index_pool if c in index_panel.columns]
        index_panel = index_panel[available_cols].copy()

        # 运行 v7.3 回测 (返回 NAV + 季频权重)
        nav_weekly, quarter_weights = run_v7_3_backtest(
            index_panel, factor_returns, cfg, return_weights=True,
        )

        # 日频价格 = exp(累积对数收益), 起点 = 1.0
        daily_prices = np.exp(index_panel.cumsum())
        daily_prices = daily_prices.ffill().dropna(how='all')
        # 对齐列
        daily_prices = daily_prices.reindex(columns=quarter_weights.columns)

        # 季频权重 → 日频份额
        from ..common.walk_forward import weights_to_daily_shares
        shares = weights_to_daily_shares(quarter_weights, daily_prices)

        # 统一设置 index.name='date'
        shares = _normalize_y(shares)
        daily_prices = _normalize_y(daily_prices)
        quarter_weights = _normalize_y(quarter_weights)
        return shares, daily_prices, quarter_weights

    return backtest_fn


# ============================================================
# 策略注册表
# ============================================================
STRATEGY_REGISTRY = {
    "v7.3": {
        "data_loader": load_v7_3_data_uniform,
        "backtest_fn_factory": make_v7_3_backtest_fn,
        "param_space": {
            "quarter_window": [4, 6, 8],
            "max_weight": [0.4, 0.5, 0.6],
            "bootstrap_times": [200, 500],
        },
    },
    "v7.5": {
        "data_loader": load_v7_3_data_uniform,
        "backtest_fn_factory": make_v7_3_backtest_fn,
        "param_space": {
            "quarter_window": [4, 8],
            "max_weight": [0.4, 0.5],
            "bootstrap_times": [500],
        },
    },
    "v7.6": {
        "data_loader": load_v7_6_data_uniform,
        "backtest_fn_factory": lambda: make_v7_6_backtest_fn("v7.6"),
        "param_space": {
            "top_n": [5, 10, 15],
            "vol_window": [13, 26, 52],
            "max_weight": [0.15, 0.25, 0.35],
            "lambda_tv": [0.05, 0.10, 0.15],
            "lambda_l1": [0.03, 0.05, 0.08],
        },
    },
    "v7.10": {
        "data_loader": load_v7_10_data_uniform,
        "backtest_fn_factory": lambda: make_v7_6_backtest_fn("v7.10"),
        "param_space": {
            "top_n": [5, 10, 15],
            "vol_window": [13, 26, 52],
            "max_weight": [0.15, 0.25, 0.35],
        },
    },
    "v7.11": {
        "data_loader": load_v7_11_data_uniform,
        "backtest_fn_factory": lambda: make_v7_6_backtest_fn("v7.11"),
        "param_space": {
            "top_n": [5, 10, 15],
            "vol_window": [13, 26, 52],
            "max_weight": [0.15, 0.25, 0.35],
        },
    },
    "v7.12": {
        "data_loader": load_v7_12_data_uniform,
        "backtest_fn_factory": lambda: make_v7_6_backtest_fn("v7.12"),
        "param_space": {
            "top_n": [5, 10, 15],
            "vol_window": [13, 26, 52],
            "max_weight": [0.15, 0.25, 0.35],
        },
    },
    "v7.13": {
        "data_loader": load_v7_13_data_uniform,
        "backtest_fn_factory": lambda: make_v7_6_backtest_fn("v7.13"),
        "param_space": {
            "top_n": [5, 10, 15],
            "vol_window": [13, 26, 52],
            "max_weight": [0.15, 0.25, 0.35],
        },
    },
    "v7.14": {
        "data_loader": load_v7_14_data_uniform,
        "backtest_fn_factory": lambda: make_v7_6_backtest_fn("v7.14"),
        "param_space": {
            "top_n": [5, 10, 15],
            "vol_window": [13, 26, 52],
            "max_weight": [0.15, 0.25, 0.35],
        },
    },
}


def get_strategy(version: str) -> dict:
    """获取策略配置."""
    if version not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown version: {version}, available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[version]


def list_versions() -> list[str]:
    """列出所有支持的版本."""
    return list(STRATEGY_REGISTRY.keys())


__all__ = [
    "load_v7_3_data_uniform",
    "load_v7_6_data_uniform",
    "load_v7_10_data_uniform",
    "load_v7_11_data_uniform",
    "load_v7_12_data_uniform",
    "load_v7_13_data_uniform",
    "load_v7_14_data_uniform",
    "DATA_LOADERS",
    "make_v7_6_backtest_fn",
    "make_v7_3_backtest_fn",
    "weights_to_daily_shares",
    "STRATEGY_REGISTRY",
    "get_strategy",
    "list_versions",
]
