# coding=utf-8
"""v7.6 宏观子策略: TV-PR (9 macro + 11 量价, 周频).

Cui et al. (2025) "Breaks and trends in factor premia."

v7.6 = v7.3 (9 macro) + v5 (11 量价) + TV-PR 时变 β_t

回测流程:
  1. 加载数据: 9 macro + 11 量价 → 周频
  2. 滚动估计: 用 TV-PR 估计 β_t
  3. 构造组合: 按 β_t 预测收益排序, 逆波动率加权
  4. 扣除成本: 5bp 佣金 + 5bp 滑点
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .data_loader_v7_6 import (
    load_weekly_macro_factors,
    load_weekly_pv_factors,
    load_weekly_asset_returns,
    build_mixed_factor_panel,
    load_v7_6_data,
)
from .tvpr_estimator import tvpr_estimator
from .data_loader import (
    EXPANDED_COLS,
    EQUITY_ETF_COLS,
    COMMODITY_ETF_COLS,
    EXPANDED_BOND_INDICES,
)
from ..v5.industry_factors import FactorEngineConfig


# ============================================================
# Config
# ============================================================
@dataclass
class V7_6Config:
    """v7.6 TV-PR 配置 (9 macro + 11 量价, 周频)."""
    name: str = "v7_6_tvpr"

    # 资产池
    asset_pool: str = "expanded"
    index_pool: tuple[str, ...] = tuple(EXPANDED_COLS)
    equity_cols: tuple[str, ...] = tuple(EQUITY_ETF_COLS)
    commodity_cols: tuple[str, ...] = tuple(COMMODITY_ETF_COLS)
    bond_cols: tuple[str, ...] = tuple(EXPANDED_BOND_INDICES)

    # 因子池 (8 macro + 11 量价 = 19 维)
    macro_cols: tuple[str, ...] = (
        "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
        "无风险收益率", "信用利差因子", "期限利差因子_债",
        "期限利差因子_股", "宏观汇率因子",
    )
    pv_factors: tuple[str, ...] = (
        "f1_second_mom", "f2_mom_term",
        "f3_amt_vol", "f4_vol_vol",
        "f5_turnover", "f6_ls_total", "f7_ls_change",
        "f8_pv_rankcov", "f9_pv_corr",
        "f10_first_div", "f11_vol_range",
    )

    # TV-PR 参数
    lambda_tv: float = 0.05
    lambda_l1: float = 0.01
    method: str = "admm"
    max_iter: int = 200
    tol: float = 1e-5

    # 调仓 (周频)
    rebalance_freq: str = "W"
    min_history: int = 52  # 周频 52 周 = 1 年
    window_size: int = 52  # 滚动窗口 52 周 = 1 年

    # 成本
    commission_bp: float = 5.0
    slippage_bp: float = 5.0
    cost_enabled: bool = True

    # 选股
    top_n: int = 10
    max_weight: float = 0.25
    vol_window: int = 26  # 周频 26 周 ≈ 半年
    vol_floor: float = 0.01


# ============================================================
# 回测主函数
# ============================================================
def run_v7_6_backtest(
    X_panel: np.ndarray | None = None,
    Y: pd.DataFrame | None = None,
    valid_codes: list[str] | None = None,
    cfg: V7_6Config | None = None,
) -> pd.Series:
    """v7.6 TV-PR 端到端回测.

    Args:
        X_panel: (T, N, K) 周频因子值面板 (None = 自动加载)
        Y: (T, N) 周频资产收益 (None = 自动加载)
        valid_codes: 有效资产代码列表
        cfg: v7.6 配置

    Returns:
        nav: pd.Series, 周频 NAV
    """
    cfg = cfg or V7_6Config()

    # 1. 加载数据
    if X_panel is None or Y is None:
        X_panel, Y, valid_codes = load_v7_6_data()

    T, N, K = X_panel.shape

    # 2. 滚动估计 β_t
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        method=cfg.method,
        min_history=cfg.min_history,
        window_size=cfg.window_size,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )

    # 3. 构造组合 (使用 β_t 预测收益)
    nav = construct_portfolio(Y, X_panel, beta_path, cfg)

    return nav


def construct_portfolio(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    beta_path: pd.DataFrame,
    cfg: V7_6Config,
) -> pd.Series:
    """根据 β_t 构造组合.

    逻辑:
      1. 计算预测收益: r_hat = X[t] @ beta_path[t-1]
      2. 按 r_hat 排序, 选 top_n ETF
      3. 逆波动率加权
      4. 扣除成本

    Args:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 周频因子值面板
        beta_path: (T, K) 时变 β_t
        cfg: 配置

    Returns:
        nav: pd.Series, 周频 NAV
    """
    T, N = Y.shape
    nav = pd.Series(1.0, index=Y.index, dtype=float)

    prev_weights = {}

    for t in range(1, T):
        # 1. 用 TV-PR 预测收益 (避免未来函数: 用 beta_path[t-1])
        beta_prev = beta_path.iloc[t - 1].values  # (K,) 上期估计的 β
        scores = X_panel[t] @ beta_prev  # (N,) 预测收益

        # 转为 Series 并过滤 NaN
        scores = pd.Series(scores, index=Y.columns)
        scores = scores.dropna()

        # 2. 选 top_n
        if len(scores) >= cfg.top_n:
            chosen = scores.nlargest(cfg.top_n).index.tolist()
        else:
            chosen = scores.index.tolist()

        # 3. 逆波动率加权
        if len(chosen) > 0 and t >= cfg.vol_window:
            # 计算波动率
            vol_window = Y.iloc[max(0, t - cfg.vol_window):t]
            vols = vol_window[chosen].std()
            vols = vols.fillna(cfg.vol_floor)  # NaN 用默认波动率填充
            vols = vols.clip(lower=cfg.vol_floor)

            # 逆波动率权重
            inv_vol = 1.0 / vols
            weights = inv_vol / inv_vol.sum()

            # 限制最大权重
            weights = weights.clip(upper=cfg.max_weight)
            weights = weights / weights.sum()
        else:
            # 等权
            weights = pd.Series(1.0 / len(chosen), index=chosen)

        # 4. 计算收益
        daily_ret = 0.0
        for code in chosen:
            if code in Y.columns:
                ret = Y[code].iloc[t]
                if pd.notna(ret):
                    daily_ret += weights.get(code, 0.0) * ret

        # 5. 交易成本
        if cfg.cost_enabled:
            turnover = 0.0
            for code in set(list(prev_weights.keys()) + list(weights.keys())):
                w_old = prev_weights.get(code, 0.0)
                w_new = weights.get(code, 0.0) if code in weights else 0.0
                turnover += abs(w_new - w_old)
            cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000
            daily_ret -= turnover * cost_rate

        nav.iloc[t] = nav.iloc[t - 1] * (1 + daily_ret)
        prev_weights = weights.to_dict()

    return nav


# ============================================================
# 工厂函数
# ============================================================
def v7_6_baseline(**overrides) -> V7_6Config:
    """v7.6 baseline: TV-PR (9 macro + 11 量价, 周频).

    预期性能:
      - OOS Calmar: 0.5-0.7 (估)
      - 起点 CV%: ≤25% (目标)
    """
    return V7_6Config(**overrides)


def v7_6_no_pv(**overrides) -> V7_6Config:
    """v7.6 变体: 只用 9 macro, 不用量价.

    用于对比 11 量价的增量贡献.
    """
    cfg = V7_6Config(**overrides)
    cfg.pv_factors = ()  # 清空量价因子
    return cfg


def v7_6_with_stop_loss(**overrides) -> V7_6Config:
    """v7.6 变体: 加硬止损.

    当 NAV 回撤 > 10% 时, 全仓债券.
    """
    cfg = V7_6Config(**overrides)
    # TODO: 实现止损逻辑
    return cfg


__all__ = [
    "V7_6Config",
    "run_v7_6_backtest",
    "construct_portfolio",
    "v7_6_baseline",
    "v7_6_no_pv",
    "v7_6_with_stop_loss",
]
