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
    load_daily_etf_returns,
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
    return_weights: bool = False,
    return_daily: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame] | tuple[pd.Series, pd.Series]:
    """v7.6 TV-PR 端到端回测.

    Args:
        X_panel: (T, N, K) 周频因子值面板 (None = 自动加载)
        Y: (T, N) 周频资产收益 (None = 自动加载)
        valid_codes: 有效资产代码列表
        cfg: v7.6 配置
        return_weights: 是否返回持仓权重
        return_daily: 是否返回日频 NAV (用周频权重 × 日频收益计算)

    Returns:
        nav: pd.Series, 周频 NAV
        weights_df: pd.DataFrame (如果 return_weights=True)
        daily_nav: pd.Series (如果 return_daily=True)
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

    # 3. 构造组合 (使用 β_t 预测收益, 必须返回权重)
    nav, weights_df = construct_portfolio(Y, X_panel, beta_path, cfg, return_weights=True)

    # 4. 计算日频 NAV (如果需要)
    if return_daily:
        daily_returns = load_daily_etf_returns()
        daily_nav = calculate_daily_nav(weights_df, daily_returns, cfg)
        return nav, daily_nav

    if return_weights:
        return nav, weights_df
    else:
        return nav


def construct_portfolio(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    beta_path: pd.DataFrame,
    cfg: V7_6Config,
    return_weights: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
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
        return_weights: 是否返回持仓权重

    Returns:
        nav: pd.Series, 周频 NAV
        weights_df: pd.DataFrame (如果 return_weights=True), columns=[date, code, weight]
    """
    T, N = Y.shape
    nav = pd.Series(1.0, index=Y.index, dtype=float)
    weights_history = []  # 存储持仓权重

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
        elif len(scores) > 0:
            chosen = scores.index.tolist()
        else:
            # 无有效资产，跳过本期
            nav.iloc[t] = nav.iloc[t - 1]
            continue

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

        # 存储持仓权重
        date = Y.index[t]
        for code in weights.index:
            weights_history.append({
                'date': date,
                'code': code,
                'weight': weights[code],
            })

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

    if return_weights:
        weights_df = pd.DataFrame(weights_history)
        return nav, weights_df
    else:
        return nav


# ============================================================
# 日频 NAV 计算
# ============================================================
def calculate_daily_nav(
    weights_df: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cfg: V7_6Config | None = None,
) -> pd.Series:
    """用周频持仓权重 × 日频收益, 计算日频 NAV.

    核心逻辑:
      1. 周频权重在 Sunday 计算 (信号日)
      2. 权重应用于该周的日收益 (周一~周五)
      3. 对于每个交易日, 找到对应的调仓日, 获取权重
      4. 用权重 × 日频收益, 累积得到日频 NAV

    Parameters:
        weights_df: (date, code, weight) 周频持仓权重
        daily_returns: (T_daily, N_etf) 日频 ETF 收益
        cfg: 配置 (用于交易成本)

    Returns:
        daily_nav: (T_daily,) 日频 NAV
    """
    cfg = cfg or V7_6Config()

    # 获取所有交易日和调仓日
    all_dates = daily_returns.index
    rebal_dates = sorted(weights_df["date"].unique())

    # 构建映射: 交易日 -> 对应的调仓日
    date_to_rebal = {}
    for idx, rebal_date in enumerate(rebal_dates):
        # 找到该调仓日对应的交易周结束日
        prev_dates = all_dates[all_dates <= rebal_date]
        if len(prev_dates) == 0:
            continue
        week_end = prev_dates[-1]

        # 找到该周的开始日 (上一个调仓日之后的第一个交易日)
        if idx > 0:
            prev_rebal = rebal_dates[idx - 1]
            next_day_idx = all_dates.searchsorted(prev_rebal)
            if next_day_idx < len(all_dates):
                week_start = all_dates[next_day_idx]
            else:
                continue
        else:
            # 第一个调仓日: 调仓日前一周的首个交易日
            week_start_idx = all_dates.searchsorted(rebal_date) - 5
            if week_start_idx < 0:
                week_start_idx = 0
            week_start = all_dates[week_start_idx]

        # 为该周的每个交易日分配权重
        week_mask = (all_dates >= week_start) & (all_dates <= week_end)
        for date in all_dates[week_mask]:
            date_to_rebal[date] = rebal_date

    # 计算日频 NAV
    daily_nav = pd.Series(1.0, index=all_dates, dtype=float)
    current_weights = {}

    for i in range(1, len(all_dates)):
        date = all_dates[i]

        # 更新权重 (如果该交易日有对应的调仓日)
        rebal_date = date_to_rebal.get(date)
        if rebal_date is not None:
            new_weights_df = weights_df[weights_df["date"] == rebal_date]
            current_weights = {
                str(k): v
                for k, v in new_weights_df.set_index("code")["weight"]
                .to_dict()
                .items()
            }

        # 计算日频组合收益
        daily_ret = 0.0
        for code, weight in current_weights.items():
            if code in daily_returns.columns:
                ret = daily_returns.loc[date, code]
                if pd.notna(ret):
                    daily_ret += weight * ret

        # 累积 NAV
        daily_nav.iloc[i] = daily_nav.iloc[i - 1] * (1 + daily_ret)

    return daily_nav


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
    "calculate_daily_nav",
    "v7_6_baseline",
    "v7_6_no_pv",
    "v7_6_with_stop_loss",
]
