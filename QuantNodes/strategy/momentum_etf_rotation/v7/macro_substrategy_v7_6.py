# coding=utf-8
"""v7.6 宏观子策略: TV-PR (8 macro + 11 量价, 周频).

Cui et al. (2025) "Breaks and trends in factor premia."

v7.6 = v7.3 (8 macro) + v5 (11 量价) + TV-PR 时变 β_t
v7.10 = v7.9 (17 macro + 19 PV) + 混合标准化 + 两阶段 CV

回测流程:
  1. 加载数据: 8 macro + 11 量价 → 周频 (v7.6) 或 17 macro + 19 PV (v7.10)
  2. 滚动估计: 用 TV-PR 估计 β_t
  3. 构造组合: 按 β_t 预测收益排序, 逆波动率加权
  4. 扣除成本: 5bp 佣金 + 5bp 滑点
"""
from __future__ import annotations

from dataclasses import dataclass

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


# ============================================================
# Config
# ============================================================
@dataclass
class V7_6Config:
    """v7.6 TV-PR 配置 (8 macro + 11 量价, 周频). v7.10 扩展到 17 macro + 19 PV."""
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
    lambda_tv: float = 0.15
    lambda_l1: float = 0.05
    method: str = "admm"
    rho: float = 1.0
    max_iter: int = 200
    tol: float = 1e-5

    # 调仓 (周频)
    rebalance_freq: str = "W"
    min_history: int = 52  # 周频 52 周 = 1 年

    # 成本
    commission_bp: float = 5.0
    slippage_bp: float = 5.0
    cost_enabled: bool = True

    # 选股
    top_n: int = 10
    max_weight: float = 0.25
    vol_window: int = 26  # 周频 26 周 ≈ 半年
    vol_floor: float = 0.01

    # 硬止损 (v7.10 Stage 32)
    stop_loss_threshold: float | None = None  # e.g. -0.10 = 10% DD 触发
    stop_loss_cooldown: int = 5  # 止损后冷却周数 (避免震荡来回止损)

    # 趋势过滤 (v7.10 Stage 32)
    trend_filter_enabled: bool = False
    trend_filter_benchmark: str = "510300"  # 沪深300 ETF
    trend_filter_ma: int = 200  # 日频 MA 窗口
    trend_filter_bear: float = 0.5  # 熊市时权益仓位缩放比例


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

    # 2. 扩展窗口估计 β_t
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        method=cfg.method,
        min_history=cfg.min_history,
        rho=cfg.rho,
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
    monday_open_returns: pd.DataFrame | None = None,
    use_latest_beta: bool = False,
    benchmark_daily: pd.Series | None = None,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """根据 β_t 构造组合 (无未来函数版本).

    逻辑 (X[t] → Y[t+1]):
      1. 周五 t 收盘: 用 X[t] @ beta_path[t] 生成信号
      2. 周一 t+1 开盘: 执行 pending_weights (如果 monday_open_returns 提供)
      3. 周五 t+1 收盘: 赚取收益
      4. 逆波动率加权
      5. 扣除成本

    时间线 (周一开盘执行模式):
      周五 t 收盘   → 估计 beta_path[t], 生成信号 → pending_weights
      周一 t+1 开盘 → 执行 pending_weights
      周五 t+1 收盘 → 赚取 NAV_friday / NAV_monday_open - 1 (不含周末隔夜)

    时间线 (原始模式):
      周五 t 收盘   → 生成信号 (用 beta_path[t-1])
      周五 t+1 收盘 → 赚取 Y[t+1] (包含周末隔夜)

    Args:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 周频因子值面板
        beta_path: (T, K) 时变 β_t
        cfg: 配置
        return_weights: 是否返回持仓权重
        monday_open_returns: (T, N) 周一开盘到周五收盘收益 (不含周末隔夜).
            如果提供, 使用周一开盘执行模式.
        use_latest_beta: 是否用本周 β (True) 还是上期 β[t-1] (False, 默认).
        benchmark_daily: 日频基准价格 (用于趋势过滤). 如果 TF 启用但未提供, 自动从 load_daily_etf_returns() 加载.

    Returns:
        nav: pd.Series, 周频 NAV
        weights_df: pd.DataFrame (如果 return_weights=True), columns=[date, code, weight]
    """
    T, N = Y.shape
    nav = pd.Series(1.0, index=Y.index, dtype=float)
    weights_history = []  # 存储持仓权重

    prev_weights = {}
    pending_weights = None  # 待执行的权重 (来自上一周的信号)

    # 判断执行模式: 周一开盘 or 原始 (周五到周五)
    use_monday_open = monday_open_returns is not None

    # 止损状态
    _stop_loss_active = cfg.stop_loss_threshold is not None
    _stop_loss_triggered = False
    _stop_loss_cooldown_remaining = 0
    _peak_nav = 1.0

    # 趋势过滤状态
    _tf_active = cfg.trend_filter_enabled
    _tf_benchmark_prices = None
    if _tf_active:
        if benchmark_daily is not None:
            _tf_benchmark_prices = benchmark_daily.dropna()
        else:
            from .data_loader_v7_6 import load_daily_etf_returns
            _daily_rets = load_daily_etf_returns()
            if cfg.trend_filter_benchmark in _daily_rets.columns:
                _tf_benchmark_prices = (1 + _daily_rets[cfg.trend_filter_benchmark]).cumprod().dropna()

    for t in range(T):
        # 1. 先执行上周信号, 赚取本周收益
        #    pending_weights 在 t-1 时由 Block 2 生成, 本 iteration 执行
        if pending_weights is not None and len(pending_weights) > 0 and t > 0:
            weekly_ret = 0.0
            for code, w in pending_weights.items():
                if code in Y.columns:
                    if use_monday_open:
                        # 周一开盘执行: 赚取周一开盘到周五收盘收益 (不含周末隔夜)
                        ret = monday_open_returns[code].iloc[t]
                    else:
                        # 原始: 赚取周五到周五收益 (含周末隔夜)
                        ret = Y[code].iloc[t]
                    if pd.notna(ret):
                        weekly_ret += w * ret

            # 2. 交易成本
            if cfg.cost_enabled:
                turnover = 0.0
                for code in set(list(prev_weights.keys()) + list(pending_weights.keys())):
                    w_old = prev_weights.get(code, 0.0)
                    w_new = pending_weights.get(code, 0.0)
                    turnover += abs(w_new - w_old)
                cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000
                weekly_ret -= turnover * cost_rate

            nav.iloc[t] = nav.iloc[t - 1] * (1 + weekly_ret)
            prev_weights = pending_weights.to_dict()

            # 止损检查 (仅在有持仓时检查 DD, 触发止损)
            if _stop_loss_active:
                _peak_nav = max(_peak_nav, nav.iloc[t])
                drawdown = nav.iloc[t] / _peak_nav - 1.0
                if drawdown < cfg.stop_loss_threshold and not _stop_loss_triggered:
                    _stop_loss_triggered = True
                    _stop_loss_cooldown_remaining = cfg.stop_loss_cooldown

            # 记录实际持仓权重 (用于 daily_nav)
            for code, w in pending_weights.items():
                weights_history.append({
                    'date': Y.index[t],
                    'code': code,
                    'weight': w,
                })
        elif t > 0:
            nav.iloc[t] = nav.iloc[t - 1]

        # 止损冷却递减 (每期都执行, 包括现金期)
        if _stop_loss_active and _stop_loss_triggered and _stop_loss_cooldown_remaining > 0:
            _stop_loss_cooldown_remaining -= 1
            if _stop_loss_cooldown_remaining <= 0:
                _stop_loss_triggered = False
                _peak_nav = nav.iloc[t]  # 重置峰值, 从新起点计 DD

        # 3. 再生成下周信号 (用本周 X[t] 和 beta_path[t] 或 beta_path[t-1])
        #    生成的权重将在 t+1 时执行, 赚取 Y[t+1]
        if t >= 1:
            if use_latest_beta:
                beta_prev = beta_path.iloc[t].values  # (K,) 本周估计的 β (用 Y[:t])
            else:
                beta_prev = beta_path.iloc[t - 1].values  # (K,) 上期估计的 β
            
            # 处理 NaN: 对每个资产，只用非NaN的因子计算 scores
            N_assets = X_panel[t].shape[0]
            scores = np.full(N_assets, np.nan)
            for i in range(N_assets):
                x_i = X_panel[t, i, :]
                valid_mask = ~np.isnan(x_i) & ~np.isnan(beta_prev)
                if np.sum(valid_mask) > 0:
                    scores[i] = np.dot(x_i[valid_mask], beta_prev[valid_mask])

            # 转为 Series 并过滤 NaN
            scores = pd.Series(scores, index=Y.columns)
            scores = scores.dropna()

            # 4. 选 top_n
            if len(scores) >= cfg.top_n:
                chosen = scores.nlargest(cfg.top_n).index.tolist()
            elif len(scores) > 0:
                chosen = scores.index.tolist()
            else:
                chosen = []

            # 5. 逆波动率加权
            if len(chosen) > 0 and t >= cfg.vol_window:
                vol_window = Y.iloc[max(0, t - cfg.vol_window):t]
                vols = vol_window[chosen].std()
                vols = vols.fillna(cfg.vol_floor).clip(lower=cfg.vol_floor)
                inv_vol = 1.0 / vols
                weights = inv_vol / inv_vol.sum()
                weights = weights.clip(upper=cfg.max_weight)
                weights = weights / weights.sum()
            elif len(chosen) > 0:
                weights = pd.Series(1.0 / len(chosen), index=chosen)
            else:
                weights = pd.Series(dtype=float)

            # 趋势过滤: 熊市时缩放权重 (benchmark < MA200)
            if _tf_active and _tf_benchmark_prices is not None and len(weights) > 0:
                current_date = Y.index[t]
                bm = _tf_benchmark_prices.loc[:current_date].dropna()
                if len(bm) >= cfg.trend_filter_ma:
                    ma = bm.iloc[-cfg.trend_filter_ma:].mean()
                    if bm.iloc[-1] < ma:
                        weights = weights * cfg.trend_filter_bear

            # 存储待执行的权重 (将在 t+1 时执行)
            pending_weights = weights

            # 止损覆盖: 如果触发止损, 全仓现金 (0% 收益)
            if _stop_loss_active and _stop_loss_triggered:
                pending_weights = pd.Series(dtype=float)
            weights_history.append({
                'date': Y.index[t],
                'code': None,  # 标记: 信号生成日
                'weight': 0,
            })

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

    核心逻辑 (X[t] → Y[t+1], 无未来函数):
      1. 周频权重在周五(t)计算 (信号日)
      2. 权重应用于下一周的日收益 (周一(t+1)~周五(t+1))
      3. 对于每个交易日, 找到上一周的调仓日, 获取权重
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
    # 关键: 权重在周五(t)生成, 应用于下周一(t+1)~周五(t+1)
    date_to_rebal = {}
    for idx, rebal_date in enumerate(rebal_dates):
        # 找到该调仓日之后的下一周的交易日范围
        # 调仓日 = 周五(t), 应用日期 = 周一(t+1)~周五(t+1)
        if idx + 1 < len(rebal_dates):
            next_rebal = rebal_dates[idx + 1]
            # 下一周的交易日: 从调仓日后第一个交易日到下一个调仓日
            after_rebal = all_dates[all_dates > rebal_date]
            if len(after_rebal) == 0:
                continue
            week_start = after_rebal[0]
            # 找到下一个调仓日对应的交易日
            before_next = all_dates[all_dates <= next_rebal]
            if len(before_next) == 0:
                continue
            week_end = before_next[-1]
        else:
            # 最后一个调仓日: 应用到之后的所有交易日
            after_rebal = all_dates[all_dates > rebal_date]
            if len(after_rebal) == 0:
                continue
            week_start = after_rebal[0]
            week_end = all_dates[-1]

        # 为下一周的每个交易日分配权重
        week_mask = (all_dates >= week_start) & (all_dates <= week_end)
        for date in all_dates[week_mask]:
            date_to_rebal[date] = rebal_date

    # 计算日频 NAV
    daily_nav = pd.Series(1.0, index=all_dates, dtype=float)
    current_weights = {}
    cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000 if cfg.cost_enabled else 0.0

    for i in range(1, len(all_dates)):
        date = all_dates[i]

        # 更新权重 (如果该交易日有对应的调仓日)
        rebal_date = date_to_rebal.get(date)
        if rebal_date is not None:
            new_weights_df = weights_df[weights_df["date"] == rebal_date]
            # 过滤掉 NaN 行 (信号生成标记)
            new_weights_df = new_weights_df[new_weights_df["code"].notna()]
            new_weights = {
                str(k): v
                for k, v in new_weights_df.set_index("code")["weight"]
                .to_dict()
                .items()
            }

            # 计算换手率 (调仓日扣减交易成本)
            if cfg.cost_enabled:
                turnover = 0.0
                all_codes = set(list(current_weights.keys()) + list(new_weights.keys()))
                for code in all_codes:
                    w_old = current_weights.get(code, 0.0)
                    w_new = new_weights.get(code, 0.0)
                    turnover += abs(w_new - w_old)

            current_weights = new_weights

        # 计算日频组合收益
        daily_ret = 0.0
        for code, weight in current_weights.items():
            if code in daily_returns.columns:
                ret = daily_returns.loc[date, code]
                if pd.notna(ret):
                    daily_ret += weight * ret

        # 调仓日扣减交易成本 (在收益计算之后, NAV 更新之前)
        if rebal_date is not None and cfg.cost_enabled:
            daily_ret -= turnover * cost_rate

        # 累积 NAV
        daily_nav.iloc[i] = daily_nav.iloc[i - 1] * (1 + daily_ret)

    return daily_nav


# ============================================================
# 工厂函数
# ============================================================
def v7_6_baseline(**overrides) -> V7_6Config:
    """v7.6 baseline: TV-PR (8 macro + 11 量价, 周频).

    预期性能:
      - OOS Calmar: 0.5-0.7 (估)
      - 起点 CV%: ≤25% (目标)
    """
    return V7_6Config(**overrides)


def v7_6_no_pv(**overrides) -> V7_6Config:
    """v7.6 变体: 只用 8 macro, 不用量价.

    用于对比 11 量价的增量贡献.
    """
    cfg = V7_6Config(**overrides)
    cfg.pv_factors = ()  # 清空量价因子
    return cfg


def v7_6_with_stop_loss(**overrides) -> V7_6Config:
    """v7.6 变体: 加硬止损.

    当 NAV 回撤 > 10% 时, 全仓现金 (0% 收益, 等效 "flight to safety").
    止损后冷却 5 周, 避免震荡市来回止损.

    设计动机: "金融预测存在极限, 这条止损线是所有逻辑假设失效时的最后实盘生存保障"
    """
    cfg = V7_6Config(**overrides)
    cfg.stop_loss_threshold = -0.10
    cfg.stop_loss_cooldown = 5
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
