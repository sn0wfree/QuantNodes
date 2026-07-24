# coding=utf-8
"""通用 Walk-Forward 回测框架 (修复 lookahead bias).

设计原则:
  1. 策略无关 — 只要提供 backtest_fn(Y_train, X_train, **params) → (weights, returns) 即可
  2. **NO LOOKAHEAD** — beta 估计只用训练数据 (Y_train = Y.iloc[:test_start])
  3. NAV 最后生成 — backtest_fn 返回 (weights_df, returns_series), 由框架累积 NAV
  4. 日期索引 — 所有遍历/切片都用 pd.Timestamp, 不用 int 索引
  5. 结构化输出 — WalkForwardResult 包含每个 walk 的 weights / returns / nav

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.walk_forward import (
        walk_forward, WalkForwardConfig, GridSearchSpace, generate_nav,
    )

    # 定义回测函数 (返回 weights 和 returns, 不用 NAV)
    def my_backtest(Y_train, X_train, **params):
        beta = expanding_window_tvpr(Y_train, X_train, ...)
        weights, returns = construct_portfolio(
            Y_train, X_train, beta, cfg, return_components=True,
        )
        return weights, returns

    # 跑 walk-forward
    result = walk_forward(
        Y=Y, X=X, backtest_fn=my_backtest, param_space=space,
         train_weeks=156, step=13,
    )
    print(result.summary())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any
import itertools

import numpy as np
import pandas as pd

from ..fi_plus import performance_metrics


# ============================================================
# 配置
# ============================================================
@dataclass
class GridSearchSpace:
    """参数网格空间.

    Examples:
        space = GridSearchSpace({
            'top_n': [5, 10, 15],
            'vol_window': [13, 26, 52],
            'max_weight': [0.15, 0.25, 0.35],
        })
    """
    param_grid: dict[str, list]

    @property
    def combinations(self) -> list[dict]:
        """返回所有参数组合."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    @property
    def n_combinations(self) -> int:
        return len(self.combinations)


@dataclass
class WalkForwardConfig:
    """Walk-Forward 配置.

    Parameters:
        train_weeks: 训练窗口长度 (周)
        step: 测试窗口 = 滚动步长 (周), step=13 表示每季度测试+滚动
        min_history: β 估计最少历史期数
        metric: 优化目标指标 ('sharpe', 'calmar', 'ann_return')
        fixed_params: 固定参数 (不参与搜索)
    """
    train_weeks: int = 156       # 3 年
    step: int = 13               # 测试窗口 = 滚动步长 (季度)
    min_history: int = 52        # 1 年预热
    metric: str = "sharpe"       # 优化目标
    fixed_params: dict = field(default_factory=dict)  # 固定参数
    n_jobs: int = 1              # 并行 jobs (1 = sequential, -1 = CPU 数)


# ============================================================
# 结果
# ============================================================
@dataclass
class WalkResult:
    """单个 walk 的结果."""
    walk_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    train_metric: float
    oos_shares: pd.DataFrame    # (T_test_daily, N) 日频份额 DataFrame
    oos_prices: pd.DataFrame    # (T_test_daily, N) 日频价格 DataFrame
    oos_weights: pd.DataFrame   # (T_test_native, N) 原频率权重 (周频/季频)
    oos_nav: pd.Series          # 由 (shares, prices) 累积的日频 NAV
    oos_metrics: dict


@dataclass
class WalkForwardResult:
    """Walk-Forward 完整结果."""
    version: str
    walks: list[WalkResult]
    oos_shares: list[pd.DataFrame]    # 每个 walk 的日频 shares
    oos_prices: list[pd.DataFrame]    # 每个 walk 的日频 prices
    oos_weights: list[pd.DataFrame]   # 每个 walk 的原频率 weights
    oos_nav: pd.Series                # 拼接后的 OOS NAV
    oos_metrics: dict                 # 拼接后的 OOS 指标
    param_space: GridSearchSpace
    config: WalkForwardConfig

    def summary(self) -> pd.DataFrame:
        """返回汇总 DataFrame."""
        rows = []
        for w in self.walks:
            rows.append({
                'walk': w.walk_id,
                'test': f"{w.test_start.date()}~{w.test_end.date()}",
                'params': str(w.best_params),
                'train': f"{w.train_metric:.3f}",
                'oos_sharpe': f"{w.oos_metrics['sharpe']:.3f}",
                'oos_calmar': f"{w.oos_metrics['calmar']:.3f}",
                'oos_dd': f"{w.oos_metrics['max_drawdown']:.2%}",
            })
        return pd.DataFrame(rows)

    def print_summary(self):
        """打印汇总."""
        m = self.oos_metrics
        print(f"\n{'='*60}")
        print(f"  {self.version} Walk-Forward OOS")
        print(f"  Walks: {len(self.walks)}, Param space: {self.param_space.n_combinations}")
        print(f"  Sharpe={m['sharpe']:.3f}, Calmar={m['calmar']:.3f}, "
              f"DD={m['max_drawdown']:.2%}")
        print(f"  AnnRet={m['ann_return']:.2%}, Vol={m['ann_vol']:.2%}")
        print(f"{'='*60}")
        print(self.summary().to_string(index=False))


# ============================================================
# NAV 生成 (由 weights × returns 计算 — 日频)
# ============================================================
def generate_nav_from_weights(
    weights: pd.DataFrame,
    daily_prices: pd.DataFrame,
    cost_bp: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """用权重 × 收益率计算 NAV (日频).

    公式:
      daily_returns = daily_prices.pct_change()
      NAV[t] = NAV[t-1] × (1 + Σ weights[i] × daily_returns[t, i] - cost)

    注意: daily_prices 需要包含测试窗口前一天的价格, 用于计算第一天的收益率.
    如果 daily_prices 不包含前一天, 第一天收益率为 0.

    Parameters:
        weights: (T_freq, N) 任意频率权重 (调仓日确定)
        daily_prices: (T_daily, N) 日频价格 (需包含测试窗口前一天)
        cost_bp: 交易成本 (bp), 默认 0

    Returns:
        nav: (T_daily,) 日频 NAV (起点=1.0)
        daily_returns: (T_daily,) 日频组合收益率
    """
    T_daily = len(daily_prices)
    if T_daily == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    codes = list(daily_prices.columns)

    # 1. 计算日频收益率
    daily_rets = daily_prices.pct_change()

    # 2. 预计算: 每个日频日期对应的调仓日 (或 None)
    weights_diff = weights.diff().abs().sum(axis=1)
    # 第一行 diff 是 NaN (sum(skipna=True) 返回 0), 强制设为权重绝对值和
    # 这样第一行非零权重时, 会被算作调仓日
    if len(weights) > 0:
        weights_diff.iloc[0] = weights.iloc[0].abs().sum()
    rebal_dates = weights.index[weights_diff > 1e-10]

    # 用 searchsorted 预计算映射: 每个 daily 日期 >= 哪个 rebal_date
    rebal_map = {}
    for rd in rebal_dates:
        pos = daily_prices.index.searchsorted(rd, side='left')
        if pos < T_daily:
            rebal_map[pos] = rd

    # 3. 初始化
    nav_vals = np.ones(T_daily)
    ret_vals = np.zeros(T_daily)
    current_weights = pd.Series(0.0, index=codes)
    prev_nav = 1.0

    # 4. 逐日计算
    for t in range(T_daily):
        # 检查是否是调仓日
        cost_factor = 1.0
        if t in rebal_map:
            rd = rebal_map[t]
            new_weights = weights.loc[rd].fillna(0.0).reindex(codes, fill_value=0.0)
            turnover = float((new_weights - current_weights).abs().sum())
            cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
            current_weights = new_weights

        # 当日组合收益 = Σ weights[i] × returns[i] (skipna)
        day_rets = daily_rets.iloc[t]
        valid_mask = day_rets.notna() & current_weights.notna()
        if valid_mask.any():
            daily_ret = float((current_weights[valid_mask] * day_rets[valid_mask]).sum())
        else:
            daily_ret = 0.0

        nav_vals[t] = prev_nav * (1 + daily_ret) * cost_factor
        ret_vals[t] = nav_vals[t] / prev_nav - 1 if prev_nav > 0 else 0.0
        prev_nav = nav_vals[t]

    nav = pd.Series(nav_vals, index=daily_prices.index, name='nav')
    daily_ret_series = pd.Series(ret_vals, index=daily_prices.index, name='daily_return')
    nav.index.name = 'date'
    daily_ret_series.index.name = 'date'
    return nav, daily_ret_series


# ============================================================
# NAV 生成 (由 shares 和 prices 累积 — 日频, legacy)
# ============================================================
def generate_nav_from_shares(
    shares: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bp: float = 0.0,
) -> pd.Series:
    """由份额（shares）和价格（prices）累积生成日频 NAV.

    公式 (按 NAV=1 基准 + 归一化):
      shares (按 NAV=1): shares[t] = weights[t] / prices[t] (调仓日)
                         shares[t] = shares[t-1] (非调仓日, forward-fill)

      portfolio_value (累积):
        portfolio_value[0] = 1.0
        if shares[t] > 0:
          portfolio_value[t] = Σ shares[t] × prices[t]   # mark-to-market
        else:
          portfolio_value[t] = portfolio_value[t-1]       # 空仓, 现金冻结

      调仓日扣成本 (每天检查, 只在交易时扣):
        turnover = Σ |Δshares × prices| / portfolio_value
        portfolio_value[t:] *= (1 - turnover × cost_bp / 10000)

      归一化:
        NAV = portfolio_value / portfolio_value[0]   # 起点 = 1.0

    Parameters:
        shares: (T, N) DataFrame, 调仓日确定的各资产份额
                非调仓日的 shares 等于上一次调仓的份额 (forward-fill)
        prices: (T, N) DataFrame, 各资产日频价格
        cost_bp: 交易成本 (bp), 默认 0

    Returns:
        nav: (T,) pd.Series, 日频 NAV (起点=1.0)
    """
    T = len(prices)
    if T == 0:
        return pd.Series(dtype=float)

    # 1. 对齐 shares 和 prices 索引
    shares = shares.reindex(prices.index).ffill().fillna(0.0)
    prices = prices.ffill().fillna(0.0)

    # 2. 计算 portfolio_value = Σ shares × prices
    portfolio_value = (shares * prices).sum(axis=1)

    # 3. 空仓期 forward-fill (shares=0 时 portfolio_value 保持不变)
    # 注意: 第一天的 shares 全 0 时, portfolio_value[0]=0, 需要初始化为 1.0
    portfolio_value = portfolio_value.where(portfolio_value > 1e-10).ffill().fillna(1.0)

    # 4. 调仓日按 turnover 扣成本 (每天都检查, 只在有交易时扣)
    if cost_bp > 0:
        shares_diff = shares.diff().fillna(0.0)
        rebal_mask = (shares_diff.abs().sum(axis=1) > 1e-10)
        # 用底层 numpy 数组修改, 避免 SettingWithCopyWarning
        pv = portfolio_value.values.copy()
        sd = shares_diff.values
        pr = prices.values
        for t_idx in range(T):
            if rebal_mask.iloc[t_idx] and pv[t_idx] > 1e-10:
                turnover_value = float(np.abs(sd[t_idx] * pr[t_idx]).sum())
                turnover_rate = turnover_value / pv[t_idx]
                cost_factor = max(1.0 - turnover_rate * cost_bp / 10000.0, 0.0)
                pv[t_idx:] = pv[t_idx:] * cost_factor
        portfolio_value = pd.Series(pv, index=portfolio_value.index)

    # 5. 归一化: NAV = portfolio_value / portfolio_value[0]
    if portfolio_value.iloc[0] > 0:
        nav = portfolio_value / portfolio_value.iloc[0]
    else:
        # 起点 portfolio_value[0] 异常, 用 1.0 兜底
        nav = portfolio_value.copy()
        nav[:] = 1.0

    return nav


# ============================================================
# 通用: DataFrame 保存/加载工具 (统一 index.name='date')
# ============================================================
def save_dataframe(df: pd.DataFrame, path) -> None:
    """统一保存 DataFrame (CSV).

    规则:
      - 自动设置 index.name='date' (若为 DatetimeIndex 且 name 为空)
      - CSV 第一列 = 'date' (不再出现 Unnamed: 0)

    Args:
        df: 待保存的 DataFrame
        path: 文件路径 (str 或 Path)
    """
    import pathlib
    path = pathlib.Path(path)
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and not df.index.name:
        df.index.name = 'date'
    df.to_csv(path)


def load_dataframe(path, parse_dates: list | None = None) -> pd.DataFrame:
    """统一加载 DataFrame (CSV).

    规则:
      - 自动 parse 日期列 ('date')
      - index_col='date'

    Args:
        path: 文件路径 (str 或 Path)
        parse_dates: 要 parse 的列名列表 (默认 ['date'])

    Returns:
        DataFrame, 索引 = DatetimeIndex (name='date')
    """
    import pathlib
    path = pathlib.Path(path)
    if parse_dates is None:
        parse_dates = ['date']
    df = pd.read_csv(path, parse_dates=parse_dates, index_col='date')
    return df


# ============================================================
# 通用: 把任意频率的 weights 转日频 shares (按 NAV=1 基准)
# ============================================================
def weights_to_daily_shares(
    weights: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> pd.DataFrame:
    """把任意频率的目标权重转为日频份额 (NAV=1 基准).

    公式:
      shares[t] = weights[t] / prices[t]   (调仓日)
      shares[t] = shares[t-1]              (非调仓日, forward-fill)

    调仓判定: weights 实际发生变化的日期 (diff.sum > eps).

    Parameters:
        weights: (T_freq, N) 任意频率权重 (T_freq = 周频/月频/季频)
        daily_prices: (T_daily, N) 日频价格

    Returns:
        shares: (T_daily, N) DataFrame, index=daily_prices.index
    """
    T_daily = len(daily_prices)
    if T_daily == 0:
        return pd.DataFrame(columns=weights.columns)
    codes = list(weights.columns)

    # 初始化 shares 全 0
    shares = pd.DataFrame(0.0, index=daily_prices.index, columns=codes)

    # 找到 weights 实际变化的索引 (调仓日)
    weights_diff = weights.diff().abs().sum(axis=1)
    rebal_mask = weights_diff > 1e-10
    rebal_dates = weights.index[rebal_mask]

    # 每个调仓日: shares = weights / prices (NAV=1 基准)
    for rebal_date in rebal_dates:
        future_dates = daily_prices.index[daily_prices.index >= rebal_date]
        if len(future_dates) == 0:
            continue
        daily_rebal_date = future_dates[0]

        target_w = weights.loc[rebal_date].fillna(0.0)
        prices_at_rebal = daily_prices.loc[daily_rebal_date]

        new_shares = pd.Series(0.0, index=codes)
        for code in codes:
            if target_w[code] > 0 and prices_at_rebal[code] > 0:
                new_shares[code] = target_w[code] / prices_at_rebal[code]

        # 从调仓日起 shares 锁定 (非调仓日 forward-fill 自动)
        for code in codes:
            shares.loc[daily_rebal_date:, code] = new_shares[code]

    return shares


def generate_nav(weights: pd.DataFrame, returns: pd.Series, cost_bp: float = 0.0) -> pd.Series:
    """由权重和收益率累积生成 NAV (兼容旧接口).

    公式: nav[t] = nav[t-1] * (1 + sum_i(weights[t, i] * returns[t, i]) - turnover * cost_bp/10000)
    其中 weights[t] 是 t 时执行的权重 (信号日在 t-1)

    Parameters:
        weights: (T, N) DataFrame, weights[t] 是 t 时执行的权重 (信号日在 t-1)
        returns: (T, N) DataFrame or Series, 资产收益
        cost_bp: 交易成本 (bp), 默认 0

    Returns:
        nav: (T,) pd.Series, 起点=1.0
    """
    T = len(weights)
    if T == 0:
        return pd.Series(dtype=float)
    nav_vals = [1.0]
    prev_w = pd.Series(0.0, index=weights.columns)
    for t in range(1, T):
        wt = weights.iloc[t].fillna(0.0)
        if isinstance(returns, pd.DataFrame):
            rt = returns.iloc[t].fillna(0.0)
        else:
            rt = pd.Series(returns.iloc[t], index=weights.columns).fillna(0.0)
        weekly_ret = float((wt * rt).sum())
        turnover = float((wt - prev_w).abs().sum())
        cost = turnover * cost_bp / 10000.0
        nav_vals.append(nav_vals[-1] * (1 + weekly_ret - cost))
        prev_w = wt
    return pd.Series(nav_vals, index=weights.index[:T])


def concat_components(components_list: list[pd.DataFrame | pd.Series]) -> pd.DataFrame:
    """拼接多个 walk 的 weights/returns.

    每个 walk 的起点为 1.0, 通过收益链接:
      nav[t] = nav[t-1] * (1 + returns[t])
    """
    if not components_list:
        return pd.DataFrame()
    out = []
    prev_nav = 1.0
    for comp in components_list:
        if isinstance(comp, pd.DataFrame):
            out.append(comp)
        else:
            out.append(comp.to_frame())
    return pd.concat(out, axis=0)


def concat_oos_nav(walk_navs: list[pd.Series]) -> pd.Series:
    """将多个 walk 的 OOS NAV 拼接成连续序列.

    每段 NAV 独立从 1.0 开始, 用收益率链接:
      nav[t] = nav[t-1] * (1 + ret[t])

    Parameters:
        walk_navs: 每个 walk 的 NAV Series (各自从 1.0 开始)

    Returns:
        拼接后的连续 NAV Series
    """
    if not walk_navs:
        return pd.Series(dtype=float)

    nav_vals = [1.0]
    nav_dates = []

    for seg in walk_navs:
        rets = seg.pct_change().fillna(0)
        for i in range(len(rets)):
            nav_vals.append(nav_vals[-1] * (1 + rets.iloc[i]))
            nav_dates.append(seg.index[i])

    return pd.Series(nav_vals[1:], index=nav_dates)


# ============================================================
# 参数搜索
# ============================================================
def grid_search(
    backtest_fn: Callable,
    Y_train: pd.DataFrame,
    X_train: np.ndarray,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    param_space: GridSearchSpace,
    fixed_params: dict | None = None,
    metric: str = "sharpe",
    cost_bp: float = 10.0,
) -> tuple[dict, float]:
    """网格搜索最佳参数 (NO LOOKAHEAD).

    Parameters:
        backtest_fn: 回测函数, 签名 (Y_train, X_train, **params) → (shares_df, prices_df, weights_df)
                     Y_train/X_train: **训练数据** (不含测试窗口)
        Y_train: 训练数据 (到 test_start)
        X_train: 训练因子面板 (到 test_start)
        train_start: 训练窗口起始日期
        train_end: 训练窗口结束日期 (test_start)
        param_space: 参数空间
        fixed_params: 固定参数
        metric: 优化目标
        cost_bp: 交易成本 (bp)

    Returns:
        (best_params, best_score)
    """
    fixed = fixed_params or {}
    best_score = -999.0
    best_params = {}

    for params in param_space.combinations:
        full_params = {**fixed, **params}
        try:
            out = backtest_fn(Y_train, X_train, **full_params)
            # 兼容 2-tuple 和 3-tuple 返回
            if len(out) == 3:
                shares, prices, _ = out
            else:
                shares, prices = out
            if shares is None or prices is None or len(shares) < 10:
                continue
            # 在训练窗口上评估
            mask = (shares.index >= train_start) & (shares.index <= train_end)
            s_train = shares.loc[mask]
            p_train = prices.loc[mask] if prices.index.equals(s_train.index) else prices.reindex(s_train.index)
            if len(s_train) < 5:
                continue
            nav = generate_nav_from_shares(s_train, p_train, cost_bp=cost_bp)
            m = performance_metrics(nav)
            score = m.get(metric, -999)
            if score > best_score:
                best_score = score
                best_params = params.copy()
        except Exception:
            continue

    return best_params, best_score


# ============================================================
# Walk-Forward 主函数 (修复 lookahead)
# ============================================================
def walk_forward(
    Y: pd.DataFrame,
    X: np.ndarray,
    backtest_fn: Callable,
    param_space: GridSearchSpace | None = None,
    config: WalkForwardConfig | None = None,
    version: str = "",
    cost_bp: float = 10.0,
) -> WalkForwardResult:
    """通用 Walk-Forward 回测 (NO LOOKAHEAD).

    修复要点:
      1. beta 估计只用 Y_train = Y.iloc[:test_start] (训练数据)
      2. backtest_fn 返回 (shares_df, prices_df, weights_df), 不再返回 nav
      3. NAV 在 walk_forward 框架内由 shares+prices 累积生成
      4. 索引用 pd.Timestamp, 不用 int

    流程:
      1. 对每个 walk:
         a. Y_train = Y.iloc[:test_start], X_train = X[:test_start]
         b. backtest_fn 在训练数据上估计, 返回测试窗口的 (shares, prices, weights)
         c. nav = generate_nav_from_shares(shares, prices) — 在框架内累积
      2. 拼接所有 walk 的 OOS NAV

    Parameters:
        Y: (T, N) 周频资产收益 (日期索引)
        X: (T, N, K) 因子面板 (日期索引对齐)
        backtest_fn: 签名 (Y_train, X_train, **params) → (shares_df, prices_df, weights_df)
                     Y_train/X_train: 训练数据 (不含测试窗口)
                     params: fixed_params + 搜索参数
                     返回: shares_df (T_daily, N) + prices_df (T_daily, N) + weights_df (T_native, N)
        param_space: 参数搜索空间 (None 表示用默认参数, 不做 grid_search)
        config: walk-forward 配置
        version: 版本标签 (用于输出)
        cost_bp: 交易成本 (bp), 默认 10bp (5bp commission + 5bp slippage)

    Returns:
        WalkForwardResult
    """
    cfg = config or WalkForwardConfig()
    use_grid_search = param_space is not None and param_space.n_combinations > 0
    walks = []
    walk_shares = []
    walk_prices = []
    walk_weights = []
    walk_navs = []

    # 用日期索引遍历 (不用 int 索引)
    test_start_idx = cfg.train_weeks
    test_start_date = Y.index[test_start_idx]
    walk_id = 0

    while test_start_idx + cfg.step <= len(Y):
        test_end_idx = test_start_idx + cfg.step
        test_end_date = Y.index[test_end_idx - 1]
        train_start_idx = max(0, test_start_idx - cfg.train_weeks)
        train_start_date = Y.index[train_start_idx]
        train_end_date = Y.index[test_start_idx - 1]
        walk_id += 1

        # **关键修复**: 训练数据只用 test_start 之前的数据 (NO LOOKAHEAD)
        Y_train = Y.iloc[:test_start_idx]
        X_train = X[:test_start_idx]

        # 参数搜索 (只用训练数据)
        if use_grid_search:
            best_params, train_score = grid_search(
                backtest_fn, Y_train, X_train,
                train_start_date, train_end_date,
                param_space, fixed_params=cfg.fixed_params, metric=cfg.metric,
                cost_bp=cost_bp,
            )
            full_params = {**cfg.fixed_params, **best_params}
        else:
            best_params = {}
            full_params = dict(cfg.fixed_params)
            train_score = float('nan')

        # 用最佳参数跑回测, 取测试窗口的份额和价格
        try:
            out = backtest_fn(Y_train, X_train, **full_params)
            if len(out) == 3:
                shares_all, prices_all, weights_all = out
            else:
                shares_all, prices_all = out
                weights_all = pd.DataFrame()
        except Exception as e:
            print(f"  Walk {walk_id} backtest failed: {e}")
            test_start_idx += cfg.step
            continue

        if shares_all is None or prices_all is None or len(shares_all) < 2:
            test_start_idx += cfg.step
            continue

        # 取测试窗口的份额和价格
        mask = (shares_all.index >= test_start_date) & (shares_all.index <= test_end_date)
        s_test = shares_all.loc[mask]
        p_test = prices_all.loc[mask] if prices_all.index.equals(shares_all.index) else prices_all.reindex(s_test.index)

        # 取测试窗口的 weights (用 test_start_date 之后的数据)
        if len(weights_all) > 0:
            w_mask = weights_all.index >= test_start_date
            w_test = weights_all.loc[w_mask]
        else:
            w_test = pd.DataFrame()

        if len(s_test) < 2:
            test_start_idx += cfg.step
            continue

        # 在框架内累积 NAV
        nav_test = generate_nav_from_shares(s_test, p_test, cost_bp=cost_bp)
        oos_metrics = performance_metrics(nav_test)

        walk = WalkResult(
            walk_id=walk_id,
            train_start=train_start_date,
            train_end=train_end_date,
            test_start=test_start_date,
            test_end=test_end_date,
            best_params=best_params,
            train_metric=train_score,
            oos_shares=s_test,
            oos_prices=p_test,
            oos_weights=w_test,
            oos_nav=nav_test,
            oos_metrics=oos_metrics,
        )
        walks.append(walk)
        walk_shares.append(s_test)
        walk_prices.append(p_test)
        walk_weights.append(w_test)
        walk_navs.append(nav_test)

        # 推进到下一个 walk
        test_start_idx += cfg.step
        if test_start_idx < len(Y):
            test_start_date = Y.index[test_start_idx]

    # 拼接 OOS NAV
    oos_nav = concat_oos_nav(walk_navs)
    oos_metrics = performance_metrics(oos_nav) if len(oos_nav) > 0 else {}

    return WalkForwardResult(
        version=version,
        walks=walks,
        oos_shares=walk_shares,
        oos_prices=walk_prices,
        oos_weights=walk_weights,
        oos_nav=oos_nav,
        oos_metrics=oos_metrics,
        param_space=param_space or GridSearchSpace({}),
        config=cfg,
    )


# ============================================================
# Rolling Walk-Forward + 真实生产模拟 (sequential)
# ============================================================
def walk_forward_rolling(
    Y: pd.DataFrame,
    X: np.ndarray,
    backtest_fn: Callable,
    config: WalkForwardConfig | None = None,
    version: str = "",
    cost_bp: float = 10.0,
    verbose: bool = True,
) -> list[dict]:
    """Rolling walk-forward + 真实生产模拟 (sequential).

    每个 walk 两阶段:
      1. 训练期 (in-sample): backtest_fn on rolling window [train_start, test_start)
      2. 部署期 (真实生产): backtest_fn on extended window [train_start, test_end)
         模型持续运行, 用截至 test_end 的累积数据重新估计参数

    Parameters:
        Y: (T, N) 周频资产收益 (日期索引)
        X: (T, N, K) 周频因子面板
        backtest_fn: (Y, X, **params) → (shares, prices, weights) 或 (shares, prices)
        config: WalkForwardConfig (train_weeks, step)
        version: 版本标签
        cost_bp: 交易成本 (bp)
        verbose: 是否打印进度

    Returns:
        walks: list[dict], 每个 walk 包含:
            - walk_id, train_start_idx, train_end_idx, test_start_idx, test_end_idx
            - train_nav, train_weights, train_shares, train_prices
            - test_nav, test_weights, test_shares, test_prices
    """
    cfg = config or WalkForwardConfig()
    walks = []

    test_start_idx = cfg.train_weeks
    walk_id = 0

    total_possible = max(0, (len(Y) - cfg.train_weeks) // cfg.step)
    if verbose:
        print(f"  Rolling walk-forward: train={cfg.train_weeks}, step={cfg.step}")
        print(f"  Estimated walks: ~{total_possible}")

    while test_start_idx + cfg.step <= len(Y):
        test_end_idx = test_start_idx + cfg.step
        train_start_idx = test_start_idx - cfg.train_weeks
        walk_id += 1

        # === 阶段 1: 训练期 (rolling window, in-sample) ===
        Y_train = Y.iloc[train_start_idx:test_start_idx]
        X_train = X[train_start_idx:test_start_idx]
        try:
            train_out = backtest_fn(Y_train, X_train, **cfg.fixed_params)
            if len(train_out) == 3:
                train_shares, train_prices, train_weights = train_out
            else:
                train_shares, train_prices = train_out
                train_weights = pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"  Walk {walk_id} train failed: {e}")
            test_start_idx += cfg.step
            continue

        if len(train_shares) < 2:
            test_start_idx += cfg.step
            continue

        # 修复: train_shares/prices 由 backtest_fn 返回时覆盖全部日频日期
        # (因为 construct_portfolio_components 加载了全部 daily_prices)
        # 必须 slice 到训练窗口, 否则 train_nav 与 test_nav 重叠
        train_end_date = Y_train.index[-1]
        train_daily_mask = train_prices.index <= train_end_date
        train_shares = train_shares.loc[train_daily_mask]
        train_prices = train_prices.loc[train_daily_mask]

        if len(train_shares) < 2:
            test_start_idx += cfg.step
            continue

        train_nav, train_daily_ret = generate_nav_from_weights(
            train_weights, train_prices, cost_bp=cost_bp,
        )

        # === 阶段 2: 部署期 (真实生产, 模型持续运行) ===
        Y_extended = Y.iloc[train_start_idx:test_end_idx]
        X_extended = X[train_start_idx:test_end_idx]
        try:
            ext_out = backtest_fn(Y_extended, X_extended, **cfg.fixed_params)
            if len(ext_out) == 3:
                ext_shares, ext_prices, ext_weights = ext_out
            else:
                ext_shares, ext_prices = ext_out
                ext_weights = pd.DataFrame()
        except Exception as e:
            if verbose:
                print(f"  Walk {walk_id} deploy failed: {e}")
            test_start_idx += cfg.step
            continue

        if len(ext_shares) < 2:
            test_start_idx += cfg.step
            continue

        # 修复: ext_shares 也覆盖全部日频日期, slice 到 [train_start, test_end]
        ext_start_date = Y_train.index[0]
        ext_end_date = Y.index[min(test_end_idx, len(Y) - 1)]
        ext_daily_mask = (ext_prices.index >= ext_start_date) & (ext_prices.index <= ext_end_date)
        ext_shares = ext_shares.loc[ext_daily_mask]
        ext_prices = ext_prices.loc[ext_daily_mask]
        if len(ext_weights) > 0:
            ext_weights = ext_weights.loc[(ext_weights.index >= ext_start_date) & (ext_weights.index <= ext_end_date)]

        if len(ext_shares) < 2:
            test_start_idx += cfg.step
            continue

        # 切出测试段 ([test_start_date, test_end_date))
        # Y 是周频 (周末日期, 周日为周末), daily 是工作日
        # weekly [a, b) → daily [first_daily >= a, last_daily < b)
        weekly_test_start = Y.index[test_start_idx]
        # test_end_idx 可能 == len(Y) (越界), 用最后一个 weekly 索引
        weekly_test_end = Y.index[min(test_end_idx, len(Y) - 1)]

        # 找 daily 中 >= weekly_test_start 的第一个日期
        future_dates = ext_shares.index[ext_shares.index >= weekly_test_start]
        # 找 daily 中 < weekly_test_end 的最后一个日期
        past_dates = ext_shares.index[ext_shares.index < weekly_test_end]

        if len(future_dates) == 0 or len(past_dates) == 0:
            test_start_idx += cfg.step
            continue
        daily_start = future_dates[0]
        daily_end = past_dates[-1]

        if daily_start > daily_end:
            test_start_idx += cfg.step
            continue

        # 包含测试窗口前一天 (用于 pct_change 计算第一天收益率)
        prev_day_mask = ext_prices.index < daily_start
        if prev_day_mask.any():
            prev_day = ext_prices.index[prev_day_mask][-1]
            test_prices_full = ext_prices.loc[prev_day:daily_end]
            test_shares_full = ext_shares.loc[prev_day:daily_end]
        else:
            test_prices_full = ext_prices.loc[daily_start:daily_end]
            test_shares_full = ext_shares.loc[daily_start:daily_end]

        # weights 仍按周频切 (用 weekly 索引)
        if len(ext_weights) > 0:
            w_mask = (ext_weights.index >= weekly_test_start) & (ext_weights.index < weekly_test_end)
            test_weights = ext_weights.loc[w_mask] if w_mask.any() else pd.DataFrame()
        else:
            test_weights = pd.DataFrame()

        # 用包含前一天的 prices 计算 NAV (generate_nav_from_weights 会正确处理)
        test_shares = test_shares_full
        test_prices = test_prices_full

        if len(test_shares) < 2:
            test_start_idx += cfg.step
            continue

        test_nav, test_daily_ret = generate_nav_from_weights(
            test_weights, test_prices, cost_bp=cost_bp,
        )

        # === 记录 walk 结果 ===
        walks.append({
            'walk_id': walk_id,
            'train_start_idx': train_start_idx,
            'train_end_idx': test_start_idx,
            'test_start_idx': test_start_idx,
            'test_end_idx': test_end_idx,
            'train_nav': train_nav,
            'train_daily_ret': train_daily_ret,
            'train_weights': train_weights,
            'train_shares': train_shares,
            'train_prices': train_prices,
            'test_nav': test_nav,
            'test_daily_ret': test_daily_ret,
            'test_weights': test_weights,
            'test_shares': test_shares,
            'test_prices': test_prices,
        })

        if verbose and (walk_id % 50 == 0 or walk_id == total_possible):
            print(f"  Walk {walk_id}/{total_possible}: "
                  f"train [{Y.index[train_start_idx].date()}~{Y.index[test_start_idx-1].date()}], "
                  f"test [{Y.index[test_start_idx].date()}~{Y.index[test_end_idx-1].date()}]")

        test_start_idx += cfg.step

    if verbose:
        print(f"  Total completed: {len(walks)} walks")

    return walks


def concatenate_full_picture(walks: list[dict]) -> tuple[pd.Series, pd.Series]:
    """第一段 train + 每段 rolling OOS, 复利链接.

    Walk 1 的 train NAV 作为基础 (in-sample, 起点 = 1.0),
    之后每个 walk 的 test OOS NAV 接续 (复利).

    Parameters:
        walks: walk_forward_rolling 返回的 list[dict]

    Returns:
        full_nav: pd.Series, 连续的日频 NAV
        full_daily_ret: pd.Series, 连续的日频组合收益率
    """
    if not walks:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    nav_segments = []
    ret_segments = []
    prev_nav = 1.0

    # Walk 1 的 train NAV + daily_ret
    first_train = walks[0].get('train_nav', pd.Series(dtype=float))
    first_train_ret = walks[0].get('train_daily_ret', pd.Series(dtype=float))
    if len(first_train) > 0:
        seg_nav = first_train / first_train.iloc[0] * prev_nav
        seg_nav.index.name = 'date'
        nav_segments.append(seg_nav)
        ret_segments.append(first_train_ret)
        prev_nav = float(seg_nav.iloc[-1])

    # 每个 walk 的 test OOS NAV + daily_ret (复利接续)
    for walk in walks:
        test_nav = walk.get('test_nav', pd.Series(dtype=float))
        test_ret = walk.get('test_daily_ret', pd.Series(dtype=float))
        if len(test_nav) == 0:
            continue
        seg_nav = test_nav / test_nav.iloc[0] * prev_nav
        seg_nav.index.name = 'date'
        nav_segments.append(seg_nav)
        ret_segments.append(test_ret)
        prev_nav = float(seg_nav.iloc[-1])

    if not nav_segments:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    full_nav = pd.concat(nav_segments)
    full_nav.index.name = 'date'

    full_daily_ret = pd.concat(ret_segments)
    full_daily_ret.index.name = 'date'

    return full_nav, full_daily_ret


__all__ = [
    "GridSearchSpace",
    "WalkForwardConfig",
    "WalkResult",
    "WalkForwardResult",
    "generate_nav",
    "generate_nav_from_shares",
    "weights_to_daily_shares",
    "save_dataframe",
    "load_dataframe",
    "concat_oos_nav",
    "grid_search",
    "walk_forward",
    "walk_forward_rolling",
    "concatenate_full_picture",
]