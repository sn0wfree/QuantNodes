# coding=utf-8
"""V7_3 v2 完整还原 — 忠实于 source notebook (NowCasting之Factor Mimicking组合-高频宏观因子.ipynb).

[重大错误修正]
v7.3 v1 失败: OOS Calmar 0.010 vs source 1.626.
根因: 5 个差异 (见 docs/38):
  1. 数据池: 5 ETF → 13 INDICES (level-1, 沪深300/500/1000/恒生 + 4 中债 + 4 商品)
  2. 调仓窗口: 8 quarter = 2 年
  3. Bootstrap times: 500
  4. Bootstrap resample: 78-104 周
  5. Symmetry 时机: 窗口全样本 Symmetry (Klein 2013)

[忠实复刻 source cell 102 + 104]
- main_idx.resample('W').last().pct_change(1)    [w=13 indices, weekly]
- pd.concat([idx, factor]).dropna()             [dropna 严格]
- quarter_window = 8                            [2 years]
- bootstrap_lasso_mapping(times=500, resample=78-104)
- Symmetry(rolling_window)                      [窗口全样本白化]
- FactorRiskParity(beta.T[factor_cols], factor_cov, sum=[0.9, 1.0])
- max_weight=0.5
- 简单回测: np.dot(opt_weight, NV.T), cumprod()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .bootstrap_lasso import BootstrapLassoMapping
from .data_loader import INDEX_COLS, EXPANDED_COLS, EQUITY_ETF_COLS, COMMODITY_ETF_COLS, EXPANDED_BOND_INDICES
from .factor_risk_parity import FactorRiskParityOptimizer

# [Stage 6 TF 权益专属化] 债券指数常量 (与 INDEX_COLS 同级)
# 熊市时 freed weight 按比例分配到这些债券指数
BOND_INDICES = [
    '中债10年期国债指数', '中债3-5年期国债指数', '中债1-3年国债财富指数',
    '中债国开行债券总指数', '中债企业债总指数',
]


@dataclass
class V7_3Config:
    """v7.3 v2 配置 (忠实于 source)."""
    rebalance_freq: str = "QE"

    # 回看窗口 (source cell 102 length=8)
    quarter_window: int = 8

    # Bootstrap-Lasso (source cell 102)
    bootstrap_times: int = 500
    bootstrap_resample_min: int = 52 * 1 + 26   # 78 周 = 1.5 年
    bootstrap_resample_max: int = 52 * 2       # 104 周 = 2 年
    bootstrap_random_state: int = 42
    bootstrap_cache_alpha: bool = True

    # Symmetry (窗口内全样本, source cell 102)
    symmetry_min_periods: int = 12

    # FactorRiskParity (source cell 94)
    max_weight: float = 0.5
    sum_lower: float = 0.9
    sum_upper: float = 1.0
    rp_max_iter: int = 200
    rp_tol: float = 1e-8

    # 池 (源 cell 102)
    index_pool: tuple[str, ...] = tuple(INDEX_COLS)  # 13 indices
    factor_cols: tuple[str, ...] = (
        "宏观增长因子", "宏观通胀因子_生活端", "宏观通胀因子_生产端",
        "无风险收益率", "信用利差因子", "期限利差因子_债",
        "期限利差因子_股", "宏观汇率因子",  # 8 factors (source 不含加权)
    )

    # 成本
    commission_bp: float = 5.0
    slippage_bp: float = 5.0

    # [Stage 4 v2 新增 2026-07-13] 趋势过滤 (Trend Filter)
    # 当 benchmark 跌破 ma_window 日均线时, 缩放权益仓位到 exposure_bear
    # 释放权重按比例分配给债券 (flight to safety)
    # v7_macro_baseline 默认 False (不启用); v7_macro_baseline_v2_tf 启用
    trend_filter_enabled: bool = False
    trend_filter_benchmark: str = "沪深300指数"
    trend_filter_ma: int = 200
    trend_filter_bear: float = 0.5

    # [Stage 6 TF 权益专属化] 权益指数列表 (可自由添加)
    equity_indices: list = field(default_factory=lambda: [
        '沪深300指数', '中证500指数', '中证1000', '恒生指数',
    ])


@dataclass
class V7_4Config(V7_3Config):
    """v7.4 扩大资产池配置 (51 ETFs + 5 bond indices = 56 assets)."""
    asset_pool: str = "expanded"  # "index" | "expanded"

    # 资产分类 (expanded pool)
    index_pool: tuple[str, ...] = tuple(EXPANDED_COLS)  # 56 assets
    equity_cols: tuple[str, ...] = tuple(EQUITY_ETF_COLS)      # 45 equity ETFs
    commodity_cols: tuple[str, ...] = tuple(COMMODITY_ETF_COLS) # 6 commodity ETFs
    bond_cols: tuple[str, ...] = tuple(EXPANDED_BOND_INDICES)   # 5 bond indices


def symmetry_full_window(
    sample: pd.DataFrame,
    factor_cols: Sequence[str],
) -> pd.DataFrame:
    """对 sample 整个窗口做 Symmetry (仿 source cell 102, Klein 2013)."""
    if not isinstance(factor_cols, list):
        factor_cols = list(factor_cols)
    factor_cols = [c for c in factor_cols if c in sample.columns]
    if not factor_cols:
        return None
    F = sample[factor_cols].dropna()
    if len(F) < 12:
        return None
    cov = np.cov(F.values, rowvar=False)
    D, U = np.linalg.eigh(cov)
    D = np.maximum(D, 1e-8)
    S = U @ np.diag(D ** -0.5) @ U.T
    out = F.values @ S
    return pd.DataFrame(out, index=F.index, columns=F.columns)


def apply_trend_filter(
    w: pd.Series,
    benchmark_price: pd.Series,
    as_of: pd.Timestamp,
    cfg: V7_3Config,
) -> pd.Series:
    """[Stage 6 TF 权益专属化] 应用趋势过滤 (equity→bonds).

    熊市 (benchmark < ma_window 日均线): 只减权益仓位 × bear,
                                         释放权重按比例分配给债券 (flight to safety).
    多头: 返回 w 不变.

    Args:
        w: 当前 FRP 算出的权重 (index = INDEX_COLS)
        benchmark_price: benchmark 日价格 (pd.Series)
        as_of: 当前调仓日
        cfg: V7_3Config 配置 (含 trend_filter_* 字段)

    Returns:
        应用 TF 后的新权重 Series.
    """
    if not cfg.trend_filter_enabled:
        return w
    s = benchmark_price.loc[:as_of].dropna()
    if len(s) < cfg.trend_filter_ma:
        return w  # 数据不足, 默认多头
    ma = s.iloc[-cfg.trend_filter_ma:].mean()
    if s.iloc[-1] >= ma:
        return w  # 多头

    # 熊市: 只减权益, freed weight → 债券按比例分配
    bear = cfg.trend_filter_bear

    # 支持 expanded pool (V7_4Config) 和 index pool (V7_3Config)
    if isinstance(cfg, V7_4Config):
        eq_cols = list(cfg.equity_cols)
        bd_cols = list(cfg.bond_cols)
    else:
        eq_cols = list(cfg.equity_indices)
        bd_cols = BOND_INDICES

    equity_mask = w.index.isin(eq_cols)
    bond_mask = w.index.isin(bd_cols)

    # 计算释放的权益权重
    freed_weight = float((w[equity_mask] * (1.0 - bear)).sum())

    w_new = w.copy()
    w_new[equity_mask] = w_new[equity_mask] * bear  # 权益减仓

    # 释放权重按比例分配给债券
    bond_sum = float(w_new[bond_mask].sum())
    if bond_sum > 0:
        w_new[bond_mask] = w_new[bond_mask] + freed_weight * (w_new[bond_mask] / bond_sum)

    return w_new


class V7_3SubStrategy:
    """v7.3 v2 完整还原 — 季度调仓 (与 source 一致)."""

    def __init__(self, cfg: V7_3Config) -> None:
        self.cfg = cfg
        self.lasso = BootstrapLassoMapping(
            times=cfg.bootstrap_times,
            resample_min_weeks=cfg.bootstrap_resample_min,
            resample_max_weeks=cfg.bootstrap_resample_max,
            random_state=cfg.bootstrap_random_state,
            cache_alpha=cfg.bootstrap_cache_alpha,
        )
        self.rp = FactorRiskParityOptimizer(
            max_weight=cfg.max_weight,
            sum_lower=cfg.sum_lower,
            sum_upper=cfg.sum_upper,
        )

    def select(
        self,
        sample: pd.DataFrame,
        end_dt: pd.Timestamp,
    ) -> Mapping[str, float] | None:
        """计算 end_dt 时刻的 13 INDICES 权重 (faithful to source cell 102)."""
        cfg = self.cfg

        # 截至 end_dt 的所有数据
        s_so_far = sample.loc[:end_dt].dropna(how="all")
        if len(s_so_far) < 52 * 2 - 26:  # 至少 ~1.5 年数据
            return None

        # Quarter 滚动窗口 (source cell 102)
        end_loc = s_so_far.index.searchsorted(end_dt, side="left")
        if end_loc > len(s_so_far):
            return None
        if end_loc == len(s_so_far):
            end_loc -= 1

        # 取 end_dt 之前的 8 quarter
        quarter_last = pd.DataFrame(
            index=s_so_far.index[: end_loc + 1]
        ).resample(cfg.rebalance_freq).last().index

        if len(quarter_last) < cfg.quarter_window + 1:
            return None

        start_dt = quarter_last[-(cfg.quarter_window + 1)]
        rolling = s_so_far.loc[start_dt:end_dt].dropna(how="all")
        if len(rolling) < 52 * 2 - 26:
            return None

        # Symmetry 应用到 rolling 窗口 (source cell 102)
        sym_factors = symmetry_full_window(rolling, cfg.factor_cols)
        if sym_factors is None or len(sym_factors) < 26:
            return None

        # Concat 索引 + symmetried factor
        symmetried_sample = pd.concat(
            [rolling[list(cfg.index_pool)], sym_factors],
            axis=1,
        ).dropna(how="all")
        if len(symmetried_sample) < cfg.bootstrap_resample_min:
            return None

        # Bootstrap-Lasso
        β = self.lasso.estimate_exposure(
            asset_returns=symmetried_sample[list(cfg.index_pool)],
            factor_returns=symmetried_sample[list(cfg.factor_cols)],
            as_of_idx=len(symmetried_sample) - 1,
        )

        # 如果 β 全零, 等权兜底
        if np.abs(β.values).sum() < 1e-8:
            n = len(cfg.index_pool)
            return {col: 1.0 / n for col in cfg.index_pool}

        # 因子协方差
        factor_cov = symmetried_sample[list(cfg.factor_cols)].cov()

        # FactorRiskParity (source cell 104 factor_expo.T[factor_cols].fillna(0))
        w = self.rp.optimize(β, factor_cov)
        return {col: float(w.get(col, 0.0)) for col in cfg.index_pool}


def run_v7_3_backtest(
    index_panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    cfg: V7_3Config | None = None,
    benchmark_price: pd.Series | None = None,
    return_weights: bool = False,
):
    """v7.3 v2 端到端回测 (忠实于 source).

    Args:
        index_panel: 13 指数日对数收益
        factor_panel: 9 宏观因子周对数收益
        cfg: V7_3Config 配置 (含 trend_filter_* 字段)
        benchmark_price: [Stage 4 v2 新增] benchmark 日价格 (用于 TF).
                          默认 None 表示不加载; 当 cfg.trend_filter_enabled=True 时
                          必须传入 (推荐用 load_benchmark_price()).
        return_weights: [Stage 6 业绩对比新增] 是否同时返回调仓权重 DataFrame.
                        默认 False (向后兼容, 现有测试零影响).

    Returns:
        if return_weights=False: pd.Series 索引=业务日, 值=NAV (起点=1).
        if return_weights=True:  (nav: pd.Series, weights: pd.DataFrame)
                                  weights index=调仓日, columns=cfg.index_pool
    """
    if cfg is None:
        cfg = V7_3Config()

    # [Stage 4 v2 新增] 加载 benchmark 价格 (TF 需要)
    if cfg.trend_filter_enabled and benchmark_price is None:
        from .data_loader import load_benchmark_price
        benchmark_price = load_benchmark_price(cfg.trend_filter_benchmark)

    # Concat sample (source cell 61)
    idx_weekly = index_panel[list(cfg.index_pool)].resample("W").last().pct_change()
    factor_weekly = factor_panel[list(cfg.factor_cols)]
    sample = pd.concat(
        [idx_weekly, factor_weekly],
        axis=1,
    ).dropna(how="any")

    # Quarter 边界
    quarter_idx = pd.DataFrame(index=sample.index).resample(cfg.rebalance_freq).last().index
    quarter_idx = quarter_idx[quarter_idx <= sample.index.max()]

    if len(quarter_idx) <= cfg.quarter_window:
        raise ValueError(
            f"Insufficient data: need > {cfg.quarter_window} quarters, got {len(quarter_idx)}"
        )

    # 计算每个调仓日的权重
    sub = V7_3SubStrategy(cfg)
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    rebal_dates = list(quarter_idx[cfg.quarter_window:])

    for d_i in rebal_dates:
        w = sub.select(sample, d_i)
        if w is not None:
            w_series = pd.Series(w)
            # [Stage 4 v2 新增] 应用 TF
            if cfg.trend_filter_enabled and benchmark_price is not None:
                w_series = apply_trend_filter(w_series, benchmark_price, d_i, cfg)
            weights_history[d_i] = w_series

    if not weights_history:
        raise ValueError("No valid weights generated")

    # simple_backtest (source cell 105)
    weight_dates = sorted(weights_history.keys())
    cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000.0

    all_ret = []
    for s, e in zip(weight_dates[:-1], weight_dates[1:]):
        mask = (index_panel.index >= s) & (index_panel.index < e)
        if not mask.any():
            continue
        idx_ret_window = index_panel.loc[mask, list(cfg.index_pool)]
        # 收益率 (np.dot, source cell 93)
        # Fill NaN with 0 for expanded pool (structural NaN from pre-listing ETFs)
        ret_data = idx_ret_window.fillna(0).values @ weights_history[s].reindex(cfg.index_pool).fillna(0).values
        ret_series = pd.Series(ret_data, index=idx_ret_window.index)

        # 调仓日成本
        turnover = np.abs(
            weights_history[e].reindex(cfg.index_pool).fillna(0).values
            - weights_history[s].reindex(cfg.index_pool).fillna(0).values
        ).sum() / 2.0
        cost = turnover * cost_rate
        ret_series.iloc[0] -= cost
        all_ret.append(ret_series)

    if not all_ret:
        raise ValueError("No returns computed")

    all_ret_series = pd.concat(all_ret)
    nav = (1 + all_ret_series).cumprod()
    nav = nav / nav.iloc[0]

    if return_weights:
        weights_df = pd.DataFrame(
            {d: w.reindex(cfg.index_pool) for d, w in weights_history.items()},
        ).T
        weights_df.index.name = "rebalance_date"
        return nav, weights_df

    return nav


__all__ = ["V7_3Config", "V7_3SubStrategy", "run_v7_3_backtest", "symmetry_full_window"]
