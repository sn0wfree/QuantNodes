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
from .data_loader import INDEX_COLS
from .factor_risk_parity import FactorRiskParityOptimizer


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
) -> pd.Series:
    """v7.3 v2 端到端回测 (忠实于 source)."""
    if cfg is None:
        cfg = V7_3Config()

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
            weights_history[d_i] = pd.Series(w)

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
        ret_data = idx_ret_window.values @ weights_history[s].reindex(cfg.index_pool).fillna(0).values
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
    return nav / nav.iloc[0]


__all__ = ["V7_3Config", "V7_3SubStrategy", "run_v7_3_backtest", "symmetry_full_window"]
