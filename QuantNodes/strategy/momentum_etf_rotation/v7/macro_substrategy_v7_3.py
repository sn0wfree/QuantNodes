# coding=utf-8
"""V7_3 独立宏观子策略 (Stage 30.4 完整版).

[流程]
1. Rolling Symmetry 正交 9 因子 → 滚动 52 周白化 (避免 look-ahead)
2. Bootstrap-Lasso (2000 次) → 资产对正交因子的暴露 β (5×9)
3. FactorRiskParity 优化 → 5 资产权重
4. 周频调仓 + 5bp 成本 (借鉴 v6 backtest)
5. 输出 NAV 时间序列

[ETF 池] (用户决策: 沪深300/中证500/创业板 + 恒生 + 国债)
    510300 沪深300 ETF
    510500 中证500 ETF
    159915 创业板 ETF (代替 512100 因为不在 44 池)
    510900 恒生 ETF (港股)
    511260 国债 ETF (利率)

[月度 vs 周度]
- 调仓: 周频 (W-FRI)
- 因子窗口: 52 周 (1 年滚动 Symmetry)
- Bootstrap 窗口: 104-156 周 (2-3 年滚动 Lasso)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .bootstrap_lasso import BootstrapLassoMapping
from .factor_risk_parity import FactorRiskParityOptimizer
from .symmetry import RollingSymmetry


@dataclass
class V7_3Config:
    """v7.3 完整版配置.

    Attributes:
        rebalance_freq: 调仓频率 ("W" 周频, 默认)
        factor_window: Symmetry 滚动窗口周数 (默认 52)
        factor_min_periods: Symmetry 最小可用期数 (默认 26)

        bootstrap_times: bootstrap 次数 (默认 200, 推荐 2000)
        bootstrap_min_weeks: bootstrap 最小重抽样周数 (默认 104)
        bootstrap_max_weeks: bootstrap 最大重抽样周数 (默认 156)

        max_weight: 单 ETF 权重上限 (默认 0.5)
        max_iter: FactorRiskParity 最大迭代 (默认 200)

        etf_pool: ETF 池 (默认 5 宽基)
        min_history_weeks: 启动前最小数据周数 (默认 156 = 3 年)

        commission_bp: 单边佣金 (默认 5 bp)
        slippage_bp: 单边滑点 (默认 10 bp)
    """
    rebalance_freq: str = "W-FRI"

    # Symmetry
    factor_window: int = 52
    factor_min_periods: int = 26

    # Bootstrap-Lasso
    bootstrap_times: int = 200
    bootstrap_min_weeks: int = 52 * 2
    bootstrap_max_weeks: int = 52 * 3
    bootstrap_random_state: int = 42

    # FactorRiskParity
    max_weight: float = 0.5
    rp_max_iter: int = 200
    rp_tol: float = 1e-8

    # ETF 池
    etf_pool: tuple[str, ...] = (
        "510300", "510500", "159915", "510900", "511260",
    )

    # 启动
    min_history_weeks: int = 52 * 3  # 3 年

    # 成本 (5bp 单边)
    commission_bp: float = 5.0
    slippage_bp: float = 10.0


class V7_3SubStrategy:
    """v7.3 独立宏观子策略.

    组合三大模块:
        - RollingSymmetry: 9 因子滚动白化
        - BootstrapLassoMapping: 2000 次稀疏回归估因子暴露
        - FactorRiskParityOptimizer: 因子空间风险平价
    """

    def __init__(self, cfg: V7_3Config) -> None:
        self.cfg = cfg
        self.symmetry = RollingSymmetry(
            window=cfg.factor_window,
            min_periods=cfg.factor_min_periods,
        )
        self.lasso = BootstrapLassoMapping(
            times=cfg.bootstrap_times,
            resample_min_weeks=cfg.bootstrap_min_weeks,
            resample_max_weeks=cfg.bootstrap_max_weeks,
            random_state=cfg.bootstrap_random_state,
        )
        self.rp = FactorRiskParityOptimizer(
            max_weight=cfg.max_weight,
            tol=cfg.rp_tol,
            max_iter=cfg.rp_max_iter,
        )

    def select(
        self,
        factor_navs: pd.DataFrame,    # (T_f, 9)
        asset_navs: pd.DataFrame,     # (T_a, 5)
        as_of: pd.Timestamp,
    ) -> Mapping[str, float] | None:
        """计算 as_of 时刻的 5 ETF 权重.

        Args:
            factor_navs: 9 宏观因子周频净值
            asset_navs: 5 ETF 日频净值
            as_of: 当前时刻 (用于切片)

        Returns:
            字典 {etf_code: weight}, 或 None (数据不足).
        """
        cfg = self.cfg

        # 1. 切片截至 as_of
        factor_navs = factor_navs.loc[:as_of].dropna(how="all")
        asset_navs = asset_navs.loc[:as_of].dropna(how="all")

        # 2. Symmetry 正交因子 (滚动 52 周, 已实现 no-leakage)
        factor_rets = factor_navs.pct_change().dropna(how="all")
        if len(factor_rets) < cfg.min_history_weeks:
            return None

        # 取截至 as_of 周数的 symmetric 因子 (rolling output 最后一行)
        # 因为 transform_panel 是 0/1/2... 的索引, 我们取最后一行
        sym_rets = self.symmetry.transform_panel(factor_rets)
        if sym_rets.iloc[-1].isna().any():
            return None

        sym_rets = sym_rets.dropna()

        # 3. 因子协方差 (最近 52 周)
        recent_window = sym_rets.iloc[-cfg.factor_window:]
        if len(recent_window) < cfg.factor_window // 2:
            return None
        Σf = recent_window.cov()

        # 4. Bootstrap-Lasso 因子暴露 β
        # 对齐资产和因子: 因子是 weekly (Sundays), 资产是 daily
        # 用 ffill 把因子正交收益扩展到日频
        asset_rets = (
            np.log(asset_navs / asset_navs.shift(1))
            .dropna(how="all")
        )
        sym_aligned_daily = sym_rets.reindex(asset_rets.index, method="ffill")
        # 丢掉开头仍为 NaN 的行
        valid_mask = sym_aligned_daily.notna().all(axis=1)
        asset_rets_a = asset_rets.loc[valid_mask]
        sym_aligned_daily = sym_aligned_daily.loc[valid_mask]

        if len(asset_rets_a) < cfg.bootstrap_min_weeks:
            return None

        as_of_idx = len(asset_rets_a) - 1
        β = self.lasso.estimate_exposure(
            asset_rets_a, sym_aligned_daily, as_of_idx,
        )

        # 如果 β 全零 (Lasso 太稀疏), 用等权兜底
        if np.abs(β.values).sum() < 1e-8:
            n = len(cfg.etf_pool)
            return {code: 1.0 / n for code in cfg.etf_pool}

        # 5. FactorRiskParity
        w = self.rp.optimize(β, Σf)

        # 6. 仅保留 etf_pool 中的代码
        return {code: float(w.get(code, 0.0)) for code in cfg.etf_pool}


def run_v7_3_backtest(
    factor_panel: pd.DataFrame,         # (T_f, 9) 周频宏观因子净值
    etf_panel: pd.DataFrame,            # (T_a, 5) ETF 日频净值
    cfg: V7_3Config | None = None,
    start_dt: pd.Timestamp | None = None,
) -> pd.Series:
    """v7.3 完整版端到端回测.

    返回值: pd.Series, 起点=1.0, 日频 NAV.

    调仓: 周五 (W-FRI). 成本: 5bp 佣金 + 5bp 滑点 (总 10bp).
    """
    if cfg is None:
        cfg = V7_3Config()

    sub = V7_3SubStrategy(cfg)

    # 周频调仓日 (W-FRI)
    end_dt = etf_panel.index.max()
    if start_dt is None:
        start_dt = etf_panel.index.min()
    rebal_start = start_dt + pd.Timedelta(days=cfg.min_history_weeks * 7)
    # 边界: rebal_start 应在 [start_dt, end_dt] 之间
    if rebal_start > end_dt:
        raise ValueError(
            f"Insufficient data: rebal_start={rebal_start.date()} > end_dt={end_dt.date()}"
        )
    week_dates = pd.date_range(
        max(rebal_start, start_dt),
        end_dt,
        freq=cfg.rebalance_freq,
    )

    # 每个 rebal 日计算权重
    weights_history: dict[pd.Timestamp, pd.Series] = {}
    for d in week_dates:
        w = sub.select(factor_panel, etf_panel, d)
        if w is not None:
            w_series = pd.Series(w)
            if w_series.sum() > 1e-8:
                weights_history[d] = w_series / w_series.sum()

    if not weights_history:
        raise ValueError("No valid weights generated — check data availability")

    weights_df = pd.DataFrame(weights_history).T
    weights_df = weights_df.reindex(etf_panel.index, method="ffill")
    # 起始处未填值, 用等权兜底
    weights_df = weights_df.fillna(1.0 / len(cfg.etf_pool))
    weights_df = weights_df.reindex(columns=list(cfg.etf_pool))

    # 调仓日持仓锁仓 (T+1), 调仓那天权重应用到下一天
    # 简化: 直接相乘 + 调仓日扣成本
    asset_rets = etf_panel[list(cfg.etf_pool)].pct_change().fillna(0.0)

    # 调仓日权重变化计算换手成本
    nav = pd.Series(index=etf_panel.index, dtype=float)
    nav.iloc[0] = 1.0

    # 单边成本率 (5bp + 10bp 滑点 * 1 折算)
    cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000.0

    rebal_dates = sorted(weights_history.keys())
    rebal_set = set(rebal_dates)

    for i in range(1, len(etf_panel)):
        d = etf_panel.index[i]
        prev_d = etf_panel.index[i - 1]
        w_curr = weights_df.loc[d]
        w_prev = weights_df.loc[prev_d]

        # 收益
        port_ret = (w_prev * asset_rets.loc[d]).sum()

        # 调仓日扣成本 (基于换手率)
        if d in rebal_set:
            turnover = np.abs(w_curr - w_prev).sum() / 2.0  # 单边换手
            cost = turnover * cost_rate
            port_ret -= cost

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret)

    # 截断起点
    nav = nav.dropna()
    if start_dt is not None:
        nav = nav.loc[start_dt:]
    return nav


__all__ = ["V7_3Config", "V7_3SubStrategy", "run_v7_3_backtest"]
