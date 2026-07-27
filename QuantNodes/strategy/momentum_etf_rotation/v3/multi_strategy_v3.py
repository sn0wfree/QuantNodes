# coding=utf-8
"""多策略主回测 (Stage 16A, v3.0).

主回测入口, 编排 3 个子策略:
- 动量策略 (v2 内置)
- 均值反转策略 (ReversionSubStrategy)
- 行业轮动策略 (IndustryRotationSubStrategy)

主回测职责:
1. 在调仓日 (默认月末) 调用各子策略
2. 收集子策略结果
3. 计算子策略权重 (等权/风险平价/信号加权)
4. 合并成最终权重
5. 应用 cap + max_weight 约束
6. NAV 计算 + 调仓成本扣减

子策略可独立调仓 (动量月度, 反转半月, 行业轮动周度)
但权重合并只在主调仓日 (月末) 进行

参考: reports/momentum_etf_rotation/v2/stage16a_plan.md §2.3
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..common.universe import ETFPool
from ..common.metrics import performance_metrics_legacy as performance_metrics
from .industry_rotation_v3 import (
    IndustryRotationConfig,
    IndustryRotationSubStrategy,
    get_industry_codes,
    get_rebalance_dates,
)
from .reversion_v3 import ReversionConfig, ReversionSubStrategy
from .sub_strategy_v3 import SubStrategy, SubStrategyConfig, SubStrategyResult
from .sub_weighting_v3 import (
    combine_sub_results,
    equal_sub_weights,
    risk_parity_sub_weights,
    signal_weighted_sub_weights,
)


@dataclass
class MultiStrategyConfig:
    """多策略组合配置 (Stage 16A)."""
    # 子策略配置
    momentum_enabled: bool = True
    reversion_enabled: bool = True
    industry_rotation_enabled: bool = True

    reversion: ReversionConfig = field(default_factory=ReversionConfig)
    industry_rotation: IndustryRotationConfig = field(default_factory=IndustryRotationConfig)

    # 子策略权重方法
    weight_method: str = "equal"  # "equal" | "risk_parity" | "signal"

    # 主调仓频率
    main_rebal_freq: str = "M"  # "M" / "W"

    # 持仓 cap
    a_share_total: int = 3       # A 股宽基+行业上限
    max_weight: float = 0.15     # 单只 ETF 权重上限

    # 交易成本
    cost_bps: float = 5.0        # 单边成本 (bp)

    # 模拟动量策略 (从 v2 复制简化版, 避免循环依赖)
    momentum_lookback: int = 144
    momentum_top_n: int = 10


@dataclass
class MultiStrategyResult:
    """多策略回测结果."""
    nav: pd.Series
    states: list[dict] = field(default_factory=list)
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    sub_navs: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, float] = field(default_factory=dict)


def _simple_momentum_select(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    pool: ETFPool,
    lookback: int = 144,
    top_n: int = 10,
) -> list[str]:
    """简化动量选股 (复用 v2 hybrid momentum 信号, 避免 v2→v3 循环依赖).

    Returns:
        list[str]: 选中的 ETF code 列表
    """
    sub = nav_df.loc[:as_of]
    if len(sub) < lookback + 1:
        return []

    # 简单动量: 144 日收益率
    ret = sub.iloc[-1] / sub.iloc[-lookback - 1] - 1.0
    ret = ret.dropna()

    # 排除不在 pool 的
    ret = ret[[c for c in ret.index if c in pool.codes]]

    # 排名
    ranked = ret.sort_values(ascending=False)

    # 简化 cap: A 股宽基+行业 ≤ 3
    chosen: list[str] = []
    a_count = 0
    for code in ranked.index:
        cat = pool.category_of(code)
        cat_name = cat.value
        if cat_name in ("a_broad", "a_sector"):
            if a_count >= 3:
                continue
            a_count += 1
        chosen.append(code)
        if len(chosen) >= top_n:
            break
    return chosen


def _simple_momentum_weight(
    nav_df: pd.DataFrame,
    codes: Sequence[str],
    as_of: pd.Timestamp,
) -> dict[str, float]:
    """简化动量加权: 逆波动."""
    if not codes:
        return {}
    sub = nav_df.loc[:as_of, [c for c in codes if c in nav_df.columns]]
    if len(sub) < 2:
        return {c: 1.0 / len(codes) for c in codes}

    log_ret = np.log(sub / sub.shift(1))
    vols = {}
    for c in log_ret.columns:
        valid = log_ret[c].dropna()
        vols[c] = valid.std() * np.sqrt(252) if len(valid) >= 2 else 1.0

    inv = {c: 1.0 / v if v > 0 else 0.0 for c, v in vols.items()}
    total = sum(inv.values())
    if total <= 0:
        return {c: 1.0 / len(codes) for c in codes}
    return {c: inv.get(c, 0.0) / total for c in codes}


def run_multi_strategy_backtest(
    etf_nav: pd.DataFrame,
    pool: ETFPool,
    cfg: MultiStrategyConfig | None = None,
) -> MultiStrategyResult:
    """多策略组合回测主入口.

    Args:
        etf_nav: 价格面板 (index=date, columns=code)
        pool: ETF 池
        cfg: 多策略配置

    Returns:
        MultiStrategyResult: NAV + 子策略 NAV + 状态
    """
    cfg = cfg or MultiStrategyConfig()
    cfg.rebalance_freq = cfg.main_rebal_freq

    etf = etf_nav.dropna(how="all")
    etf_norm = etf / etf.iloc[0]

    dates = etf.index

    # 主调仓日
    if cfg.main_rebal_freq == "M":
        rebal_dates = pd.Series(dates).groupby(
            dates.to_period("M")
        ).max().tolist()
    else:
        rebal_dates = dates.tolist()

    # 子策略初始化
    rev_sub = ReversionSubStrategy(cfg.reversion, pool) if cfg.reversion_enabled else None
    ind_sub = IndustryRotationSubStrategy(cfg.industry_rotation, pool) if cfg.industry_rotation_enabled else None

    # 模拟用占位 SubStrategy (动量)
    class _MomentumShim(SubStrategy):
        def select(self, nav_df, as_of):
            return _simple_momentum_select(
                nav_df, as_of, pool,
                lookback=cfg.momentum_lookback, top_n=cfg.momentum_top_n,
            )
        def weight(self, nav_df, codes, as_of):
            return _simple_momentum_weight(nav_df, codes, as_of)
        def run_step(self, nav_df, as_of):
            codes = self.select(nav_df, as_of)
            weights = self.weight(nav_df, codes, as_of)
            return SubStrategyResult(
                date=as_of, chosen=codes, weights=weights,
                meta={"strategy": "momentum"},
            )

    mom_shim = _MomentumShim(
        SubStrategyConfig(
            name="momentum", top_n=cfg.momentum_top_n, min_history=cfg.momentum_lookback,
        ),
        pool,
    ) if cfg.momentum_enabled else None

    # 子策略 NAV 跟踪
    sub_navs: dict[str, pd.Series] = {
        "combined": pd.Series(dtype=float),
    }
    if cfg.momentum_enabled:
        sub_navs["momentum"] = pd.Series(dtype=float)
    if cfg.reversion_enabled:
        sub_navs["reversion"] = pd.Series(dtype=float)
    if cfg.industry_rotation_enabled:
        sub_navs["industry_rotation"] = pd.Series(dtype=float)

    # 子策略最近一次结果 (用于非主调仓日复用)
    last_sub_results: dict[str, SubStrategyResult] = {}
    # 各子策略 NAV 状态
    sub_nav_state: dict[str, tuple[dict[str, float], float]] = {}

    nav = np.ones(len(dates))
    states: list[dict] = []
    rebal_actual: list[pd.Timestamp] = []

    for i, date in enumerate(dates):
        # 1. 调仓日: 运行所有子策略
        if date in rebal_dates:
            sub_results: list[SubStrategyResult] = []

            # 动量子策略 (主调仓日运行)
            if mom_shim is not None:
                r = mom_shim.run_step(etf_norm, date)
                sub_results.append(r)
                last_sub_results["momentum"] = r

            # 反转子策略 (每次主调仓都跑, 但反转的内部信号会变)
            if rev_sub is not None:
                r = rev_sub.run_step(etf_norm, date)
                sub_results.append(r)
                last_sub_results["reversion"] = r

            # 行业轮动子策略 (主调仓日, 实际可独立周度)
            if ind_sub is not None:
                r = ind_sub.run_step(etf_norm, date)
                sub_results.append(r)
                last_sub_results["industry_rotation"] = r

            # 2. 子策略权重
            if cfg.weight_method == "equal":
                sub_w = sub_weights_from_results_safe(sub_results, "equal")
            elif cfg.weight_method == "signal":
                sub_w = sub_weights_from_results_safe(sub_results, "signal")
            elif cfg.weight_method == "risk_parity":
                # 需要历史 sub_navs 来算协方差
                if sub_navs["combined"].shape[0] >= 30:
                    sub_w = risk_parity_sub_weights(
                        pd.DataFrame({
                            name: s for name, s in sub_navs.items()
                            if name != "combined"
                        }),
                        method="ledoit_wolf",
                    )
                else:
                    sub_w = sub_weights_from_results_safe(sub_results, "equal")
            else:
                sub_w = sub_weights_from_results_safe(sub_results, "equal")

            # 3. 合并
            combined = combine_sub_results(sub_results, sub_w, pool_codes=set(pool.codes))

            # 4. 应用 max_weight
            combined = _apply_max_weight(combined, cfg.max_weight)

            # 5. 归一化
            total = sum(combined.values())
            if total > 0:
                combined = {k: v / total for k, v in combined.items()}

            # 6. 调仓成本
            prev_total = sum(sub_nav_state.get("combined", ({}, 0.0))[0].values()) if "combined" in sub_nav_state else 0
            cost = 0.0
            if i > 0 and sub_nav_state.get("combined"):
                old_w = sub_nav_state["combined"][0]
                all_codes = set(old_w.keys()) | set(combined.keys())
                turnover = sum(
                    abs(combined.get(c, 0) - old_w.get(c, 0))
                    for c in all_codes
                ) / 2
                cost = turnover * cfg.cost_bps / 10000

            sub_nav_state["combined"] = (combined, 1.0)
            states.append({
                "date": date, "weights": combined,
                "sub_weights": sub_w, "sub_results": sub_results,
            })
            rebal_actual.append(date)

            if i > 0:
                nav[i] = nav[i - 1] * (1 - cost)
            else:
                nav[i] = 1.0
        else:
            # 非调仓日: 用前一日 NAV 累加
            if i > 0 and sub_nav_state.get("combined"):
                weights = sub_nav_state["combined"][0]
                daily_ret = 0.0
                for code, w in weights.items():
                    if code in etf_norm.columns:
                        col = etf_norm[code]
                        a = col.iloc[i] if hasattr(col, 'iloc') else col
                        b = col.iloc[i - 1] if hasattr(col, 'iloc') else col
                        if isinstance(a, pd.Series): a = a.iloc[0]
                        if isinstance(b, pd.Series): b = b.iloc[0]
                        if not pd.isna(a) and not pd.isna(b) and b != 0:
                            daily_ret += w * (a / b - 1)
                nav[i] = nav[i - 1] * (1 + daily_ret)
            else:
                nav[i] = 1.0 if i == 0 else nav[i - 1]

        # 跟踪子策略 NAV (用 last_sub_results)
        for name, r in last_sub_results.items():
            if r.weights:
                ret = 0.0
                for code, w in r.weights.items():
                    if code in etf_norm.columns:
                        col = etf_norm[code]
                        if i > 0:
                            a = col.iloc[i] if hasattr(col, 'iloc') else col
                            b = col.iloc[i - 1] if hasattr(col, 'iloc') else col
                            if isinstance(a, pd.Series): a = a.iloc[0]
                            if isinstance(b, pd.Series): b = b.iloc[0]
                            if not pd.isna(a) and not pd.isna(b) and b != 0:
                                ret += w * (a / b - 1)
                prev = sub_nav_state.get(name, ({}, 1.0))[1]
                if i == 0:
                    sub_navs[name] = pd.concat([sub_navs[name], pd.Series([1.0], index=[date])])
                else:
                    sub_navs[name] = pd.concat([sub_navs[name], pd.Series([prev * (1 + ret)], index=[date])])
                sub_nav_state[name] = (r.weights, sub_navs[name].iloc[-1])

    nav_series = pd.Series(nav, index=dates, name="multi_strategy")

    # 计算子策略 DataFrame
    sub_nav_df = pd.DataFrame({k: v for k, v in sub_navs.items() if len(v) > 0})

    return MultiStrategyResult(
        nav=nav_series,
        states=states,
        rebalance_dates=rebal_actual,
        sub_navs=sub_nav_df,
        metrics=performance_metrics(nav_series),
    )


def sub_weights_from_results_safe(
    sub_results: Sequence[SubStrategyResult],
    method: str,
) -> dict[str, float]:
    """从子策略结果构造权重的安全 wrapper."""
    if not sub_results:
        return {}
    names = [r.meta.get("strategy", f"strategy_{i}") for i, r in enumerate(sub_results)]
    if method == "signal":
        return signal_weighted_sub_weights(sub_results)
    return equal_sub_weights(names)


def _apply_max_weight(
    weights: dict[str, float],
    max_w: float,
) -> dict[str, float]:
    """约束单只 ETF 权重上限."""
    if not weights or max_w >= 1.0:
        return weights
    result = dict(weights)
    for _ in range(10):
        excess_total = 0.0
        for c, w in result.items():
            if w > max_w:
                excess_total += w - max_w
                result[c] = max_w
        if excess_total <= 1e-6:
            break
        non_capped = [c for c, w in result.items() if w < max_w]
        non_capped_sum = sum(result[c] for c in non_capped)
        if non_capped_sum > 0 and non_capped:
            for c in non_capped:
                result[c] += excess_total * (result[c] / non_capped_sum)
    return result


__all__ = [
    "MultiStrategyConfig",
    "MultiStrategyResult",
    "run_multi_strategy_backtest",
    "sub_weights_from_results_safe",
]
