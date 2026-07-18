# coding=utf-8
"""v4 多策略回测主入口 (Stage 17, v4.0).

3 个子策略 (可独立启用):
- 风格轮动 (style_rotation_v4)
- Smart β (smart_beta_v4)
- 因子择时 (factor_timing_v4) — IC 驱动

6 回测模式:
- v4A: 仅风格轮动
- v4B: 仅 Smart β
- v4C: 风格 + Smart β (无因子择时)
- v4D: + 因子择时 (IC only)
- v4E: + 因子择时 (HMM, 待实施)
- v4F: + 因子择时 (IC + HMM 融合, 待实施)

参考: reports/momentum_etf_rotation/v4/STAGE17_PLAN.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from .factor_timing_v4 import (
    FactorTimingConfig,
    compute_factor_weights,
    compute_strategy_weights,
    backtest_factor_timing,
)
from .smart_beta_v4 import SmartBetaConfig, SmartBetaSubStrategy
from .style_rotation_v4 import StyleRotationConfig, StyleRotationSubStrategy
from .sub_strategy_v4 import SubStrategy, SubStrategyConfig, SubStrategyResult
from .universe_v4 import ALL_V4_CODES, load_smartbeta_panel

logger = logging.getLogger(__name__)


class V4Mode(Enum):
    """v4 回测模式."""
    STYLE_ONLY = "v4A_style"
    SMART_BETA_ONLY = "v4B_smartbeta"
    COMBO_NO_TIMING = "v4C_combo"
    WITH_IC_TIMING = "v4D_ic"
    WITH_HMM_TIMING = "v4E_hmm"      # 待实施
    WITH_FUSION_TIMING = "v4F_fusion" # 待实施


@dataclass
class V4Config:
    """v4 多策略配置."""
    # 模式
    mode: str = "v4C_combo"  # V4Mode value

    # 子策略开关
    style_enabled: bool = True
    smart_beta_enabled: bool = True
    factor_timing_enabled: bool = False

    # 子策略配置
    style: StyleRotationConfig = field(default_factory=lambda: StyleRotationConfig(
        top_n_styles=3, top_n_per_style=1, max_weight=0.20,
    ))
    smart_beta: SmartBetaConfig = field(default_factory=lambda: SmartBetaConfig(
        top_n=3, max_weight=0.20,
    ))
    factor_timing: FactorTimingConfig = field(default_factory=FactorTimingConfig)

    # 持仓 cap (放宽到 0.20, 避免 cap 吸收差异)
    max_weight: float = 0.20

    # 调仓
    main_rebal_freq: str = "M"

    # 交易成本
    cost_bps: float = 5.0

    # 子策略初始权重 (无因子择时时用)
    style_weight: float = 0.5
    smart_beta_weight: float = 0.5

    # 预热期 (用等权)
    warmup_days: int = 252  # 1 年


@dataclass
class V4Result:
    """v4 回测结果."""
    nav: pd.Series
    states: list[dict] = field(default_factory=list)
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)
    sub_navs: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict[str, float] = field(default_factory=dict)
    mode: str = ""


def _get_rebal_dates(panel: pd.DataFrame, freq: str = "M") -> list[pd.Timestamp]:
    """取主调仓日 (默认月末)."""
    if freq == "M":
        return (
            pd.Series(panel.index)
            .groupby(panel.index.to_period("M"))
            .max()
            .tolist()
        )
    return panel.index.tolist()


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


def _combine_sub_results(
    sub_results: list[SubStrategyResult],
    sub_weights: dict[str, float],
) -> dict[str, float]:
    """合并子策略结果 (按 sub_weights 加权)."""
    combined: dict[str, float] = {}
    for r in sub_results:
        name = r.meta.get("strategy", "unknown")
        sub_w = sub_weights.get(name, 0.0)
        for code, w in r.weights.items():
            combined[code] = combined.get(code, 0.0) + sub_w * w
    # 归一化
    total = sum(combined.values())
    if total > 0:
        combined = {k: v / total for k, v in combined.items()}
    return combined


def _performance_metrics(nav: pd.Series) -> dict:
    """计算关键业绩指标."""
    n = len(nav)
    if n < 2:
        return {}
    ann_ret = nav.iloc[-1] ** (252 / n) - 1
    daily_ret = nav.pct_change().dropna()
    if len(daily_ret) < 2:
        return {"ann_return": float(ann_ret)}
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    max_dd = float((nav / nav.cummax() - 1).min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "final_nav": float(nav.iloc[-1]),
    }


def run_v4_backtest(
    panel: pd.DataFrame,
    cfg: V4Config | None = None,
    hmm_regime_series: pd.Series | None = None,
) -> V4Result:
    """v4 多策略回测主入口.

    Args:
        panel: 12 只 Smart β ETF 价格面板
        cfg: v4 配置
        hmm_regime_series: (可选) HMM 预测的 regime 时序, 用于 v4E/v4F 融合

    Returns:
        V4Result: NAV + 子策略 NAV + 状态
    """
    cfg = cfg or V4Config()
    cfg.mode = cfg.mode  # validate

    panel = panel.dropna(how="all")
    dates = panel.index
    rebal_dates = _get_rebal_dates(panel, cfg.main_rebal_freq)

    # 子策略初始化
    style_sub = StyleRotationSubStrategy(cfg.style) if cfg.style_enabled else None
    sb_sub = SmartBetaSubStrategy(cfg.smart_beta) if cfg.smart_beta_enabled else None

    # 因子择时 IC 历史 (如果启用)
    ic_history: pd.DataFrame = pd.DataFrame()
    if cfg.factor_timing_enabled:
        ic_history = backtest_factor_timing(
            panel, list(panel.columns), cfg.factor_timing,
        )

    # 子策略权重 (默认或动态)
    sub_weights: dict[str, float] = {}
    if cfg.style_enabled:
        sub_weights["style_rotation"] = cfg.style_weight
    if cfg.smart_beta_enabled:
        sub_weights["smart_beta"] = cfg.smart_beta_weight
    # 归一化
    total = sum(sub_weights.values())
    if total > 0:
        sub_weights = {k: v / total for k, v in sub_weights.items()}

    # 子策略 NAV 跟踪
    sub_navs: dict[str, pd.Series] = {"combined": pd.Series(dtype=float)}
    if cfg.style_enabled:
        sub_navs["style_rotation"] = pd.Series(dtype=float)
    if cfg.smart_beta_enabled:
        sub_navs["smart_beta"] = pd.Series(dtype=float)

    # 状态
    last_sub_results: dict[str, SubStrategyResult] = {}
    sub_nav_state: dict[str, tuple[dict[str, float], float]] = {}

    nav = np.ones(len(dates))
    states: list[dict] = []
    rebal_actual: list[pd.Timestamp] = []
    weights_combined: dict[str, float] = {}

    for i, date in enumerate(dates):
        if date in rebal_dates:
            # 1. 跑子策略
            sub_results: list[SubStrategyResult] = []
            if style_sub is not None:
                r = style_sub.run_step(panel, date)
                sub_results.append(r)
                last_sub_results["style_rotation"] = r
            if sb_sub is not None:
                r = sb_sub.run_step(panel, date)
                sub_results.append(r)
                last_sub_results["smart_beta"] = r

            # 2. 因子择时 (IC + HMM 融合)
            current_regime = -1
            if cfg.factor_timing_enabled and not ic_history.empty:
                idx = ic_history.index.get_indexer([date], method="ffill")[0]
                if idx >= 0:
                    ic_dict = ic_history.iloc[idx].to_dict()
                else:
                    ic_dict = {n: 0.0 for n in ic_history.columns}
                f_w = compute_factor_weights(
                    pd.DataFrame([ic_dict], index=[date]),
                    cfg.factor_timing,
                )

                # HMM regime 调整
                if hmm_regime_series is not None and date in hmm_regime_series.index:
                    current_regime = int(hmm_regime_series.loc[date])
                elif hmm_regime_series is not None and len(hmm_regime_series) > 0:
                    idx_r = hmm_regime_series.index.get_indexer([date], method="ffill")[0]
                    if idx_r >= 0:
                        current_regime = int(hmm_regime_series.iloc[idx_r])

                if current_regime >= 0:
                    # HMM 调整 IC 权重: 按 regime 偏好乘
                    from .regime_detector_v4 import get_regime_factor_weight
                    adjusted = {}
                    for f, w in f_w.items():
                        regime_w = get_regime_factor_weight(current_regime, f)
                        adjusted[f] = w * regime_w
                    # 重新归一化
                    total_adj = sum(adjusted.values())
                    if total_adj > 0:
                        adjusted = {k: v / total_adj for k, v in adjusted.items()}
                    f_w = adjusted

                s_w = compute_strategy_weights(
                    f_w, cfg.factor_timing.factor_to_strategy,
                )
                sub_weights = s_w if s_w else sub_weights
                # 归一化
                total = sum(sub_weights.values())
                if total > 0:
                    sub_weights = {k: v / total for k, v in sub_weights.items()}

            # 3. 合并
            combined = _combine_sub_results(sub_results, sub_weights)
            combined = _apply_max_weight(combined, cfg.max_weight)
            total = sum(combined.values())
            if total > 0:
                combined = {k: v / total for k, v in combined.items()}

            # 4. 调仓成本
            cost = 0.0
            if i > 0 and weights_combined:
                all_codes = set(weights_combined.keys()) | set(combined.keys())
                turnover = sum(
                    abs(combined.get(c, 0) - weights_combined.get(c, 0))
                    for c in all_codes
                ) / 2
                cost = turnover * cfg.cost_bps / 10000

            weights_combined = combined
            sub_nav_state["combined"] = (combined, 1.0)
            states.append({
                "date": date, "weights": combined,
                "sub_weights": sub_weights, "sub_results": sub_results,
                "regime": current_regime,
            })
            rebal_actual.append(date)

            if i > 0:
                nav[i] = nav[i - 1] * (1 - cost)
            else:
                nav[i] = 1.0
        else:
            # 非调仓日: 用前一日 NAV 累加
            if i > 0 and weights_combined:
                daily_ret = 0.0
                for code, w in weights_combined.items():
                    if code in panel.columns:
                        a, b = panel[code].iloc[i], panel[code].iloc[i - 1]
                        if not pd.isna(a) and not pd.isna(b) and b != 0:
                            daily_ret += w * (a / b - 1)
                nav[i] = nav[i - 1] * (1 + daily_ret)
            else:
                nav[i] = 1.0 if i == 0 else nav[i - 1]

        # 子策略 NAV 跟踪 (用 last_sub_results)
        for name, r in last_sub_results.items():
            if r.weights:
                ret = 0.0
                for code, w in r.weights.items():
                    if code in panel.columns:
                        if i > 0:
                            a, b = panel[code].iloc[i], panel[code].iloc[i - 1]
                            if not pd.isna(a) and not pd.isna(b) and b != 0:
                                ret += w * (a / b - 1)
                prev = sub_nav_state.get(name, ({}, 1.0))[1]
                if i == 0:
                    sub_navs[name] = pd.concat([sub_navs[name], pd.Series([1.0], index=[date])])
                else:
                    sub_navs[name] = pd.concat([sub_navs[name], pd.Series([prev * (1 + ret)], index=[date])])
                sub_nav_state[name] = (r.weights, sub_navs[name].iloc[-1])

    nav_series = pd.Series(nav, index=dates, name="v4")
    sub_nav_df = pd.DataFrame({k: v for k, v in sub_navs.items() if len(v) > 0})

    return V4Result(
        nav=nav_series,
        states=states,
        rebalance_dates=rebal_actual,
        sub_navs=sub_nav_df,
        metrics=_performance_metrics(nav_series),
        mode=cfg.mode,
    )


def run_v4_mode(
    panel: pd.DataFrame,
    mode: str,
    factor_timing_cfg: FactorTimingConfig | None = None,
    hmm_detector: "RegimeDetector | None" = None,
) -> V4Result:
    """按 mode 跑 v4 回测 (便捷接口).

    Args:
        panel: 价格面板
        mode: V4Mode value
        factor_timing_cfg: 因子择时配置
        hmm_detector: 已训练好的 HMM 检测器 (v4E/v4F 必需)
    """
    cfg = V4Config()
    cfg.mode = mode

    hmm_series: pd.Series | None = None

    if mode == "v4A_style":
        cfg.style_enabled = True
        cfg.smart_beta_enabled = False
    elif mode == "v4B_smartbeta":
        cfg.style_enabled = False
        cfg.smart_beta_enabled = True
    elif mode == "v4C_combo":
        cfg.style_enabled = True
        cfg.smart_beta_enabled = True
    elif mode == "v4D_ic":
        cfg.style_enabled = True
        cfg.smart_beta_enabled = True
        cfg.factor_timing_enabled = True
        cfg.factor_timing = factor_timing_cfg or FactorTimingConfig()
    elif mode == "v4E_hmm":
        # 仅 HMM 因子择时
        cfg.style_enabled = True
        cfg.smart_beta_enabled = True
        cfg.factor_timing_enabled = True  # 必须开启
        cfg.factor_timing = factor_timing_cfg or FactorTimingConfig()
        if hmm_detector is not None:
            hmm_series = hmm_detector.predict_series(
                panel, panel.index[0], panel.index[-1], step=5,
            )
        else:
            logger.warning("v4E_hmm 需要 hmm_detector, 退化为 v4D_ic")
    elif mode == "v4F_fusion":
        # IC + HMM 融合
        cfg.style_enabled = True
        cfg.smart_beta_enabled = True
        cfg.factor_timing_enabled = True
        cfg.factor_timing = factor_timing_cfg or FactorTimingConfig()
        if hmm_detector is not None:
            hmm_series = hmm_detector.predict_series(
                panel, panel.index[0], panel.index[-1], step=5,
            )
        else:
            logger.warning("v4F_fusion 需要 hmm_detector, 退化为 v4D_ic")
    else:
        raise ValueError(f"未知 mode: {mode}")

    return run_v4_backtest(panel, cfg, hmm_regime_series=hmm_series)


__all__ = [
    "V4Config",
    "V4Mode",
    "V4Result",
    "run_v4_backtest",
    "run_v4_mode",
]
