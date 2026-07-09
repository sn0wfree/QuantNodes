# coding=utf-8
"""v4 因子择时 (Stage 18 合并版) — IC 驱动 + 多改进.

核心思想 (金融街证券《风格轮动与因子择时》):
- 把因子作为"可交易品种"
- 用滚动 IC 监控哪个因子最近有效
- 给子策略动态分配权重: IC 高的因子权重大

Stage 18 升级 (基于 v4 诊断 SUB_STRATEGY_DIAGNOSTIC.md §2):
1. **因子特异性 forward_window** (诊断: momentum 120d, value 40d, reversal 60d)
2. **因子特异性 lag 平滑** (诊断: 5 因子 lag1=0.48-0.69 高持续, reversal 不用)
3. **Regime-conditioned 因子选择** (诊断: bull 仅 m+v, bear v+d+q, sideways 仅 v)
4. **删除 low_vol 因子** (诊断: IC vs forward 相关 -0.454, 反指因子)
5. **IC 质量过滤** (诊断: |IC|<0.05 视为噪声, 84-94% 频率)

向后兼容:
- `forward_window` 字段保留 (默认 None = 走因子特异 FW)
- `use_low_vol: bool = True` 字段保留 (默认 False, 走 v5 优化)
- `factor_to_strategy` 字段保留 (默认 None = 走因子→ETF 直接映射)

权重公式:
    raw_weight[name] = max(0, IC[name] + base) ** power
    weight[name] = raw_weight / sum(raw_weight)

参数:
- factor_fw: 因子→前向窗口 (默认 5 因子特异)
- factor_smooth_window: 因子→lag 平滑窗口
- factor_ic_threshold: |IC| 阈值
- regime_factors: regime→可用因子
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .factor_ic import (
    FACTOR_NAMES,
    factor_ic_at,
    factor_ic_rolling_mean,
    rolling_factor_ic,
)
from .universe_v4 import ALL_V4_CODES, SMART_BETA_CODES, STYLE_GROUP_CODES


@dataclass
class FactorTimingConfig:
    """v4 因子择时配置 (Stage 18 升级版).

    Stage 18 新增字段 (defaults = v5 优化值):
    - factor_fw: 因子→前向窗口
    - factor_smooth_window: 因子→lag 平滑
    - factor_ic_threshold: IC 质量阈值
    - regime_factors: regime→可用因子
    - factor_to_etf: 因子→具体 ETF (替代 factor_to_strategy)
    - use_low_vol: 是否启用 low_vol (默认 False)

    向后兼容字段:
    - forward_window: 统一前向窗口 (默认 None = 走 factor_fw)
    - factor_to_strategy: 因子→子策略名 (默认 None = 走 factor_to_etf)
    """
    # 通用
    ic_step: int = 5
    lookback: int = 60
    base: float = 0.05
    power: float = 2.0
    min_weight: float = 0.10
    max_weight: float = 0.50
    warmup_period: int = 120

    # Stage 18 #1: 因子特异性 forward_window
    forward_window: int | None = None
    factor_fw: dict[str, int] = field(default_factory=lambda: {
        "momentum": 120,   # 诊断: 最优 120d
        "reversal": 60,    # 诊断: 最优 60d
        "value":    40,    # 诊断: 最优 40d (峰值)
        "dividend": 180,   # 诊断: 最优 180d
        "quality":  252,   # 诊断: 最优 252d
    })

    # Stage 18 #2: 因子特异性 lag 平滑
    smooth_window: int = 12
    factor_smooth_window: dict[str, int] = field(default_factory=lambda: {
        "momentum": 4,     # 诊断: lag1=+0.59, 用 4w 平滑
        "value":    4,     # 诊断: lag1=+0.58
        "dividend": 4,     # 诊断: lag1=+0.48
        "quality":  4,     # 诊断: lag1=+0.53
        "reversal": 1,     # 诊断: lag1=-0.01, 不平滑
    })

    # Stage 18 #5: IC 质量阈值
    factor_ic_threshold: float = 0.05

    # Stage 18 #4: low_vol 开关 (默认 False = 删除反指因子)
    use_low_vol: bool = False

    # Stage 18 #3: Regime-conditioned 因子选择
    regime_factors: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "bull":     ("momentum", "value"),
        "bear":     ("value", "dividend", "quality"),
        "sideways": ("value",),
    })

    # 因子 → ETF 直接映射 (Stage 18 新增, 替代 factor_to_strategy)
    factor_to_etf: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "momentum": ("510300", "510500", "159915", "588000", "510880"),
        "reversal": ("510300", "510500", "159915", "588000", "510880"),
        "value":    ("512040",),
        "dividend": ("510880", "512890", "515080", "515100"),
        "quality":  ("515900",),
    })

    # 向后兼容: 因子→子策略名映射
    factor_to_strategy: dict[str, str] = field(default_factory=lambda: {
        "momentum": "style_rotation",
        "reversal": "style_rotation",
        "value":    "smart_beta",
        "low_vol":  "smart_beta",
        "dividend": "smart_beta",
        "quality":  "smart_beta",
    })


def get_active_factors(cfg: FactorTimingConfig) -> tuple[str, ...]:
    """当前活跃因子 (考虑 use_low_vol)."""
    if cfg.use_low_vol:
        return FACTOR_NAMES
    return tuple(f for f in FACTOR_NAMES if f != "low_vol")


def compute_factor_weights(
    ic_history: pd.DataFrame,
    cfg: FactorTimingConfig | None = None,
    regime: str = "sideways",
) -> dict[str, float]:
    """根据最新一期 IC 计算因子权重 (Stage 18 升级).

    算法:
        1. 选可用因子 (regime 过滤 + use_low_vol)
        2. |IC| < threshold → weight = 0 (Stage 18 #5)
        3. raw_w[name] = max(0, IC[name] + cfg.base) ** cfg.power
        4. 应用 min_weight / max_weight
        5. 归一化

    Args:
        ic_history: 滚动 IC DataFrame (index=date, columns=factor)
        cfg: 配置
        regime: "bull" | "bear" | "sideways"

    Returns:
        dict, factor name → weight (sum=1, 或 sum<1 表示有 cash)
    """
    cfg = cfg or FactorTimingConfig()
    active = get_active_factors(cfg)
    available = cfg.regime_factors.get(regime, active)

    if ic_history.empty:
        return {n: 1.0 / len(available) for n in available}

    latest = ic_history.iloc[-1].fillna(0.0)

    raw: dict[str, float] = {}
    for name in available:
        if name not in latest.index:
            continue
        ic = float(latest.get(name, 0.0))
        if abs(ic) < cfg.factor_ic_threshold:
            continue
        raw[name] = max(0.0, ic + cfg.base) ** cfg.power

    if not raw:
        return {n: 0.0 for n in available}

    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total for k, v in raw.items()}

    for name in raw:
        if raw[name] < cfg.min_weight:
            raw[name] = cfg.min_weight
        if raw[name] > cfg.max_weight:
            raw[name] = cfg.max_weight

    total = sum(raw.values())
    if total > 0:
        raw = {k: v / total for k, v in raw.items()}

    return raw


def compute_strategy_weights(
    factor_weights: dict[str, float],
    factor_to_strategy: dict[str, str] | None = None,
) -> dict[str, float]:
    """聚合因子权重到子策略权重 (向后兼容).

    多个因子映射到同一子策略 → 权重累加.

    Args:
        factor_weights: factor → weight
        factor_to_strategy: factor → strategy name (默认 FactorTimingConfig 默认值)

    Returns:
        strategy name → weight (sum=1)
    """
    f2s = factor_to_strategy or FactorTimingConfig().factor_to_strategy
    out: dict[str, float] = {}
    for f, w in factor_weights.items():
        s = f2s.get(f, "unknown")
        out[s] = out.get(s, 0.0) + w

    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def aggregate_factor_to_etf(
    factor_weights: dict[str, float],
    cfg: FactorTimingConfig,
) -> dict[str, float]:
    """因子权重 → ETF 权重 (因子内等权, Stage 18 #1 主用法).

    多个因子同时选中同一只 ETF → 权重累加.
    """
    out: dict[str, float] = {}
    for fac, w in factor_weights.items():
        if w <= 0:
            continue
        codes = cfg.factor_to_etf.get(fac, ())
        if not codes:
            continue
        per_etf = w / len(codes)
        for c in codes:
            out[c] = out.get(c, 0.0) + per_etf

    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def backtest_factor_timing(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str] = ALL_V4_CODES,
    cfg: FactorTimingConfig | None = None,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """回测因子择时的 IC 时序 (Stage 18 升级).

    Stage 18 #1: 因子特异性 forward_window
    Stage 18 #2: 因子特异性 lag 平滑

    Args:
        nav_df: 价格面板
        all_codes: 候选 ETF codes
        cfg: 因子择时配置
        start/end: 范围

    Returns:
        pd.DataFrame, index=date, columns=factor (平滑后 IC)
    """
    cfg = cfg or FactorTimingConfig()
    active = get_active_factors(cfg)

    if cfg.forward_window is not None:
        fw_dict = {n: cfg.forward_window for n in active}
    else:
        fw_dict = {n: cfg.factor_fw.get(n, 60) for n in active}

    panel = nav_df.loc[start:end]
    dates = panel.index
    sample_dates = dates[::cfg.ic_step]
    if len(dates) > 0 and dates[-1] not in sample_dates:
        sample_dates = sample_dates.append(pd.Index([dates[-1]]))

    rows = []
    for ts in sample_dates:
        if panel.index.get_loc(ts) < cfg.lookback + max(fw_dict.values()):
            continue
        row = {"date": ts}
        for fac in active:
            fw = fw_dict[fac]
            ic_val = factor_ic_at(
                panel, ts, list(all_codes),
                forward_window=fw, lookback=cfg.lookback,
            )
            row[fac] = ic_val.get(fac, 0.0)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")

    smooth = pd.DataFrame(index=df.index)
    for fac in df.columns:
        w = cfg.factor_smooth_window.get(fac, cfg.smooth_window)
        if w <= 1:
            smooth[fac] = df[fac]
        else:
            lookback_steps = max(1, w)
            smooth[fac] = (df[fac] * 2 + df[fac].shift(lookback_steps).fillna(df[fac])) / 3

    return smooth


def backtest_factor_weights_history(
    nav_df: pd.DataFrame,
    all_codes: Sequence[str] = ALL_V4_CODES,
    cfg: FactorTimingConfig | None = None,
    start: str | pd.Timestamp = "2020-01-01",
    end: str | pd.Timestamp = "2026-06-30",
) -> pd.DataFrame:
    """回测因子权重时序 (Stage 18)."""
    cfg = cfg or FactorTimingConfig()
    ic_history = backtest_factor_timing(
        nav_df, all_codes, cfg, start, end,
    )
    if ic_history.empty:
        return pd.DataFrame()

    weights_list = []
    dates = []
    for ts, row in ic_history.iterrows():
        f_w = compute_factor_weights(pd.DataFrame([row.to_dict()], index=[ts]), cfg)
        weights_list.append(f_w)
        dates.append(ts)

    return pd.DataFrame(weights_list, index=dates)


__all__ = [
    "FactorTimingConfig",
    "compute_factor_weights",
    "compute_strategy_weights",
    "aggregate_factor_to_etf",
    "backtest_factor_timing",
    "backtest_factor_weights_history",
    "get_active_factors",
]
