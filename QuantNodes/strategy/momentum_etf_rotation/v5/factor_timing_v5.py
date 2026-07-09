# coding=utf-8
"""v5 因子择时 (FactorTiming v5.0).

基于 Stage 17 v4 诊断 (SUB_STRATEGY_DIAGNOSTIC.md) 5 大改进:

1. **因子特异性 forward_window**:
   - momentum: 120d (vs 统一 20d)
   - reversal: 60d
   - value: 40d
   - dividend: 180d
   - quality: 252d
   - **low_vol: 不用** (IC vs forward 强负相关 -0.454, 反指因子)

2. **lag1 平滑 (momentum/value/dividend/quality)**:
   - momentum/value/dividend/quality lag1=0.48-0.69 高持续
   - reversal lag1=-0.01 无持续 → 不平滑

3. **regime-conditioned 因子选择**:
   - bull: 仅 momentum + value
   - bear: value + dividend + quality (防御)
   - sideways: 仅 value

4. **low_vol 移除**: 反指因子, 直接删除

5. **IC 质量过滤**: |IC| < 0.05 时该因子 weight=0 (|IC|>0.05 频率 84-94% 噪声)

诊断基础:
- reports/momentum_etf_rotation/v4/SUB_STRATEGY_DIAGNOSTIC.md §2.1-2.7
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..v4.factor_ic import (
    factor_ic_at,
)
from ..v4.universe_v4 import (
    ALL_V4_CODES,
    SMART_BETA_CODES,
    STYLE_GROUP_CODES,
    SmartBetaFactor,
)
from ..v4.sub_strategy_v4 import SubStrategy, SubStrategyConfig, SubStrategyResult


@dataclass
class FactorTimingV5Config(SubStrategyConfig):
    """v5 因子择时配置 (5 改进)."""
    name: str = "factor_timing_v5"

    factor_fw: dict[str, int] = field(default_factory=lambda: {
        "momentum": 120,    # 改进 1
        "reversal": 60,
        "value":    40,
        "dividend": 180,
        "quality":  252,
    })
    factor_lookback: int = 60

    factor_smooth_window: dict[str, int] = field(default_factory=lambda: {
        "momentum": 4,      # 改进 2: lag1 平滑 4 周
        "value":    4,
        "dividend": 4,
        "quality":  4,
        "reversal": 1,      # reversal lag1=-0.01, 不平滑
    })

    factor_ic_threshold: float = 0.05  # 改进 5: |IC|<0.05 视为噪声

    regime_factors: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "bull":     ("momentum", "value"),            # 改进 3
        "bear":     ("value", "dividend", "quality"),
        "sideways": ("value",),
    })

    factor_to_etf: dict[str, tuple[str, ...]] = field(default_factory=lambda: {
        "momentum": ("510300", "510500", "159915", "588000", "510880"),
        "reversal": ("510300", "510500", "159915", "588000", "510880"),
        "value":    ("512040",),
        "dividend": ("510880", "512890", "515080", "515100"),
        "quality":  ("515900",),
    })

    base: float = 0.05
    power: float = 2.0
    min_weight: float = 0.10
    max_weight: float = 0.50

    ic_step: int = 5
    min_history: int = 252

    rebalance_freq: str = "M"


def compute_v5_factor_weights(
    ic_smooth: dict[str, float],
    cfg: FactorTimingV5Config,
    regime: str,
) -> dict[str, float]:
    """v5 因子权重计算.

    算法:
        1. 根据 regime 选择可用因子 (改进 3)
        2. |IC| < threshold 的因子 weight=0 (改进 5)
        3. raw_w[name] = max(0, IC[name] + base) ** power
        4. 应用 min_weight / max_weight
        5. 归一化

    Returns:
        dict, factor name → weight (sum=1, 或 sum<1 表示有 cash)
    """
    available_factors = cfg.regime_factors.get(regime, ("value",))

    weights: dict[str, float] = {}
    for name in available_factors:
        ic = ic_smooth.get(name, 0.0)
        if abs(ic) < cfg.factor_ic_threshold:
            continue
        raw = max(0.0, ic + cfg.base) ** cfg.power
        weights[name] = raw

    if not weights:
        return {n: 0.0 for n in available_factors}

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    for name in weights:
        if weights[name] < cfg.min_weight:
            weights[name] = cfg.min_weight
        if weights[name] > cfg.max_weight:
            weights[name] = cfg.max_weight

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def aggregate_factor_to_etf(
    factor_weights: dict[str, float],
    cfg: FactorTimingV5Config,
) -> dict[str, float]:
    """因子权重 → ETF 权重 (因子内等权).

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


def compute_ic_history_v5(
    nav_df: pd.DataFrame,
    cfg: FactorTimingV5Config,
    all_codes: Sequence[str],
    rebal_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    """v5 IC 时序: 用因子特异性 FW + lag 平滑.

    Returns:
        DataFrame, index=date, columns=factor (smoothed IC)
    """
    ic_records = []
    for date in rebal_dates:
        row = {"date": date}
        for fac, fw in cfg.factor_fw.items():
            ic = factor_ic_at(
                nav_df, date, all_codes,
                forward_window=fw, lookback=cfg.factor_lookback,
            )
            row[fac] = ic.get(fac, 0.0)
        ic_records.append(row)

    df = pd.DataFrame(ic_records).set_index("date")

    smooth = pd.DataFrame(index=df.index)
    for fac in df.columns:
        w = cfg.factor_smooth_window.get(fac, 1)
        if w <= 1:
            smooth[fac] = df[fac]
        else:
            smooth[fac] = df[fac].rolling(window=w, min_periods=1).mean()

    return smooth


def classify_regime_v5(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp,
    short_window: int = 60,
    long_window: int = 252,
    bull_short: float = 0.05,
    bear_short: float = -0.05,
    long_th: float = 0.10,
    long_neg_th: float = -0.10,
) -> str:
    """用 HS300 (510300) 动量分类 regime."""
    if "510300" not in nav_df.columns:
        return "sideways"
    sub = nav_df.loc[:as_of, "510300"]
    if len(sub) < long_window + 1:
        return "sideways"

    mom_short = float(sub.iloc[-1] / sub.iloc[-short_window - 1] - 1.0)
    mom_long = float(sub.iloc[-1] / sub.iloc[-long_window - 1] - 1.0)

    if mom_short > bull_short and mom_long > long_th:
        return "bull"
    if mom_short < bear_short and mom_long < long_neg_th:
        return "bear"
    return "sideways"


class FactorTimingV5SubStrategy(SubStrategy):
    """v5 因子择时子策略.

    选股逻辑:
        1. 计算 v5 IC 时序 (因子特异性 FW + lag 平滑) — 改进 1+2
        2. regime-conditioned 因子选择 — 改进 3
        3. |IC| < 0.05 过滤 — 改进 5
        4. raw_w = max(0, IC+base)^power
        5. 因子权重 → ETF 权重 (因子内等权)
        6. 不使用 low_vol — 改进 4 (反指因子)
    """

    def __init__(self, config: FactorTimingV5Config):
        super().__init__(config)
        self.config: FactorTimingV5Config = config

    def select(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> list[str]:
        return list(set(c for codes in self.config.factor_to_etf.values() for c in codes))

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        return {}

    def run_step(
        self,
        nav_df: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> SubStrategyResult:
        cfg = self.config
        if cfg.min_history > 0 and len(nav_df) < cfg.min_history:
            return SubStrategyResult(date=as_of, meta={"strategy": cfg.name})

        all_codes = list(ALL_V4_CODES)
        available = [c for c in all_codes if c in nav_df.columns]

        ic_now: dict[str, float] = {}
        for fac, fw in cfg.factor_fw.items():
            ic = factor_ic_at(nav_df, as_of, available,
                              forward_window=fw, lookback=cfg.factor_lookback)
            ic_now[fac] = ic.get(fac, 0.0)

        if "510300" in nav_df.columns:
            recent = nav_df.loc[:as_of, "510300"].iloc[-1 - cfg.factor_smooth_window.get("momentum", 4):]
            if len(recent) >= 2:
                avg_recent = float(recent.mean())
                sub_past = nav_df.loc[:as_of, "510300"].iloc[-cfg.factor_smooth_window.get("momentum", 4) - 1 - 1]
                smoothed_ic_mom = float(recent.iloc[-1] / sub_past - 1.0) * 0
            else:
                smoothed_ic_mom = 0.0
        else:
            smoothed_ic_mom = 0.0

        ic_smooth = dict(ic_now)
        for fac, w in cfg.factor_smooth_window.items():
            if w > 1 and fac in ic_smooth:
                idx = nav_df.index.get_loc(as_of)
                look_back = min(w * cfg.ic_step, idx)
                if look_back > 0:
                    ic_smooth[fac] = float(ic_now[fac])
                    past_idx = max(0, idx - look_back)
                    past_date = nav_df.index[past_idx]
                    try:
                        ic_past = factor_ic_at(nav_df, past_date, available,
                                               forward_window=cfg.factor_fw[fac],
                                               lookback=cfg.factor_lookback)
                        ic_smooth[fac] = (ic_now[fac] * 2 + ic_past.get(fac, 0.0)) / 3
                    except Exception:
                        pass

        regime = classify_regime_v5(nav_df, as_of)
        f_weights = compute_v5_factor_weights(ic_smooth, cfg, regime)
        etf_weights = aggregate_factor_to_etf(f_weights, cfg)

        if not etf_weights:
            etf_weights = {self.config.factor_to_etf["value"][0]: 1.0}

        etf_weights = self._apply_max_weight(etf_weights, cfg.max_weight)
        total = sum(etf_weights.values())
        if total > 0:
            etf_weights = {k: v / total for k, v in etf_weights.items()}

        cash = 1.0 - sum(etf_weights.values())
        if cash < 0:
            cash = 0.0
            total = sum(etf_weights.values())
            etf_weights = {k: v / total for k, v in etf_weights.items()}

        return SubStrategyResult(
            date=as_of,
            chosen=list(etf_weights.keys()),
            weights=etf_weights,
            signal_strength=float(np.mean([abs(ic_smooth.get(f, 0.0)) for f in f_weights if f_weights.get(f, 0) > 0])) if f_weights else 0.0,
            meta={
                "strategy": cfg.name,
                "regime": regime,
                "ic_smooth": ic_smooth,
                "factor_weights": f_weights,
                "cash_weight": cash,
            },
        )


__all__ = [
    "FactorTimingV5Config",
    "FactorTimingV5SubStrategy",
    "compute_v5_factor_weights",
    "aggregate_factor_to_etf",
    "compute_ic_history_v5",
    "classify_regime_v5",
]
