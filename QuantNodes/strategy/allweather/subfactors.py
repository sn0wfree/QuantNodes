# coding=utf-8
"""AllWeather V6 三子因子计算 (PLAN §13.3.1).

复用 CTA 因子库已有因子, 输出 T × N 日度因子分数 ∈ [-1, +1].

子因子映射 (文章"基于时序波动率、趋势力度、量能确认三维合成"):
    Sub-A 时序波动率: VolatilityConeFactor (10d/120d)
    Sub-B 趋势力度:  TrendQualityFactor + MomentumSlopeFactor
    Sub-C 量能确认:  VolumeTrendFactor + OIPriceDivergenceFactor

ETF 无 OI → Sub-C 仅用 VolumeTrendFactor (路径分支).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from QuantNodes.strategy.cta.factors.trend_quality import (
    MomentumSlopeFactor,
    TrendQualityFactor,
)
from QuantNodes.strategy.cta.factors.volume import (
    OIPriceDivergenceFactor,
    VolumeTrendFactor,
)
from QuantNodes.strategy.cta.factors.volume_decay import VolatilityConeFactor


@dataclass
class SubFactorConfig:
    """三子因子计算配置 (全程固定, 与 CTA 因子库默认窗口对齐)."""

    # Sub-A 时序波动率: 10d/120d = 2周/24周 (6个月)
    vol_cone_sw: int = 10
    vol_cone_lw: int = 120
    vol_cone_z: float = 1.0

    # Sub-B 趋势力度: R² 30d (6周) + slope 20d (4周)
    trend_quality_w: int = 30
    trend_quality_th: float = 0.6
    mom_slope_w: int = 20
    mom_slope_sw: int = 10

    # Sub-C 量能确认: 5d (1周)
    vol_trend_w: int = 5
    vol_trend_pw: int = 5
    oi_div_w: int = 5
    oi_div_pw: int = 5


def compute_subfactor_a(
    prices: pd.DataFrame,
    cfg: Optional[SubFactorConfig] = None,
) -> pd.DataFrame:
    """Sub-A 时序波动率: VolatilityConeFactor.

    输出: T × N, 范围 [-1, +1].
        +1: 短期波动相对长期放大 (regime 转折预警).
         0: 中性 (短长波动比接近历史均值).
        -1: 短期波动相对长期收缩 (regime 平稳).
    """
    cfg = cfg or SubFactorConfig()
    factor = VolatilityConeFactor(
        short_window=cfg.vol_cone_sw,
        long_window=cfg.vol_cone_lw,
        z_threshold=cfg.vol_cone_z,
    )
    return factor.compute(prices=prices)


def compute_subfactor_b(
    prices: pd.DataFrame,
    cfg: Optional[SubFactorConfig] = None,
) -> pd.DataFrame:
    """Sub-B 趋势力度: TrendQualityFactor + MomentumSlopeFactor.

    输出: T × N, 范围 [-1, +1] (等权均值, 与黄金多因子对齐).
        +1: 高 R² (趋势纯) + 动量加速 → 强趋势多.
         0: 低 R² 或动量中性 → 震荡.
        -1: 高 R² + 动量加速下行 → 强趋势空.
    """
    cfg = cfg or SubFactorConfig()
    tq = TrendQualityFactor(
        window=cfg.trend_quality_w,
        threshold=cfg.trend_quality_th,
    )
    tq_sig = tq.compute(prices=prices)

    ms = MomentumSlopeFactor(
        lookback=cfg.mom_slope_w,
        slope_window=cfg.mom_slope_sw,
    )
    ms_sig = ms.compute(prices=prices)

    return (tq_sig + ms_sig) / 2.0


def compute_subfactor_c(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    open_interest: Optional[pd.DataFrame] = None,
    cfg: Optional[SubFactorConfig] = None,
) -> pd.DataFrame:
    """Sub-C 量能确认: VolumeTrendFactor + OIPriceDivergenceFactor.

    输出: T × N, 范围 [-1, +1].
        +1: 量价齐升 (期货 OI 也升) → 趋势确认多.
         0: 量价背离 → 信号矛盾.
        -1: 量价齐跌 (期货 OI 升空) → 趋势确认空.

    ETF 无 OI: open_interest=None 时仅用 VolumeTrendFactor.
    """
    cfg = cfg or SubFactorConfig()
    vt = VolumeTrendFactor(
        window=cfg.vol_trend_w,
        price_window=cfg.vol_trend_pw,
    )
    vt_sig = vt.compute(prices=prices, volumes=volumes)

    if open_interest is not None:
        common_cols = prices.columns.intersection(open_interest.columns)
        if len(common_cols) > 0:
            oi = OIPriceDivergenceFactor(
                window=cfg.oi_div_w,
                price_window=cfg.oi_div_pw,
            )
            oi_sig = oi.compute(
                prices=prices[common_cols],
                open_interest=open_interest[common_cols],
            )
            combined = vt_sig.copy()
            combined[common_cols] = (vt_sig[common_cols] + oi_sig) / 2.0
            return combined
    return vt_sig


def compute_all_subfactors(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    open_interest: Optional[pd.DataFrame] = None,
    cfg: Optional[SubFactorConfig] = None,
) -> dict[str, pd.DataFrame]:
    """三子因子一次性计算.

    返回: {"a": T×N, "b": T×N, "c": T×N}.
    """
    cfg = cfg or SubFactorConfig()
    return {
        "a": compute_subfactor_a(prices, cfg),
        "b": compute_subfactor_b(prices, cfg),
        "c": compute_subfactor_c(prices, volumes, open_interest, cfg),
    }
