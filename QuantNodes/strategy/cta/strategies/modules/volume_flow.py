# coding=utf-8
"""量能资金流组 — 量价/持仓/资金流类因子.

从 factor_modules.py 拆出 (pure move).
"""
from __future__ import annotations

from ..base import _ModuleStrategy

from ...factors.volume import (
    AmihudFactor, OBVFactor, OIPriceDivergenceFactor, OpenInterestTrendFactor,
    VolOIRatioFactor, VolumeBreakoutFactor, VolumePriceCorrFactor, VolumeTrendFactor,
)
from ...factors.divergence import PriceVolumeDivergenceFactor, VolumeConfirmationFactor
from ...factors.volume_decay import VolumeDecayFactor
from ...factors.market_structure import VolumeReturnResidualFactor
from ...factors.capital import CapitalInflowFactor, TurnoverFactor
from ...factors.money_flow import CMFFactor, ForceIndexFactor, MFIFactor, PVTFactor
from ...factors.volume_indicators import (
    EaseOfMovementFactor, KlingerOscillatorFactor, VolumeMACrossFactor,
    VolumeRatioFactor,
)
from ...factors.price_location import VWAPDeviationFactor
from ...factors.momentum_xs import AmihudXSFactor


class VolumeFlowModuleStrategy(_ModuleStrategy):
    """量能资金流组: 量价/持仓/资金流类因子."""

    FACTORS = [
        ("volume_trend", VolumeTrendFactor, "price_vol"),
        ("volume_breakout", VolumeBreakoutFactor, "vols"),
        ("obv", OBVFactor, "price_vol"),
        ("oi_trend", OpenInterestTrendFactor, "ois"),
        ("vol_oi_ratio", VolOIRatioFactor, "price_oi"),
        ("amihud", AmihudFactor, "price_vol"),
        ("volume_price_corr", VolumePriceCorrFactor, "price_vol"),
        ("volume_confirmation", VolumeConfirmationFactor, "price_vol"),
        ("price_vol_divergence", PriceVolumeDivergenceFactor, "price_vol"),
        ("oi_price_divergence", OIPriceDivergenceFactor, "price_oi"),
        ("volume_decay", VolumeDecayFactor, "vols"),
        ("volume_ret_residual", VolumeReturnResidualFactor, "price_vol"),
        ("capital_inflow", CapitalInflowFactor, "price_oi"),
        ("turnover", TurnoverFactor, "price_vol"),
        ("mfi", MFIFactor, "ohlcv"),
        ("cmf", CMFFactor, "ohlcv"),
        ("ease_of_movement", EaseOfMovementFactor, "ohlcv"),
        ("klinger", KlingerOscillatorFactor, "ohlcv"),
        ("volume_ratio", VolumeRatioFactor, "vols"),
        ("volume_ma_cross", VolumeMACrossFactor, "vols"),
        ("pvt", PVTFactor, "price_vol"),
        ("force_index", ForceIndexFactor, "price_vol"),
        ("vwap_deviation", VWAPDeviationFactor, "ohlcv"),
        ("amihud_xs", AmihudXSFactor, "price_vol"),
    ]
