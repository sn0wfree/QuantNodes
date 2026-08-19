# coding=utf-8
"""CTA 因子库.

时序因子:
    - trend: Donchian / MA / Momentum / TSMOM / RSI / Bollinger / Efficiency / ADX
    - volatility: HV / Parkinson / VolRegime / VolSpread / vol_target_positions
    - volume: VolumeTrend / VolumeBreakout / OI / OBV / VolOIRatio / Amihud / VolumePriceCorr
    - term_structure: Basis / RollYield / TermSlope / TermCurvature / NearBasis
    - statistical: Hurst / VarianceRatio / Autocorrelation / Skewness
    - microstructure: Overnight / Intraday / Divergence / Gap / CloseLocation
    - composite: MultiHorizonMomentum / SharpeMomentum / MomentumCrashFilter

横截面因子:
    - momentum_xs: MomentumXS / VolatilityXS / BasisXS

技术指标:
    - indicators: RSI / ATR / ADX / Bollinger / EfficiencyRatio / Donchian / ZScore
"""
from __future__ import annotations

from .base import Factor, PanelFactor, time_series_factor
from .indicators import (
    adx,
    atr,
    bollinger_bands,
    bollinger_percent_b,
    donchian,
    efficiency_ratio,
    rsi,
    zscore,
)
from .trend import (
    ADXFilterFactor,
    ATRBreakoutFactor,
    BIASFactor,
    BollingerFactor,
    DonchianChannelFactor,
    EfficiencyFactor,
    MAFactor,
    MomentumFactor,
    RSIFactor,
    TSMAFactor,
    compute_atr,
)
from .term_structure import (
    BasisFactor,
    NearBasisFactor,
    RollYieldFactor,
    TermCurvatureFactor,
    TermSlopeFactor,
    basis_to_signal,
)
from .volatility import (
    GARCHFactor,
    HVFactor,
    ParkinsonFactor,
    VolRegimeFactor,
    VolSkewnessFactor,
    VolSpreadFactor,
    vol_target_positions,
)
from .volume import (
    AmihudFactor,
    OIPriceDivergenceFactor,
    OBVFactor,
    OpenInterestTrendFactor,
    VolOIRatioFactor,
    VolumeBreakoutFactor,
    VolumePriceCorrFactor,
    VolumeTrendFactor,
)
from .momentum_xs import (
    AmihudXSFactor,
    BasisXSFactor,
    MomentumXSFactor,
    VolatilityXSFactor,
)
from .statistical import (
    AutocorrelationFactor,
    FractalDimensionFactor,
    HurstExponentFactor,
    SkewnessFactor,
    VarianceRatioFactor,
)
from .microstructure import (
    CloseLocationFactor,
    GapFactor,
    IntradayReturnFactor,
    OvernightIntradayDivergenceFactor,
    OvernightReturnFactor,
)
from .composite import (
    MomentumCrashFilterFactor,
    MultiHorizonMomentumFactor,
    SharpeMomentumFactor,
)
from .industrial_chain import (
    CHAIN_FORMULAS,
    ChainMarginFactor,
    LeadLagFactor,
    MarginMomentumFactor,
    PriceRatioFactor,
    chain_margin_series,
)
from .calendar import (
    MonthTurnFactor,
    SeasonalityFactor,
    WeekdayFactor,
)
from .capital import (
    CapitalInflowFactor,
    TurnoverFactor,
)
from .position import (
    FrontMonthOIShareFactor,
    OIConcentrationFactor,
)
from .divergence import (
    PriceVolumeDivergenceFactor,
    VolumeConfirmationFactor,
)
from .correlation import (
    AvgCorrelationFactor,
    DispersionFactor,
)
from .basis_momentum import (
    BasisLevelFactor,
    BasisMomentumFactor,
)
from .risk import (
    DownsideVolFactor,
    MaxDrawdownFactor,
    TailRiskFactor,
)
from .macro_link import (
    DXYLinkFactor,
    MacroStateFactor,
    VIXLinkFactor,
)
from .event import (
    ExtremeMoveFactor,
    GapStreakFactor,
    LimitMoveFactor,
)
from .price_location import (
    OpenLocationFactor,
    PriceLocationFactor,
    VWAPDeviationFactor,
)
from .trend_quality import (
    MomentumSlopeFactor,
    TrendQualityFactor,
)
from .volume_decay import (
    VolatilityConeFactor,
    VolumeDecayFactor,
)
from .candlestick import (
    DojiFactor,
    EngulfingFactor,
    HammerFactor,
    LongBodyFactor,
)
from .cointegration import (
    PairRatioMeanReversionFactor,
    SpreadMeanReversionFactor,
)
from .market_structure import (
    NewHighLowFactor,
    VolumeReturnResidualFactor,
)
from .money_flow import (
    CMFFactor,
    ForceIndexFactor,
    MFIFactor,
    PVTFactor,
)
from .volume_indicators import (
    EaseOfMovementFactor,
    KlingerOscillatorFactor,
    VolumeMACrossFactor,
    VolumeRatioFactor,
)
from .oscillators import (
    CCIFactor,
    MACDFactor,
    StochasticFactor,
    WilliamsRFactor,
)
from .trend_advanced import (
    AroonFactor,
    ChandelierExitFactor,
    LinearSlopeFactor,
    TRIXFactor,
    VortexFactor,
)
from .mean_reversion import (
    HalfLifeFactor,
    SqueezeFactor,
)
from .short_term_reversal import ShortTermReversalTrendFactor
from .turn_point import (
    TurnPointBreakoutFactor,
    FastTurnPointFactor,
)

__all__ = [
    # base
    "Factor",
    "PanelFactor",
    "time_series_factor",
    # indicators
    "adx",
    "atr",
    "bollinger_bands",
    "bollinger_percent_b",
    "donchian",
    "efficiency_ratio",
    "rsi",
    "zscore",
    # trend
    "ADXFilterFactor",
    "BollingerFactor",
    "DonchianChannelFactor",
    "EfficiencyFactor",
    "MAFactor",
    "MomentumFactor",
    "RSIFactor",
    "TSMAFactor",
    "compute_atr",
    # term structure
    "BasisFactor",
    "NearBasisFactor",
    "RollYieldFactor",
    "TermCurvatureFactor",
    "TermSlopeFactor",
    "basis_to_signal",
    # volatility
    "GARCHFactor",
    "HVFactor",
    "ParkinsonFactor",
    "VolRegimeFactor",
    "VolSkewnessFactor",
    "VolSpreadFactor",
    "vol_target_positions",
    # volume
    "AmihudFactor",
    "OIPriceDivergenceFactor",
    "OBVFactor",
    "OpenInterestTrendFactor",
    "VolOIRatioFactor",
    "VolumeBreakoutFactor",
    "VolumePriceCorrFactor",
    "VolumeTrendFactor",
    # cross-sectional
    "BasisXSFactor",
    "MomentumXSFactor",
    "VolatilityXSFactor",
    # statistical
    "AutocorrelationFactor",
    "HurstExponentFactor",
    "SkewnessFactor",
    "VarianceRatioFactor",
    # microstructure
    "CloseLocationFactor",
    "GapFactor",
    "IntradayReturnFactor",
    "OvernightIntradayDivergenceFactor",
    "OvernightReturnFactor",
    # composite
    "MomentumCrashFilterFactor",
    "MultiHorizonMomentumFactor",
    "SharpeMomentumFactor",
    # industrial chain
    "CHAIN_FORMULAS",
    "ChainMarginFactor",
    "LeadLagFactor",
    "MarginMomentumFactor",
    "PriceRatioFactor",
    "chain_margin_series",
    # calendar
    "MonthTurnFactor",
    "SeasonalityFactor",
    "WeekdayFactor",
    # capital
    "CapitalInflowFactor",
    "TurnoverFactor",
    # position
    "FrontMonthOIShareFactor",
    "OIConcentrationFactor",
    # divergence
    "PriceVolumeDivergenceFactor",
    "VolumeConfirmationFactor",
    # correlation
    "AvgCorrelationFactor",
    "DispersionFactor",
    # basis momentum
    "BasisLevelFactor",
    "BasisMomentumFactor",
    # risk
    "DownsideVolFactor",
    "MaxDrawdownFactor",
    "TailRiskFactor",
    # macro link
    "DXYLinkFactor",
    "MacroStateFactor",
    "VIXLinkFactor",
    # event
    "ExtremeMoveFactor",
    "GapStreakFactor",
    "LimitMoveFactor",
    # price location
    "OpenLocationFactor",
    "PriceLocationFactor",
    "VWAPDeviationFactor",
    # trend quality
    "MomentumSlopeFactor",
    "TrendQualityFactor",
    # volume decay / volatility cone
    "VolatilityConeFactor",
    "VolumeDecayFactor",
    # cross-sectional amihud
    "AmihudXSFactor",
    # candlestick
    "DojiFactor",
    "EngulfingFactor",
    "HammerFactor",
    "LongBodyFactor",
    # cointegration
    "PairRatioMeanReversionFactor",
    "SpreadMeanReversionFactor",
    # market structure
    "NewHighLowFactor",
    "VolumeReturnResidualFactor",
    # trend extras
    "ATRBreakoutFactor",
    "BIASFactor",
    # statistical extras
    "FractalDimensionFactor",
    # money flow
    "CMFFactor",
    "ForceIndexFactor",
    "MFIFactor",
    "PVTFactor",
    # volume indicators
    "EaseOfMovementFactor",
    "KlingerOscillatorFactor",
    "VolumeMACrossFactor",
    "VolumeRatioFactor",
    # oscillators
    "CCIFactor",
    "MACDFactor",
    "StochasticFactor",
    "WilliamsRFactor",
    # trend advanced
    "AroonFactor",
    "ChandelierExitFactor",
    "LinearSlopeFactor",
    "TRIXFactor",
    "VortexFactor",
    # mean reversion
    "HalfLifeFactor",
    "SqueezeFactor",
    # turn point
    "TurnPointBreakoutFactor",
    "FastTurnPointFactor",
    # short term reversal
    "ShortTermReversalTrendFactor",
]
