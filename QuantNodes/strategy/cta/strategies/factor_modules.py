# coding=utf-8
"""多因子 CTA 策略 — 四模块子策略 + 组合 (策略 v3).

设计依据: docs/94-多因子CTA策略设计.md + 106 因子全库扩展调研

结构:
    信号层: 全因子 compute() 信号 (-1/0/+1) 统一投票, 因子级面板保留 (归因)
    合成层: 模块内等权求和 → 模块方向; 4 组: 趋势动量/均值回归/量能资金流/事件日历
    执行层: 模块加权合成净信号 → sign → vol_target_positions (单一资金池) → 回测
    过滤层: 波动率风险类因子 (HV/GARCH/统计/回撤等) 第二阶段接入

输入类型约定 (FACTORS 元组):
    panel      → 作用于 close 面板
    ohlc       → 逐品种 OHLC
    ohlcv      → 逐品种 OHLCV
    price_vol  → (close 面板, volume 面板)
    price_oi   → (close 面板, OI 面板)
    vols       → 逐品种 volume 面板
    ois        → 逐品种 OI 面板
    spread     → 全品种近远月期限结构 long-form (load_term_structure)
    basis      → 全品种现货基差 long-form (spot_basis_daily)
    macro_vix  → (close 面板, VIX 序列)
    macro_dxy  → (close 面板, DXY 序列)
    series     → 逐品种单序列 (每列独立 compute)
    chain      → 产业链因子, 按模块 CHAIN_MAP 映射到品种列

用法:
    from QuantNodes.strategy.cta.strategies.factor_modules import (
        TrendMomentumModuleStrategy, FactorModuleComboStrategy,
    )

    strat = TrendMomentumModuleStrategy(products=["AU", "RB"], start="2018-01-01")
    result = strat.run()
    print(result.metrics)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..backtest.cta_engine import CtaBacktestResult
from ..factors.calendar import MonthTurnFactor, SeasonalityFactor, WeekdayFactor
from ..factors.candlestick import (
    DojiFactor, EngulfingFactor, HammerFactor, LongBodyFactor,
)
from ..factors.capital import CapitalInflowFactor, TurnoverFactor
from ..factors.composite import (
    MomentumCrashFilterFactor, MultiHorizonMomentumFactor, SharpeMomentumFactor,
)
from ..factors.divergence import (
    PriceVolumeDivergenceFactor, VolumeConfirmationFactor,
)
from ..factors.event import ExtremeMoveFactor, GapStreakFactor, LimitMoveFactor
from ..factors.market_structure import NewHighLowFactor, VolumeReturnResidualFactor
from ..factors.mean_reversion import HalfLifeFactor, SqueezeFactor
from ..factors.microstructure import (
    CloseLocationFactor, GapFactor, IntradayReturnFactor,
    OvernightIntradayDivergenceFactor, OvernightReturnFactor,
)
from ..factors.momentum_xs import AmihudXSFactor, MomentumXSFactor
from ..factors.money_flow import CMFFactor, ForceIndexFactor, MFIFactor, PVTFactor
from ..factors.oscillators import (
    CCIFactor, MACDFactor, StochasticFactor, WilliamsRFactor,
)
from ..factors.price_location import (
    OpenLocationFactor, PriceLocationFactor, VWAPDeviationFactor,
)
from ..factors.trend import (
    ATRBreakoutFactor, BIASFactor, BollingerFactor,
    EfficiencyFactor, MAFactor, MomentumFactor, RSIFactor,
)
from ..factors.trend_advanced import (
    AroonFactor, ChandelierExitFactor, LinearSlopeFactor, TRIXFactor, VortexFactor,
)
from ..factors.trend_quality import MomentumSlopeFactor, TrendQualityFactor
from ..factors.volume import (
    AmihudFactor, OBVFactor, OIPriceDivergenceFactor, OpenInterestTrendFactor,
    VolOIRatioFactor, VolumeBreakoutFactor, VolumePriceCorrFactor, VolumeTrendFactor,
)
from ..factors.volume_decay import VolumeDecayFactor
from ..factors.volume_indicators import (
    EaseOfMovementFactor, KlingerOscillatorFactor, VolumeMACrossFactor,
    VolumeRatioFactor,
)
from ..factors.term_structure import TermSlopeFactor, BasisConvergenceFactor, CalendarSpreadMeanReversion
from ..factors.basis_momentum import BasisLevelFactor
from ..factors.macro_link import DXYLinkFactor, MacroStateFactor, VIXLinkFactor
from ..factors.turn_point import TurnPointBreakoutFactor, FastTurnPointFactor
from ..factors.short_term_reversal import ShortTermReversalTrendFactor
from ..factors.industrial_chain import (
    ChainMarginFactor, LeadLagFactor, MarginMomentumFactor, PriceRatioFactor,
)
from ..factors.cointegration import SpreadMeanReversionFactor


# ── 基类/配置/品种池 (已拆至 base.py, 此处 re-export 保持兼容) ───────────
from .base import (  # noqa: E402,F401
    DEFAULT_PRODUCTS,
    ModuleConfig,
    _ModuleStrategy,
    vol_target_positions,
)

# ── 波动率风险过滤层 (已拆至 filters.py, 此处 re-export 保持兼容) ──────────
from .filters import MarketRegimeFilter, VolRiskFilter  # noqa: E402,F401

# ── 模块子策略 (Phase3 将拆至 modules/) ─────────────────────────────────

class TrendMomentumModuleStrategy(_ModuleStrategy):
    """趋势动量组: 价格趋势/动量/突破类因子.

    去重 (docs/94 §10.4): donchian×ma 相关 0.798 (72 品种离散信号最高对)
    且 donchian 单因子 Sharpe 最弱 (-0.52), 删除.
    """

    FACTORS = [
        ("ma", MAFactor, "panel"),
        ("momentum", MomentumFactor, "panel"),
        ("rsi", RSIFactor, "panel"),
        ("efficiency", EfficiencyFactor, "panel"),
        ("aroon", AroonFactor, "panel"),
        ("linear_slope", LinearSlopeFactor, "panel"),
        ("trix", TRIXFactor, "panel"),
        ("macd", MACDFactor, "panel"),
        ("multi_horizon_mom", MultiHorizonMomentumFactor, "panel"),
        ("sharpe_mom", SharpeMomentumFactor, "panel"),
        ("mom_crash_filter", MomentumCrashFilterFactor, "panel"),
        ("mom_slope", MomentumSlopeFactor, "panel"),
        ("trend_quality", TrendQualityFactor, "panel"),
        ("new_high_low", NewHighLowFactor, "panel"),
        ("momentum_xs", MomentumXSFactor, "panel"),
        ("atr_breakout", ATRBreakoutFactor, "ohlc"),
        ("turn_point_breakout", TurnPointBreakoutFactor, "ohlc"),
        ("fast_turn_point", FastTurnPointFactor, "ohlc"),
        ("short_term_reversal_trend", ShortTermReversalTrendFactor, "panel"),
        ("overnight_ret", OvernightReturnFactor, "ohlc"),
        ("intraday_ret", IntradayReturnFactor, "ohlc"),
        ("overnight_intraday_div", OvernightIntradayDivergenceFactor, "ohlc"),
        ("long_body", LongBodyFactor, "ohlc"),
        ("vortex", VortexFactor, "ohlc"),
        ("chandelier", ChandelierExitFactor, "ohlc"),
    ]


class OscillatorAlphaModuleStrategy(_ModuleStrategy):
    """振荡器Alpha组: 15因子振荡器+inv_vol加权+双向交易.

    核心逻辑:
        1. 15个振荡器/反转因子独立计算信号
        2. inv_vol加权: 波动率低的因子权重更高
        3. 双向交易: 55%做多, 44%做空
        4. 始终在场: 仅1%时间空仓
        5. vol_targeting: 仓位按波动率调整

    收益来源:
        - 做多贡献: +11.0% 年化 (主要来源)
        - 做空贡献: +6.7% 年化 (次要来源)
        - 始终在场: 避免空仓机会成本

    适用场景:
        - 趋势市: 更赚钱 (33%年化)
        - 震荡市: 表现一般 (2%年化)
    """
    # 保留旧名称作为别名
    MeanReversionModuleStrategy = None  # 向后兼容

    FACTORS = [
        ("bollinger", BollingerFactor, "panel"),
        ("bias", BIASFactor, "panel"),
        ("half_life", HalfLifeFactor, "panel"),
        ("price_location", PriceLocationFactor, "panel"),
        ("spread_mr", SpreadMeanReversionFactor, "series"),
        ("stochastic", StochasticFactor, "ohlc"),
        ("cci", CCIFactor, "ohlc"),
        ("williams_r", WilliamsRFactor, "ohlc"),
        ("squeeze", SqueezeFactor, "ohlc"),
        ("close_location", CloseLocationFactor, "ohlc"),
        ("open_location", OpenLocationFactor, "ohlc"),
        ("doji", DojiFactor, "ohlc"),
        ("hammer", HammerFactor, "ohlc"),
        ("engulfing", EngulfingFactor, "ohlc"),
        ("gap", GapFactor, "ohlc"),
    ]


# 向后兼容
MeanReversionModuleStrategy = OscillatorAlphaModuleStrategy


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


class EventCalendarModuleStrategy(_ModuleStrategy):
    """事件日历组: 极端波动/跳变/日历效应类因子."""

    FACTORS = [
        ("extreme_move", ExtremeMoveFactor, "panel"),
        ("gap_streak", GapStreakFactor, "panel"),
        ("limit_move", LimitMoveFactor, "panel"),
        ("seasonality", SeasonalityFactor, "panel"),
        ("weekday", WeekdayFactor, "panel"),
        ("month_turn", MonthTurnFactor, "panel"),
    ]


class TermBasisModuleStrategy(_ModuleStrategy):
    """期限结构/基差组: 近远月价差 + 现货基差类因子.

    数据: load_term_structure (近远月 OI top-4 合约) + spot_basis_daily.
    因子选择 (docs/94 §7.5/§8.9): 9 因子归因仅 term_slope/basis_level 为正
    (basis/roll_yield/term_curvature/oi_concentration/front_month_oi/
    near_basis/basis_momentum 单边信号在国内商品系统性失效, 剔除).
    basis_level 单因子 72 品种归因转负 (-0.44) 但组合层面为正贡献
    (sign 非线性), 保留.
    IF/T 等无现货基差品种无信号 (NaN → 0).
    """

    FACTORS = [
        ("term_slope", TermSlopeFactor, "spread"),
        ("basis_level", BasisLevelFactor, "basis"),
        ("basis_convergence", BasisConvergenceFactor, "basis"),
        ("calendar_spread_mr", CalendarSpreadMeanReversion, "spread"),
    ]


class MacroModuleStrategy(_ModuleStrategy):
    """宏观联动组: VIX/DXY 与品种联动类因子."""

    FACTORS = [
        ("vix_link", VIXLinkFactor, "macro_vix"),
        ("dxy_link", DXYLinkFactor, "macro_dxy"),
        ("macro_state", MacroStateFactor, "macro_vix"),
    ]


class IndustrialChainModuleStrategy(_ModuleStrategy):
    """产业链组: 盘面利润/利润动量/比价/上下游传导.

    信号 → 品种映射 (CHAIN_MAP):
        chain_margin_rb  → RB   (螺纹盘面利润, 交易下游成品)
        chain_margin_hc  → HC   (热卷盘面利润)
        chain_margin_pp  → PP   (MTO 利润)
        margin_momentum_m → M   (压榨利润动量)
        price_ratio_au_cu → AU-CU 配对 (比价高估 → 空 AU 多 CU)
        lead_lag_i_rb    → RB   (铁矿传导顺畅 → 做多下游螺纹)
        chain_margin_j   → J    (焦化利润)
        chain_margin_ta  → TA   (PTA加工费)
        chain_margin_lh  → LH   (生猪饲料成本)
    """

    FACTORS = [
        # 钢铁链: 铁矿→焦煤→焦炭→螺纹/热卷
        ("chain_margin_rb", lambda: ChainMarginFactor("steel_profit_rb"), "chain"),
        ("chain_margin_hc", lambda: ChainMarginFactor("steel_profit_hc"), "chain"),
        # 化工链: 石化→化工
        ("chain_margin_pp", lambda: ChainMarginFactor("mto_profit"), "chain"),
        # 农产品链: 大豆压榨
        ("margin_momentum_m", lambda: MarginMomentumFactor("crush_profit"), "chain"),
        # 跨品种
        ("price_ratio_au_cu", lambda: PriceRatioFactor("AU", "CU"), "chain"),
        ("lead_lag_i_rb", lambda: LeadLagFactor("I", "RB"), "chain"),
        # 新增: 焦化链
        ("chain_margin_j", lambda: ChainMarginFactor("coke_profit"), "chain"),
        # 新增: PTA加工费
        ("chain_margin_ta", lambda: ChainMarginFactor("pta_fee"), "chain"),
        # 新增: 生猪饲料成本
        ("chain_margin_lh", lambda: ChainMarginFactor("hog_feed_cost"), "chain"),
    ]

    CHAIN_MAP = {
        "chain_margin_rb": "RB",       # 螺纹盘面利润 → RB
        "chain_margin_hc": "HC",       # 热卷盘面利润 → HC
        "chain_margin_pp": "PP",       # MTO利润 → PP
        "margin_momentum_m": "M",      # 大豆压榨利润 → M
        "price_ratio_au_cu": "AU-CU",  # 比价 → AU和CU
        "lead_lag_i_rb": "RB",         # 铁矿传导 → RB
        "chain_margin_j": "J",         # 焦化利润 → J
        "chain_margin_ta": "TA",       # PTA加工费 → TA
        "chain_margin_lh": "LH",       # 生猪饲料成本 → LH
    }


# ── 组合策略 ───────────────────────────────────────────────────────────────

@dataclass
class ComboConfig(ModuleConfig):
    """组合策略配置: 模块权重 (趋势/量能/期限基差/宏观/产业链) + 过滤层.

    权重策略: 默认等权 1/5 (5 组), 简单稳健, 不做样本内调优
    (调优结果仅记录于 docs/94, 不设为默认).
    """

    module_weights: tuple = (
        1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5,
    )
    module_weighting: str = "fixed"  # 组间加权: fixed | inv_vol | regression | tvpr
    filter_enabled: bool = True      # 市场级波动危机过滤 (Sharpe 1.109→1.190)
    momentum_filter_window: int = 0  # 信号动量过滤窗口 (天), 0=关闭 (docs/94 §10.7)
    momentum_filter_coef: float = 0.0  # 动量为负时仓位系数 (0=全空仓, 1=不过滤)
    dd_filter_enabled: bool = False  # NAV 回撤降杠杆: 回撤超阈值 → 全局降杠杆
    dd_filter_threshold: float = 0.05  # 回撤触发阈值 (5% = 0.05)
    dd_filter_coef: float = 0.5     # 触发后杠杆系数 (0.5 = 半仓)
    leverage_hard_cap: float = 0.0   # 组合总杠杆硬上限 (逐日 gross/equity), 0=关闭
    min_position_change: float = 0.30  # 仓位变化 < 此比例不调仓 (降换手), 0=关闭
    slippage_bps: float | None = None  # L3滑点校准系数 (None=无滑点, 1.0=标准)
    roll_cost_bps: float = 0.0  # 换月额外成本 (基点), 0=不模拟换月
    roll_days: int = 5          # 换月成本摊销天数
    roll_model: str = "amortized"  # 换月模型: 'amortized'均摊 / 'lump'一次性
    signal_ema_span: int = 0    # 信号EMA平滑跨度 (天), 0=关闭
    min_daily_volume: float = 1000.0  # 品种日均成交量阈值 (手), 低于此剔除
    position_ema_alpha: float = 0.0  # 持仓EMA平滑 (推荐0.15), 0=关闭
    # 信号一致性过滤 (cluster内信号不一致→取消)
    enable_cluster_filter: bool = True
    cluster_conflict_threshold: float = 0.5  # cluster内冲突比例>此值→取消信号
    # 尾部风险保护 (高相关性→全局降仓)
    tail_risk_enabled: bool = True
    tail_risk_corr_window: int = 60   # 相关性计算窗口
    tail_risk_corr_threshold: float = 0.30  # 高相关阈值 (自适应: 品种少时自动提高)
    tail_risk_scale: float = 0.7      # 高相关时仓位乘数 (0.7=降仓30%)
    # 市场状态过滤 (震荡市→降仓)
    regime_filter_enabled: bool = False  # 市场状态过滤开关
    regime_adx_threshold: float = 20.0  # ADX阈值: <此值=震荡市
    regime_chop_coef: float = 0.4      # 震荡市仓位系数
    # 单品种名义敞口系数: 上限 = equity × max_leverage × 此系数 (默认0.25=旧行为)
    max_product_notional_coef: float = 0.25
    # 全局杠杆系数: 资金等分基数放大 (1.0=现状; 1.7≈杠杆0.53→0.90, 线性放大器)
    equity_leverage_k: float = 1.0


class FactorModuleComboStrategy(_ModuleStrategy):
    """组合策略: 五模块方向信号加权 → 单一资金池 vol targeting.

    模块: TrendMomentum + VolumeFlow + TermBasis + Macro + IndustrialChain
    权重默认等权 0.2×5 (可调, 趋势主导可改为 0.4/0.2/0.3/0.1).
    波动率风险过滤层 (HV/GARCH/统计/回撤类) 第二阶段接入.
    """

    def __init__(self, **kwargs):
        config = kwargs.pop("config", None)
        super().__init__(config=config, **kwargs)
        if not isinstance(self._config, ComboConfig):
            self._config = ComboConfig(**{
                f: getattr(self._config, f) for f in ModuleConfig.__dataclass_fields__
            })
        cfg = self._config
        self._modules = [
            TrendMomentumModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            VolumeFlowModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            TermBasisModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            MacroModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            IndustrialChainModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
        ]
        self.module_signals: dict[str, pd.DataFrame] = {}
        self._filter = VolRiskFilter() if self._config.filter_enabled else None

    # 品种Cluster定义 (板块分组, 覆盖全部品种, 供一致性过滤/尾部风险/敞口上限共用)
    CLUSTERS = {
        'precious': ['AU', 'AG'],
        'black': ['RB', 'HC', 'I', 'J', 'JM', 'SF', 'SM', 'SS'],
        'base': ['CU', 'AL', 'ZN', 'PB', 'NI', 'SN', 'SI', 'AO', 'LC'],
        'energy_chem': ['SC', 'FU', 'LU', 'PG', 'TA', 'RU', 'NR', 'BR', 'PP',
                        'MA', 'EG', 'PF', 'V', 'EB', 'BU', 'PX', 'SP', 'SH',
                        'SA', 'FG', 'UR', 'L'],
        'agri': ['A', 'B', 'M', 'Y', 'P', 'C', 'CS', 'RM', 'OI', 'CF', 'SR',
                 'JD', 'LH', 'AP', 'CJ', 'PK'],
        'financial': ['IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS', 'TL'],
        'shipping': ['EC'],
    }
    # tail_risk_filter 相关性计算只用这些板块（排除shipping等单品种板块）
    VALID_TAIL_RISK_SECTORS = ('precious', 'black', 'base', 'energy_chem', 'agri', 'financial')

    def _filter_conflicting_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Cluster内信号一致性衰减 (仅极端冲突时).

        只有一致性极低 (<30%) 时才取消信号 (置零).
        """
        cfg = self._config
        if not cfg.enable_cluster_filter:
            return signals

        filtered = signals.copy().astype(float)

        for cluster_name, products in self.CLUSTERS.items():
            avail = [p for p in products if p in signals.columns]
            if len(avail) < 2:
                continue

            cluster_sigs = signals[avail]
            signs = np.sign(cluster_sigs)

            for p in avail:
                others = [o for o in avail if o != p]
                if not others:
                    continue
                my_sign = signs[p].values
                other_signs = signs[others].values
                agree = (other_signs == my_sign[:, np.newaxis]).sum(axis=1)
                consistency = agree / len(others)
                # 极端冲突 (<30%一致性) → 取消信号
                extreme_conflict = consistency < 0.3
                filtered.loc[extreme_conflict, p] = 0.0

        return filtered

    def _tail_risk_filter(self, weights: pd.DataFrame) -> pd.DataFrame:
        """板块间尾部风险保护: 板块相关性飙升时全局降仓.

        自适应阈值: 板块内品种越少 (噪声越大), 阈值越高.
        """
        cfg = self._config
        if not cfg.tail_risk_enabled:
            return weights

        # 构造板块收益序列, 统计每个板块的品种数 (只用 VALID_TAIL_RISK_SECTORS)
        ret = self._price_panel.pct_change(fill_method=None)
        sector_returns = {}
        sector_sizes = {}
        for sector_name, products in self.CLUSTERS.items():
            if sector_name not in self.VALID_TAIL_RISK_SECTORS:
                continue
            avail = [p for p in products if p in ret.columns]
            if avail:
                sector_returns[sector_name] = ret[avail].mean(axis=1)
                sector_sizes[sector_name] = len(avail)

        n_sectors = len(sector_returns)
        if n_sectors < 2:
            return weights

        # 自适应阈值: 板块内平均品种数越少 → 阈值越高 (噪声大)
        avg_products = np.mean(list(sector_sizes.values()))
        if avg_products <= 1.5:
            threshold = cfg.tail_risk_corr_threshold + 0.20  # 极少: 0.30→0.50
        elif avg_products <= 2.5:
            threshold = cfg.tail_risk_corr_threshold + 0.10  # 少量: 0.30→0.40
        else:
            threshold = cfg.tail_risk_corr_threshold          # 充足: 0.30

        sector_df = pd.DataFrame(sector_returns)

        # 计算板块间滚动平均相关性
        sectors = list(sector_df.columns)
        corr_sum = None
        n_pairs = 0
        for i in range(len(sectors)):
            for j in range(i + 1, len(sectors)):
                c = sector_df[sectors[i]].rolling(
                    cfg.tail_risk_corr_window, min_periods=20
                ).corr(sector_df[sectors[j]])
                corr_sum = c if corr_sum is None else corr_sum.add(c, fill_value=0)
                n_pairs += 1

        if corr_sum is None or n_pairs == 0:
            return weights

        avg_sector_corr = corr_sum / n_pairs

        # 板块相关性超过阈值 → 降仓
        scale = pd.Series(1.0, index=weights.index)
        high_corr = avg_sector_corr > threshold
        scale[high_corr] = cfg.tail_risk_scale

        return weights.mul(scale, axis=0)

    def _apply_weight_filters(self, weights: pd.DataFrame,
                              cfg: ModuleConfig) -> pd.DataFrame:
        """权重层过滤链: VolRiskFilter × 尾部风险 × 市场状态.

        _signal_to_lots (无硬限路径) 与 run() (leverage_hard_cap>0 路径) 共用,
        保证过滤器在任意执行路径下都生效.
        """
        out = weights
        if self._filter is not None:
            out = out * self._filter.coefficients(self._price_panel)
        # 尾部风险保护: 高相关性时全局降仓
        out = self._tail_risk_filter(out)
        # 市场状态过滤: 震荡市降仓
        if getattr(cfg, "regime_filter_enabled", False):
            regime_filter = MarketRegimeFilter(
                er_threshold=cfg.regime_adx_threshold,
                chop_coef=cfg.regime_chop_coef,
            )
            out = out * regime_filter.coefficients(self._price_panel)
        return out

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """组合层: vol_target × 权重过滤链 → 手数."""
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        weights = self._apply_weight_filters(weights, cfg)
        return self._weights_to_lots(weights, cfg)

    def _weights_to_lots(self, weights: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        equity = cfg.initial_cash
        # 全局杠杆系数: 等分资金基数放大 (线性放大暴露, 不改Sharpe)
        lev_k = getattr(cfg, "equity_leverage_k", 1.0)
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for product in weights.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            capital_per_product = signal_on[product] * equity * lev_k / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            # 单品种名义敞口上限: equity × max_leverage × coef
            # (ModuleConfig无此字段时保持旧行为 coef=0.25)
            coef = getattr(cfg, "max_product_notional_coef", 0.25)
            max_notional = equity * cfg.max_leverage * coef
            max_lots = (max_notional / notional_per_lot).fillna(0)
            target = target.clip(lower=-max_lots, upper=max_lots).round()
            lots[product] = target

        # Cluster级别敞口上限: 同cluster品种合计 ≤ equity × max_leverage × 0.5
        lots = self._cluster_limit(lots, cfg)
        return lots

    def _cluster_limit(self, lots: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """Cluster级别敞口上限, 防止同板块高乘数品种合计敞口过大.

        上限随 equity_leverage_k 等比放大, 保持线性放大语义.
        """
        clusters = self.CLUSTERS
        equity = cfg.initial_cash
        lev_k = getattr(cfg, "equity_leverage_k", 1.0)
        max_cluster_notional = equity * 1.0 * lev_k

        for cluster_name, products in clusters.items():
            avail = [p for p in products if p in lots.columns and p in self._price_panel.columns]
            if len(avail) < 2:
                continue
            notional = pd.DataFrame(0.0, index=lots.index, columns=lots.columns)
            for p in avail:
                spec = self._registry.get_or_default(p)
                price = self._price_panel[p]
                notional[p] = lots[p].abs() * price * spec.contract_multiplier
            cluster_total = notional[avail].sum(axis=1)
            exceed = cluster_total > max_cluster_notional
            if exceed.any():
                scale = (max_cluster_notional / cluster_total).clip(upper=1.0)
                for p in avail:
                    lots.loc[exceed, p] = (lots.loc[exceed, p] * scale[exceed]).round()
        return lots

    def load_panel(self) -> pd.DataFrame:
        panel = super().load_panel()
        for m in self._modules:
            m._price_panel = panel
            m._ohlc_map = self._ohlc_map
            m._vols = self._vols
            m._ois = self._ois
        return panel

    def _signals_raw(self) -> pd.DataFrame:
        """模块信号加权求和 (fixed/inv_vol/regression).

        未上市/无价格数据的品种信号统一置NaN:
        - 不污染 inv_vol 模块权重的行均值 (0列会稀释权重)
        - 不参与 n_active 资金等分 (signal_on=False)
        - 不产生虚假交易 (lots=0)
        """
        self.module_signals = {}
        sigs = []
        price_valid = self._price_panel.notna()
        for m in self._modules:
            sig = m._signals_raw()
            sig = sig.where(price_valid)
            self.module_signals[m.__class__.__name__] = sig
            sigs.append(sig)
        if self._config.module_weighting == "inv_vol":
            w = pd.concat([
                s.rolling(self._config.agg_window, min_periods=20).std().mean(axis=1)
                for s in sigs
            ], axis=1)
            inv = 1.0 / w.replace(0, np.nan)
            w_norm = inv.div(inv.sum(axis=1), axis=0).fillna(1 / len(sigs))
            combo = None
            for sig, wi in zip(sigs, [w_norm[c] for c in w_norm.columns]):
                combo = sig.mul(wi, axis=0) if combo is None else combo + sig.mul(wi, axis=0)
            return combo
        if self._config.module_weighting == "regression":
            return self._agg_module_regression(sigs)
        if self._config.module_weighting == "tvpr":
            return self._agg_module_tvpr(sigs)
        w = self._config.module_weights
        if len(w) != len(sigs):
            raise ValueError(
                f"module_weights 长度 {len(w)} != 模块数 {len(sigs)}, "
                "zip会静默丢弃多余信号, 请检查 ComboConfig.module_weights"
            )
        combo = None
        for sig, w_i in zip(sigs, w):
            combo = sig * w_i if combo is None else combo + sig * w_i
        return combo

    def _agg_module_regression(self, sigs: list[pd.DataFrame]) -> pd.DataFrame:
        """滚动 OLS 学习模块间权重: sig → 未来收益, 每 agg_update 天更新."""
        T = len(self._price_panel)
        cols = list(self._price_panel.columns)
        n_modules = len(sigs)
        # 构造 X: (T, N, K) — N=品种, K=模块数
        X = np.stack([
            s.reindex(columns=cols).values for s in sigs
        ], axis=-1)
        # Y: 下一日收益
        Y = self._price_panel.pct_change(fill_method=None).shift(-1).values
        win = self._config.agg_window
        agg_update = self._config.agg_update
        # 初始化等权
        W = np.full((T, n_modules), 1.0 / n_modules)
        for t in range(win, T, agg_update):
            x = X[t - win:t].reshape(-1, n_modules)
            y = Y[t - win:t].reshape(-1)
            mask = ~np.isnan(y) & ~np.any(np.isnan(x), axis=1)
            if mask.sum() > n_modules * 2:
                beta, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
                W[t:min(t + agg_update, T)] = beta
        # 加权求和
        combo = None
        for j in range(n_modules):
            w_j = pd.Series(W[:, j], index=self._price_panel.index)
            contrib = sigs[j].reindex(columns=cols).mul(w_j, axis=0)
            combo = contrib if combo is None else combo + contrib
        return combo

    def _agg_module_tvpr(self, sigs: list[pd.DataFrame]) -> pd.DataFrame:
        """TVPR 约束版模块间权重: L1+TV罚项防止过拟合."""
        from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
            expanding_window_tvpr,
        )
        cfg = self._config
        cols = list(self._price_panel.columns)
        n_modules = len(sigs)
        X = np.stack([
            s.reindex(columns=cols).values for s in sigs
        ], axis=-1)
        Y = self._price_panel.pct_change(fill_method=None).shift(-1)
        beta = expanding_window_tvpr(
            Y, X,
            lambda_tv=cfg.tvpr_lambda_tv,
            lambda_l1=cfg.tvpr_lambda_l1,
            min_history=cfg.tvpr_min_history,
            step=cfg.agg_update,
        )
        combo = None
        for j in range(n_modules):
            w_j = beta.iloc[:, j]
            contrib = sigs[j].reindex(columns=cols).mul(w_j, axis=0)
            combo = contrib if combo is None else combo + contrib
        return combo

    def _signals(self) -> pd.DataFrame:
        """组合方向 (-1/0/+1), 可选动量过滤 + cluster一致性过滤."""
        raw = self._signals_raw()
        cfg = self._config
        if cfg.momentum_filter_window > 0:
            mom = raw.rolling(cfg.momentum_filter_window, min_periods=10).mean()
            coef = pd.DataFrame(1.0, index=raw.index, columns=raw.columns)
            coef[mom < 0] = cfg.momentum_filter_coef
            raw = raw * coef
        if cfg.signal_ema_span > 0:
            raw = raw.ewm(span=cfg.signal_ema_span, min_periods=1).mean()
        # cluster过滤: 在sign()之前作用于连续信号, 衰减冲突信号
        raw = self._filter_conflicting_signals(raw)
        return np.sign(raw)

    def _leverage_cap_coef(self, sig: pd.DataFrame) -> pd.Series:
        """两遍法计算杠杆硬限系数 (T,).

        第一遍回测拿 NAV/仓位 → 逐日 gross/equity → 超限日等比缩放.
        系数因果 (只用第一遍 t 期数据), 第二遍用缩放权重重跑.
        """
        r1 = self._run_signal(sig)
        cap = self._config.leverage_hard_cap
        idx = pd.DatetimeIndex(r1.dates)
        equity = pd.Series(r1.equity, index=idx)
        prices = self._price_panel.reindex(idx)
        pos = pd.DataFrame(r1.positions, index=idx)
        gross = pd.DataFrame({
            p: pos[p].abs() * prices[p]
            * self._registry.get_or_default(p).contract_multiplier
            for p in prices.columns if p in pos.columns
        }).sum(axis=1)
        lev = gross / equity.replace(0, np.nan)
        coef = (cap / lev).clip(upper=1.0).fillna(1.0)
        return coef.clip(lower=0.0)

    def _dd_coef(self, nav: pd.Series) -> pd.Series:
        """NAV 回撤降杠杆系数 (T,) ∈ [0,1], 无前视.

        回撤超阈值 → 降杠杆系数持续生效, 直到 NAV 创新高才恢复 1.0.
        (触发后保持低仓, 避免回撤中反复抄底; 新高恢复, 滞后但稳健)
        """
        cfg = self._config
        thr = cfg.dd_filter_threshold
        coef_val = cfg.dd_filter_coef
        dd = nav / nav.cummax() - 1
        coef = pd.Series(1.0, index=nav.index)
        in_dd = False
        for t in range(len(nav)):
            if dd.iloc[t] < -thr:
                in_dd = True
            if in_dd and nav.iloc[t] == nav.iloc[:t + 1].max():
                in_dd = False
            if in_dd:
                coef.iloc[t] = coef_val
        return coef

    def run(self) -> CtaBacktestResult:
        """运行完整回测.

        leverage_hard_cap > 0 时两遍法: 第一遍无硬限拿 NAV → 缩放权重 →
        第二遍带杠杆硬限重跑 (总杠杆 ≤ cap).
        dd_filter_enabled 时两遍法: 第一遍拿 NAV → 回撤系数 → 第二遍乘系数重跑
        (回撤系数无前视, 只用第一遍 t 期及之前数据).
        """
        self.load_panel()
        sig = self._signals()
        cfg = self._config
        dd_coef = None
        if cfg.dd_filter_enabled:
            nav0 = self._run_signal(sig).nav
            dd_coef = self._dd_coef(nav0)
        if cfg.leverage_hard_cap > 0:
            sig = self._exec_sig(sig)
            weights = vol_target_positions(
                sig, self._price_panel, target_vol=cfg.target_vol,
                vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
            )
            weights = self._apply_weight_filters(weights, cfg)
            if dd_coef is not None:
                weights = weights.mul(dd_coef, axis=0)
            # 多轮迭代: 权重层等比缩放 → 手数层 floor 缩放, 保证总杠杆 ≤ cap
            coef = pd.Series(1.0, index=sig.index)
            lots = None
            for _ in range(3):
                lots = self._weights_to_lots(weights.mul(coef, axis=0), cfg)
                r_cur = self._run_with_lots(lots)
                idx = pd.DatetimeIndex(r_cur.dates)
                equity = pd.Series(r_cur.equity, index=idx)
                prices = self._price_panel.reindex(idx)
                pos = pd.DataFrame(r_cur.positions, index=idx)
                gross = pd.DataFrame({
                    p: pos[p].abs() * prices[p]
                    * self._registry.get_or_default(p).contract_multiplier
                    for p in prices.columns if p in pos.columns
                }).sum(axis=1)
                lev = gross / equity.replace(0, np.nan)
                if lev.max() <= cfg.leverage_hard_cap * 1.02:
                    return r_cur
                coef_new = coef * (cfg.leverage_hard_cap / lev).clip(upper=1.0).fillna(1.0)
                coef_new = coef_new.clip(lower=0.0)
                if (coef_new - coef).abs().max() < 1e-6:
                    break
                coef = coef_new
            # 手数层 floor 缩放兜底 (取整粒度无法由权重缩放消除)
            scale = (cfg.leverage_hard_cap / lev).clip(upper=1.0).fillna(1.0)
            lots_final = lots.mul(scale, axis=0).apply(np.floor)
            return self._run_with_lots(lots_final)
        if dd_coef is not None:
            sig = sig.mul(dd_coef, axis=0)
        return self._run_signal(sig)

    def run_factor(self, name: str) -> CtaBacktestResult:
        """组合策略按模块归因 (而非单因子)."""
        if not self.module_signals:
            self._signals_raw()
        return self._run_signal(np.sign(self.module_signals[name]))

    def module_correlation(self) -> pd.DataFrame:
        """模块级信号相关矩阵."""
        if not self.module_signals:
            self._signals_raw()
        frame = pd.DataFrame({
            name: s.to_numpy().ravel() for name, s in self.module_signals.items()
        })
        return frame.corr()
