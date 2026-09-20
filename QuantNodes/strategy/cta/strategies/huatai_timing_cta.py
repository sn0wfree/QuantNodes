# coding=utf-8
"""华泰时序CTA方法论落地: 期货实证 (IF + AU + T).

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).

标的池:  IF (沪深300股指期货) | AU (沪金) | T (10年国债期货)
       对应文章原 ETF 标的 (510300 / 518880 / 511010) 的期货映射,
       保持"权益 - 商品 - 债券"三足鼎立结构.

完整流程:
    1. 加载主力连续面板 (continuous_main_{product}_daily)
    2. ARGarchEstimator → 滚动估计 (μ, ρ, σ)
    3. compute_tsi_panel → TSI + 95% 置信区间
    4. classify_regime_from_estimator → 状态分类 (trend_up/trend_down/range)
    5. 三层开仓信号:
        - TREND: 自适应均线交叉 (EMA fast/slow)
        - RANGE: 14 日 RSI 超超反转 (30/70)
    6. 四层退出机制 (evaluate_exit 一站式并联):
        - 技术止损: ATR 吊灯 (按品种动态乘数)
        - 逻辑证伪: 状态反转确认 (3 日)
        - 时间强制: max_holding_days + 未盈利
        - 全局风控: 账户回撤熔断

绝对收益导向: 评价标准是卡玛比 (提升持有体验), 非超额收益.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_risk import PositionLimitRisk
from ..data_loader import CtaDataLoader
from ..exits import (
    DrawdownCircuitBreaker,
    evaluate_exit,
)
from ..exits.range_exit import range_rsi_exit_check
from ..factors.indicators import rsi as compute_rsi
from ..factors.volatility import vol_target_positions
from ..factors.composite import (
    CrossDimensionalResonanceFactor,
    MultiPeriodResonanceFactor,
    VolatilityRegimeResonanceFactor,
)
from ..regime import (
    ARGarchEstimator,
    STATE_RANGE,
    STATE_TREND_DOWN,
    STATE_TREND_UP,
    classify_regime_from_estimator,
    compute_tsi_panel,
    kaufman_adaptive_window,
)

# 21品种板块配置 (7板块 × 3-5品种)
# 来源: 华泰金工《时序CTA方法论综述》扩展 + 行业流动性筛选
# 排除 IF (Walk-Forward 窗口差异 4.7, 过拟合风险高)
HUATAI_PRODUCTS = [
    # 股指 (2): IC, IH (排除 IF)
    "IC", "IH",
    # 国债 (3): 低波动, 小 ATR
    "T", "TF", "TS",
    # 贵金属 (2): 中高波动
    "AU", "AG",
    # 有色 (4): 中波动, NI 波动大
    "CU", "AL", "ZN", "NI",
    # 黑色 (2): 中高波动
    "RB", "I",
    # 能化 (3): 中高波动
    "SC", "MA", "TA",
    # 农产品 (4): 中波动
    "M", "Y", "P", "SR",
]

# 推荐组合 (Tier1+2: 高/中可靠性品种)
RECOMMENDED_PORTFOLIO = [
    "AU",   # 贵金属: Calmar=4.266, stable=4
    "NI",   # 有色: Calmar=2.364, stable=2
    "AL",   # 有色: Calmar=1.418, stable=2
    "SC",   # 能化: Calmar=1.125, stable=2
    "T",    # 国债: Calmar=1.075, stable=3
]

# ATR 乘数配置: 按板块 × 品种特性
# 文章推荐: IF=4.5, AU=3.0, T=1.5
# 扩展逻辑: 高波动 → 大乘数 (宽松止损), 低波动 → 小乘数 (紧凑止损)
DEFAULT_CHANDLER_MULT: dict[str, float] = {
    # 股指: 日内 2-3% 波动 → 大 ATR
    "IF": 4.5, "IC": 5.0, "IH": 4.5,
    # 国债: 日内 0.3% 波动 → 小 ATR
    "T": 1.5, "TF": 1.5, "TS": 2.0,
    # 贵金属: AG 波动 > AU
    "AU": 3.0, "AG": 4.0,
    # 有色: NI 镍波动大
    "CU": 3.5, "AL": 3.0, "ZN": 3.5, "NI": 5.0,
    # 黑色: I 铁矿石波动大
    "RB": 3.5, "I": 4.0,
    # 能化: SC 原油波动大
    "SC": 4.5, "MA": 3.5, "TA": 3.5,
    # 农产品: SR 白糖波动小, P 棕榈大
    "M": 3.0, "Y": 3.0, "P": 3.5, "SR": 2.5,
    # 扩展品种 (Phase A2)
    "pb": 3.5, "sn": 4.0,  # 有色
    "hc": 3.5, "sf": 3.0, "sm": 3.0, "j": 4.0, "jm": 4.0,  # 黑色
    "bu": 3.5, "eg": 4.0, "eb": 4.0, "v": 3.5, "l": 3.5, "pp": 3.5,  # 能化
    "cf": 3.5, "rm": 3.5, "oi": 3.5,  # 农产品
}

# 扩展品种池 (15原 + 15新 = 30品种)
EXTENDED_PRODUCTS = HUATAI_PRODUCTS + [
    # 有色 (2)
    "pb", "sn",
    # 黑色 (5)
    "hc", "sf", "sm", "j", "jm",
    # 能化 (6)
    "bu", "eg", "eb", "v", "l", "pp",
    # 农产品 (3)
    "cf", "rm", "oi",
]


@dataclass
class HuataiTimingConfig:
    """华泰时序CTA策略配置 (期货实证)."""

    # ── 参数估计 (regime/) ──
    base_window: int = 60
    use_adaptive: bool = True
    h_min: int = 21
    h_max: int = 250
    er_window: int = 30
    confirm_window: int = 10
    rho_window: int = 60
    use_garch: bool = False

    # ── 开仓信号 ──
    ema_fast_min: int = 5
    ema_fast_max: int = 20
    ema_slow: int = 60
    use_ema_slope: bool = False       # True: 用EMA斜率替代绝对位置
    rsi_window: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # ── 退出: ATR 吊灯乘数 (按品种, 文章推荐 + 扩展) ──
    chandelier_mult: dict = field(default_factory=lambda: DEFAULT_CHANDLER_MULT.copy())
    chandelier_window: int = 22
    chandelier_atr_window: int = 14
    tr_clip_enabled: bool = True      # TR 5%-95% 分位截断 (研报原文要求)

    # ── 退出: 震荡交易 RSI 回落退出 (研报原文) ──
    # 空单(开仓时RSI超买): RSI 回落至 60 以内平空
    # 多单(开仓时RSI超卖): RSI 回升至 40 以上平多
    range_rsi_exit_enabled: bool = True   # False: range 交易也走吊灯 (v1 行为)
    range_rsi_exit_long: float = 40.0
    range_rsi_exit_short: float = 60.0
    # 逆 μ 反转做空已被证伪 (20品种×2020-2026 归因: -0.415%, t=-2.53,
    # AU 2025 牛市做空爆仓的根因)。开启后 range 做空需 μ<0 (顺势做空)。
    range_short_require_mu_neg: bool = True
    # ρ>0 是否为趋势必要条件。False 适用于"阶梯式牛市" (μ 强但日度自相关
    # 为负, 如 AU 2022-2026 +164% 行情 ρ>0 仅 2.8% → 原规则全程误判 range)。
    # 20品种实测: 平均 Calmar -0.038 → +0.042 (金属/TA 大幅改善, 少数震荡
    # 品种如 I/TF 恶化, 品种级差异留给 walk-forward 标定)
    rho_required: bool = False
    # ── 做空开关 (Phase 10c 实证: 99.1% PnL 来自做多, 空信号仅 48/5741;
    #    long-only Calmar 1.901 > 全信号 1.866 — 空头为负贡献, 默认关闭) ──
    # False: TREND_DOWN / RANGE(RSI>70) 不产生空信号, 且跳过 short_gate 计算
    # True: 恢复 v9 完整多空行为 (研究用)
    allow_short: bool = False
    # ── 做空宏观闸门 (归因: trend_down 做空 -0.153%, t=-3.50, 组合亏损主因) ──
    # 商品广度 (close>MA250 品种占比) >= threshold → 结构性牛市 → 禁 trend_down 做空
    # mode: breadth / external / both_and / both_or (见 regime.macro_gate)
    short_gate_enabled: bool = True
    short_gate_mode: str = "both_and"  # 实测与 external 等价 (+0.123), 阈值 0.5-0.7 稳定
    breadth_ma: int = 250
    breadth_threshold: float = 0.6
    # ── 短空闸门各宏观维度开关 (Phase 10 扩展 CS+COR) ──
    short_gate_use_vix: bool = True
    short_gate_use_crb: bool = True
    short_gate_use_cs: bool = True   # 中美利差
    short_gate_use_cor: bool = True  # 金油比相关性
    # ── 做多宏观闸门 (与 short_gate 对称, 用于衰退/避险期拒做多) ──
    long_gate_enabled: bool = False   # 默认关闭 (向后兼容)
    long_gate_mode: str = "external"  # external/breadth/off (见 regime.macro_long_gate)
    long_gate_use_vix: bool = True
    long_gate_use_crb: bool = True
    long_gate_use_cs: bool = True
    long_gate_use_cor: bool = True

    # ── 再入场冷却期 (防止止损-开仓循环, 研报原文 §"趋势的可交易性"痛点) ──
    cooldown_days: int = 5

    # ── 持仓模式 ──
    # True: 研报原文持仓优先 (持仓日只判退出, 趋势死扛/反转RSI回落)
    # False: 信号驱动 (entry 信号消失即平仓, 实测整体 Calmar 更优, 默认)
    hold_priority: bool = False

    # ── 退出: 时间 ──
    max_holding_days: int = 60
    time_min_pnl: float = 0.0

    # ── 退出: 全局风控 ──
    drawdown_threshold: float = -0.15
    drawdown_recovery: float = 1.0   # 兼容参数 (已被观察期语义取代)
    observation_days: int = 60       # 熔断触发后的观察期天数 (期满解除+峰值重置)

    # ── 仓位 ──
    target_vol: float = 0.15
    vol_lookback: int = 20
    max_leverage: float = 3.0
    # 低波品种杠杆上限 (超出此值的杠杆将被截断, 解决成本侵蚀问题)
    # 品种年化波动 3% × 杠杆 3x = 权益放大 9 倍, 成本也被放大 → 净亏
    # 品种级 cap: 确保 |effective_leverage| ≤ product_max_leverage[product]
    product_max_leverage: dict = field(default_factory=lambda: {
        "T": 1.5, "TF": 1.5, "TS": 1.5,  # 国债期货: 低波, 杠杆 cap 1.5x
        "IC": 1.5, "IH": 1.5,             # 股指: 高波动, cap 1.5x
    })
    # 组合总杠杆上限 (逐日 sum(|notional|) / equity ≤ portfolio_max_leverage)
    # 0 = 不限制。设置后逐日等比缩放所有品种仓位使总杠杆不超限。
    portfolio_max_leverage: float = 3.0

    # ── 回测基础设施 ──
    initial_cash: float = 2_000_000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.001

    @staticmethod
    def presets() -> dict[str, "HuataiTimingConfig"]:
        """参数预设集, 用于敏感性分析.

        Returns:
            dict[name → HuataiTimingConfig]
        """
        return {
            # 文章原参数
            "default": HuataiTimingConfig(),
            # 状态确认窗口缩短 (10→5), 状态切换更灵敏
            "responsive": HuataiTimingConfig(confirm_window=5),
            # 目标波动率放大 (0.15→0.20), 提升杠杆利用率
            "levered": HuataiTimingConfig(target_vol=0.20),
            # 最长持仓天数延长 (60→120), 避免截断长趋势
            "patient": HuataiTimingConfig(max_holding_days=120),
            # 三者综合
            "aggressive": HuataiTimingConfig(
                confirm_window=5,
                target_vol=0.20,
                max_holding_days=120,
            ),
            # EMA斜率替代绝对位置 (诊断发现: IF trend_up中fast>slow仅26%)
            "slope": HuataiTimingConfig(use_ema_slope=True),
            # 斜率 + 缩短确认窗口
            "slope_responsive": HuataiTimingConfig(
                use_ema_slope=True, confirm_window=5,
            ),
        }


class HuataiTimingCTAStrategy:
    """华泰时序CTA方法论: 期货实证策略."""

    def __init__(
        self,
        products: Optional[list[str]] = None,
        config: Optional[HuataiTimingConfig] = None,
        data_loader: Optional[CtaDataLoader] = None,
        start: str = "2020-01-01",
        end: str = "2026-05-31",
    ):
        self._products = products or HUATAI_PRODUCTS
        self._config = config or HuataiTimingConfig()
        self._dl = data_loader or CtaDataLoader()
        self._start = start
        self._end = end
        self._registry = CtaContractRegistry()

        # 加载标定参数 (用于 regime-aware 参数热切换)
        from ..optimization.adaptive import load_21_product_calibration, get_all_calibrated
        load_21_product_calibration()
        self._calibrated_params = get_all_calibrated()

        self._price_panel: pd.DataFrame | None = None
        self._ohlc_panel: pd.DataFrame | None = None
        self._states: pd.DataFrame | None = None
        self._mu: pd.DataFrame | None = None
        self._entry: pd.DataFrame | None = None
        self._dates: list[pd.Timestamp] | None = None

    # ── 数据 ────────────────────────────────────────────────────────────

    def load_panel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """加载 IF/AU/T 主力连续 OHLC 面板.

        Returns:
            (price_panel, ohlc_panel) 两 DataFrame 索引/列一致.
        """
        prices: dict[str, pd.Series] = {}
        ohlcs: dict[str, pd.DataFrame] = {}
        for product in self._products:
            try:
                df = self._dl.load_continuous_main(
                    product, start=self._start, end=self._end,
                )
            except Exception as e:
                print(f"  [{product}] load failed: {e}")
                continue
            if df is None or df.empty:
                continue
            df = df.set_index("trade_date").copy()
            df.index = pd.to_datetime(df.index)
            prices[product] = df["close"]
            ohlcs[product] = df[["open", "high", "low", "close"]]
        price_panel = pd.DataFrame(prices).sort_index().dropna(how="all")
        ohlc_panel = pd.concat(ohlcs, axis=1).sort_index().dropna(how="all")
        valid_dates = price_panel.dropna(how="any").index
        self._price_panel = price_panel.loc[valid_dates]
        self._ohlc_panel = ohlc_panel.loc[valid_dates]
        self._dates = list(self._price_panel.index)
        return self._price_panel, self._ohlc_panel

    # ── 状态估计 ────────────────────────────────────────────────────────

    def _estimate_states(self) -> pd.DataFrame:
        cfg = self._config
        if self._price_panel is None:
            self.load_panel()
        estimator = ARGarchEstimator(
            base_window=cfg.base_window,
            use_adaptive=cfg.use_adaptive,
            h_min=cfg.h_min, h_max=cfg.h_max, er_window=cfg.er_window,
            use_garch=cfg.use_garch,
        )
        params = estimator.estimate(self._price_panel)
        panel = compute_tsi_panel(params)
        self._mu = params["mu"]
        self._states = classify_regime_from_estimator(
            params, panel,
            confirm_window=cfg.confirm_window,
            rho_window=cfg.rho_window,
            rho_required=cfg.rho_required,
        )
        # 做空宏观闸门 (归因: trend_down 做空 t=-3.50)
        # allow_short=False 时空头信号不产生, 跳过宏观序列加载与广度计算
        self._short_gate = None
        if cfg.short_gate_enabled and cfg.allow_short:
            from ..regime.macro_gate import macro_short_gate
            self._short_gate = macro_short_gate(
                self._price_panel,
                mode=cfg.short_gate_mode,
                breadth_threshold=cfg.breadth_threshold,
                breadth_ma=cfg.breadth_ma,
                use_vix=cfg.short_gate_use_vix,
                use_crb_proxy=cfg.short_gate_use_crb,
                use_credit_spread=cfg.short_gate_use_cs,
                use_copper_oil_ratio=cfg.short_gate_use_cor,
            )
        # 做多宏观闸门 (与 short_gate 对称)
        self._long_gate = None
        if cfg.long_gate_enabled:
            from ..regime.macro_long_gate import macro_long_gate
            self._long_gate = macro_long_gate(
                self._price_panel,
                mode=cfg.long_gate_mode,
                breadth_threshold=cfg.breadth_threshold,
                breadth_ma=cfg.breadth_ma,
                use_vix=cfg.long_gate_use_vix,
                use_crb_proxy=cfg.long_gate_use_crb,
                use_credit_spread=cfg.long_gate_use_cs,
                use_copper_oil_ratio=cfg.long_gate_use_cor,
            )
        return self._states

    # ── 开仓信号 ────────────────────────────────────────────────────────

    def _entry_signals(self) -> pd.DataFrame:
        cfg = self._config
        if self._price_panel is None or self._states is None:
            self.load_panel()
            self._estimate_states()

        out = pd.DataFrame(
            0.0, index=self._price_panel.index, columns=self._price_panel.columns,
        )
        slow_ema = self._price_panel.ewm(span=cfg.ema_slow, adjust=False).mean()
        for col in self._price_panel.columns:
            close = self._price_panel[col]
            er_series = kaufman_adaptive_window(
                close, cfg.ema_fast_min, cfg.ema_fast_max, cfg.er_window,
            )
            states_col = self._states[col]
            n = len(close)
            for t in range(n):
                if t < cfg.ema_slow:
                    continue
                state = states_col.iloc[t]
                h_raw = er_series.iloc[t]
                h = int(h_raw) if pd.notna(h_raw) else cfg.ema_fast_min
                h = max(cfg.ema_fast_min, min(h, cfg.ema_fast_max))
                fast = close.iloc[t - h + 1: t + 1].mean()
                slow = slow_ema[col].iloc[t]
                use_slope = cfg.use_ema_slope
                if state == STATE_TREND_UP:
                    # 做多宏观闸门 (衰退/避险期拒做多)
                    long_gate_ok = (
                        self._long_gate is None
                        or bool(self._long_gate[col].iloc[t])
                    )
                    if not long_gate_ok:
                        sig = 0.0
                    elif use_slope:
                        fast_prev = close.iloc[t - h: t].mean() if t >= h + 1 else fast
                        sig = 1.0 if fast > fast_prev else 0.0
                    else:
                        sig = 1.0 if fast > slow else 0.0
                elif state == STATE_TREND_DOWN:
                    # 做空宏观闸门 (归因 t=-3.50): 结构性牛市禁空
                    gate_ok = (
                        self._short_gate is None
                        or bool(self._short_gate[col].iloc[t])
                    )
                    if not gate_ok:
                        sig = 0.0
                    elif use_slope:
                        fast_prev = close.iloc[t - h: t].mean() if t >= h + 1 else fast
                        sig = -1.0 if fast < fast_prev else 0.0
                    else:
                        sig = -1.0 if fast < slow else 0.0
                elif state == STATE_RANGE:
                    rsi_v = compute_rsi(close.iloc[:t + 1], cfg.rsi_window).iloc[t]
                    if rsi_v < cfg.rsi_oversold:
                        sig = 1.0
                    elif rsi_v > cfg.rsi_overbought:
                        # 逆 μ 反转做空被证伪 (归因 t=-2.53): 需 μ<0 顺势做空
                        if cfg.range_short_require_mu_neg:
                            mu_today = self._mu[col].iloc[t] if self._mu is not None else np.nan
                            sig = -1.0 if (pd.notna(mu_today) and mu_today < 0) else 0.0
                        else:
                            sig = -1.0
                    else:
                        sig = 0.0
                else:
                    sig = 0.0
                out.iat[t, out.columns.get_loc(col)] = sig
        self._entry = out
        return out

    def _entry_signals_multi_period(
        self,
        windows: tuple[int, ...] = (5, 20, 60),
        weights: tuple[float, ...] | None = None,
        min_agree: int | None = None,
    ) -> pd.DataFrame:
        """多周期共振开仓信号 (状态分支 + EMA 共振确认).

        与 `_entry_signals()` 差异 (用于消融研究):
            - TREND_UP/DOWN: 额外要求共振因子在该 bar 与原 EMA 交叉方向一致才输出.
            - RANGE: 同 RSI 反转 (不受共振影响).

        研报依据: §"开仓信号: 第三层 - 复合信号层" 多周期共振.

        Args:
            windows: 共振因子周期 (默认 (5, 20, 60))
            weights: 可选加权 (默认 None 等权多数决)
            min_agree: 等权多数决阈值 (默认 len(windows)//2 + 1)
        """
        return self._entry_signals_with_resonance(
            MultiPeriodResonanceFactor(
                windows=windows, weights=weights, min_agree=min_agree,
            ),
        )

    def _entry_signals_vol_regime(
        self,
        windows: tuple[int, ...] = (5, 20, 60),
        percentile: float = 0.75,
    ) -> pd.DataFrame:
        """波动率体制感知: 仅在 vol > 历史 percentile 时启用共振.

        设计动机: 解决 multi_period 在低波动率震荡市频繁反向的问题.
        """
        return self._entry_signals_with_resonance(
            VolatilityRegimeResonanceFactor(
                windows=windows, percentile=percentile,
            ),
        )

    def _entry_signals_cross_dim(self) -> pd.DataFrame:
        """跨维度共振: EMA 方向 + 动量符号 + 波动率压缩 三维度投票.

        设计动机: 跨维度能提供更丰富的信号特征, 缓解多周期共振的
        "快速反转早期被过滤"问题 (动量与波动率压缩对反转有反应).
        """
        return self._entry_signals_with_resonance(CrossDimensionalResonanceFactor())

    def _entry_signals_with_resonance(self, resonance_factor) -> pd.DataFrame:
        """共振因子驱动的统一开仓信号入口.

        Args:
            resonance_factor: 任何输出 (T, N) {-1, 0, +1} 的共振因子.
        """
        cfg = self._config
        if self._price_panel is None or self._states is None:
            self.load_panel()
            self._estimate_states()

        resonance = resonance_factor.compute(self._price_panel)

        out = pd.DataFrame(
            0.0, index=self._price_panel.index, columns=self._price_panel.columns,
        )
        slow_ema = self._price_panel.ewm(span=cfg.ema_slow, adjust=False).mean()
        for col in self._price_panel.columns:
            close = self._price_panel[col]
            er_series = kaufman_adaptive_window(
                close, cfg.ema_fast_min, cfg.ema_fast_max, cfg.er_window,
            )
            states_col = self._states[col]
            res_col = resonance[col]
            n = len(close)
            for t in range(n):
                if t < cfg.ema_slow:
                    continue
                state = states_col.iloc[t]
                h_raw = er_series.iloc[t]
                h = int(h_raw) if pd.notna(h_raw) else cfg.ema_fast_min
                h = max(cfg.ema_fast_min, min(h, cfg.ema_fast_max))
                fast = close.iloc[t - h + 1: t + 1].mean()
                slow = slow_ema[col].iloc[t]
                res = res_col.iloc[t]
                use_slope = cfg.use_ema_slope
                if state == STATE_TREND_UP:
                    if use_slope:
                        fast_prev = close.iloc[t - h: t].mean() if t >= h + 1 else fast
                        sig = 1.0 if (fast > fast_prev and res >= 1) else 0.0
                    else:
                        sig = 1.0 if (fast > slow and res >= 1) else 0.0
                elif state == STATE_TREND_DOWN:
                    # 做空宏观闸门 (归因 t=-3.50): 结构性牛市禁空
                    gate_ok = (
                        self._short_gate is None
                        or bool(self._short_gate[col].iloc[t])
                    )
                    if not gate_ok:
                        sig = 0.0
                    elif use_slope:
                        fast_prev = close.iloc[t - h: t].mean() if t >= h + 1 else fast
                        sig = -1.0 if (fast < fast_prev and res <= -1) else 0.0
                    else:
                        sig = -1.0 if (fast < slow and res <= -1) else 0.0
                elif state == STATE_RANGE:
                    rsi_v = compute_rsi(close.iloc[:t + 1], cfg.rsi_window).iloc[t]
                    if rsi_v < cfg.rsi_oversold:
                        sig = 1.0
                    elif rsi_v > cfg.rsi_overbought:
                        # 逆 μ 反转做空被证伪 (归因 t=-2.53): 需 μ<0 顺势做空
                        if cfg.range_short_require_mu_neg:
                            mu_today = self._mu[col].iloc[t] if self._mu is not None else np.nan
                            sig = -1.0 if (pd.notna(mu_today) and mu_today < 0) else 0.0
                        else:
                            sig = -1.0
                    else:
                        sig = 0.0
                else:
                    sig = 0.0
                out.iat[t, out.columns.get_loc(col)] = sig
        self._entry = out
        return out

    # ── 目标仓位 ────────────────────────────────────────────────────────

    def _positions(self) -> pd.DataFrame:
        cfg = self._config
        if self._entry is None:
            self._entry_signals()
        # allow_short=False: 裁剪全部空信号 (覆盖 _entry_signals 及共振变体)
        entry = self._entry if cfg.allow_short else self._entry.clip(lower=0.0)
        weights = vol_target_positions(
            entry, self._price_panel,
            target_vol=cfg.target_vol, vol_lookback=cfg.vol_lookback,
            max_leverage=cfg.max_leverage,
        )
        equity = cfg.initial_cash
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for product in weights.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional = price * spec.contract_multiplier
            capital = signal_on[product] * equity / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital / notional
            lots[product] = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            # 品种级杠杆上限 (vol cap): 解决低波品种成本侵蚀
            # 原理: lots × notional / (equity/n_active) = effective_leverage
            #       cap lots 使 effective_leverage ≤ product_max_leverage[product]
            max_lev = cfg.product_max_leverage.get(product)
            if max_lev is not None:
                equity_per_product = equity / n_active
                lots_cap = (max_lev * equity_per_product / notional).round()
                lots_cap = lots_cap.replace([np.inf, -np.inf], np.nan).fillna(0)
                lots[product] = lots[product].clip(
                    upper=lots_cap, lower=-lots_cap,
                )

        # ── 组合总杠杆上限: sum(|notional|) / equity ≤ portfolio_max_leverage ──
        pml = cfg.portfolio_max_leverage
        if pml > 0:
            total_notional = pd.DataFrame(0.0, index=lots.index, columns=lots.columns)
            for product in lots.columns:
                price = self._price_panel[product]
                spec = self._registry.get_or_default(product)
                notional_per_lot = price * spec.contract_multiplier
                total_notional[product] = lots[product].abs() * notional_per_lot
            gross = total_notional.sum(axis=1)
            lev = gross / equity
            scale = (pml / lev).clip(upper=1.0).fillna(1.0)
            # 逐日等比缩放所有品种手数
            lots = lots.multiply(scale, axis=0).round()

        return lots

    # ── 回测 ────────────────────────────────────────────────────────────

    def run(self) -> CtaBacktestResult:
        cfg = self._config
        self.load_panel()
        self._estimate_states()
        self._entry_signals()
        positions = self._positions()

        cols = [p for p in self._products if p in self._price_panel.columns]
        prices_map = {p: self._price_panel[p].tolist() for p in cols}
        products_map = {p: p for p in prices_map}
        dates = self._dates

        chandelier_kw = {
            p: {
                "window": cfg.chandelier_window,
                "atr_window": cfg.chandelier_atr_window,
                "mult": cfg.chandelier_mult.get(p, 3.0),
                "tr_clip": cfg.tr_clip_enabled,
            }
            for p in cols
        }
        cb = DrawdownCircuitBreaker(
            threshold=cfg.drawdown_threshold,
            recovery_ratio=cfg.drawdown_recovery,
            observation_days=cfg.observation_days,
        )

        strategy = _HuataiPanelStrategy(
            positions=positions,
            ohlc=self._ohlc_panel,
            states=self._states,
            chandelier_kwargs=chandelier_kw,
            breaker=cb,
            max_holding_days=cfg.max_holding_days,
            calibrated_params=self._calibrated_params,
            range_rsi_exit_enabled=cfg.range_rsi_exit_enabled,
            range_rsi_exit_long=cfg.range_rsi_exit_long,
            range_rsi_exit_short=cfg.range_rsi_exit_short,
            rsi_window=cfg.rsi_window,
            cooldown_days=cfg.cooldown_days,
            hold_priority=cfg.hold_priority,
        )

        risk = PositionLimitRisk(products=products_map, registry=self._registry)
        result = run_cta_backtest(
            dates=dates,
            prices=prices_map,
            products=products_map,
            strategy=strategy,
            config=CtaBacktestConfig(
                initial_cash=cfg.initial_cash,
                commission_rate=cfg.commission_rate,
            ),
            registry=self._registry,
            risk=risk,
        )
        result.metrics["drawdown_circuit_breaker_engaged"] = cb.is_active
        return result

    # ── 便捷属性 ────────────────────────────────────────────────────────

    @property
    def products(self) -> list[str]:
        return self._products

    @property
    def states(self) -> pd.DataFrame | None:
        return self._states

    @property
    def entry_signals(self) -> pd.DataFrame | None:
        return self._entry


class _HuataiPanelStrategy(CtaStrategy):
    """华泰策略回调: 每个 bar 复用 evaluate_exit 判定四层退出.

    支持 regime-aware 参数热切换:
        - 追踪每个品种的当前状态
        - 当状态变化时, 从 calibrated_params 中查找对应参数
        - 使用新参数决定退出逻辑
    """

    def __init__(
        self,
        positions: pd.DataFrame,
        ohlc: pd.DataFrame,
        states: pd.DataFrame,
        chandelier_kwargs: dict,
        breaker: DrawdownCircuitBreaker,
        max_holding_days: int,
        calibrated_params: dict | None = None,
        range_rsi_exit_enabled: bool = True,
        range_rsi_exit_long: float = 40.0,
        range_rsi_exit_short: float = 60.0,
        rsi_window: int = 14,
        cooldown_days: int = 5,
        hold_priority: bool = False,
    ):
        self._pos = positions
        self._ohlc = ohlc
        self._states = states
        self._chand = chandelier_kwargs
        self._cb = breaker
        self._max_hold = max_holding_days
        self._calibrated = calibrated_params or {}
        self._range_rsi_exit_enabled = range_rsi_exit_enabled
        self._range_rsi_exit_long = range_rsi_exit_long
        self._range_rsi_exit_short = range_rsi_exit_short
        self._rsi_window = rsi_window
        self._cooldown_days = cooldown_days
        self._hold_priority = hold_priority
        self._last: dict[str, float] = {}
        self._entry_day: dict[str, int] = {}
        self._entry_state: dict[str, str] = {}  # product → 开仓时的状态 (决定退出机制)
        self._exit_day: dict[str, int] = {}  # product → 退出日 idx (冷却期判定)
        self._current_regime: dict[str, str] = {}  # product → regime
        self._regime_changed: dict[str, bool] = {}  # product → regime_changed
        # 预计算 RSI (退出判定用)
        self._rsi_cache: dict[str, pd.Series] = {}

    def _get_rsi(self, product: str) -> pd.Series:
        """获取品种 RSI 序列 (懒加载缓存)."""
        if product not in self._rsi_cache:
            close = self._ohlc[product]["close"]
            self._rsi_cache[product] = compute_rsi(close, self._rsi_window)
        return self._rsi_cache[product]

    def _hold_days(self, product: str, current_idx: int) -> int:
        start = self._entry_day.get(product)
        return max(0, current_idx - start) if start is not None else 0

    def _track_entry(self, current_idx: int, targets: dict[str, float]) -> None:
        """基于实际决策 (targets) 记录开仓日 / 开仓状态 / 退出日 / 清理."""
        for product in self._pos.columns:
            old_qty = self._last.get(product, 0)
            new_qty = targets.get(product, 0)
            if old_qty == 0 and abs(new_qty) > 1e-9:
                self._entry_day[product] = current_idx
                # 记录开仓时的状态 (决定退出机制: trend→吊灯, range→RSI回落)
                self._entry_state[product] = str(
                    self._states[product].iloc[current_idx],
                )
                self._exit_day.pop(product, None)
            elif old_qty != 0 and abs(new_qty) <= 1e-9:
                self._entry_day.pop(product, None)
                self._entry_state.pop(product, None)
                self._exit_day[product] = current_idx  # 冷却期起点

    def _update_regime(self, product: str, idx: int) -> None:
        """更新品种状态, 并检测状态切换."""
        if product not in self._states.columns:
            return
        new_regime = str(self._states[product].iloc[idx])
        old_regime = self._current_regime.get(product, "")
        if new_regime != old_regime:
            self._regime_changed[product] = True
        else:
            self._regime_changed[product] = False
        self._current_regime[product] = new_regime

    def _get_chandelier_mult(self, product: str) -> float:
        """获取品种的吊灯乘数 (支持标定参数热切换)."""
        # 优先使用标定参数
        if product in self._calibrated:
            regime = self._current_regime.get(product, "global")
            regime_params = self._calibrated[product].get(regime, {})
            if "chandelier_mult" in regime_params:
                return regime_params["chandelier_mult"]
        # fallback 到默认
        return self._chand.get(product, {}).get("mult", 3.0)

    def _exit_triggered(self, product: str, i: int, cur_qty: float) -> bool:
        """趋势持仓退出判定 (研报原文 §"离场信号"): 吊灯 (TR截断) + 状态反转证伪 + 时间."""
        # 时间强制退出
        if self._hold_days(product, i) >= self._max_hold:
            return True

        level0 = self._ohlc.columns.get_level_values(0)
        product_ohlc = self._ohlc[product] if product in level0 else None
        if product_ohlc is None:
            return False

        pos_series = pd.Series(0.0, index=self._ohlc.index)
        for j in range(i):
            pos_series.iloc[j] = self._last.get(product, 0)
        pos_series.iloc[i] = cur_qty
        risk_halt = pd.Series(False, index=self._ohlc.index)

        exits = evaluate_exit(
            position=pos_series,
            ohlc=product_ohlc,
            states=self._states[[product]],
            unrealized_pnl=pd.Series(0.0, index=self._ohlc.index),
            max_holding_days=self._max_hold,
            chandelier_kwargs=self._chand[product],
            logical_confirm_window=3,
            risk_halt=risk_halt,
            mode="parallel",
        )
        return bool(exits.iloc[i])

    def _range_exit_triggered(
        self, product: str, i: int, cur_qty: float, entry_today: float,
    ) -> bool:
        """反转持仓退出: 信号消失 OR RSI 回落 (研报原文) OR 时间.

        反转交易高胜率低赔率 → 信号中断或 RSI 回落兑现即离场 (先到者平).
        """
        if self._hold_days(product, i) >= self._max_hold:
            return True
        cur_sign = 1 if cur_qty > 0 else (-1 if cur_qty < 0 else 0)
        if cur_sign == 0:
            return True
        # 信号消失: 当日无信号或方向反转 → 反转逻辑失效
        if abs(entry_today) <= 1e-9:
            return True
        entry_sign = 1 if entry_today > 0 else -1
        if entry_sign != cur_sign:
            return True
        # RSI 回落获利了结 (研报原文)
        rsi_today = float(self._get_rsi(product).iloc[i])
        return range_rsi_exit_check(
            cur_sign, rsi_today,
            exit_long=self._range_rsi_exit_long,
            exit_short=self._range_rsi_exit_short,
        )

    def target_positions(self, date, data) -> dict[str, float]:
        """按 hold_priority 分流:
            True  → 研报原文持仓优先状态机 (持仓日只判退出)
            False → 信号驱动 (entry 信号消失即平仓, 实测默认更优)
        """
        if date not in self._pos.index:
            return {}
        i = self._pos.index.get_loc(date)

        # 更新每个品种的状态 (regime-aware)
        for product in self._pos.columns:
            self._update_regime(product, i)

        if not self._hold_priority:
            # ── 信号驱动模式 (默认): entry 驱动持仓 + 退出判定仅在有信号日生效 ──
            raw_targets = {
                p: float(self._pos[p].iloc[i])
                for p in self._pos.columns
                if abs(self._pos[p].iloc[i]) > 1e-9
            }
            # 冷却期: 退出后 cooldown_days 内禁止再入场
            raw_targets = {
                p: q for p, q in raw_targets.items()
                if not (
                    (last_exit := self._exit_day.get(p)) is not None
                    and (i - last_exit) < self._cooldown_days
                )
            }
            # 全局回撤熔断: 清空全部目标
            if self._cb.is_active:
                raw_targets = {}
            # 退出判定 (仅对当日有信号的品种)
            for product in list(raw_targets.keys()):
                hold_d = self._hold_days(product, i)
                if hold_d >= self._max_hold:
                    raw_targets.pop(product, None)
                    continue
                level0 = self._ohlc.columns.get_level_values(0)
                product_ohlc = self._ohlc[product] if product in level0 else None
                if product_ohlc is None:
                    continue
                entry_state = self._entry_state.get(product, "")
                if entry_state == STATE_RANGE and self._range_rsi_exit_enabled:
                    cur_sign = 1 if self._last.get(product, 0) > 0 else (
                        -1 if self._last.get(product, 0) < 0 else 0
                    )
                    if cur_sign != 0:
                        rsi_today = float(self._get_rsi(product).iloc[i])
                        if range_rsi_exit_check(
                            cur_sign, rsi_today,
                            exit_long=self._range_rsi_exit_long,
                            exit_short=self._range_rsi_exit_short,
                        ):
                            raw_targets.pop(product, None)
                else:
                    pos_series = pd.Series(0.0, index=self._ohlc.index)
                    pos_series.iloc[i] = raw_targets[product]
                    for j in range(i):
                        pos_series.iloc[j] = self._last.get(product, 0)
                    risk_halt = pd.Series(False, index=self._ohlc.index)
                    exits = evaluate_exit(
                        position=pos_series,
                        ohlc=product_ohlc,
                        states=self._states[[product]],
                        unrealized_pnl=pd.Series(0.0, index=self._ohlc.index),
                        max_holding_days=self._max_hold,
                        chandelier_kwargs=self._chand[product],
                        logical_confirm_window=3,
                        risk_halt=risk_halt,
                        mode="parallel",
                    )
                    if bool(exits.iloc[i]):
                        raw_targets.pop(product, None)

            # 显式平仓指令: _last 持仓但当日无信号/被退出 → 必须下发 qty=0
            # (缺失此步会导致 broker 孤儿仓位: 策略认为已平, 引擎从未下单,
            #  仓位无人管理死扛到爆仓 — AU 2021-2025 爆仓的根因)
            for product, cur in self._last.items():
                if abs(cur) > 1e-9 and product not in raw_targets:
                    raw_targets[product] = 0.0

            self._track_entry(i, raw_targets)
            if raw_targets == self._last:
                return {}
            self._last = raw_targets
            return raw_targets

        # ── 持仓优先模式 (研报原文, hold_priority=True) ──
        # 全局回撤熔断: 平掉全部持仓, 且不开新仓
        if self._cb.is_active:
            targets = {p: 0.0 for p, v in self._last.items() if abs(v) > 1e-9}
            self._track_entry(i, targets)
            self._last = {p: 0.0 for p in self._last}
            return targets if targets else {}

        targets: dict[str, float] = {}
        for product in self._pos.columns:
            cur = self._last.get(product, 0)
            if abs(cur) > 1e-9:
                # ── 持仓日: 只判退出, 不考虑入场信号 ──
                entry_state = self._entry_state.get(product, "")
                if entry_state == STATE_RANGE and self._range_rsi_exit_enabled:
                    entry_today = float(self._pos[product].iloc[i])
                    exit_now = self._range_exit_triggered(product, i, cur, entry_today)
                else:
                    exit_now = self._exit_triggered(product, i, cur)
                if exit_now:
                    targets[product] = 0.0
                # 未触发 → 维持持仓
            else:
                # ── 空仓日: 只判入场信号 ──
                # 冷却期: 退出后 cooldown_days 内禁止再入场 (防止损-开仓循环)
                last_exit = self._exit_day.get(product)
                if last_exit is not None and (i - last_exit) < self._cooldown_days:
                    continue
                q = float(self._pos[product].iloc[i])
                if abs(q) > 1e-9:
                    targets[product] = q

        # 补全"维持"项: 与 _last 对齐以便正确检测变化
        for product, cur in self._last.items():
            if abs(cur) > 1e-9 and product not in targets:
                targets[product] = cur

        self._track_entry(i, targets)
        if targets == self._last:
            return {}
        self._last = targets
        return targets

    def post_step(self, nav: float) -> None:
        """回测引擎每步回调: 更新 DrawdownCircuitBreaker."""
        self._cb.update(nav)
