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
from typing import Optional

import numpy as np
import pandas as pd

from ..data_loader import CtaDataLoader
from ..backtest.cta_engine import (
    CtaBacktestConfig,
    CtaBacktestResult,
    CtaStrategy,
    run_cta_backtest,
)
from ..backtest.cta_contract import CtaContractRegistry
from ..backtest.cta_risk import PositionLimitRisk
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
from ..factors.volatility import vol_target_positions
from ..factors.term_structure import TermSlopeFactor
from ..factors.basis_momentum import BasisLevelFactor
from ..factors.macro_link import DXYLinkFactor, MacroStateFactor, VIXLinkFactor
from ..factors.industrial_chain import (
    ChainMarginFactor, LeadLagFactor, MarginMomentumFactor, PriceRatioFactor,
)
from ..factors.cointegration import SpreadMeanReversionFactor

# 默认品种池: 全部有交易量品种 (2023-2025 日均成交量 ≥ 1 万手, 82→72)
# 覆盖全部产业链配方品种: 钢铁 RB/I/J/HC, 焦化 J/JM, PTA/聚酯 TA/PX/PF/EG,
# MTO PP/MA, 压榨 M/Y/A, 饲料 LH/C
DEFAULT_PRODUCTS = [
    "RB", "FG", "SA", "M", "TA", "MA", "V", "P", "RM", "HC",
    "LC", "Y", "AG", "FU", "I", "C", "UR", "OI", "JM", "SM",
    "PP", "CF", "L", "RU", "SI", "EB", "SR", "PS", "SP", "SH",
    "EG", "SF", "JD", "AO", "BU", "PX", "AP", "AL", "IM", "PF",
    "AU", "ZN", "SS", "NI", "PK", "A", "CS", "CJ", "EC", "B",
    "T", "TL", "PG", "IF", "CU", "TF", "IC", "LU", "SN", "BR",
    "NR", "LH", "PT", "PR", "SC", "PB", "TS", "IH", "PD", "LG",
    "J", "PL",
]


@dataclass
class ModuleConfig:
    """子策略/组合策略共用配置."""

    target_vol: float = 0.15        # 目标年化波动率
    vol_lookback: int = 20          # 波动率回看
    max_leverage: float = 3.0       # 最大杠杆
    initial_cash: float = 2_000_000.0
    commission_rate: float = 0.0001
    aggregation: str = "equal"      # 组内聚合: equal | inv_vol | regression | tvpr
    agg_window: int = 252           # inv_vol/regression/tvpr 滚动窗口
    agg_update: int = 21            # 权重更新频率 (天)
    tvpr_lambda_tv: float = 0.5     # 组内 TV-PR TV 罚项 (β 平滑度)
    tvpr_lambda_l1: float = 0.0     # 组内 TV-PR L1 罚项 (>0 组内稀疏过滤因子)
    tvpr_min_history: int = 252     # 组内 TV-PR 最少历史天数
    execution_lag: int = 1          # 信号执行滞后 (天): 1 = T+1 生产口径
                                    # (T 收盘算信号, T+1 收盘成交); 0 = 回测口径
    margin_call: bool = False       # True: 保证金不足拒绝开仓 (真实保证金率)
    continuous: bool = True         # 连续信号模式: 有 raw_value 的因子用
                                    # 连续值 + 滚动 z-score, 其余保持 -1/0/1
    cont_z_window: int = 252        # 连续信号 z-score 窗口
    cont_z_clip: float = 3.0        # 连续信号 z-score 截断


class _ModuleStrategy:
    """模块子策略执行层基类.

    子类定义 FACTORS: list[(因子名, 因子类, 输入类型)].
    信号 = 组内因子 compute() 信号 (-1/0/+1) 等权求和 → sign.
    """

    FACTORS: list[tuple[str, object, str]] = []
    CHAIN_MAP: dict[str, str] = {}   # 产业链因子名 → 目标品种/配对 ("A-B")

    def __init__(
        self,
        products: Optional[list[str]] = None,
        config: Optional[ModuleConfig] = None,
        data_loader: Optional[CtaDataLoader] = None,
        start: str = "2018-01-01",
        end: str = "2025-12-31",
    ):
        self._products = products or DEFAULT_PRODUCTS
        self._config = config or ModuleConfig()
        self._dl = data_loader or CtaDataLoader()
        self._start = start
        self._end = end
        self._registry = CtaContractRegistry()

        self._price_panel: pd.DataFrame | None = None
        self._ohlc_map: dict[str, pd.DataFrame] = {}
        self._vols: pd.DataFrame | None = None
        self._ois: pd.DataFrame | None = None
        self._spread_df: pd.DataFrame | None = None
        self._basis_df: pd.DataFrame | None = None
        self._vix: pd.Series | None = None
        self._dxy: pd.Series | None = None
        self.factor_signals: dict[str, pd.DataFrame] = {}

    # ── 数据 ────────────────────────────────────────────────────────────

    def load_panel(self) -> pd.DataFrame:
        """加载全部产品主力连续 → close/volume/OI 面板 + OHLC 映射."""
        closes, ohlcs, vols, ois = {}, {}, {}, {}
        for product in self._products:
            try:
                df = self._dl.load_continuous_main(
                    product, start=self._start, end=self._end,
                )
                if df is None or df.empty:
                    continue
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("trade_date").sort_index()
                closes[product] = df["close"]
                ohlcs[product] = df[["open", "high", "low", "close"]]
                vols[product] = df["volume"]
                ois[product] = df["open_interest"]
            except Exception as e:
                print(f"  [{product}] load failed: {e}")
        panel = pd.DataFrame(closes).sort_index().dropna(how="all")
        # 固定品种列 (新上市品种早期无数据 → NaN 列, 回测自动跳过)
        panel = panel.reindex(columns=self._products)
        self._price_panel = panel
        self._ohlc_map = ohlcs
        self._vols = pd.DataFrame(vols).reindex(index=panel.index, columns=self._products)
        self._ois = pd.DataFrame(ois).reindex(index=panel.index, columns=self._products)
        return panel

    # ── 信号层 ──────────────────────────────────────────────────────────

    def _ensure_external(self) -> None:
        """懒加载外部数据 (期限结构/基差/宏观序列)."""
        if self._spread_df is None:
            parts = []
            for product in self._products:
                ts = self._dl.load_term_structure(
                    product, start=self._start, end=self._end,
                )
                if not ts.empty:
                    parts.append(ts)
            self._spread_df = pd.concat(parts) if parts else pd.DataFrame()
        if self._basis_df is None:
            try:
                self._basis_df = self._dl.load_spot_basis(
                    start=self._start, end=min(self._end, "2025-07-31"),
                )
            except Exception:
                self._basis_df = pd.DataFrame()
        if self._vix is None:
            try:
                self._vix = self._dl.load_macro_series("vix")
            except Exception:
                self._vix = pd.Series(dtype=float)
        if self._dxy is None:
            try:
                self._dxy = self._dl.load_macro_series("dxy")
            except Exception:
                self._dxy = pd.Series(dtype=float)

    def _raw_value_panel(self, factor, input_type: str) -> pd.DataFrame:
        """因子 raw_value() → (T,N) 连续值面板 (continuous 模式)."""
        if input_type == "panel":
            out = factor.raw_value(self._price_panel)
        elif input_type == "ohlc":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                r = factor.raw_value(ohlc)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "ohlcv":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                df = ohlc.copy()
                df["volume"] = self._vols[p]
                r = factor.raw_value(df)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "series":
            cols = {}
            for p in self._price_panel.columns:
                r = factor.raw_value(self._price_panel[p])
                cols[p] = r if isinstance(r, pd.Series) else r.iloc[:, 0]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type in ("spread", "basis"):
            self._ensure_external()
            src = self._spread_df if input_type == "spread" else self._basis_df
            out = factor.raw_value(src)
            if isinstance(out, pd.Series):
                out = out.unstack()
            if input_type == "basis":
                out.columns = [str(c).upper() for c in out.columns]
                cols = [p for p in self._products if p in out.columns]
                return out.reindex(index=self._price_panel.index, columns=cols)
            return out.reindex(self._price_panel.index)
        elif input_type == "macro_vix":
            self._ensure_external()
            out = factor.raw_value(self._price_panel, self._vix)
        elif input_type == "macro_dxy":
            self._ensure_external()
            out = factor.raw_value(self._price_panel, self._dxy)
        else:
            raise ValueError(f"continuous mode unsupported for input_type: {input_type}")
        return out.reindex(self._price_panel.index)

    def _signal_panel(self, factor, input_type: str, name: str | None = None,
                      continuous: bool = False) -> pd.DataFrame:
        """因子 compute() 信号 → (T,N) 面板 (-1/0/+1).

        continuous=True 且因子有 raw_value 时, 返回连续值 (不 z-score,
        z-score 在 _signals_raw 聚合前统一处理).
        """
        if input_type == "panel":
            out = factor.compute(self._price_panel)
        elif input_type == "series":
            cols = {}
            for p in self._price_panel.columns:
                r = factor.compute(self._price_panel[p])
                cols[p] = r if isinstance(r, pd.Series) else r.iloc[:, 0]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "chain":
            out = factor.compute(self._price_panel)
            col = out.columns[0]
            target = self.CHAIN_MAP[name]
            panel = pd.DataFrame(0.0, index=self._price_panel.index,
                                 columns=self._price_panel.columns)
            if "-" in target:  # 配对: 分子=sig, 分母=-sig
                num, den = target.split("-")
                panel[num] = out[col]
                panel[den] = -out[col]
            else:
                panel[target] = out[col]
            return panel
        elif input_type == "ohlc":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                r = factor.compute(ohlc)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "ohlcv":
            cols = {}
            for p, ohlc in self._ohlc_map.items():
                df = ohlc.copy()
                df["volume"] = self._vols[p]
                r = factor.compute(df)
                cols[p] = r[r.columns[0]]
            return pd.DataFrame(cols).reindex(self._price_panel.index)
        elif input_type == "price_vol":
            out = factor.compute(self._price_panel, self._vols)
        elif input_type == "price_oi":
            out = factor.compute(self._price_panel, self._ois)
        elif input_type == "vols":
            out = factor.compute(self._vols)
        elif input_type == "ois":
            out = factor.compute(self._ois)
        elif input_type == "spread":
            self._ensure_external()
            out = factor.compute(self._spread_df)
            if isinstance(out, pd.Series):
                out = out.unstack()
            return out.reindex(self._price_panel.index)
        elif input_type == "basis":
            self._ensure_external()
            out = factor.compute(self._basis_df)
            if isinstance(out, pd.Series):
                out = out.unstack()
            out.columns = [str(c).upper() for c in out.columns]
            cols = [p for p in self._products if p in out.columns]
            return out.reindex(index=self._price_panel.index, columns=cols)
        elif input_type == "macro_vix":
            self._ensure_external()
            out = factor.compute(self._price_panel, self._vix)
        elif input_type == "macro_dxy":
            self._ensure_external()
            out = factor.compute(self._price_panel, self._dxy)
        else:
            raise ValueError(f"unknown input_type: {input_type}")
        return out.reindex(self._price_panel.index)

    def _signals_raw(self) -> pd.DataFrame:
        """模块内因子信号 → 模块信号 (按 aggregation 聚合).

        equal:      等权平均 (Σsig/n, ∈ [-1,1], 修复组间尺度不均衡)
        inv_vol:    滚动窗口信号波动率倒数加权 (w_k ∝ 1/σ_k)
        regression: 滚动窗口 OLS 学权重 (sig → 未来收益, w = 系数, 无前视)
        """
        if self._price_panel is None:
            self.load_panel()
        self.factor_signals = {}
        panels = {}
        cont = self._config.continuous
        for name, cls, input_type in self.FACTORS:
            factor = cls()
            use_raw = False
            if cont and hasattr(factor, "raw_value"):
                probe = factor.raw_value
                # 仅真连续值 (raw != compute) 才连续化; 事件型 raw=compute
                # 保持 -1/0/1 离散, 避免 z-score 损坏
                use_raw = (probe.__func__ is not factor.compute.__func__)
            if use_raw:
                sig = self._raw_value_panel(factor, input_type)
                # 连续值: 时序滚动 z-score + clip (无前视)
                z = self._cont_zscore(sig)
                self.factor_signals[name] = z
                panels[name] = z.fillna(0)
            else:
                sig = self._signal_panel(factor, input_type, name)
                # 统一行列: 缺列品种 (早期无数据) → NaN → 求和前 fillna(0)
                sig = sig.reindex(index=self._price_panel.index,
                                  columns=self._price_panel.columns)
                self.factor_signals[name] = sig
                panels[name] = sig.fillna(0)
        agg = self._config.aggregation
        if agg == "equal":
            return sum(panels.values()) / len(panels)
        if agg in ("inv_vol", "regression"):
            return self._agg_weighted(panels)
        if agg == "tvpr":
            return self._agg_tvpr(panels)
        raise ValueError(f"unknown aggregation: {agg}")

    def _cont_zscore(self, raw: pd.DataFrame) -> pd.DataFrame:
        """连续信号时序滚动 z-score (逐品种, 无前视) + clip."""
        cfg = self._config
        win = cfg.cont_z_window
        mu = raw.rolling(win, min_periods=20).mean()
        sd = raw.rolling(win, min_periods=20).std()
        z = (raw - mu) / sd.replace(0, np.nan)
        # 归一化到 [-1,1]: 与离散 ±1 信号同尺度, 聚合时公平混合
        return z.clip(-cfg.cont_z_clip, cfg.cont_z_clip) / cfg.cont_z_clip

    def _agg_tvpr(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """组内 TV-PR 聚合: 因子信号 → 未来收益 的时变权重 β_t.

        X (T,N,K) = 组内因子信号, Y (T,N) = 品种未来 1 日收益,
        β_t = expanding TV-PR (ADMM, 无前视), λ_l1>0 可在组内稀疏过滤因子.
        模块信号 = Σ_k β_t,k × X_t,i,k.
        """
        from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
            tvpr_estimator,
        )
        cfg = self._config
        names = list(panels)
        cols = list(self._price_panel.columns)
        X = np.stack([
            panels[n].reindex(columns=cols).values for n in names
        ], axis=-1)
        Y = self._price_panel.pct_change().shift(-1)
        beta = tvpr_estimator(
            Y, X,
            lambda_tv=cfg.tvpr_lambda_tv,
            lambda_l1=cfg.tvpr_lambda_l1,
            method="expanding",
            min_history=cfg.tvpr_min_history,
            step=cfg.agg_update,
        )
        combo = np.einsum("tni,ti->tn", X, beta.values)
        return pd.DataFrame(combo, index=self._price_panel.index, columns=cols)

    def _agg_weights(self, panels: dict[str, pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
        """滚动窗口因子权重 (T, K): inv_vol = 1/σ, regression = OLS 系数.

        无前视: t 期权重只用 [t-window, t) 数据, 每 agg_update 天更新一次.
        """
        names = list(panels)
        T = len(self._price_panel)
        W = np.full((T, len(names)), 1.0 / len(names))
        cols = list(self._price_panel.columns)
        unified = [panels[n].reindex(columns=cols, fill_value=0.0) for n in names]
        stack = np.stack([p.values for p in unified], axis=-1)
        win = self._config.agg_window
        if self._config.aggregation == "inv_vol":
            for t in range(win, T, self._config.agg_update):
                x = stack[t - win:t].reshape(-1, len(names))
                std = np.nanstd(x, axis=0)
                safe = np.where(std > 1e-9, std, np.nan)
                w = np.where(np.isnan(safe), 0.0, 1.0 / safe)
                w = w / w.sum() if w.sum() > 0 else np.full(len(names), 1.0 / len(names))
                W[t:min(t + self._config.agg_update, T)] = w
            return W, names
        if self._config.aggregation == "regression":
            Y = self._price_panel.pct_change().shift(-1).values
            for t in range(win, T, self._config.agg_update):
                x = stack[t - win:t].reshape(-1, len(names))
                y = Y[t - win:t].reshape(-1)
                mask = ~np.isnan(y) & ~np.any(np.isnan(x), axis=1)
                if mask.sum() > len(names) * 2:
                    beta, *_ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
                    W[t:min(t + self._config.agg_update, T)] = beta
            return W, names
        raise ValueError(f"unknown aggregation: {self._config.aggregation}")

    def _agg_weighted(self, panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """按 (T,K) 权重矩阵加权合成模块信号."""
        W, names = self._agg_weights(panels)
        out = None
        for j, n in enumerate(names):
            contrib = panels[n].mul(W[:, j], axis=0)
            out = contrib if out is None else out + contrib
        return out

    def _signals(self) -> pd.DataFrame:
        """模块信号 → 方向 (-1/0/+1)."""
        return np.sign(self._signals_raw())

    # ── 仓位与回测 ──────────────────────────────────────────────────────

    def _positions(self) -> pd.DataFrame:
        """方向 × 波动率目标 → 目标手数 (当日活跃品种等分资金)."""
        cfg = self._config
        signals = self._signals()
        return self._signal_to_lots(signals, cfg)

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        weights = vol_target_positions(
            signals,
            self._price_panel,
            target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback,
            max_leverage=cfg.max_leverage,
        )
        equity = cfg.initial_cash
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        for product in signals.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            capital_per_product = signal_on[product] * equity / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            lots[product] = target
        return lots

    def _exec_sig(self, sig: pd.DataFrame) -> pd.DataFrame:
        """应用执行滞后: T 收盘算信号 → T+lag 成交 (生产口径)."""
        lag = self._config.execution_lag
        return sig.shift(lag).fillna(0) if lag > 0 else sig

    def _run_signal(self, sig: pd.DataFrame) -> CtaBacktestResult:
        """给定方向面板直接跑回测 (组合/归因共用).

        统一应用 execution_lag (T+1 生产口径).
        """
        cfg = self._config
        lots = self._signal_to_lots(self._exec_sig(sig), cfg)
        return self._run_with_lots(lots)

    def _run_with_lots(self, lots: pd.DataFrame) -> CtaBacktestResult:
        """给定目标手数面板直接跑回测 (方向变化才调仓)."""
        cfg = self._config
        price_panel = self._price_panel

        class _PanelStrategy(CtaStrategy):
            def __init__(self, pos_df):
                self._pos = pos_df
                self._last = {}

            def target_positions(self, date, data):
                if date not in self._pos.index:
                    return {}
                row = self._pos.loc[date]
                targets = {
                    p: float(row[p]) for p in self._pos.columns if abs(row[p]) > 1e-9
                }
                changed = False
                for p in set(targets) | set(self._last):
                    def _sgn(x):
                        return 1 if x > 0 else (-1 if x < 0 else 0)
                    new_s = _sgn(targets.get(p, 0))
                    old_s = _sgn(self._last.get(p, 0))
                    if new_s != old_s:
                        changed = True
                        break
                if not changed:
                    return {}
                self._last = targets
                return targets

        cols = [p for p in lots.columns if p in price_panel.columns]
        prices_map = {p: price_panel[p].tolist() for p in cols}
        products_map = {p: p for p in prices_map}
        risk = PositionLimitRisk(products=products_map, registry=self._registry)
        return run_cta_backtest(
            dates=list(price_panel.index),
            prices=prices_map,
            products=products_map,
            strategy=_PanelStrategy(lots),
            config=CtaBacktestConfig(
                initial_cash=cfg.initial_cash, commission_rate=cfg.commission_rate,
                margin_call=cfg.margin_call,
            ),
            registry=self._registry,
            risk=risk,
        )

    def run(self) -> CtaBacktestResult:
        """运行完整回测 (方向变化才调仓)."""
        self.load_panel()
        return self._run_signal(self._signals())

    # ── 归因 ────────────────────────────────────────────────────────────

    def run_factor(self, name: str) -> CtaBacktestResult:
        """单因子独立回测 (因子级归因)."""
        if name not in self.factor_signals:
            self._signals_raw()
        return self._run_signal(np.sign(self.factor_signals[name]))

    def signal_correlation(self) -> pd.DataFrame:
        """因子级信号相关矩阵 (列两两 Pearson, 逐日逐品种配对)."""
        if not self.factor_signals:
            self._signals_raw()
        frame = pd.DataFrame({
            name: s.to_numpy().ravel() for name, s in self.factor_signals.items()
        })
        return frame.corr()

    @property
    def products(self) -> list[str]:
        return self._products


# ── 波动率风险过滤层 ───────────────────────────────────────────────────────

class VolRiskFilter:
    """波动率风险过滤层: 输出全局仓位系数 (T,) ∈ [0,1].

    市场级危机过滤 (业界标准, 实验增益): 全市场平均 HV 滚动 z-score
    超过阈值 → 全局降仓. 10 品种实验: z>1.0 → 0.5x 使 Sharpe 1.109→1.190.
    单品种级过滤 (HV 逐品种 z-score) 实验无增益, 默认关闭.
    """

    def __init__(
        self,
        z_window: int = 252,
        mkt_z_threshold: float = 1.0,
        mkt_coef: float = 0.5,
        product_z_threshold: float = 2.0,
        product_coef: float = 0.5,
        apply_per_product: bool = False,
    ):
        self.z_window = z_window
        self.mkt_z_threshold = mkt_z_threshold
        self.mkt_coef = mkt_coef
        self.product_z_threshold = product_z_threshold
        self.product_coef = product_coef
        self.apply_per_product = apply_per_product

    def coefficients(self, price_panel: pd.DataFrame) -> pd.DataFrame:
        """返回 (T,N) 仓位系数矩阵."""
        from ..factors.volatility import HVFactor
        hv = HVFactor().compute(price_panel)
        mu = hv.rolling(self.z_window, min_periods=20).mean()
        sd = hv.rolling(self.z_window, min_periods=20).std()
        hv_z = (hv - mu) / sd.replace(0, np.nan)
        coef = pd.DataFrame(1.0, index=price_panel.index, columns=price_panel.columns)
        mkt_z = hv_z.mean(axis=1)
        coef.loc[mkt_z > self.mkt_z_threshold, :] = self.mkt_coef
        if self.apply_per_product:
            coef[hv_z > self.product_z_threshold] = self.product_coef
        return coef


# ── 四个方向子策略 ─────────────────────────────────────────────────────────

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
        ("overnight_ret", OvernightReturnFactor, "ohlc"),
        ("intraday_ret", IntradayReturnFactor, "ohlc"),
        ("overnight_intraday_div", OvernightIntradayDivergenceFactor, "ohlc"),
        ("long_body", LongBodyFactor, "ohlc"),
        ("vortex", VortexFactor, "ohlc"),
        ("chandelier", ChandelierExitFactor, "ohlc"),
    ]


class MeanReversionModuleStrategy(_ModuleStrategy):
    """均值回归组: 超买超卖/价格位置/K线反转/价差回归类因子."""

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
        chain_margin_pp  → PP   (MTO 利润)
        margin_momentum_m → M   (压榨利润动量)
        price_ratio_au_cu → AU-CU 配对 (比价高估 → 空 AU 多 CU)
        lead_lag_i_rb    → RB   (铁矿传导顺畅 → 做多下游螺纹)
    """

    FACTORS = [
        ("chain_margin_rb", lambda: ChainMarginFactor("steel_profit_rb"), "chain"),
        ("chain_margin_pp", lambda: ChainMarginFactor("mto_profit"), "chain"),
        ("margin_momentum_m", lambda: MarginMomentumFactor("crush_profit"), "chain"),
        ("price_ratio_au_cu", lambda: PriceRatioFactor("AU", "CU"), "chain"),
        ("lead_lag_i_rb", lambda: LeadLagFactor("I", "RB"), "chain"),
    ]

    CHAIN_MAP = {
        "chain_margin_rb": "RB",
        "chain_margin_pp": "PP",
        "margin_momentum_m": "M",
        "price_ratio_au_cu": "AU-CU",
        "lead_lag_i_rb": "RB",
    }


# ── 组合策略 ───────────────────────────────────────────────────────────────

@dataclass
class ComboConfig(ModuleConfig):
    """组合策略配置: 模块权重 (趋势/均值回归/量能/事件/期限基差/宏观/产业链) + 过滤层.

    权重策略: 默认等权 1/7 (7 组), 简单稳健, 不做样本内调优
    (调优结果仅记录于 docs/94, 不设为默认).
    """

    module_weights: tuple[float, float, float, float, float, float, float] = (
        1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7,
    )
    module_weighting: str = "fixed"  # 组间加权: fixed | inv_vol (波动率倒数)
    filter_enabled: bool = True      # 市场级波动危机过滤 (Sharpe 1.109→1.190)
    momentum_filter_window: int = 0  # 信号动量过滤窗口 (天), 0=关闭 (docs/94 §10.7)
    momentum_filter_coef: float = 0.0  # 动量为负时仓位系数 (0=全空仓, 1=不过滤)
    dd_filter_enabled: bool = False  # NAV 回撤降杠杆: 回撤超阈值 → 全局降杠杆
    dd_filter_threshold: float = 0.05  # 回撤触发阈值 (5% = 0.05)
    dd_filter_coef: float = 0.5     # 触发后杠杆系数 (0.5 = 半仓)
    leverage_hard_cap: float = 0.0   # 组合总杠杆硬上限 (逐日 gross/equity), 0=关闭


class FactorModuleComboStrategy(_ModuleStrategy):
    """组合策略: 四模块方向信号加权 → 单一资金池 vol targeting.

    模块权重默认等权 0.25×4 (可调, 趋势主导可改为 0.4/0.2/0.3/0.1).
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
            MeanReversionModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            VolumeFlowModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            EventCalendarModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            TermBasisModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            MacroModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            IndustrialChainModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
        ]
        self.module_signals: dict[str, pd.DataFrame] = {}
        self._filter = VolRiskFilter() if self._config.filter_enabled else None

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """组合层: vol_target 权重 × 波动率过滤系数 → 手数."""
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if self._filter is not None:
            weights = weights * self._filter.coefficients(self._price_panel)
        return self._weights_to_lots(weights, cfg)

    def _weights_to_lots(self, weights: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        equity = cfg.initial_cash
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=weights.index, columns=weights.columns)
        for product in weights.columns:
            price = self._price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            capital_per_product = signal_on[product] * equity / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            lots[product] = target
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
        """模块信号加权求和 (fixed: 固定权重; inv_vol: 滚动波动率倒数)."""
        self.module_signals = {}
        sigs = []
        for m in self._modules:
            sig = m._signals_raw()
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
        w = self._config.module_weights
        combo = None
        for sig, w_i in zip(sigs, w):
            combo = sig * w_i if combo is None else combo + sig * w_i
        return combo

    def _signals(self) -> pd.DataFrame:
        """组合方向 (-1/0/+1), 可选信号动量过滤 (见 ComboConfig)."""
        raw = self._signals_raw()
        cfg = self._config
        if cfg.momentum_filter_window > 0:
            mom = raw.rolling(cfg.momentum_filter_window, min_periods=10).mean()
            coef = pd.DataFrame(1.0, index=raw.index, columns=raw.columns)
            coef[mom < 0] = cfg.momentum_filter_coef
            raw = raw * coef
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
            if self._filter is not None:
                weights = weights * self._filter.coefficients(self._price_panel)
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
