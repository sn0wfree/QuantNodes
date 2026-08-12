# coding=utf-8
"""多因子 CTA 策略 — 三模块子策略 + 组合 (策略 v2).

设计依据: docs/94-多因子CTA策略设计.md (业界主流: 连续信号加权 + 单一资金池)

结构:
    信号层: 因子 raw_value → winsorize → 滚动 z-score (clip ±3)
            事件型因子 (Chandelier/Squeeze) 保持 0/±1, 不标准化
    合成层: 模块内等权求和 → 模块连续信号; 因子级信号面板保留 (归因)
    执行层: sign(模块信号) → vol_target_positions (单一资金池) → 手数 → 回测

用法:
    from QuantNodes.strategy.cta.strategies.factor_modules import (
        OscillatorModuleStrategy, FactorModuleComboStrategy,
    )

    strat = OscillatorModuleStrategy(products=["AU", "RB"], start="2018-01-01")
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
from ..factors.mean_reversion import HalfLifeFactor, SqueezeFactor
from ..factors.oscillators import (
    CCIFactor, MACDFactor, StochasticFactor, WilliamsRFactor,
)
from ..factors.trend_advanced import (
    AroonFactor, ChandelierExitFactor, LinearSlopeFactor, TRIXFactor, VortexFactor,
)
from ..factors.volatility import vol_target_positions

# 默认中池 10 品种 (设计文档 §一)
DEFAULT_PRODUCTS = ["AU", "RB", "M", "CU", "IF", "I", "T", "TA", "MA", "C"]

Z_WINDOW = 252        # 滚动 z-score 窗口
Z_CLIP = 3.0          # z-score 截断
WINSOR_P = 0.01       # winsorize 分位


@dataclass
class ModuleConfig:
    """子策略/组合策略共用配置."""

    target_vol: float = 0.15        # 目标年化波动率
    vol_lookback: int = 20          # 波动率回看
    max_leverage: float = 3.0       # 最大杠杆
    initial_cash: float = 2_000_000.0
    commission_rate: float = 0.0001
    z_window: int = Z_WINDOW
    z_clip: float = Z_CLIP
    winsor_p: float = WINSOR_P


class _ModuleStrategy:
    """模块子策略执行层基类.

    子类只需定义 FACTORS: list[(因子名, 因子实例, 输入类型, 是否事件型)]
    输入类型: "panel" → compute 于 close 面板; "ohlc" → 逐品种 OHLC.
    """

    FACTORS: list[tuple[str, object, str, bool]] = []

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
        self.factor_signals: dict[str, pd.DataFrame] = {}

    # ── 数据 ────────────────────────────────────────────────────────────

    def load_panel(self) -> pd.DataFrame:
        """加载全部产品主力连续 → close 面板 + OHLC 映射."""
        closes, ohlcs = {}, {}
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
            except Exception as e:
                print(f"  [{product}] load failed: {e}")
        panel = pd.DataFrame(closes).sort_index().dropna(how="all")
        self._price_panel = panel
        self._ohlc_map = ohlcs
        return panel

    # ── 信号层 ──────────────────────────────────────────────────────────

    def _raw_panel(self, factor, input_type: str) -> pd.DataFrame:
        """因子 raw_value → (T,N) 面板.

        panel: 直接作用于 close 面板; ohlc: 逐品种 OHLC 合并.
        """
        if input_type == "panel":
            return factor.raw_value(self._price_panel)
        cols = {}
        for p, ohlc in self._ohlc_map.items():
            out = factor.raw_value(ohlc)
            cols[p] = out[out.columns[0]]
        return pd.DataFrame(cols).reindex(self._price_panel.index)

    @staticmethod
    def _normalize(raw: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        """winsorize → 滚动 z-score → clip (逐品种时序, 无前视)."""
        lo = raw.rolling(cfg.z_window, min_periods=20).quantile(cfg.winsor_p)
        hi = raw.rolling(cfg.z_window, min_periods=20).quantile(1 - cfg.winsor_p)
        clipped = raw.clip(lo, hi, axis=0)
        mu = clipped.rolling(cfg.z_window, min_periods=20).mean()
        sd = clipped.rolling(cfg.z_window, min_periods=20).std()
        z = (clipped - mu) / sd.replace(0, np.nan)
        return z.clip(-cfg.z_clip, cfg.z_clip)

    def _signals_raw(self) -> pd.DataFrame:
        """模块内因子 z-score 等权求和 (事件型直接 0/±1)."""
        if self._price_panel is None:
            self.load_panel()
        cfg = self._config
        self.factor_signals = {}
        summed = pd.DataFrame(0.0, index=self._price_panel.index,
                              columns=self._price_panel.columns)
        for name, factor, input_type, event in self.FACTORS:
            raw = self._raw_panel(factor, input_type)
            if event:
                self.factor_signals[name] = raw
                summed += raw
            else:
                z = self._normalize(raw, cfg)
                self.factor_signals[name] = z
                summed += z.fillna(0)
        return summed

    def _signals(self) -> pd.DataFrame:
        """模块连续信号 → 方向 (-1/0/+1)."""
        return np.sign(self._signals_raw())

    # ── 仓位与回测 ──────────────────────────────────────────────────────

    def _positions(self) -> pd.DataFrame:
        """方向 × 波动率目标 → 目标手数 (当日活跃品种等分资金)."""
        cfg = self._config
        signals = self._signals()
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

    def run(self) -> CtaBacktestResult:
        """运行完整回测 (方向变化才调仓)."""
        price_panel = self.load_panel()
        positions = self._positions()

        cols = [p for p in self._products if p in price_panel.columns]
        prices_map = {p: price_panel[p].tolist() for p in cols}
        products_map = {p: p for p in prices_map}

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

        risk = PositionLimitRisk(products=products_map, registry=self._registry)
        result = run_cta_backtest(
            dates=list(price_panel.index),
            prices=prices_map,
            products=products_map,
            strategy=_PanelStrategy(positions),
            config=CtaBacktestConfig(
                initial_cash=self._config.initial_cash,
                commission_rate=self._config.commission_rate,
            ),
            registry=self._registry,
            risk=risk,
        )
        return result

    # ── 归因 ────────────────────────────────────────────────────────────

    def run_factor(self, name: str) -> CtaBacktestResult:
        """单因子独立回测 (因子级归因)."""
        if name not in self.factor_signals:
            self._signals_raw()
        z = self.factor_signals[name]
        sig = np.sign(z)
        return self._run_signal(sig)

    def _run_signal(self, sig: pd.DataFrame) -> CtaBacktestResult:
        """给定方向面板直接跑回测 (组合/归因共用)."""
        cfg = self._config
        price_panel = self._price_panel
        weights = vol_target_positions(
            sig, price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        equity = cfg.initial_cash
        signal_on = (weights.abs() > 1e-9).astype(float)
        n_active = signal_on.sum(axis=1).clip(lower=1)
        lots = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
        for product in sig.columns:
            price = price_panel[product]
            spec = self._registry.get_or_default(product)
            notional_per_lot = price * spec.contract_multiplier
            capital_per_product = signal_on[product] * equity / n_active
            with np.errstate(divide="ignore", invalid="ignore"):
                target = weights[product] * capital_per_product / notional_per_lot
            target = target.replace([np.inf, -np.inf], np.nan).fillna(0).round()
            lots[product] = target

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

        cols = [p for p in sig.columns if p in price_panel.columns]
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
            ),
            registry=self._registry,
            risk=risk,
        )

    def signal_correlation(self) -> pd.DataFrame:
        """因子级 z-score 信号相关矩阵 (列两两 Pearson)."""
        if not self.factor_signals:
            self._signals_raw()
        frame = pd.DataFrame({
            name: s.to_numpy().ravel() for name, s in self.factor_signals.items()
        })
        return frame.corr()

    @property
    def products(self) -> list[str]:
        return self._products


# ── 三个模块子策略 ─────────────────────────────────────────────────────────

class OscillatorModuleStrategy(_ModuleStrategy):
    """振荡模块: MACD / Stochastic / CCI / WilliamsR."""

    FACTORS = [
        ("macd", MACDFactor(), "panel", False),
        ("stochastic", StochasticFactor(), "ohlc", False),
        ("cci", CCIFactor(), "ohlc", False),
        ("williams_r", WilliamsRFactor(), "ohlc", False),
    ]


class TrendAdvancedModuleStrategy(_ModuleStrategy):
    """趋势进阶模块: Aroon / Vortex / LinearSlope / TRIX / Chandelier."""

    FACTORS = [
        ("aroon", AroonFactor(), "panel", False),
        ("vortex", VortexFactor(), "ohlc", False),
        ("linear_slope", LinearSlopeFactor(), "panel", False),
        ("trix", TRIXFactor(), "panel", False),
        ("chandelier", ChandelierExitFactor(), "ohlc", True),
    ]


class MeanReversionModuleStrategy(_ModuleStrategy):
    """均值回归模块: HalfLife / Squeeze."""

    FACTORS = [
        ("half_life", HalfLifeFactor(), "panel", False),
        ("squeeze", SqueezeFactor(), "ohlc", True),
    ]


# ── 组合策略 ───────────────────────────────────────────────────────────────

@dataclass
class ComboConfig(ModuleConfig):
    """组合策略配置: 模块权重 (振荡/趋势/均值回归)."""

    module_weights: tuple[float, float, float] = (0.3, 0.5, 0.2)


class FactorModuleComboStrategy(_ModuleStrategy):
    """组合策略: 三模块信号加权 → 单一资金池 vol targeting.

    模块权重默认 0.3/0.5/0.2 (趋势主导, 业界 CTA 惯例).
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
            OscillatorModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            TrendAdvancedModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
            MeanReversionModuleStrategy(self._products, cfg, self._dl, self._start, self._end),
        ]
        self.module_signals: dict[str, pd.DataFrame] = {}
        self._combo_raw: pd.DataFrame | None = None

    def load_panel(self) -> pd.DataFrame:
        panel = super().load_panel()
        for m in self._modules:
            m._price_panel = panel
            m._ohlc_map = self._ohlc_map
        return panel

    def _signals_raw(self) -> pd.DataFrame:
        """加权合成: 0.3×振荡 + 0.5×趋势 + 0.2×均值回归."""
        w = self._config.module_weights
        self.module_signals = {}
        combo = None
        for m, w_i in zip(self._modules, w):
            sig = m._signals_raw()
            self.module_signals[m.__class__.__name__] = sig
            combo = sig * w_i if combo is None else combo + sig * w_i
        self._combo_raw = combo
        return combo

    def _signals(self) -> pd.DataFrame:
        return np.sign(self._signals_raw())

    def run_factor(self, name: str) -> CtaBacktestResult:
        """组合策略按模块归因 (而非单因子)."""
        if not self.module_signals:
            self._signals_raw()
        sig = np.sign(self.module_signals[name])
        return self._run_signal(sig)

    def module_correlation(self) -> pd.DataFrame:
        """模块级信号相关矩阵."""
        if not self.module_signals:
            self._signals_raw()
        frame = pd.DataFrame({
            name: s.to_numpy().ravel() for name, s in self.module_signals.items()
        })
        return frame.corr()
