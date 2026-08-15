# coding=utf-8
"""Multi-Factor Options Strategy: 多因子期权策略.

综合使用 IV/Skew/VRP/Momentum/Term Structure 因子,
动态选择最优期权策略, 严格管理保证金和换月.

核心逻辑:
    1. 因子计算 → 生成综合信号
    2. 信号 → 路由到最优策略
    3. 仓位管理 → 保证金 ≤ 50%
    4. 换月管理 → 到期前 7 天自动换月
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from ..options.iv_panel import load_iv_panel
from ..options.hedge_data import load_vrp_panel, vrp_percentile_panel
from ..options.vol_signals import VolSignalGenerator, Regime
from ..options.option_factors import (
    iv_zscore,
    skew_rolling_panel,
    skew_zscore,
    skew_momentum,
    put_call_volume_ratio,
    oi_change_rate,
    realized_vol_asymmetry,
)
from ..options.selector import (
    select_call, select_put, select_straddle, select_put_spread,
)
from ..backtest.cta_contract import CtaContractRegistry


class Signal(IntEnum):
    """策略信号."""
    SELL_PUT_SPREAD = 0     # 卖 put spread (低波 + 看涨/中性)
    SELL_STRANGLE = 1       # 卖 strangle (高波 + 横盘)
    BUY_PUT_SPREAD = 2      # 买 put spread (看跌保护)
    BUY_STRADDLE = 3        # 买 straddle (低波 + 预期大波动)
    BUY_LONG_PUT = 4        # 买 long put (高波 + 看跌)
    SELL_CALL_SPREAD = 5    # 卖 call spread (看跌 + 高 IV)
    CALENDAR_SPREAD = 6     # 日历价差 (IV contango)
    NO_TRADE = 7            # 不交易


@dataclass
class MultiFactorConfig:
    """多因子策略配置."""
    # 品种
    products: list[str] = field(default_factory=lambda: ["RB"])

    # 仓位管理
    initial_cash: float = 1_000_000.0
    max_margin_pct: float = 0.50       # 保证金上限 50%
    max_position_per_product: int = 2  # 每品种最大持仓组数

    # 换月
    roll_days: int = 7                 # 到期前 N 天换月
    min_dte: int = 20                  # 最小到期天数
    max_dte: int = 60                  # 最大到期天数

    # 因子权重
    w_iv_rank: float = 0.25            # IV 分位权重
    w_vrp: float = 0.20                # VRP 权重
    w_skew: float = 0.15               # Skew 权重
    w_momentum: float = 0.15           # 动量权重
    w_trend: float = 0.15              # 趋势权重 (CTA)
    w_pcvr: float = 0.10               # P/C Volume 权重

    # 信号阈值
    signal_threshold: float = 0.3      # 综合信号 > threshold 才交易

    # CTA 因子
    use_cta_factors: bool = True       # 是否使用 CTA 趋势因子


@dataclass
class Position:
    """持仓记录."""
    product: str
    symbols: dict[str, float]   # {option_symbol: qty}
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    strategy_name: str
    margin_used: float = 0.0


class MultiFactorOptionsStrategy:
    """多因子期权策略引擎.

    使用方式:
        strategy = MultiFactorOptionsStrategy(config)
        result = strategy.run(start="2023-01-01", end="2025-12-31")
    """

    def __init__(self, config: MultiFactorConfig | None = None):
        self._cfg = config or MultiFactorConfig()
        self._loader = None
        self._um = None
        self._oe = None
        self._registry = CtaContractRegistry()

        # 数据缓存
        self._iv_panel: pd.DataFrame | None = None
        self._vrp_panel: pd.DataFrame | None = None
        self._price_data: dict[str, pd.Series] = {}
        self._option_data: dict[str, pd.DataFrame] = {}
        self._hv_data: dict[str, pd.Series] = {}

        # 因子缓存
        self._iv_zscore: pd.DataFrame | None = None
        self._skew_panel: pd.DataFrame | None = None
        self._pcvr: pd.Series | None = None
        self._oi_delta: pd.DataFrame | None = None
        self._vol_asym: pd.DataFrame | None = None
        self._vol_gen: VolSignalGenerator | None = None

        # 状态
        self._positions: dict[str, Position] = {}
        self._trades: list[dict] = []
        self._daily_pnl: list[dict] = []

    def load_data(self, start: str, end: str):
        """预加载所有数据."""
        from ..data_loader import CtaDataLoader
        from ..options.iv_panel import add_percentile

        self._loader = CtaDataLoader()
        from ..options.underlying_map import UnderlyingMap
        from ..options.expiry import OptionExpiry
        self._um = UnderlyingMap(self._loader)
        self._oe = OptionExpiry(self._loader)

        # 标的价格
        for prod in self._cfg.products:
            try:
                px = self._loader.load_continuous_main(prod)
                if px is not None and not px.empty:
                    px["trade_date"] = pd.to_datetime(px["trade_date"])
                    px = px.set_index("trade_date")
                    self._price_data[prod] = px["close"]
            except Exception:
                pass

        # 期权数据
        for prod in self._cfg.products:
            try:
                df = self._loader.load_v_all_options(products=[prod], start=start, end=end)
                if df is not None and not df.empty:
                    df["trade_date"] = pd.to_datetime(df["trade_date"])
                    self._option_data[prod] = df
            except Exception:
                pass

        # IV 面板
        try:
            self._iv_panel = load_iv_panel(start=start, end=end)
        except Exception:
            self._iv_panel = None

        # VRP 面板
        try:
            self._vrp_panel = load_vrp_panel(products=self._cfg.products, start=start, end=end)
        except Exception:
            self._vrp_panel = None

        # HV 数据
        for prod, series in self._price_data.items():
            log_ret = np.log(series / series.shift(1))
            hv = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
            self._hv_data[prod] = hv

        # VolSignalGenerator
        if self._iv_panel is not None:
            hv_panel = pd.DataFrame(self._hv_data)
            self._vol_gen = VolSignalGenerator(self._iv_panel, hv_panel, iv_pct_window=20)

        # 预计算因子
        self._compute_factors()

    def _compute_factors(self):
        """预计算所有因子."""
        if self._iv_panel is None:
            return

        # IV Z-Score
        self._iv_zscore = iv_zscore(self._iv_panel, window=60)

        # Skew panel
        for prod, df in self._option_data.items():
            if not df.empty:
                # 统一列名
                df2 = df.copy()
                if "dt" not in df2.columns and "trade_date" in df2.columns:
                    df2["dt"] = df2["trade_date"]
                if "k" not in df2.columns and "strike_price" in df2.columns:
                    df2["k"] = df2["strike_price"]
                if "f" not in df2.columns and "underlying" in df2.columns:
                    # 用标的价格填充
                    series = self._price_data.get(prod)
                    if series is not None:
                        df2["f"] = df2["dt"].map(series) if series.index.dtype == df2["dt"].dtype else np.nan
                if "bs_iv" not in df2.columns and "close" in df2.columns:
                    df2["bs_iv"] = df2["close"]  # 近似
                self._skew_panel = skew_rolling_panel(df2, window=20)
                break

        # P/C Volume Ratio
        for prod, df in self._option_data.items():
            if not df.empty and "volume" in df.columns:
                df2 = df.copy()
                if "dt" not in df2.columns and "trade_date" in df2.columns:
                    df2["dt"] = df2["trade_date"]
                if "cp" not in df2.columns and "option_type" in df2.columns:
                    df2["cp"] = df2["option_type"]
                self._pcvr = put_call_volume_ratio(df2)
                break

        # OI Change
        for prod, df in self._option_data.items():
            if not df.empty and "open_interest" in df.columns:
                df2 = df.copy()
                if "dt" not in df2.columns and "trade_date" in df2.columns:
                    df2["dt"] = df2["trade_date"]
                if "cp" not in df2.columns and "option_type" in df2.columns:
                    df2["cp"] = df2["option_type"]
                self._oi_delta = oi_change_rate(df2)
                break

        # Realized Vol Asymmetry
        for prod, series in self._price_data.items():
            returns = series.pct_change()
            self._vol_asym = realized_vol_asymmetry(returns, window=60)
            break

        # CTA Trend Factors
        if self._cfg.use_cta_factors:
            self._compute_cta_factors()

    def _compute_cta_factors(self):
        """计算 CTA 趋势因子."""
        from ..factors.trend import (
            DonchianChannelFactor, MAFactor, MomentumFactor,
            TSMAFactor, RSIFactor, BollingerFactor, EfficiencyFactor,
        )
        from ..factors.composite import MultiHorizonMomentumFactor, SharpeMomentumFactor

        # 构造价格面板
        price_panel = pd.DataFrame(self._price_data)
        price_panel = price_panel.dropna(how="all")
        if price_panel.empty:
            return

        self._cta_signals = {}

        # 1. Donchian Channel (20/10)
        donchian = DonchianChannelFactor(entry_window=20, exit_window=10)
        self._cta_signals["donchian"] = donchian.compute(price_panel)

        # 2. MA Crossover (5/20)
        ma = MAFactor(short_window=5, long_window=20)
        self._cta_signals["ma"] = ma.compute(price_panel)

        # 3. Momentum (20-day)
        mom = MomentumFactor(lookback=20)
        self._cta_signals["momentum"] = mom.compute(price_panel)

        # 4. TSMOM (60-day, AQR style)
        tsom = TSMAFactor(lookback=60)
        self._cta_signals["tsmom"] = tsom.compute(price_panel)

        # 5. Multi-Horizon Momentum (3/6/12m weighted)
        multi_mom = MultiHorizonMomentumFactor()
        self._cta_signals["multi_mom"] = multi_mom.compute(price_panel)

        # 6. Sharpe Momentum (momentum / vol)
        sharpe_mom = SharpeMomentumFactor()
        self._cta_signals["sharpe_mom"] = sharpe_mom.compute(price_panel)

        # 7. RSI (14-day)
        rsi = RSIFactor(window=14)
        self._cta_signals["rsi"] = rsi.compute(price_panel)

        # 8. Bollinger %B
        boll = BollingerFactor(window=20, n_std=2)
        self._cta_signals["bollinger"] = boll.compute(price_panel)

        # 9. Efficiency Ratio (10-day)
        eff = EfficiencyFactor(window=10)
        self._cta_signals["efficiency"] = eff.compute(price_panel)

    def _get_cta_signal(self, product: str, date: pd.Timestamp) -> float | None:
        """获取 CTA 综合趋势信号.

        多因子投票: 多个趋势因子一致 → 强信号
        """
        if not self._cfg.use_cta_factors or not hasattr(self, "_cta_signals"):
            return None

        votes = []
        for name, signal_df in self._cta_signals.items():
            if product in signal_df.columns and date in signal_df.index:
                val = signal_df.loc[date, product]
                if not np.isnan(val):
                    votes.append(val)

        if not votes:
            return None

        # 加权投票 (Donchian/MultiMom 权重更高)
        weights_map = {
            "donchian": 2.0, "multi_mom": 2.0, "sharpe_mom": 1.5,
            "tsmom": 1.5, "momentum": 1.0, "ma": 1.0,
            "rsi": 0.5, "bollinger": 0.5, "efficiency": 1.0,
        }
        weighted_votes = []
        weighted_weights = []
        for i, (name, signal_df) in enumerate(self._cta_signals.items()):
            if product in signal_df.columns and date in signal_df.index:
                val = signal_df.loc[date, product]
                if not np.isnan(val):
                    weighted_votes.append(val)
                    weighted_weights.append(weights_map.get(name, 1.0))

        if not weighted_votes:
            return None

        return np.average(weighted_votes, weights=weighted_weights)

    def _get_signal(self, product: str, date: pd.Timestamp) -> Signal:
        """计算综合信号 → 选择策略."""
        spot = self._get_price(product, date)
        if spot is None or spot <= 0:
            return Signal.NO_TRADE

        # 收集各因子信号
        signals = []
        weights = []

        # 1. IV Rank (权重 0.25)
        iv_rank = self._get_iv_rank(product, date)
        if iv_rank is not None:
            if iv_rank < 0.3:
                signals.append(1.0)   # IV 低 → 买方
            elif iv_rank > 0.7:
                signals.append(-1.0)  # IV 高 → 卖方
            else:
                signals.append(0.0)
            weights.append(self._cfg.w_iv_rank)

        # 2. VRP (权重 0.20)
        vrp = self._get_vrp(product, date)
        if vrp is not None:
            if vrp > 0.02:
                signals.append(-1.0)  # VRP 正 → 期权贵 → 卖
            elif vrp < -0.02:
                signals.append(1.0)   # VRP 负 → 期权便宜 → 买
            else:
                signals.append(0.0)
            weights.append(self._cfg.w_vrp)

        # 3. Skew (权重 0.15)
        skew_val = self._get_skew(product, date)
        if skew_val is not None:
            if skew_val > 0.02:
                signals.append(-0.5)  # skew 陡 → put 贵 → 卖 put spread
            elif skew_val < -0.01:
                signals.append(0.5)   # skew 平 → 买 put spread
            else:
                signals.append(0.0)
            weights.append(self._cfg.w_skew)

        # 4. Momentum (权重 0.15)
        mom = self._get_momentum(product, date, lookback=20)
        if mom is not None:
            signals.append(np.sign(mom))
            weights.append(self._cfg.w_momentum)

        # 5. CTA Trend (动态权重)
        cta_signal = self._get_cta_signal(product, date)
        if cta_signal is not None:
            trend_strength = abs(cta_signal)
            if trend_strength > 0.5:
                trend_weight = self._cfg.w_trend * 1.5
            elif trend_strength < 0.2:
                trend_weight = self._cfg.w_trend * 0.5
            else:
                trend_weight = self._cfg.w_trend
            signals.append(np.clip(cta_signal, -1, 1))
            weights.append(trend_weight)

        # 6. P/C Volume (权重 0.10)
        pcvr = self._get_pcvr(date)
        if pcvr is not None:
            if pcvr > 1.3:
                signals.append(-0.5)  # put 成交量高 → 看跌
            elif pcvr < 0.7:
                signals.append(0.5)   # call 成交量高 → 看涨
            else:
                signals.append(0.0)
            weights.append(self._cfg.w_pcvr)

        if not signals:
            return Signal.NO_TRADE

        # 加权平均
        composite = np.average(signals, weights=weights)

        # 路由到策略 (CTA 趋势增强方向性)
        iv_rank = iv_rank if iv_rank is not None else 0.5

        # CTA 趋势增强: 强趋势时偏向方向性策略
        if cta_signal is not None and abs(cta_signal) > 0.5:
            if cta_signal > 0.5:
                # 强看涨
                if iv_rank < 0.4:
                    return Signal.BUY_STRADDLE
                else:
                    return Signal.SELL_PUT_SPREAD
            else:
                # 强看跌
                if iv_rank > 0.6:
                    return Signal.SELL_STRANGLE
                elif iv_rank < 0.4:
                    return Signal.BUY_LONG_PUT
                else:
                    return Signal.SELL_CALL_SPREAD

        if composite > self._cfg.signal_threshold:
            if iv_rank < 0.4:
                return Signal.BUY_STRADDLE
            elif composite > 0.5:
                return Signal.BUY_PUT_SPREAD
            else:
                return Signal.SELL_PUT_SPREAD
        elif composite < -self._cfg.signal_threshold:
            if iv_rank > 0.6:
                return Signal.SELL_STRANGLE
            elif composite < -0.5:
                return Signal.SELL_CALL_SPREAD
            else:
                return Signal.BUY_LONG_PUT
        else:
            # 中性区域: 看 IV 水平
            if iv_rank > 0.65:
                return Signal.SELL_STRANGLE
            elif iv_rank < 0.35:
                return Signal.CALENDAR_SPREAD
            return Signal.NO_TRADE

    def _get_price(self, product: str, date: pd.Timestamp) -> float | None:
        series = self._price_data.get(product)
        if series is None:
            return None
        if date in series.index:
            return float(series.loc[date])
        idx = series.index.get_indexer([date], method="ffill")
        if idx[0] >= 0:
            return float(series.iloc[idx[0]])
        return None

    def _get_iv_rank(self, product: str, date: pd.Timestamp) -> float | None:
        if self._iv_panel is None or product not in self._iv_panel.columns:
            return None
        if date not in self._iv_panel.index:
            return None
        iv = self._iv_panel.loc[date, product]
        if pd.isna(iv):
            return None
        # 计算分位
        hist = self._iv_panel[product].dropna()
        if len(hist) < 60:
            return None
        return float((hist < iv).sum() / len(hist))

    def _get_vrp(self, product: str, date: pd.Timestamp) -> float | None:
        if self._vrp_panel is None or product not in self._vrp_panel.columns:
            return None
        if date not in self._vrp_panel.index:
            return None
        v = self._vrp_panel.loc[date, product]
        return float(v) if pd.notna(v) else None

    def _get_skew(self, product: str, date: pd.Timestamp) -> float | None:
        if self._skew_panel is None or product not in self._skew_panel.columns:
            return None
        if date not in self._skew_panel.index:
            return None
        v = self._skew_panel.loc[date, product]
        return float(v) if pd.notna(v) else None

    def _get_momentum(self, product: str, date: pd.Timestamp, lookback: int = 20) -> float | None:
        series = self._price_data.get(product)
        if series is None:
            return None
        if date not in series.index:
            return None
        idx = series.index.get_loc(date)
        if idx < lookback:
            return None
        return float(series.iloc[idx] / series.iloc[idx - lookback] - 1)

    def _get_trend(self, product: str, date: pd.Timestamp) -> float | None:
        """简单 MA 趋势: price > MA20 → +1, < MA20 → -1."""
        series = self._price_data.get(product)
        if series is None or len(series) < 20:
            return None
        if date not in series.index:
            return None
        idx = series.index.get_loc(date)
        if idx < 20:
            return None
        ma = series.iloc[idx - 20:idx].mean()
        price = series.iloc[idx]
        return 1.0 if price > ma else -1.0

    def _get_pcvr(self, date: pd.Timestamp) -> float | None:
        if self._pcvr is None:
            return None
        if date in self._pcvr.index:
            v = self._pcvr.loc[date]
            return float(v) if pd.notna(v) else None
        return None

    def _select_strategy(
        self, signal: Signal, product: str, date: pd.Timestamp, spot: float,
    ) -> dict[str, float]:
        """根据信号选择具体策略."""
        loader = self._loader
        um = self._um
        oe = self._oe

        if signal == Signal.SELL_PUT_SPREAD:
            return self._sell_put_spread(product, date, spot, loader, um, oe)
        elif signal == Signal.SELL_STRANGLE:
            return self._sell_strangle(product, date, spot, loader, um, oe)
        elif signal == Signal.BUY_PUT_SPREAD:
            return self._buy_put_spread(product, date, spot, loader, um, oe)
        elif signal == Signal.BUY_STRADDLE:
            return self._buy_straddle(product, date, spot, loader, um, oe)
        elif signal == Signal.BUY_LONG_PUT:
            return self._buy_long_put(product, date, spot, loader, um, oe)
        elif signal == Signal.SELL_CALL_SPREAD:
            return self._sell_call_spread(product, date, spot, loader, um, oe)
        elif signal == Signal.CALENDAR_SPREAD:
            return self._calendar_spread(product, date, spot, loader, um, oe)
        return {}

    def _sell_put_spread(self, product, date, spot, loader, um, oe):
        long_put = select_put(product, date, spot, target_delta=0.25,
                              min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                              loader=loader, um=um, oe=oe)
        short_put = select_put(product, date, spot, target_delta=0.10,
                               min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                               loader=loader, um=um, oe=oe)
        targets = {}
        if long_put:
            targets[long_put.symbol] = 1.0
        if short_put and short_put.strike < (long_put.strike if long_put else 0):
            targets[short_put.symbol] = -1.0
        return targets

    def _sell_strangle(self, product, date, spot, loader, um, oe):
        call = select_call(product, date, spot, target_delta=0.25,
                           min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                           loader=loader, um=um, oe=oe)
        put = select_put(product, date, spot, target_delta=0.25,
                         min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                         loader=loader, um=um, oe=oe)
        targets = {}
        if call:
            targets[call.symbol] = -1.0
        if put:
            targets[put.symbol] = -1.0
        return targets

    def _buy_put_spread(self, product, date, spot, loader, um, oe):
        long_put = select_put(product, date, spot, target_delta=0.25,
                              min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                              loader=loader, um=um, oe=oe)
        short_put = select_put(product, date, spot, target_delta=0.10,
                               min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                               loader=loader, um=um, oe=oe)
        targets = {}
        if long_put:
            targets[long_put.symbol] = 1.0
        if short_put and short_put.strike < (long_put.strike if long_put else 0):
            targets[short_put.symbol] = -1.0
        return targets

    def _buy_straddle(self, product, date, spot, loader, um, oe):
        s = select_straddle(product, date, spot,
                            min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                            loader=loader, um=um, oe=oe)
        if s:
            return {s.call.symbol: 1.0, s.put.symbol: 1.0}
        return {}

    def _buy_long_put(self, product, date, spot, loader, um, oe):
        put = select_put(product, date, spot, target_delta=0.25,
                         min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                         loader=loader, um=um, oe=oe)
        if put:
            return {put.symbol: 1.0}
        return {}

    def _sell_call_spread(self, product, date, spot, loader, um, oe):
        long_call = select_call(product, date, spot, target_delta=0.25,
                                min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                                loader=loader, um=um, oe=oe)
        short_call = select_call(product, date, spot, target_delta=0.10,
                                 min_days=self._cfg.min_dte, max_days=self._cfg.max_dte,
                                 loader=loader, um=um, oe=oe)
        targets = {}
        if long_call:
            targets[long_call.symbol] = 1.0
        if short_call and short_call.strike > (long_call.strike if long_call else 0):
            targets[short_call.symbol] = -1.0
        return targets

    def _calendar_spread(self, product, date, spot, loader, um, oe):
        near_put = select_put(product, date, spot, target_delta=0.25,
                              min_days=15, max_days=35,
                              loader=loader, um=um, oe=oe)
        far_put = select_put(product, date, spot, target_delta=0.25,
                             min_days=45, max_days=75,
                             loader=loader, um=um, oe=oe)
        targets = {}
        if near_put:
            targets[near_put.symbol] = -1.0  # 卖近月
        if far_put:
            targets[far_put.symbol] = 1.0    # 买远月
        return targets

    def _check_roll(self, product: str, date: pd.Timestamp) -> bool:
        """检查是否需要换月."""
        pos = self._positions.get(product)
        if pos is None:
            return False
        days_left = (pos.expiry - date).days
        return days_left <= self._cfg.roll_days

    def _check_signal_change(self, product: str, date: pd.Timestamp, new_signal: Signal) -> bool:
        """检查信号是否变化."""
        pos = self._positions.get(product)
        if pos is None:
            return True  # 无持仓, 需要入场
        # 信号分类: 买方/卖方/中性
        old_type = self._signal_type(pos.strategy_name)
        new_type = self._signal_type(new_signal)
        return old_type != new_type

    def _signal_type(self, signal_or_name) -> str:
        """信号分类."""
        if isinstance(signal_or_name, Signal):
            s = signal_or_name
        else:
            name_map = {
                "sell_put_spread": Signal.SELL_PUT_SPREAD,
                "sell_strangle": Signal.SELL_STRANGLE,
                "buy_put_spread": Signal.BUY_PUT_SPREAD,
                "buy_straddle": Signal.BUY_STRADDLE,
                "buy_long_put": Signal.BUY_LONG_PUT,
                "sell_call_spread": Signal.SELL_CALL_SPREAD,
                "calendar_spread": Signal.CALENDAR_SPREAD,
            }
            s = name_map.get(signal_or_name, Signal.NO_TRADE)

        if s in (Signal.SELL_PUT_SPREAD, Signal.SELL_STRANGLE, Signal.SELL_CALL_SPREAD):
            return "sell"
        elif s in (Signal.BUY_PUT_SPREAD, Signal.BUY_STRADDLE, Signal.BUY_LONG_PUT):
            return "buy"
        elif s == Signal.CALENDAR_SPREAD:
            return "calendar"
        return "neutral"

    def _estimate_margin(self, targets: dict[str, float], spot: float) -> float:
        """估算保证金占用 (简化: 用 spot × 手数 × 乘数 × 保证金率)."""
        margin = 0.0
        for sym, qty in targets.items():
            if qty < 0:  # 卖方需要保证金
                mult = self._registry.get_or_default("RB").contract_multiplier
                margin += abs(qty) * spot * mult * 0.15  # 简化保证金率
        return margin

    def run(self, start: str = "2023-01-01", end: str = "2025-12-31") -> dict:
        """运行回测.

        Returns:
            dict 含 trades/daily_pnl/factor_values
        """
        import time
        t0 = time.time()

        self.load_data(start, end)

        # 逐日模拟
        for product in self._cfg.products:
            series = self._price_data.get(product)
            if series is None:
                continue

            dates = series.index[(series.index >= start) & (series.index <= end)]

            for date in dates:
                spot = float(series.loc[date])
                if np.isnan(spot) or spot <= 0:
                    continue

                # 1. 检查换月
                if self._check_roll(product, date):
                    self._close_position(product, date, spot, "roll")

                # 2. 计算信号
                signal = self._get_signal(product, date)

                # 3. 信号变化 → 平旧仓
                if not self._check_signal_change(product, date, signal):
                    continue  # 信号不变, 继续持有

                if signal != Signal.NO_TRADE:
                    # 平旧仓
                    if product in self._positions:
                        self._close_position(product, date, spot, "signal_change")

                    # 开新仓
                    targets = self._select_strategy(signal, product, date, spot)
                    if targets:
                        # 保证金检查
                        margin = self._estimate_margin(targets, spot)
                        if margin > self._cfg.initial_cash * self._cfg.max_margin_pct:
                            continue  # 超过保证金上限

                        self._open_position(product, date, spot, targets, signal)

        elapsed = time.time() - t0

        # 统计
        n_trades = len(self._trades)
        n_buys = sum(1 for t in self._trades if t["side"] == "open")
        n_sells = sum(1 for t in self._trades if t["side"] == "close")

        print(f"回测完成: {n_trades} 笔交易 (开仓{n_buys}, 平仓{n_sells}), {elapsed:.0f}s")

        return {
            "trades": self._trades,
            "daily_pnl": self._daily_pnl,
            "n_trades": n_trades,
            "elapsed": elapsed,
        }

    def _open_position(self, product, date, spot, targets, signal):
        pos = Position(
            product=product,
            symbols=targets,
            entry_date=date,
            expiry=date + pd.Timedelta(days=self._cfg.max_dte),
            strategy_name=signal.name.lower(),
            margin_used=self._estimate_margin(targets, spot),
        )
        self._positions[product] = pos

        for sym, qty in targets.items():
            self._trades.append({
                "date": date, "symbol": sym, "side": "open",
                "qty": qty, "product": product, "strategy": signal.name,
            })

    def _close_position(self, product, date, spot, reason):
        pos = self._positions.get(product)
        if pos is None:
            return

        for sym, qty in pos.symbols.items():
            self._trades.append({
                "date": date, "symbol": sym, "side": "close",
                "qty": -qty, "product": product, "strategy": pos.strategy_name,
                "reason": reason,
            })

        self._positions.pop(product, None)
