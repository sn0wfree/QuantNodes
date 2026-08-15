# coding=utf-8
"""Options-Enhanced fut_v1 v3: 裸买 Call/Put + Roll + VRP 动态仓位 + Put Spread 保护.

三项升级:
    1. Roll 管理: 到期前自动换月 + 信号反转退出
    2. 动态 VRP 仓位: VRP 强度连续调整仓位大小
    3. Put Spread 尾部保护: 降低 MaxDD
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy, ComboConfig, ModuleConfig
from ..factors.volatility import vol_target_positions
from ..options.iv_panel import load_iv_panel
from ..options.hedge_data import load_vrp_panel
from ..options.selector import select_call, select_put, select_put_spread
from ..backtest.cta_contract import CtaContractRegistry, OptionContractSpec, CtaContract


@dataclass
class OptionsBoostConfig:
    """增强配置."""
    # 加仓比例
    boost_notional_pct: float = 1.0   # 加仓名义 = 期货仓位 × 100% (覆盖全部方向敞口)
    boost_max_lots: int = 60          # 最大单品种期权总手数
    max_options_per_product: int = 2  # 每品种最多同时持有期权数
    max_premium_pct: float = 0.10     # 组合权利金预算: 所有期权权利金 ≤ 账户 10%

    # IV 过滤 (连续)
    iv_high_threshold: float = 0.7
    iv_zscore_max: float = 2.0      # IV z-score > 2: 不加仓

    # DTE
    min_dte: int = 30
    max_dte: int = 60
    roll_dte: int = 10              # 到期前 N 天换月

    # 动态 VRP 仓位
    vrp_boost_high: float = 0.8     # VRP 分位 > 80%: 仓位 ×1.3
    vrp_boost_mid: float = 0.5      # VRP 分位 50-80%: 仓位 ×1.0
    vrp_boost_low: float = 0.2      # VRP 分位 20-50%: 仓位 ×0.5
    vrp_min_threshold: float = 0.2  # VRP 分位 < 20%: 不交易

    # Put Spread 保护
    enable_put_spread: bool = True
    put_spread_budget_pct: float = 0.005  # 年化预算 = NAV × 0.5%
    put_spread_delta: float = 0.25
    put_spread_wing: float = 0.10


class OptionsBoostedFutV1(FactorModuleComboStrategy):
    """增强版 fut_v1: CTA + 裸买期权 + Roll + VRP 动态仓位 + Put Spread 保护."""

    def __init__(
        self,
        boost_config: OptionsBoostConfig | None = None,
        option_contracts: dict | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._boost_cfg = boost_config or OptionsBoostConfig()
        self._option_contracts = option_contracts or {}
        self._option_data = option_data or {}
        self._iv_panel = None
        self._vrp_panel = None
        self._option_holdings: dict[str, dict] = {}
        self._put_spread_holdings: dict[str, dict] = {}
        self._load_options_data()

    def _load_options_data(self):
        try:
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            pass
        try:
            self._vrp_panel = load_vrp_panel(
                products=self._products, start=self._start, end=self._end,
            )
        except Exception:
            pass

    # ── 信号获取 ──────────────────────────────────────────────────────────

    def _get_vrp_signal(self, product: str, date: pd.Timestamp) -> float:
        if self._vrp_panel is None or product not in self._vrp_panel.columns:
            return 0.0
        if date not in self._vrp_panel.index:
            return 0.0
        vrp = self._vrp_panel.loc[date, product]
        return float(vrp) if pd.notna(vrp) else 0.0

    def _get_vrp_percentile(self, product: str, date: pd.Timestamp) -> float:
        """VRP 分位 (0~1) — 只用截至 date 的历史数据, 无前视."""
        if self._vrp_panel is None or product not in self._vrp_panel.columns:
            return 0.5
        if date not in self._vrp_panel.index:
            return 0.5
        vrp = self._vrp_panel.loc[date, product]
        if pd.isna(vrp):
            return 0.5
        # 只用历史: 截至 date 的已实现值 (不含未来)
        hist = self._vrp_panel[product].loc[:date].dropna()
        if len(hist) < 60:
            return 0.5
        return float((hist < vrp).sum() / len(hist))

    def _get_iv_rank(self, product: str, date: pd.Timestamp) -> float:
        if self._iv_panel is None or product not in self._iv_panel.columns:
            return 0.5
        if date not in self._iv_panel.index:
            return 0.5
        iv_val = self._iv_panel.loc[date, product]
        if pd.isna(iv_val):
            return 0.5
        # 只用历史: 截至 date 的已实现值 (不含未来)
        hist = self._iv_panel[product].loc[:date].dropna()
        if len(hist) < 60:
            return 0.5
        return float((hist < iv_val).sum() / len(hist))

    def _get_iv_zscore(self, product: str, date: pd.Timestamp) -> float:
        """IV Z-Score — 只用截至 date 的历史数据, 无前视."""
        if self._iv_panel is None or product not in self._iv_panel.columns:
            return 0.0
        if date not in self._iv_panel.index:
            return 0.0
        iv_val = self._iv_panel.loc[date, product]
        if pd.isna(iv_val):
            return 0.0
        hist = self._iv_panel[product].loc[:date].dropna()
        if len(hist) < 60:
            return 0.0
        mu = hist.mean()
        std = hist.std()
        if std <= 0:
            return 0.0
        return float((iv_val - mu) / std)

    def _get_vrp_size_multiplier(self, product: str, date: pd.Timestamp) -> float:
        """VRP 动态仓位乘数 (0~1.3) — 用于裸买期权 (long gamma).

        核心逻辑: 本策略是买方 (买 Call/Put 加仓),
        VRP 低 = IV < HV = 期权便宜 = 应加大买入; 反之减少.

        Returns:
            VRP 分位 < 0.2 → 1.3 (期权便宜, 加仓)
            VRP 分位 0.2-0.5 → 1.0 (正常)
            VRP 分位 0.5-0.8 → 0.5 (期权偏贵, 减仓)
            VRP 分位 > 0.8 → 0.0 (期权太贵, 不买)
        """
        vrp_pct = self._get_vrp_percentile(product, date)
        cfg = self._boost_cfg

        if vrp_pct < cfg.vrp_min_threshold:
            return 1.3  # VRP 极低: 期权最便宜, 加大买入
        elif vrp_pct < cfg.vrp_boost_low:
            return 1.0  # 低 VRP: 期权便宜, 正常买
        elif vrp_pct < cfg.vrp_boost_mid:
            return 1.0  # 中 VRP: 正常
        elif vrp_pct < cfg.vrp_boost_high:
            return 0.5  # 高 VRP: 期权偏贵, 减仓
        else:
            return 0.0  # VRP 极高: 期权太贵, 不买

    # ── 期权选择 ──────────────────────────────────────────────────────────

    def _select_atm_from_cache(
        self, product: str, date: pd.Timestamp, spot: float, direction: float,
    ) -> tuple[str, float, str] | None:
        """内存版 ATM 期权选择 (避免每次查 DuckDB).

        从 self._option_data 缓存中选 delta 最接近 0.40 的合约.
        方向: >0 买 Call, <0 买 Put.

        Returns:
            (symbol, premium, expiry) 或 None
        """
        df = self._option_data.get(product)
        if df is None or df.empty:
            return None

        # 找当日或最近的期权行情
        day = df[df["trade_date"] == date]
        if day.empty:
            dates = df["trade_date"].unique()
            idx = np.searchsorted(dates, date, side="right") - 1
            if idx < 0:
                return None
            day = df[df["trade_date"] == dates[idx]]
        if day.empty:
            return None

        # 过滤流动性
        liq = day[(day.get("volume", pd.Series(0, index=day.index)).fillna(0) > 0)
                  | (day.get("open_interest", pd.Series(0, index=day.index)).fillna(0) > 0)]
        day = liq if not liq.empty else day

        # 选目标期权类型
        opt_type = "C" if direction > 0 else "P"
        subset = day[day["option_type"].str.upper() == opt_type]
        if subset.empty:
            return None

        # 用 strike 近似 delta (ATM = strike 接近 spot 的 0.40 delta 通常偏 OTM)
        # 简化: 选 strike 最接近 spot 且 <=/>= spot 的 OTM 合约
        if opt_type == "C":
            otm = subset[subset["strike_price"] >= spot]
        else:
            otm = subset[subset["strike_price"] <= spot]
        if otm.empty:
            otm = subset

        otm = otm.copy()
        otm["_dist"] = (otm["strike_price"] - spot).abs()
        best = otm.sort_values("_dist").iloc[0]

        sym = best["symbol"]
        if sym not in self._option_contracts:
            return None

        premium = float(best["close"]) if pd.notna(best["close"]) else 0.0
        if premium <= 0:
            return None

        # expiry: 用 underlying (如 rb2307) 推断到期月, 失败则近似 date + 60天
        expiry = ""
        underlying = str(best.get("underlying", ""))
        if underlying:
            # 提取 YYMM
            digits = "".join(ch for ch in underlying if ch.isdigit())
            if len(digits) >= 4:
                yy = int(digits[:2])
                mm = int(digits[2:4])
                year = 2000 + yy
                # 到期约在合约月的 15 日
                try:
                    expiry = pd.Timestamp(year=year, month=mm, day=15).strftime("%Y-%m-%d")
                except Exception:
                    pass
        if not expiry:
            expiry = (pd.Timestamp(date) + pd.Timedelta(days=60)).strftime("%Y-%m-%d")

        return (sym, premium, expiry)

    def _select_option_for_boost(
        self, product: str, date: pd.Timestamp, spot: float, direction: float,
    ) -> tuple[str, float] | None:
        from ..options.underlying_map import UnderlyingMap
        from ..options.expiry import OptionExpiry
        loader = self._dl
        um = UnderlyingMap(loader)
        oe = OptionExpiry(loader)

        if direction > 0:
            sel = select_call(
                product, date, spot, target_delta=0.40,
                min_days=self._boost_cfg.min_dte, max_days=self._boost_cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )
        else:
            sel = select_put(
                product, date, spot, target_delta=0.40,
                min_days=self._boost_cfg.min_dte, max_days=self._boost_cfg.max_dte,
                loader=loader, um=um, oe=oe,
            )

        if sel is None or sel.symbol not in self._option_contracts:
            return None

        spec = self._registry.get_or_default(product)
        boost_notional = spot * spec.contract_multiplier * self._boost_cfg.boost_notional_pct
        opt_price = sel.premium
        if opt_price <= 0:
            return None

        lots = int(boost_notional / (opt_price * spec.contract_multiplier))
        lots = min(lots, self._boost_cfg.boost_max_lots)
        lots = max(lots, 1)

        return (sel.symbol, float(lots), sel.expiry)

    def _select_put_spread_for_hedge(
        self, product: str, date: pd.Timestamp, spot: float,
    ) -> tuple[str, float, str] | None:
        """选择 Put Spread 尾部保护."""
        from ..options.underlying_map import UnderlyingMap
        from ..options.expiry import OptionExpiry
        loader = self._dl
        um = UnderlyingMap(loader)
        oe = OptionExpiry(loader)

        spread = select_put_spread(
            product, date, spot,
            buy_delta=self._boost_cfg.put_spread_delta,
            sell_delta=self._boost_cfg.put_spread_wing,
            min_days=self._boost_cfg.min_dte, max_days=self._boost_cfg.max_dte,
            loader=loader, um=um, oe=oe,
        )
        if spread is None:
            return None

        # 检查两腿都有 option_contracts
        if (spread.long_put.symbol not in self._option_contracts or
            spread.short_put.symbol not in self._option_contracts):
            return None

        return (spread, spread.net_premium)

    # ── Roll 管理 ──────────────────────────────────────────────────────────

    def _check_option_roll(self, product: str, date: pd.Timestamp) -> bool:
        """检查期权是否需要换月."""
        if product not in self._option_holdings:
            return False
        h = self._option_holdings[product]
        expiry = pd.Timestamp(h.get("expiry", "2099-01-01"))
        days_left = (expiry - date).days
        return days_left <= self._boost_cfg.roll_dte

    def _check_option_signal_exit(self, product: str, date: pd.Timestamp) -> bool:
        """检查是否需要因信号反转而退出."""
        if product not in self._option_holdings:
            return False
        h = self._option_holdings[product]
        # 持仓超过 30 天退出
        days_held = (date - pd.Timestamp(h["entry_date"])).days
        return days_held > 30

    def _close_option(self, product: str, targets: dict):
        """平仓期权."""
        if product in self._option_holdings:
            h = self._option_holdings[product]
            targets[h["symbol"]] = 0.0
            self._option_holdings.pop(product, None)

    # ── Put Spread 保护 ──────────────────────────────────────────────────

    def _manage_put_spread(self, date: pd.Timestamp, futures_targets: dict):
        """管理 Put Spread 尾部保护."""
        if not self._boost_cfg.enable_put_spread:
            return

        # 检查现有 put spread 持仓是否需要换月
        for product in list(self._put_spread_holdings.keys()):
            h = self._put_spread_holdings[product]
            expiry = pd.Timestamp(h.get("expiry", "2099-01-01"))
            if (expiry - date).days <= self._boost_cfg.roll_dte:
                # 平旧仓
                for sym in h.get("symbols", {}):
                    futures_targets[sym] = 0.0
                self._put_spread_holdings.pop(product, None)

        # 对有期货多头的品种, 买 put spread 保护
        for product in self._products:
            if product in self._put_spread_holdings:
                continue  # 已持有

            # 检查是否有期货多头
            futures_qty = futures_targets.get(product, 0)
            if futures_qty <= 0:
                continue  # 只保护多头

            # 检查预算
            # (简化: 每品种最多 1 个 put spread)

            # 选择 put spread
            spot = self._price_panel.loc[date, product] if date in self._price_panel.index else None
            if spot is None or np.isnan(spot):
                continue

            result = self._select_put_spread_for_hedge(product, date, spot)
            if result is not None:
                spread, cost = result
                futures_targets[spread.long_put.symbol] = 1.0
                futures_targets[spread.short_put.symbol] = -1.0
                self._put_spread_holdings[product] = {
                    "symbols": {spread.long_put.symbol: 1.0, spread.short_put.symbol: -1.0},
                    "expiry": spread.long_put.expiry,
                    "cost": cost,
                }

    # ── 主逻辑 ──────────────────────────────────────────────────────────

    def _signal_to_lots(self, signals: pd.DataFrame, cfg: ModuleConfig) -> pd.DataFrame:
        if self._iv_panel is None:
            self._load_options_data()
        weights = vol_target_positions(
            signals, self._price_panel, target_vol=cfg.target_vol,
            vol_lookback=cfg.vol_lookback, max_leverage=cfg.max_leverage,
        )
        if self._filter is not None:
            weights = weights * self._filter.coefficients(self._price_panel)
        lots = self._weights_to_lots(weights, cfg)
        self._precomputed_lots = lots  # 存储预计算结果
        return lots

    def _precompute_option_targets(self) -> dict[pd.Timestamp, dict[str, float]]:
        """预计算每个交易日的期权目标 (避免日循环中加载数据).

        支持:
        - 每品种多个期权同时加仓 (max_options_per_product)
        - 期权覆盖 100% 期货方向敞口 (boost_notional_pct=1.0)
        - 组合权利金预算 (max_premium_pct)

        Returns:
            {date: {option_symbol: qty}}
        """
        targets_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        # holdings: {product: [{symbol, qty, entry_date, expiry, ...}]}
        holdings: dict[str, list[dict]] = {}

        dates = self._precomputed_lots.index
        cfg = self._boost_cfg

        for date in dates:
            targets: dict[str, float] = {}

            # 1. Roll / 退出检查 (每个持仓独立)
            for product in list(holdings.keys()):
                remaining = []
                for h in holdings[product]:
                    exp = h.get("expiry") or "2099-01-01"
                    expiry = pd.Timestamp(exp) if exp else pd.Timestamp("2099-01-01")
                    days_left = (expiry - date).days
                    days_held = (date - pd.Timestamp(h["entry_date"])).days
                    if days_left <= cfg.roll_dte or days_held > 30:
                        targets[h["symbol"]] = 0.0  # 平仓
                    else:
                        remaining.append(h)
                if remaining:
                    holdings[product] = remaining
                else:
                    holdings.pop(product, None)

            # 2. 计算当前组合权利金占用 (估算)
            # 用持仓量 × 权利金近似 (需要记录 premium)
            total_premium_used = sum(
                h.get("premium", 0.0) * h.get("qty", 0) * h.get("mult", 1.0)
                for hs in holdings.values() for h in hs
            )

            # 3. 期货多头/空头 + VRP 确认 → 裸买 Call/Put
            row = self._precomputed_lots.loc[date]
            for product in self._products:
                if product not in row.index or abs(row[product]) < 1:
                    continue

                futures_qty = float(row[product])
                vrp_signal = self._get_vrp_signal(product, date)
                if abs(vrp_signal) < 0.01:
                    continue

                futures_direction = np.sign(futures_qty)
                if np.sign(vrp_signal) != futures_direction:
                    continue

                iv_rank = self._get_iv_rank(product, date)
                if iv_rank > cfg.iv_high_threshold:
                    continue

                iv_z = self._get_iv_zscore(product, date)
                if abs(iv_z) > cfg.iv_zscore_max:
                    continue

                size_mult = self._get_vrp_size_multiplier(product, date)
                if size_mult <= 0:
                    continue

                # 每品种持仓数限制
                n_hold = len(holdings.get(product, []))
                if n_hold >= cfg.max_options_per_product:
                    continue

                spot = self._price_panel.loc[date, product] if date in self._price_panel.index else None
                if spot is None or np.isnan(spot) or spot <= 0:
                    continue

                # 选择裸买 Call/Put (内存版, 快)
                sel = self._select_atm_from_cache(product, date, spot, futures_direction)
                if sel is None:
                    continue
                opt_symbol, opt_premium, opt_expiry = sel
                if opt_symbol not in self._option_contracts:
                    continue

                spec = self._registry.get_or_default(product)
                mult = spec.contract_multiplier

                # 单个期权覆盖的敞口 = 期货名义 × boost_pct / max_options
                per_option_ratio = cfg.boost_notional_pct / max(1, cfg.max_options_per_product)
                boost_notional = spot * mult * per_option_ratio
                if opt_premium <= 0:
                    continue
                opt_qty = int(boost_notional / (opt_premium * mult))
                opt_qty = min(max(opt_qty, 1), cfg.boost_max_lots)
                opt_qty = max(1, int(opt_qty * size_mult))  # VRP 动态仓位

                # 组合权利金预算检查
                premium_cost = opt_premium * opt_qty * mult
                equity_base = getattr(self._config, "initial_cash", 5_000_000)
                if total_premium_used + premium_cost > equity_base * cfg.max_premium_pct:
                    continue

                targets[opt_symbol] = float(opt_qty)
                holdings.setdefault(product, []).append({
                    "symbol": opt_symbol,
                    "qty": opt_qty,
                    "entry_date": str(date.date()),
                    "direction": futures_direction,
                    "expiry": opt_expiry,
                    "premium": opt_premium,
                    "mult": mult,
                })
                total_premium_used += premium_cost

            targets_by_date[date] = targets

        self._precomputed_option_targets = targets_by_date
        return targets_by_date

    def target_positions(self, date, data) -> dict[str, float]:
        """增强版: 期货 + 裸买期权 + Put Spread 保护."""
        if not hasattr(self, "_precomputed_lots") or self._precomputed_lots is None:
            return {}

        if date not in self._precomputed_lots.index:
            return {}

        futures_targets = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                futures_targets[product] = float(row[product])

        # 叠加预计算的期权目标
        if hasattr(self, "_precomputed_option_targets") and date in self._precomputed_option_targets:
            opt_targets = self._precomputed_option_targets[date]
            # 平仓 (target=0) 优先
            for sym, qty in opt_targets.items():
                futures_targets[sym] = qty

        # Put Spread 尾部保护 (当前简化跳过, 保留接口)
        # self._manage_put_spread(date, futures_targets)

        return futures_targets
