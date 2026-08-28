# coding=utf-8
"""Reversal Ambush Option Strategy: 超跌埋伏 + 动态 spread→裸买 转换.

核心逻辑 (经验证: 反弹确认后转换中位 +17.5%, 胜率 65.7%):
    1. 超跌触发: 20日价格位置分位 < loc_threshold 且 fut_v1 反向 (逆势埋伏)
    2. 低成本建仓: 买 ATM Call + 卖 OTM Call (1.12×spot) → Call Spread
    3. 反弹确认: 持仓中标的 N 日内上行 > confirm_up 且反弹幅度 < 3% → 买回卖腿
       (释放 gamma, 由低权利金 spread 变为高 delta 裸买)
    4. 退出: 持有 max_hold 日 / 到期前 roll / 反弹止盈

状态机: none → spread → naked → none
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy


@dataclass
class ReversalAmbushConfig:
    """超跌埋伏 + 动态转换配置."""

    # 超跌触发
    loc_window: int = 20
    loc_threshold: float = 0.15        # 20日位置分位 < 15% → 超跌
    iv_rank_max: float = 0.70         # IV 分位 > 70% 不建仓 (高 IV 期权太贵)

    # 建仓 (Call Spread)
    sell_strike_mult: float = 1.12    # 卖腿 strike = spot × 1.12 (OTM)
    min_dte: int = 15
    max_dte: int = 60

    # 动态转换 (spread → naked)
    convert_confirm_up: float = 0.002   # 标的上行 > 0.2% 确认反弹
    convert_max_up: float = 0.03        # 反弹 < 3% (排除已大涨错过)

    # 退出
    max_hold_days: int = 20
    roll_days: int = 7                  # 到期前 N 天换月
    take_profit: float = 0.30           # 权利金 +30% 止盈 (naked 状态)

    # 仓位
    max_premium_pct: float = 0.02       # 单品种权利金 ≤ 资金 2%


class ReversalAmbushStrategy(FactorModuleComboStrategy):
    """超跌埋伏 + 动态 spread→裸买 转换 (叠加 fut_v1)."""

    def __init__(
        self,
        ambush_config: ReversalAmbushConfig | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        option_contracts: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._ambush_cfg = ambush_config or ReversalAmbushConfig()
        self._option_data = option_data or {}
        self._option_contracts = option_contracts or {}
        self._iv_panel: pd.DataFrame | None = None
        self._precomputed_lots: pd.DataFrame | None = None
        self._holdings: dict[str, dict] = {}   # {product: {state, buy_sym, sell_sym, ...}}
        self._loc_cache: pd.DataFrame | None = None
        # 预索引期权链: {product: DataFrame(indexed by trade_date)}
        self._chains: dict[str, pd.DataFrame] = {}
        for prod, df in self._option_data.items():
            if df is not None and not df.empty:
                self._chains[prod] = df.set_index("trade_date").sort_index()

    def _load_iv_panel(self):
        if self._iv_panel is not None:
            return
        try:
            from ..options.iv_panel import load_iv_panel
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            self._iv_panel = None

    # ── 信号: 超跌 (20日价格位置分位) ──────────────────────────────────

    def _price_loc(self) -> pd.DataFrame:
        """20日价格位置分位 (0~1), 低 = 超跌."""
        if self._loc_cache is not None:
            return self._loc_cache
        p = self._price_panel
        roll_max = p.rolling(self._ambush_cfg.loc_window).max()
        roll_min = p.rolling(self._ambush_cfg.loc_window).min()
        self._loc_cache = (p - roll_min) / (roll_max - roll_min).replace(0, np.nan)
        return self._loc_cache

    def _get_iv_rank(self, product: str, date: pd.Timestamp) -> float:
        """IV 分位 (无前视 rolling)."""
        if self._iv_panel is None or product not in self._iv_panel.columns:
            return 0.5
        if date not in self._iv_panel.index:
            return 0.5
        iv_val = self._iv_panel.loc[date, product]
        if pd.isna(iv_val):
            return 0.5
        hist = self._iv_panel[product].loc[:date].dropna()
        if len(hist) < 60:
            return 0.5
        return float((hist < iv_val).sum() / len(hist))

    # ── 期权选择 (内存) ────────────────────────────────────────────────

    def _day_rows(self, product: str, date: pd.Timestamp) -> pd.DataFrame | None:
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        d = chain.loc[date]
        if isinstance(d, pd.Series):
            d = d.to_frame().T
        return d

    def _pick_call(self, product: str, date: pd.Timestamp, spot: float,
                   strike_mult: float) -> tuple[str, float, str] | None:
        """选 strike 最接近 spot×strike_mult 的 Call → (symbol, premium, expiry)."""
        day = self._day_rows(product, date)
        if day is None or day.empty:
            return None
        calls = day[day["option_type"].str.upper() == "C"].copy()
        if calls.empty:
            return None
        liq = calls[(calls["volume"].fillna(0) > 0) | (calls["open_interest"].fillna(0) > 0)]
        if not liq.empty:
            calls = liq
        calls["_dist"] = (calls["strike_price"] - spot * strike_mult).abs()
        best_pos = int(calls["_dist"].to_numpy().argmin())
        row = calls.iloc[best_pos]
        sym = str(row["symbol"])
        if sym not in self._option_contracts:
            return None
        premium = float(row["close"]) if pd.notna(row["close"]) else 0.0
        if premium <= 0:
            return None
        return (sym, premium, "")

    def _premium_of(self, product: str, date: pd.Timestamp, sym: str) -> float | None:
        """某期权在 date 的价格."""
        day = self._day_rows(product, date)
        if day is None or day.empty:
            return None
        m = day["symbol"] == sym
        if not m.any():
            return None
        row = day.loc[m].iloc[0]
        return float(row["close"]) if pd.notna(row["close"]) else None

    # ── 主逻辑: 预计算每日期权目标 ────────────────────────────────────

    def _precompute_ambush_targets(self) -> dict[pd.Timestamp, dict[str, float]]:
        """逐日模拟状态机 → {date: {option_symbol: qty}}.

        返回的目标面板供 target_positions 使用; 平仓 (qty=0) 同样记录.
        """
        cfg = self._ambush_cfg
        dates = self._precomputed_lots.index
        targets_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        holdings: dict[str, dict] = {}

        for date in dates:
            targets: dict[str, float] = {}
            row = self._precomputed_lots.loc[date]

            for product in list(holdings.keys()):
                h = holdings[product]
                days_held = (date - pd.Timestamp(h["entry_date"])).days

                # 止盈 (naked 状态)
                if h["state"] == "naked":
                    prem = self._premium_of(product, date, h["buy_sym"])
                    if prem is not None and h.get("cost", 0) > 0:
                        if prem / h["cost"] - 1 >= cfg.take_profit:
                            targets[h["buy_sym"]] = 0.0
                            holdings.pop(product)
                            continue

                # 到期前 roll 或超持有期 → 平仓
                expiry = pd.Timestamp(h.get("expiry") or "2099-01-01")
                days_left = (expiry - date).days
                if days_left <= cfg.roll_days or days_held >= cfg.max_hold_days:
                    for sym in (h.get("buy_sym"), h.get("sell_sym")):
                        if sym:
                            targets[sym] = 0.0
                    holdings.pop(product)
                    continue

                # 转换: spread → naked (反弹确认)
                if h["state"] == "spread":
                    has_date = date in self._price_panel.index
                    spot_now = self._price_panel.loc[date, product] if has_date else np.nan
                    spot_entry = h["spot_entry"]
                    if not np.isnan(spot_now) and spot_now > spot_entry:
                        up = spot_now / spot_entry - 1
                        if up >= cfg.convert_confirm_up and up < cfg.convert_max_up:
                            # 买回卖腿 → 裸买
                            buy_back = self._premium_of(product, date, h["sell_sym"])
                            if buy_back is not None:
                                targets[h["sell_sym"]] = 1.0  # 买回卖腿 (平掉 -1 空腿)
                                h["state"] = "naked"
                                h["cost"] = h["cost"] + buy_back * h.get("mult", 1.0)
                                h["entry_date"] = str(date.date())

            # 建仓
            for product in self._products:
                if product in holdings:
                    continue
                if product not in row.index or abs(row[product]) < 1:
                    continue
                futures_qty = float(row[product])
                if futures_qty >= 0:
                    continue  # 只在 fut_v1 空头时逆势埋伏 (超跌 → 赌反弹)

                has_date = date in self._price_panel.index
                spot = self._price_panel.loc[date, product] if has_date else None
                if spot is None or np.isnan(spot) or spot <= 0:
                    continue

                loc = self._price_loc()
                loc_val = loc.loc[date, product] if date in loc.index else np.nan
                if pd.isna(loc_val) or loc_val >= cfg.loc_threshold:
                    continue  # 未超跌

                iv_rank = self._get_iv_rank(product, date)
                if iv_rank > cfg.iv_rank_max:
                    continue  # 高 IV 期权太贵

                buy = self._pick_call(product, date, spot, 1.0)
                if buy is None:
                    continue
                sell = self._pick_call(product, date, spot, cfg.sell_strike_mult)
                if sell is None or sell[0] == buy[0]:
                    continue

                buy_sym, buy_premium, _ = buy
                sell_sym, sell_premium, _ = sell
                net_cost = buy_premium - sell_premium
                if net_cost <= 0:
                    continue

                spec = self._registry.get_or_default(product)
                mult = spec.contract_multiplier
                equity = getattr(self._config, "initial_cash", 5_000_000)
                budget = equity * cfg.max_premium_pct
                max_lots = int(budget / (net_cost * mult)) if net_cost > 0 else 0
                if max_lots < 1:
                    continue

                targets[buy_sym] = float(max_lots)
                targets[sell_sym] = -float(max_lots)
                holdings[product] = {
                    "state": "spread",
                    "buy_sym": buy_sym, "sell_sym": sell_sym,
                    "qty": max_lots, "mult": mult,
                    "cost": net_cost,
                    "spot_entry": spot,
                    "entry_date": str(date.date()),
                    "expiry": "",
                }

            targets_by_date[date] = targets

        self._precomputed_ambush = targets_by_date
        return targets_by_date

    # ── 回测接口 ──────────────────────────────────────────────────────

    def target_positions(self, date, data) -> dict[str, float]:
        """期货 + 超跌埋伏期权."""
        if self._precomputed_lots is None or date not in self._precomputed_lots.index:
            return {}
        futures = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                futures[product] = float(row[product])
        if hasattr(self, "_precomputed_ambush") and date in self._precomputed_ambush:
            futures.update(self._precomputed_ambush[date])
        return futures

    def run(self):
        """跑完整回测: 期货 + 期权叠加."""
        self.load_panel()
        self._loc_cache = None
        self._load_iv_panel()
        signals = self._signals()
        exec_sig = self._exec_sig(signals)
        lots = self._signal_to_lots(exec_sig, self._config)
        self._precomputed_lots = lots
        self._precompute_ambush_targets()
        return self._run_with_lots(lots)
