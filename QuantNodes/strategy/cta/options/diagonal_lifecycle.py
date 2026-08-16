# coding=utf-8
"""Diagonal Lifecycle Strategy: 对角线生命周期期权策略 (docs/103).

三腿结构 + 日频 gamma scalping:
    Phase 1 — 底部入场 (loc<threshold 且 fut_v1 空头):
        买 near OTM call (高gamma, 捕捉反弹启动)
        买 far  OTM call (低theta, 保留尾部收益)
    Phase 2 — 反弹确认 (窗口内上行 ∈ [min, max)):
        卖 near ATM call (short theta/vega) → 对角线价差
    Phase 3 — Gamma Scalping (日频):
        组合 |delta| > 阈值 → 期货调回 0
    Phase 4 — 退出:
        到期前 roll_days 平腿 / 权利金止盈 (所有腿)
        止损仅对裸卖腿 (买权保护消失后) — 买权腿风险已定义为权利金, 不止损
        确认超时平仓

状态机: flat → entered → diagonal → flat
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy


@dataclass
class DiagonalLifecycleConfig:
    """对角线生命周期配置."""

    # 底部触发
    loc_window: int = 20
    loc_threshold: float = 0.15      # 20日价格位置分位 < 15% → 超跌
    iv_rank_max: float = 0.70        # IV 分位 > 70% 不入场

    # 腿构建
    near_min_dte: int = 15
    near_max_dte: int = 40
    far_min_dte: int = 45
    far_max_dte: int = 75
    buy_strike_mult: float = 1.00    # 底仓/远仓 OTM 倍数
    sell_strike_mult: float = 1.00   # 卖腿 ATM 倍数 (相对入场 spot)

    # 反弹确认
    confirm_window: int = 5          # 确认窗口 (日)
    confirm_min_up: float = 0.002    # 上行 > 0.2%
    confirm_max_up: float = 0.03     # 上行 < 3% (排除已大涨)

    # gamma scalping
    scalp_threshold: float = 0.5     # 组合 |delta| 超过 → 期货调整

    # 退出
    roll_days: int = 7               # 到期前 N 天平腿
    take_profit: float = 0.50        # 权利金 +50% 止盈
    stop_loss: float = 0.30          # 止损仅对裸卖腿生效 (无买权保护时)

    # 仓位 (delta 市值口径)
    delta_notional_pct: float = 0.30   # 期权层 delta 市值占 NAV 比例
    far_share: float = 0.5             # far 腿占 delta 预算比例 (near 拿 1-far_share)
    far_enabled: bool = True           # False 时只建 near 腿 (单腿对照)


class DiagonalLifecycleStrategy(FactorModuleComboStrategy):
    """对角线生命周期期权策略 (叠加 fut_v1)."""

    def __init__(
        self,
        lifecycle_config: DiagonalLifecycleConfig | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        option_contracts: dict | None = None,
        greeks_cache_path: str | None = None,
        greeks_parquet: str | None = None,
        greeks_cache=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._lc_cfg = lifecycle_config or DiagonalLifecycleConfig()
        self._option_contracts = option_contracts or {}
        self._iv_panel: pd.DataFrame | None = None
        self._precomputed_lots: pd.DataFrame | None = None
        self._loc_cache: pd.DataFrame | None = None
        self._greeks_cache_path = greeks_cache_path
        self._greeks_parquet = greeks_parquet
        self._greeks_cache = greeks_cache
        # 预索引期权链
        self._chains: dict[str, pd.DataFrame] = {}
        for prod, df in (option_data or {}).items():
            if df is not None and not df.empty:
                self._chains[prod] = df.set_index("trade_date").sort_index()
        # 到期日映射 (数据驱动)
        self._expiry_map: dict[str, pd.Timestamp] = {}
        self._build_expiry_map(option_data or {})
        # 统计
        self.stats = {
            "entries": 0, "conversions": 0, "scalps": 0,
            "tp_exits": 0, "sl_exits": 0, "timeout_exits": 0, "expiry_exits": 0,
        }

    def _build_expiry_map(self, option_data: dict[str, pd.DataFrame]):
        """从最后交易日推拿到期日 (每合约链内末次出现日期)."""
        for prod, df in option_data.items():
            if df is None or df.empty:
                continue
            last = df.groupby("symbol")["trade_date"].max()
            for sym, dt in last.items():
                self._expiry_map[str(sym).upper()] = pd.Timestamp(dt)

    # ── 信号 ──────────────────────────────────────────────────────────

    def _price_loc(self) -> pd.DataFrame:
        """20日价格位置分位 (0~1), 低 = 超跌 (缓存)."""
        if self._loc_cache is not None:
            return self._loc_cache
        p = self._price_panel
        w = self._lc_cfg.loc_window
        roll_max = p.rolling(w).max()
        roll_min = p.rolling(w).min()
        self._loc_cache = (p - roll_min) / (roll_max - roll_min).replace(0, np.nan)
        return self._loc_cache

    def _load_iv_panel(self):
        if self._iv_panel is not None:
            return
        try:
            from .iv_panel import load_iv_panel
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            self._iv_panel = None

    def _iv_rank(self, product: str, date: pd.Timestamp) -> float:
        """IV 分位 (无前视)."""
        if self._iv_panel is None or product not in self._iv_panel.columns:
            return 0.5
        if date not in self._iv_panel.index:
            return 0.5
        v = self._iv_panel.loc[date, product]
        if pd.isna(v):
            return 0.5
        hist = self._iv_panel[product].loc[:date].dropna()
        if len(hist) < 60:
            return 0.5
        return float((hist < v).sum() / len(hist))

    # ── 期权链工具 (内存) ─────────────────────────────────────────────

    def _day_calls(self, product: str, date: pd.Timestamp) -> pd.DataFrame | None:
        """当日 call 链 (含流动性过滤)."""
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        day = chain.loc[date]
        if isinstance(day, pd.Series):
            day = day.to_frame().T
        calls = day[day["option_type"].str.upper() == "C"].copy()
        if calls.empty:
            return None
        liq = calls[(calls["volume"].fillna(0) > 0)
                    | (calls["open_interest"].fillna(0) > 0)]
        return liq if not liq.empty else calls

    def _pick_call(
        self, product: str, date: pd.Timestamp, spot: float,
        strike_mult: float, min_dte: int, max_dte: int,
    ) -> dict | None:
        """按 DTE 窗口 + strike 倍数选 call → {symbol, strike, price, expiry}."""
        calls = self._day_calls(product, date)
        if calls is None:
            return None
        exp = calls["symbol"].str.upper().map(self._expiry_map)
        ttm = (exp - date).dt.days
        win = calls[(ttm >= min_dte) & (ttm <= max_dte)]
        if win.empty:
            return None
        dist = (win["strike_price"] - spot * strike_mult).abs()
        row = win.iloc[int(dist.to_numpy().argmin())]
        sym = str(row["symbol"]).upper()
        if sym not in self._option_contracts:
            return None
        px = float(row["close"]) if pd.notna(row["close"]) else 0.0
        if px <= 0:
            return None
        return {
            "symbol": sym, "strike": float(row["strike_price"]),
            "price": px, "expiry": self._expiry_map.get(sym),
        }

    def _price_of(self, product: str, date: pd.Timestamp, sym: str) -> float | None:
        """某期权在 date 的收盘价."""
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        day = chain.loc[date]
        if isinstance(day, pd.Series):
            day = day.to_frame().T
        m = day["symbol"].str.upper() == sym.upper()
        if not m.any():
            return None
        v = day.loc[m].iloc[0]["close"]
        return float(v) if pd.notna(v) else None

    def _ensure_greeks_cache(self):
        if self._greeks_cache is None:
            from .greeks_cache import GreeksCache
            self._greeks_cache = GreeksCache(
                parquet_path=self._greeks_parquet,
                pickle_path=self._greeks_cache_path,
            )

    def _leg_delta(self, product: str, date: pd.Timestamp, leg: dict, spot: float) -> float:
        """反解单腿 delta (优先 parquet 缓存 O(1), 未命中按需反解)."""
        self._ensure_greeks_cache()
        try:
            rec = self._greeks_cache.lookup(product, leg["symbol"], date)
            if rec is not None:
                return float(rec["delta"]) if pd.notna(rec["delta"]) else 0.0
            px = self._price_of(product, date, leg["symbol"])
            if px is None or px <= 0:
                return 0.0
            rec2 = self._greeks_cache.get_or_compute(
                product, date, leg["symbol"], leg["strike"], px, spot,
                leg.get("expiry"), option_type="C",
            )
            return float(rec2["delta"]) if rec2 else 0.0
        except Exception:
            return 0.0

    # ── 状态机: 预计算每日目标 ────────────────────────────────────────

    def _precompute_lifecycle_targets(self) -> dict[pd.Timestamp, dict[str, float]]:
        """逐日状态机 → {date: {symbol: qty}} + 期货 scalp 调整."""
        cfg = self._lc_cfg
        dates = self._precomputed_lots.index
        targets_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        self._scalp_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        holdings: dict[str, dict] = {}

        for date in dates:
            targets: dict[str, float] = {}
            scalp: dict[str, float] = {}
            lots_row = self._precomputed_lots.loc[date]
            has_px = date in self._price_panel.index

            # ── 1. 持仓管理 ──
            for product in list(holdings.keys()):
                h = holdings[product]
                spot = self._price_panel.loc[date, product] if has_px else np.nan
                if np.isnan(spot) or spot <= 0:
                    continue
                days_held = (date - pd.Timestamp(h["entry_date"])).days

                # 1a. entered: 确认转换 / 超时平仓
                if h["state"] == "entered":
                    up = spot / h["entry_spot"] - 1
                    confirmed = (up >= cfg.confirm_min_up and up < cfg.confirm_max_up
                                 and days_held <= cfg.confirm_window
                                 and h.get("near_long") is not None)
                    if confirmed:
                        sell = self._pick_call(
                            product, date, h["entry_spot"],
                            cfg.sell_strike_mult, cfg.near_min_dte, cfg.near_max_dte,
                        )
                        if sell is not None:
                            qty = h["near_long"]["qty"]
                            targets[sell["symbol"]] = -float(qty)
                            h["near_short"] = {
                                "symbol": sell["symbol"], "qty": -float(qty),
                                "entry_px": sell["price"], "strike": sell["strike"],
                                "expiry": sell["expiry"],
                            }
                            h["state"] = "diagonal"
                            self.stats["conversions"] += 1
                    elif days_held > cfg.confirm_window:
                        # 确认超时 → 全平
                        for leg in self._long_legs(h):
                            targets[leg["symbol"]] = 0.0
                        holdings.pop(product)
                        self.stats["timeout_exits"] += 1

                # 1b. 各腿退出检查 (entered/diagonal 通用)
                if product in holdings:
                    h = holdings[product]
                    # 买权保护量: 存活买腿总手数 (卖腿有买权保护 → 风险已定义, 不止损)
                    protected_qty = sum(
                        leg["qty"] for leg in self._all_legs(h) if leg["qty"] > 0
                    )
                    alive = []
                    for leg in self._all_legs(h):
                        px = self._price_of(product, date, leg["symbol"])
                        expiry = leg.get("expiry")
                        days_left = ((pd.Timestamp(expiry) - date).days
                                     if expiry is not None else 999)
                        ret = (px / leg["entry_px"] - 1) if (px and leg["entry_px"] > 0) else 0.0
                        # 卖腿盈利 = 权利金下跌
                        if leg["qty"] < 0:
                            ret = -ret
                        if ret >= cfg.take_profit:
                            targets[leg["symbol"]] = 0.0
                            self.stats["tp_exits"] += 1
                        elif (leg["qty"] < 0
                              and protected_qty < abs(leg["qty"]) - 1e-9
                              and ret <= -cfg.stop_loss):
                            # 止损仅对裸卖腿: 买权保护消失后卖腿变为裸卖才触发
                            targets[leg["symbol"]] = 0.0
                            self.stats["sl_exits"] += 1
                        elif days_left <= cfg.roll_days:
                            targets[leg["symbol"]] = 0.0
                            self.stats["expiry_exits"] += 1
                        else:
                            alive.append(leg)
                    self._set_legs(h, alive)
                    if not alive:
                        holdings.pop(product)

                # 1c. gamma scalping (diagonal: 期货调组合 delta → 0)
                if product in holdings and holdings[product]["state"] == "diagonal":
                    h = holdings[product]
                    port_delta = sum(
                        leg["qty"] * self._leg_delta(product, date, leg, spot)
                        for leg in self._all_legs(h)
                    )
                    adj = -round(port_delta)
                    if abs(port_delta) >= cfg.scalp_threshold and adj != 0:
                        scalp[product] = float(adj)
                        self.stats["scalps"] += 1

            # ── 2. 新入场 ──
            for product in self._products:
                if product in holdings:
                    continue
                if product not in lots_row.index or lots_row[product] >= 0:
                    continue  # 仅 fut_v1 空头时逆势埋伏
                if not has_px:
                    continue
                spot = self._price_panel.loc[date, product]
                if np.isnan(spot) or spot <= 0:
                    continue
                loc = self._price_loc()
                loc_val = loc.loc[date, product] if date in loc.index else np.nan
                if pd.isna(loc_val) or loc_val >= cfg.loc_threshold:
                    continue
                if self._iv_rank(product, date) > cfg.iv_rank_max:
                    continue

                near = self._pick_call(product, date, spot, cfg.buy_strike_mult,
                                       cfg.near_min_dte, cfg.near_max_dte)
                if near is None:
                    continue
                far = None
                if cfg.far_enabled:
                    far = self._pick_call(product, date, spot, cfg.buy_strike_mult,
                                          cfg.far_min_dte, cfg.far_max_dte)
                    if far is None or far["symbol"] == near["symbol"]:
                        continue

                spec = self._registry.get_or_default(product)
                mult = spec.contract_multiplier
                equity = getattr(self._config, "initial_cash", 5_000_000)

                near_delta = abs(self._leg_delta(product, date, near, spot))
                if near_delta < 1e-6:
                    continue
                delta_budget = equity * cfg.delta_notional_pct
                if far is not None:
                    far_delta = abs(self._leg_delta(product, date, far, spot))
                    if far_delta < 1e-6:
                        continue
                    near_qty = int(delta_budget * (1 - cfg.far_share)
                                   / (near_delta * spot * mult))
                    far_qty = int(delta_budget * cfg.far_share
                                  / (far_delta * spot * mult))
                    if near_qty < 1 or far_qty < 1:
                        continue
                else:
                    near_qty = int(delta_budget / (near_delta * spot * mult))
                    far_qty = 0
                    if near_qty < 1:
                        continue

                targets[near["symbol"]] = float(near_qty)
                if far:
                    targets[far["symbol"]] = float(far_qty)
                holdings[product] = {
                    "state": "entered",
                    "entry_date": str(date.date()),
                    "entry_spot": float(spot),
                    "near_long": {
                        "symbol": near["symbol"], "qty": float(near_qty),
                        "entry_px": near["price"], "strike": near["strike"],
                        "expiry": near["expiry"],
                    },
                    "far_long": ({
                        "symbol": far["symbol"], "qty": float(far_qty),
                        "entry_px": far["price"], "strike": far["strike"],
                        "expiry": far["expiry"],
                    } if far else None),
                    "near_short": None,
                }
                self.stats["entries"] += 1

            targets_by_date[date] = targets
            self._scalp_by_date[date] = scalp

        self._precomputed_lifecycle = targets_by_date
        # 持久化 greeks 缓存 (下次回测直接查)
        if self._greeks_cache is not None and self._greeks_cache_path:
            try:
                self._greeks_cache.save_pickle(self._greeks_cache_path)
            except Exception:
                pass
        return targets_by_date

    @staticmethod
    def _long_legs(h: dict) -> list[dict]:
        return [h[k] for k in ("near_long", "far_long") if h.get(k)]

    @staticmethod
    def _all_legs(h: dict) -> list[dict]:
        return [h[k] for k in ("near_long", "far_long", "near_short") if h.get(k)]

    @staticmethod
    def _set_legs(h: dict, alive: list[dict]):
        by_sym = {leg["symbol"] for leg in alive}
        for key in ("near_long", "far_long", "near_short"):
            leg = h.get(key)
            if leg is not None and leg["symbol"] not in by_sym:
                h[key] = None

    # ── 回测接口 ──────────────────────────────────────────────────────

    def target_positions(self, date, data) -> dict[str, float]:
        """fut_v1 期货 + 期权 targets + scalp 期货调整."""
        if self._precomputed_lots is None or date not in self._precomputed_lots.index:
            return {}
        out: dict[str, float] = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                out[product] = float(row[product])
        if hasattr(self, "_precomputed_lifecycle") and date in self._precomputed_lifecycle:
            out.update(self._precomputed_lifecycle[date])
        if hasattr(self, "_scalp_by_date") and date in self._scalp_by_date:
            for product, adj in self._scalp_by_date[date].items():
                out[product] = out.get(product, 0.0) + adj
        return out

    def run(self):
        """完整回测: fut_v1 + 对角线生命周期期权."""
        self.load_panel()
        self._loc_cache = None
        self._load_iv_panel()
        signals = self._signals()
        exec_sig = self._exec_sig(signals)
        lots = self._signal_to_lots(exec_sig, self._config)
        self._precomputed_lots = lots
        self._precompute_lifecycle_targets()
        return self._run_with_lots(lots)
