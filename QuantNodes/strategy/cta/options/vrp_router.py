# coding=utf-8
"""VRP 路由买方策略 (docs/105 + 106 研究结论).

核心逻辑:
    - 低VRP (< buy_vrp_max): 期权便宜 → 买 spread
      · 超跌反转 (mom20<0): long_products → 买 call spread
      · 超涨转跌 (mom20>0): short_products → 买 put spread
    - 中段 VRP (≥ 0.30): 不参与

仓位: delta 市值口径 (delta_notional_pct × NAV).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy


@dataclass
class VrpRouterConfig:
    """双向 VRP 路由配置."""

    # 买方 VRP 触发
    buy_vrp_max: float = 0.30        # VRP 分位 < 30% → 买 spread
    iv_rank_max: float | None = None # IV 分位过滤 (None=不启用)

    # 品种白名单 (方向由 20 日动量决定)
    long_products: list[str] = field(default_factory=lambda: ["AG", "AU", "CU", "I"])
    short_products: list[str] = field(default_factory=lambda: ["M", "SR", "TA"])

    # 持有期
    long_hold: int = 5               # call spread (超跌反转)
    short_hold: int = 10             # put spread (超涨转跌)

    # 结构
    atm_mult: float = 1.00
    call_otm_mult: float = 1.12      # call spread 卖腿 strike 倍数
    put_otm_mult: float = 0.88       # put spread 卖腿 strike 倍数
    min_dte: int = 15
    max_dte: int = 40

    # 仓位
    delta_notional_pct: float = 0.25 # delta 市值预算 (NAV %)


class VrpRouterStrategy(FactorModuleComboStrategy):
    """低VRP买 spread: 超跌买 call / 超涨买 put (叠加 fut_v1)."""

    def __init__(
        self,
        router_config: VrpRouterConfig | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        option_contracts: dict | None = None,
        greeks_cache=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._rr_cfg = router_config or VrpRouterConfig()
        self._option_contracts = option_contracts or {}
        self._iv_panel: pd.DataFrame | None = None
        self._precomputed_lots: pd.DataFrame | None = None
        self._greeks_cache = greeks_cache
        self._vrp_pct: pd.DataFrame | None = None
        self._mom20: pd.DataFrame | None = None
        self._iv_pct: pd.DataFrame | None = None
        self._chains: dict[str, pd.DataFrame] = {}
        for prod, df in (option_data or {}).items():
            if df is not None and not df.empty:
                self._chains[prod] = df.set_index("trade_date").sort_index()
        self._expiry_map: dict[str, pd.Timestamp] = {}
        self._build_expiry_map(option_data or {})
        self.stats = {"long_buys": 0, "short_buys": 0,
                      "exits": 0, "skip_vrp": 0, "skip_dir": 0}

    def _build_expiry_map(self, option_data: dict[str, pd.DataFrame]):
        for prod, df in option_data.items():
            if df is None or df.empty:
                continue
            last = df.groupby("symbol")["trade_date"].max()
            for sym, dt in last.items():
                self._expiry_map[str(sym).upper()] = pd.Timestamp(dt)

    # ── 信号 ──────────────────────────────────────────────────────────

    def _load_iv_panel(self):
        if self._iv_panel is not None:
            return
        try:
            from .iv_panel import load_iv_panel
            self._iv_panel = load_iv_panel(start=self._start, end=self._end)
        except Exception:
            self._iv_panel = None

    def _compute_signal_panels(self):
        """VRP 分位 / 20日动量 / IV 分位 (rolling, 无前视)."""
        if self._vrp_pct is not None or self._iv_panel is None:
            return
        iv = self._iv_panel.reindex(index=self._price_panel.index,
                                    columns=self._price_panel.columns)
        ret = self._price_panel.pct_change()
        hv20 = ret.rolling(20).std() * np.sqrt(252)
        vrp = iv - hv20
        self._vrp_pct = vrp.rolling(252, min_periods=60).rank(pct=True)
        self._mom20 = ret.rolling(20).sum()
        if self._rr_cfg.iv_rank_max is not None:
            self._iv_pct = iv.rolling(252, min_periods=60).rank(pct=True)

    # ── 期权链工具 ────────────────────────────────────────────────────

    def _day_rows(self, product: str, date: pd.Timestamp) -> pd.DataFrame | None:
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        d = chain.loc[date]
        return d.to_frame().T if isinstance(d, pd.Series) else d

    def _pick(self, product: str, date: pd.Timestamp, spot: float,
              opt_type: str, sm: float) -> dict | None:
        d = self._day_rows(product, date)
        if d is None:
            return None
        sub = d[d["option_type"].str.upper() == opt_type].copy()
        if sub.empty:
            return None
        liq = sub[(sub["volume"].fillna(0) > 0) | (sub["open_interest"].fillna(0) > 0)]
        if not liq.empty:
            sub = liq
        exp = sub["symbol"].str.upper().map(self._expiry_map)
        ttm = (exp - date).dt.days
        win = sub[(ttm >= self._rr_cfg.min_dte) & (ttm <= self._rr_cfg.max_dte)]
        if win.empty:
            return None
        dist = (win["strike_price"] - spot * sm).abs()
        row = win.iloc[int(dist.to_numpy().argmin())]
        sym = str(row["symbol"]).upper()
        if sym not in self._option_contracts:
            return None
        px = float(row["close"]) if pd.notna(row["close"]) else 0.0
        if px <= 0:
            return None
        return {"symbol": sym, "px": px, "strike": float(row["strike_price"]),
                "expiry": self._expiry_map.get(sym)}

    def _price_of(self, product: str, date: pd.Timestamp, sym: str) -> float | None:
        chain = self._chains.get(product)
        if chain is None or date not in chain.index:
            return None
        d = chain.loc[date]
        if isinstance(d, pd.Series):
            d = d.to_frame().T
        m = d["symbol"].str.upper() == sym.upper()
        if not m.any():
            return None
        v = d.loc[m].iloc[0]["close"]
        return float(v) if pd.notna(v) else None

    def _delta_of(self, product: str, date: pd.Timestamp, pick: dict, spot: float) -> float:
        if self._greeks_cache is not None:
            rec = self._greeks_cache.lookup(product, pick["symbol"], date)
            if rec is not None and pd.notna(rec.get("delta")):
                return float(rec["delta"])
        return 0.0

    # ── 状态机 ────────────────────────────────────────────────────────

    def _precompute_router_targets(self) -> dict[pd.Timestamp, dict[str, float]]:
        """逐日: 持仓到期平仓 + 双向 VRP 触发建仓."""
        cfg = self._rr_cfg
        dates = self._precomputed_lots.index
        targets_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        holdings: dict[str, dict] = {}
        hold_pos: dict[str, int] = {}

        for i, date in enumerate(dates):
            targets: dict[str, float] = {}
            has_px = date in self._price_panel.index

            # 1. 到期平仓
            for product in list(holdings.keys()):
                if i >= hold_pos[product] + holdings[product]["hold"]:
                    h = holdings[product]
                    targets[h["atm_sym"]] = 0.0
                    targets[h["otm_sym"]] = 0.0
                    holdings.pop(product)
                    hold_pos.pop(product)
                    self.stats["exits"] += 1

            # 2. 新触发
            if has_px and self._vrp_pct is not None:
                for product in self._products:
                    if product in holdings:
                        continue
                    if date not in self._vrp_pct.index:
                        continue
                    v = self._vrp_pct.loc[date, product]
                    if pd.isna(v):
                        continue
                    # 中段 VRP 不参与
                    if v >= cfg.buy_vrp_max:
                        self.stats["skip_vrp"] += 1
                        continue
                    if self._iv_pct is not None:
                        if date not in self._iv_pct.index:
                            continue
                        ivq = self._iv_pct.loc[date, product]
                        if pd.isna(ivq) or ivq >= cfg.iv_rank_max:
                            self.stats["skip_vrp"] += 1
                            continue
                    spot = self._price_panel.loc[date, product]
                    if np.isnan(spot) or spot <= 0:
                        continue
                    if self._chains.get(product) is None:
                        continue

                    mom = self._mom20.loc[date, product] if date in self._mom20.index else np.nan
                    if pd.isna(mom):
                        continue

                    # ── 方向判定 (买方) ──
                    if mom < 0 and product in cfg.long_products:
                        side, hold = "buy_call", cfg.long_hold
                        otm_mult = cfg.call_otm_mult
                    elif mom > 0 and product in cfg.short_products:
                        side, hold = "buy_put", cfg.short_hold
                        otm_mult = cfg.put_otm_mult
                    else:
                        self.stats["skip_dir"] += 1
                        continue

                    opt_type = "C" if side == "buy_call" else "P"
                    atm = self._pick(product, date, spot, opt_type, cfg.atm_mult)
                    otm = self._pick(product, date, spot, opt_type, otm_mult)
                    if atm is None or otm is None or atm["symbol"] == otm["symbol"]:
                        continue
                    atm_delta = abs(self._delta_of(product, date, atm, spot))
                    if atm_delta < 1e-6:
                        atm_delta = 0.5
                    spec = self._registry.get_or_default(product)
                    mult = spec.contract_multiplier
                    equity = getattr(self._config, "initial_cash", 5_000_000)
                    qty = int(equity * cfg.delta_notional_pct
                              / (spot * mult * atm_delta))
                    if qty < 1:
                        continue

                    targets[atm["symbol"]] = float(qty)
                    targets[otm["symbol"]] = -float(qty)
                    holdings[product] = {
                        "atm_sym": atm["symbol"], "otm_sym": otm["symbol"],
                        "qty": qty, "hold": hold,
                    }
                    hold_pos[product] = i
                    if side == "buy_call":
                        self.stats["long_buys"] += 1
                    else:
                        self.stats["short_buys"] += 1

            targets_by_date[date] = targets

        self._precomputed_router = targets_by_date
        return targets_by_date

    # ── 回测接口 ──────────────────────────────────────────────────────

    def target_positions(self, date, data) -> dict[str, float]:
        if self._precomputed_lots is None or date not in self._precomputed_lots.index:
            return {}
        out: dict[str, float] = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                out[product] = float(row[product])
        if hasattr(self, "_precomputed_router") and date in self._precomputed_router:
            out.update(self._precomputed_router[date])
        return out

    def run(self):
        self.load_panel()
        self._load_iv_panel()
        self._compute_signal_panels()
        signals = self._signals()
        exec_sig = self._exec_sig(signals)
        lots = self._signal_to_lots(exec_sig, self._config)
        self._precomputed_lots = lots
        self._precompute_router_targets()
        return self._run_with_lots(lots)
