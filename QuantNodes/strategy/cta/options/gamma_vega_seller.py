# coding=utf-8
"""Gamma-Vega 卖方策略: 大跌后分步建仓卖 put spread (docs/108 §5).

核心逻辑 (验证: 中位+44%, 胜率71%, 优于一次性+38%/67%):
    触发: 连续3日跌 + VRP分位>0.5 + 品种∈good_products
    建仓: 分步加仓 — T日 1/3 仓位; T+1 继续跌加 1/3; T+2 继续跌加 1/3
    (跌时加仓 = 在 put 价格更高时卖出更多 → 平均入场价格更好)
    持有: 每个批次独立持有 hold_days 日后平仓
    结构: 卖 ATM put spread (ATM - OTM put_otm_mult)

注意: 每批次独立核算 (不能用平均成本统一平仓).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy


@dataclass
class GammaVegaConfig:
    """Gamma-Vega 卖方配置."""

    # 触发
    down_days: int = 3                # 连续下跌日数
    vrp_min: float = 0.50             # VRP 分位阈值
    good_products: list[str] = field(
        default_factory=lambda: ["I", "AG", "CU", "SR", "TA", "RU", "M"])

    # 分步建仓
    max_batches: int = 3              # 最大批次
    batch_add_if_down: bool = True    # True=继续跌才加仓; False=时间固定加仓
    batch_size: float = 1.0 / 3.0     # 每批次占目标仓位的比例

    # 结构
    atm_mult: float = 1.00
    put_otm_mult: float = 0.88        # OTM put strike 倍数
    min_dte: int = 15
    max_dte: int = 40

    # 持有
    hold_days: int = 7                # 每批次独立持有天数

    # 仓位
    delta_notional_pct: float = 0.10  # 单品种 delta 市值预算 (NAV %)


class GammaVegaSeller(FactorModuleComboStrategy):
    """大跌后分步建仓卖 put spread (叠加 fut_v1)."""

    def __init__(
        self,
        gv_config: GammaVegaConfig | None = None,
        option_data: dict[str, pd.DataFrame] | None = None,
        option_contracts: dict | None = None,
        greeks_cache=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._gv_cfg = gv_config or GammaVegaConfig()
        self._option_contracts = option_contracts or {}
        self._iv_panel: pd.DataFrame | None = None
        self._precomputed_lots: pd.DataFrame | None = None
        self._greeks_cache = greeks_cache
        self._vrp_pct: pd.DataFrame | None = None
        self._down_cnt: pd.DataFrame | None = None
        self._chains: dict[str, pd.DataFrame] = {}
        for prod, df in (option_data or {}).items():
            if df is not None and not df.empty:
                self._chains[prod] = df.set_index("trade_date").sort_index()
        self._expiry_map: dict[str, pd.Timestamp] = {}
        self._build_expiry_map(option_data or {})
        self.stats = {"entries": 0, "adds": 0, "batch_exits": 0, "skips": 0}

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
        """VRP 分位 / 近5日下跌天数 (rolling, 无前视)."""
        if self._vrp_pct is not None or self._iv_panel is None:
            return
        iv = self._iv_panel.reindex(index=self._price_panel.index,
                                    columns=self._price_panel.columns)
        ret = self._price_panel.pct_change()
        hv20 = ret.rolling(20).std() * np.sqrt(252)
        vrp = iv - hv20
        self._vrp_pct = vrp.rolling(252, min_periods=60).rank(pct=True)
        self._ret = ret
        self._down_cnt = (ret < 0).rolling(5).sum()

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
        win = sub[(ttm >= self._gv_cfg.min_dte) & (ttm <= self._gv_cfg.max_dte)]
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

    def _delta_of(self, product: str, date: pd.Timestamp, pick: dict, spot: float) -> float:
        if self._greeks_cache is not None:
            rec = self._greeks_cache.lookup(product, pick["symbol"], date)
            if rec is not None and pd.notna(rec.get("delta")):
                return float(rec["delta"])
        return 0.0

    # ── 状态机: 分步建仓 ──────────────────────────────────────────────

    def _precompute_gv_targets(self) -> dict[pd.Timestamp, dict[str, float]]:
        """逐日: 批次平仓 + 触发建仓/加仓.

        持仓结构: holdings[product] = {"batches": [{atm_sym, otm_sym, entry_i, qty}], "last_add_i"}
        每批次独立持有 hold_days 日后平仓.
        """
        cfg = self._gv_cfg
        dates = self._precomputed_lots.index
        targets_by_date: dict[pd.Timestamp, dict[str, float]] = {}
        holdings: dict[str, dict] = {}

        for i, date in enumerate(dates):
            targets: dict[str, float] = {}
            has_px = date in self._price_panel.index

            # 1. 批次平仓 (独立持有)
            for product in list(holdings.keys()):
                h = holdings[product]
                remaining = []
                for b in h["batches"]:
                    if i >= b["entry_i"] + cfg.hold_days:
                        targets[b["atm_sym"]] = 0.0
                        targets[b["otm_sym"]] = 0.0
                        self.stats["batch_exits"] += 1
                    else:
                        remaining.append(b)
                h["batches"] = remaining
                if not remaining:
                    holdings.pop(product)

            # 2. 新触发 / 加仓
            if has_px and self._vrp_pct is not None:
                for product in self._products:
                    if product not in cfg.good_products:
                        continue
                    if date not in self._vrp_pct.index:
                        continue
                    v = self._vrp_pct.loc[date, product]
                    if pd.isna(v) or v <= cfg.vrp_min:
                        continue
                    spot = self._price_panel.loc[date, product]
                    if np.isnan(spot) or spot <= 0:
                        continue
                    if self._chains.get(product) is None:
                        continue

                    if product in holdings:
                        # ── 加仓 (已有持仓) ──
                        h = holdings[product]
                        if len(h["batches"]) >= cfg.max_batches:
                            continue
                        if i <= h["last_add_i"]:
                            continue  # 同日不重复加
                        # 加仓条件: 继续跌 (默认) 或 时间固定
                        if cfg.batch_add_if_down:
                            prev = self._ret.loc[dates[i - 1], product] if i > 0 else 0.0
                            if pd.notna(prev) and prev >= 0:
                                continue  # 未继续跌, 不加
                        self._open_batch(product, date, spot, i, targets, holdings,
                                         add=True)
                    else:
                        # ── 新触发 (首批) ──
                        dc = np.nan
                        if date in self._down_cnt.index:
                            dc = self._down_cnt.loc[date, product]
                        if pd.isna(dc) or dc < cfg.down_days:
                            self.stats["skips"] += 1
                            continue
                        self._open_batch(product, date, spot, i, targets, holdings,
                                         add=False)

            targets_by_date[date] = targets

        self._precomputed_gv = targets_by_date
        return targets_by_date

    def _open_batch(self, product: str, date: pd.Timestamp, spot: float, i: int,
                    targets: dict, holdings: dict, add: bool):
        """开一批卖 put spread."""
        cfg = self._gv_cfg
        atm = self._pick(product, date, spot, "P", cfg.atm_mult)
        otm = self._pick(product, date, spot, "P", cfg.put_otm_mult)
        if atm is None or otm is None or atm["symbol"] == otm["symbol"]:
            return
        atm_delta = abs(self._delta_of(product, date, atm, spot))
        if atm_delta < 1e-6:
            atm_delta = 0.5
        spec = self._registry.get_or_default(product)
        mult = spec.contract_multiplier
        equity = getattr(self._config, "initial_cash", 5_000_000)
        # 每批次 delta 市值 = 预算 × batch_size
        qty = int(equity * cfg.delta_notional_pct * cfg.batch_size
                  / (spot * mult * atm_delta))
        if qty < 1:
            return
        # 卖方: 卖 ATM put (负 qty), 买 OTM put (正 qty)
        targets[atm["symbol"]] = -float(qty)
        targets[otm["symbol"]] = float(qty)

        h = holdings.setdefault(product, {"batches": [], "last_add_i": -1})
        h["batches"].append({"atm_sym": atm["symbol"], "otm_sym": otm["symbol"],
                             "entry_i": i, "qty": qty})
        h["last_add_i"] = i
        if add:
            self.stats["adds"] += 1
        else:
            self.stats["entries"] += 1

    # ── 回测接口 ──────────────────────────────────────────────────────

    def target_positions(self, date, data) -> dict[str, float]:
        if self._precomputed_lots is None or date not in self._precomputed_lots.index:
            return {}
        out: dict[str, float] = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                out[product] = float(row[product])
        if hasattr(self, "_precomputed_gv") and date in self._precomputed_gv:
            out.update(self._precomputed_gv[date])
        return out

    def run(self):
        self.load_panel()
        self._load_iv_panel()
        self._compute_signal_panels()
        signals = self._signals()
        exec_sig = self._exec_sig(signals)
        lots = self._signal_to_lots(exec_sig, self._config)
        self._precomputed_lots = lots
        self._precompute_gv_targets()
        return self._run_with_lots(lots)
