# coding=utf-8
"""CTA 期权覆盖 A/B: 裸买 Put vs Put Spread (Track A, docs/100 §三).

对比:
    baseline  (无对冲)
    naked_put (裸买 0.25Δ Put)
    put_spread (买 0.25Δ + 卖 0.10Δ Put, 保护成本降 50-60%)

hedge_ratio=0.25: 对冲 25% 的 delta 敞口 (n_lots = qty × 0.25 / |delta|).

用法:
    python scripts/cta_overlay_ab.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.hedge_data import load_basis_panel
from QuantNodes.strategy.cta.options.hedge_layer import (
    HedgeConfig, timing_coefficient,
)
from QuantNodes.strategy.cta.options.iv_panel import add_percentile, load_iv_panel
from QuantNodes.strategy.cta.options.selector import select_put, select_put_spread
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)
from QuantNodes.strategy.cta.backtest.cta_contract import (
    CtaContractRegistry, CtaContract, OptionContractSpec,
)
from QuantNodes.strategy.cta.backtest.cta_broker import CtaBroker
from QuantNodes.strategy.cta.backtest.cta_engine import CtaBacktestConfig, run_cta_backtest

CFG = ComboConfig(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True,
)

INITIAL_CASH = 1_000_000.0


class _OverlaySim:
    """买入持有保护模拟 (支持裸 Put 和 Put Spread)."""

    def __init__(self, mode: str, prem_frames: dict, iv_pct, basis_pct,
                 cfg: HedgeConfig | None = None):
        self.mode = mode  # 'naked_put' | 'put_spread'
        self.prem_frames = prem_frames
        self.iv_pct = iv_pct
        self.basis_pct = basis_pct
        self.cfg = cfg or HedgeConfig(target_delta=0.25, hedge_ratio=0.25)
        self.loader = CtaDataLoader()
        self.um = UnderlyingMap(self.loader)
        self.oe = OptionExpiry(self.loader)
        self._sel_cache: dict = {}
        self._spread_cache: dict = {}

    def _select_naked(self, product, ts, px, key):
        if key in self._sel_cache:
            return self._sel_cache[key]
        sel = select_put(product, ts, px, target_delta=self.cfg.target_delta,
                         min_days=self.cfg.min_days, max_days=self.cfg.max_days,
                         loader=self.loader, um=self.um, oe=self.oe)
        self._sel_cache[key] = sel
        return sel

    def _select_spread(self, product, ts, px, key):
        if key in self._spread_cache:
            return self._spread_cache[key]
        spread = select_put_spread(product, ts, px,
                                   buy_delta=self.cfg.target_delta,
                                   sell_delta=0.10,
                                   min_days=self.cfg.min_days,
                                   max_days=self.cfg.max_days,
                                   loader=self.loader, um=self.um, oe=self.oe)
        self._spread_cache[key] = spread
        return spread

    def run(self, positions, prices, registry, dates):
        idx = pd.DatetimeIndex(dates)
        n = len(dates)
        pos_arr = {p: positions[p].reindex(idx).fillna(0.0).to_numpy()
                    for p in positions.columns}
        px_arr = {p: prices[p].reindex(idx).to_numpy() if p in prices.columns
                  else np.full(n, np.nan) for p in positions.columns}

        holdings: dict[str, dict] = {}
        cash = 0.0
        equity = np.zeros(n)
        n_opened = 0
        n_skipped = 0

        for i in range(n):
            ts = dates[i]
            if i % 500 == 0:
                print(f"  [{self.mode}] day {i}/{n} opened={n_opened} skipped={n_skipped} "
                      f"holdings={len(holdings)} cache={len(self._spread_cache if self.mode=='put_spread' else self._sel_cache)}", flush=True)

            for product in positions.columns:
                qty = pos_arr[product][i]
                if qty <= 0:
                    continue
                px = px_arr[product][i]
                if np.isnan(px) or px <= 0:
                    continue

                h = holdings.get(product)
                if h is not None:
                    if (pd.Timestamp(h["expiry"]) - ts).days <= self.cfg.roll_days:
                        holdings.pop(product)
                        h = None
                if h is not None:
                    n_skipped += 1
                    continue

                key = f"{product}|{ts.strftime('%Y-%m')}"
                mult = registry.get_or_default(product).contract_multiplier

                if self.mode == "naked_put":
                    sel = self._select_naked(product, ts, px, key)
                    if sel is None:
                        n_skipped += 1
                        continue
                    n_lots = int(np.floor(qty * self.cfg.hedge_ratio / abs(sel.delta)))
                    if n_lots <= 0:
                        n_skipped += 1
                        continue
                    cash -= n_lots * sel.premium * mult
                    holdings[product] = {
                        "type": "naked", "qty": n_lots,
                        "symbol": sel.symbol, "expiry": pd.Timestamp(sel.expiry),
                        "mult": mult, "last_prem": sel.premium,
                    }
                else:  # put_spread
                    spread = self._select_spread(product, ts, px, key)
                    if spread is None:
                        n_skipped += 1
                        continue
                    n_lots = int(np.floor(qty * self.cfg.hedge_ratio / abs(spread.long_put.delta)))
                    if n_lots <= 0:
                        n_skipped += 1
                        continue
                    # 净支出 (多腿 premium − 空腿 premium) × 手数 × 乘数
                    net_cost = n_lots * spread.net_premium * mult
                    cash -= net_cost
                    holdings[product] = {
                        "type": "spread", "qty": n_lots,
                        "long_sym": spread.long_put.symbol,
                        "short_sym": spread.short_put.symbol,
                        "long_prem": spread.long_put.premium,
                        "short_prem": spread.short_put.premium,
                        "expiry": pd.Timestamp(spread.long_put.expiry),
                        "mult": mult, "last_prem": spread.net_premium,
                        "strike_long": spread.long_put.strike,
                        "strike_short": spread.short_put.strike,
                    }
                    n_opened += 1

            # mark-to-market
            value = 0.0
            for product, h in list(holdings.items()):
                expired = pd.Timestamp(h["expiry"]) == ts
                if expired:
                    s = px_arr[product][i]
                    if h["type"] == "naked":
                        settle = max((h.get("strike", 0) - s), 0) * h["qty"] * h["mult"] \
                            if not np.isnan(s) else 0.0
                        # 简化: 用 last_prem 作为参考
                    else:
                        settle = 0.0  # Spread 到期: 两腿同时结算, 净值 = 0
                    cash += settle
                    holdings.pop(product)
                else:
                    # 用 last_prem 标记
                    prem = h["last_prem"]
                    if h["type"] == "naked":
                        piv = self.prem_frames.get(product)
                        if piv is not None and h["symbol"] in piv.columns:
                            ser = piv[h["symbol"]]
                            known = ser[ser.index <= ts]
                            if not known.empty:
                                prem = float(known.iloc[-1])
                        value += h["qty"] * prem * h["mult"]
                    else:
                        # Spread mark: 两腿分别标记
                        for sym, is_long in [(h["long_sym"], True), (h["short_sym"], False)]:
                            piv = self.prem_frames.get(product)
                            mp = 0.0
                            if piv is not None and sym in piv.columns:
                                ser = piv[sym]
                                known = ser[ser.index <= ts]
                                if not known.empty:
                                    mp = float(known.iloc[-1])
                            sign = 1.0 if is_long else -1.0
                            value += sign * h["qty"] * mp * h["mult"]
                    h["last_prem"] = prem
            equity[i] = cash + value

        return pd.Series(np.diff(np.concatenate([[0], equity])) / INITIAL_CASH, index=idx)


def main():
    import pickle
    from dataclasses import replace
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

    cache = Path("/tmp/opencode/cta_hedge_cache.pkl")
    if cache.exists():
        with cache.open("rb") as f:
            d = pickle.load(f)
        dates = d["dates"]
        positions = d["positions"]
        price_panel = d["price_panel"]
        base_nav = d["base_nav"]
        registry = d["registry"]
        print(f"CTA (cached): Sharpe={d['metrics']['Sharpe']:.3f}")
    else:
        print("Running CTA ...")
        s = FactorModuleComboStrategy(config=CFG)
        r = s.run()
        dates = [pd.Timestamp(dd) for dd in r.dates]
        idx = pd.DatetimeIndex(dates)
        positions = pd.DataFrame(r.positions, index=idx)
        price_panel = pd.DataFrame(
            {p: s._price_panel[p] for p in positions.columns if p in s._price_panel.columns}, index=idx)
        registry = s._registry
        base_nav = r.nav
        with cache.open("wb") as f:
            pickle.dump({"dates": dates, "positions": positions, "price_panel": price_panel,
                         "base_nav": base_nav, "metrics": r.metrics, "registry": registry}, f)

    # 子集: 2023-01 起 (期权覆盖充分)
    mask = pd.DatetimeIndex(dates) >= "2023-01-01"
    dates = [t for t, m in zip(dates, mask) if m]
    positions = positions.loc[dates]
    price_panel = price_panel.loc[dates]
    base_nav = base_nav.loc[dates]

    dl = CtaDataLoader()
    opt_products = set(dl.list_option_products())
    ever_long = set(positions.columns[positions.where(positions > 0).notna().any()])
    hedge_products = sorted(ever_long & opt_products)

    iv_pct, basis_pct = load_iv_panel(start=dates[0].strftime("%Y-%m-%d"),
                                       end=dates[-1].strftime("%Y-%m-%d")), None
    iv_pct = add_percentile(iv_pct)
    basis_raw = load_basis_panel(start=dates[0].strftime("%Y-%m-%d"),
                                  end=dates[-1].strftime("%Y-%m-%d"))
    basis_pct = add_percentile(basis_raw) if not basis_raw.empty else basis_raw
    print(f"IV: {list(iv_pct.columns)[:5]}... basis: {list(basis_pct.columns)[:5]}...")

    # 加载期权 premium
    print(f"Loading option close ({len(hedge_products)} products) ...")
    t1 = time.time()
    prem_frames = {}
    start, end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    for p in hedge_products:
        df = dl.load_v_all_options(products=[p], start=start, end=end)
        if df is not None and not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["symbol"] = df["symbol"].str.upper()
            prem_frames[p] = df.pivot_table(index="trade_date", columns="symbol", values="close")
    print(f"Loaded in {time.time()-t1:.0f}s")

    base_ret = base_nav.pct_change().dropna()
    base_m = compute_metrics(base_nav)

    print(f"\n{'mode':12s} {'Sharpe':>7s} {'AnnRet':>7s} {'MaxDD':>7s} "
          f"{'最差单日':>8s} {'成本/年':>8s}")
    print(f"{'baseline':12s} {base_m['Sharpe']:7.3f} {base_m['AnnRet']:7.3f} "
          f"{base_m['MaxDD']:7.3f} {base_ret.min()*100:7.2f}% {'—':>8s}")

    cfg = HedgeConfig(target_delta=0.25, hedge_ratio=0.25, min_days=30, max_days=60, roll_days=7)
    for mode in ["naked_put", "put_spread"]:
        sim = _OverlaySim(mode, prem_frames, iv_pct, basis_pct, cfg=cfg)
        t0 = time.time()
        pnl = sim.run(positions, price_panel, registry, dates)
        hedged = base_nav + pnl.cumsum()
        m = compute_metrics(hedged)
        ret = hedged.pct_change().dropna()
        yrs = len(dates) / 252
        print(f"{mode:12s} {m['Sharpe']:7.3f} {m['AnnRet']:7.3f} {m['MaxDD']:7.3f} "
              f"{ret.min()*100:7.2f}% {pnl.sum()*100/yrs:7.2f}%  "
              f"[{time.time()-t0:.0f}s cache={len(sim._sel_cache)}]")


if __name__ == "__main__":
    main()