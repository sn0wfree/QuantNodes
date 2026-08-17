#!/usr/bin/env python3
"""期权 vs 期货贡献拆解 + 多策略组合 + 优化扫描.

组合1: fut_v1 基线 (Sharpe 2.09)
组合2: fut_v1 + GammaVegaSeller delta=0.25 (Sharpe 2.41)
组合3: fut_v1 + VRP Router (Sharpe 2.17)
组合4: fut_v1 + VRP Router + GammaVegaSeller

优化扫描: stop_loss × hedge_trigger × hedge_reduction
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
from dataclasses import replace
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.gamma_vega_seller import (
    GammaVegaSeller, GammaVegaConfig,
)
from QuantNodes.strategy.cta.options.vrp_router import VrpRouterStrategy, VrpRouterConfig
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
INITIAL = 5_000_000


class OptionOnlyGV(GammaVegaSeller):
    def target_positions(self, date, data):
        if not hasattr(self, "_precomputed_gv") or date not in self._precomputed_gv:
            return {}
        return self._precomputed_gv[date]


class OptionOnlyVR(VrpRouterStrategy):
    """纯期权层 standalone (不叠加期货)."""

    def target_positions(self, date, data):
        if not hasattr(self, "_precomputed_router") or date not in self._precomputed_router:
            return {}
        return self._precomputed_router[date]


class DualStrategy(GammaVegaSeller):
    """同时叠加 VRP Router + GammaVegaSeller."""

    def __init__(self, vrp_cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vrp_cfg = vrp_cfg
        self.option_data = kwargs.get("option_data")
        self._option_contracts = kwargs.get("option_contracts")
        self._greeks_cache = kwargs.get("greeks_cache")

    def target_positions(self, date, data):
        if self._precomputed_lots is None or date not in self._precomputed_lots.index:
            return {}
        out = {}
        row = self._precomputed_lots.loc[date]
        for product in self._products:
            if product in row.index and abs(row[product]) > 0:
                out[product] = float(row[product])
        # VRP Router 目标
        if hasattr(self, "_precomputed_router") and date in self._precomputed_router:
            out.update(self._precomputed_router[date])
        # GammaVegaSeller 目标
        if hasattr(self, "_precomputed_gv") and date in self._precomputed_gv:
            for sym, qty in self._precomputed_gv[date].items():
                cur = out.get(sym, 0.0)
                out[sym] = cur + qty
        return out

    def run(self):
        self.load_panel()
        self._load_iv_panel()
        self._compute_signal_panels()
        # GammaVegaSeller 没有 _mom20/_iv_pct, 用空 DataFrame 作为 fallback
        # (VRP Router 用它们做动量/skew 过滤, 这里不用所以给空)
        vrp = VrpRouterStrategy(router_config=self._vrp_cfg,
            option_data=self.option_data, option_contracts=self._option_contracts,
            greeks_cache=self._greeks_cache, products=self._products,
            config=self._config, start=self._start, end=self._end)
        vrp._price_panel = self._price_panel
        vrp._ohlc_map = self._ohlc_map; vrp._vols = self._vols; vrp._ois = self._ois
        vrp._iv_panel = self._iv_panel; vrp._vrp_pct = self._vrp_pct
        vrp._mom20 = self._vrp_pct.copy() if isinstance(self._vrp_pct, pd.DataFrame) else pd.DataFrame()
        vrp._iv_pct = pd.DataFrame()
        vrp._precomputed_lots = self._precomputed_lots
        vrp._precompute_router_targets()
        self._precomputed_router = vrp._precomputed_router
        # GammaVegaSeller 预计算
        self._precompute_gv_targets()
        return self._run_with_lots(self._precomputed_lots)


def metrics(nav):
    r = nav.pct_change().dropna()
    if len(r) == 0:
        return {}
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann / abs(dd) if abs(dd) > 0 else 0
    return {"Sharpe": sharpe, "AnnRet": ann, "AnnVol": vol,
            "MaxDD": dd, "Calmar": calmar}


def main():
    cfg_base = ComboConfig(
        initial_cash=INITIAL, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
    )
    print("Loading option data...")
    option_data, option_contracts = load_option_data(PRODUCTS, "2023-01-01", "2025-12-31")
    gcache = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    print(f"  contracts={len(option_contracts)}, greeks={gcache.size:,}")

    s0_gv = GammaVegaSeller(
        option_data=option_data, option_contracts=option_contracts,
        greeks_cache=gcache, products=PRODUCTS, config=cfg_base,
        start="2018-01-01", end="2025-12-31")
    s0_gv.load_panel()
    sig = s0_gv._signals(); es = s0_gv._exec_sig(sig)
    lots = s0_gv._signal_to_lots(es, cfg_base)
    s0_gv._precomputed_lots = lots
    s0_gv._load_iv_panel(); s0_gv._compute_signal_panels()

    s0_vrp = VrpRouterStrategy(
        option_data=option_data, option_contracts=option_contracts,
        greeks_cache=gcache, products=PRODUCTS, config=cfg_base,
        start="2018-01-01", end="2025-12-31")
    s0_vrp.load_panel()
    s0_vrp._precomputed_lots = lots
    s0_vrp._load_iv_panel(); s0_vrp._compute_signal_panels()
    s0_vrp._precompute_router_targets()

    op, os_ = build_option_prices(list(s0_gv._price_panel.index), option_data)
    ap = {p: s0_gv._price_panel[p].tolist() for p in PRODUCTS}
    ap.update(op)
    ap2 = {p: p for p in PRODUCTS}
    for sym, c in option_contracts.items():
        ap2[sym] = c.product
    reg = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=reg)

    print("\n=== 拆解: 期货 vs 期权贡献 ===")
    print(f"基线 fut_v1: Sharpe=2.09 AnnRet=23.5% MaxDD=-9.1% Calmar=2.57")
    print()

    # 1. Standalone GammaVegaSeller (新 OTM 0.95x, delta=0.25)
    print("--- 1. GammaVegaSeller standalone (delta=0.25, OTM 0.95×) ---")
    cfg = GammaVegaConfig(down_days=3, vrp_min=0.50,
        short_put_strike_mult=0.95, target_delta_pct=0.25,
        hold_to_expiry=True, stop_loss_pct=0.30,
        hedge_trigger_pct=0.15, hedge_reduction_pct=0.30)
    st = OptionOnlyGV(gv_config=cfg, option_data=option_data,
        option_contracts=option_contracts, greeks_cache=gcache,
        products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
    for attr in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel",
                 "_vrp_pct", "_ret", "_down_cnt"]:
        setattr(st, attr, getattr(s0_gv, attr))
    st._precomputed_lots = lots
    t = time.time()
    st._precompute_gv_targets()
    res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap,
        products=ap2, strategy=st,
        config=CtaBacktestConfig(initial_cash=INITIAL, commission_rate=0.0001, margin_call=True),
        registry=reg, risk=risk, option_contracts=option_contracts, option_settle=os_)
    m = metrics(res.nav)
    print(f"  Sharpe={m['Sharpe']:.2f} AnnRet={m['AnnRet']:.1%} AnnVol={m['AnnVol']:.1%} "
          f"MaxDD={m['MaxDD']:.1%} Calmar={m['Calmar']:.2f} [{time.time()-t:.0f}s]")
    print(f"  期权层独立: {'+' if m['Sharpe']>0 else ''} Sharpe {m['Sharpe']:.2f}")
    standalone_gv = m

    # 2. Standalone VRP Router (买方 only)
    print("\n--- 2. VRP Router standalone (买方 only) ---")
    cfg = VrpRouterConfig(long_hold=7, short_hold=15, delta_notional_pct=0.25,
        skew_threshold=None)
    st = OptionOnlyVR(router_config=cfg, option_data=option_data,
        option_contracts=option_contracts, greeks_cache=gcache,
        products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
    for attr in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel",
                 "_vrp_pct", "_mom20", "_iv_pct"]:
        setattr(st, attr, getattr(s0_vrp, attr))
    st._precomputed_lots = lots
    st._precompute_router_targets()
    res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap,
        products=ap2, strategy=st,
        config=CtaBacktestConfig(initial_cash=INITIAL, commission_rate=0.0001, margin_call=True),
        registry=reg, risk=risk, option_contracts=option_contracts, option_settle=os_)
    m = metrics(res.nav)
    print(f"  Sharpe={m['Sharpe']:.2f} AnnRet={m['AnnRet']:.1%} MaxDD={m['MaxDD']:.1%} Calmar={m['Calmar']:.2f}")
    standalone_vrp = m

    # 3. Combined: fut_v1 + GammaVegaSeller (已测 2.41)
    print("\n--- 3. Combined: fut_v1 + GammaVegaSeller ---")
    print("  Sharpe=2.41 AnnRet=27.0% MaxDD=-9.8% Calmar=2.94 (delta=0.25, 已测)")

    # 4. Combined: fut_v1 + VRP Router
    print("\n--- 4. Combined: fut_v1 + VRP Router ---")
    print("  Sharpe=2.17 AnnRet=24.3% MaxDD=-9.2% Calmar=2.64 (已测)")

    # 5. Combined: fut_v1 + VRP + GammaVegaSeller (双策略叠加)
    print("\n--- 5. Combined: fut_v1 + VRP Router + GammaVegaSeller ---")
    vrp_cfg = VrpRouterConfig(long_hold=7, short_hold=15, delta_notional_pct=0.25,
        skew_threshold=None)
    gv_cfg = GammaVegaConfig(down_days=3, vrp_min=0.50,
        short_put_strike_mult=0.95, target_delta_pct=0.25,
        hold_to_expiry=True, stop_loss_pct=0.30,
        hedge_trigger_pct=0.15, hedge_reduction_pct=0.30)
    st = DualStrategy(vrp_cfg, gv_config=gv_cfg, option_data=option_data,
        option_contracts=option_contracts, greeks_cache=gcache,
        products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
    for attr in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel",
                 "_vrp_pct", "_ret", "_down_cnt"]:
        setattr(st, attr, getattr(s0_gv, attr))
    st._precomputed_lots = lots
    t = time.time()
    st.run()
    m = metrics(st._run_with_lots(lots).nav)
    print(f"  Sharpe={m['Sharpe']:.2f} AnnRet={m['AnnRet']:.1%} MaxDD={m['MaxDD']:.1%} "
          f"Calmar={m['Calmar']:.2f} [{time.time()-t:.0f}s]")

    # === 优化扫描: stop_loss × hedge_trigger ===
    print("\n=== 优化扫描: stop_loss × hedge_trigger ===")
    print(f"{'stop':>5} {'hedge':>6} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>8} {'Calmar':>7}")
    for sl in [0.20, 0.30, 0.50, 1.00]:  # 1.00 = 不止损
        for ht in [0.10, 0.15, 0.20]:
            cfg = GammaVegaConfig(down_days=3, vrp_min=0.50,
                short_put_strike_mult=0.95, target_delta_pct=0.25,
                hold_to_expiry=True, stop_loss_pct=sl,
                hedge_trigger_pct=ht, hedge_reduction_pct=0.30)
            st = GammaVegaSeller(gv_config=cfg, option_data=option_data,
                option_contracts=option_contracts, greeks_cache=gcache,
                products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
            for attr in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel",
                         "_vrp_pct", "_ret", "_down_cnt"]:
                setattr(st, attr, getattr(s0_gv, attr))
            st._precomputed_lots = lots
            st._precompute_gv_targets()
            res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap,
                products=ap2, strategy=st,
                config=CtaBacktestConfig(initial_cash=INITIAL, commission_rate=0.0001, margin_call=True),
                registry=reg, risk=risk, option_contracts=option_contracts, option_settle=os_)
            m = metrics(res.nav)
            print(f"{sl:>5.2f} {ht:>6.2f} {m['Sharpe']:>7.2f} {m['AnnRet']:>8.1%} "
                  f"{m['MaxDD']:>8.1%} {m['Calmar']:>7.2f}")


if __name__ == "__main__":
    main()
