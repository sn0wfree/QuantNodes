#!/usr/bin/env python3
"""GammaVegaSeller 期权层独立回测 (不含 fut_v1 期货)."""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np

from QuantNodes.strategy.cta.options.gamma_vega_seller import (
    GammaVegaSeller, GammaVegaConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
INITIAL = 5_000_000


class OptionOnlyGV(GammaVegaSeller):
    """仅交易期权 targets, 不叠加期货."""

    def target_positions(self, date, data) -> dict[str, float]:
        if not hasattr(self, "_precomputed_gv") or date not in self._precomputed_gv:
            return {}
        return self._precomputed_gv[date]


def full_metrics(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann / abs(dd) if abs(dd) > 0 else 0
    return {"AnnRet": ann, "AnnVol": vol, "Sharpe": sharpe,
            "MaxDD": dd, "Calmar": calmar, "TotalRet": total}


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

    s0 = GammaVegaSeller(gv_config=GammaVegaConfig(), option_data=option_data,
        option_contracts=option_contracts, greeks_cache=gcache,
        products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
    s0.load_panel()
    sig = s0._signals(); es = s0._exec_sig(sig)
    lots = s0._signal_to_lots(es, cfg_base)
    s0._precomputed_lots = lots
    s0._load_iv_panel(); s0._compute_signal_panels()

    op, os_ = build_option_prices(list(s0._price_panel.index), option_data)
    all_prices = {p: s0._price_panel[p].tolist() for p in PRODUCTS}
    all_prices.update(op)
    all_products = {p: p for p in PRODUCTS}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product
    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=registry)

    # 测试不同 delta
    print(f"\n{'delta/批':>8} {'Sharpe':>7} {'AnnRet':>8} {'AnnVol':>7} {'MaxDD':>8} {'Calmar':>7} {'entry':>5} {'adds':>5} {'exits':>6}")
    for dp in [0.10, 0.30, 0.50]:
        cfg = GammaVegaConfig(down_days=4, vrp_min=0.50, good_products=["I","AG","CU","SR","TA","RU","M"],
                              max_batches=3, batch_add_if_down=True, batch_size=1/3,
                              atm_mult=1.00, put_otm_mult=0.88, min_dte=15, max_dte=40,
                              hold_to_expiry=True, hold_days=7, stop_loss_pct=0.30,
                              batch_delta_pct=dp)
        st = OptionOnlyGV(gv_config=cfg, option_data=option_data,
            option_contracts=option_contracts, greeks_cache=gcache,
            products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
        st._price_panel = s0._price_panel; st._ohlc_map = s0._ohlc_map
        st._vols = s0._vols; st._ois = s0._ois
        st._iv_panel = s0._iv_panel; st._vrp_pct = s0._vrp_pct
        st._ret = s0._ret; st._down_cnt = s0._down_cnt
        st._precomputed_lots = lots
        t0 = time.time()
        st._precompute_gv_targets()
        res = run_cta_backtest(
            dates=list(st._price_panel.index), prices=all_prices, products=all_products,
            strategy=st, config=CtaBacktestConfig(initial_cash=INITIAL,
                commission_rate=0.0001, margin_call=True),
            registry=registry, risk=risk, option_contracts=option_contracts,
            option_settle=os_)
        m = full_metrics(res.nav)
        s = st.stats
        print(f"{dp:>8.2f} {m['Sharpe']:>7.2f} {m['AnnRet']:>8.1%} {m['AnnVol']:>7.1%} "
              f"{m['MaxDD']:>8.1%} {m['Calmar']:>7.2f} {s['entries']:>5} {s['adds']:>5} "
              f"{s['batch_exits']:>6}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
