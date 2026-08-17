#!/usr/bin/env python3
"""GammaVegaSeller standalone — 期权层独立回测 (不含 fut_v1).
验证真实 MDD 和 P&L 分布."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import numpy as np, pandas as pd
from QuantNodes.strategy.cta.options.gamma_vega_seller import GammaVegaSeller, GammaVegaConfig
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

PRODUCTS = ["RB","I","AG","AU","CU","M","TA","SR","RU","PP"]

class OptionOnlyGV(GammaVegaSeller):
    """纯期权层 (不含期货仓位)."""
    def target_positions(self, date, data):
        if not hasattr(self, "_precomputed_gv") or date not in self._precomputed_gv:
            return {}
        return self._precomputed_gv[date]

def full_metrics(nav):
    r = nav.pct_change().dropna()
    if len(r)==0: return {}
    total = nav.iloc[-1]/nav.iloc[0]-1
    years = len(r)/252
    ann = (1+total)**(1/max(years,0.01))-1
    vol = r.std()*np.sqrt(252)
    sharpe = (ann-0.02)/vol if vol>0 else 0
    dd = ((nav-nav.cummax())/nav.cummax()).min()
    calmar = ann/abs(dd) if abs(dd)>0 else 0
    # 额外指标
    win_days = (r>0).mean()
    best_day = r.max()
    worst_day = r.min()
    return {"AnnRet":ann, "AnnVol":vol, "Sharpe":sharpe, "MaxDD":dd,
            "Calmar":calmar, "WinRate":win_days, "BestDay":best_day, "WorstDay":worst_day,
            "TotalRet":total, "Years":years}

def main():
    cfg = ComboConfig(initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0,
        vol_lookback=20, momentum_filter_window=60, momentum_filter_coef=0.0,
        filter_enabled=True, module_weighting="inv_vol", continuous=True,
        cont_z_window=252, cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr")
    od, oc = load_option_data(PRODUCTS, "2023-01-01", "2025-12-31")
    gc = GreeksCache(parquet_path="data/cache/option_greeks.parquet")
    s0 = GammaVegaSeller(option_data=od, option_contracts=oc, greeks_cache=gc,
        products=PRODUCTS, config=cfg, start="2018-01-01", end="2025-12-31")
    s0.load_panel()
    sig = s0._signals(); es = s0._exec_sig(sig)
    lots = s0._signal_to_lots(es, cfg)
    s0._precomputed_lots = lots; s0._load_iv_panel(); s0._compute_signal_panels()
    op, os_ = build_option_prices(list(s0._price_panel.index), od)
    ap = {p:s0._price_panel[p].tolist() for p in PRODUCTS}; ap.update(op)
    ap2 = {p:p for p in PRODUCTS}
    for sym, c in oc.items(): ap2[sym] = c.product
    reg = CtaContractRegistry()
    risk = PositionLimitRisk(products={p:p for p in PRODUCTS}, registry=reg)

    print("=" * 80)
    print("GammaVegaSeller Standalone (纯期权层, 不含期货)")
    print("=" * 80)
    print(f"{'d_pct':>6} {'Sharpe':>7} {'AnnRet':>8} {'AnnVol':>8} {'MaxDD':>8} "
          f"{'Calmar':>7} {'WinR':>6} {'BestD':>7} {'WorstD':>7} {'entry':>5} {'hedge':>5}")
    for dp in [0.15, 0.20, 0.25, 0.30, 0.50]:
        t=time.time()
        cfg2 = GammaVegaConfig(down_days=3, vrp_min=0.50,
            short_put_strike_mult=0.95, target_delta_pct=dp,
            hold_to_expiry=True, stop_loss_pct=0.20,
            hedge_trigger_pct=0.10, hedge_reduction_pct=0.30)
        st = OptionOnlyGV(gv_config=cfg2, option_data=od, option_contracts=oc,
            greeks_cache=gc, products=PRODUCTS, config=cfg, start="2018-01-01", end="2025-12-31")
        for a in ["_price_panel","_ohlc_map","_vols","_ois","_iv_panel","_vrp_pct","_ret","_down_cnt"]:
            setattr(st, a, getattr(s0, a))
        st._precomputed_lots=lots
        st._precompute_gv_targets()
        res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap, products=ap2,
            strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000,commission_rate=0.0001,margin_call=True),
            registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
        m = full_metrics(res.nav)
        s = st.stats
        print(f"{dp:>6.2f} {m['Sharpe']:>7.2f} {m['AnnRet']:>8.1%} {m['AnnVol']:>8.1%} "
              f"{m['MaxDD']:>8.1%} {m['Calmar']:>7.2f} {m['WinRate']:>5.0%} "
              f"{m['BestDay']*100:>6.1f}% {m['WorstDay']*100:>6.1f}% {s['entries']:>5} {s['hedges']:>5}  [{time.time()-t:.0f}s]", flush=True)

    # 组合对比
    print("\n--- 组合: fut_v1 + GV (delta=0.30) ---")
    gv_st = GammaVegaSeller(gv_config=GammaVegaConfig(down_days=3, vrp_min=0.50,
        short_put_strike_mult=0.95, target_delta_pct=0.30,
        hold_to_expiry=True, stop_loss_pct=0.20,
        hedge_trigger_pct=0.10, hedge_reduction_pct=0.30),
        option_data=od, option_contracts=oc, greeks_cache=gc,
        products=PRODUCTS, config=cfg, start="2018-01-01", end="2025-12-31")
    for a in ["_price_panel","_ohlc_map","_vols","_ois","_iv_panel","_vrp_pct","_ret","_down_cnt"]:
        setattr(gv_st, a, getattr(s0, a))
    gv_st._precomputed_lots = lots
    res2 = run_cta_backtest(dates=list(gv_st._price_panel.index), prices=ap, products=ap2,
        strategy=gv_st, config=CtaBacktestConfig(initial_cash=5_000_000,commission_rate=0.0001,margin_call=True),
        registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
    m2 = full_metrics(res2.nav)
    print(f"{'d=0.30':>6} {m2['Sharpe']:>7.2f} {m2['AnnRet']:>8.1%} {m2['AnnVol']:>8.1%} "
          f"{m2['MaxDD']:>8.1%} {m2['Calmar']:>7.2f} {m2['WinRate']:>5.0%} "
          f"{m2['BestDay']*100:>6.1f}% {m2['WorstDay']*100:>6.1f}%")

import time
if __name__ == "__main__":
    main()
