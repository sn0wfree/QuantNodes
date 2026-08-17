#!/usr/bin/env python3
"""GammaVegaSeller 参数扫描: delta cap × 品种过滤 × 持仓周期 × 入场深度."""
import warnings; warnings.filterwarnings("ignore")

import sys
import time
from dataclasses import replace
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd
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
GOOD_ALL = ["I", "AG", "CU", "SR", "TA", "RU", "M"]
GOOD_NO_M = ["I", "AG", "CU", "SR", "TA", "RU"]


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
    return {"Sharpe": sharpe, "AnnRet": ann, "MaxDD": dd, "Calmar": calmar}


def main():
    cfg_base = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0, vol_lookback=20,
        momentum_filter_window=60, momentum_filter_coef=0.0, filter_enabled=True,
        module_weighting="inv_vol", continuous=True, cont_z_window=252,
        cont_z_clip=3.0, margin_call=True, execution_lag=1, aggregation="tvpr",
    )
    od, oc = load_option_data(PRODUCTS, "2023-01-01", "2025-12-31")
    gc = GreeksCache(parquet_path="data/cache/option_greeks.parquet")

    # 共享面板
    s0 = GammaVegaSeller(option_data=od, option_contracts=oc, greeks_cache=gc,
        products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
    s0.load_panel()
    sig = s0._signals(); es = s0._exec_sig(sig)
    lots = s0._signal_to_lots(es, cfg_base)
    s0._precomputed_lots = lots
    s0._load_iv_panel(); s0._compute_signal_panels()

    op, os_ = build_option_prices(list(s0._price_panel.index), od)
    ap = {p: s0._price_panel[p].tolist() for p in PRODUCTS}; ap.update(op)
    ap2 = {p: p for p in PRODUCTS}
    for sym, c in oc.items():
        ap2[sym] = c.product
    reg = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=reg)

    # 基线
    print("baseline: Sharpe=2.09 AnnRet=23.5% MaxDD=-9.1% Calmar=2.57")
    print()

    # ── 1. Delta cap 扫描 ──
    print("=== 1. Delta cap 扫描 ===")
    print(f"{'d_pct':>6} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>8} {'Calmar':>7} {'entry':>5} {'hedge':>5} {'stops':>5}")
    for dp in [0.10, 0.15, 0.20, 0.25, 0.30, 0.50]:
        cfg2 = GammaVegaConfig(down_days=3, vrp_min=0.50,
            short_put_strike_mult=0.95, target_delta_pct=dp,
            hold_to_expiry=True, stop_loss_pct=0.20,
            hedge_trigger_pct=0.10, hedge_reduction_pct=0.30)
        st = GammaVegaSeller(gv_config=cfg2, option_data=od, option_contracts=oc,
            greeks_cache=gc, products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
        for a in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel", "_vrp_pct", "_ret", "_down_cnt"]:
            setattr(st, a, getattr(s0, a))
        st._precomputed_lots = lots
        t0 = time.time()
        st._precompute_gv_targets()
        res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap, products=ap2,
            strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001, margin_call=True),
            registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
        m = metrics(res.nav)
        s = st.stats
        print(f"{dp:>6.2f} {m['Sharpe']:>7.2f} {m['AnnRet']:>8.1%} {m['MaxDD']:>8.1%} "
              f"{m['Calmar']:>7.2f} {s['entries']:>5} {s['hedges']:>5} {s['stops']:>5}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    # ── 2. 品种过滤 ──
    print("\n=== 2. 品种过滤 (delta=0.25) ===")
    for wl_label, wl in [("全品种M", GOOD_ALL), ("去M", GOOD_NO_M)]:
        cfg2 = GammaVegaConfig(down_days=3, vrp_min=0.50,
            short_put_strike_mult=0.95, target_delta_pct=0.25,
            good_products=wl,
            hold_to_expiry=True, stop_loss_pct=0.20,
            hedge_trigger_pct=0.10, hedge_reduction_pct=0.30)
        st = GammaVegaSeller(gv_config=cfg2, option_data=od, option_contracts=oc,
            greeks_cache=gc, products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
        for a in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel", "_vrp_pct", "_ret", "_down_cnt"]:
            setattr(st, a, getattr(s0, a))
        st._precomputed_lots = lots
        t0 = time.time()
        st._precompute_gv_targets()
        res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap, products=ap2,
            strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001, margin_call=True),
            registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
        m = metrics(res.nav)
        s = st.stats
        print(f"  {wl_label:<12} Sharpe={m['Sharpe']:.2f} AnnRet={m['AnnRet']:.1%} MaxDD={m['MaxDD']:.1%} "
              f"Calmar={m['Calmar']:.2f} [{time.time()-t0:.0f}s]", flush=True)

    # ── 3. 持仓周期 (hold_to_expiry vs 截断) ──
    print("\n=== 3. 持仓周期 (delta=0.25, stop=0.20) ===")
    for hold, label in [(30, "30日"), (60, "60日"), (999, "无限制")]:
        cfg2 = GammaVegaConfig(down_days=3, vrp_min=0.50,
            short_put_strike_mult=0.95, target_delta_pct=0.25,
            hold_to_expiry=(hold > 100), hold_days=hold if hold <= 100 else 7,
            stop_loss_pct=0.20, hedge_trigger_pct=0.10, hedge_reduction_pct=0.30)
        st = GammaVegaSeller(gv_config=cfg2, option_data=od, option_contracts=oc,
            greeks_cache=gc, products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
        for a in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel", "_vrp_pct", "_ret", "_down_cnt"]:
            setattr(st, a, getattr(s0, a))
        st._precomputed_lots = lots
        t0 = time.time()
        st._precompute_gv_targets()
        res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap, products=ap2,
            strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001, margin_call=True),
            registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
        m = metrics(res.nav)
        s = st.stats
        print(f"  {label:<12} Sharpe={m['Sharpe']:.2f} AnnRet={m['AnnRet']:.1%} MaxDD={m['MaxDD']:.1%} "
              f"Calmar={m['Calmar']:.2f} [{time.time()-t0:.0f}s]", flush=True)

    # ── 4. 入场深度 (down_days 加强) ──
    print("\n=== 4. 入场深度 (delta=0.25, stop=0.20) ===")
    for nd in [3, 4, 5]:
        cfg2 = GammaVegaConfig(down_days=nd, vrp_min=0.50,
            short_put_strike_mult=0.95, target_delta_pct=0.25,
            hold_to_expiry=True, stop_loss_pct=0.20,
            hedge_trigger_pct=0.10, hedge_reduction_pct=0.30)
        st = GammaVegaSeller(gv_config=cfg2, option_data=od, option_contracts=oc,
            greeks_cache=gc, products=PRODUCTS, config=cfg_base, start="2018-01-01", end="2025-12-31")
        for a in ["_price_panel", "_ohlc_map", "_vols", "_ois", "_iv_panel", "_vrp_pct", "_ret", "_down_cnt"]:
            setattr(st, a, getattr(s0, a))
        st._precomputed_lots = lots
        t0 = time.time()
        st._precompute_gv_targets()
        res = run_cta_backtest(dates=list(st._price_panel.index), prices=ap, products=ap2,
            strategy=st, config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001, margin_call=True),
            registry=reg, risk=risk, option_contracts=oc, option_settle=os_)
        m = metrics(res.nav)
        s = st.stats
        print(f"  down={nd}  Sharpe={m['Sharpe']:.2f} AnnRet={m['AnnRet']:.1%} MaxDD={m['MaxDD']:.1%} "
              f"Calmar={m['Calmar']:.2f} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
