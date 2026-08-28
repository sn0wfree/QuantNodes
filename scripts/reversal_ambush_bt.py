#!/usr/bin/env python3
"""Reversal Ambush (超跌埋伏 + 动态转换) 回测 — 叠加 fut_v1.

对比: fut_v1 基线 vs fut_v1 + 超跌埋伏动态转换期权.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import pandas as pd
import numpy as np

from QuantNodes.strategy.cta.options.reversal_ambush import (
    ReversalAmbushStrategy, ReversalAmbushConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig,
)
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry, CtaContract, OptionContractSpec


def compute_full_metrics(nav):
    returns = nav.pct_change().dropna()
    if len(returns) == 0:
        return {}
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0
    downside = returns[returns < 0]
    down_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (ann_return - 0.02) / down_vol if down_vol > 0 else 0
    cummax = nav.cummax()
    max_dd = ((nav - cummax) / cummax).min()
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0
    win_rate = (returns > 0).sum() / len(returns)
    return {
        "ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "max_drawdown": max_dd, "calmar": calmar, "win_rate": win_rate,
    }


def load_option_data(products, start, end):
    loader = CtaDataLoader()
    registry = CtaContractRegistry()
    option_data = {}
    option_contracts = {}
    for prod in products:
        try:
            opt_df = loader.load_v_all_options(products=[prod], start=start, end=end)
            if opt_df is not None and not opt_df.empty:
                opt_df["trade_date"] = pd.to_datetime(opt_df["trade_date"])
                option_data[prod] = opt_df
                for sym, row in opt_df.groupby("symbol").first().iterrows():
                    opt_spec = OptionContractSpec(
                        exchange="", product=prod,
                        underlying=str(row.get("underlying", "")),
                        option_type=str(row.get("option_type", "")).upper(),
                        strike=float(row["strike_price"]),
                        expiry_date="",
                        contract_multiplier=registry.get_or_default(prod).contract_multiplier,
                    )
                    option_contracts[sym] = CtaContract(
                        product=prod, symbol=sym, expiry_year=0, expiry_month=0,
                        spec=registry.get_or_default(prod), option_spec=opt_spec,
                    )
        except Exception:
            pass
    return option_data, option_contracts


def build_option_prices(dates, option_data):
    """构造期权逐日价格序列 {symbol: [price]×len(dates)} + 到期结算价."""
    idx = pd.DatetimeIndex(dates)
    prices = {}
    settle = {}
    for prod, odf in option_data.items():
        for sym, grp in odf.groupby("symbol"):
            ser = grp.set_index("trade_date")["close"].sort_index()
            ser = ser[~ser.index.duplicated(keep="last")]
            prices[sym] = ser.reindex(idx, method="ffill").fillna(0.0).tolist()
            settle[sym] = {d.strftime("%Y-%m-%d"): v for d, v in ser.items()}
    return prices, settle


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    start = "2018-01-01"
    end = "2025-12-31"
    opt_start = "2023-01-01"
    opt_end = "2025-12-31"

    print("Reversal Ambush (超跌埋伏 + 动态转换) — fut_v1 overlay")
    print("=" * 70)

    option_data, option_contracts = load_option_data(products, opt_start, opt_end)
    print(f"  Option contracts: {len(option_contracts)}")

    cfg_base = ComboConfig(
        initial_cash=5_000_000,
        target_vol=0.30,
        max_leverage=3.0,
        vol_lookback=20,
        momentum_filter_window=60,
        momentum_filter_coef=0.0,
        filter_enabled=True,
        module_weighting="inv_vol",
        continuous=True,
        cont_z_window=252,
        cont_z_clip=3.0,
        margin_call=True,
        execution_lag=1,
        aggregation="tvpr",
    )

    # 基线
    print("\n--- Baseline fut_v1 ---")
    t0 = time.time()
    strat_base = FactorModuleComboStrategy(
        products=products, config=cfg_base, start=start, end=end,
    )
    result_base = strat_base.run()
    m_base = compute_full_metrics(result_base.nav)
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"MaxDD={m_base['max_drawdown']:>6.1%}  Calmar={m_base['calmar']:>6.2f}  ({time.time()-t0:.0f}s)")

    # 增强策略
    print("\n--- fut_v1 + Reversal Ambush ---")
    t0 = time.time()
    ambush_cfg = ReversalAmbushConfig(
        loc_window=20, loc_threshold=0.15, iv_rank_max=0.70,
        sell_strike_mult=1.12, min_dte=15, max_dte=60,
        convert_confirm_up=0.002, convert_max_up=0.03,
        max_hold_days=20, roll_days=7, take_profit=0.30,
        max_premium_pct=0.02,
    )
    strat = ReversalAmbushStrategy(
        ambush_config=ambush_cfg,
        option_data=option_data,
        option_contracts=option_contracts,
        products=products,
        config=cfg_base,
        start=start,
        end=end,
    )
    strat.load_panel()
    signals = strat._signals()
    exec_sig = strat._exec_sig(signals)
    lots = strat._signal_to_lots(exec_sig, cfg_base)
    strat._precomputed_lots = lots
    print("  Precomputing ambush targets...")
    n_opt = strat._precompute_ambush_targets()
    n_days = sum(1 for v in n_opt.values() if v)
    n_targets = sum(len(v) for v in n_opt.values())
    print(f"  Precomputed: {len(n_opt)} days, {n_days} days with targets, {n_targets} option targets")

    from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    # 期权价格序列 + 到期结算价
    opt_prices, opt_settle = build_option_prices(
        list(strat._price_panel.index), option_data
    )
    all_prices = {p: strat._price_panel[p].tolist() for p in products}
    all_prices.update(opt_prices)
    # products map 需覆盖期权 symbol → 品种 (引擎逐 symbol 查 product)
    all_products = {p: p for p in products}
    sym_to_prod = {}
    for sym, c in option_contracts.items():
        sym_to_prod[sym] = c.product
    all_products.update(sym_to_prod)

    result = run_cta_backtest(
        dates=list(strat._price_panel.index),
        prices=all_prices,
        products=all_products,
        strategy=strat,
        config=CtaBacktestConfig(initial_cash=5_000_000, commission_rate=0.0001, margin_call=True),
        registry=registry,
        risk=risk,
        option_contracts=option_contracts,
        option_settle=opt_settle,
    )
    m = compute_full_metrics(result.nav)
    print(f"  AnnRet={m['ann_return']:>6.1%}  Sharpe={m['sharpe']:>6.2f}  "
          f"MaxDD={m['max_drawdown']:>6.1%}  Calmar={m['calmar']:>6.2f}  "
          f"Trades={len(result.trades):>4d}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  {'':>15} {'Baseline':>12} {'Ambush':>12} {'Improvement':>12}")
    print("  " + "-" * 50)
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]:
        v1 = m_base.get(key, 0)
        v2 = m.get(key, 0)
        diff = v2 - v1
        if key in ["ann_return", "ann_vol", "max_drawdown", "win_rate"]:
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()
