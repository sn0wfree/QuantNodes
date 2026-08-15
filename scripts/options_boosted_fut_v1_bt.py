#!/usr/bin/env python3
"""Options-Boosted fut_v1 回测 — 裸买 Call/Put 增强.

对比: fut_v1 vs fut_v1 + 期权因子 vs fut_v1 + 裸买期权
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import pandas as pd
import numpy as np
from QuantNodes.strategy.cta.options.options_boosted_fut_v1 import (
    OptionsBoostedFutV1,
    OptionsBoostConfig,
)
from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy,
    ComboConfig,
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
    """加载期权数据 + option_contracts."""
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

                for sym in opt_df["symbol"].unique():
                    row = opt_df[opt_df["symbol"] == sym].iloc[0]
                    opt_spec = OptionContractSpec(
                        exchange="", product=prod,
                        underlying=str(row.get("underlying", "")),
                        option_type=str(row.get("option_type", "")).upper(),
                        strike=float(row["strike_price"]),
                        expiry_date="",
                        contract_multiplier=registry.get_or_default(prod).contract_multiplier,
                    )
                    contract = CtaContract(
                        product=prod, symbol=sym, expiry_year=0, expiry_month=0,
                        spec=registry.get_or_default(prod), option_spec=opt_spec,
                    )
                    option_contracts[sym] = contract
        except Exception:
            pass

    return option_data, option_contracts


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    start = "2023-01-01"
    end = "2025-12-31"

    print("Options-Boosted fut_v1 — Multi-Product Portfolio")
    print("=" * 70)

    # 加载期权数据
    print("\nLoading option data...")
    option_data, option_contracts = load_option_data(products, start, end)
    print(f"  Option contracts: {len(option_contracts)}")
    print(f"  Products with options: {list(option_data.keys())}")

    # ═══ 策略 1: 基线 fut_v1 ═══
    print("\n--- Baseline fut_v1 (72 factors) ---")
    t0 = time.time()
    cfg_base = ComboConfig(
        initial_cash=10_000_000, target_vol=0.15, max_leverage=3.0,
        vol_lookback=20, momentum_filter_window=60, momentum_filter_coef=0.5,
        filter_enabled=True, module_weighting="inv_vol",
    )
    strat_base = FactorModuleComboStrategy(products=products, config=cfg_base)
    result_base = strat_base.run()
    m_base = compute_full_metrics(result_base.nav)
    t1 = time.time() - t0
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"MaxDD={m_base['max_drawdown']:>6.1%}  Calmar={m_base['calmar']:>6.2f}  ({t1:.0f}s)")

    # ═══ 策略 2: fut_v1 + 裸买期权 ═══
    print("\n--- fut_v1 + Naked Long Options (VRP boost) ---")
    t0 = time.time()
    boost_cfg = OptionsBoostConfig(
        boost_notional_pct=0.3,
        boost_max_lots=5,
        iv_high_threshold=0.7,
        min_dte=30, max_dte=60,
    )
    strat_boost = OptionsBoostedFutV1(
        boost_config=boost_cfg,
        option_contracts=option_contracts,
        option_data=option_data,
        products=products,
        config=cfg_base,
    )

    # 运行回测
    registry = CtaContractRegistry()
    from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    # 先跑信号
    strat_boost.load_panel()
    signals = strat_boost._signals()
    lots = strat_boost._signal_to_lots(signals, cfg_base)

    # 用自定义策略跑回测
    class BoostedStrategy:
        def __init__(self, lots_df, boost_strat):
            self._lots = lots_df
            self._boost = boost_strat
            self._last = {}

        def target_positions(self, date, data):
            if date not in self._lots.index:
                return {}
            row = self._lots.loc[date]
            targets = {p: float(row[p]) for p in self._lots.columns if abs(row[p]) > 1e-9}

            # 方向变化才调仓
            changed = False
            for p in set(targets) | set(self._last):
                def _sgn(x):
                    return 1 if x > 0 else (-1 if x < 0 else 0)
                if _sgn(targets.get(p, 0)) != _sgn(self._last.get(p, 0)):
                    changed = True
                    break
            if not changed:
                return {}
            self._last = targets.copy()

            # 加仓: VRP 确认时买 Call/Put
            for product in list(targets.keys()):
                futures_qty = targets[product]
                if abs(futures_qty) < 1:
                    continue

                vrp_signal = self._boost._get_vrp_signal(product, date)
                iv_rank = self._boost._get_iv_rank(product, date)

                if abs(vrp_signal) < 0.01:
                    continue
                if iv_rank > self._boost._boost_cfg.iv_high_threshold:
                    continue

                futures_dir = np.sign(futures_qty)
                vrp_dir = np.sign(vrp_signal)
                if futures_dir != vrp_dir:
                    continue

                # 检查是否已持有期权
                if product in self._boost._option_holdings:
                    continue

                # 选择裸买 Call/Put
                spot = self._boost._price_panel.loc[date, product] if date in self._boost._price_panel.index else None
                if spot is None or np.isnan(spot):
                    continue

                result = self._boost._select_option_for_boost(product, date, spot, futures_dir)
                if result is not None:
                    opt_symbol, opt_qty = result
                    targets[opt_symbol] = opt_qty
                    self._boost._option_holdings[product] = {
                        "symbol": opt_symbol, "qty": opt_qty,
                        "entry_date": str(date.date()), "direction": futures_dir,
                    }

            return targets

    strategy = BoostedStrategy(lots, strat_boost)
    t0 = time.time()
    result_boost = run_cta_backtest(
        dates=list(strat_boost._price_panel.index),
        prices={p: strat_boost._price_panel[p].tolist() for p in products},
        products={p: p for p in products},
        strategy=strategy,
        config=CtaBacktestConfig(initial_cash=10_000_000, commission_rate=0.0001),
        registry=registry,
        risk=risk,
        option_contracts=option_contracts,
    )
    m_boost = compute_full_metrics(result_boost.nav)
    t2 = time.time() - t0
    print(f"  AnnRet={m_boost['ann_return']:>6.1%}  Sharpe={m_boost['sharpe']:>6.2f}  "
          f"MaxDD={m_boost['max_drawdown']:>6.1%}  Calmar={m_boost['calmar']:>6.2f}  ({t2:.0f}s)")

    # 对比
    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  {'':>15} {'Baseline':>12} {'Boosted':>12} {'Improvement':>12}")
    print("  " + "-" * 50)
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]:
        v1 = m_base.get(key, 0)
        v2 = m_boost.get(key, 0)
        diff = v2 - v1
        if key in ["ann_return", "ann_vol", "max_drawdown", "win_rate"]:
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()
