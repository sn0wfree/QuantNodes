#!/usr/bin/env python3
"""Multi-Factor Options 完整回测 + 业绩指标.

策略核心:
    强趋势 (|CTA| > 0.5) → 买 Call/Put (gamma 凸性, 方向性)
    弱趋势 (|CTA| < 0.3) → 卖期权 (theta 收割)
    中性  → Calendar Spread (期限结构)

用法:
    cd QuantNodes && python scripts/mf_options_bt.py
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.options.iv_panel import load_iv_panel
from QuantNodes.strategy.cta.options.vol_signals import VolSignalGenerator
from QuantNodes.strategy.cta.options.multi_factor_options import (
    MultiFactorOptionsStrategy, MultiFactorConfig, Signal
)
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig, CtaStrategy
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry, CtaContract, OptionContractSpec
from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics


def compute_full_metrics(nav: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """计算完整业绩指标."""
    if len(nav) < 2:
        return {}

    # 日收益率
    returns = nav.pct_change().dropna()
    if len(returns) == 0:
        return {}

    # 年化收益
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # 年化波动率
    ann_vol = returns.std() * np.sqrt(252)

    # Sharpe
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # 最大回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()

    # Calmar
    calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0

    # 胜率
    win_rate = (returns > 0).sum() / len(returns)

    # 盈亏比
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    # 最长回撤天数
    underwater = drawdown < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_lengths = underwater.groupby(groups).sum()
        max_dd_days = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
    else:
        max_dd_days = 0

    return {
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "payoff_ratio": payoff_ratio,
        "max_dd_days": max_dd_days,
        "n_years": n_years,
    }


def run_backtest(product="RB", start="2023-01-01", end="2025-12-31"):
    """运行单品种回测."""
    t0 = time.time()
    loader = CtaDataLoader()
    um = UnderlyingMap(loader)
    oe = OptionExpiry(loader)
    registry = CtaContractRegistry()

    # 价格数据
    px = loader.load_continuous_main(product)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.set_index("trade_date")
    price_series = px["close"]

    # IV + HV
    iv_panel = load_iv_panel(start=start, end=end)
    log_ret = np.log(price_series / price_series.shift(1))
    hv_series = log_ret.rolling(20, min_periods=10).std() * np.sqrt(252)
    hv_panel = pd.DataFrame({product: hv_series}).dropna()
    vol_gen = VolSignalGenerator(iv_panel, hv_panel, iv_pct_window=20)

    # 期权数据
    opt_df = loader.load_v_all_options(products=[product], start=start, end=end)
    opt_df["trade_date"] = pd.to_datetime(opt_df["trade_date"])

    # option_contracts
    option_contracts = {}
    for sym in opt_df["symbol"].unique():
        row = opt_df[opt_df["symbol"] == sym].iloc[0]
        opt_spec = OptionContractSpec(
            exchange="", product=product,
            underlying=str(row.get("underlying", "")),
            option_type=str(row.get("option_type", "")).upper(),
            strike=float(row["strike_price"]), expiry_date="",
            contract_multiplier=registry.get_or_default(product).contract_multiplier,
        )
        contract = CtaContract(product=product, symbol=sym, expiry_year=0, expiry_month=0,
                               spec=registry.get_or_default(product), option_spec=opt_spec)
        option_contracts[sym] = contract

    # prices_map
    dates = price_series.index[(price_series.index >= start) & (price_series.index <= end)]
    prices_map = {product: price_series.reindex(dates).tolist()}
    products_map = {product: product}
    for sym in opt_df["symbol"].unique():
        sym_df = opt_df[opt_df["symbol"] == sym].set_index("trade_date")
        prices_map[sym] = sym_df["close"].reindex(dates).tolist()
        products_map[sym] = product

    # Multi-Factor Strategy
    mf_cfg = MultiFactorConfig(
        products=[product], initial_cash=1_000_000, max_margin_pct=0.50,
        roll_days=7, min_dte=20, max_dte=60, use_cta_factors=True,
    )
    mf = MultiFactorOptionsStrategy(mf_cfg)
    mf.load_data(start, end)

    # CtaStrategy wrapper
    state = {"holdings": {}, "current_signal": None}

    class S(CtaStrategy):
        def target_positions(self, date, data):
            spot = price_series.loc[date] if date in price_series.index else None
            if spot is None or np.isnan(spot):
                return {}

            # 换月检查
            if product in state["holdings"]:
                h = state["holdings"][product]
                days_left = (h["expiry"] - date).days
                if days_left <= mf_cfg.roll_days:
                    state["holdings"].pop(product, None)
                    state["current_signal"] = None
                    # 返回平仓目标
                    for sym in list(h["symbols"].keys()):
                        return {sym: 0.0}

            # 信号
            signal = mf._get_signal(product, date)
            new_type = mf._signal_type(signal)
            old_type = mf._signal_type(state["current_signal"]) if state["current_signal"] else None

            # 信号变化 → 平旧仓 + 开新仓
            if product in state["holdings"] and new_type == old_type:
                return {}  # 持有

            if product in state["holdings"]:
                h = state["holdings"][product]
                state["holdings"].pop(product)
                for sym in list(h["symbols"].keys()):
                    return {sym: 0.0}

            if signal != Signal.NO_TRADE:
                targets = mf._select_strategy(signal, product, date, spot)
                if targets:
                    margin = mf._estimate_margin(targets, spot)
                    if margin <= mf_cfg.initial_cash * mf_cfg.max_margin_pct:
                        state["holdings"][product] = {
                            "symbols": targets,
                            "expiry": date + pd.Timedelta(days=mf_cfg.max_dte),
                            "signal": signal,
                        }
                        state["current_signal"] = signal
                        return targets
            return {}

    result = run_cta_backtest(
        dates=list(dates), prices=prices_map, products=products_map,
        strategy=S(), config=CtaBacktestConfig(initial_cash=1_000_000, commission_rate=0.0001),
        registry=registry, option_contracts=option_contracts,
    )

    elapsed = time.time() - t0

    # 计算完整指标
    metrics = compute_full_metrics(result.nav)

    return {
        "product": product,
        "nav": result.nav,
        "metrics": metrics,
        "trades": result.trades,
        "n_trades": len(result.trades),
        "elapsed": elapsed,
    }


def print_results(results: list[dict]):
    """打印业绩报告."""
    print("\n" + "=" * 80)
    print("Multi-Factor Options Strategy — Performance Report")
    print("=" * 80)

    # 汇总表
    print(f"\n{'Product':>8} {'Return':>8} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>8} {'Trades':>8}")
    print("-" * 80)

    for r in results:
        m = r["metrics"]
        if not m:
            print(f"{r['product']:>8} {'N/A':>8}")
            continue
        print(f"{r['product']:>8} {m['total_return']:>7.1%} {m['ann_return']:>7.1%} "
              f"{m['ann_vol']:>7.1%} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
              f"{m['max_drawdown']:>7.1%} {m['calmar']:>8.2f} {m['win_rate']:>7.1%} "
              f"{r['n_trades']:>8d}")

    # 策略分布
    print("\n\nStrategy Distribution:")
    print("-" * 40)
    for r in results:
        trades = [t for t in r["trades"] if hasattr(t, "side") and t.side == "open"
                  or isinstance(t, dict) and t.get("side") == "open"]
        strategies = {}
        for t in trades:
            s = t.strategy if hasattr(t, "strategy") else t.get("strategy", "UNKNOWN") if isinstance(t, dict) else "UNKNOWN"
            strategies[s] = strategies.get(s, 0) + 1
        print(f"\n  {r['product']}:")
        for s, n in sorted(strategies.items(), key=lambda x: -x[1]):
            print(f"    {s:25s} {n:4d}")


if __name__ == "__main__":
    results = []
    for prod in ["RB", "I", "AG"]:
        try:
            r = run_backtest(prod)
            results.append(r)
        except Exception as e:
            print(f"{prod} ERROR: {e}")

    print_results(results)
