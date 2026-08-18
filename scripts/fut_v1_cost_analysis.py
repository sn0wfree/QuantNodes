#!/usr/bin/env python3
"""fut_v1 策略交易成本分析 — 交易统计、换手率、手续费、滑点、成本后净收益."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time
import numpy as np
import pandas as pd

from scripts.common import (
    compute_metrics, run_strategy, print_metrics, parse_args, get_products,
    DEFAULT_CONFIG,
)


def analyze_trades(trades):
    """分析交易记录."""
    if not trades:
        return {}
    df = pd.DataFrame([{
        "date": t.date, "symbol": t.symbol, "product": t.product,
        "side": t.side, "qty": t.qty, "price": t.price,
        "fee": t.fee, "notional": t.notional,
    } for t in trades])
    df["date"] = pd.to_datetime(df["date"])

    total_trades = len(df)
    total_notional = df["notional"].sum()
    total_fee = df["fee"].sum()

    opens = df[df["side"] == "open"]
    closes = df[df["side"] == "close"]

    by_product = df.groupby("product").agg(
        trades=("notional", "count"),
        notional=("notional", "sum"),
        fee=("fee", "sum"),
    ).sort_values("notional", ascending=False)

    daily_notional = df.groupby("date")["notional"].sum()

    return {
        "total_trades": total_trades,
        "n_opens": len(opens),
        "n_closes": len(closes),
        "total_notional": total_notional,
        "total_fee": total_fee,
        "avg_trade_notional": total_notional / total_trades if total_trades else 0,
        "daily_notional": daily_notional,
        "by_product": by_product,
        "df": df,
    }


def estimate_holding_period(trades, dates):
    """估算平均持仓天数."""
    if not trades:
        return 0.0, 0.0
    from collections import defaultdict
    date_str_to_idx = {str(d.date()): i for i, d in enumerate(dates)}
    opens_by_symbol = defaultdict(list)
    holdings = []
    for t in trades:
        idx = date_str_to_idx.get(t.date)
        if t.side == "open":
            opens_by_symbol[t.symbol].append(idx)
        elif t.side == "close":
            if opens_by_symbol[t.symbol]:
                open_i = opens_by_symbol[t.symbol].pop(0)
                if open_i is not None and idx is not None:
                    holdings.append(idx - open_i)
    if not holdings:
        return 0.0, 0.0
    return np.mean(holdings), np.median(holdings)


def main():
    args = parse_args("fut_v1 Transaction Cost Analysis")
    products = get_products(args.quick)
    n = len(products)

    t0 = time.time()
    print("=" * 70)
    print(f"fut_v1 Strategy — Transaction Cost Analysis ({n} Products)")
    print("=" * 70)

    result, metrics = run_strategy(products=products, start=args.start, end=args.end)
    elapsed = time.time() - t0
    print(f"Backtest done in {elapsed:.0f}s\n")

    # 1. 基线指标
    print_metrics(metrics, "BASELINE PERFORMANCE (no extra costs)")

    # 2. 交易统计
    stats = analyze_trades(result.trades)
    print(f"\n{'=' * 70}")
    print("TRADE STATISTICS")
    print("=" * 70)
    print(f"  Total trades:          {stats['total_trades']:>8d}")
    print(f"    Opens:               {stats['n_opens']:>8d}")
    print(f"    Closes:              {stats['n_closes']:>8d}")
    print(f"  Total notional:        {stats['total_notional']:>15,.0f} CNY")
    print(f"  Avg trade notional:    {stats['avg_trade_notional']:>15,.0f} CNY")
    print(f"  Total commission paid: {stats['total_fee']:>15,.0f} CNY")

    # 3. 持仓天数
    avg_hold, med_hold = estimate_holding_period(result.trades, result.dates)
    print(f"\n{'=' * 70}")
    print("HOLDING PERIOD")
    print("=" * 70)
    print(f"  Average holding period:  {avg_hold:.1f} days")
    print(f"  Median holding period:   {med_hold:.1f} days")

    # 4. 换手率
    dates_idx = pd.DatetimeIndex(result.dates)
    equity_series = pd.Series(result.equity, index=dates_idx)
    daily_not_full = stats["daily_notional"].reindex(dates_idx, fill_value=0.0)
    equity_aligned = equity_series.reindex(dates_idx).ffill()

    ann_turnover = (daily_not_full / equity_aligned).mean() * 252

    print(f"\n{'=' * 70}")
    print("TURNOVER RATE")
    print("=" * 70)
    print(f"  Daily notional traded (avg): {daily_not_full.mean():>15,.0f} CNY")
    print(f"  Annualized turnover rate:    {ann_turnover:>10.2f}x")

    # 5. 成本敏感性分析
    initial_cash = DEFAULT_CONFIG["initial_cash"]
    comm_rates = [0.0001, 0.0002, 0.0005]
    comm_labels = ["0.01%", "0.02%", "0.05%"]
    slippage_rate = 0.0005

    print(f"\n{'=' * 70}")
    print("COST SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"  {'Scenario':<38s} {'AnnCost':>8s} {'NetSharpe':>10s} {'NetRet':>8s}")
    print(f"  {'-'*68}")

    scenarios = [
        ("Low (0.01% comm + 0.05% slip)", 0.0001, slippage_rate),
        ("Mid  (0.02% comm + 0.05% slip)", 0.0002, slippage_rate),
        ("High (0.05% comm + 0.05% slip)", 0.0005, slippage_rate),
    ]

    for name, comm, slip in scenarios:
        daily_cost = daily_not_full * (comm + slip)
        total_cost = daily_cost.sum()
        ann_cost_pct = total_cost / metrics["Years"] / initial_cash
        net_ret = metrics["AnnRet"] - ann_cost_pct
        net_sharpe = (net_ret - 0.02) / metrics["AnnVol"] if metrics["AnnVol"] > 0 else 0
        print(f"  {name:<38s} {ann_cost_pct:>7.2%} {net_sharpe:>10.2f} {net_ret:>7.2%}")

    # 6. 按品种成交额分布
    print(f"\n{'=' * 70}")
    print("TOP 15 PRODUCTS BY TRADED NOTIONAL")
    print("=" * 70)
    bp = stats["by_product"].head(15)
    print(f"  {'Product':>8s} {'Trades':>8s} {'Notional':>18s} {'Fee':>12s}")
    print(f"  {'-'*50}")
    for prod, row in bp.iterrows():
        print(f"  {prod:>8s} {int(row['trades']):>8d} {row['notional']:>18,.0f} {row['fee']:>12,.0f}")

    # 7. 总结
    low_cost_pct = (daily_not_full * 0.0006).sum() / metrics["Years"] / initial_cash
    net_sharpe_low = (metrics["AnnRet"] - low_cost_pct - 0.02) / metrics["AnnVol"] if metrics["AnnVol"] > 0 else 0

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"  Baseline Sharpe:          {metrics['Sharpe']:.2f}")
    print(f"  Total cost (0.01%+slip):  {low_cost_pct:.2%}/yr")
    print(f"  Net Sharpe (low cost):    {net_sharpe_low:.2f}")
    print(f"  Annualized turnover:      {ann_turnover:.1f}x")
    print(f"  Avg holding period:       {avg_hold:.1f} days")

    if net_sharpe_low > 0.5:
        print(f"  Verdict: Strategy SURVIVES realistic costs with margin.")
    elif net_sharpe_low > 0:
        print(f"  Verdict: Strategy is MARGINAL after costs — monitor closely.")
    else:
        print(f"  Verdict: Strategy does NOT survive realistic costs.")

    print(f"\nTotal analysis time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
