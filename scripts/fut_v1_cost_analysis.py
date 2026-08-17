#!/usr/bin/env python3
"""fut_v1 策略交易成本分析 — 交易统计、换手率、手续费、滑点、成本后净收益."""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS

PRODUCTS = DEFAULT_PRODUCTS
START, END = "2018-01-01", "2025-12-31"


def compute_metrics(nav, rf=0.02):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann_ret = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 0 else 0
    sortino_denom = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 0.001
    sortino = (ann_ret - rf) / sortino_denom
    return {
        "AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": dd, "Calmar": calmar, "TotalRet": total, "Years": years,
    }


def analyze_trades(trades):
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
    total_qty = df["qty"].sum()

    # 开平仓分离
    opens = df[df["side"] == "open"]
    closes = df[df["side"] == "close"]
    n_opens = len(opens)
    n_closes = len(closes)

    # 逐品种统计
    by_product = df.groupby("product").agg(
        trades=("notional", "count"),
        notional=("notional", "sum"),
        fee=("fee", "sum"),
    ).sort_values("notional", ascending=False)

    # 逐日成交额
    daily_notional = df.groupby("date")["notional"].sum()

    # 持仓日数 (按 symbol 看 position 变化)
    return {
        "total_trades": total_trades,
        "n_opens": n_opens,
        "n_closes": n_closes,
        "total_notional": total_notional,
        "total_fee": total_fee,
        "total_qty": total_qty,
        "avg_trade_notional": total_notional / total_trades if total_trades else 0,
        "avg_fee_per_trade": total_fee / total_trades if total_trades else 0,
        "daily_notional": daily_notional,
        "by_product": by_product,
        "df": df,
    }


def estimate_holding_period(trades, dates):
    """估算平均持仓天数: 用 position 面板中每个品种持仓存在的天数."""
    if not trades:
        return 0.0, 0.0
    # trade.date 是字符串 "YYYY-MM-DD", dates 是 pd.Timestamp
    date_str_to_idx = {str(d.date()): i for i, d in enumerate(dates)}
    from collections import defaultdict
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
    t0 = time.time()
    print("=" * 70)
    print("fut_v1 Strategy — Transaction Cost Analysis")
    print("=" * 70)

    cfg = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0,
        vol_lookback=20, momentum_filter_window=60, momentum_filter_coef=0.0,
        filter_enabled=True, module_weighting="inv_vol", continuous=True,
        cont_z_window=252, cont_z_clip=3.0, margin_call=True,
        execution_lag=1, aggregation="tvpr",
    )

    print(f"Products: {len(PRODUCTS)} | Period: {START} ~ {END}")
    print(f"Initial cash: 5,000,000 CNY\n")

    print("Running strategy...")
    strat = FactorModuleComboStrategy(products=PRODUCTS, config=cfg, start=START, end=END)
    result = strat.run()
    elapsed = time.time() - t0
    print(f"Backtest done in {elapsed:.0f}s\n")

    # ── 1. 基线指标 ──────────────────────────────────────────────────────
    m = compute_metrics(result.nav)
    print("=" * 70)
    print("1. BASELINE PERFORMANCE (no extra costs)")
    print("=" * 70)
    print(f"  Annual Return:  {m['AnnRet']:>8.2%}")
    print(f"  Annual Vol:     {m['AnnVol']:>8.2%}")
    print(f"  Sharpe Ratio:   {m['Sharpe']:>8.2f}")
    print(f"  Sortino Ratio:  {m['Sortino']:>8.2f}")
    print(f"  Max Drawdown:   {m['MaxDD']:>8.2%}")
    print(f"  Calmar Ratio:   {m['Calmar']:>8.2f}")
    print(f"  Total Return:   {m['TotalRet']:>8.2%}")
    print(f"  Period:         {m['Years']:>8.1f} years")

    # ── 2. 交易统计 ──────────────────────────────────────────────────────
    stats = analyze_trades(result.trades)
    print(f"\n{'=' * 70}")
    print("2. TRADE STATISTICS")
    print("=" * 70)
    print(f"  Total trades:          {stats['total_trades']:>8d}")
    print(f"    Opens:               {stats['n_opens']:>8d}")
    print(f"    Closes:              {stats['n_closes']:>8d}")
    print(f"  Total notional:        {stats['total_notional']:>15,.0f} CNY")
    print(f"  Avg trade notional:    {stats['avg_trade_notional']:>15,.0f} CNY")
    print(f"  Total commission paid: {stats['total_fee']:>15,.0f} CNY")
    print(f"  Avg fee per trade:     {stats['avg_fee_per_trade']:>15,.0f} CNY")

    # ── 3. 平均持仓天数 ──────────────────────────────────────────────────
    avg_hold, med_hold = estimate_holding_period(result.trades, result.dates)
    print(f"\n{'=' * 70}")
    print("3. HOLDING PERIOD")
    print("=" * 70)
    print(f"  Average holding period:  {avg_hold:.1f} days")
    print(f"  Median holding period:   {med_hold:.1f} days")

    # ── 4. 换手率 ─────────────────────────────────────────────────────────
    dates_idx = pd.DatetimeIndex(result.dates)
    nav_series = result.nav.reindex(dates_idx)
    equity_series = pd.Series(result.equity, index=dates_idx)

    daily_notional = stats["daily_notional"]
    # 对齐到交易日
    all_dates = dates_idx
    daily_not_full = daily_notional.reindex(all_dates, fill_value=0.0)
    # NAV/Equity 作为分母
    nav_aligned = nav_series.reindex(all_dates).ffill()
    equity_aligned = equity_series.reindex(all_dates).ffill()
    initial_cash = cfg.initial_cash

    turnover_daily = daily_not_full / equity_aligned
    # 年化换手率 = 日均换手率 × 252
    avg_daily_turnover = turnover_daily.mean()
    ann_turnover = avg_daily_turnover * 252

    # 月度换手率
    monthly_notional = daily_not_full.resample("M").sum()
    monthly_equity = equity_aligned.resample("M").last()
    monthly_turnover = monthly_notional / monthly_equity

    print(f"\n{'=' * 70}")
    print("4. TURNOVER RATE")
    print("=" * 70)
    print(f"  Daily notional traded (avg): {daily_not_full.mean():>15,.0f} CNY")
    print(f"  Daily notional traded (max): {daily_not_full.max():>15,.0f} CNY")
    print(f"  Daily turnover rate (avg):   {avg_daily_turnover:>10.4%}")
    print(f"  Annualized turnover rate:    {ann_turnover:>10.2f}x")
    print(f"  Monthly turnover (avg):      {monthly_turnover.mean():>10.2%}")
    print(f"  Monthly turnover (max):      {monthly_turnover.max():>10.2%}")

    # ── 5. 不同手续费率下的年化成本 ──────────────────────────────────────
    comm_rates = [0.0001, 0.0002, 0.0005]  # 0.01%, 0.02%, 0.05%
    comm_labels = ["0.01%", "0.02%", "0.05%"]

    print(f"\n{'=' * 70}")
    print("5. ESTIMATED ANNUAL TRANSACTION COSTS (commission only)")
    print("=" * 70)
    print(f"  {'Rate':>8s}  {'Daily Cost':>15s}  {'Annual Cost':>15s}  {'Ann Cost %':>10s}")
    print(f"  {'-'*52}")

    annual_costs_comm = {}
    for rate, label in zip(comm_rates, comm_labels):
        daily_cost = daily_not_full * rate
        total_cost = daily_cost.sum()
        annual_cost = total_cost / m["Years"]
        annual_cost_pct = annual_cost / initial_cash
        annual_costs_comm[label] = annual_cost_pct
        print(f"  {label:>8s}  {daily_cost.mean():>15,.0f}  {annual_cost:>15,.0f}  {annual_cost_pct:>9.2%}")

    # 实际已付手续费
    actual_fee_annual = stats["total_fee"] / m["Years"]
    actual_fee_pct = stats["total_fee"] / initial_cash / m["Years"]
    print(f"\n  Actual (0.01%):   {actual_fee_annual:>15,.0f}/yr  ({actual_fee_pct:.2%} of capital)")

    # ── 6. 滑点影响 ───────────────────────────────────────────────────────
    slippage_rate = 0.0005  # 0.05% per trade

    print(f"\n{'=' * 70}")
    print("6. SLIPPAGE IMPACT (0.05% per trade side)")
    print("=" * 70)

    # 滑点 = 成交额 × slippage_rate (每笔单边)
    daily_slippage = daily_not_full * slippage_rate
    total_slippage = daily_slippage.sum()
    annual_slippage = total_slippage / m["Years"]
    annual_slippage_pct = annual_slippage / initial_cash

    print(f"  Slippage rate:        {slippage_rate:.2%} per trade side")
    print(f"  Daily slippage (avg): {daily_slippage.mean():>15,.0f} CNY")
    print(f"  Annual slippage:      {annual_slippage:>15,.0f} CNY")
    print(f"  Annual slippage %:    {annual_slippage_pct:>10.2%}")

    # ── 7. 综合成本后净收益 ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("7. PERFORMANCE AFTER REALISTIC COSTS")
    print("=" * 70)

    scenarios = [
        ("Low (0.01% comm + 0.05% slip)", 0.0001, 0.0005),
        ("Mid  (0.02% comm + 0.05% slip)", 0.0002, 0.0005),
        ("High (0.05% comm + 0.05% slip)", 0.0005, 0.0005),
    ]

    print(f"\n  {'Scenario':<38s} {'AnnCost':>8s} {'NetSharpe':>10s} {'NetRet':>8s} {'Survives?':>10s}")
    print(f"  {'-'*78}")

    for name, comm, slip in scenarios:
        daily_cost = daily_not_full * (comm + slip)
        total_cost = daily_cost.sum()
        ann_cost_pct = total_cost / m["Years"] / initial_cash
        net_ret = m["AnnRet"] - ann_cost_pct
        net_sharpe = (net_ret - 0.02) / m["AnnVol"] if m["AnnVol"] > 0 else 0
        survives = "YES" if net_sharpe > 0.5 else ("MARGINAL" if net_sharpe > 0 else "NO")
        print(f"  {name:<38s} {ann_cost_pct:>7.2%} {net_sharpe:>10.2f} {net_ret:>7.2%} {survives:>10s}")

    # 额外: 保守估计 (双边滑点)
    print(f"\n  --- Conservative (0.05% slippage BOTH sides) ---")
    for name, comm, slip in scenarios:
        daily_cost = daily_not_full * (comm + slip * 2)
        total_cost = daily_cost.sum()
        ann_cost_pct = total_cost / m["Years"] / initial_cash
        net_ret = m["AnnRet"] - ann_cost_pct
        net_sharpe = (net_ret - 0.02) / m["AnnVol"] if m["AnnVol"] > 0 else 0
        survives = "YES" if net_sharpe > 0.5 else ("MARGINAL" if net_sharpe > 0 else "NO")
        print(f"  {name:<38s} {ann_cost_pct:>7.2%} {net_sharpe:>10.2f} {net_ret:>7.2%} {survives:>10s}")

    # ── 8. 按品种成交额分布 ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("8. TOP 15 PRODUCTS BY TRADED NOTIONAL")
    print("=" * 70)
    bp = stats["by_product"].head(15)
    print(f"  {'Product':>8s} {'Trades':>8s} {'Notional':>18s} {'Fee':>12s} {'% of Total':>10s}")
    print(f"  {'-'*60}")
    for prod, row in bp.iterrows():
        pct = row["notional"] / stats["total_notional"]
        print(f"  {prod:>8s} {int(row['trades']):>8d} {row['notional']:>18,.0f} {row['fee']:>12,.0f} {pct:>9.1%}")

    # ── 9. 总结 ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("9. SUMMARY")
    print("=" * 70)
    total_cost_pct = annual_costs_comm["0.01%"] + annual_slippage_pct
    net_sharpe_low = (m["AnnRet"] - total_cost_pct - 0.02) / m["AnnVol"] if m["AnnVol"] > 0 else 0
    print(f"  Baseline Sharpe:          {m['Sharpe']:.2f}")
    print(f"  Total cost (0.01%+slip):  {total_cost_pct:.2%}/yr")
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
