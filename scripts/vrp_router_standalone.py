#!/usr/bin/env python3
"""VRP Router 期权层独立回测 (不含 fut_v1 期货).

只交易双向 VRP 路由的期权 targets, 独立账户核算.
输出: 年化收益/年化波动/Sharpe/MaxDD/Calmar + 日胜率 + 交易级统计.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.vrp_router import VrpRouterStrategy, VrpRouterConfig
from QuantNodes.strategy.cta.strategies.factor_modules import ComboConfig
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from scripts.diagonal_lifecycle_bt import load_option_data, build_option_prices

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
INITIAL = 5_000_000


class OptionOnlyRouter(VrpRouterStrategy):
    """仅交易期权 targets, 不叠加期货."""

    def target_positions(self, date, data) -> dict[str, float]:
        if not hasattr(self, "_precomputed_router") or date not in self._precomputed_router:
            return {}
        return self._precomputed_router[date]


def full_metrics(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann = (1 + total) ** (1 / max(years, 0.01)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (ann - 0.02) / vol if vol > 0 else 0
    downside = r[r < 0]
    sortino = (ann - 0.02) / (downside.std() * np.sqrt(252)) if len(downside) > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann / abs(dd) if abs(dd) > 0 else 0
    daily_win = (r > 0).mean()
    return {"AnnRet": ann, "AnnVol": vol, "Sharpe": sharpe, "Sortino": sortino,
            "MaxDD": dd, "Calmar": calmar, "DailyWin": daily_win,
            "TotalRet": total, "Years": years}


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

    # 生成日期序列 (借 fut_v1 面板)
    s0 = VrpRouterStrategy(
        router_config=VrpRouterConfig(), option_data=option_data,
        option_contracts=option_contracts, greeks_cache=gcache, products=PRODUCTS,
        config=cfg_base, start="2018-01-01", end="2025-12-31",
    )
    s0.load_panel()
    sig = s0._signals()
    es = s0._exec_sig(sig)
    lots = s0._signal_to_lots(es, cfg_base)
    s0._precomputed_lots = lots
    s0._load_iv_panel()
    s0._compute_signal_panels()

    opt_prices, opt_settle = build_option_prices(list(s0._price_panel.index), option_data)
    all_prices = {p: s0._price_panel[p].tolist() for p in PRODUCTS}
    all_prices.update(opt_prices)
    all_products = {p: p for p in PRODUCTS}
    for sym, c in option_contracts.items():
        all_products[sym] = c.product

    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=registry)

    rr = VrpRouterConfig(long_hold=7, short_hold=15, delta_notional_pct=0.20)
    st = OptionOnlyRouter(
        router_config=rr, option_data=option_data, option_contracts=option_contracts,
        greeks_cache=gcache, products=PRODUCTS, config=cfg_base,
        start="2018-01-01", end="2025-12-31",
    )
    st._price_panel = s0._price_panel
    st._ohlc_map = s0._ohlc_map
    st._vols = s0._vols
    st._ois = s0._ois
    st._iv_panel = s0._iv_panel
    st._vrp_pct = s0._vrp_pct
    st._mom20 = s0._mom20
    st._iv_pct = s0._iv_pct
    st._precomputed_lots = lots

    print("\nPrecomputing router targets...")
    t0 = time.time()
    st._precompute_router_targets()
    print(f"  done ({time.time()-t0:.0f}s)  stats={st.stats}")

    t0 = time.time()
    result = run_cta_backtest(
        dates=list(st._price_panel.index),
        prices=all_prices, products=all_products, strategy=st,
        config=CtaBacktestConfig(initial_cash=INITIAL, commission_rate=0.0001,
                                 margin_call=True),
        registry=registry, risk=risk,
        option_contracts=option_contracts, option_settle=opt_settle,
    )
    m = full_metrics(result.nav)
    print(f"  backtest done ({time.time()-t0:.0f}s), trades={len(result.trades)}")

    print("\n" + "=" * 66)
    print("期权层独立业绩 (仅 VRP 路由, 2018-2025)")
    print("=" * 66)
    print(f"  年化收益   : {m['AnnRet']:>8.2%}")
    print(f"  年化波动   : {m['AnnVol']:>8.2%}")
    print(f"  Sharpe     : {m['Sharpe']:>8.2f}")
    print(f"  Sortino    : {m['Sortino']:>8.2f}")
    print(f"  MaxDD      : {m['MaxDD']:>8.2%}")
    print(f"  Calmar     : {m['Calmar']:>8.2f}")
    print(f"  日胜率     : {m['DailyWin']:>8.1%}")
    print(f"  总收益     : {m['TotalRet']:>8.2%}  ({m['Years']:.1f}年)")

    # 交易级统计
    print("\n" + "=" * 66)
    print("交易级统计")
    print("=" * 66)
    df_t = pd.DataFrame([{
        "date": t.date, "symbol": t.symbol, "side": t.side,
        "qty": t.qty, "price": t.price, "notional": t.notional,
    } for t in result.trades])
    if len(df_t):
        opt_mask = df_t["symbol"].isin(option_contracts.keys())
        opt_trades = df_t[opt_mask]
        n_open = len(opt_trades[opt_trades["side"] == "open"])
        n_close = len(opt_trades[opt_trades["side"] == "close"])
        n_expire = len(opt_trades[opt_trades["side"] == "expire"])
        print(f"  期权交易总数: {len(opt_trades)}")
        print(f"    开仓 (open)  : {n_open}")
        print(f"    平仓 (close) : {n_close}")
        print(f"    到期 (expire): {n_expire}")
        if len(opt_trades):
            print(f"    平均单笔金额: {opt_trades['notional'].mean():,.0f} 元")
            print(f"    单笔最大    : {opt_trades['notional'].max():,.0f} 元")

        # 配对盈亏: 同一 symbol 同一方向 open → close/expire
        pair_pnls = []
        for sym, grp in opt_trades.groupby("symbol"):
            grp = grp.sort_values("date")
            open_px = None
            for _, row in grp.iterrows():
                if row["side"] == "open":
                    open_px = row["price"]
                elif row["side"] in ("close", "expire") and open_px is not None:
                    # 期权买方: 盈利 = (平仓价 - 开仓价) × qty
                    # 卖方 (avg 记录方式): 盈利 = (开仓价 - 平仓价) × qty
                    # 此处 trades 无方向信息, 用 open/close 价格差近似单腿
                    pair_pnls.append((row["price"] - open_px) * row["qty"])
                    open_px = None
        if pair_pnls:
            a = np.array(pair_pnls)
            wins = a[a > 0]
            losses = a[a < 0]
            print(f"\n  配对笔数      : {len(a)}")
            print(f"  配对胜率      : {(a > 0).mean():.1%}")
            if len(wins):
                print(f"  平均盈利/笔  : {wins.mean():+,.0f} 元")
            if len(losses):
                print(f"  平均亏损/笔  : {losses.mean():+,.0f} 元")
            if len(wins) and len(losses):
                print(f"  盈亏比        : {abs(wins.mean() / losses.mean()):.2f}")
            print(f"  配对总盈亏   : {a.sum():+,.0f} 元")

    # 期权层资金曲线: NAV 相对初始
    nav = result.nav
    print(f"\n  首日 NAV: {nav.iloc[0]:.4f}, 末日 NAV: {nav.iloc[-1]:.4f}")
    print(f"  期权层累计盈亏: {(nav.iloc[-1]-1)*INITIAL:+,.0f} 元")


if __name__ == "__main__":
    main()
