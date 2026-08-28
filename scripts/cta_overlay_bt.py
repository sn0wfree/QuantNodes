#!/usr/bin/env python3
"""CTA + Options Overlay 回测脚本.

对比: 纯 CTA vs CTA + 期权 overlay
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import pandas as pd
import numpy as np
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.options.iv_panel import load_iv_panel
from QuantNodes.strategy.cta.options.hedge_data import load_vrp_panel
from QuantNodes.strategy.cta.options.cta_overlay import CTAOptionsOverlay, OverlayConfig
from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig, CtaStrategy
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


def run_backtest(product, use_overlay=True, start="2023-01-01", end="2025-12-31"):
    """运行单品种 CTA + Overlay 回测."""
    t0 = time.time()
    loader = CtaDataLoader()
    um = UnderlyingMap(loader)
    oe = OptionExpiry(loader)
    registry = CtaContractRegistry()

    # 加载价格
    px = loader.load_continuous_main(product)
    px["trade_date"] = pd.to_datetime(px["trade_date"])
    px = px.set_index("trade_date")
    price_series = px["close"]

    # 加载期权数据
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
        contract = CtaContract(
            product=product, symbol=sym, expiry_year=0, expiry_month=0,
            spec=registry.get_or_default(product), option_spec=opt_spec,
        )
        option_contracts[sym] = contract

    # 价格映射
    dates = price_series.index[(price_series.index >= start) & (price_series.index <= end)]
    prices_map = {product: price_series.reindex(dates).tolist()}
    products_map = {product: product}
    for sym in opt_df["symbol"].unique():
        sym_df = opt_df[opt_df["symbol"] == sym].set_index("trade_date")
        prices_map[sym] = sym_df["close"].reindex(dates).tolist()
        products_map[sym] = product

    # Overlay 配置
    overlay_cfg = OverlayConfig()

    # CTA 信号 (MA 趋势)
    ma5 = price_series.rolling(5).mean()
    ma20 = price_series.rolling(20).mean()
    ma60 = price_series.rolling(60).mean()
    mom20 = price_series.pct_change(20)
    mom60 = price_series.pct_change(60)

    cta_signal = (
        (ma5 > ma20).astype(float) * 0.3
        + (ma20 > ma60).astype(float) * 0.3
        + (mom20 > 0).astype(float) * 0.2
        + (mom60 > 0).astype(float) * 0.2
    ) * 2 - 1

    # IV rank (简化: 用 rolling percentile)
    iv_panel = None
    try:
        iv_panel = load_iv_panel(start=start, end=end)
    except Exception:
        pass

    # VRP
    vrp_panel = None
    try:
        vrp_panel = load_vrp_panel(products=[product], start=start, end=end)
    except Exception:
        pass

    # 状态
    state = {
        "holdings": {},
        "overlay_holdings": {},
        "current_signal": None,
    }

    overlay = CTAOptionsOverlay(overlay_cfg)

    class Strategy(CtaStrategy):
        def target_positions(self, date, data):
            spot = price_series.loc[date] if date in price_series.index else None
            if spot is None or np.isnan(spot):
                return {}

            all_targets = {}

            # ── CTA 期货仓位 (90% 资金) ──
            cta_val = cta_signal.get(date, 0) if date in cta_signal.index else 0
            vol = price_series.pct_change().rolling(20).std().get(date, 0.02) * np.sqrt(252)
            if vol > 0:
                target_weight = cta_val * (0.15 / vol)
                target_weight = np.clip(target_weight, -3, 3)
            else:
                target_weight = 0

            spec = registry.get_or_default(product)
            notional = spot * spec.contract_multiplier
            cta_lots = round(target_weight * 1_000_000 * overlay_cfg.cta_pct / notional)
            if cta_lots != 0:
                all_targets[product] = cta_lots

            # ── 期权 Overlay (10% 资金) ──
            if use_overlay:
                # 获取 IV rank
                iv_rank = 0.5
                if iv_panel is not None and product in iv_panel.columns:
                    if date in iv_panel.index:
                        iv_val = iv_panel.loc[date, product]
                        if pd.notna(iv_val):
                            hist = iv_panel[product].dropna()
                            if len(hist) > 60:
                                iv_rank = float((hist < iv_val).sum() / len(hist))

                # 获取 VRP
                vrp_pct = 0.5
                if vrp_panel is not None and product in vrp_panel.columns:
                    if date in vrp_panel.index:
                        vrp_val = vrp_panel.loc[date, product]
                        if pd.notna(vrp_val):
                            hist = vrp_panel[product].dropna()
                            if len(hist) > 60:
                                vrp_pct = float((hist < vrp_val).sum() / len(hist))

                # 检查 overlay 持仓换月
                if product in state["overlay_holdings"]:
                    h = state["overlay_holdings"][product]
                    if (h["expiry"] - date).days <= overlay_cfg.roll_days:
                        for sym in h["symbols"]:
                            all_targets[sym] = 0.0
                        state["overlay_holdings"].pop(product, None)

                # 最小持仓期
                if product in state["overlay_holdings"]:
                    h = state["overlay_holdings"][product]
                    if (date - h["entry_date"]).days < 15:
                        pass  # 继续持有
                    else:
                        # 平旧仓
                        state["overlay_holdings"].pop(product)
                        for sym in h["symbols"]:
                            all_targets[sym] = 0.0

                # 新开 overlay
                if product not in state["overlay_holdings"]:
                    overlay_targets = overlay.get_overlay_targets(
                        product, date, spot,
                        iv_rank=iv_rank, vrp_pct=vrp_pct,
                        loader=loader, um=um, oe=oe,
                    )
                    if overlay_targets:
                        state["overlay_holdings"][product] = {
                            "symbols": overlay_targets,
                            "expiry": date + pd.Timedelta(days=overlay_cfg.max_dte),
                            "entry_date": date,
                        }
                        all_targets.update(overlay_targets)

            return all_targets

    result = run_cta_backtest(
        dates=list(dates),
        prices=prices_map,
        products=products_map,
        strategy=Strategy(),
        config=CtaBacktestConfig(
            initial_cash=1_000_000,
            commission_rate=0.0001,
        ),
        registry=registry,
        option_contracts=option_contracts,
    )

    metrics = compute_full_metrics(result.nav)
    elapsed = time.time() - t0
    return result.nav, result.trades, metrics, elapsed


def main():
    print("CTA + Options Overlay Strategy — 2023~2025")
    print("=" * 70)

    products = ["RB", "I", "AG"]

    # 基线: 纯 CTA (无 overlay)
    print("\n--- Baseline: Pure CTA (no overlay) ---")
    for prod in products:
        nav, trades, metrics, elapsed = run_backtest(prod, use_overlay=False)
        m = metrics
        print(
            f"  {prod}: AnnRet={m['ann_return']:>6.1%}  Sharpe={m['sharpe']:>6.2f}  "
            f"MaxDD={m['max_drawdown']:>6.1%}  Calmar={m['calmar']:>6.2f}  "
            f"Trades={len(trades):>4d}  ({elapsed:.0f}s)"
        )

    # CTA + Overlay
    print("\n--- CTA + Options Overlay ---")
    for prod in products:
        nav, trades, metrics, elapsed = run_backtest(prod, use_overlay=True)
        m = metrics
        print(
            f"  {prod}: AnnRet={m['ann_return']:>6.1%}  Sharpe={m['sharpe']:>6.2f}  "
            f"MaxDD={m['max_drawdown']:>6.1%}  Calmar={m['calmar']:>6.2f}  "
            f"Trades={len(trades):>4d}  ({elapsed:.0f}s)"
        )

    # 对比
    print("\n" + "=" * 70)
    print("Comparison:")
    print(f"  {'':>10} {'Pure CTA':>15} {'CTA+Overlay':>15} {'Improvement':>15}")
    print("  " + "-" * 55)

    for prod in products:
        _, _, m_base, _ = run_backtest(prod, use_overlay=False)
        _, _, m_over, _ = run_backtest(prod, use_overlay=True)

        sharpe_imp = m_over["sharpe"] - m_base["sharpe"]
        dd_imp = m_over["max_drawdown"] - m_base["max_drawdown"]
        ret_imp = m_over["ann_return"] - m_base["ann_return"]

        print(
            f"  {prod:>10} "
            f"Sharpe={m_base['sharpe']:>5.2f}  DD={m_base['max_drawdown']:>5.1%}  Ret={m_base['ann_return']:>5.1%}  "
            f"Sharpe={m_over['sharpe']:>5.2f}  DD={m_over['max_drawdown']:>5.1%}  Ret={m_over['ann_return']:>5.1%}  "
            f"dSh={sharpe_imp:+.2f}  dDD={dd_imp:+.1%}"
        )


if __name__ == "__main__":
    main()
