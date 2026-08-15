#!/usr/bin/env python3
"""Options-Enhanced CTA 回测对比.

对比: 纯 CTA vs 期权因子增强 CTA
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import pandas as pd
import numpy as np
from QuantNodes.strategy.cta.options.options_enhanced_cta import (
    OptionsEnhancedCTA,
    OptionsEnhancedCTAConfig,
    run_options_enhanced_cta,
)
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig, CtaStrategy
from QuantNodes.strategy.cta.backtest.cta_contract import CtaContractRegistry
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk


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


def run_baseline_cta(products, start="2023-01-01", end="2025-12-31"):
    """运行基线 CTA (无期权因子), 使用 CtaBroker."""
    loader = CtaDataLoader()
    registry = CtaContractRegistry()

    # 加载价格面板
    price_data = {}
    for prod in products:
        try:
            px = loader.load_continuous_main(prod)
            if px is not None and not px.empty:
                px["trade_date"] = pd.to_datetime(px["trade_date"])
                px = px.set_index("trade_date")
                price_data[prod] = px["close"]
        except Exception:
            pass

    price_panel = pd.DataFrame(price_data).dropna(how="all")
    price_panel = price_panel.loc[(price_panel.index >= start) & (price_panel.index <= end)]

    # CTA 信号
    cta_signals = {}
    for product in price_panel.columns:
        series = price_panel[product]
        ma5 = series.rolling(5).mean()
        ma20 = series.rolling(20).mean()
        ma60 = series.rolling(60).mean()
        mom5 = series.pct_change(5)
        mom20 = series.pct_change(20)
        mom60 = series.pct_change(60)

        sig = (
            (ma5 > ma20).astype(float) * 0.25
            + (ma20 > ma60).astype(float) * 0.25
            + (mom5 > 0).astype(float) * 0.15
            + (mom20 > 0).astype(float) * 0.20
            + (mom60 > 0).astype(float) * 0.15
        ) * 2 - 1
        cta_signals[product] = sig

    # vol_target
    vol = price_panel.pct_change().rolling(20).std() * np.sqrt(252)
    target_vol = 0.15

    weights = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)
    for product in price_panel.columns:
        sig = cta_signals[product]
        v = vol[product]
        raw = sig * (target_vol / v.replace(0, np.nan))
        weights[product] = raw.clip(-3, 3).fillna(0)

    # 转为手数
    equity = 10_000_000
    n_active = (weights.abs() > 1e-9).sum(axis=1).clip(lower=1)
    lots = pd.DataFrame(0.0, index=price_panel.index, columns=price_panel.columns)
    for product in price_panel.columns:
        spec = registry.get_or_default(product)
        notional = price_panel[product] * spec.contract_multiplier
        capital = equity / n_active
        with np.errstate(divide="ignore", invalid="ignore"):
            lots[product] = (weights[product] * capital / notional).round()

    # 使用 CtaBroker 回测
    class BaselineStrategy(CtaStrategy):
        def __init__(self, lots_df):
            self._lots = lots_df
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
            self._last = targets
            return targets

    products_map = {p: p for p in price_panel.columns}
    risk = PositionLimitRisk(products=products_map, registry=registry)

    result = run_cta_backtest(
        dates=list(price_panel.index),
        prices={p: price_panel[p].tolist() for p in price_panel.columns},
        products=products_map,
        strategy=BaselineStrategy(lots),
        config=CtaBacktestConfig(initial_cash=equity, commission_rate=0.0001),
        registry=registry,
        risk=risk,
    )
    return result.nav


def main():
    products = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]

    print("Options-Enhanced CTA — Multi-Product Portfolio")
    print("=" * 70)

    # 基线 CTA
    print("\nBaseline CTA (no options factors):")
    t0 = time.time()
    nav_base = run_baseline_cta(products)
    m_base = compute_full_metrics(nav_base)
    t1 = time.time() - t0
    print(f"  AnnRet={m_base['ann_return']:>6.1%}  Sharpe={m_base['sharpe']:>6.2f}  "
          f"Sortino={m_base['sortino']:>6.2f}  MaxDD={m_base['max_drawdown']:>6.1%}  "
          f"Calmar={m_base['calmar']:>6.2f}  WinRate={m_base['win_rate']:>5.1%}  ({t1:.0f}s)")

    # 期权增强 CTA
    print("\nOptions-Enhanced CTA:")
    cfg = OptionsEnhancedCTAConfig(
        products=products,
        initial_cash=10_000_000,
        target_vol=0.15,
        start="2023-01-01",
        end="2025-12-31",
    )

    # 加载价格面板 (与 baseline 相同)
    loader = CtaDataLoader()
    price_data = {}
    for prod in products:
        try:
            px = loader.load_continuous_main(prod)
            if px is not None and not px.empty:
                px["trade_date"] = pd.to_datetime(px["trade_date"])
                px = px.set_index("trade_date")
                price_data[prod] = px["close"]
        except Exception:
            pass
    price_panel = pd.DataFrame(price_data).dropna(how="all")
    price_panel = price_panel.loc[(price_panel.index >= cfg.start) & (price_panel.index <= cfg.end)]

    # 加载期权因子
    iv_panel = None
    vrp_panel = None
    try:
        iv_panel = load_iv_panel(start=cfg.start, end=cfg.end)
    except Exception:
        pass
    try:
        vrp_panel = load_vrp_panel(products=products, start=cfg.start, end=cfg.end)
    except Exception:
        pass

    # 创建增强策略
    enhanced_strategy = OptionsEnhancedCTA(
        price_panel=price_panel,
        config=cfg,
        iv_panel=iv_panel,
        vrp_panel=vrp_panel,
    )

    # 运行回测 (CtaBroker)
    registry = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in products}, registry=registry)

    t0 = time.time()
    result = run_cta_backtest(
        dates=list(price_panel.index),
        prices={p: price_panel[p].tolist() for p in products},
        products={p: p for p in products},
        strategy=enhanced_strategy,
        config=CtaBacktestConfig(initial_cash=cfg.initial_cash, commission_rate=0.0001),
        registry=registry,
        risk=risk,
    )
    m_enh = compute_full_metrics(result.nav)
    t2 = time.time() - t0
    print(f"  AnnRet={m_enh['ann_return']:>6.1%}  Sharpe={m_enh['sharpe']:>6.2f}  "
          f"Sortino={m_enh['sortino']:>6.2f}  MaxDD={m_enh['max_drawdown']:>6.1%}  "
          f"Calmar={m_enh['calmar']:>6.2f}  WinRate={m_enh['win_rate']:>5.1%}  ({t2:.0f}s)")

    # 对比
    print("\n" + "=" * 70)
    print("Improvement:")
    for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar", "win_rate"]:
        v1 = m_base.get(key, 0)
        v2 = m_enh.get(key, 0)
        diff = v2 - v1
        if key in ["ann_return", "ann_vol", "max_drawdown", "win_rate"]:
            print(f"  {key:>15}: {v1:>7.1%} → {v2:>7.1%} ({diff:+.1%})")
        else:
            print(f"  {key:>15}: {v1:>7.2f} → {v2:>7.2f} ({diff:+.2f})")


if __name__ == "__main__":
    main()
