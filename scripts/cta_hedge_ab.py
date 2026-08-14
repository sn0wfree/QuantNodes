# coding=utf-8
"""CTA 期权对冲 A/B 对照: fut_v1 + 纯 Put 尾部保护 (docs/99).

对比 timing_mode: none / iv / basis / both.

流程:
    1. 跑 fut_v1 → 逐日持仓 + NAV + 价格面板
    2. 构建 IV 分位面板 与 月差分位面板
    3. 对每个 timing_mode, 叠加买入持有的虚值 Put 对冲:
       - 净多头品种且择时系数 > 0 → 选 Put (按 (product, expiry) 缓存)
       - 逐日按期权 close 市价结算 (mark-to-market)
       - 到期按内在价值 max(K-S, 0) 结算
    4. 输出各模式指标对比 (NAV/Sharpe/MaxDD/最差单日/对冲净成本)

用法:
    python scripts/cta_hedge_ab.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from QuantNodes.strategy.cta.options.expiry import OptionExpiry
from QuantNodes.strategy.cta.options.hedge_data import load_basis_panel
from QuantNodes.strategy.cta.options.hedge_layer import (
    HedgeConfig, timing_coefficient,
)
from QuantNodes.strategy.cta.options.iv_panel import add_percentile, load_iv_panel
from QuantNodes.strategy.cta.options.selector import select_put
from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

CFG = ComboConfig(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True,
)

HEDGE_CFG = HedgeConfig(
    timing_mode="iv", target_delta=0.25, hedge_ratio=0.10,
    min_days=30, max_days=60, roll_days=7,
)

INITIAL_CASH = 1_000_000.0


def _load_iv_basis_panels(dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """加载 IV 分位面板 + 月差分位面板 (对齐 CTA 日期)."""
    start, end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    iv = load_iv_panel(start=start, end=end)
    iv_pct = add_percentile(iv)
    basis = load_basis_panel(start=start, end=end)
    basis_pct = add_percentile(basis)
    idx = pd.DatetimeIndex(dates)
    return iv_pct.reindex(idx), basis_pct.reindex(idx)


def _build_premium_frames(products: list[str], loader: CtaDataLoader,
                          dates: list[pd.Timestamp]) -> dict[str, pd.DataFrame]:
    """预加载期权 close 面板 {product: (date × symbol → close)} 用于市价结算."""
    start, end = dates[0].strftime("%Y-%m-%d"), dates[-1].strftime("%Y-%m-%d")
    frames = {}
    for p in products:
        df = loader.load_v_all_options(products=[p], start=start, end=end)
        if df is None or df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["symbol"] = df["symbol"].str.upper()
        piv = df.pivot_table(index="trade_date", columns="symbol", values="close")
        frames[p] = piv
    return frames


class _PutHedgeSim:
    """买入持有的 Put 尾部保护模拟 (逐日市价结算 + 到期内在价值结算)."""

    def __init__(self, cfg: HedgeConfig, premium_frames: dict,
                 iv_pct: pd.DataFrame, basis_pct: pd.DataFrame,
                 shared_sel_cache: dict | None = None):
        self.cfg = cfg
        self.premium_frames = premium_frames
        self.iv_pct = iv_pct
        self.basis_pct = basis_pct
        self.loader = CtaDataLoader()
        self.um = UnderlyingMap(self.loader)
        self.oe = OptionExpiry(self.loader)
        self._sel_cache: dict = shared_sel_cache if shared_sel_cache is not None else {}

    def _select(self, product: str, ts: pd.Timestamp, px: float, key: str):
        if key in self._sel_cache:
            return self._sel_cache[key]
        sel = select_put(product, ts, px, target_delta=self.cfg.target_delta,
                         min_days=self.cfg.min_days, max_days=self.cfg.max_days,
                         loader=self.loader, um=self.um, oe=self.oe)
        self._sel_cache[key] = sel
        return sel

    def _premium(self, product: str, symbol: str, ts: pd.Timestamp) -> float | None:
        piv = self.premium_frames.get(product)
        if piv is None or symbol not in piv.columns:
            return None
        ser = piv[symbol]
        known = ser[ser.index <= ts]
        if known.empty:
            return None
        return float(known.iloc[-1])

    def run(self, positions: pd.DataFrame, prices: pd.DataFrame,
            registry, dates: list[pd.Timestamp]) -> pd.Series:
        """返回逐日对冲 PnL (占初始资金比例).

        会计: 对冲账户权益 = 现金 (已付权利金负数) + 持仓市值.
        open 时 现金 -= qty×premium×mult; 逐日 mark 市值; 到期结算现金 += 内在价值.
        日 PnL = 当日权益 − 昨日权益.

        优化: 预提取各品种为 numpy 数组, 避免逐日 .loc 逐格查找.
        """
        cfg = self.cfg
        idx = pd.DatetimeIndex(dates)
        n_days = len(dates)

        # ── 预提取为 dict[str, np.array] (按 dates 对齐) ──
        pos_arr = {p: positions[p].reindex(idx).fillna(0.0).to_numpy()
                    for p in positions.columns}
        px_arr = {p: prices[p].reindex(idx).to_numpy() if p in prices.columns
                  else np.full(n_days, np.nan)
                  for p in positions.columns}
        iv_arr = {p: self.iv_pct[p].reindex(idx).to_numpy()
                   if self.iv_pct is not None and p in self.iv_pct.columns
                   else np.full(n_days, np.nan)
                   for p in positions.columns}
        basis_arr = {p: self.basis_pct[p].reindex(idx).to_numpy()
                      if self.basis_pct is not None and p in self.basis_pct.columns
                      else np.full(n_days, np.nan)
                      for p in positions.columns}

        products = list(positions.columns)
        holdings: dict[str, dict] = {}
        cash = 0.0
        equity = np.zeros(n_days)

        for i in range(n_days):
            ts = dates[i]
            if i % 500 == 0:
                print(f"  day {i}/{n_days} selects={len(self._sel_cache)} "
                      f"holdings={len(holdings)}", flush=True)
            # ── 开仓 / 展期 ──
            for product in products:
                qty = pos_arr[product][i]
                if qty <= 0:
                    continue
                px = px_arr[product][i]
                if np.isnan(px) or px <= 0:
                    continue
                coef = timing_coefficient(iv_arr[product][i],
                                          basis_arr[product][i], cfg)
                if coef <= 0:
                    continue
                h = holdings.get(product)
                if h is not None:
                    exp = pd.Timestamp(h["expiry"])
                    if (exp - ts).days <= cfg.roll_days:
                        holdings.pop(product)
                        h = None
                if h is None:
                    key = f"{product}|{ts.strftime('%Y-%m')}"
                    sel = self._select(product, ts, px, key)
                    if sel is None:
                        continue
                    mult = registry.get_or_default(product).contract_multiplier
                    abs_delta = abs(sel.delta)
                    if abs_delta < 1e-6:
                        continue
                    n = int(np.floor(qty * cfg.hedge_ratio * coef / abs_delta))
                    if n <= 0:
                        continue
                    holdings[product] = {
                        "symbol": sel.symbol, "qty": n, "strike": sel.strike,
                        "expiry": pd.Timestamp(sel.expiry), "mult": mult,
                        "last_prem": sel.premium,
                    }
                    cash -= n * sel.premium * mult

            # ── 逐日市价结算 ──
            value = 0.0
            for product in list(holdings.keys()):
                h = holdings[product]
                if pd.Timestamp(h["expiry"]) == ts:
                    s = px_arr[product][i]
                    settle = max((h["strike"] - s) * h["mult"], 0.0) * h["qty"] \
                        if not np.isnan(s) else 0.0
                    cash += settle
                    holdings.pop(product)
                else:
                    prem = self._premium(product, h["symbol"], ts)
                    if prem is None:
                        prem = h["last_prem"]
                    value += h["qty"] * prem * h["mult"]
                    h["last_prem"] = prem
            equity[i] = cash + value

        return pd.Series(np.diff(np.concatenate([[0], equity])) / INITIAL_CASH,
                         index=idx)


def main():
    import pickle
    import time
    from dataclasses import replace
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

    cache = Path("/tmp/opencode/cta_hedge_cache.pkl")
    cache.parent.mkdir(parents=True, exist_ok=True)

    # 1. CTA 回测 (缓存)
    if cache.exists():
        with cache.open("rb") as f:
            d = pickle.load(f)
        dates = d["dates"]
        positions = d["positions"]
        price_panel = d["price_panel"]
        base_nav = d["base_nav"]
        metrics = d["metrics"]
        registry = d["registry"]
        print(f"fut_v1 (cached): Sharpe={metrics['Sharpe']:.3f} MaxDD={metrics['MaxDD']:.3f}")
    else:
        print("运行 fut_v1 ...")
        t0 = time.time()
        s = FactorModuleComboStrategy(config=CFG)
        r = s.run()
        dates = [pd.Timestamp(dd) for dd in r.dates]
        idx = pd.DatetimeIndex(dates)
        positions = pd.DataFrame(r.positions, index=idx)
        price_panel = pd.DataFrame(
            {p: s._price_panel[p] for p in positions.columns
             if p in s._price_panel.columns}, index=idx,
        )
        registry = s._registry
        base_nav = r.nav
        metrics = r.metrics
        print(f"fut_v1: Sharpe={metrics['Sharpe']:.3f} MaxDD={metrics['MaxDD']:.3f} "
              f"({time.time()-t0:.0f}s)")
        with cache.open("wb") as f:
            pickle.dump({"dates": dates, "positions": positions,
                         "price_panel": price_panel, "base_nav": base_nav,
                         "metrics": metrics, "registry": registry}, f)

    # 2. IV + 月差分位面板
    print("加载 IV / 月差分位面板 ...")
    iv_pct, basis_pct = _load_iv_basis_panels(dates)

    # 3. 只加载「有期权 + 曾出现净多头」的品种
    dl = CtaDataLoader()
    opt_products = set(dl.list_option_products())
    ever_long = set(positions.columns[positions.where(positions > 0).notna().any()])
    hedge_products = sorted(ever_long & opt_products)
    print(f"对冲品种 ({len(hedge_products)}): {hedge_products[:10]}...")
    print(f"预加载期权 close ({len(hedge_products)} 品种) ...")
    t1 = time.time()
    prem_frames = _build_premium_frames(hedge_products, dl, dates)
    print(f"期权数据加载完成 ({time.time()-t1:.0f}s)")

    # 4. A/B 对照 (共享 select 缓存: none 模式预填, 后续模式复用)
    print("\n=== 对冲 A/B 对照 ===", flush=True)
    print(f"{'mode':5s} {'Sharpe':>7s} {'AnnRet':>7s} {'MaxDD':>7s} "
          f"{'最差单日':>8s} {'对冲成本':>8s}", flush=True)
    shared_cache: dict = {}
    for mode in ["none", "iv", "basis", "both"]:
        try:
            cfg = replace(HEDGE_CFG, timing_mode=mode)
            sim = _PutHedgeSim(cfg, prem_frames, iv_pct, basis_pct,
                               shared_sel_cache=shared_cache)
            t2 = time.time()
            pnl = sim.run(positions, price_panel, registry, dates)
            hedged_nav = base_nav + pnl.cumsum()
            ret = hedged_nav.pct_change().dropna()
            m = compute_metrics(hedged_nav)
            worst = ret.min() * 100 if len(ret) else float("nan")
            print(f"{mode:5s} {m['Sharpe']:7.3f} {m['AnnRet']:7.3f} {m['MaxDD']:7.3f} "
                  f"{worst:7.2f}% {pnl.sum()*100:7.2f}%  "
                  f"[{time.time()-t2:.0f}s cache={len(shared_cache)}]", flush=True)
        except Exception as e:
            import traceback
            print(f"{mode:5s} ERROR: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
