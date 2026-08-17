#!/usr/bin/env python3
"""DynamicVolStrategy 独立回测

纯期权层回测，不含CTA期货叠加
对比GV独立回测和DVS回测
"""
import warnings
warnings.filterwarnings("ignore")
import sys
import time
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd
from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.backtest.cta_engine import run_cta_backtest, CtaBacktestConfig
from QuantNodes.strategy.cta.backtest.cta_contract import (
    CtaContractRegistry, CtaContract, OptionContractSpec,
)
from QuantNodes.strategy.cta.backtest.cta_risk import PositionLimitRisk
from QuantNodes.strategy.cta.options.dynamic_vol.regime_detector import RegimeDetector, CooldownManager
from QuantNodes.strategy.cta.options.dynamic_vol.product_scorer import ProductScorer
from QuantNodes.strategy.cta.options.dynamic_vol.surface_selector import SurfaceSelector
from QuantNodes.strategy.cta.options.dynamic_vol.strategy_router import StrategyRouter
from QuantNodes.strategy.cta.options.dynamic_vol.theta_engine import ThetaEngine
from QuantNodes.strategy.cta.options.dynamic_vol.gamma_engine import GammaEngine
from QuantNodes.strategy.cta.options.dynamic_vol.risk_manager import RiskManager
from QuantNodes.strategy.cta.options.iv_panel import load_iv_panel
from QuantNodes.strategy.cta.options.hedge_data import load_vrp_panel
from QuantNodes.strategy.cta.options.greeks_cache import GreeksCache
from QuantNodes.strategy.cta.options.selector import select_put, select_call, select_straddle

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU"]
START = "2019-01-01"
END = "2026-07-21"
OPT_START = "2019-01-01"
OPT_END = "2026-07-21"


def full_metrics(nav):
    """计算完整指标"""
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
    win_days = (r > 0).mean()
    best_day = r.max()
    worst_day = r.min()
    return {
        "AnnRet": ann, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": dd,
        "Calmar": calmar, "WinRate": win_days,
        "BestDay": best_day, "WorstDay": worst_day,
        "TotalRet": total, "Years": years,
    }


def load_option_data(products, start, end):
    """加载期权链 + contracts"""
    loader = CtaDataLoader()
    registry = CtaContractRegistry()
    option_data = {}
    option_contracts = {}
    for prod in products:
        try:
            opt_df = loader.load_v_all_options(products=[prod], start=start, end=end)
            if opt_df is not None and not opt_df.empty:
                opt_df["trade_date"] = pd.to_datetime(opt_df["trade_date"])
                opt_df["symbol"] = opt_df["symbol"].str.upper()
                option_data[prod] = opt_df
                spec = registry.get_or_default(prod)
                for sym, row in opt_df.groupby("symbol").first().iterrows():
                    opt_spec = OptionContractSpec(
                        exchange="", product=prod,
                        underlying=str(row.get("underlying", "")),
                        option_type=str(row.get("option_type", "")).upper(),
                        strike=float(row["strike_price"]),
                        expiry_date="",
                        contract_multiplier=spec.contract_multiplier,
                    )
                    option_contracts[sym] = CtaContract(
                        product=prod, symbol=sym, expiry_year=0, expiry_month=0,
                        spec=spec, option_spec=opt_spec,
                    )
        except Exception as e:
            print(f"Warning: failed to load {prod}: {e}")
    return option_data, option_contracts


def build_option_prices(dates, option_data):
    """期权逐日价格序列 {symbol: [price]} + 到期结算价 map"""
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


class OptionOnlyDVS:
    """纯期权层DVS (不含期货仓位)"""

    def __init__(
        self, option_data, iv_panel, vrp_panel, vix_series,
        products, initial_cash=5_000_000,
        cooldown_days=10, min_position_pct=0.20,
        min_dte=15, max_dte=40,
    ):
        self.option_data = option_data
        self.iv_panel = iv_panel
        self.vrp_panel = vrp_panel
        self.vix_series = vix_series
        self.products = products
        self.initial_cash = initial_cash

        # 初始化各层
        self.regime_detector = RegimeDetector()
        self.product_scorer = ProductScorer()
        self.surface_selector = SurfaceSelector()
        self.strategy_router = StrategyRouter()
        self.theta_engine = ThetaEngine()
        self.gamma_engine = GammaEngine()
        self.risk_manager = RiskManager()
        self.cooldown_manager = CooldownManager(
            cooldown_days=cooldown_days,
            min_position_pct=min_position_pct,
        )

        self.min_dte = min_dte
        self.max_dte = max_dte

        # 预计算目标
        self._precomputed: dict = {}
        self.stats = {
            "entries": 0, "gamma_entries": 0,
            "exits": 0, "stops": 0,
            "iv_crush_exits": 0, "skips": 0,
        }

    def precompute_all_targets(self):
        """预计算所有日期的目标持仓"""
        if self.iv_panel is None or self.iv_panel.empty:
            return {}

        dates = self.iv_panel.index
        all_targets = {}
        total = len(dates)

        for i, date in enumerate(dates):
            if (i + 1) % 200 == 0:
                print(f"  Progress: {i+1}/{total} dates ({len(all_targets)} with targets)")

            daily_targets = {}

            # 0. 检查现有持仓的止损/退出
            for product in list(self.theta_engine.positions.keys()):
                pos = self.theta_engine.positions[product]
                if pos.is_active:
                    # 简化：持有超过20天就平仓
                    days_held = (date - pos.entry_date).days
                    if days_held >= 20:
                        close_targets = self.theta_engine.close_position(product)
                        daily_targets.update(close_targets)
                        self.stats["exits"] += 1

            for product in list(self.gamma_engine.positions.keys()):
                pos = self.gamma_engine.positions[product]
                if pos.is_active:
                    # 检查IV crush退出
                    regime_result = self.regime_detector.compute_regime(
                        product, date, self.iv_panel, self.vix_series,
                        self.option_data.get(product),
                    )
                    if self.gamma_engine.check_iv_crush_exit(
                        product, date,
                        regime_result.iv_zscore if hasattr(regime_result, 'iv_zscore') else 0,
                        regime_result.term_structure_slope,
                        regime_result.iv_momentum_5d,
                    ):
                        close_targets = self.gamma_engine.close_position(product)
                        daily_targets.update(close_targets)
                        self.stats["iv_crush_exits"] += 1
                    # 持有超过30天也平仓
                    elif (date - pos.entry_date).days >= 30:
                        close_targets = self.gamma_engine.close_position(product)
                        daily_targets.update(close_targets)
                        self.stats["exits"] += 1

            # 获取当日可用品种
            available = [
                p for p in self.products
                if p in self.iv_panel.columns and date in self.iv_panel.index
            ]
            if not available:
                if daily_targets:
                    all_targets[date] = daily_targets
                continue

            # 1. 识别各品种的市场状态
            regimes = {}
            for product in available:
                regime_result = self.regime_detector.compute_regime(
                    product, date, self.iv_panel, self.vix_series,
                    self.option_data.get(product),
                )
                regimes[product] = regime_result
                if regime_result.iv_crush_detected:
                    self.cooldown_manager.on_iv_crush(date)

            # 2. 选择品种
            primary_regime = list(regimes.values())[0].regime if regimes else Regime.CALM
            selected = self.product_scorer.score_products(
                date=date, regime=primary_regime,
                equity=self.initial_cash, products=available,
                iv_panel=self.iv_panel, vrp_panel=self.vrp_panel,
            )

            # 3. 为每个品种路由策略
            cooldown_mult = self.cooldown_manager.get_position_multiplier(date)

            for product in selected:
                # 跳过已有持仓的品种
                if product in self.theta_engine.positions and \
                   self.theta_engine.positions[product].is_active:
                    continue
                if product in self.gamma_engine.positions and \
                   self.gamma_engine.positions[product].is_active:
                    continue

                regime_result = regimes[product]
                regime = regime_result.regime

                # 获取标的价格
                spot = self._get_spot_price(product, date)
                if spot is None or spot <= 0:
                    continue

                # 路由到策略
                route_result = self.strategy_router.route(
                    regime=regime, product=product,
                    trade_date=str(date.date()),
                    underlying_price=spot,
                    option_data=self.option_data.get(product),
                )

                if route_result is None:
                    self.stats["skips"] += 1
                    continue

                engine = route_result["engine"]
                strategy = route_result["strategy"]
                position = route_result["position"]

                if engine == "theta":
                    targets = self.theta_engine.execute(
                        strategy=strategy, product=product,
                        date=date, position_data=position,
                        equity=self.initial_cash, spot=spot,
                        multiplier=self._get_multiplier(product),
                        cooldown_multiplier=cooldown_mult,
                    )
                    self.stats["entries"] += 1
                elif engine == "gamma":
                    if not self.risk_manager.can_open_gamma_position(
                        len(self.gamma_engine.positions)
                    ):
                        self.stats["skips"] += 1
                        continue
                    targets = self.gamma_engine.execute(
                        strategy=strategy, product=product,
                        date=date, position_data=position,
                        equity=self.initial_cash, spot=spot,
                        multiplier=self._get_multiplier(product),
                        iv_percentile=regime_result.iv_percentile,
                        cooldown_multiplier=cooldown_mult,
                    )
                    self.stats["gamma_entries"] += 1
                else:
                    continue

                if targets:
                    daily_targets.update(targets)

            all_targets[date] = daily_targets

        self._precomputed = all_targets
        return all_targets

    def target_positions(self, date, data):
        """回测引擎调用"""
        return self._precomputed.get(date, {})

    def _get_spot_price(self, product, date):
        """获取标的价格（用期权数据中的underlying_price）"""
        df = self.option_data.get(product)
        if df is None or df.empty:
            return None
        day_data = df[df["trade_date"] == date]
        if day_data.empty:
            return None
        if "underlying_price" in day_data.columns:
            val = day_data["underlying_price"].iloc[0]
            if pd.notna(val) and val > 0:
                return float(val)
        # fallback: 用strike_price的中位数近似
        if "strike_price" in day_data.columns:
            return float(day_data["strike_price"].median())
        return None

    def _get_multiplier(self, product):
        """获取合约乘数"""
        multipliers = {
            "RB": 10, "I": 100, "CU": 5, "AU": 1000, "AG": 15,
            "M": 10, "SR": 10, "TA": 5, "RU": 10, "PP": 5,
            "AL": 5, "NI": 1, "ZN": 5, "PB": 5,
            "SC": 1000, "FU": 10, "LU": 10, "NR": 10, "BC": 5,
        }
        return multipliers.get(product, 10)


def main():
    print("=" * 80)
    print("DynamicVolStrategy (DVS) 独立回测 — 纯期权层")
    print("=" * 80)

    t0 = time.time()

    # 加载数据
    print("\n[1/4] Loading option data...")
    od, oc = load_option_data(PRODUCTS, OPT_START, OPT_END)
    print(f"  Option contracts: {len(oc)} ({time.time()-t0:.0f}s)")

    t1 = time.time()
    print("\n[2/4] Loading IV/VRP panels...")
    iv_panel = load_iv_panel(products=PRODUCTS, start=START, end=END)
    vrp_panel = load_vrp_panel(products=PRODUCTS, start=START, end=END)
    print(f"  IV panel: {iv_panel.shape}, VRP panel: {vrp_panel.shape if vrp_panel is not None else 'None'} ({time.time()-t1:.0f}s)")

    # 加载VIX
    try:
        vix = pd.read_parquet("data/high_freq_macro/macro_vix_daily.parquet")["vix"]
        vix.index = pd.to_datetime(vix.index)
        print(f"  VIX loaded: {len(vix)} rows")
    except Exception as e:
        print(f"  VIX load failed: {e}")
        vix = None

    # 预计算目标
    print("\n[3/4] Precomputing DVS targets...")
    t2 = time.time()

    st = OptionOnlyDVS(
        option_data=od, iv_panel=iv_panel, vrp_panel=vrp_panel,
        vix_series=vix, products=PRODUCTS, initial_cash=5_000_000,
        cooldown_days=10, min_position_pct=0.20,
        min_dte=15, max_dte=40,
    )
    targets = st.precompute_all_targets()
    print(f"  Targets computed in {time.time()-t2:.0f}s")
    print(f"  Dates with targets: {len(targets)}")
    print(f"  Stats: {st.stats}")

    # 运行回测
    print("\n[4/4] Running backtest...")
    t3 = time.time()

    # 构建价格面板
    dates = sorted(targets.keys())
    if not dates:
        print("No targets computed, exiting.")
        return

    op, os_ = build_option_prices(dates, od)

    # 构建价格面板：期货 + 期权
    loader = CtaDataLoader()
    ap = {}
    ap2 = {}  # {symbol: product}

    # 期货价格
    for prod in PRODUCTS:
        try:
            futs = loader.load_futures_daily(prod, start=str(dates[0].date()), end=str(dates[-1].date()))
            if futs is not None and not futs.empty:
                futs["trade_date"] = pd.to_datetime(futs["trade_date"])
                ser = futs.set_index("trade_date")["close"].sort_index()
                ser = ser[~ser.index.duplicated(keep="last")]
                ap[prod] = ser.reindex(dates, method="ffill").fillna(0.0).tolist()
                ap2[prod] = prod
        except Exception:
            pass

    # 期权价格
    ap.update(op)
    for sym, c in oc.items():
        ap2[sym] = c.product

    # 注册合约
    reg = CtaContractRegistry()
    risk = PositionLimitRisk(products={p: p for p in PRODUCTS}, registry=reg)

    # 运行回测
    res = run_cta_backtest(
        dates=dates, prices=ap, products=ap2,
        strategy=st,
        config=CtaBacktestConfig(
            initial_cash=5_000_000, commission_rate=0.0001, margin_call=True,
        ),
        registry=reg, risk=risk,
        option_contracts=oc, option_settle=os_,
    )

    m = full_metrics(res.nav)
    print(f"\nBacktest completed in {time.time()-t3:.0f}s")

    # 输出结果
    print("\n" + "=" * 80)
    print("DVS Results (纯期权层)")
    print("=" * 80)
    print(f"AnnRet:    {m.get('AnnRet', 0):>8.1%}")
    print(f"AnnVol:    {m.get('AnnVol', 0):>8.1%}")
    print(f"Sharpe:    {m.get('Sharpe', 0):>8.2f}")
    print(f"MaxDD:     {m.get('MaxDD', 0):>8.1%}")
    print(f"Calmar:    {m.get('Calmar', 0):>8.2f}")
    print(f"WinRate:   {m.get('WinRate', 0):>8.0%}")
    print(f"BestDay:   {m.get('BestDay', 0)*100:>7.1f}%")
    print(f"WorstDay:  {m.get('WorstDay', 0)*100:>7.1f}%")
    print(f"TotalRet:  {m.get('TotalRet', 0):>8.1%}")
    print(f"Years:     {m.get('Years', 0):>8.1f}")

    print(f"\nStats: {st.stats}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
