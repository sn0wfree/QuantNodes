"""动态波动率策略 (Dynamic Volatility Strategy)

双引擎设计：
- Theta引擎：卖put spread/铁鹰/宽跨，收theta
- Gamma引擎：买跨式，赚gamma

核心创新：
- IV Crush预警退出
- 波面曲面套利选股
- 冷静期机制
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..strategies.factor_modules import FactorModuleComboStrategy
from .dynamic_vol.regime_detector import RegimeDetector, Regime, RegimeConfig, CooldownManager
from .dynamic_vol.product_scorer import ProductScorer, ScorerConfig
from .dynamic_vol.surface_selector import SurfaceSelector, SurfaceSelectorConfig
from .dynamic_vol.strategy_router import StrategyRouter, StrategyConfig
from .dynamic_vol.theta_engine import ThetaEngine
from .dynamic_vol.gamma_engine import GammaEngine
from .dynamic_vol.risk_manager import RiskManager, RiskConfig
from .dynamic_vol.greeks_hedger import GreeksHedger, HedgerConfig

logger = logging.getLogger(__name__)


@dataclass
class DynamicVolConfig:
    """DVS配置"""
    # 品种
    products: List[str] = field(default_factory=lambda: [
        "RB", "I", "CU", "AU", "AG", "M", "SR", "TA", "RU"
    ])

    # 账户
    initial_cash: float = 5_000_000

    # 冷静期
    cooldown_days: int = 10
    min_position_pct: float = 0.20

    # 合约选择
    min_dte: int = 15
    max_dte: int = 40

    # 曲面选股
    outlier_threshold: float = 1.5


class DynamicVolStrategy(FactorModuleComboStrategy):
    """动态波动率策略

    双引擎设计：
    - Theta引擎：卖put spread/铁鹰/宽跨，收theta
    - Gamma引擎：买跨式，赚gamma

    核心创新：
    - IV Crush预警退出
    - 波面曲面套利选股
    - 冷静期机制
    """

    def __init__(
        self,
        dv_config: Optional[DynamicVolConfig] = None,
        option_data: Optional[Dict[str, pd.DataFrame]] = None,
        option_contracts: Optional[dict] = None,
        greeks_cache=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._dv_cfg = dv_config or DynamicVolConfig()
        self._option_contracts = option_contracts or {}
        self._greeks_cache = greeks_cache

        # 初始化各层
        self.regime_detector = RegimeDetector()
        self.product_scorer = ProductScorer()
        self.surface_selector = SurfaceSelector(
            SurfaceSelectorConfig(outlier_threshold=self._dv_cfg.outlier_threshold)
        )
        self.strategy_router = StrategyRouter(
            StrategyConfig(min_dte=self._dv_cfg.min_dte, max_dte=self._dv_cfg.max_dte)
        )
        self.theta_engine = ThetaEngine()
        self.gamma_engine = GammaEngine()
        self.risk_manager = RiskManager()
        self.greeks_hedger = GreeksHedger()
        self.cooldown_manager = CooldownManager(
            cooldown_days=self._dv_cfg.cooldown_days,
            min_position_pct=self._dv_cfg.min_position_pct,
        )

        # 预计算目标
        self._precomputed_gv: Dict[pd.Timestamp, Dict[str, Any]] = {}

        # 统计
        self.stats = {
            "entries": 0, "gamma_entries": 0,
            "exits": 0, "stops": 0,
            "iv_crush_exits": 0, "skips": 0,
        }

    def precompute_all_targets(
        self,
        option_data: Dict[str, pd.DataFrame],
        iv_panel: pd.DataFrame,
        vrp_panel: Optional[pd.DataFrame] = None,
        vix_series: Optional[pd.Series] = None,
    ) -> Dict[pd.Timestamp, Dict[str, Any]]:
        """预计算所有日期的目标持仓

        Parameters
        ----------
        option_data : {product: DataFrame} 各品种期权数据
        iv_panel : IV面板 (date x product)
        vrp_panel : VRP面板
        vix_series : VIX序列

        Returns
        -------
        {date: {symbol: target_qty}}
        """
        if iv_panel is None or iv_panel.empty:
            return {}

        dates = iv_panel.index
        all_targets = {}

        for date in dates:
            daily_targets = {}

            # 获取当日可用品种
            available_products = [
                p for p in self._dv_cfg.products
                if p in iv_panel.columns and date in iv_panel.index
            ]
            if not available_products:
                continue

            # 1. 识别各品种的市场状态
            regimes = {}
            for product in available_products:
                regime_result = self.regime_detector.compute_regime(
                    product, date, iv_panel, vix_series, option_data.get(product)
                )
                regimes[product] = regime_result

                # 更新冷静期
                if regime_result.iv_crush_detected:
                    self.cooldown_manager.on_iv_crush(date)

            # 2. 选择品种
            equity = self._dv_cfg.initial_cash
            # 简化：用第一个品种的状态作为全局状态
            # 实际应该按品种分别评分
            primary_regime = list(regimes.values())[0].regime if regimes else Regime.CALM

            selected_products = self.product_scorer.score_products(
                date=date,
                regime=primary_regime,
                equity=equity,
                products=available_products,
                iv_panel=iv_panel,
                vrp_panel=vrp_panel,
            )

            # 3. 为每个品种路由策略
            cooldown_mult = self.cooldown_manager.get_position_multiplier(date)

            for product in selected_products:
                regime_result = regimes[product]
                regime = regime_result.regime

                # 获取标的价格
                spot = self._get_spot_price(product, date, option_data)
                if spot is None or spot <= 0:
                    continue

                # 路由到策略
                route_result = self.strategy_router.route(
                    regime=regime,
                    product=product,
                    trade_date=str(date.date()),
                    underlying_price=spot,
                    option_data=option_data.get(product),
                )

                if route_result is None:
                    self.stats["skips"] += 1
                    continue

                # 执行策略
                engine = route_result["engine"]
                strategy = route_result["strategy"]
                position = route_result["position"]

                if engine == "theta":
                    targets = self.theta_engine.execute(
                        strategy=strategy,
                        product=product,
                        date=date,
                        position_data=position,
                        equity=equity,
                        spot=spot,
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
                        strategy=strategy,
                        product=product,
                        date=date,
                        position_data=position,
                        equity=equity,
                        spot=spot,
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

        self._precomputed_gv = all_targets
        return all_targets

    def target_positions(self, date, data):
        """回测引擎调用：返回目标持仓"""
        return self._precomputed_gv.get(date, {})

    def _get_spot_price(
        self, product: str, date: pd.Timestamp,
        option_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """获取标的价格"""
        df = option_data.get(product)
        if df is None or df.empty:
            return None
        if "underlying_price" in df.columns:
            day_data = df[df["dt"] == date]
            if not day_data.empty:
                return float(day_data["underlying_price"].iloc[0])
        return None

    def _get_multiplier(self, product: str) -> float:
        """获取合约乘数"""
        multipliers = {
            "RB": 10, "I": 100, "CU": 5, "AU": 1000, "AG": 15,
            "M": 10, "SR": 10, "TA": 5, "RU": 10, "PP": 5,
            "AL": 5, "NI": 1, "ZN": 5, "PB": 5, "SN": 1,
            "AU": 1000, "AG": 15, "SC": 1000, "FU": 10, "LU": 10,
            "RU": 10, "NR": 10, "BC": 5, "SP": 10, "CJ": 5,
            "AP": 10, "CF": 5, "RM": 10, "OI": 10, "RS": 10,
            "WH": 20, "PM": 50, "RI": 20, "JR": 20, "LR": 20,
            "CY": 5, "PF": 5, "PK": 5, "SH": 30, "SA": 20,
            "FG": 20, "ZC": 100, "JM": 60, "JD": 10, "LH": 16,
            "BB": 500, "FB": 500, "RR": 10, "LH": 16,
            "T": 10000, "TF": 10000, "TS": 20000, "TL": 10000,
        }
        return multipliers.get(product, 10)
