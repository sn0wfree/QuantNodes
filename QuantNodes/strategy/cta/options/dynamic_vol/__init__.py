"""动态波动率策略 (Dynamic Volatility Strategy)

双引擎设计：
- Theta引擎：卖put spread/铁鹰/宽跨，收theta
- Gamma引擎：买跨式，赚gamma

核心创新：
- IV Crush预警退出
- 波面曲面套利选股
- 冷静期机制
"""

from .regime_detector import RegimeDetector, Regime, CooldownManager
from .product_scorer import ProductScorer
from .surface_selector import SurfaceSelector
from .strategy_router import StrategyRouter
from .theta_engine import ThetaEngine
from .gamma_engine import GammaEngine
from .risk_manager import RiskManager
from .greeks_hedger import GreeksHedger

__all__ = [
    "RegimeDetector", "Regime", "CooldownManager",
    "ProductScorer", "SurfaceSelector", "StrategyRouter",
    "ThetaEngine", "GammaEngine", "RiskManager", "GreeksHedger",
]
