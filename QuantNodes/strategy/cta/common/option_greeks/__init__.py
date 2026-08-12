from .bsm import OptionPricing, ImpliedVolatility, BlackScholesModel, BSPricing, MCPricing, calculate_iv
from .greeks import (
    GreeksCalculator,
    calculate_greeks,
    calculate_all_greeks,
    calculate_delta,
    calculate_gamma,
    calculate_vega,
    calculate_theta,
    calculate_rho,
    delta,
    gamma,
    vega,
    theta,
    rho,
)
from .pricing import (
    binomial_tree,
    trinomial_tree,
    finite_difference,
    american_approx,
    PricingResult,
)
from .greeks_advanced import (
    calculate_vanna,
    calculate_volga,
    calculate_charm,
    calculate_vera,
    calculate_speed,
    calculate_zomma,
    calculate_color,
    calculate_veta,
    calculate_all_advanced_greeks,
    AdvancedGreeksResult,
)
from .risk import (
    HedgeAnalyzer,
    StressTester,
    MarginCalculator,
    calculate_margin,
    HedgeResult,
    StressResult,
)
from .arbitrage import ArbitrageAnalyzer, detect_arbitrage, ArbitrageOpportunity
from .smile import IVSmileAnalyzer, analyze_iv_smile, SmileAnalysis
from .skew import SkewAnalyzer, analyze_skew, SkewMetrics
from .surface import IVSurfaceAnalyzer, build_iv_surface, IVSurface
from .analysis import (
    PortfolioAnalyzer,
    analyze_portfolio,
    PortfolioGreeks,
    analyze_iron_condor,
    analyze_jade_lizard,
    analyze_ratio_spread,
    analyze_calendar_spread,
    analyze_straddle,
    analyze_strangle,
    calculate_breakeven,
    plot_payoff,
)

__version__ = '0.3.0'

__all__ = [
    # Pricing & IV
    'OptionPricing',
    'ImpliedVolatility',
    'BlackScholesModel',
    'BSPricing',
    'MCPricing',
    'calculate_iv',
    # Advanced Pricing
    'binomial_tree',
    'trinomial_tree',
    'finite_difference',
    'american_approx',
    'PricingResult',
    # Greeks
    'GreeksCalculator',
    'calculate_greeks',
    'calculate_all_greeks',
    'calculate_delta',
    'calculate_gamma',
    'calculate_vega',
    'calculate_theta',
    'calculate_rho',
    'delta',
    'gamma',
    'vega',
    'theta',
    'rho',
    # Advanced Greeks
    'calculate_vanna',
    'calculate_volga',
    'calculate_charm',
    'calculate_vera',
    'calculate_speed',
    'calculate_zomma',
    'calculate_color',
    'calculate_veta',
    'calculate_all_advanced_greeks',
    'AdvancedGreeksResult',
    # Risk Management
    'HedgeAnalyzer',
    'StressTester',
    'MarginCalculator',
    'calculate_margin',
    'HedgeResult',
    'StressResult',
    # Analysis
    'ArbitrageAnalyzer',
    'detect_arbitrage',
    'ArbitrageOpportunity',
    'IVSmileAnalyzer',
    'analyze_iv_smile',
    'SmileAnalysis',
    'SkewAnalyzer',
    'analyze_skew',
    'SkewMetrics',
    'IVSurfaceAnalyzer',
    'build_iv_surface',
    'IVSurface',
    'PortfolioAnalyzer',
    'analyze_portfolio',
    'PortfolioGreeks',
    # Strategies
    'analyze_iron_condor',
    'analyze_jade_lizard',
    'analyze_ratio_spread',
    'analyze_calendar_spread',
    'analyze_straddle',
    'analyze_strangle',
    'calculate_breakeven',
    'plot_payoff',
]
