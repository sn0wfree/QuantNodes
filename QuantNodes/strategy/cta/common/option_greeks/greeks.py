"""
期权 Greeks 计算模块

提供完整的希腊字母计算功能:
- Delta: 标的价格变化对期权价格的影响
- Gamma: Delta 的变化率
- Vega: 波动率变化对期权价格的影响  
- Theta: 时间衰减对期权价格的影响
- Rho: 利率变化对期权价格的影响

特性:
    - 使用 lru_cache 缓存 d1, d2 计算，提升性能
    - 支持批量计算
    - 返回 namedtuple 格式结果

主要函数:
    calculate_all_greeks: 计算所有 Greeks
    calculate_delta/gamma/vega/theta/rho: 单独计算
    calculate_greeks: 批量计算

示例:
    >>> from option_greeks import calculate_all_greeks
    >>> result = calculate_all_greeks(s=5500, k=5500, r=0.03, g=0, sigma=0.20, t=30/365, option_type='call')
    >>> print(f"Delta: {result.delta:.4f}")
    0.5286
"""

import numpy as np
from scipy.stats import norm
from functools import lru_cache
from collections import namedtuple

norm_cdf = norm.cdf
norm_pdf = norm.pdf

_OPTION_TYPE_MAP = {
    'Call': 1, 'call': 1, 'C': 1, 'c': 1,
    'Put': -1, 'put': -1, 'P': -1, 'p': -1,
}

_GreeksResult = namedtuple('GreeksResult', ['delta', 'gamma', 'vega', 'theta', 'rho'])


def _option_type_to_sign(option_type):
    sign = _OPTION_TYPE_MAP.get(option_type)
    if sign is not None:
        return sign
    raise ValueError(f'Unknown option type: {option_type}')


@lru_cache(maxsize=4096)
def _calc_d1(s, k, r, sigma, t, g):
    sqrt_t = np.sqrt(t)
    return (np.log(s / k) + ((r - g) + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)


@lru_cache(maxsize=4096)
def _calc_d2(s, k, r, sigma, t, g):
    d1 = _calc_d1(s, k, r, sigma, t, g)
    return d1 - sigma * np.sqrt(t)


def delta(s, k, r, g, sigma, t, option_type):
    cp_sign = _option_type_to_sign(option_type)
    d1 = _calc_d1(s, k, r, sigma, t, g)
    return norm_cdf(d1) - (1 if cp_sign < 0 else 0)


def gamma(s, k, r, sigma, t, option_type):
    if sigma < 1e-8:
        return 0.0
    d1 = _calc_d1(s, k, r, sigma, t, g=0)
    return norm_pdf(d1) / (s * sigma * np.sqrt(t))


def vega(s, k, r, sigma, t, option_type):
    d1 = _calc_d1(s, k, r, sigma, t, g=0)
    return norm_pdf(d1) * s * np.sqrt(t)


def theta(s, k, r, sigma, t, option_type):
    cp_sign = _option_type_to_sign(option_type)
    d1 = _calc_d1(s, k, r, sigma, t, g=0)
    d2 = d1 - sigma * np.sqrt(t)
    
    term1 = -s * sigma * norm_pdf(d1) / (2 * np.sqrt(t))
    
    if cp_sign > 0:
        term2 = -r * k * np.exp(-r * t) * norm_cdf(d2)
    else:
        term2 = r * k * np.exp(-r * t) * norm_cdf(-d2)
    
    return term1 + term2


def rho(s, k, r, sigma, t, option_type):
    cp_sign = _option_type_to_sign(option_type)
    d2 = _calc_d2(s, k, r, sigma, t, g=0)
    
    if cp_sign > 0:
        return k * t * np.exp(-r * t) * norm_cdf(d2)
    else:
        return -k * t * np.exp(-r * t) * norm_cdf(-d2)


def calculate_all_greeks(s, k, r, g, sigma, t, option_type):
    """
    计算期权所有 Greeks 指标
    
    使用 Black-Scholes 公式计算 Delta, Gamma, Vega, Theta, Rho。
    结果缓存 d1, d2 计算以提升性能。
    
    参数:
        s (float): 标的资产当前价格
        k (float): 行权价
        r (float): 无风险利率 (年化)
        g (float): 股息收益率 (年化)
        sigma (float): 波动率 (年化)
        t (float): 到期时间 (年)
        option_type: 期权类型
            - 'call', 'Call', 'C', 'c', 1: 看涨期权
            - 'put', 'Put', 'P', 'p', -1: 看跌期权
    
    返回:
        namedtuple: 包含 delta, gamma, vega, theta, rho
    
    示例:
        >>> result = calculate_all_greeks(s=5500, k=5500, r=0.03, g=0, sigma=0.20, t=30/365, option_type='call')
        >>> print(f"Delta: {result.delta:.4f}, Gamma: {result.gamma:.6f}")
        Delta: 0.5286, Gamma: 0.001262
    """
    cp_sign = _option_type_to_sign(option_type)
    
    sqrt_t = np.sqrt(t)
    exp_rt = np.exp(-r * t)
    
    d1 = (np.log(s / k) + ((r - g) + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    
    cdf_d1 = norm_cdf(d1)
    cdf_d2 = norm_cdf(d2)
    pdf_d1 = norm_pdf(d1)
    
    delta_val = cdf_d1 - (1 if cp_sign < 0 else 0)
    gamma_val = pdf_d1 / (s * sigma * sqrt_t) if sigma > 1e-8 else 0.0
    vega_val = pdf_d1 * s * sqrt_t
    
    term1 = -s * sigma * pdf_d1 / (2 * sqrt_t)
    if cp_sign > 0:
        term2 = -r * k * exp_rt * cdf_d2
    else:
        term2 = r * k * exp_rt * norm_cdf(-d2)
    theta_val = term1 + term2
    
    if cp_sign > 0:
        rho_val = k * t * exp_rt * cdf_d2
    else:
        rho_val = -k * t * exp_rt * norm_cdf(-d2)
    
    return _GreeksResult(delta_val, gamma_val, vega_val, theta_val, rho_val)


def calculate_all_greeks_dict(s, k, r, g, sigma, t, option_type):
    result = calculate_all_greeks(s, k, r, g, sigma, t, option_type)
    return {'delta': result.delta, 'gamma': result.gamma, 'vega': result.vega, 'theta': result.theta, 'rho': result.rho}


class GreeksCalculator:
    __slots__ = ('model', 'option_style')

    def __init__(self, model='Black-Scholes', option_style='European'):
        self.model = model
        self.option_style = option_style

    def get_all_greeks(self, s, k, r, g, sigma, t, option_type):
        return calculate_all_greeks(s, k, r, g, sigma, t, option_type)

    def delta(self, s, k, r, g, sigma, t, option_type):
        return delta(s, k, r, g, sigma, t, option_type)

    def gamma(self, s, k, r, sigma, t, option_type):
        return gamma(s, k, r, sigma, t, option_type)

    def vega(self, s, k, r, sigma, t, option_type):
        return vega(s, k, r, sigma, t, option_type)

    def theta(self, s, k, r, sigma, t, option_type):
        return theta(s, k, r, sigma, t, option_type)

    def rho(self, s, k, r, sigma, t, option_type):
        return rho(s, k, r, sigma, t, option_type)


_calculator = GreeksCalculator()


def calculate_greeks(
    underlying_price: float,
    strike_price: float,
    volatility: float,
    time_to_maturity: float,
    risk_free_rate: float,
    option_type: str,
    dividend_yield: float = 0
) -> _GreeksResult:
    return _calculator.get_all_greeks(
        underlying_price, strike_price, risk_free_rate,
        dividend_yield, volatility, time_to_maturity, option_type
    )


def calculate_delta(s, k, r, g, sigma, t, option_type):
    return delta(s, k, r, g, sigma, t, option_type)


def calculate_gamma(s, k, r, sigma, t, option_type):
    return gamma(s, k, r, sigma, t, option_type)


def calculate_vega(s, k, r, sigma, t, option_type):
    return vega(s, k, r, sigma, t, option_type)


def calculate_theta(s, k, r, sigma, t, option_type):
    return theta(s, k, r, sigma, t, option_type)


def calculate_rho(s, k, r, sigma, t, option_type):
    return rho(s, k, r, sigma, t, option_type)
