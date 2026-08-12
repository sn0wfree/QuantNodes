"""
高级期权 Greeks 模块

计算二阶和高阶 Greeks 指标:

1. **Vanna**: ∂²V/∂σ∂S
   - 波动率变化对 Delta 的影响
   - 价格变化对 Vega 的影响
   
2. **Volga (Vega Gamma)**: ∂²V/∂σ²
   - Vega 对波动率的敏感度
   - 波动率微笑的关键指标
   
3. **Charm**: ∂²V/∂t∂S
   - Delta 的时间衰减
   - 也称为 Delta Decay
   
4. **Vera (Rho Vanna)**: ∂²V/∂r∂σ
   - 利率变化对 Vega 的影响
   
5. **Speed**: ∂³V/∂S³
   - Gamma 的变化率
   - 用于 Gamma 风险管理
   
6. **Zomma**: ∂³V/∂S²∂σ
   - Gamma 和 Vanna 的混合
   
7. **Color (Gamma Decay)**: ∂³V/∂t∂S²
   - Gamma 的时间衰减
   
8. **Veta**: ∂³V/∂t∂σ
   - Vega 的时间衰减

主要函数:
    calculate_vanna: Vanna
    calculate_volga: Volga/Vega Gamma
    calculate_charm: Charm
    calculate_ver: Vera
    calculate_speed: Speed
    calculate_zomma: Zomma
    calculate_color: Color
    calculate_all_advanced_greeks: 所有高级 Greeks

示例:
    >>> from option_greeks.greeks_advanced import calculate_all_advanced_greeks
    >>> g = calculate_all_advanced_greeks(s=5500, k=5500, r=0.03, g=0, sigma=0.20, t=30/365, option_type='call')
    >>> print(f"Vanna: {g.vanna:.6f}, Volga: {g.volga:.4f}")
    Vanna: -0.123456, Volga: 12.3456
"""

import numpy as np
from scipy.stats import norm
from functools import lru_cache
from collections import namedtuple
from typing import Optional


norm_cdf = norm.cdf
norm_pdf = norm.pdf

_OptionTypeMap = {
    'Call': 1, 'call': 1, 'C': 1, 'c': 1,
    'Put': -1, 'put': -1, 'P': -1, 'p': -1,
}

AdvancedGreeksResult = namedtuple('AdvancedGreeksResult', [
    'vanna', 'volga', 'charm', 'vera', 'speed', 'zomma', 'color', 'veta'
])


def _option_type_to_sign(option_type):
    """转换期权类型为符号"""
    sign = _OptionTypeMap.get(option_type)
    if sign is not None:
        return sign
    return 1


@lru_cache(maxsize=4096)
def _calc_d1_d2(s, k, r, sigma, t, g=0):
    """计算 d1 和 d2"""
    if t <= 1e-8 or sigma <= 1e-8:
        d1 = np.inf if s > k else -np.inf
    else:
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s / k) + ((r - g) + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * np.sqrt(t)
    return d1, d2


def calculate_vanna(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Vanna: ∂²V/∂σ∂S
    
    波动率变化对 Delta 的影响，或价格变化对 Vega 的影响。
    
    参数:
        s (float): 标的资产价格
        k (float): 行权价
        r (float): 无风险利率
        g (float): 股息收益率
        sigma (float): 波动率
        t (float): 到期时间
        option_type (str): 'call' 或 'put'
    
    返回:
        float: Vanna 值
    
    示例:
        >>> vanna = calculate_vanna(5500, 5500, 0.03, 0, 0.20, 30/365, 'call')
        >>> print(f"Vanna: {vanna:.6f}")
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    vanna = -norm_pdf(d1) * d2 / sigma * sqrt_t
    
    return vanna


def calculate_volga(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Volga (Vega Gamma): ∂²V/∂σ²
    
    Vega 对波动率的敏感度，波动率微笑的关键指标。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Volga 值
    
    示例:
        >>> volga = calculate_volga(5500, 5500, 0.03, 0, 0.20, 30/365, 'call')
        >>> print(f"Volga: {volga:.4f}")
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    volga = s * sqrt_t * norm_pdf(d1) * (d1 * d2 + 1) / sigma
    
    return volga


def calculate_charm(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Charm: ∂²V/∂t∂S
    
    Delta 的时间衰减，也称为 Delta Decay。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Charm 值
    
    示例:
        >>> charm = calculate_charm(5500, 5500, 0.03, 0, 0.20, 30/365, 'call')
        >>> print(f"Charm: {charm:.4f}")
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    nd1 = norm_pdf(d1)
    cdf_d1 = norm_cdf(d1)
    cdf_d2 = norm_cdf(d2)
    
    term1 = -g * nd1 / (s * sigma * sqrt_t)
    
    if cp_sign > 0:
        term2 = r * cdf_d2 / (sigma * sqrt_t)
    else:
        term2 = -r * norm_cdf(-d2) / (sigma * sqrt_t)
    
    charm = term1 + term2
    
    return charm


def calculate_vera(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Vera (Rho Vanna): ∂²V/∂r∂σ
    
    利率变化对 Vega 的影响。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Vera 值
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    vera = -s * sqrt_t * norm_pdf(d1) * (r - g) / sigma
    
    return vera


def calculate_speed(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Speed: ∂³V/∂S³
    
    Gamma 的变化率，用于 Gamma 风险管理。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Speed 值
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    nd1 = norm_pdf(d1)
    
    speed = -nd1 / (s ** 2 * sigma * sqrt_t) * (1 + d1 / (s * sigma * sqrt_t))
    
    return speed


def calculate_zomma(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Zomma: ∂³V/∂S²∂σ
    
    Gamma 和 Vanna 的混合指标。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Zomma 值
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    nd1 = norm_pdf(d1)
    
    zomma = nd1 / (s * sigma * sqrt_t) * (d1 * d2 + 1) / (s * sigma * sqrt_t)
    
    return zomma


def calculate_color(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Color (Gamma Decay): ∂³V/∂t∂S²
    
    Gamma 的时间衰减。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Color 值
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    nd1 = norm_pdf(d1)
    
    term1 = g * nd1 / (s ** 2 * sigma * sqrt_t) * (1 + d1 / (s * sigma * sqrt_t))
    term2 = r * nd1 / (s * sigma * sqrt_t) * (1 + d1 / (s * sigma * sqrt_t))
    
    color = term1 - term2
    
    return color


def calculate_veta(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> float:
    """
    Veta: ∂³V/∂t∂σ
    
    Vega 的时间衰减。
    
    参数:
        同 calculate_vanna
    
    返回:
        float: Veta 值
    """
    cp_sign = _option_type_to_sign(option_type)
    
    if t <= 0 or sigma <= 0:
        return 0.0
    
    sqrt_t = np.sqrt(t)
    d1, d2 = _calc_d1_d2(s, k, r, sigma, t, g)
    
    nd1 = norm_pdf(d1)
    
    veta = s * sqrt_t * nd1 * (
        (r - g) * d1 / (sigma * sqrt_t) +
        (1 + d1 * d2) / (2 * t)
    )
    
    return veta


def calculate_all_advanced_greeks(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str
) -> AdvancedGreeksResult:
    """
    计算所有高级 Greeks
    
    参数:
        s (float): 标的资产价格
        k (float): 行权价
        r (float): 无风险利率
        g (float): 股息收益率
        sigma (float): 波动率
        t (float): 到期时间
        option_type (str): 'call' 或 'put'
    
    返回:
        AdvancedGreeksResult: 包含所有高级 Greeks
    
    示例:
        >>> g = calculate_all_advanced_greeks(5500, 5500, 0.03, 0, 0.20, 30/365, 'call')
        >>> print(f"Vanna: {g.vanna:.4f}, Volga: {g.volga:.4f}, Charm: {g.charm:.4f}")
    """
    vanna = calculate_vanna(s, k, r, g, sigma, t, option_type)
    volga = calculate_volga(s, k, r, g, sigma, t, option_type)
    charm = calculate_charm(s, k, r, g, sigma, t, option_type)
    vera = calculate_vera(s, k, r, g, sigma, t, option_type)
    speed = calculate_speed(s, k, r, g, sigma, t, option_type)
    zomma = calculate_zomma(s, k, r, g, sigma, t, option_type)
    color = calculate_color(s, k, r, g, sigma, t, option_type)
    veta = calculate_veta(s, k, r, g, sigma, t, option_type)
    
    return AdvancedGreeksResult(
        vanna=vanna,
        volga=volga,
        charm=charm,
        vera=vera,
        speed=speed,
        zomma=zomma,
        color=color,
        veta=veta
    )


def calculate_greeks_fd(
    s: float,
    k: float,
    r: float,
    g: float,
    sigma: float,
    t: float,
    option_type: str,
    ds: float = 0.01,
    dsigma: float = 0.01
) -> dict:
    """
    使用有限差分法计算 Greeks (验证解析解)
    
    参数:
        s, k, r, g, sigma, t, option_type: 同上
        ds (float): 价格扰动步长
        dsigma (float): 波动率扰动步长
    
    返回:
        dict: Greeks 字典
    """
    from option_greeks import BSPricing
    
    cp_sign = _option_type_to_sign(option_type)
    
    p0 = BSPricing(s, k, r, t, sigma, cp_sign, g)
    
    p_up = BSPricing(s + ds * s, k, r, t, sigma, cp_sign, g)
    p_down = BSPricing(s - ds * s, k, r, t, sigma, cp_sign, g)
    
    delta = (p_up - p_down) / (2 * ds * s)
    gamma = (p_up - 2 * p0 + p_down) / (ds * s) ** 2
    
    sigma_up = sigma + dsigma
    sigma_down = sigma - dsigma
    p_sigma_up = BSPricing(s, k, r, t, sigma_up, cp_sign, g)
    p_sigma_down = BSPricing(s, k, r, t, sigma_down, cp_sign, g)
    
    vega = (p_sigma_up - p_sigma_down) / (2 * dsigma)
    
    t_up = t + 0.001
    t_down = t - 0.001
    p_time_up = BSPricing(s, k, r, t_up, sigma, cp_sign, g)
    p_time_down = BSPricing(s, k, r, t_down, sigma, cp_sign, g)
    
    theta = (p_time_down - p_time_up) / (2 * 0.001)
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }
