"""
Black-Scholes-Merton 期权定价模块

提供期权定价和隐含波动率计算的完整功能:
- Black-Scholes-Merton 定价公式
- Newton-Raphson + Brent 组合算法计算隐含波动率
- Monte Carlo 模拟定价

主要函数:
    BSPricing: Black-Scholes 期权定价
    MCPricing: Monte Carlo 期权定价
    calculate_iv: 计算隐含波动率

示例:
    >>> from option_greeks import BSPricing, calculate_iv
    >>> # 计算期权价格
    >>> price = BSPricing(s=5500, k=5500, r=0.03, t=30/365, sigma=0.20, cp_sign=1)
    >>> print(f"期权价格: {price:.2f}")
    132.53
    
    >>> # 计算隐含波动率
    >>> iv, diff = calculate_iv(S=5500, K=5500, r=0.03, T=30/365, option_price=132.5, cp_sign=1)
    >>> print(f"隐 {iv:.4含波动率:f}")
    0.2000
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from functools import lru_cache

norm_cdf = norm.cdf
norm_pdf = norm.pdf

_CP_SIGN_MAP = {
    'Call': 1, 'call': 1, 'c': 1, 'C': 1,
    'Put': -1, 'put': -1, 'p': -1, 'P': -1,
}


def _detect_cp_sign(cp_sign):
    if isinstance(cp_sign, (int, np.integer)):
        return 1 if cp_sign == 1 else -1
    sign = _CP_SIGN_MAP.get(cp_sign)
    if sign is not None:
        return sign
    raise ValueError(f'cp_sign ({cp_sign}) is Unknown input')


@lru_cache(maxsize=4096)
def _bs_price(s, k, r, t, sigma, cp_sign, g):
    if sigma <= 1e-8:
        d1 = np.inf if s > k else -np.inf
    else:
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s / k) + ((r - g) + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    
    d2 = d1 - sigma * np.sqrt(t)
    
    return cp_sign * s * np.exp(-g * t) * norm_cdf(cp_sign * d1) - cp_sign * k * np.exp(-r * t) * norm_cdf(cp_sign * d2)


def _bs_vega(s, k, r, t, sigma, g=0):
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + ((r - g) + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    return s * sqrt_t * norm_pdf(d1)


def BSPricing(s, k, r, t, sigma, cp_sign, g=0, dividends='continuous', IV_LOWER_BOUND=1e-8):
    """
    Black-Scholes-Merton 期权定价
    
    使用 Black-Scholes-Merton 公式计算欧式期权价格。
    
    参数:
        s (float): 标的资产当前价格
        k (float): 行权价
        r (float): 无风险利率 (年化)
        t (float): 到期时间 (年)
        sigma (float): 波动率 (年化)
        cp_sign: 期权类型
            - 1, 'call', 'Call', 'C': 看涨期权
            - -1, 'put', 'Put', 'P': 看跌期权
        g (float): 股息收益率 (默认 0, 连续股息)
        dividends (str): 股息处理方式
            - 'continuous': 连续股息 (默认)
            - 其他: 视为离散股息调整
        IV_LOWER_BOUND (float): 隐含波动率下界
    
    返回:
        float: 期权理论价格
    
    示例:
        >>> BSPricing(s=5500, k=5500, r=0.03, t=30/365, sigma=0.20, cp_sign=1)
        132.53
    """
    cp_sign = _detect_cp_sign(cp_sign)
    
    if dividends != 'continuous':
        s = s - g * np.exp(-r * t)
        g = 0
    
    return _bs_price(s, k, r, t, sigma, cp_sign, g)


def MCPricing(s, k, r, t, sigma, cp_sign, g, dividends='continuous', iteration=100000):
    cp_sign = _detect_cp_sign(cp_sign)
    
    if dividends != 'continuous':
        s = s - g * np.exp(-r * t)
        g = 0
    
    zt = np.random.normal(0, 1, iteration)
    st = s * np.exp((r - g - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * zt)
    _p = np.maximum((st - k) * cp_sign, 0)

    return np.average(_p) * np.exp(-r * t)


def _iv_newton(S, K, r, T, option_price, cp_sign, g, max_iter=50, tol=1e-6):
    if T <= 0:
        return 0.3, 0.0
    
    intrinsic = max(cp_sign * (S - K), 0)
    extrinsic = max(option_price - intrinsic, 0)
    
    if extrinsic <= 0:
        sigma = 0.001 if S > K else 1.0
        return sigma, _bs_price(S, K, r, T, sigma, cp_sign, g) - option_price
    
    sqrt_T = np.sqrt(T)
    if S <= 0 or K <= 0 or sqrt_T <= 0:
        sigma = 0.3
    else:
        moneyness = np.log(S / K)
        if abs(moneyness) < 0.01:
            sigma = min(max(extrinsic / (S * sqrt_T), 0.1), 1.0)
        else:
            term = abs(moneyness) / sqrt_T
            sigma = min(max(term * 0.5 + extrinsic / (S * sqrt_T) * 0.3, 0.05), 2.0)
    
    prev_sigma = sigma
    for i in range(max_iter):
        price = _bs_price(S, K, r, T, sigma, cp_sign, g)
        diff = price - option_price
        
        if abs(diff) < tol:
            return sigma, diff
        
        vega = _bs_vega(S, K, r, T, sigma, g)
        if abs(vega) < 1e-12:
            break
        
        new_sigma = sigma - diff / vega
        if new_sigma <= 0.001:
            new_sigma = 0.001
        elif new_sigma > 5.0:
            new_sigma = 5.0
        
        if abs(new_sigma - prev_sigma) < 1e-10:
            break
            
        prev_sigma = sigma
        sigma = new_sigma
    
    return sigma, _bs_price(S, K, r, T, sigma, cp_sign, g) - option_price


def _iv_brent(S, K, r, T, option_price, cp_sign, g, IV_LOWER_BOUND=1e-8):
    def loss(sigma):
        return _bs_price(S, K, r, T, sigma, cp_sign, g) - option_price

    a, b = 0.01, 0.9
    fa, fb = loss(a), loss(b)
    
    max_iter = 2000
    iter_count = 0
    while fa * fb > 0 and iter_count < max_iter:
        if abs(fa) < abs(fb):
            a = max(a / 2, IV_LOWER_BOUND)
        else:
            b = min(b * 2, 1.0)
        fa, fb = loss(a), loss(b)
        iter_count += 1

    if fa * fb < 0:
        iv = brentq(loss, a, b)
        return iv, loss(iv)
    
    if abs(fb) < abs(fa):
        return b, fb
    return IV_LOWER_BOUND, loss(IV_LOWER_BOUND)


class ImpliedVolatility:
    __slots__ = ('pricing_func',)

    def __init__(self, pricing_func=None):
        self.pricing_func = pricing_func or BSPricing

    def calculate(self, S, K, r, T, option_price, cp_sign, g=0, dividends='continuous', IV_LOWER_BOUND=1e-8):
        cp_sign = _detect_cp_sign(cp_sign)
        
        if dividends != 'continuous':
            S = S - g * np.exp(-r * T)
            g = 0
        
        iv, diff = _iv_newton(S, K, r, T, option_price, cp_sign, g)
        
        if abs(diff) > 1e-4:
            iv, diff = _iv_brent(S, K, r, T, option_price, cp_sign, g, IV_LOWER_BOUND)
        
        return iv, diff


class BlackScholesModel:
    __slots__ = ('s', 'k', 'r', 't', 'sigma', 'cp_sign', 'g', 'dividends', '_d1', '_d2')

    def __init__(self, s, k, r, t, sigma, cp_sign, g=0, dividends='continuous'):
        if dividends != 'continuous':
            s = s - g * np.exp(-r * t)
            g = 0
        self.s = s
        self.k = k
        self.r = r
        self.t = t
        self.sigma = sigma
        self.cp_sign = _detect_cp_sign(cp_sign)
        self.g = g
        self.dividends = dividends
        self._d1 = None
        self._d2 = None

    @property
    def d1(self):
        if self._d1 is None:
            if self.sigma * np.sqrt(self.t) > 1e-8:
                self._d1 = (np.log(self.s / self.k) + ((self.r - self.g) + 0.5 * self.sigma ** 2) * self.t) / (self.sigma * np.sqrt(self.t))
            else:
                self._d1 = np.inf if self.s > self.k else -np.inf
        return self._d1

    @property
    def d2(self):
        if self._d2 is None:
            self._d2 = self.d1 - self.sigma * np.sqrt(self.t)
        return self._d2

    @property
    def nd1(self):
        return norm_cdf(self.cp_sign * self.d1)

    @property
    def nd2(self):
        return norm_cdf(self.cp_sign * self.d2)

    @property
    def pdf_d1(self):
        return norm_pdf(self.d1)

    def price(self):
        return self.cp_sign * self.s * np.exp(-self.g * self.t) * self.nd1 - self.cp_sign * self.k * np.exp(-self.r * self.t) * self.nd2


class OptionPricing:
    @staticmethod
    def BSPricing(s, k, r, t, sigma, cp_sign, g=0, dividends='continuous', IV_LOWER_BOUND=1e-8):
        return BSPricing(s, k, r, t, sigma, cp_sign, g, dividends, IV_LOWER_BOUND)

    @staticmethod
    def MCPricing(s, k, r, t, sigma, cp_sign, g, dividends='continuous', iteration=100000):
        return MCPricing(s, k, r, t, sigma, cp_sign, g, dividends, iteration)


_iv_calculator = ImpliedVolatility()


def calculate_iv(S, K, r, T, option_price, cp_sign, g=0, dividends='continuous'):
    """计算隐含波动率 (Newton + Brent 组合方法)

    参数:
        S: 标的资产价格
        K: 行权价
        r: 无风险利率
        T: 到期时间 (年)
        option_price: 期权市场价格
        cp_sign: 期权类型 ('call'/'put' 或 1/-1)
        g: 股息收益率
        dividends: 股息方式 ('continuous' 或 'discrete')

    返回:
        (iv, diff): 隐含波动率和计算误差

    示例:
        >>> iv, diff = calculate_iv(5500, 5400, 0.03, 30/365, 150, 'call')
        >>> print(f"IV: {iv:.4f}")
    """
    return _iv_calculator.calculate(S, K, r, T, option_price, cp_sign, g, dividends)
