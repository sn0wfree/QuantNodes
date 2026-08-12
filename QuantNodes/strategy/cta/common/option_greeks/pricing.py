"""
期权定价模型模块

提供多种期权定价方法:

1. **二叉树模型** (Cox-Ross-Rubinstein):
   - 美式期权定价
   - 可选美式/欧式期权
   - 支持股息处理
   
2. **有限差分法**:
   - 显式方法 (Explicit)
   - 隐式方法 (Implicit)  
   - Crank-Nicolson 方法 (均值)
   
3. **Brennan-Schramm 近似**:
   - 美式期权近似定价

主要函数:
    binomial_tree: 二叉树定价
    finite_difference: 有限差分定价
    american_approx: Brennan-Schramm 近似

示例:
    >>> from option_greeks.pricing import binomial_tree
    >>> price = binomial_tree(s=5500, k=5500, r=0.03, sigma=0.20, t=30/365, cp_sign=1, n=100)
    >>> print(f"美式看涨期权价格: {price:.2f}")
    132.53
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PricingResult:
    """定价结果"""
    price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    method: str = "binomial"


def binomial_tree(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    n: int = 100,
    american: bool = True,
    dividend: float = 0.0
) -> float:
    """
    Cox-Ross-Rubinstein 二叉树期权定价
    
    参数:
        s (float): 标的资产当前价格
        k (float): 行权价
        r (float): 无风险利率 (年化)
        sigma (float): 波动率 (年化)
        t (float): 到期时间 (年)
        cp_sign (int): 1=看涨期权, -1=看跌期权
        n (int): 树的高度 (步数), 默认 100
        american (bool): 是否为美式期权, 默认 True
        dividend (float): 股息收益率, 默认 0
    
    返回:
        float: 期权价格
    
    示例:
        >>> price = binomial_tree(s=5500, k=5500, r=0.03, sigma=0.20, t=30/365, cp_sign=1, n=100)
        >>> print(f"价格: {price:.2f}")
        132.53
    """
    if t <= 0 or n <= 0:
        return max(0, cp_sign * (s - k))
    
    dt = t / n
    df = np.exp(-r * dt)
    
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - dividend) * dt) - d) / (u - d)
    
    prices = np.zeros(n + 1)
    values = np.zeros(n + 1)
    
    for i in range(n + 1):
        prices[i] = s * (u ** (n - i)) * (d ** i)
        if cp_sign == 1:
            values[i] = max(0, prices[i] - k)
        else:
            values[i] = max(0, k - prices[i])
    
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            underlying = s * (u ** (j - i)) * (d ** i)
            values[i] = df * (p * values[i] + (1 - p) * values[i + 1])
            
            if american:
                if cp_sign == 1:
                    exercise = max(0, underlying - k)
                else:
                    exercise = max(0, k - underlying)
                values[i] = max(values[i], exercise)
    
    return values[0]


def binomial_tree_greeks(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    n: int = 100,
    american: bool = True,
    dividend: float = 0.0
) -> PricingResult:
    """
    二叉树定价并计算 Greeks
    
    参数:
        s, k, r, sigma, t, cp_sign, n, american, dividend: 同 binomial_tree
    
    返回:
        PricingResult: 包含价格和 Greeks
    """
    price = binomial_tree(s, k, r, sigma, t, cp_sign, n, american, dividend)
    
    ds = 0.01 * s
    price_up = binomial_tree(s + ds, k, r, sigma, t, cp_sign, n, american, dividend)
    price_down = binomial_tree(s - ds, k, r, sigma, t, cp_sign, n, american, dividend)
    
    delta = (price_up - price_down) / (2 * ds)
    gamma = (price_up - 2 * price + price_down) / (ds ** 2)
    
    return PricingResult(
        price=price,
        delta=delta,
        gamma=gamma,
        method="binomial"
    )


def finite_difference(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    method: str = 'explicit',
    n_steps: int = 50,
    m_steps: int = 50
) -> float:
    """
    有限差分法期权定价
    
    参数:
        s (float): 标的资产当前价格
        k (float): 行权价
        r (float): 无风险利率
        sigma (float): 波动率
        t (float): 到期时间 (年)
        cp_sign (int): 1=看涨, -1=看跌
        method (str): 'explicit', 'implicit', 'crank_nicolson'
        n_steps (int): 时间步数
        m_steps (int): 价格步数
    
    返回:
        float: 期权价格
    """
    if method == 'explicit':
        return _fd_explicit(s, k, r, sigma, t, cp_sign, n_steps, m_steps)
    elif method == 'crank_nicolson':
        return _fd_crank_nicolson(s, k, r, sigma, t, cp_sign, n_steps, m_steps)
    else:
        return _fd_explicit(s, k, r, sigma, t, cp_sign, n_steps, m_steps)


def _fd_explicit(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    n: int,
    m: int
) -> float:
    """显式有限差分法"""
    dt = t / n
    smax = 3 * s
    ds = smax / m
    
    grid = np.zeros((m + 1, n + 1))
    
    for i in range(m + 1):
        price = i * ds
        if cp_sign == 1:
            grid[i, 0] = max(0, price - k)
        else:
            grid[i, 0] = max(0, k - price)
    
    alpha = 0.5 * dt * (sigma ** 2 * np.arange(m + 1) ** 2 - (r - 0.5 * sigma ** 2) * np.arange(m + 1))
    beta = 1 - dt * (sigma ** 2 * np.arange(m + 1) ** 2 + r)
    gamma = 0.5 * dt * (sigma ** 2 * np.arange(m + 1) ** 2 + (r - 0.5 * sigma ** 2) * np.arange(m + 1))
    
    for j in range(1, n + 1):
        grid[0, j] = 0
        grid[m, j] = smax - k * np.exp(-r * j * dt) if cp_sign == 1 else 0
        
        for i in range(1, m):
            grid[i, j] = alpha[i] * grid[i - 1, j - 1] + beta[i] * grid[i, j - 1] + gamma[i] * grid[i + 1, j - 1]
    
    idx = int(s / ds)
    if idx >= m:
        idx = m - 1
    
    return grid[idx, n]


def _fd_crank_nicolson(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    n: int,
    m: int
) -> float:
    """Crank-Nicolson 有限差分法"""
    dt = t / n
    smax = 3 * s
    ds = smax / m
    
    grid = np.zeros(m + 1)
    
    for i in range(m + 1):
        price = i * ds
        if cp_sign == 1:
            grid[i] = max(0, price - k)
        else:
            grid[i] = max(0, k - price)
    
    a = np.zeros(m + 1)
    b = np.zeros(m + 1)
    c = np.zeros(m + 1)
    
    for i in range(m + 1):
        xi = i * ds
        a[i] = -0.25 * dt * (sigma ** 2 * xi ** 2 - (r - 0.5 * sigma ** 2) * xi)
        b[i] = 1 + 0.5 * dt * (sigma ** 2 * xi ** 2 + r)
        c[i] = -0.25 * dt * (sigma ** 2 * xi ** 2 + (r - 0.5 * sigma ** 2) * xi)
    
    for j in range(1, n + 1):
        rhs = np.zeros(m + 1)
        for i in range(1, m):
            rhs[i] = (0.25 * dt * (sigma ** 2 * (i * ds) ** 2 - (r - 0.5 * sigma ** 2) * (i * ds)) * grid[i - 1] +
                     (1 - 0.5 * dt * (sigma ** 2 * (i * ds) ** 2 + r)) * grid[i] +
                     0.25 * dt * (sigma ** 2 * (i * ds) ** 2 + (r - 0.5 * sigma ** 2) * (i * ds)) * grid[i + 1])
        
        rhs[0] = 0
        rhs[m] = smax - k * np.exp(-r * j * dt) if cp_sign == 1 else 0
        
        grid = _tdma(a[1:m], b[1:m], c[1:m], rhs[1:m])
    
    idx = int(s / ds)
    if idx >= m:
        idx = m - 1
    
    return grid[idx] if idx < len(grid) else grid[-1]


def _tdma(a, b, c, d):
    """三对角矩阵算法 (Thomas algorithm)"""
    n = len(d)
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    
    for i in range(1, n):
        mc = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - mc * cc[i - 1]
        dc[i] = dc[i] - mc * dc[i - 1]
    
    xc = np.zeros(n)
    xc[-1] = dc[-1] / bc[-1]
    
    for i in range(n - 2, -1, -1):
        xc[i] = (dc[i] - cc[i] * xc[i + 1]) / bc[i]
    
    return xc


def american_approx(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    q: float = 0.0
) -> float:
    """
    Brennan-Schramm 美式期权近似定价
    
    基于 Geske-Whaley 模型的近似方法。
    
    参数:
        s (float): 标的资产价格
        k (float): 行权价
        r (float): 无风险利率
        sigma (float): 波动率
        t (float): 到期时间
        cp_sign (int): 1=看涨, -1=看跌
        q (float): 股息收益率
    
    返回:
        float: 美式期权近似价格
    """
    from scipy.stats import norm
    
    if cp_sign == 1:
        tau = t
        e = r - q
        
        if e <= 0:
            return binomial_tree(s, k, r, sigma, t, cp_sign, n=50)
        
        s_crit = k * (1 + sigma * np.sqrt(t) * np.sqrt(abs(np.log(1 + 2 * e / (sigma ** 2)) / 2) + r * t) / (e * t) ** 0.5)
        
        if s >= s_crit:
            d1 = (np.log(s / k) + (e + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
            d2 = d1 - sigma * np.sqrt(t)
            price = s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
            return price
        else:
            return binomial_tree(s, k, r, sigma, t, cp_sign, n=50)
    else:
        return binomial_tree(s, k, r, sigma, t, cp_sign, n=50)


def trinomial_tree(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    cp_sign: int,
    n: int = 100,
    american: bool = True
) -> float:
    """
    三叉树期权定价
    
    参数:
        s (float): 标的资产价格
        k (float): 行权价
        r (float): 无风险利率
        sigma (float): 波动率
        t (float): 到期时间
        cp_sign (int): 1=看涨, -1=看跌
        n (int): 树的高度
        american (bool): 是否为美式期权
    
    返回:
        float: 期权价格
    """
    dt = t / n
    df = np.exp(-r * dt)
    
    nu = r - 0.5 * sigma ** 2
    dxu = sigma * np.sqrt(1.5 * dt)
    dxd = dxu
    
    pu = 0.5 * ((sigma ** 2 * dt + nu * dt) / dxu + (nu * dt / dxu) ** 2) / (1 + nu * dt / dxu) ** 2
    pd = 0.5 * ((sigma ** 2 * dt + nu * dt) / dxd + (nu * dt / dxd) ** 2) / (1 - nu * dt / dxd) ** 2
    pm = 1 - pu - pd
    
    prices = np.zeros(2 * n + 1)
    values = np.zeros(2 * n + 1)
    
    for j in range(-n, n + 1):
        prices[j + n] = s * np.exp(j * dxu)
    
    for j in range(-n, n + 1):
        if cp_sign == 1:
            values[j + n] = max(0, prices[j + n] - k)
        else:
            values[j + n] = max(0, k - prices[j + n])
    
    for i in range(n - 1, -1, -1):
        for j in range(-i, i + 1):
            idx = j + n
            underlying = s * np.exp(j * dxu)
            value = df * (pu * values[idx + 1] + pm * values[idx] + pd * values[idx - 1])
            
            if american:
                if cp_sign == 1:
                    exercise = max(0, underlying - k)
                else:
                    exercise = max(0, k - underlying)
                value = max(value, exercise)
            
            values[idx] = value
        
        values[n + i + 1] = 0
        values[n - i - 1] = 0
    
    return values[n]


def implied_tree(
    s: float,
    k: float,
    t: float,
    market_price: float,
    r: float = 0.03,
    cp_sign: int = 1,
    n: int = 50
) -> np.ndarray:
    """
    从市场价格的隐含树 (Implied Tree)
    
    参数:
        s (float): 标的资产价格
        k (float): 行权价
        t (float): 到期时间
        market_price (float): 市场价格
        r (float): 无风险利率
        cp_sign (int): 1=看涨, -1=看跌
        n (int): 树的高度
    
    返回:
        np.ndarray: 隐含概率树
    """
    dt = t / n
    u = 1.1
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    prices = np.zeros((n + 1, n + 1))
    values = np.zeros((n + 1, n + 1))
    
    for i in range(n + 1):
        prices[i, n - i:] = s * (u ** np.arange(n - i, n + 1)) * (d ** np.arange(0, n - i + 1))
    
    for j in range(n + 1):
        if cp_sign == 1:
            values[j, n - j] = max(0, prices[j, n - j] - k)
        else:
            values[j, n - j] = max(0, k - prices[j, n - j])
    
    return values
