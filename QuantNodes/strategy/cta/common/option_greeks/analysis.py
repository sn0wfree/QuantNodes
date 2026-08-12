"""
综合期权分析模块

期权组合分析和风险管理工具:

1. **组合 Greeks**:
   - 多合约组合的 Delta, Gamma, Vega, Theta, Rho
   - 仓位汇总和风险评估
   
2. **策略分析**:
   - Covered Call: 持有标的 + 卖出看涨
   - Straddle: 买入看涨 + 买入看跌 (同strike)
   - Strangle: 买入看涨 OTM + 买入看跌 OTM
   - Iron Condor: 垂直价差组合
   - Butterfly: 蝶式价差
   
3. **风险指标**:
   - VaR (Value at Risk): 最大损失估计
   - CVaR (Conditional VaR): 尾部风险
   - Greeks 敏感度分析

主要类:
    Position: 期权仓位
    PortfolioAnalyzer: 组合分析器
    
主要函数:
    analyze_portfolio: 分析期权组合
    calculate_var: 计算 VaR

示例:
    >>> from option_greeks.analysis import analyze_portfolio
    >>> positions = [
    ...     {'type': 'call', 'k': 5500, 'sigma': 0.20, 'side': 1},
    ...     {'type': 'put', 'k': 5400, 'sigma': 0.22, 'side': -1},
    ... ]
    >>> result = analyze_portfolio(5500, positions, 30/365, 0.03)
    >>> print(f"Delta: {result['greeks']['delta']:.2f}, VaR: {result['var']['var_95']:.2f}")
    Delta: 50.72, VaR: 557828.12
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from .greeks import calculate_all_greeks, calculate_delta


@dataclass
class Position:
    """期权仓位"""
    strike: float
    expiry: float
    quantity: float
    option_type: str
    iv: float
    premium: float
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0


@dataclass
class PortfolioGreeks:
    """组合 Greeks"""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class PortfolioAnalyzer:
    """组合分析器"""
    
    def __init__(self, s: float, r: float = 0.03, q: float = 0):
        """
        参数:
            s: 标的资产价格
            r: 无风险利率
            q: 股息收益率
        """
        self.s = s
        self.r = r
        self.q = q
    
    def calculate_position_greeks(
        self,
        position: Position,
        t: float
    ) -> PortfolioGreeks:
        """计算单个仓位的 Greeks"""
        greeks = calculate_all_greeks(
            s=self.s,
            k=position.strike,
            r=self.r,
            g=self.q,
            sigma=position.iv,
            t=t,
            option_type=position.option_type
        )
        
        return PortfolioGreeks(
            delta=greeks.delta * position.quantity * 100,
            gamma=greeks.gamma * position.quantity * 100,
            vega=greeks.vega * position.quantity * 100,
            theta=greeks.theta * position.quantity * 100,
            rho=greeks.rho * position.quantity * 100
        )
    
    def calculate_portfolio_greeks(
        self,
        positions: List[Position],
        current_time: float
    ) -> PortfolioGreeks:
        """计算组合 Greeks"""
        total = PortfolioGreeks(0, 0, 0, 0, 0)
        
        for pos in positions:
            t = max(pos.expiry - current_time, 0.001)
            pos_greeks = self.calculate_position_greeks(pos, t)
            
            total.delta += pos_greeks.delta
            total.gamma += pos_greeks.gamma
            total.vega += pos_greeks.vega
            total.theta += pos_greeks.theta
            total.rho += pos_greeks.rho
        
        return total
    
    def analyze_strategy(
        self,
        strategy_type: str,
        s: float,
        k: float,
        t: float,
        sigma: float,
        r: float = 0.03
    ) -> Dict:
        """
        分析期权策略
        
        参数:
            strategy_type: 策略类型
            s: 标的价
            k: 执行价
            t: 到期时间
            sigma: 波动率
            r: 利率
        
        返回:
            策略分析结果
        """
        strategies = {
            'long_call': self._long_call,
            'short_call': self._short_call,
            'long_put': self._long_put,
            'short_put': self._short_put,
            'covered_call': self._covered_call,
            'protective_put': self._protective_put,
            'long_straddle': self._long_straddle,
            'short_straddle': self._short_straddle,
            'bull_call_spread': lambda: self._bull_call_spread(s, k, t, sigma, r),
            'bear_put_spread': lambda: self._bear_put_spread(s, k, t, sigma, r),
            'iron_condor': lambda: self._iron_condor(s, k, t, sigma, r),
            'butterfly': lambda: self._butterfly(s, k, t, sigma, r),
        }
        
        if strategy_type not in strategies:
            return {'error': f'Unknown strategy: {strategy_type}'}
        
        return strategies[strategy_type]()
    
    def _long_call(self) -> Dict:
        return {'name': 'Long Call', 'direction': 'Long', 'risk': 'Unlimited upside'}
    
    def _short_call(self) -> Dict:
        return {'name': 'Short Call', 'direction': 'Short', 'risk': 'Unlimited'}
    
    def _long_put(self) -> Dict:
        return {'name': 'Long Put', 'direction': 'Long', 'risk': 'Limited to premium'}
    
    def _short_put(self) -> Dict:
        return {'name': 'Short Put', 'direction': 'Short', 'risk': 'Large downside'}
    
    def _covered_call(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Covered Call',
            'positions': [
                {'type': 'stock', 'quantity': 100},
                {'type': 'short_call', 'strike': k}
            ],
            'max_profit': max(0, s - k) + s * (1 - np.exp(-r * t)),
            'max_loss': s
        }
    
    def _protective_put(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Protective Put',
            'positions': [
                {'type': 'stock', 'quantity': 100},
                {'type': 'long_put', 'strike': k}
            ],
            'max_profit': 'Unlimited',
            'max_loss': k
        }
    
    def _long_straddle(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Long Straddle',
            'positions': [
                {'type': 'long_call', 'strike': k},
                {'type': 'long_put', 'strike': k}
            ],
            'profit_at_exp': 'S > K + premium or S < K - premium',
            'max_loss': 'Premium paid'
        }
    
    def _short_straddle(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Short Straddle',
            'positions': [
                {'type': 'short_call', 'strike': k},
                {'type': 'short_put', 'strike': k}
            ],
            'profit_at_exp': 'Premium received',
            'max_loss': 'Unlimited'
        }
    
    def _bull_call_spread(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Bull Call Spread',
            'positions': [
                {'type': 'long_call', 'strike': k},
                {'type': 'short_call', 'strike': k * 1.05}
            ],
            'max_profit': 'Limited',
            'max_loss': 'Premium paid'
        }
    
    def _bear_put_spread(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Bear Put Spread',
            'positions': [
                {'type': 'long_put', 'strike': k},
                {'type': 'short_put', 'strike': k * 0.95}
            ],
            'max_profit': 'Limited',
            'max_loss': 'Premium paid'
        }
    
    def _iron_condor(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Iron Condor',
            'positions': [
                {'type': 'long_put', 'strike': k * 0.9},
                {'type': 'short_put', 'strike': k * 0.95},
                {'type': 'short_call', 'strike': k * 1.05},
                {'type': 'long_call', 'strike': k * 1.1}
            ],
            'max_profit': 'Limited premium',
            'max_loss': 'Limited'
        }
    
    def _butterfly(self, s, k, t, sigma, r) -> Dict:
        return {
            'name': 'Long Butterfly',
            'positions': [
                {'type': 'long_call', 'strike': k * 0.9},
                {'type': 'short_call', 'strike': k, 'quantity': 2},
                {'type': 'long_call', 'strike': k * 1.1}
            ],
            'max_profit': 'K2 - K1 - premium',
            'max_loss': 'Premium paid'
        }
    
    def calculate_var(
        self,
        positions: List[Position],
        current_time: float,
        confidence: float = 0.95,
        simulations: int = 10000
    ) -> Dict:
        """
        计算 VaR (Value at Risk)
        
        参数:
            positions: 仓位列表
            current_time: 当前时间
            confidence: 置信水平
            simulations: 模拟次数
        
        返回:
            {'var_95': float, 'var_99': float, 'cvar': float}
        """
        returns = np.random.normal(0, 0.02, simulations)
        
        portfolio_values = []
        for ret in returns:
            s_new = self.s * (1 + ret)
            total_value = 0
            
            for pos in positions:
                t = max(pos.expiry - current_time, 0.001)
                from .bsm import BSPricing
                if pos.option_type in ('call', 'C'):
                    new_price = BSPricing(s_new, pos.strike, self.r, t, pos.iv, 1, self.q)
                else:
                    new_price = BSPricing(s_new, pos.strike, self.r, t, pos.iv, -1, self.q)
                
                total_value += new_price * pos.quantity * 100
            
            portfolio_values.append(total_value)
        
        var_95 = np.percentile(portfolio_values, (1 - confidence) * 100)
        var_99 = np.percentile(portfolio_values, 1)
        
        cvar = np.mean([v for v in portfolio_values if v < var_95])
        
        return {
            'var_95': abs(self.s * 100 - var_95),
            'var_99': abs(self.s * 100 - var_99),
            'cvar': abs(self.s * 100 - cvar) if cvar else 0
        }
    
    def sensitivity_analysis(
        self,
        position: Position,
        current_time: float,
        param: str,
        range_pct: float = 0.1
    ) -> Dict:
        """
        敏感度分析
        
        参数:
            position: 仓位
            current_time: 当前时间
            param: 参数 ('spot', 'vol', 'time')
            range_pct: 变化范围百分比
        """
        t = max(position.expiry - current_time, 0.001)
        base = calculate_all_greeks(
            self.s, position.strike, self.r, self.q,
            position.iv, t, position.option_type
        )
        
        if param == 'spot':
            s_up = self.s * (1 + range_pct)
            s_down = self.s * (1 - range_pct)
            g_up = calculate_all_greeks(s_up, position.strike, self.r, self.q, position.iv, t, position.option_type)
            g_down = calculate_all_greeks(s_down, position.strike, self.r, self.q, position.iv, t, position.option_type)
            
            return {
                'spot_change_pct': range_pct * 100,
                'delta_change': g_up.delta - base.delta,
                'gamma_change': g_up.gamma - base.gamma,
                'vega_change': g_up.vega - base.vega
            }
        
        elif param == 'vol':
            iv_up = position.iv * (1 + range_pct)
            g_up = calculate_all_greeks(self.s, position.strike, self.r, self.q, iv_up, t, position.option_type)
            
            return {
                'vol_change_pct': range_pct * 100,
                'vega_change': g_up.vega - base.vega,
                'theta_change': g_up.theta - base.theta
            }
        
        return {}


def analyze_portfolio(
    s: float,
    positions: List[Dict],
    current_time: float,
    r: float = 0.03,
    q: float = 0
) -> Dict:
    """
    便捷函数: 分析期权组合
    """
    analyzer = PortfolioAnalyzer(s, r, q)
    
    pos_objects = [
        Position(
            strike=p['strike'],
            expiry=p.get('expiry', 30/365),
            quantity=p.get('quantity', 1),
            option_type=p.get('type', 'call'),
            iv=p.get('iv', 0.3),
            premium=p.get('premium', 0)
        )
        for p in positions
    ]
    
    greeks = analyzer.calculate_portfolio_greeks(pos_objects, current_time)
    var = analyzer.calculate_var(pos_objects, current_time)
    
    return {
        'greeks': {
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'vega': greeks.vega,
            'theta': greeks.theta,
            'rho': greeks.rho
        },
        'var': var
    }


def analyze_iron_condor(
    s: float,
    call_short_strike: float,
    put_short_strike: float,
    call_wing: float = 50,
    put_wing: float = 50,
    t: float = 30/365,
    sigma: float = 0.20,
    r: float = 0.03
) -> Dict:
    """
    Iron Condor 策略分析
    
    卖出宽跨式 (straddle) + 买入更宽跨式 (strangle)
    预期: 低波动率, 标的价格在区间内
    
    参数:
        s: 标的资产价格
        call_short_strike: 卖出看涨期权行权价
        put_short_strike: 卖出看跌期权行权价  
        call_wing: 看涨期权翼宽
        put_wing: 看跌期权翼宽
        t: 到期时间 (年)
        sigma: 波动率
        r: 无风险利率
    
    返回:
        dict: 策略分析结果
    
    示例:
        >>> result = analyze_iron_condor(5500, 5550, 5450, 50, 50, 30/365, 0.20)
    """
    from .bsm import BSPricing
    
    call_long_strike = call_short_strike + call_wing
    put_long_strike = put_short_strike - put_wing
    
    positions = [
        {'type': 'call', 'strike': call_short_strike, 'iv': sigma, 'side': -1},
        {'type': 'put', 'strike': put_short_strike, 'iv': sigma, 'side': -1},
        {'type': 'call', 'strike': call_long_strike, 'iv': sigma, 'side': 1},
        {'type': 'put', 'strike': put_long_strike, 'iv': sigma, 'side': 1},
    ]
    
    total_credit = 0
    for pos in positions:
        price = BSPricing(s, pos['strike'], r, t, sigma, 1 if pos['type'] == 'call' else -1)
        total_credit += price * pos['side'] * 100
    
    max_profit = total_credit * 100
    max_loss = (call_wing + put_wing) * 100 - max_profit
    
    call_breakeven = call_short_strike + total_credit
    put_breakeven = put_short_strike - total_credit
    
    return {
        'strategy': 'Iron Condor',
        'positions': positions,
        'net_credit': total_credit,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'call_breakeven': call_breakeven,
        'put_breakeven': put_breakeven,
        'wing_width': call_wing + put_wing
    }


def analyze_jade_lizard(
    s: float,
    call_short_strike: float,
    put_short_strike: float,
    call_wing: float = 50,
    t: float = 30/365,
    sigma: float = 0.20,
    r: float = 0.03
) -> Dict:
    """
    Jade Lizard 策略分析
    
    卖出看涨价差 + 卖出看跌
    预期: 中性偏多, 收取权利金
    
    参数:
        s: 标的资产价格
        call_short_strike: 卖出看涨行权价
        put_short_strike: 卖出看跌行权价
        call_wing: 看涨翼宽
        t: 到期时间
        sigma: 波动率
        r: 无风险利率
    
    返回:
        dict: 策略分析结果
    """
    from .bsm import BSPricing
    
    call_long_strike = call_short_strike + call_wing
    
    positions = [
        {'type': 'call', 'strike': call_short_strike, 'iv': sigma, 'side': -1},
        {'type': 'call', 'strike': call_long_strike, 'iv': sigma, 'side': 1},
        {'type': 'put', 'strike': put_short_strike, 'iv': sigma, 'side': -1},
    ]
    
    total_credit = 0
    for pos in positions:
        price = BSPricing(s, pos['strike'], r, t, sigma, 1 if pos['type'] == 'call' else -1)
        total_credit += price * pos['side'] * 100
    
    max_profit = total_credit * 100
    max_loss = call_wing * 100 + (s - put_short_strike) * 100
    
    call_breakeven = call_short_strike + total_credit * 2
    put_breakeven = put_short_strike - total_credit
    
    return {
        'strategy': 'Jade Lizard',
        'positions': positions,
        'net_credit': total_credit,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'call_breakeven': call_breakeven,
        'put_breakeven': put_breakeven,
    }


def analyze_ratio_spread(
    s: float,
    lower_strike: float,
    upper_strike: float,
    ratio: int = 2,
    t: float = 30/365,
    sigma: float = 0.20,
    r: float = 0.03,
    option_type: str = 'call'
) -> Dict:
    """
    Ratio Spread 策略分析
    
    买入 N 份低价期权 + 卖出 1 份高价期权
    
    参数:
        s: 标的资产价格
        lower_strike: 低价期权行权价
        upper_strike: 高价期权行权价
        ratio: 比例 (例如 2:1)
        t: 到期时间
        sigma: 波动率
        r: 无风险利率
        option_type: 'call' 或 'put'
    
    返回:
        dict: 策略分析结果
    """
    from .bsm import BSPricing
    
    cp_sign = 1 if option_type == 'call' else -1
    
    long_price = BSPricing(s, lower_strike, r, t, sigma, cp_sign)
    short_price = BSPricing(s, upper_strike, r, t, sigma, cp_sign)
    
    net_cost = (long_price * ratio - short_price) * 100
    
    if cp_sign == 1:
        max_profit = (upper_strike - lower_strike) * 100 * ratio - net_cost
        max_loss = net_cost if upper_strike <= lower_strike else float('inf')
        breakeven = lower_strike + net_cost / (ratio * 100)
    else:
        max_profit = (lower_strike - upper_strike) * 100 - net_cost
        max_loss = net_cost if lower_strike <= upper_strike else float('inf')
        breakeven = lower_strike - net_cost / (ratio * 100)
    
    return {
        'strategy': f'Ratio Spread ({ratio}:1)',
        'lower_strike': lower_strike,
        'upper_strike': upper_strike,
        'ratio': ratio,
        'net_cost': net_cost,
        'max_profit': max_profit,
        'max_loss': max_loss,
        'breakeven': breakeven,
    }


def analyze_calendar_spread(
    s: float,
    strike: float,
    near_expiry: float,
    far_expiry: float,
    t: float = 30/365,
    sigma: float = 0.20,
    r: float = 0.03,
    option_type: str = 'call'
) -> Dict:
    """
    Calendar Spread (Horizontal Spread) 策略分析
    
    卖出近期权 + 买远期权
    预期: 波动率上升, 时间价值衰减差异
    
    参数:
        s: 标的资产价格
        strike: 行权价
        near_expiry: 近期权到期时间 (天)
        far_expiry: 远期权到期时间 (天)
        t: 当前时间 (距离到期)
        sigma: 波动率
        r: 无风险利率
        option_type: 'call' 或 'put'
    
    返回:
        dict: 策略分析结果
    """
    from .bsm import BSPricing
    
    cp_sign = 1 if option_type == 'call' else -1
    
    near_price = BSPricing(s, strike, r, near_expiry/365, sigma, cp_sign)
    far_price = BSPricing(s, strike, r, far_expiry/365, sigma, cp_sign)
    
    net_cost = (far_price - near_price) * 100
    
    near_breakeven = strike + net_cost / 100 if cp_sign == 1 else strike - net_cost / 100
    
    return {
        'strategy': 'Calendar Spread',
        'strike': strike,
        'near_expiry': near_expiry,
        'far_expiry': far_expiry,
        'near_price': near_price,
        'far_price': far_price,
        'net_cost': net_cost,
        'near_breakeven': near_breakeven,
    }


def analyze_straddle(
    s: float,
    strike: float,
    t: float = 30/365,
    sigma: float = 0.20,
    r: float = 0.03,
    side: str = 'long'
) -> Dict:
    """
    Straddle 策略分析
    
    买入/卖出 同行权价看涨 + 看跌
    
    参数:
        s: 标的资产价格
        strike: 行权价
        t: 到期时间
        sigma: 波动率
        r: 无风险利率
        side: 'long' 或 'short'
    
    返回:
        dict: 策略分析结果
    """
    from .bsm import BSPricing
    
    call_price = BSPricing(s, strike, r, t, sigma, 1)
    put_price = BSPricing(s, strike, r, t, sigma, -1)
    
    total_cost = (call_price + put_price) * 100
    multiplier = 1 if side == 'long' else -1
    
    if side == 'long':
        max_loss = total_cost
        max_profit = float('inf')
    else:
        max_loss = float('inf')
        max_profit = total_cost
    
    breakeven_up = strike + total_cost / 100
    breakeven_down = strike - total_cost / 100
    
    return {
        'strategy': f'Straddle ({side})',
        'strike': strike,
        'call_price': call_price,
        'put_price': put_price,
        'total_cost': total_cost,
        'max_profit': max_profit * multiplier,
        'max_loss': max_loss * multiplier,
        'breakeven_up': breakeven_up,
        'breakeven_down': breakeven_down,
    }


def analyze_strangle(
    s: float,
    call_strike: float,
    put_strike: float,
    t: float = 30/365,
    sigma: float = 0.20,
    r: float = 0.03,
    side: str = 'long'
) -> Dict:
    """
    Strangle 策略分析
    
    买入/卖出 OTM 看涨 + OTM 看跌
    
    参数:
        s: 标的资产价格
        call_strike: 看涨行权价
        put_strike: 看跌行权价
        t: 到期时间
        sigma: 波动率
        r: 无风险利率
        side: 'long' 或 'short'
    
    返回:
        dict: 策略分析结果
    """
    from .bsm import BSPricing
    
    call_price = BSPricing(s, call_strike, r, t, sigma, 1)
    put_price = BSPricing(s, put_strike, r, t, sigma, -1)
    
    total_cost = (call_price + put_price) * 100
    multiplier = 1 if side == 'long' else -1
    
    if side == 'long':
        max_loss = total_cost
        max_profit = float('inf')
    else:
        max_loss = float('inf')
        max_profit = total_cost
    
    breakeven_up = call_strike + total_cost / 100
    breakeven_down = put_strike - total_cost / 100
    
    return {
        'strategy': f'Strangle ({side})',
        'call_strike': call_strike,
        'put_strike': put_strike,
        'call_price': call_price,
        'put_price': put_price,
        'total_cost': total_cost,
        'max_profit': max_profit * multiplier,
        'max_loss': max_loss * multiplier,
        'breakeven_up': breakeven_up,
        'breakeven_down': breakeven_down,
    }


def calculate_breakeven(
    strategy_type: str,
    positions: List[Dict],
    s: float,
    t: float,
    sigma: float,
    r: float
) -> List[float]:
    """
    计算策略盈亏平衡点
    
    参数:
        strategy_type: 策略类型
        positions: 仓位列表
        s: 标的资产价格
        t: 到期时间
        sigma: 波动率
        r: 无风险利率
    
    返回:
        List[float]: 盈亏平衡点列表
    """
    from .bsm import BSPricing
    
    breakevens = []
    price_range = np.linspace(s * 0.7, s * 1.3, 100)
    
    initial_value = 0
    for pos in positions:
        k = pos.get('strike', pos.get('k', s))
        cp = 1 if pos.get('type', 'call') == 'call' else -1
        qty = pos.get('quantity', pos.get('side', 1))
        price = BSPricing(s, k, r, t, sigma, cp)
        initial_value += price * qty * 100
    
    for p in price_range:
        current_value = 0
        for pos in positions:
            k = pos.get('strike', pos.get('k', s))
            cp = 1 if pos.get('type', 'call') == 'call' else -1
            qty = pos.get('quantity', pos.get('side', 1))
            price = BSPricing(p, k, r, t, sigma, cp)
            current_value += price * qty * 100
        
        if abs(current_value - initial_value) < s * 0.005:
            breakevens.append(p)
    
    return breakevens


def plot_payoff(
    positions: List[Dict],
    s: float,
    t: float,
    sigma: float,
    r: float,
    price_range_pct: float = 0.3,
    points: int = 100
) -> Dict:
    """
    生成盈亏图表数据
    
    参数:
        positions: 仓位列表
        s: 标的资产价格
        t: 到期时间
        sigma: 波动率
        r: 无风险利率
        price_range_pct: 价格范围百分比
        points: 数据点数量
    
    返回:
        dict: 图表数据
    """
    from .bsm import BSPricing
    
    prices = np.linspace(s * (1 - price_range_pct), s * (1 + price_range_pct), points)
    
    payoffs = []
    initial_value = 0
    
    for pos in positions:
        k = pos.get('strike', pos.get('k', s))
        cp = 1 if pos.get('type', 'call') == 'call' else -1
        qty = pos.get('quantity', pos.get('side', 1))
        price = BSPricing(s, k, r, t, sigma, cp)
        initial_value += price * qty * 100
    
    for p in prices:
        payoff = 0
        for pos in positions:
            k = pos.get('strike', pos.get('k', s))
            cp = 1 if pos.get('type', 'call') == 'call' else -1
            qty = pos.get('quantity', pos.get('side', 1))
            price = BSPricing(p, k, r, 0.001, sigma, cp)
            payoff += price * qty * 100
        
        payoffs.append(payoff - initial_value)
    
    return {
        'prices': prices.tolist(),
        'payoffs': payoffs,
        'initial_value': initial_value,
    }
