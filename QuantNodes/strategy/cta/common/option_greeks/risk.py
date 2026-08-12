"""
期权风险管理模块

提供全面的风险管理工具:

1. **对冲分析**:
   - Delta 对冲: 动态对冲标的价格变动
   - Gamma 扫描: 分析 Gamma 风险
   - Vega 对冲: 波动率对冲
   
2. **压力测试**:
   - 极端价格情景 (±10%, ±20%, ±30%)
   - 极端波动率情景 (±20%, ±50%)
   - 历史情景回测
   - 自定义情景
   
3. **保证金计算**:
   - 组合保证金需求
   - 单一期权保证金
   - 价差保证金减免
   
4. **情景分析**:
   - 多情景 P&L 模拟
   - 盈亏平衡分析
   - 敏感度矩阵

主要类:
    HedgeAnalyzer: 对冲分析器
    StressTester: 压力测试器
    MarginCalculator: 保证金计算器
    
主要函数:
    calculate_margin: 计算保证金
    run_stress_test: 运行压力测试
    calculate_scenario_pnl: 计算情景 P&L

示例:
    >>> from option_greeks.risk import HedgeAnalyzer, StressTester
    >>> # Delta 对冲分析
    >>> hedge = HedgeAnalyzer()
    >>> result = hedge.delta_hedge(positions, 5500, 0.1)
    >>> # 压力测试
    >>> stress = StressTester()
    >>> results = stress.run_stress_test(positions)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from .greeks import calculate_all_greeks, calculate_delta, calculate_gamma
from .bsm import BSPricing


@dataclass
class Position:
    """期权仓位"""
    strike: float
    expiry: float
    option_type: str
    quantity: float
    iv: float
    premium: float = 0.0


@dataclass
class HedgeResult:
    """对冲结果"""
    delta: float
    gamma: float
    vega: float
    theta: float
    hedge_ratio: float
    rebalance_signals: int


@dataclass
class StressResult:
    """压力测试结果"""
    scenario: str
    price_change: float
    iv_change: float
    pnl: float
    delta: float
    gamma: float


class HedgeAnalyzer:
    """对冲分析器"""
    
    def __init__(self, s: float = 0, r: float = 0.03):
        """
        初始化对冲分析器
        
        参数:
            s: 标的资产价格
            r: 无风险利率
        """
        self.s = s
        self.r = r
    
    def calculate_portfolio_greeks(
        self,
        positions: List[Dict],
        current_price: float,
        current_time: float = 0
    ) -> Dict[str, float]:
        """
        计算组合 Greeks
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
            current_time: 距离到期时间
        
        返回:
            dict: Greeks 字典
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        
        for pos in positions:
            k = pos.get('strike', pos.get('k', 5500))
            iv = pos.get('iv', pos.get('sigma', 0.2))
            qty = pos.get('quantity', pos.get('side', 1))
            opt_type = pos.get('type', 'call')
            t = pos.get('expiry', pos.get('t', 30/365))
            
            g = calculate_all_greeks(
                s=current_price,
                k=k,
                r=self.r,
                g=0,
                sigma=iv,
                t=t,
                option_type=opt_type
            )
            
            total_delta += g.delta * qty * 100
            total_gamma += g.gamma * qty * 100
            total_vega += g.vega * qty * 100
            total_theta += g.theta * qty * 100
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta
        }
    
    def delta_hedge(
        self,
        positions: List[Dict],
        current_price: float,
        rebalance_threshold: float = 0.1,
        current_time: float = 0
    ) -> HedgeResult:
        """
        Delta 对冲分析
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
            rebalance_threshold: 再平衡阈值
            current_time: 当前时间
        
        返回:
            HedgeResult: 对冲结果
        """
        greeks = self.calculate_portfolio_greeks(positions, current_price, current_time)
        
        hedge_ratio = -greeks['delta'] / current_price
        
        price_range = np.linspace(current_price * 0.9, current_price * 1.1, 21)
        rebalance_count = 0
        prev_delta = greeks['delta']
        
        for p in price_range:
            g = self.calculate_portfolio_greeks(positions, p, current_time)
            if abs(g['delta'] - prev_delta) > current_price * rebalance_threshold:
                rebalance_count += 1
                prev_delta = g['delta']
        
        return HedgeResult(
            delta=greeks['delta'],
            gamma=greeks['gamma'],
            vega=greeks['vega'],
            theta=greeks['theta'],
            hedge_ratio=hedge_ratio,
            rebalance_signals=rebalance_count
        )
    
    def gamma_scan(
        self,
        positions: List[Dict],
        current_price: float,
        price_range_pct: float = 0.1,
        current_time: float = 0
    ) -> Dict[str, float]:
        """
        Gamma 风险扫描
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
            price_range_pct: 价格扫描范围百分比
            current_time: 当前时间
        
        返回:
            dict: Gamma 扫描结果
        """
        prices = np.linspace(
            current_price * (1 - price_range_pct),
            current_price * (1 + price_range_pct),
            21
        )
        
        gammas = []
        for p in prices:
            g = self.calculate_portfolio_greeks(positions, p, current_time)
            gammas.append(g['gamma'])
        
        return {
            'gamma_mean': np.mean(gammas),
            'gamma_max': np.max(gammas),
            'gamma_min': np.min(gammas),
            'gamma_std': np.std(gammas)
        }
    
    def vega_hedge(
        self,
        positions: List[Dict],
        current_price: float,
        current_iv: float,
        target_iv: float,
        current_time: float = 0
    ) -> Dict[str, float]:
        """
        Vega 对冲分析
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
            current_iv: 当前隐含波动率
            target_iv: 目标隐含波动率
            current_time: 当前时间
        
        返回:
            dict: Vega 对冲结果
        """
        greeks = self.calculate_portfolio_greeks(positions, current_price, current_time)
        
        vega_change = greeks['vega'] * (target_iv - current_iv) * 100
        
        return {
            'current_vega': greeks['vega'],
            'vega_change': vega_change,
            'hedged_vega': greeks['vega'] + vega_change,
            'iv_sensitivity': greeks['vega'] * 0.01
        }


class StressTester:
    """压力测试器"""
    
    DEFAULT_SCENARIOS = {
        'price_up_10': {'price_change': 0.10, 'iv_change': 0.0},
        'price_up_20': {'price_change': 0.20, 'iv_change': 0.0},
        'price_up_30': {'price_change': 0.30, 'iv_change': 0.0},
        'price_down_10': {'price_change': -0.10, 'iv_change': 0.0},
        'price_down_20': {'price_change': -0.20, 'iv_change': 0.0},
        'price_down_30': {'price_change': -0.30, 'iv_change': 0.0},
        'iv_up_20': {'price_change': 0.0, 'iv_change': 0.20},
        'iv_up_50': {'price_change': 0.0, 'iv_change': 0.50},
        'iv_down_20': {'price_change': 0.0, 'iv_change': -0.20},
        'iv_down_50': {'price_change': 0.0, 'iv_change': -0.50},
        'crash': {'price_change': -0.20, 'iv_change': 0.50},
        'rally': {'price_change': 0.20, 'iv_change': -0.20},
    }
    
    def __init__(self, r: float = 0.03):
        self.r = r
    
    def run_stress_test(
        self,
        positions: List[Dict],
        current_price: float,
        current_iv: float,
        scenarios: Optional[Dict[str, Dict]] = None,
        current_time: float = 0
    ) -> List[StressResult]:
        """
        运行压力测试
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
            current_iv: 当前隐含波动率
            scenarios: 自定义情景 (可选)
            current_time: 当前时间
        
        返回:
            List[StressResult]: 压力测试结果列表
        """
        if scenarios is None:
            scenarios = self.DEFAULT_SCENARIOS
        
        results = []
        
        for name, params in scenarios.items():
            price_change_pct = params['price_change']
            iv_change_pct = params['iv_change']
            
            new_price = current_price * (1 + price_change_pct)
            new_iv = current_iv * (1 + iv_change_pct)
            
            pnl = self._calculate_pnl(
                positions, current_price, new_price,
                current_iv, new_iv, current_time
            )
            
            greeks = self._calculate_greeks_at(
                positions, new_price, new_iv, current_time
            )
            
            results.append(StressResult(
                scenario=name,
                price_change=price_change_pct * 100,
                iv_change=iv_change_pct * 100,
                pnl=pnl,
                delta=greeks['delta'],
                gamma=greeks['gamma']
            ))
        
        return results
    
    def _calculate_pnl(
        self,
        positions: List[Dict],
        old_price: float,
        new_price: float,
        old_iv: float,
        new_iv: float,
        current_time: float
    ) -> float:
        """计算 P&L"""
        total_pnl = 0.0
        
        for pos in positions:
            k = pos.get('strike', pos.get('k', 5500))
            iv_old = pos.get('iv', pos.get('sigma', old_iv))
            qty = pos.get('quantity', pos.get('side', 1))
            opt_type = pos.get('type', 'call')
            t = pos.get('expiry', pos.get('t', 30/365))
            premium = pos.get('premium', 0)
            
            old_price_opt = BSPricing(old_price, k, self.r, t, iv_old, 1 if opt_type == 'call' else -1)
            new_price_opt = BSPricing(new_price, k, self.r, t, new_iv, 1 if opt_type == 'call' else -1)
            
            pnl = (new_price_opt - old_price_opt) * qty * 100 + premium * qty
            total_pnl += pnl
        
        return total_pnl
    
    def _calculate_greeks_at(
        self,
        positions: List[Dict],
        price: float,
        iv: float,
        current_time: float
    ) -> Dict[str, float]:
        """计算特定价格和 IV 下的 Greeks"""
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        
        for pos in positions:
            k = pos.get('strike', pos.get('k', 5500))
            qty = pos.get('quantity', pos.get('side', 1))
            opt_type = pos.get('type', 'call')
            t = pos.get('expiry', pos.get('t', 30/365))
            
            g = calculate_all_greeks(price, k, self.r, 0, iv, t, opt_type)
            
            total_delta += g.delta * qty * 100
            total_gamma += g.gamma * qty * 100
            total_vega += g.vega * qty * 100
        
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega
        }
    
    def calculate_var(
        self,
        positions: List[Dict],
        current_price: float,
        current_iv: float,
        confidence: float = 0.95,
        simulations: int = 10000,
        current_time: float = 0
    ) -> Dict[str, float]:
        """
        计算 VaR (Value at Risk)
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
            current_iv: 当前隐含波动率
            confidence: 置信水平
            simulations: 模拟次数
            current_time: 当前时间
        
        返回:
            dict: VaR 结果
        """
        price_returns = np.random.normal(0, 0.02, simulations)
        iv_returns = np.random.normal(0, 0.10, simulations)
        
        pnls = []
        for pr, ir in zip(price_returns, iv_returns):
            new_price = current_price * (1 + pr)
            new_iv = current_iv * (1 + ir)
            
            pnl = self._calculate_pnl(
                positions, current_price, new_price,
                current_iv, new_iv, current_time
            )
            pnls.append(pnl)
        
        pnls = np.array(pnls)
        var_95 = np.percentile(pnls, (1 - confidence) * 100)
        var_99 = np.percentile(pnls, 1)
        cvar_95 = pnls[pnls <= var_95].mean()
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'max_loss': np.min(pnls),
            'max_gain': np.max(pnls),
            'mean_pnl': np.mean(pnls)
        }


class MarginCalculator:
    """保证金计算器"""
    
    def __init__(self, initial_margin_rate: float = 0.12, maintenance_rate: float = 0.10):
        """
        初始化保证金计算器
        
        参数:
            initial_margin_rate: 初始保证金比例
            maintenance_rate: 维持保证金比例
        """
        self.initial_margin_rate = initial_margin_rate
        self.maintenance_rate = maintenance_rate
    
    def calculate_option_margin(
        self,
        s: float,
        k: float,
        premium: float,
        iv: float,
        cp_sign: int
    ) -> float:
        """
        计算单一期权保证金
        
        基于 CME 规则简化:
        期权保证金 = max(期权价值, 标的价值 * 保证金比例)
        
        参数:
            s: 标的资产价格
            k: 行权价
            premium: 期权权利金
            iv: 隐含波动率
            cp_sign: 1=看涨, -1=看跌
        
        返回:
            float: 保证金
        """
        from .bsm import BSPricing
        
        option_value = premium
        underlying_value = s
        
        if cp_sign == 1:
            intrinsic = max(0, s - k)
        else:
            intrinsic = max(0, k - s)
        
        margin = max(option_value, intrinsic * 0.10 + underlying_value * 0.10)
        
        return margin * 100
    
    def calculate_spread_margin(
        self,
        positions: List[Dict],
        current_price: float
    ) -> Dict[str, float]:
        """
        计算价差保证金
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
        
        返回:
            dict: 保证金结果
        """
        long_margin = 0.0
        short_margin = 0.0
        
        for pos in positions:
            qty = pos.get('quantity', pos.get('side', 1))
            
            if qty > 0:
                margin = self.calculate_option_margin(
                    current_price,
                    pos.get('strike', pos.get('k', 5500)),
                    pos.get('premium', 0),
                    pos.get('iv', pos.get('sigma', 0.2)),
                    1 if pos.get('type', 'call') == 'call' else -1
                )
                long_margin += margin * qty
            else:
                margin = self.calculate_option_margin(
                    current_price,
                    pos.get('strike', pos.get('k', 5500)),
                    pos.get('premium', 0),
                    pos.get('iv', pos.get('sigma', 0.2)),
                    1 if pos.get('type', 'call') == 'call' else -1
                )
                short_margin += abs(margin * qty)
        
        spread_discount = min(short_margin * 0.5, long_margin)
        
        total_margin = long_margin + short_margin - spread_discount
        
        return {
            'long_margin': long_margin,
            'short_margin': short_margin,
            'spread_discount': spread_discount,
            'total_margin': max(0, total_margin)
        }
    
    def calculate_portfolio_margin(
        self,
        positions: List[Dict],
        current_price: float
    ) -> float:
        """
        计算组合总保证金
        
        参数:
            positions: 仓位列表
            current_price: 当前标的价格
        
        返回:
            float: 总保证金
        """
        spread_result = self.calculate_spread_margin(positions, current_price)
        
        underlying_exposure = 0
        for pos in positions:
            qty = pos.get('quantity', pos.get('side', 1))
            underlying_exposure += abs(qty * current_price * 100)
        
        return max(spread_result['total_margin'], underlying_exposure * self.maintenance_rate)


def calculate_margin(
    positions: List[Dict],
    current_price: float,
    initial_margin_rate: float = 0.12,
    maintenance_rate: float = 0.10
) -> Dict[str, float]:
    """
    便捷函数: 计算期权组合保证金
    
    参数:
        positions: 仓位列表
        current_price: 当前标的价格
        initial_margin_rate: 初始保证金比例
        maintenance_rate: 维持保证金比例
    
    返回:
        dict: 保证金详情
    
    示例:
        >>> positions = [{'type': 'call', 'k': 5500, 'iv': 0.20, 'side': -1, 'premium': 100}]
        >>> result = calculate_margin(positions, 5500)
        >>> print(f"Total Margin: {result['total_margin']:.2f}")
    """
    calc = MarginCalculator(initial_margin_rate, maintenance_rate)
    spread_result = calc.calculate_spread_margin(positions, current_price)
    total = calc.calculate_portfolio_margin(positions, current_price)
    return {
        'total_margin': total,
        'long_margin': spread_result.get('long_margin', 0),
        'short_margin': spread_result.get('short_margin', 0),
        'spread_discount': spread_result.get('spread_discount', 0)
    }
