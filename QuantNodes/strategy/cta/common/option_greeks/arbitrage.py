"""
期权套利分析模块

检测期权市场中的潜在套利机会:

1. **Put-Call Parity 违背**: C - P = S - K*e^(-rT)
   - 如果市场价格违背此等式，存在理论套利机会

2. **Box Spread 套利**: 
   - 买入看涨期权 K1 + 卖出看涨期权 K2 + 买入看跌期权 K2 + 卖出看跌期权 K1
   - 价值 = (K2 - K1) * e^(-rT)

3. **垂直价差套利**:
   - Bull Call Spread: 买入低行权价看涨 + 卖出高行权价看涨
   - Bear Put Spread: 买入高行权价看跌 + 卖出低行权价看跌

4. **Butterfly Spread 套利**:
   - 买入看涨 K1 + 卖出2个看涨 K2 + 买入看涨 K3

主要类:
    ArbitrageAnalyzer: 套利分析器
    
主要函数:
    detect_arbitrage: 检测期权链中的套利机会

示例:
    >>> from option_greeks.arbitrage import detect_arbitrage
    >>> options = [
    ...     {'strike': 5400, 'call': 125.5, 'put': 18.2, 'iv': 0.152},
    ...     {'strike': 5500, 'call': 75.8, 'put': 37.5, 'iv': 0.150},
    ... ]
    >>> opportunities = detect_arbitrage(options, 5500, 30/365)
    >>> for opp in opportunities:
    ...     print(f"{opp.type}: {opp.profit:.2f}")
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from .bsm import BSPricing
from .greeks import calculate_all_greeks


@dataclass
class ArbitrageOpportunity:
    """套利机会"""
    type: str
    description: str
    profit: float
    profit_pct: float
    details: Dict


class ArbitrageAnalyzer:
    """套利分析器"""
    
    def __init__(self, r: float = 0.03, commission: float = 0.65):
        """
        参数:
            r: 无风险利率
            commission: 每手合约佣金
        """
        self.r = r
        self.commission = commission
    
    def check_put_call_parity(
        self, 
        s: float, 
        k: float, 
        t: float, 
        call_price: float, 
        put_price: float,
        g: float = 0
    ) -> Optional[ArbitrageOpportunity]:
        """
        检查 Put-Call Parity 违背
        
        公式: C - P = S - K*e^(-rT)
        违背时存在套利机会
        """
        synthetic = s - k * np.exp(-self.r * t)
        actual = call_price - put_price
        
        parity_error = actual - synthetic
        
        max_error = s * 0.01
        if abs(parity_error) > max_error:
            return ArbitrageOpportunity(
                type="Put-Call Parity Violation",
                description=f"Parity deviation: {parity_error:.4f}",
                profit=abs(parity_error),
                profit_pct=abs(parity_error) / s * 100,
                details={
                    'synthetic': synthetic,
                    'actual': actual,
                    'deviation': parity_error
                }
            )
        return None
    
    def check_box_spread(
        self,
        k1: float,
        k2: float,
        t: float,
        c1: float,
        p1: float,
        c2: float,
        p2: float,
        use_commission: bool = True
    ) -> Optional[ArbitrageOpportunity]:
        """
        检查 Box Spread 套利
        
        Box = (Call at K1 + Put at K2) - (Call at K2 + Put at K1)
        价值 = (K2 - K1) * e^(-rT)
        """
        comm = 4 * self.commission if use_commission else 0
        
        box_value = (c1 + p2) - (c2 + p1)
        box_theoretical = (k2 - k1) * np.exp(-self.r * t)
        
        profit = box_theoretical - box_value - comm
        
        if profit > 0.01:
            return ArbitrageOpportunity(
                type="Box Spread",
                description=f"Risk-free profit opportunity",
                profit=profit,
                profit_pct=profit / (k2 - k1) * 100,
                details={
                    'box_value': box_value,
                    'theoretical': box_theoretical,
                    'commission': comm,
                    'k1': k1,
                    'k2': k2
                }
            )
        return None
    
    def check_vertical_spread(
        self,
        k1: float,
        k2: float,
        t: float,
        sigma: float,
        spread_price: float,
        is_bull: bool = True,
        use_commission: bool = True
    ) -> Optional[ArbitrageOpportunity]:
        """
        检查垂直价差套利
        
        Bull Call Spread: Buy K1 Call, Sell K2 Call (K1 < K2)
        Bear Put Spread: Buy K2 Put, Sell K1 Put (K1 < K2)
        """
        comm = 2 * self.commission if use_commission else 0
        
        k_buy = k1 if is_bull else k2
        k_sell = k2 if is_bull else k1
        
        theoretical = BSPricing(
            s=0, k=k_buy, r=self.r, t=t, sigma=sigma, cp_sign=1, g=0
        ) - BSPricing(
            s=0, k=k_sell, r=self.r, t=t, sigma=sigma, cp_sign=1, g=0
        )
        
        profit = theoretical - spread_price - comm
        
        if profit > 0.01:
            return ArbitrageOpportunity(
                type=f"{'Bull Call' if is_bull else 'Bear Put'} Spread",
                description=f"Vertical spread mispricing",
                profit=profit,
                profit_pct=profit / spread_price * 100 if spread_price > 0 else 0,
                details={
                    'theoretical': theoretical,
                    'market': spread_price,
                    'commission': comm,
                    'k1': k1,
                    'k2': k2
                }
            )
        return None
    
    def check_butterfly_spread(
        self,
        k1: float,
        k2: float,
        k3: float,
        c1: float,
        c2: float,
        c3: float,
        t: float,
        sigma: float,
        use_commission: bool = True
    ) -> Optional[ArbitrageOpportunity]:
        """
        检查 Butterfly Spread 套利
        
        Long Butterfly: Buy 1 Call K1 + 1 Call K3 - 2 Call K2
        价值范围: [0, K3-K1], 内在价值在 K2 处最大
        """
        comm = 4 * self.commission if use_commission else 0
        
        butterfly_value = c1 + c3 - 2 * c2
        
        theoretical_min = max(0, k2 - k1) * np.exp(-self.r * t)
        theoretical_max = max(0, k3 - k2) * np.exp(-self.r * t)
        
        profit = theoretical_min - butterfly_value - comm
        
        if abs(butterfly_value) > 0.01 and profit > 0.01:
            return ArbitrageOpportunity(
                type="Butterfly Spread",
                description=f"Butterfly mispricing",
                profit=profit,
                profit_pct=profit / butterfly_value * 100 if butterfly_value != 0 else 0,
                details={
                    'market_value': butterfly_value,
                    'theoretical_min': theoretical_min,
                    'theoretical_max': theoretical_max,
                    'k1': k1,
                    'k2': k2,
                    'k3': k3
                }
            )
        return None
    
    def check_intrinsic_arbitrage(
        self,
        s: float,
        k: float,
        t: float,
        call_price: float,
        put_price: float,
        cp_sign: int = 1
    ) -> Optional[ArbitrageOpportunity]:
        """
        检查内在价值套利
        
        看涨期权价格不能低于内在价值: max(S - K*e^(-rT), 0)
        看跌期权价格不能低于内在价值: max(K*e^(-rT) - S, 0)
        """
        intrinsic_call = max(s - k * np.exp(-self.r * t), 0)
        intrinsic_put = max(k * np.exp(-self.r * t) - s, 0)
        
        if cp_sign == 1:
            if call_price < intrinsic_call - 0.01:
                return ArbitrageOpportunity(
                    type="Intrinsic Call Arbitrage",
                    description="Call price below intrinsic value",
                    profit=intrinsic_call - call_price,
                    profit_pct=(intrinsic_call - call_price) / call_price * 100 if call_price > 0 else 0,
                    details={
                        'intrinsic': intrinsic_call,
                        'market': call_price,
                        's': s,
                        'k': k
                    }
                )
        else:
            if put_price < intrinsic_put - 0.01:
                return ArbitrageOpportunity(
                    type="Intrinsic Put Arbitrage",
                    description="Put price below intrinsic value",
                    profit=intrinsic_put - put_price,
                    profit_pct=(intrinsic_put - put_price) / put_price * 100 if put_price > 0 else 0,
                    details={
                        'intrinsic': intrinsic_put,
                        'market': put_price,
                        's': s,
                        'k': k
                    }
                )
        return None
    
    def analyze_opportunities(
        self,
        options_data: List[Dict],
        s: float,
        t: float
    ) -> List[ArbitrageOpportunity]:
        """
        分析期权链中的所有套利机会
        
        参数:
            options_data: 期权数据列表 [{'strike': K, 'call': C, 'put': P, 'iv': sigma}, ...]
            s: 标的资产价格
            t: 到期时间 (年)
        """
        opportunities = []
        
        for opt in options_data:
            k = opt['strike']
            c = opt.get('call', 0)
            p = opt.get('put', 0)
            sigma = opt.get('iv', 0.3)
            
            pc = self.check_put_call_parity(s, k, t, c, p)
            if pc:
                opportunities.append(pc)
            
            ia = self.check_intrinsic_arbitrage(s, k, t, c, p, 1)
            if ia:
                opportunities.append(ia)
            
            ia_put = self.check_intrinsic_arbitrage(s, k, t, c, p, -1)
            if ia_put:
                opportunities.append(ia_put)
        
        for i in range(len(options_data)):
            for j in range(i + 1, len(options_data)):
                k1 = options_data[i]['strike']
                k2 = options_data[j]['strike']
                c1 = options_data[i].get('call', 0)
                c2 = options_data[j].get('call', 0)
                p1 = options_data[i].get('put', 0)
                p2 = options_data[j].get('put', 0)
                sigma = (options_data[i].get('iv', 0.3) + options_data[j].get('iv', 0.3)) / 2
                
                if c1 > 0 and c2 > 0 and p1 > 0 and p2 > 0:
                    box = self.check_box_spread(
                        k1, k2, t, c1, p1, c2, p2
                    )
                    if box:
                        opportunities.append(box)
        
        return sorted(opportunities, key=lambda x: -x.profit)


def detect_arbitrage(
    options_chain: List[Dict],
    underlying_price: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.03,
    commission: float = 0.65
) -> List[ArbitrageOpportunity]:
    """
    便捷函数: 检测期权链中的套利机会
    
    参数:
        options_chain: 期权链数据
            [{'strike': K, 'call': C, 'put': P, 'iv': sigma}, ...]
        underlying_price: 标的资产价格
        time_to_expiry: 到期时间 (年)
        risk_free_rate: 无风险利率
        commission: 手续费
    
    返回:
        套利机会列表
    """
    analyzer = ArbitrageAnalyzer(r=risk_free_rate, commission=commission)
    return analyzer.analyze_opportunities(options_chain, underlying_price, time_to_expiry)
