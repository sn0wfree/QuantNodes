"""
波动率偏斜 (Skew) 分析模块

分析期权偏斜特征和尾部风险:

1. **Delta 概念**:
   - 25-delta: 期权 Delta 绝对值为 0.25 的行权价
   - 10-delta: 期权 Delta 绝对值为 0.10 的行权价
   
2. **风险逆转指标 (Risk Reversal)**:
   - RR = IV(25D Call) - IV(25D Put)
   - 正值: 市场偏向看涨 (上涨预期)
   - 负值: 市场偏向看跌 (下跌预期)
   
3. **蝶式价差指标 (Butterfly)**:
   - BF = (IV(25D Call) + IV(25D Put)) / 2 - IV(ATM)
   - 正值: 尾部风险溢价高 (波动率更高)
   - 负值: 微笑平滑

4. **偏斜度 (Skew)**:
   - Skew = RR / ATM IV
   - 衡量相对偏斜程度

主要类:
    SkewAnalyzer: 偏斜分析器
    
主要函数:
    analyze_skew: 分析偏斜指标

示例:
    >>> from option_greeks.skew import analyze_skew
    >>> options = [
    ...     {'strike': 5400, 'call_iv': 0.152, 'put_iv': 0.155},
    ...     {'strike': 5500, 'call_iv': 0.150, 'put_iv': 0.150},
    ...     {'strike': 5600, 'call_iv': 0.153, 'put_iv': 0.152},
    ... ]
    >>> result = analyze_skew(s=5500, t=30/365, options_data=options)
    >>> print(f"Risk Reversal: {result.rr_25d:.4f}, ATM IV: {result.atm_iv:.4f}")
    Risk Reversal: 0.0000, ATM IV: 0.1500
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from scipy.interpolate import CubicSpline
from scipy.stats import norm


@dataclass
class SkewMetrics:
    """Skew 指标"""
    rr_25d: float
    rr_10d: float
    bf_25d: float
    bf_10d: float
    skew_25d: float
    skew_10d: float
    call_25d_iv: Optional[float] = None
    put_25d_iv: Optional[float] = None
    call_10d_iv: Optional[float] = None
    put_10d_iv: Optional[float] = None
    atm_iv: Optional[float] = None


class SkewAnalyzer:
    """Skew 分析器"""
    
    def __init__(self):
        pass
    
    def delta_to_strike(
        self, 
        s: float, 
        t: float, 
        iv: float, 
        target_delta: float,
        r: float = 0.03,
        cp_sign: int = 1
    ) -> float:
        """
        根据 Delta 计算执行价
        
        d1 = N^{-1}(delta / cp_sign)
        K = S * exp(-d1 * sigma * sqrt(T) + (r - q)*T)
        """
        if cp_sign == 1:
            d1 = norm.ppf(target_delta)
        else:
            d1 = norm.ppf(-target_delta)
        
        k = s * np.exp(-d1 * iv * np.sqrt(t) + r * t)
        return k
    
    def strike_to_delta(
        self,
        s: float,
        k: float,
        t: float,
        iv: float,
        r: float = 0.03,
        cp_sign: int = 1
    ) -> float:
        """
        根据执行价计算 Delta
        """
        d1 = (np.log(s / k) + (r + 0.5 * iv ** 2) * t) / (iv * np.sqrt(t))
        
        if cp_sign == 1:
            return float(norm.cdf(d1))
        else:
            return float(norm.cdf(d1) - 1)
    
    def calculate_rr(
        self,
        call_iv: float,
        put_iv: float
    ) -> float:
        """
        计算风险逆转指标 (Risk Reversal)
        
        RR = IV(call) - IV(put)
        正值: 上涨预期 (更多上涨杠杆)
        负值: 下跌预期 (更多下跌杠杆)
        """
        return call_iv - put_iv
    
    def calculate_bf(
        self,
        call_iv: float,
        atm_iv: float,
        put_iv: float
    ) -> float:
        """
        计算蝶式价差指标 (Butterfly)
        
        BF = (IV(call) + IV(put)) / 2 - IV(ATM)
        正值: 尾部风险溢价高
        负值: 微笑平滑
        """
        return (call_iv + put_iv) / 2 - atm_iv
    
    def interpolate_iv_at_delta(
        self,
        options: List[Dict],
        target_delta: float,
        cp_sign: int = 1
    ) -> Optional[float]:
        """
        插值计算指定 Delta 的 IV
        """
        if cp_sign == 1:
            deltas = [opt.get('call_delta', 0) for opt in options if 'call_delta' in opt]
            ivs = [opt.get('call_iv', 0) for opt in options if 'call_iv' in opt]
        else:
            deltas = [opt.get('put_delta', 0) for opt in options if 'put_delta' in opt]
            ivs = [opt.get('put_iv', 0) for opt in options if 'put_iv' in opt]
        
        if len(deltas) < 2 or len(ivs) < 2:
            return None
        
        valid = [(d, i) for d, i in zip(deltas, ivs) if d != 0 and i > 0]
        if len(valid) < 2:
            return None
        
        valid = sorted(valid, key=lambda x: x[0])
        deltas = [v[0] for v in valid]
        ivs = [v[1] for v in valid]
        
        if target_delta < min(deltas) or target_delta > max(deltas):
            return None
        
        spline = CubicSpline(deltas, ivs)
        return float(spline(target_delta))
    
    def analyze_skew(
        self,
        s: float,
        t: float,
        options_data: List[Dict]
    ) -> SkewMetrics:
        """
        全面分析 Skew 指标
        
        参数:
            s: 标的资产价格
            t: 到期时间
            options_data: 期权数据
                [{'strike': K, 'call_iv': civ, 'put_iv': piv, 'call_delta': cd, 'put_delta': pd, ...}, ...]
        """
        call_options = [opt for opt in options_data if 'call_iv' in opt and opt['call_iv'] > 0]
        put_options = [opt for opt in options_data if 'put_iv' in opt and opt['put_iv'] > 0]
        
        if not call_options or not put_options:
            return SkewMetrics(0, 0, 0, 0, 0, 0)
        
        call_options = sorted(call_options, key=lambda x: x['strike'])
        put_options = sorted(put_options, key=lambda x: x['strike'])
        
        atm_call = min(call_options, key=lambda x: abs(x['strike'] - s))
        atm_put = min(put_options, key=lambda x: abs(x['strike'] - s))
        atm_iv = (atm_call.get('call_iv', 0) + atm_put.get('put_iv', 0)) / 2
        
        iv_25d_call = self.interpolate_iv_at_delta(call_options, 0.25, 1)
        iv_25d_put = self.interpolate_iv_at_delta(put_options, 0.25, -1)
        iv_10d_call = self.interpolate_iv_at_delta(call_options, 0.10, 1)
        iv_10d_put = self.interpolate_iv_at_delta(put_options, 0.10, -1)
        
        rr_25d = self.calculate_rr(iv_25d_call, iv_25d_put) if iv_25d_call and iv_25d_put else 0
        rr_10d = self.calculate_rr(iv_10d_call, iv_10d_put) if iv_10d_call and iv_10d_put else 0
        
        bf_25d = self.calculate_bf(iv_25d_call, atm_iv, iv_25d_put) if iv_25d_call and iv_25d_put else 0
        bf_10d = self.calculate_bf(iv_10d_call, atm_iv, iv_10d_put) if iv_10d_call and iv_10d_put else 0
        
        skew_25d = rr_25d / atm_iv if atm_iv > 0 else 0
        skew_10d = rr_10d / atm_iv if atm_iv > 0 else 0
        
        return SkewMetrics(
            rr_25d=rr_25d,
            rr_10d=rr_10d,
            bf_25d=bf_25d,
            bf_10d=bf_10d,
            skew_25d=skew_25d,
            skew_10d=skew_10d,
            call_25d_iv=iv_25d_call,
            put_25d_iv=iv_25d_put,
            call_10d_iv=iv_10d_call,
            put_10d_iv=iv_10d_put,
            atm_iv=atm_iv
        )
    
    def calculate_time_spread_skew(
        self,
        skew_history: List[Dict]
    ) -> Dict:
        """
        计算 Skew 时间序列统计
        
        参数:
            skew_history: Skew 历史数据 [{'date': ..., 'rr_25d': ..., 'bf_25d': ...}, ...]
        """
        if len(skew_history) < 2:
            return {'mean': 0, 'std': 0, 'trend': 0}
        
        rr_25d_values = [s['rr_25d'] for s in skew_history if 'rr_25d' in s]
        
        if not rr_25d_values:
            return {'mean': 0, 'std': 0, 'trend': 0}
        
        return {
            'mean': np.mean(rr_25d_values),
            'std': np.std(rr_25d_values),
            'min': np.min(rr_25d_values),
            'max': np.max(rr_25d_values),
            'current': rr_25d_values[-1] if rr_25d_values else 0,
            'z_score': (rr_25d_values[-1] - np.mean(rr_25d_values)) / np.std(rr_25d_values) if np.std(rr_25d_values) > 0 else 0
        }


def analyze_skew(
    s: float,
    t: float,
    options_data: List[Dict]
) -> SkewMetrics:
    """
    便捷函数: 分析 Skew 指标
    """
    analyzer = SkewAnalyzer()
    return analyzer.analyze_skew(s, t, options_data)
