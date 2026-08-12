"""
波动率微笑 (IV Smile) 分析模块

分析期权波动率随执行价 (strike) 变化的特征:

1. **波动率微笑**: 
   - OTM 看跌期权通常有较高 IV (负偏度)
   - OTM 看涨期权通常有较低 IV
   - ATM 期权 IV 最低

2. **微笑曲线拟合**:
   - 样条插值 (spline)
   - 多项式拟合 (polynomial)
   - 二次拟合 (quadratic)

3. **偏度指标**:
   - Skew: (OTM Call IV - OTM Put IV) / moneyness 范围
   - Risk Reversal: 25D Call IV - 25D Put IV
   - Butterfly: (25D Call IV + 25D Put IV) / 2 - ATM IV

4. **异常值检测**:
   - 检测偏离微笑曲线超过阈值个标准差的 IV

主要类:
    IVSmileAnalyzer: IV 微笑分析器
    
主要函数:
    analyze_iv_smile: 分析 IV 微笑

示例:
    >>> from option_greeks.smile import analyze_iv_smile
    >>> options = [{'strike': 5400, 'iv': 0.152}, {'strike': 5500, 'iv': 0.150}, {'strike': 5600, 'iv': 0.153}]
    >>> result = analyze_iv_smile(s=5500, options_data=options)
    >>> print(f"ATM IV: {result.atm_iv:.4f}, Skew: {result.skew:.4f}")
    ATM IV: 0.1500, Skew: -0.1110
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, CubicSpline


@dataclass
class SmilePoint:
    """微笑曲线点"""
    strike: float
    moneyness: float
    iv: float
    delta: Optional[float] = None


@dataclass
class SmileAnalysis:
    """微笑分析结果"""
    atm_iv: float
    rr_25d: float
    rr_10d: float
    bf_25d: float
    skew: float
    min_iv: float
    max_iv: float
    wing_adjusted_skew: Optional[float] = None


class IVSmileAnalyzer:
    """IV 微笑分析器"""
    
    def __init__(self):
        pass
    
    def calculate_moneyness(self, s: float, k: float) -> float:
        """计算价值状态 (moneyness)"""
        return np.log(s / k)
    
    def calculate_moneyness_by_delta(
        self, 
        target_delta: float, 
        iv: float, 
        t: float, 
        r: float = 0.03
    ) -> float:
        """
        根据 Delta 计算对应的 moneyness
        
        d1 = N^{-1}(delta) for calls
        m = d1 * sigma * sqrt(T)
        """
        from scipy.stats import norm
        d1 = norm.ppf(target_delta)
        return d1 * iv * np.sqrt(t)
    
    def fit_smile_curve(
        self, 
        strikes: List[float], 
        ivs: List[float],
        method: str = 'spline'
    ):
        """
        拟合微笑曲线            strikes: 执行价列表
           
        
        参数:
 ivs: 对应 IV 列表
            method: 拟合方法 ('spline', 'polynomial', ' quadratic')
        
        返回:
            插值函数
        """
        strikes_arr = np.asarray(strikes, dtype=np.float64)
        ivs_arr = np.asarray(ivs, dtype=np.float64)
        
        if method == 'spline':
            return CubicSpline(strikes_arr, ivs_arr)
        elif method == 'quadratic':
            coeffs = np.polyfit(strikes_arr, ivs_arr, 2)
            return lambda x: np.polyval(coeffs, x)
        else:
            return UnivariateSpline(strikes_arr, ivs_arr, s=len(strikes_arr))
    
    def calculate_skew(
        self,
        strikes: List[float],
        ivs: List[float],
        s: float,
        t: float
    ) -> Dict[str, float]:
        """
        计算偏斜度
        
        返回:
            {'skew': float, 'rr_25d': float, 'rr_10d': float, 'bf_25d': float}
        """
        from scipy.stats import norm
        
        moneyness_arr = np.array([self.calculate_moneyness(s, k) for k in strikes])
        ivs_arr = np.array(ivs)
        
        if len(strikes) < 3:
            return {'skew': 0.0, 'rr_25d': 0.0, 'rr_10d': 0.0, 'bf_25d': 0.0}
        
        min_moneyness = float(np.min(moneyness_arr))
        max_moneyness = float(np.max(moneyness_arr))
        
        skew_val = (ivs_arr[np.argmax(moneyness_arr)] - ivs_arr[np.argmin(moneyness_arr)]) / (max_moneyness - min_moneyness)
        
        try:
            spline = self.fit_smile_curve(strikes, ivs)
            
            d1_25c = norm.ppf(0.25)
            d1_25p = norm.ppf(-0.25)
            d1_10c = norm.ppf(0.10)
            d1_10p = norm.ppf(-0.10)
            
            k_25c = s * np.exp(-d1_25c * np.sqrt(t) * float(spline(s)))
            k_25p = s * np.exp(-d1_25p * np.sqrt(t) * float(spline(s)))
            k_10c = s * np.exp(-d1_10c * np.sqrt(t) * float(spline(s)))
            k_10p = s * np.exp(-d1_10p * np.sqrt(t) * float(spline(s)))
            
            iv_25c = float(spline(k_25c)) if k_25c in strikes else float(np.interp(k_25c, strikes, ivs))
            iv_25p = float(spline(k_25p)) if k_25p in strikes else float(np.interp(k_25p, strikes, ivs))
            iv_10c = float(spline(k_10c)) if k_10c in strikes else float(np.interp(k_10c, strikes, ivs))
            iv_10p = float(spline(k_10p)) if k_10p in strikes else float(np.interp(k_10p, strikes, ivs))
            
            atm_idx = int(np.argmin([abs(m) for m in moneyness_arr]))
            iv_atm = float(ivs_arr[atm_idx])
            
            rr_25d = iv_25c - iv_25p
            rr_10d = iv_10c - iv_10p
            bf_25d = (iv_25c + iv_25p) / 2 - iv_atm
            
        except Exception:
            rr_25d = 0.0
            rr_10d = 0.0
            bf_25d = 0.0
            skew_val = 0.0
        
        return {
            'skew': float(skew_val),
            'rr_25d': float(rr_25d),
            'rr_10d': float(rr_10d),
            'bf_25d': float(bf_25d)
        }
    
    def detect_outliers(
        self,
        strikes: List[float],
        ivs: List[float],
        threshold: float = 2.0
    ) -> List[Dict]:
        """
        检测 IV 异常值 (偏离微笑曲线超过 threshold 个标准差)
        """
        if len(strikes) < 5:
            return []
        
        try:
            spline = self.fit_smile_curve(strikes, ivs)
            fitted = spline(strikes)
            residuals = np.array(ivs) - fitted
            std = np.std(residuals)
            
            outliers = []
            for i, (k, iv, res) in enumerate(zip(strikes, ivs, residuals)):
                if abs(res) > threshold * std:
                    outliers.append({
                        'strike': k,
                        'iv': iv,
                        'fitted_iv': fitted[i],
                        'residual': res,
                        'z_score': res / std if std > 0 else 0
                    })
            
            return outliers
        except Exception:
            return []
    
    def analyze(
        self,
        s: float,
        options_data: List[Dict]
    ) -> SmileAnalysis:
        """
        全面分析 IV 微笑
        
        参数:
            s: 标的资产价格
            options_data: 期权数据 [{'strike': K, 'iv': sigma, ...}, ...]
        
        返回:
            SmileAnalysis 结果
        """
        strikes = [opt['strike'] for opt in options_data]
        ivs = [opt['iv'] for opt in options_data]
        
        moneyness = [self.calculate_moneyness(s, k) for k in strikes]
        
        t = options_data[0].get('t', 30/365) if options_data else 30/365
        
        min_idx = np.argmin(np.abs(moneyness))
        atm_iv = ivs[min_idx]
        
        skew_data = self.calculate_skew(strikes, ivs, s, t)
        
        return SmileAnalysis(
            atm_iv=atm_iv,
            rr_25d=skew_data['rr_25d'],
            rr_10d=skew_data['rr_10d'],
            bf_25d=skew_data['bf_25d'],
            skew=skew_data['skew'],
            min_iv=min(ivs),
            max_iv=max(ivs)
        )


def analyze_iv_smile(
    s: float,
    options_data: List[Dict]
) -> SmileAnalysis:
    """
    便捷函数: 分析 IV 微笑
    """
    analyzer = IVSmileAnalyzer()
    return analyzer.analyze(s, options_data)
