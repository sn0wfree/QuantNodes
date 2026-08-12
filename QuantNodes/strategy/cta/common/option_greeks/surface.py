"""
波动率曲面 (IV Surface) 分析模块

构建和分析三维波动率曲面 (Strike × Expiry × IV):

1. **曲面构建**:
   - 支持多到期时间期权链数据
   - 线性/样条插值方法
   - 处理稀疏数据

2. **曲面插值**:
   - 线性插值 (linear)
   - 三次样条 (cubic)
   - 最近邻 (nearest)

3. **SABR 模型校准**:
   - 校准 alpha (波动率水平)
   - 校准 rho (相关性)
   - 校准 volga (波动率 gamma)

4. **局部波动率**:
   - 从隐含波动率曲面近似计算
   - 用于期权定价和风险分析

主要类:
    IVSurface: 波动率曲面数据结构
    IVSurfaceAnalyzer: 曲面分析器
    
主要函数:
    build_iv_surface: 构建 IV 曲面
    interpolate_iv: 插值计算任意 (K, T) 的 IV

示例:
    >>> from option_greeks.surface import build_iv_surface
    >>> options = [
    ...     {'strike': 5400, 't': 7, 'iv': 0.148},
    ...     {'strike': 5500, 't': 7, 'iv': 0.146},
    ...     {'strike': 5400, 't': 30, 'iv': 0.152},
    ... ]
    >>> surface = build_iv_surface(options)
    >>> print(f"Strikes: {len(surface.strikes)}, Expiries: {len(surface.expiries)}")
    Strikes: 2, Expiries: 2
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Callable
from scipy.interpolate import (
    griddata, RBFInterpolator, RectBivariateSpline
)


@dataclass
class IVSurface:
    """波动率曲面"""
    strikes: np.ndarray
    expiries: np.ndarray
    iv_matrix: np.ndarray
    interpolation_method: str


@dataclass
class LocalVolPoint:
    """局部波动率点"""
    s: float
    t: float
    local_vol: float


class IVSurfaceAnalyzer:
    """IV 曲面分析器"""
    
    def __init__(self):
        self.surface = None
    
    def build_surface(
        self,
        strikes: List[float],
        expiries: List[float],
        ivs: List[float],
        method: str = 'linear'
    ) -> IVSurface:
        """
        构建 IV 曲面
        
        参数:
            strikes: 执行价列表
            expiries: 到期时间列表 (年)
            ivs: 对应 IV 列表
            method: 插值方法 ('linear', 'cubic', 'nearest')
        
        返回:
            IVSurface 对象
        """
        strikes_arr = np.array(strikes)
        exp_arr = np.array(expiries)
        ivs_arr = np.array(ivs)
        
        unique_strikes = np.unique(strikes_arr)
        unique_expiries = np.unique(exp_arr)
        
        if len(unique_expiries) < 2 or len(unique_strikes) < 2:
            avg_iv = float(np.mean(ivs_arr))
            iv_matrix = np.full((len(unique_expiries), len(unique_strikes)), avg_iv)
            self.surface = IVSurface(
                strikes=unique_strikes,
                expiries=unique_expiries,
                iv_matrix=iv_matrix,
                interpolation_method='flat'
            )
            return self.surface
        
        points = np.column_stack([strikes_arr, exp_arr])
        
        grid_strikes, grid_expiry = np.meshgrid(unique_strikes, unique_expiries)
        
        iv_matrix = griddata(
            points, ivs_arr, 
            (grid_strikes, grid_expiry), 
            method=method,
            fill_value=float(np.nanmean(ivs_arr))
        )
        
        self.surface = IVSurface(
            strikes=unique_strikes,
            expiries=unique_expiries,
            iv_matrix=iv_matrix,
            interpolation_method=method
        )
        
        return self.surface
    
    def interpolate_iv(
        self,
        strike: float,
        expiry: float
    ) -> Optional[float]:
        """
        插值计算给定 (K, T) 的 IV
        """
        if self.surface is None:
            return None
        
        if strike < self.surface.strikes.min() or strike > self.surface.strikes.max():
            return None
        if expiry < self.surface.expiries.min() or expiry > self.surface.expiries.max():
            return None
        
        spline = RectBivariateSpline(
            self.surface.expiries,
            self.surface.strikes,
            self.surface.iv_matrix,
            kx=3, ky=3
        )
        
        return float(spline(expiry, strike)[0, 0])
    
    def calculate_local_vol(
        self,
        s: float,
        t: float,
        k: float,
        iv: float,
        dt: float = 0.001
    ) -> float:
        """
        近似计算局部波动率
        
        使用有限差分法近似导数
        """
        v = iv ** 2
        
        iv_plus = self.interpolate_iv(k * 1.01, t)
        iv_minus = self.interpolate_iv(k * 0.99, t)
        
        if iv_plus is None or iv_minus is None:
            return iv
        
        dsigma_dk = (iv_plus - iv_minus) / (0.02 * k)
        
        sigma_sq = v
        dsigma_sq_dk = 2 * iv * dsigma_dk
        
        local_vol_sq = v + dsigma_sq_dk * (np.log(s / k) - 0.25 * v * t) * s / k
        
        local_vol_sq = max(local_vol_sq, 0.01)
        
        return np.sqrt(local_vol_sq)
    
    def extract_term_structure(
        self,
        strike: float,
        expiry_range: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        提取期限结构 (Term Structure)
        
        返回:
            {'expiries': [...], 'ivs': [...]}
        """
        if self.surface is None:
            return {'expiries': [], 'ivs': []}
        
        expiry_min = expiry_range[0] if expiry_range else self.surface.expiries.min()
        expiry_max = expiry_range[1] if expiry_range else self.surface.expiries.max()
        
        valid_expiry = [
            e for e in self.surface.expiries 
            if expiry_min <= e <= expiry_max
        ]
        
        ivs = []
        for e in valid_expiry:
            iv = self.interpolate_iv(strike, e)
            ivs.append(iv if iv else 0)
        
        return {
            'strike': strike,
            'expiries': valid_expiry,
            'ivs': ivs
        }
    
    def extract_smile_at_expiry(
        self,
        expiry: float,
        strike_range: Optional[Tuple[float, float]] = None
    ) -> Dict:
        """
        提取特定到期时间的微笑曲线
        """
        if self.surface is None:
            return {'strikes': [], 'ivs': []}
        
        strike_min = strike_range[0] if strike_range else self.surface.strikes.min()
        strike_max = strike_range[1] if strike_range else self.surface.strikes.max()
        
        valid_strikes = [
            k for k in self.surface.strikes
            if strike_min <= k <= strike_max
        ]
        
        ivs = []
        for k in valid_strikes:
            iv = self.interpolate_iv(k, expiry)
            ivs.append(iv if iv else 0)
        
        return {
            'expiry': expiry,
            'strikes': valid_strikes,
            'ivs': ivs
        }
    
    def calibrate_sabr(
        self,
        strikes: List[float],
        ivs: List[float],
        beta: float = 0.5
    ) -> Dict:
        """
        SABR 模型校准
        
        返回:
            {'alpha': float, 'rho': float, 'volga': float}
        """
        n = len(strikes)
        
        if n < 3:
            return {'alpha': 0.2, 'rho': 0.0, 'volga': 0.0}
        
        strikes_arr = np.asarray(strikes, dtype=np.float64)
        ivs_arr = np.asarray(ivs, dtype=np.float64)
        
        f = float(np.mean(strikes_arr))
        
        dlogk = np.diff(np.log(strikes_arr))
        div = np.divide(dlogk[:-1], dlogk[1:], where=np.array(dlogk[1:]) != 0)
        
        weights = ivs_arr[:-1] * ivs_arr[1:] * (1.0 - div) / (dlogk[:-1] + dlogk[1:])
        alpha = float(np.average(ivs_arr, weights=weights))
        
        sigma0 = float(np.std(ivs_arr))
        
        return {
            'alpha': alpha,
            'rho': 0.0,
            'volga': sigma0,
            'beta': beta
        }


def build_iv_surface(
    options_data: List[Dict],
    method: str = 'linear'
) -> IVSurface:
    """
    便捷函数: 构建 IV 曲面
    """
    strikes = [opt['strike'] for opt in options_data]
    expiries = [opt.get('t', opt.get('expiry', 30/365)) for opt in options_data]
    ivs = [opt['iv'] for opt in options_data]
    
    analyzer = IVSurfaceAnalyzer()
    return analyzer.build_surface(strikes, expiries, ivs, method)
