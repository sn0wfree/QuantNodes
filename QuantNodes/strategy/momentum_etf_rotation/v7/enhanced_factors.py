# coding=utf-8
"""增强因子库 — 6 个新量价因子 + 3 个新宏观因子.

新增量价因子 (f12-f17):
  微观结构:
    12. Amihud 非流动性 (amihud_illiquidity)
    13. Realized Volatility (realized_volatility)
    14. Realized Skewness (realized_skewness)
    15. Max5 (max5)
    16. 52-Week High (52week_high)
    17. Idiosyncratic Volatility (idiosyncratic_volatility)

新增宏观因子 (f21-f23):
    21. 美元指数 (DXY) 对数收益率
    22. 实际利率 (Real Rate)
    23. VIX 恐慌指数

参考:
  - Amihud (2002) "Illiquidity and stock returns"
  - Ang et al. (2006) "The cross-section of volatility and expected returns"
  - Bali et al. (2011) "Maxing out" 
  - George & Hwang (2004) "The 52-week high and momentum investing"
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
HF_DIR = REPO / "data" / "high_freq_macro"


# ============================================================
# 量价因子 (f12-f17)
# ============================================================

def amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Amihud 非流动性: mean(|r| / volume).

    衡量单位成交量引起的价格变动，流动性风险指标。
    因子值越大，流动性越差。

    Args:
        close: 收盘价
        volume: 成交量
        window: 滚动窗口 (默认 20 天)

    Returns:
        pd.Series, Amihud 非流动性指标
    """
    returns = close.pct_change()
    abs_ret = returns.abs()
    # 避免除以零
    volume_safe = volume.replace(0, np.nan)
    illiq = abs_ret / volume_safe
    return illiq.rolling(window, min_periods=max(1, window // 2)).mean()


def realized_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """已实现波动率: std(daily_returns).

    衡量资产风险，波动率溢价指标。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 20 天)

    Returns:
        pd.Series, 已实现波动率
    """
    returns = close.pct_change()
    return returns.rolling(window, min_periods=max(1, window // 2)).std()


def realized_skewness(close: pd.Series, window: int = 20) -> pd.Series:
    """已实现偏度: skew(daily_returns).

    衡量尾部风险，偏度溢价指标。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 20 天)

    Returns:
        pd.Series, 已实现偏度
    """
    returns = close.pct_change()
    return returns.rolling(window, min_periods=max(1, window // 2)).skew()


def max5(close: pd.Series, window: int = 5) -> pd.Series:
    """Max5: 过去 window 天最大日收益.

    彩票效应，投资者偏好近期有极端正收益的资产。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 5 天)

    Returns:
        pd.Series, 最大日收益
    """
    returns = close.pct_change()
    return returns.rolling(window, min_periods=1).max()


def week52_high(close: pd.Series, window: int = 252) -> pd.Series:
    """52-Week High: close / max(close, window) - 1.

    锚定效应，投资者参考52周高点定价。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 252 天，约1年)

    Returns:
        pd.Series, 距52周高点的距离 (负值)
    """
    max_price = close.rolling(window, min_periods=max(1, window // 2)).max()
    return close / max_price - 1


def idiosyncratic_volatility(
    close: pd.Series,
    market_close: pd.Series,
    window: int = 60,
) -> pd.Series:
    """特质波动率: 回归残差的标准差.

    低波动溢价，特质波动率低的资产收益更高。

    Args:
        close: 资产收盘价
        market_close: 市场基准收盘价 (如沪深300)
        window: 滚动窗口 (默认 60 天)

    Returns:
        pd.Series, 特质波动率
    """
    ret = close.pct_change()
    mkt_ret = market_close.pct_change()

    # 滚动协方差和方差
    cov = ret.rolling(window, min_periods=max(1, window // 2)).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=max(1, window // 2)).var()

    # 滚动 beta
    beta = cov / var.replace(0, np.nan)

    # 残差
    resid = ret - beta * mkt_ret

    return resid.rolling(window, min_periods=max(1, window // 2)).std()


# ============================================================
# 量价因子计算引擎
# ============================================================

@dataclass
class EnhancedFactorConfig:
    """增强因子配置."""

    # Amihud
    amihud_window: int = 20

    # Realized Volatility
    rv_window: int = 20

    # Realized Skewness
    rs_window: int = 20

    # Max5
    max5_window: int = 5

    # 52-Week High
    high52_window: int = 252

    # Idiosyncratic Volatility
    idio_vol_window: int = 60

    # 因子名映射
    name_map: dict[str, str] = None

    def __post_init__(self):
        if self.name_map is None:
            self.name_map = {
                "f12_amihud": "Amihud非流动性",
                "f13_rv": "已实现波动率",
                "f14_rs": "已实现偏度",
                "f15_max5": "Max5",
                "f16_52w_high": "52周高点距离",
                "f17_idio_vol": "特质波动率",
            }


def compute_enhanced_factor(
    ohlc: pd.DataFrame,
    factor: str,
    market_close: pd.Series | None = None,
    cfg: EnhancedFactorConfig | None = None,
) -> pd.Series:
    """计算单只 ETF 单个增强因子.

    Args:
        ohlc: DataFrame, columns=[open, high, low, close, volume]
        factor: 因子名 (e.g. "f12_amihud")
        market_close: 市场基准收盘价 (f17_idio_vol 需要)
        cfg: 配置

    Returns:
        pd.Series, 因子值
    """
    cfg = cfg or EnhancedFactorConfig()
    close = ohlc["close"]
    volume = ohlc["volume"]

    if factor == "f12_amihud":
        return amihud_illiquidity(close, volume, cfg.amihud_window)
    elif factor == "f13_rv":
        return realized_volatility(close, cfg.rv_window)
    elif factor == "f14_rs":
        return realized_skewness(close, cfg.rs_window)
    elif factor == "f15_max5":
        return max5(close, cfg.max5_window)
    elif factor == "f16_52w_high":
        return week52_high(close, cfg.high52_window)
    elif factor == "f17_idio_vol":
        if market_close is None:
            raise ValueError("f17_idio_vol 需要 market_close 参数")
        return idiosyncratic_volatility(close, market_close, cfg.idio_vol_window)
    else:
        raise ValueError(f"未知增强因子: {factor}")


def compute_all_enhanced_factors(
    ohlc: pd.DataFrame,
    market_close: pd.Series | None = None,
    cfg: EnhancedFactorConfig | None = None,
) -> pd.DataFrame:
    """计算单只 ETF 全部 6 个增强因子.

    Args:
        ohlc: DataFrame, columns=[open, high, low, close, volume]
        market_close: 市场基准收盘价 (f17_idio_vol 需要)
        cfg: 配置

    Returns:
        DataFrame, index=date, columns=6 factors
    """
    cfg = cfg or EnhancedFactorConfig()
    factors = list(cfg.name_map.keys())
    out = pd.DataFrame(index=ohlc.index)
    for fac in factors:
        try:
            out[fac] = compute_enhanced_factor(ohlc, fac, market_close, cfg)
        except Exception as e:
            print(f"  [{fac}] 计算失败: {e}")
            out[fac] = np.nan
    return out


def compute_all_enhanced_factors_panel(
    ohlcv_panel: pd.DataFrame,
    market_close: pd.Series | None = None,
    cfg: EnhancedFactorConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """计算多只 ETF 全部 6 个增强因子.

    Args:
        ohlcv_panel: 多级 columns, (code, field) with field in {open,high,low,close,volume}
        market_close: 市场基准收盘价 (f17_idio_vol 需要)
        cfg: 配置

    Returns:
        dict, code → DataFrame of 6 factors
    """
    cfg = cfg or EnhancedFactorConfig()
    codes = sorted(set(ohlcv_panel.columns.get_level_values(0)))
    out = {}
    for code in codes:
        try:
            sub = ohlcv_panel[code].copy()
            sub = sub.dropna(how="all")
            if len(sub) < 252:
                continue
            out[code] = compute_all_enhanced_factors(sub, market_close, cfg)
        except Exception as e:
            print(f"  [{code}] 增强因子计算失败: {e}")
            continue
    return out


# ============================================================
# 宏观因子加载 (f21-f23)
# ============================================================

def load_dxy_factor() -> pd.DataFrame:
    """加载美元指数 (DXY) 日度数据.

    Returns:
        DataFrame, index=date, columns=['dxy']
    """
    cache = HF_DIR / "macro_dxy_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raise FileNotFoundError(f"DXY 缓存不存在: {cache}")


def load_vix_factor() -> pd.DataFrame:
    """加载 VIX 恐慌指数日度数据.

    Returns:
        DataFrame, index=date, columns=['vix']
    """
    cache = HF_DIR / "macro_vix_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raise FileNotFoundError(f"VIX 缓存不存在: {cache}")


def load_real_rate_factor() -> pd.DataFrame:
    """加载实际利率月度数据.

    Returns:
        DataFrame, index=date, columns=['real_rate']
    """
    cache = HF_DIR / "macro_real_rate_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    raise FileNotFoundError(f"实际利率缓存不存在: {cache}")


def load_enhanced_macro_factors() -> pd.DataFrame:
    """加载全部增强宏观因子, 对齐到周频.

    Returns:
        DataFrame, index=weekly_date, columns=['dxy_logret', 'vix', 'real_rate']
    """
    # 加载日度数据
    dxy = load_dxy_factor()
    vix = load_vix_factor()
    real_rate = load_real_rate_factor()

    # 转换为对数收益率 (DXY) 或直接使用 (VIX, 实际利率)
    dxy_logret = np.log(dxy / dxy.shift(1))
    dxy_logret.columns = ["dxy_logret"]

    # 对齐到周频 (取周末值)
    dxy_weekly = dxy_logret.resample("W").last()
    vix_weekly = vix.resample("W").last()
    # 实际利率是月频，前向填充到周频
    real_rate_weekly = real_rate.resample("W").ffill()

    # 合并
    merged = pd.concat([dxy_weekly, vix_weekly, real_rate_weekly], axis=1)

    # 对齐列名
    merged.columns = ["dxy_logret", "vix", "real_rate"]

    return merged


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 量价因子函数
    "amihud_illiquidity",
    "realized_volatility",
    "realized_skewness",
    "max5",
    "week52_high",
    "idiosyncratic_volatility",
    # 配置和计算引擎
    "EnhancedFactorConfig",
    "compute_enhanced_factor",
    "compute_all_enhanced_factors",
    "compute_all_enhanced_factors_panel",
    # 宏观因子加载
    "load_dxy_factor",
    "load_vix_factor",
    "load_real_rate_factor",
    "load_enhanced_macro_factors",
]
