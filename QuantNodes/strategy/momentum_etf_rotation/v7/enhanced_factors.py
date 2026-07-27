# coding=utf-8
"""增强因子库 — 6 个新量价因子 + 5 个新宏观因子.

新增量价因子 (f12-f17):
  微观结构:
    12. Amihud 非流动性 (amihud_illiquidity)
    13. Realized Volatility (realized_volatility)
    14. Realized Skewness (realized_skewness)
    15. Max5 (max5)
    16. 52-Week High (52week_high)
    17. Idiosyncratic Volatility (idiosyncratic_volatility)

新增宏观因子:
    美元指数 (DXY) 对数收益率
    实际利率 (Real Rate) + diff + rank20
    VIX 恐慌指数 (pct_change) + rank20
    TF dummy (趋势过滤)
    中美利差 CN_US_SPREAD (CN_10Y - US_10Y)
    黄金原油收益率相关性 (GOLD_OIL_CORR, 20日滚动)

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
    """Amihud 非流动性: mean(|r| / amount), 取对数.

    衡量单位成交金额引起的价格变动，流动性风险指标。
    因子值越大，流动性越差。

    Args:
        close: 收盘价
        volume: 成交量
        window: 滚动窗口 (默认 20 天)

    Returns:
        pd.Series, Amihud 非流动性指标 (对数值)
    """
    returns = close.pct_change()
    abs_ret = returns.abs()
    # 用成交金额代替成交量
    amount = close * volume
    amount_safe = amount.replace(0, np.nan)
    illiq = abs_ret / amount_safe
    # 处理 inf 值 (当 abs_ret > 0 但 amount = 0 时)
    illiq = illiq.replace([np.inf, -np.inf], np.nan)
    # 取对数避免数值过小
    log_illiq = np.log(illiq.replace(0, np.nan))
    return log_illiq.rolling(window, min_periods=max(1, window // 2)).mean()


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
    # 对齐数据
    aligned = pd.concat([close, market_close], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(np.nan, index=close.index)

    ret = aligned.iloc[:, 0].pct_change()
    mkt_ret = aligned.iloc[:, 1].pct_change()

    # 滚动协方差和方差
    cov = ret.rolling(window, min_periods=max(1, window // 2)).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=max(1, window // 2)).var()

    # 滚动 beta
    beta = cov / var.replace(0, np.nan)

    # 残差
    resid = ret - beta * mkt_ret

    # 计算特质波动率
    idio_vol = resid.rolling(window, min_periods=max(1, window // 2)).std()

    # 对齐回原始索引
    return idio_vol.reindex(close.index)


def momentum_return(close: pd.Series, window: int = 20) -> pd.Series:
    """经典动量因子: pct_change(window).

    捕捉动量效应，window=5 为短期反转，window=20 为中期动量，window=60 为长期动量。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 20 天)

    Returns:
        pd.Series, 动量收益率
    """
    return close.pct_change(window)


def short_term_reversal(close: pd.Series, window: int = 5) -> pd.Series:
    """短期反转因子: -pct_change(window).

    捕捉短期反转效应，过去涨幅大的资产未来可能下跌。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 5 天)

    Returns:
        pd.Series, 短期反转因子 (取反)
    """
    return -close.pct_change(window)


def rsi_indicator(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI 技术指标: 相对强弱指数.

    衡量超买超卖，RSI > 70 超买，RSI < 30 超卖。

    Args:
        close: 收盘价
        window: 滚动窗口 (默认 14 天)

    Returns:
        pd.Series, RSI 值 [0, 100]
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window, min_periods=max(1, window // 2)).mean()
    avg_loss = loss.rolling(window, min_periods=max(1, window // 2)).mean()

    # 当 avg_loss = 0 时 (单调上涨)，RS = inf，RSI = 100
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rs = rs.fillna(np.inf)
    rsi = 100 - (100 / (1 + rs))

    return rsi


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

    # 经典动量
    momentum_short_window: int = 5   # 短期反转
    momentum_mid_window: int = 20    # 中期动量
    momentum_long_window: int = 60   # 长期动量

    # RSI
    rsi_window: int = 14

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
                "f18_mom_short": "短期动量(5日)",
                "f19_mom_mid": "中期动量(20日)",
                "f20_mom_long": "长期动量(60日)",
                "f21_reversal": "短期反转(5日)",
                "f22_rsi": "RSI(14日)",
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
    elif factor == "f18_mom_short":
        return momentum_return(close, cfg.momentum_short_window)
    elif factor == "f19_mom_mid":
        return momentum_return(close, cfg.momentum_mid_window)
    elif factor == "f20_mom_long":
        return momentum_return(close, cfg.momentum_long_window)
    elif factor == "f21_reversal":
        return short_term_reversal(close, cfg.momentum_short_window)
    elif factor == "f22_rsi":
        return rsi_indicator(close, cfg.rsi_window)
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

    优先使用 QuantNodes/data/美元指数.xlsx (1971-2026, 14224 条)，
    否则回退到 data/high_freq_macro/macro_dxy_daily.parquet。

    Returns:
        DataFrame, index=date, columns=['dxy']
    """
    # 优先使用新数据源
    dxy_xlsx = REPO / "data" / "美元指数.xlsx"
    cache_v2 = HF_DIR / "macro_dxy_daily_v2.parquet"
    cache_v1 = HF_DIR / "macro_dxy_daily.parquet"

    if cache_v2.exists():
        return pd.read_parquet(cache_v2)

    if dxy_xlsx.exists():
        dxy_raw = pd.read_excel(dxy_xlsx, skiprows=7, header=0)
        dxy_raw.columns = ['date', 'dxy']
        dxy_raw['date'] = pd.to_datetime(dxy_raw['date'])
        dxy_raw = dxy_raw.set_index('date').sort_index()
        dxy = dxy_raw.dropna()
        dxy.to_parquet(cache_v2)
        return dxy

    if cache_v1.exists():
        return pd.read_parquet(cache_v1)

    raise FileNotFoundError("DXY 数据不存在")


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
    """加载实际利率日频数据 (DFII10, 10Y TIPS yield).

    优先使用缓存的 daily parquet，否则从 FRED API 获取 DFII10。

    Returns:
        DataFrame, index=date, columns=['real_rate'], 日频
    """
    cache = HF_DIR / "macro_real_rate_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    # 从 FRED API 获取 DFII10 (10-Year TIPS yield, daily)
    from fredapi import Fred
    fred = Fred(api_key="7ee6f74caae5aa717a7e849fb14b055e")
    tips = fred.get_series("DFII10")
    tips.index.name = "date"
    df = tips.to_frame("real_rate").dropna()
    df.to_parquet(cache)
    return df


def load_cn_us_spread() -> pd.DataFrame:
    """加载中美10年期国债利差 (日频).

    CN_10Y - US_10Y, 单位: 百分点.

    Returns:
        DataFrame, index=date, columns=['cn_us_spread']
    """
    cache = HF_DIR / "cn_us_spread_10y.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("date")[["cn_us_spread"]]
    raise FileNotFoundError(f"中美利差缓存不存在: {cache}")


def load_gold_oil_correlation() -> pd.DataFrame:
    """加载黄金原油收益率20日滚动相关系数 (日频).

    沪金指数 vs 布伦特原油, 20日滚动Pearson相关系数.

    Returns:
        DataFrame, index=date, columns=['gold_oil_corr']
    """
    cache = HF_DIR / "gold_oil_corr.parquet"
    if cache.exists():
        return pd.read_parquet(cache).set_index("date")[["gold_oil_corr"]]
    raise FileNotFoundError(f"黄金原油相关性缓存不存在: {cache}")


def load_trend_filter_dummy(ma_window: int = 200) -> pd.Series:
    """加载趋势过滤 dummy 变量 (日频).

    当 benchmark < MA(ma_window) 时为 1 (熊市), 否则为 0 (牛市).

    Args:
        ma_window: 移动平均窗口 (默认 200 日)

    Returns:
        pd.Series, 日频, name="tf_dummy", 值域 {0, 1}
    """
    benchmark_path = HF_DIR / "v9_benchmark_沪深300.parquet"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark 数据不存在: {benchmark_path}")

    benchmark = pd.read_parquet(benchmark_path)
    if "沪深300指数" in benchmark.columns:
        price = benchmark["沪深300指数"]
    else:
        price = benchmark.iloc[:, 0]

    # 计算移动平均
    ma = price.rolling(window=ma_window, min_periods=ma_window).mean()

    # 创建 dummy: 1 if price < MA, 0 otherwise
    # 前 ma_window-1 天为 NaN (数据不足)
    dummy = (price < ma).astype(float)
    dummy.name = "tf_dummy"

    # 将 NaN (数据不足) 填充为 0 (默认牛市)
    dummy = dummy.fillna(0)

    return dummy


def load_enhanced_macro_factors() -> pd.DataFrame:
    """加载全部增强宏观因子, 对齐到周频.

    Returns:
        DataFrame, index=weekly_date, columns=[
            'dxy_logret', 'vix', 'vix_rank20',
            'real_rate', 'real_rate_diff', 'real_rate_rank20',
            'tf_dummy',
            'cn_us_spread', 'gold_oil_corr'
        ]
    """
    # 加载日度数据
    dxy = load_dxy_factor()
    vix = load_vix_factor()
    real_rate = load_real_rate_factor()    # 日频 DFII10
    tf_dummy = load_trend_filter_dummy()   # 日频
    cn_us_spread = load_cn_us_spread()    # 日频
    gold_oil_corr = load_gold_oil_correlation()  # 日频

    # --- DXY: 先 resample 到周频 (周五价格), 再算 log return ---
    dxy_weekly_price = dxy.resample("W").last()
    dxy_weekly = np.log(dxy_weekly_price / dxy_weekly_price.shift(1))
    dxy_weekly.columns = ["dxy_logret"]

    # --- VIX: log return of weekly level ---
    vix_weekly = vix.resample("W").last()
    vix_logret = np.log(vix_weekly / vix_weekly.shift(1))
    vix_logret.columns = ["vix"]

    # VIX 时序 rank (过去20周的排名归一化到 [0,1])
    vix_rank20 = vix_logret.rolling(20, min_periods=10).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x)
    )
    vix_rank20.columns = ["vix_rank20"]

    # --- Real Rate: 日频 → 周频 Friday close → diff ---
    real_rate_weekly = real_rate.resample("W").last()
    real_rate_diff = real_rate_weekly.diff()
    real_rate_diff.columns = ["real_rate_diff"]

    # 实际利率时序 rank (过去20周的排名归一化到 [0,1])
    real_rate_rank20 = real_rate_weekly.rolling(20, min_periods=10).apply(
        lambda x: pd.Series(x).rank().iloc[-1] / len(x)
    )
    real_rate_rank20.columns = ["real_rate_rank20"]

    # --- TF dummy, CN-US spread, Gold-Oil corr: 日频 → 周频 Friday ---
    tf_dummy_weekly = tf_dummy.resample("W").last()
    tf_dummy_weekly.name = "tf_dummy"
    cn_us_spread_weekly = cn_us_spread.resample("W").last()
    gold_oil_corr_weekly = gold_oil_corr.resample("W").last()

    # 合并
    merged = pd.concat([
        dxy_weekly, vix_logret, vix_rank20,
        real_rate_weekly, real_rate_diff, real_rate_rank20,
        tf_dummy_weekly,
        cn_us_spread_weekly, gold_oil_corr_weekly
    ], axis=1)

    # 对齐列名
    merged.columns = [
        "dxy_logret", "vix", "vix_rank20",
        "real_rate", "real_rate_diff", "real_rate_rank20",
        "tf_dummy",
        "cn_us_spread", "gold_oil_corr"
    ]

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
    "load_trend_filter_dummy",
    "load_cn_us_spread",
    "load_gold_oil_correlation",
    "load_enhanced_macro_factors",
]
