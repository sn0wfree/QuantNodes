# coding=utf-8
"""20 个内置 Composite DAG 算子 — pandas 镜像 (PR-QN-4, 2026-06-22)

与 composite_dag_ops.py 中的 polars 版同名, 用 engine="pandas" 注册到
_COMPOSITE_REGISTRY_PANDAS. 保留 polars 主路径, 为 LLM 提供 pandas 备选.

分类 (同 polars 版):
  - 中性化 (3):      industry_neutralize / market_neutralize / subindustry_neutralize
  - 横截面归一化 (3): zscore_xs / rank_xs / scale_xs
  - 滚动回归 (3):    rolling_beta / ols_simplified / residual
  - 波动率 (4):      parkinson / garman_klass / yang_zhang / realized
  - 配对交易 (2):    pair_zscore / pair_ratio
  - 缩尾异常 (3):    winsorize / mad_outlier / zscore_clip
  - 复合时序 (2):    decay_linear / momentum_accel

pandas 实现特点:
  - 输入: pd.DataFrame + 列名 (str), 输出: pd.Series
  - 用 groupby().transform() / rolling() / ewm() 等 pandas 原生 API
  - 与 polars 版同名, 通过 CompositeSpec.engine 字段区分
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .composite_dag import composite_operator


# ============== 中性化 (3) ==============

@composite_operator(
    name="industry_neutralize",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True, "description": "DataFrame"},
        "x_col": {"type": "str", "required": True, "description": "待中性化的因子列名"},
        "industry_col": {"type": "str", "default": "citic_1", "description": "行业列名"},
    },
    doc="行业中性化 (pandas): x 减去行业均值 (用 groupby().transform 实现)",
)
def industry_neutralize(df: pd.DataFrame, x_col: str, industry_col: str = "citic_1") -> pd.Series:
    return df[x_col] - df.groupby(industry_col)[x_col].transform("mean")


@composite_operator(
    name="market_neutralize",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
    },
    doc="市场中性化 (pandas): x 减去横截面均值",
)
def market_neutralize(df: pd.DataFrame, x_col: str) -> pd.Series:
    return df[x_col] - df[x_col].mean()


@composite_operator(
    name="subindustry_neutralize",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "subindustry_col": {"type": "str", "default": "citic_2", "description": "二级行业列名"},
    },
    doc="二级行业中性化 (pandas): x 减去二级行业均值",
)
def subindustry_neutralize(
    df: pd.DataFrame, x_col: str, subindustry_col: str = "citic_2",
) -> pd.Series:
    return df[x_col] - df.groupby(subindustry_col)[x_col].transform("mean")


# ============== 横截面归一化 (3) ==============

@composite_operator(
    name="zscore_xs",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
    },
    doc="横截面 zscore (pandas): (x - mean) / std",
)
def zscore_xs(df: pd.DataFrame, x_col: str) -> pd.Series:
    s = df[x_col]
    return (s - s.mean()) / s.std()


@composite_operator(
    name="rank_xs",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
    },
    doc="横截面 rank (pandas, pct 排序, [0, 1])",
)
def rank_xs(df: pd.DataFrame, x_col: str) -> pd.Series:
    s = df[x_col]
    return s.rank(pct=True)


@composite_operator(
    name="scale_xs",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "lower": {"type": "float", "default": 0.0},
        "upper": {"type": "float", "default": 1.0},
    },
    doc="横截面缩放到 [lower, upper] (pandas)",
)
def scale_xs(df: pd.DataFrame, x_col: str, lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    s = df[x_col]
    return (s - s.min()) / (s.max() - s.min()) * (upper - lower) + lower


# ============== 滚动回归 (3) ==============

def _rolling_beta_pd(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """内部: 滚动 beta = rolling_cov(y, x) / rolling_var(x) (pandas)."""
    mx = x.rolling(window).mean()
    my = y.rolling(window).mean()
    cov = ((x - mx) * (y - my)).rolling(window).mean()
    var = ((x - mx) ** 2).rolling(window).mean()
    return cov / var


@composite_operator(
    name="rolling_beta",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "y_col": {"type": "str", "required": True, "description": "因变量列名"},
        "x_col": {"type": "str", "required": True, "description": "自变量列名"},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动 beta (pandas): rolling_cov / rolling_var",
)
def rolling_beta(df: pd.DataFrame, y_col: str, x_col: str, window: int = 20) -> pd.Series:
    return _rolling_beta_pd(df[y_col], df[x_col], window)


@composite_operator(
    name="rolling_ols_simplified",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "y_col": {"type": "str", "required": True},
        "x_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动 OLS 简化 (pandas): beta * (x - mean_x) + mean_y",
)
def rolling_ols_simplified(df: pd.DataFrame, y_col: str, x_col: str, window: int = 20) -> pd.Series:
    y = df[y_col]
    x = df[x_col]
    beta = _rolling_beta_pd(y, x, window)
    return beta * (x - x.rolling(window).mean()) + y.rolling(window).mean()


@composite_operator(
    name="rolling_residual",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "y_col": {"type": "str", "required": True},
        "x_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动回归残差 (pandas): y - beta * x",
)
def rolling_residual(df: pd.DataFrame, y_col: str, x_col: str, window: int = 20) -> pd.Series:
    y = df[y_col]
    x = df[x_col]
    beta = _rolling_beta_pd(y, x, window)
    return y - beta * x


# ============== 波动率 (4) ==============

@composite_operator(
    name="parkinson_vol",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "high_col": {"type": "str", "required": True},
        "low_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Parkinson 波动率 (pandas): sqrt( (log(high/low))² / (4*ln(2)) rolling_mean )",
)
def parkinson_vol(df: pd.DataFrame, high_col: str, low_col: str, window: int = 20) -> pd.Series:
    log_hl = np.log(df[high_col] / df[low_col])
    return np.sqrt((log_hl ** 2 / (4 * math.log(2))).rolling(window).mean())


@composite_operator(
    name="garman_klass_vol",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "high_col": {"type": "str", "required": True},
        "low_col": {"type": "str", "required": True},
        "close_col": {"type": "str", "required": True},
        "open_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Garman-Klass 波动率 (pandas)",
)
def garman_klass_vol(
    df: pd.DataFrame, high_col: str, low_col: str,
    close_col: str, open_col: str, window: int = 20,
) -> pd.Series:
    log_hl = np.log(df[high_col] / df[low_col])
    log_co = np.log(df[close_col] / df[open_col])
    return np.sqrt(
        (0.5 * log_hl ** 2 - (2 * math.log(2)) * log_co ** 2).rolling(window).mean()
    )


@composite_operator(
    name="yang_zhang_vol",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "high_col": {"type": "str", "required": True},
        "low_col": {"type": "str", "required": True},
        "close_col": {"type": "str", "required": True},
        "open_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Yang-Zhang 波动率简化 (pandas): log(high/low) 的 rolling_std",
)
def yang_zhang_vol(
    df: pd.DataFrame, high_col: str, low_col: str,
    close_col: str, open_col: str, window: int = 20,
) -> pd.Series:
    return np.log(df[high_col] / df[low_col]).rolling(window).std()


@composite_operator(
    name="realized_vol",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "returns_col": {"type": "str", "required": True, "description": "收益率列名"},
        "window": {"type": "int", "default": 20},
    },
    doc="已实现波动率 (pandas): 收益率 rolling std",
)
def realized_vol(df: pd.DataFrame, returns_col: str, window: int = 20) -> pd.Series:
    return df[returns_col].rolling(window).std()


# ============== 配对交易 (2) ==============

@composite_operator(
    name="pair_zscore",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "a_col": {"type": "str", "required": True, "description": "股票 A 价列名"},
        "b_col": {"type": "str", "required": True, "description": "股票 B 价列名"},
        "window": {"type": "int", "default": 60},
    },
    doc="配对 zscore (pandas): (a-b) / rolling_std(a-b)",
)
def pair_zscore(df: pd.DataFrame, a_col: str, b_col: str, window: int = 60) -> pd.Series:
    spread = df[a_col] - df[b_col]
    return spread / spread.rolling(window).std()


@composite_operator(
    name="pair_ratio",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "a_col": {"type": "str", "required": True},
        "b_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 60},
    },
    doc="配对比率 (pandas): rolling_mean(a/b)",
)
def pair_ratio(df: pd.DataFrame, a_col: str, b_col: str, window: int = 60) -> pd.Series:
    return (df[a_col] / df[b_col]).rolling(window).mean()


# ============== 缩尾/异常 (3) ==============

@composite_operator(
    name="winsorize",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "lower_q": {"type": "float", "default": 0.01, "description": "下分位数"},
        "upper_q": {"type": "float", "default": 0.99, "description": "上分位数"},
    },
    doc="缩尾 (pandas): 用分位数 clip x 到边界",
)
def winsorize(
    df: pd.DataFrame, x_col: str, lower_q: float = 0.01, upper_q: float = 0.99,
) -> pd.Series:
    s = df[x_col]
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lo, hi)


@composite_operator(
    name="mad_outlier",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "n_mad": {"type": "float", "default": 3.0, "description": "MAD 倍数"},
    },
    doc="MAD 异常值标记 (pandas): |x - median| > n_mad * MAD 置为 NaN",
)
def mad_outlier(df: pd.DataFrame, x_col: str, n_mad: float = 3.0) -> pd.Series:
    s = df[x_col]
    median = s.median()
    mad = (s - median).abs().median()
    mask = (s - median).abs() <= n_mad * mad
    return s.where(mask)


@composite_operator(
    name="zscore_clip",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "n_std": {"type": "float", "default": 3.0},
    },
    doc="Z-score 截断 (pandas): |zscore| > n_std → NaN",
)
def zscore_clip(df: pd.DataFrame, x_col: str, n_std: float = 3.0) -> pd.Series:
    s = df[x_col]
    z = (s - s.mean()) / s.std()
    return s.where(z.abs() <= n_std)


# ============== 复合时序 (2) ==============

@composite_operator(
    name="decay_linear_xs",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="指数衰减移动平均 (pandas): ewm(span=window).mean()",
)
def decay_linear_xs(df: pd.DataFrame, x_col: str, window: int = 20) -> pd.Series:
    return df[x_col].ewm(span=window).mean()


@composite_operator(
    name="momentum_accel",
    engine="pandas",
    params={
        "df": {"type": "dataframe", "required": True},
        "x_col": {"type": "str", "required": True},
        "short_window": {"type": "int", "default": 5},
        "long_window": {"type": "int", "default": 20},
    },
    doc="动量加速度 (pandas): short_mom - long_mom",
)
def momentum_accel(
    df: pd.DataFrame, x_col: str, short_window: int = 5, long_window: int = 20,
) -> pd.Series:
    s = df[x_col]
    short_mom = s / s.shift(short_window) - 1
    long_mom = s / s.shift(long_window) - 1
    return short_mom - long_mom
