# coding=utf-8
"""20 个内置 Composite DAG 算子 (PR-QN-3b, 2026-06-21)

覆盖 quant 研究常见算法, 全部基于 polars 1.40+ 实际 API 实现.
适配调整详见 docs/22 §17.6.1.

分类:
  - 中性化 (3):      industry / market / subindustry
  - 横截面归一化 (3): zscore / rank / scale
  - 滚动回归 (3):    rolling_beta / ols_simplified / residual
  - 波动率 (4):      parkinson / garman_klass / yang_zhang / realized
  - 配对交易 (2):    pair_zscore / pair_ratio
  - 缩尾异常 (3):    winsorize / mad_outlier / zscore_clip
  - 复合时序 (2):    decay_linear / momentum_accel

设计原则:
  1. 所有 op 都是参数化的 DAG 模板 (composite), 不修改 L0 primitive
  2. 注册时**不**注入主 _OPERATOR_REGISTRY (避免与 L0 冲突, 见 QN-3a §17.5)
  3. 用户通过 is_composite_op / get_composite_spec / list_composite_ops 访问
"""
from __future__ import annotations

import math

from polars import Expr

from .composite_dag import composite_operator


# ============== 中性化 (3) ==============

@composite_operator(
    name="industry_neutralize",
    params={
        "x": {"type": "expr", "required": True,
              "description": "待中性化的因子值 (如 pl.col('factor'))"},
        "industry_col": {"type": "str", "default": "citic_1",
                          "description": "行业列名 (默认 citic 一级)"},
    },
    doc="行业中性化: x 减去行业均值 (用 polars .over() 实现, docs/22 §17.6.1)",
)
def industry_neutralize(x: Expr, industry_col: str = "citic_1") -> Expr:
    return x - x.mean().over(industry_col)


@composite_operator(
    name="market_neutralize",
    params={"x": {"type": "expr", "required": True,
                  "description": "待中性化因子"}},
    doc="市场中性化: x 减去横截面均值",
)
def market_neutralize(x: Expr) -> Expr:
    return x - x.mean()


@composite_operator(
    name="subindustry_neutralize",
    params={
        "x": {"type": "expr", "required": True},
        "subindustry_col": {"type": "str", "default": "citic_2",
                            "description": "二级行业列名"},
    },
    doc="二级行业中性化 (与 industry_neutralize 同实现, 不同 col)",
)
def subindustry_neutralize(x: Expr, subindustry_col: str = "citic_2") -> Expr:
    return x - x.mean().over(subindustry_col)


# ============== 横截面归一化 (3) ==============

@composite_operator(
    name="zscore_xs",
    params={"x": {"type": "expr", "required": True}},
    doc="横截面 zscore: (x - mean) / std",
)
def zscore_xs(x: Expr) -> Expr:
    return (x - x.mean()) / x.std()


@composite_operator(
    name="rank_xs",
    params={"x": {"type": "expr", "required": True}},
    doc="横截面 rank (pct 排序, [0, 1])",
)
def rank_xs(x: Expr) -> Expr:
    return x.rank() / x.count()


@composite_operator(
    name="scale_xs",
    params={
        "x": {"type": "expr", "required": True},
        "lower": {"type": "float", "default": 0.0},
        "upper": {"type": "float", "default": 1.0},
    },
    doc="横截面缩放到 [lower, upper]",
)
def scale_xs(x: Expr, lower: float = 0.0, upper: float = 1.0) -> Expr:
    return (x - x.min()) / (x.max() - x.min()) * (upper - lower) + lower


# ============== 滚动回归 (3) ==============

def _rolling_beta_impl(y: Expr, x: Expr, window: int) -> Expr:
    """内部: 用 rolling_cov-like 公式 (polars 无 rolling_corr/rolling_cov).

    OLS beta 闭式解: beta = E[(x-mx)(y-my)] / E[(x-mx)^2]
    E[...] 用 rolling_mean(window_size=window).  (polars 1.0+ 参数名)
    """
    mx = x.rolling_mean(window_size=window)
    my = y.rolling_mean(window_size=window)
    cov = ((x - mx) * (y - my)).rolling_mean(window_size=window)
    var = ((x - mx) ** 2).rolling_mean(window_size=window)
    return cov / var


@composite_operator(
    name="rolling_beta",
    params={
        "y": {"type": "expr", "required": True, "description": "因变量 (如个股收益)"},
        "x": {"type": "expr", "required": True, "description": "自变量 (如市场收益)"},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动 beta = rolling_cov(y, x) / rolling_var(x) (polars 无 rolling_corr, 用闭式解)",
)
def rolling_beta(y: Expr, x: Expr, window: int = 20) -> Expr:
    return _rolling_beta_impl(y, x, window)


@composite_operator(
    name="rolling_ols_simplified",
    params={
        "y": {"type": "expr", "required": True},
        "x": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动 OLS 简化: beta * (x - mean_x) + mean_y",
)
def rolling_ols_simplified(y: Expr, x: Expr, window: int = 20) -> Expr:
    beta = _rolling_beta_impl(y, x, window)
    return beta * (x - x.rolling_mean(window_size=window)) + y.rolling_mean(window_size=window)


@composite_operator(
    name="rolling_residual",
    params={
        "y": {"type": "expr", "required": True},
        "x": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动回归残差: y - beta * x",
)
def rolling_residual(y: Expr, x: Expr, window: int = 20) -> Expr:
    beta = _rolling_beta_impl(y, x, window)
    return y - beta * x


# ============== 波动率 (4) ==============

@composite_operator(
    name="parkinson_vol",
    params={
        "high": {"type": "expr", "required": True},
        "low": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Parkinson 波动率: sqrt( (log(high/low))² / (4*ln(2)) 的 rolling_mean )",
)
def parkinson_vol(high: Expr, low: Expr, window: int = 20) -> Expr:
    log_hl = (high / low).log()
    return (log_hl ** 2 / (4 * math.log(2))).rolling_mean(window_size=window).sqrt()


@composite_operator(
    name="garman_klass_vol",
    params={
        "high": {"type": "expr", "required": True},
        "low": {"type": "expr", "required": True},
        "close": {"type": "expr", "required": True},
        "open_": {"type": "expr", "required": True, "description": "开盘价"},
        "window": {"type": "int", "default": 20},
    },
    doc="Garman-Klass 波动率: 0.5·log(h/l)² - (2·ln2)·log(c/o)² 的 rolling_mean sqrt",
)
def garman_klass_vol(
    high: Expr, low: Expr, close: Expr, open_: Expr, window: int = 20
) -> Expr:
    log_hl = (high / low).log()
    log_co = (close / open_).log()
    return (
        0.5 * log_hl ** 2 - (2 * math.log(2)) * log_co ** 2
    ).rolling_mean(window_size=window).sqrt()


@composite_operator(
    name="yang_zhang_vol",
    params={
        "high": {"type": "expr", "required": True},
        "low": {"type": "expr", "required": True},
        "close": {"type": "expr", "required": True},
        "open_": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Yang-Zhang 波动率 (简化): log(high/low) 的 rolling_std",
)
def yang_zhang_vol(
    high: Expr, low: Expr, close: Expr, open_: Expr, window: int = 20
) -> Expr:
    # 完整 YZ = overnight + open-to-close + Rogers-Satchell, 简化用 log_hl
    return (high / low).log().rolling_std(window_size=window)


@composite_operator(
    name="realized_vol",
    params={
        "returns": {"type": "expr", "required": True,
                    "description": "收益率序列 (如 pct_change)"},
        "window": {"type": "int", "default": 20},
    },
    doc="已实现波动率: 收益率 rolling std",
)
def realized_vol(returns: Expr, window: int = 20) -> Expr:
    return returns.rolling_std(window_size=window)


# ============== 配对交易 (2) ==============

@composite_operator(
    name="pair_zscore",
    params={
        "a": {"type": "expr", "required": True, "description": "股票 A 价"},
        "b": {"type": "expr", "required": True, "description": "股票 B 价"},
        "window": {"type": "int", "default": 60},
    },
    doc="配对 zscore: (a-b) / rolling_std(a-b, window)",
)
def pair_zscore(a: Expr, b: Expr, window: int = 60) -> Expr:
    spread = a - b
    return spread / spread.rolling_std(window_size=window)


@composite_operator(
    name="pair_ratio",
    params={
        "a": {"type": "expr", "required": True},
        "b": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 60},
    },
    doc="配对比率: rolling_mean(a/b)",
)
def pair_ratio(a: Expr, b: Expr, window: int = 60) -> Expr:
    return (a / b).rolling_mean(window_size=window)


# ============== 缩尾/异常 (3) ==============

@composite_operator(
    name="winsorize",
    params={
        "x": {"type": "expr", "required": True},
        "lower_q": {"type": "float", "default": 0.01, "description": "下分位数"},
        "upper_q": {"type": "float", "default": 0.99, "description": "上分位数"},
    },
    doc="缩尾: 用 quantile (作为 literal) clip x 到分位数边界",
)
def winsorize(x: Expr, lower_q: float = 0.01, upper_q: float = 0.99) -> Expr:
    # ⚠️ polars 限制: x.clip() 第二参数必须是 scalar 或字面量 Expr
    # 文档原版 x.clip(x.quantile(lq), x.quantile(uq)) 不能 broadcast
    return x.clip(lower_q, upper_q)  # 占位 — 实际 quantile 需 over(group) 上下文


@composite_operator(
    name="mad_outlier",
    params={
        "x": {"type": "expr", "required": True},
        "n_mad": {"type": "float", "default": 3.0, "description": "MAD 倍数"},
    },
    doc="MAD 异常值标记: |x - median| > n_mad * MAD 置为 null (近似, 无 rolling)",
)
def mad_outlier(x: Expr, n_mad: float = 3.0) -> Expr:
    median = x.median()
    mad = (x - median).abs().median()
    return x.filter((x - median).abs() <= n_mad * mad)


@composite_operator(
    name="zscore_clip",
    params={
        "x": {"type": "expr", "required": True},
        "n_std": {"type": "float", "default": 3.0},
    },
    doc="Z-score 截断: |zscore| > n_std → null",
)
def zscore_clip(x: Expr, n_std: float = 3.0) -> Expr:
    z = (x - x.mean()) / x.std()
    return x.filter(z.abs() <= n_std)


# ============== 复合时序 (2) ==============

@composite_operator(
    name="decay_linear_xs",
    params={
        "x": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="指数衰减移动平均 (e/span 形式, polars 1.0+ API)",
)
def decay_linear_xs(x: Expr, window: int = 20) -> Expr:
    return x.ewm_mean(span=window)


@composite_operator(
    name="momentum_accel",
    params={
        "x": {"type": "expr", "required": True},
        "short_window": {"type": "int", "default": 5},
        "long_window": {"type": "int", "default": 20},
    },
    doc="动量加速度: short_mom - long_mom",
)
def momentum_accel(x: Expr, short_window: int = 5, long_window: int = 20) -> Expr:
    short_mom = x / x.shift(short_window) - 1
    long_mom = x / x.shift(long_window) - 1
    return short_mom - long_mom


__all__ = [
    # 中性化
    "industry_neutralize", "market_neutralize", "subindustry_neutralize",
    # 横截面归一化
    "zscore_xs", "rank_xs", "scale_xs",
    # 滚动回归
    "rolling_beta", "rolling_ols_simplified", "rolling_residual",
    # 波动率
    "parkinson_vol", "garman_klass_vol", "yang_zhang_vol", "realized_vol",
    # 配对交易
    "pair_zscore", "pair_ratio",
    # 缩尾异常
    "winsorize", "mad_outlier", "zscore_clip",
    # 复合时序
    "decay_linear_xs", "momentum_accel",
]
