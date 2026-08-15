# coding=utf-8
"""期权专用因子库 — 缺失因子补全.

基于 docs/101 调研, 补充尚未实现的期权因子:
    - IV Z-Score
    - Vanna/Volga 组合因子
    - Charm 时序因子
    - IV Term Structure Slope
    - Skew Rolling Panel
    - Gamma Exposure (GEX)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── 1. IV Z-Score ──────────────────────────────────────────────────────


def iv_zscore(
    iv_panel: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """IV Z-Score: (IV - rolling_mean) / rolling_std.

    Args:
        iv_panel: IV 面板 (date × product)
        window: 滚动窗口

    Returns:
        Z-Score 面板 (date × product), >2=极贵, <-2=极便宜
    """
    mu = iv_panel.rolling(window, min_periods=20).mean()
    sigma = iv_panel.rolling(window, min_periods=20).std()
    return (iv_panel - mu) / sigma.replace(0, np.nan)


# ── 2. Vanna/Volga 组合因子 ────────────────────────────────────────────


def portfolio_vanna(
    option_data: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.Series:
    """组合 Vanna: Σ(vanna × weight × multiplier).

    Vanna = d²V/dSdσ, 衡量 vol 变化对 delta 的影响.
    正 vanna = vol 升时 delta 增加 (方向性风险).

    Args:
        option_data: 含 vanna/weight/multiplier 列的 DataFrame
        weight_col: 权重列名

    Returns:
        按日期聚合的 portfolio vanna
    """
    if "vanna" not in option_data.columns:
        return pd.Series(dtype=float)
    return (option_data["vanna"] * option_data[weight_col] * option_data.get("multiplier", 1.0)).groupby(
        option_data.get("dt", option_data.index)
    ).sum()


def portfolio_volga(
    option_data: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.Series:
    """组合 Volga (Vega Gamma): Σ(volga × weight × multiplier).

    Volga = d²V/dσ², 衡量 vega 对 vol 变化的敏感度.
    正 volga = vol 波动时 vega 增加 (有利).

    Args:
        option_data: 含 volga/weight/multiplier 列的 DataFrame
        weight_col: 权重列名

    Returns:
        按日期聚合的 portfolio volga
    """
    if "volga" not in option_data.columns:
        return pd.Series(dtype=float)
    return (option_data["volga"] * option_data[weight_col] * option_data.get("multiplier", 1.0)).groupby(
        option_data.get("dt", option_data.index)
    ).sum()


def vanna_volga_ratio(
    vanna: pd.Series,
    volga: pd.Series,
) -> pd.Series:
    """Vanna/Volga 比率.

    > 1: 方向性风险主导 (vanna)
    < 1: 波动率风险主导 (volga)
    ≈ 0: 均衡
    """
    return vanna / volga.replace(0, np.nan)


# ── 3. Charm 时序因子 ──────────────────────────────────────────────────


def portfolio_charm(
    option_data: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.Series:
    """组合 Charm: Σ(charm × weight × multiplier).

    Charm = d²V/dtdS, 衡量 delta 随时间的衰减.
    正 charm (long call): delta 随时间增加 (有利)
    负 charm (long put): delta 随时间减少 (不利)

    用途: 判断 delta 对冲频率 — charm 大时需要更频繁对冲.
    """
    if "charm" not in option_data.columns:
        return pd.Series(dtype=float)
    return (option_data["charm"] * option_data[weight_col] * option_data.get("multiplier", 1.0)).groupby(
        option_data.get("dt", option_data.index)
    ).sum()


def charm_decay_rate(
    charm: pd.Series,
    current_delta: pd.Series,
) -> pd.Series:
    """Charm 衰减率: |charm| / |delta|.

    衡量 delta 每天衰减的比例. 高值 = 需要频繁对冲.
    """
    return charm.abs() / current_delta.abs().replace(0, np.nan)


# ── 4. IV Term Structure Slope ────────────────────────────────────────


def iv_term_structure_slope(
    iv_near: pd.Series,
    iv_far: pd.Series,
) -> pd.Series:
    """IV 期限结构斜率: (iv_far - iv_near) / iv_near.

    > 0: contango (远月 IV > 近月) — 适合卖近买远
    < 0: backwardation (近月 IV > 远月) — 适合买近卖远
    """
    return (iv_far - iv_near) / iv_near.replace(0, np.nan)


def iv_term_structure_slope_panel(
    iv_panel: pd.DataFrame,
    near_product: str,
    far_product: str,
) -> pd.Series:
    """从 IV 面板计算期限结构斜率 (简化: 用两个产品近似)."""
    if near_product not in iv_panel.columns or far_product not in iv_panel.columns:
        return pd.Series(dtype=float)
    return iv_term_structure_slope(iv_panel[near_product], iv_panel[far_product])


# ── 5. Skew Rolling Panel ─────────────────────────────────────────────


def skew_rolling_panel(
    option_data: pd.DataFrame,
    product_col: str = "product",
    date_col: str = "dt",
    k_col: str = "k",
    iv_col: str = "bs_iv",
    spot_col: str = "f",
    window: int = 20,
) -> pd.DataFrame:
    """滚动 Skew 面板 (date × product).

    对每个品种每日计算 OLS(IV ~ log(K/S)) 斜率, 然后 rolling mean.

    Args:
        option_data: 期权链数据 (长格式)
        window: 滚动平均窗口

    Returns:
        Skew 面板 (date × product)
    """
    from scipy import stats

    skew_records = []
    for (dt, product), group in option_data.groupby([date_col, product_col]):
        if len(group) < 5:
            continue
        spot = group[spot_col].iloc[0]
        if spot <= 0:
            continue

        strikes = group[k_col].values.astype(float)
        ivs = group[iv_col].values.astype(float)
        log_moneyness = np.log(strikes / spot)
        valid = np.isfinite(ivs) & np.isfinite(log_moneyness)

        if valid.sum() < 5:
            continue

        slope, _, _, _, _ = stats.linregress(log_moneyness[valid], ivs[valid])
        skew_records.append({"date": dt, "product": product, "skew": slope})

    if not skew_records:
        return pd.DataFrame()

    skew_df = pd.DataFrame(skew_records)
    panel = skew_df.pivot_table(index="date", columns="product", values="skew")
    min_p = min(5, window)
    return panel.rolling(window, min_periods=min_p).mean()


# ── 6. Gamma Exposure (GEX) ───────────────────────────────────────────


def compute_gex(
    option_data: pd.DataFrame,
    spot_col: str = "f",
    gamma_col: str = "Gamma",
    oi_col: str = "open_interest",
    multiplier_col: str = "multiplier",
    sign_col: str = "cp",
) -> pd.Series:
    """Gamma Exposure (GEX): Σ(gamma × OI × multiplier × spot × sign).

    正 GEX: 做市商做多 gamma → 标的趋向稳定 (pin risk)
    负 GEX: 做市商做空 gamma → 标的趋向波动 (gamma squeeze)

    Args:
        option_data: 含 gamma/oi/multiplier/cp/f 列的 DataFrame
        sign_col: 列名, 值 "C"=+1, "P"=-1

    Returns:
        按日期聚合的 GEX
    """
    if gamma_col not in option_data.columns:
        return pd.Series(dtype=float)

    cp_sign = option_data[sign_col].map({"C": 1, "P": -1, "call": 1, "put": -1}).fillna(1)
    gex = (
        option_data[gamma_col]
        * option_data[oi_col].fillna(0)
        * option_data.get(multiplier_col, 1.0)
        * option_data[spot_col]
        * cp_sign
    )
    return gex.groupby(option_data.get("dt", option_data.index)).sum()


def gex_zscore(
    gex: pd.Series,
    window: int = 60,
) -> pd.Series:
    """GEX Z-Score.

    > 2: 极高 gamma 暴露 → 市场稳定 (pin risk)
    < -2: 极低/负 gamma 暴露 → gamma squeeze 风险
    """
    mu = gex.rolling(window, min_periods=20).mean()
    sigma = gex.rolling(window, min_periods=20).std()
    return (gex - mu) / sigma.replace(0, np.nan)


# ── 7. 综合期权因子面板 ────────────────────────────────────────────────


def build_option_factor_panel(
    iv_panel: pd.DataFrame,
    hv_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建综合期权因子面板.

    Args:
        iv_panel: IV 面板 (date × product)
        hv_panel: HV 面板 (date × product)

    Returns:
        DataFrame 含:
        - iv_zscore: IV Z-Score
        - iv_rank: IV 分位
        - vrp: IV - HV (如有 HV)
        - iv_momentum: IV 5日变化
        - iv_term_slope: IV 短期-长期斜率 (简化)
    """
    result = pd.DataFrame(index=iv_panel.index)

    # IV Z-Score (逐品种)
    zscores = iv_zscore(iv_panel, window=60)
    result["iv_zscore"] = zscores.mean(axis=1)

    # IV Rank (分位, 逐品种取均值)
    rank_df = iv_panel.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    result["iv_rank"] = rank_df.mean(axis=1)

    # IV Momentum (5日变化)
    result["iv_momentum_5d"] = iv_panel.pct_change(5).mean(axis=1)

    # IV Momentum (20日变化)
    result["iv_momentum_20d"] = iv_panel.pct_change(20).mean(axis=1)

    # IV Short-Long 斜率 (简化: 用 rolling std 近似)
    result["iv_spread"] = (
        iv_panel.rolling(5).mean().mean(axis=1)
        - iv_panel.rolling(60).mean().mean(axis=1)
    )

    # VRP (如有 HV)
    if hv_panel is not None:
        common = iv_panel.columns.intersection(hv_panel.columns)
        if len(common) > 0:
            vrp = iv_panel[common] - hv_panel[common]
            result["vrp_mean"] = vrp.mean(axis=1)
            result["vrp_rank"] = vrp.rolling(252, min_periods=60).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
            )

    return result
