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
    result["iv_momentum_5d"] = iv_panel.pct_change(5, fill_method=None).mean(axis=1)

    # IV Momentum (20日变化)
    result["iv_momentum_20d"] = iv_panel.pct_change(20, fill_method=None).mean(axis=1)

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


# ── 7. Put/Call Volume Ratio (PCVR) ──────────────────────────────────


def put_call_volume_ratio(
    option_data: pd.DataFrame,
    product_col: str = "product",
    date_col: str = "dt",
    cp_col: str = "cp",
    vol_col: str = "volume",
) -> pd.Series:
    """Put/Call 成交量比率: Σ(put_volume) / Σ(call_volume).

    > 1: put 成交量 > call (看跌情绪)
    < 1: call 成交量 > put (看涨情绪)
    极端值 (>1.5 或 <0.7) 通常是反转信号

    Args:
        option_data: 期权链数据 (长格式, 含 cp/volume 列)

    Returns:
        按日期聚合的 PCVR
    """
    if vol_col not in option_data.columns:
        return pd.Series(dtype=float)

    put_vol = option_data[option_data[cp_col].isin(["P", "put"])].groupby(date_col)[vol_col].sum()
    call_vol = option_data[option_data[cp_col].isin(["C", "call"])].groupby(date_col)[vol_col].sum()

    return put_vol / call_vol.replace(0, np.nan)


def put_call_volume_ratio_by_delta(
    option_data: pd.DataFrame,
    delta_col: str = "Delta",
    date_col: str = "dt",
    cp_col: str = "cp",
    vol_col: str = "volume",
    put_delta_range: tuple[float, float] = (-0.5, -0.15),
    call_delta_range: tuple[float, float] = (0.15, 0.5),
) -> pd.Series:
    """按 delta 区间的 Put/Call Volume Ratio.

    只计算 OTM 期权 (delta 在指定范围内) 的成交量比率,
    更精确地衡量方向性情绪.
    """
    if vol_col not in option_data.columns or delta_col not in option_data.columns:
        return pd.Series(dtype=float)

    puts = option_data[
        (option_data[cp_col].isin(["P", "put"]))
        & (option_data[delta_col] >= put_delta_range[0])
        & (option_data[delta_col] <= put_delta_range[1])
    ]
    calls = option_data[
        (option_data[cp_col].isin(["C", "call"]))
        & (option_data[delta_col] >= call_delta_range[0])
        & (option_data[delta_col] <= call_delta_range[1])
    ]

    put_vol = puts.groupby(date_col)[vol_col].sum()
    call_vol = calls.groupby(date_col)[vol_col].sum()

    return put_vol / call_vol.replace(0, np.nan)


# ── 8. Open Interest Change Rate (OI Delta) ───────────────────────────


def oi_change_rate(
    option_data: pd.DataFrame,
    date_col: str = "dt",
    cp_col: str = "cp",
    oi_col: str = "open_interest",
) -> pd.DataFrame:
    """Open Interest 日变化率.

    Returns:
        DataFrame 含:
        - total_oi_delta: 总 OI 变化
        - call_oi_delta: Call OI 变化
        - put_oi_delta: Put OI 变化
        - net_positioning: call_oi_delta - put_oi_delta (净方向性)
    """
    if oi_col not in option_data.columns:
        return pd.DataFrame()

    total_oi = option_data.groupby(date_col)[oi_col].sum()
    call_oi = option_data[option_data[cp_col].isin(["C", "call"])].groupby(date_col)[oi_col].sum()
    put_oi = option_data[option_data[cp_col].isin(["P", "put"])].groupby(date_col)[oi_col].sum()

    result = pd.DataFrame({
        "total_oi_delta": total_oi.diff(),
        "call_oi_delta": call_oi.diff(),
        "put_oi_delta": put_oi.diff(),
    })
    result["net_positioning"] = result["call_oi_delta"] - result["put_oi_delta"]
    return result


def oi_price_divergence(
    price_returns: pd.Series,
    oi_delta: pd.Series,
) -> pd.Series:
    """OI-价格背离信号.

    价格涨 + OI 增 = 新多头入场 (趋势确认)
    价格涨 + OI 减 = 空头平仓 (趋势减弱)
    价格跌 + OI 增 = 新空头入场 (趋势确认)
    价格跌 + OI 减 = 多头平仓 (趋势减弱)

    Returns:
        信号: +1=多头确认, -1=空头确认, 0=背离
    """
    price_up = (price_returns > 0).astype(int)
    oi_up = (oi_delta > 0).astype(int)

    signal = price_up * oi_up + (1 - price_up) * (1 - oi_up)
    return signal * 2 - 1  # 映射到 [-1, 1]


# ── 9. Net Dealer Gamma Positioning ────────────────────────────────────


def net_dealer_gamma(
    option_data: pd.DataFrame,
    gamma_col: str = "Gamma",
    oi_col: str = "open_interest",
    multiplier_col: str = "multiplier",
    spot_col: str = "f",
    strike_col: str = "k",
    cp_col: str = "cp",
    date_col: str = "dt",
) -> pd.DataFrame:
    """做市商净 Gamma 定位.

    正 GEX = 做市商做多 gamma → 标的稳定 (pin risk)
    负 GEX = 做市商做空 gamma → 标的波动 (gamma squeeze)

    Returns:
        DataFrame 含:
        - total_gex: 总 gamma exposure
        - call_gex: Call GEX
        - put_gex: Put GEX
        - net_gex: call_gex - |put_gex| (净方向性)
    """
    if gamma_col not in option_data.columns:
        return pd.DataFrame()

    cp_sign = option_data[cp_col].map({"C": 1, "P": -1, "call": 1, "put": -1}).fillna(1)

    gex_per_contract = (
        option_data[gamma_col]
        * option_data[oi_col].fillna(0)
        * option_data.get(multiplier_col, 1.0)
        * option_data[spot_col]
        * cp_sign
    )

    option_data = option_data.copy()
    option_data["_gex"] = gex_per_contract

    total_gex = option_data.groupby(date_col)["_gex"].sum()
    call_gex = option_data[option_data[cp_col].isin(["C", "call"])].groupby(date_col)["_gex"].sum()
    put_gex = option_data[option_data[cp_col].isin(["P", "put"])].groupby(date_col)["_gex"].sum()

    result = pd.DataFrame({
        "total_gex": total_gex,
        "call_gex": call_gex,
        "put_gex": put_gex,
    })
    result["net_gex"] = result["call_gex"] - result["put_gex"].abs()
    return result


def gex_flip_level(
    option_data: pd.DataFrame,
    gamma_col: str = "Gamma",
    oi_col: str = "open_interest",
    multiplier_col: str = "multiplier",
    strike_col: str = "k",
    cp_col: str = "cp",
) -> float | None:
    """GEX 翻转价位: GEX 从正变负的价格水平.

    在此价位附近, 做市商从 long gamma 变为 short gamma,
    标的趋向剧烈波动 (gamma squeeze 点).
    """
    if gamma_col not in option_data.columns:
        return None

    # 按 strike 聚合 GEX
    cp_sign = option_data[cp_col].map({"C": 1, "P": -1, "call": 1, "put": -1}).fillna(1)
    gex_by_strike = (
        option_data[gamma_col]
        * option_data[oi_col].fillna(0)
        * option_data.get(multiplier_col, 1.0)
        * cp_sign
    ).groupby(option_data[strike_col]).sum()

    if len(gex_by_strike) < 3:
        return None

    # 找 GEX 累积从正变负的价位
    cumulative = gex_by_strike.sort_index().cumsum()
    sign_changes = np.diff(np.sign(cumulative.values))
    flip_indices = np.where(sign_changes != 0)[0]

    if len(flip_indices) == 0:
        return None

    # 返回第一个翻转点
    idx = flip_indices[0]
    return float(cumulative.index[idx])


# ── 10. Theta/Gamma Ratio ─────────────────────────────────────────────


def theta_gamma_ratio(
    theta: float,
    gamma: float,
    spot: float,
    implied_vol: float,
) -> float:
    """单个期权的 Theta/Gamma 比率.

    TGR = |theta| / (0.5 × gamma × S² × σ² / 365)

    高 TGR → 卖方有利 (theta 收入 > gamma 风险)
    低 TGR → 买方有利 (gamma 利润 > theta 成本)

    Returns:
        TGR (无量纲)
    """
    if gamma <= 0 or spot <= 0 or implied_vol <= 0:
        return np.nan
    gamma_daily = 0.5 * gamma * spot ** 2 * implied_vol ** 2 / 365
    if gamma_daily <= 0:
        return np.nan
    return abs(theta) / gamma_daily


def portfolio_theta_gamma_ratio(
    option_data: pd.DataFrame,
    theta_col: str = "Theta",
    gamma_col: str = "Gamma",
    spot_col: str = "f",
    iv_col: str = "bs_iv",
    weight_col: str = "weight",
    multiplier_col: str = "multiplier",
    date_col: str = "dt",
) -> pd.Series:
    """组合 Theta/Gamma Ratio 时间序列.

    高值 → 组合 theta 收入相对 gamma 风险较高 → 卖方有利
    低值 → 组合 gamma 潜力相对 theta 成本较高 → 买方有利
    """
    if theta_col not in option_data.columns or gamma_col not in option_data.columns:
        return pd.Series(dtype=float)

    def _tgr(group):
        theta_total = (group[theta_col] * group[weight_col] * group.get(multiplier_col, 1.0)).sum()
        gamma_total = (group[gamma_col] * group[weight_col] * group.get(multiplier_col, 1.0)).sum()
        spot = group[spot_col].iloc[0] if spot_col in group.columns else 0
        iv = group[iv_col].iloc[0] if iv_col in group.columns else 0
        return theta_gamma_ratio(theta_total, gamma_total, spot, iv)

    return option_data.groupby(date_col).apply(_tgr)


# ── 11. Skew Z-Score / Skew Momentum ──────────────────────────────────


def skew_zscore(
    skew_panel: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Skew Z-Score: (current - rolling_mean) / rolling_std.

    > 2: skew 极端陡峭 (tail risk 恐慌) → 反向做多
    < -2: skew 极端平缓 (complacency) → 反向做空
    """
    mu = skew_panel.rolling(window, min_periods=60).mean()
    sigma = skew_panel.rolling(window, min_periods=60).std()
    return (skew_panel - mu) / sigma.replace(0, np.nan)


def skew_momentum(
    skew_panel: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 60,
) -> pd.DataFrame:
    """Skew 动量: short_ma - long_ma.

    > 0: skew 正在陡峭化 (风险厌恶增加)
    < 0: skew 正在平缓化 (风险偏好增加)
    """
    short_ma = skew_panel.rolling(short_window, min_periods=2).mean()
    long_ma = skew_panel.rolling(long_window, min_periods=10).mean()
    return short_ma - long_ma


def skew_percentile(
    skew_panel: pd.DataFrame,
    window: int = 252,
) -> pd.DataFrame:
    """Skew 分位数 (0~1)."""
    return skew_panel.rolling(window, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )


# ── 12. IV Term Structure Curvature ────────────────────────────────────


def iv_term_structure_curvature(
    iv_near: pd.Series,
    iv_mid: pd.Series,
    iv_far: pd.Series,
) -> pd.Series:
    """IV 期限结构曲率: iv_mid - (iv_near + iv_far) / 2.

    < 0: 中间月份便宜 (适合 calendar spread 入场)
    > 0: 中间月份贵 (适合 calendar spread 出场)
    """
    return iv_mid - (iv_near + iv_far) / 2


# ── 13. Realized Upside/Downside Vol Ratio ─────────────────────────────


def realized_vol_asymmetry(
    returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """已实现波动率不对称性.

    Returns:
        DataFrame 含:
        - vol_up: 上涨日已实现波动率
        - vol_down: 下跌日已实现波动率
        - vol_ratio: vol_down / vol_up (>1 = 下跌波动更大)
        - realized_skew: (vol_up - vol_down) / (vol_up + vol_down)
    """
    up_returns = returns.clip(lower=0)
    down_returns = returns.clip(upper=0)

    vol_up = up_returns.rolling(window, min_periods=10).apply(
        lambda x: x[x > 0].std() if (x > 0).sum() > 2 else np.nan, raw=True
    )
    vol_down = down_returns.abs().rolling(window, min_periods=10).apply(
        lambda x: x[x > 0].std() if (x > 0).sum() > 2 else np.nan, raw=True
    )

    vol_ratio = vol_down / vol_up.replace(0, np.nan)
    realized_skew = (vol_up - vol_down) / (vol_up + vol_down).replace(0, np.nan)

    return pd.DataFrame({
        "vol_up": vol_up,
        "vol_down": vol_down,
        "vol_ratio": vol_ratio,
        "realized_skew": realized_skew,
    })


# ── 14. VWAP-IV (Volume-Weighted IV) ──────────────────────────────────


def vwap_iv(
    option_data: pd.DataFrame,
    date_col: str = "dt",
    iv_col: str = "bs_iv",
    vol_col: str = "volume",
    product_col: str = "product",
) -> pd.Series:
    """成交量加权 IV (VWAP-IV).

    比单点 ATM IV 更稳健, 反映市场共识.
    """
    if iv_col not in option_data.columns or vol_col not in option_data.columns:
        return pd.Series(dtype=float)

    valid = option_data.dropna(subset=[iv_col, vol_col])
    valid = valid[valid[vol_col] > 0]

    valid = valid.copy()
    valid["_weighted_iv"] = valid[iv_col] * valid[vol_col]
    total_vol = valid.groupby(date_col)[vol_col].sum()
    weighted_sum = valid.groupby(date_col)["_weighted_iv"].sum()

    return weighted_sum / total_vol.replace(0, np.nan)


# ── 15. Cross-Product IV Spread ────────────────────────────────────────


def cross_product_iv_spread(
    iv_panel: pd.DataFrame,
    product_a: str,
    product_b: str,
) -> pd.Series:
    """跨品种 IV 价差: IV_A - IV_B.

    用于相关品种的相对价值交易 (如 IO vs MO, RB vs I).
    """
    if product_a not in iv_panel.columns or product_b not in iv_panel.columns:
        return pd.Series(dtype=float)
    return iv_panel[product_a] - iv_panel[product_b]


def cross_product_iv_spread_zscore(
    iv_panel: pd.DataFrame,
    product_a: str,
    product_b: str,
    window: int = 60,
) -> pd.Series:
    """跨品种 IV 价差 Z-Score."""
    spread = cross_product_iv_spread(iv_panel, product_a, product_b)
    mu = spread.rolling(window, min_periods=20).mean()
    sigma = spread.rolling(window, min_periods=20).std()
    return (spread - mu) / sigma.replace(0, np.nan)


# ── 16. Advanced Greek Factors (Color/Veta/Speed) ─────────────────────


def portfolio_color(
    option_data: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.Series:
    """组合 Color (Gamma 衰减): Σ(color × weight × multiplier).

    Color = d³V/dtdS², 衡量 gamma 随时间的衰减.
    正 color: gamma 随时间增加 (到期前 gamma spike)
    负 color: gamma 随时间减少
    """
    if "color" not in option_data.columns:
        return pd.Series(dtype=float)
    return (option_data["color"] * option_data[weight_col] * option_data.get("multiplier", 1.0)).groupby(
        option_data.get("dt", option_data.index)
    ).sum()


def portfolio_veta(
    option_data: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.Series:
    """组合 Veta (Vega 衰减): Σ(veta × weight × multiplier).

    Veta = d³V/dtdσ, 衡量 vega 随时间的衰减.
    用于 calendar spread timing — veta 大时近月 vega 衰减快.
    """
    if "veta" not in option_data.columns:
        return pd.Series(dtype=float)
    return (option_data["veta"] * option_data[weight_col] * option_data.get("multiplier", 1.0)).groupby(
        option_data.get("dt", option_data.index)
    ).sum()


def portfolio_speed(
    option_data: pd.DataFrame,
    weight_col: str = "weight",
) -> pd.Series:
    """组合 Speed (Gamma 凸性): Σ(speed × weight × multiplier).

    Speed = d³V/dS³, 衡量 gamma 对价格变化的敏感度.
    正 speed: gamma 随价格上涨而增加 (long gamma 持仓有利)
    负 speed: gamma 随价格上涨而减少
    """
    if "speed" not in option_data.columns:
        return pd.Series(dtype=float)
    return (option_data["speed"] * option_data[weight_col] * option_data.get("multiplier", 1.0)).groupby(
        option_data.get("dt", option_data.index)
    ).sum()
