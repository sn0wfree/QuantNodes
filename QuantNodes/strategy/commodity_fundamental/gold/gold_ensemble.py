# coding=utf-8
"""gold_ensemble.py — 黄金集成策略.

从 gold_trend_bnh.py 拆分出的集成策略模块:
    - compute_adaptive_ensemble: 多信号自适应集成 (先加权后风控)
    - compute_layered_ensemble_signal: 分层集成 (先风控后加权) — 生产函数

集成信号:
    1. macro_regime (滞回带趋势 × GVZ自适应 × 实际利率门控)
    2. real_rate (TIPS MA信号)
    3. momentum (52周动量)
    4. breakout (20日高点+3%止损, 可选)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .gold_utils import (
    load_gold_price_d,
    load_indicator_d,
)


def _rolling_calmar(
    close: pd.Series,
    position: pd.Series,
    window: int = 126,
) -> pd.Series:
    """基于仓位的滚动Calmar (年化收益/|MaxDD|)."""
    sr = (position.shift(1) * close.pct_change()).dropna()
    if len(sr) < window:
        return pd.Series(0, index=close.index)
    nav = (1 + sr).cumprod()
    rolling_max = nav.rolling(window, min_periods=window // 2).max()
    dd = nav / rolling_max - 1
    rolling_dd_min = dd.rolling(window, min_periods=window // 2).min()
    mean_ret = sr.rolling(window, min_periods=window // 2).mean() * 252
    cal = mean_ret / rolling_dd_min.abs().replace(0, np.nan)
    return cal.fillna(0)


def compute_adaptive_ensemble(
    close: pd.Series = None,
    vix: pd.Series = None,
    gvz: pd.Series = None,
    tips: pd.Series = None,
    macro_weight: float = 0.40,
    real_rate_weight: float = 0.30,
    momentum_weight: float = 0.30,
    weight_method: str = "fixed",
    adaptive_lookback: int = 126,
    adaptive_temp: float = 1.0,
    use_risk_overlay: bool = True,
    hard_cap: float = -0.15,
    soft_cap: float = -0.12,
    daily_cap: float = -0.03,
    vol_cap: float = 0.20,
    cooldown_days: int = 5,
    vol_window: int = 20,
) -> pd.DataFrame:
    """多信号自适应集成 + 风控覆盖.

    集成信号:
      1. macro_regime (滞回带趋势 × GVZ自适应 × 实际利率门控)
      2. real_rate (TIPS MA信号)
      3. momentum (52周动量)

    加权方式:
      "fixed": 固定权重
      "adaptive": 滚动Calmar动态权重

    返回: DataFrame [close, position]
    """
    from .gold_trend_bnh import compute_macro_regime_signal, compute_risk_overlay
    from .gold_real_rate import compute_real_rate_signal
    from .gold_momentum import compute_momentum_signal

    if close is None:
        close = load_gold_price_d()

    if vix is None:
        vix = load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    if gvz is None:
        gvz = load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()
    if tips is None:
        tips = load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()

    # 获取3个信号 (weekly → daily resample)
    macro_df = compute_macro_regime_signal()
    macro_pos = macro_df["position"].reindex(close.index).ffill().fillna(0)

    rr_df = compute_real_rate_signal(variant="tips_ma")
    rr_pos = rr_df["position"].reindex(close.index).ffill().fillna(0)

    mom_df = compute_momentum_signal(variant="long_only")
    mom_pos = mom_df["position"].reindex(close.index).ffill().fillna(0)

    signals = {"macro": macro_pos, "real_rate": rr_pos, "momentum": mom_pos}
    weights = {"macro": macro_weight, "real_rate": real_rate_weight,
               "momentum": momentum_weight}

    if weight_method == "fixed":
        combined = sum(weights[name] * pos for name, pos in signals.items())
    elif weight_method == "adaptive":
        cal = {name: _rolling_calmar(close, pos, adaptive_lookback)
               for name, pos in signals.items()}
        cal_df = pd.DataFrame(cal)
        w_raw = np.exp(cal_df * adaptive_temp)
        w_raw.iloc[:adaptive_lookback] = 1.0
        w_sum = w_raw.sum(axis=1).replace(0, np.nan)
        w_norm = w_raw.div(w_sum, axis=0).fillna(1 / 3)

        combined = pd.Series(0.0, index=close.index)
        for name in signals:
            combined = combined.add(w_norm[name] * signals[name], fill_value=0)
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")

    combined = combined.clip(0, 1.2)

    # 应用风控覆盖
    if use_risk_overlay:
        combined = compute_risk_overlay(
            close=close, position=combined, vix=vix, gvz=gvz,
            hard_cap=hard_cap, soft_cap=soft_cap, daily_cap=daily_cap,
            vol_cap=vol_cap, cooldown_days=cooldown_days, vol_window=vol_window,
        )

    return pd.DataFrame({"close": close, "position": combined})


def compute_layered_ensemble_signal(
    close: pd.Series = None,
    vix: pd.Series = None,
    gvz: pd.Series = None,
    tips: pd.Series = None,
    use_risk_overlay: bool = True,
    hard_cap: float = -0.10,
    soft_cap: float = -0.07,
    daily_cap: float = -0.02,
    vol_cap: float = 0.15,
    cooldown_days: int = 5,
    vol_window: int = 20,
    breakout_weight: float = 0.2,
    deadband: float = 0.05,
) -> pd.DataFrame:
    """分层集成信号 (Layered Ensemble) — 各信号独立风控后再加权.

    与 compute_adaptive_ensemble 的区别:
      adaptive: 先加权 → 再风控
      layered:  先各自风控 → 再加权 (生产函数)

    集成方式:
      - 三路基础 (默认等权): macro + real_rate + momentum
      - 第4路突破 (可选): gold_breakout, 永久小仓位
      - 权重归一: 三路均分(1-breakout_weight), 第4路固定breakout_weight

    返回: DataFrame [close, open, position]
    """
    from .gold_trend_bnh import (
        compute_macro_regime_signal,
        compute_risk_overlay,
        load_gold_price_d_with_open,
    )
    from .gold_real_rate import compute_real_rate_signal
    from .gold_momentum import compute_momentum_signal
    from .gold_breakout import compute_breakout_signal

    if close is None:
        price_df = load_gold_price_d_with_open()
        close = price_df["close"]
        open_price = price_df["open"]
    else:
        open_price = close

    # 因子shift2 (更保守)
    if vix is None:
        vix = load_indicator_d("vix", "gold_fred").reindex(close.index).ffill().shift(2)
    if gvz is None:
        gvz = load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill().shift(2)
    if tips is None:
        tips = load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill().shift(2)

    # 获取4个信号
    macro_df = compute_macro_regime_signal(vix=vix, gvz=gvz, tips=tips)
    macro_pos = macro_df["position"].reindex(close.index).ffill().fillna(0)

    rr_df = compute_real_rate_signal(variant="tips_ma", tips_daily=tips)
    rr_pos = rr_df["position"].reindex(close.index).ffill().fillna(0)

    mom_df = compute_momentum_signal(variant="long_only")
    mom_pos = mom_df["position"].reindex(close.index).ffill().fillna(0)

    bo_df = compute_breakout_signal()
    bo_pos = bo_df["position"].reindex(close.index).fillna(0)

    # 各信号独立风控
    if use_risk_overlay:
        macro_ov = compute_risk_overlay(
            close=close, position=macro_pos, vix=vix, gvz=gvz,
            hard_cap=hard_cap, soft_cap=soft_cap, daily_cap=daily_cap,
            vol_cap=vol_cap, cooldown_days=cooldown_days, vol_window=vol_window,
        )
        rr_ov = compute_risk_overlay(
            close=close, position=rr_pos, vix=vix, gvz=gvz,
            hard_cap=hard_cap, soft_cap=soft_cap, daily_cap=daily_cap,
            vol_cap=vol_cap, cooldown_days=cooldown_days, vol_window=vol_window,
        )
        mom_ov = compute_risk_overlay(
            close=close, position=mom_pos, vix=vix, gvz=gvz,
            hard_cap=hard_cap, soft_cap=soft_cap, daily_cap=daily_cap,
            vol_cap=vol_cap, cooldown_days=cooldown_days, vol_window=vol_window,
        )
        bo_ov = compute_risk_overlay(
            close=close, position=bo_pos, vix=vix, gvz=gvz,
            hard_cap=hard_cap, soft_cap=soft_cap, daily_cap=daily_cap,
            vol_cap=vol_cap, cooldown_days=cooldown_days, vol_window=vol_window,
        ) if breakout_weight > 0 else None
    else:
        bo_ov = bo_pos if breakout_weight > 0 else None
        macro_ov, rr_ov, mom_ov = macro_pos, rr_pos, mom_pos

    # 加权平均
    if breakout_weight > 0 and bo_ov is not None:
        base_weight = (1 - breakout_weight) / 3
        position = (base_weight * macro_ov + base_weight * rr_ov + base_weight * mom_ov
                    + breakout_weight * bo_ov).clip(0, 1.2)
    else:
        position = ((macro_ov + rr_ov + mom_ov) / 3).clip(0, 1.2)

    # 死带平滑
    if deadband > 0:
        for i in range(1, len(position)):
            if abs(position.iloc[i] - position.iloc[i-1]) < deadband:
                position.iloc[i] = position.iloc[i-1]

    return pd.DataFrame({"close": close, "open": open_price, "position": position})


__all__ = [
    "compute_adaptive_ensemble",
    "compute_layered_ensemble_signal",
]
