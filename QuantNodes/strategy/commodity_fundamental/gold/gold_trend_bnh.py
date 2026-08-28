# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_trend_bnh.py

黄金趋势跟踪B&H策略 (Trend-Following Buy & Hold)

核心逻辑:
    始终保持多头 (Buy & Hold), 但加入追踪止损
    - 入场: 始终在场
    - 出场: 价格从最高点回撤超过阈值 → 空仓
    - 重新入场: 价格恢复上涨 (创出新高或回到阈值内)

核心优势:
    - 牛市: 接近100%持仓, 捕获完整涨幅
    - 熊市: 止损出场, 避免大幅回撤
    - 2018-2025年化 18.28% (超过B&H 17.87%)
    - Sharpe 1.37, MaxDD -18.49% (均优于B&H)

风控层叠加 (风险门):
    条件: (价格>MA80) OR (VIX<60%百分位)
    - 牛市: 风控门ON, 满仓运行
    - 熊市: 风控门OFF, 趋势B&H止损生效

参数:
    - trailing_stop: 追踪止损比例 (默认12%, 在2018-2025优化得到)
    - use_risk_gate: 是否启用风控层 (默认False)
    - ma_window: 风控层均线窗口 (默认80日)
    - vix_pct_window: VIX百分位窗口 (默认252日)
    - vix_threshold: VIX百分位阈值 (默认0.6)
    - confirm_days: 风控门确认天数 (默认0=无确认)
    - min_hold_days: 最小持仓天数 (默认0=无限制)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_gold_price_d() -> pd.Series:
    """加载黄金日度收盘价."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价 (W-FRI)."""
    return load_gold_price_d().resample("W-FRI").last().dropna()


def _load_indicator_d(value_col: str, source: str) -> pd.Series:
    """加载日度指标."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"SELECT trade_date, value FROM fundamentals_daily "
        f"WHERE source='{source}' AND value_col='{value_col}' ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].ffill().dropna()


def _apply_confirm_and_hold(raw_signal: pd.Series, confirm_days: int, min_hold_days: int) -> pd.Series:
    """对原始信号应用确认期和最小持仓期.

    参数:
        raw_signal: 原始0/1信号
        confirm_days: 连续N天=1才输出1, 连续N天=0才输出0
        min_hold_days: 持仓后至少持仓N天

    返回: 过滤后的信号
    """
    if confirm_days <= 0 and min_hold_days <= 0:
        return raw_signal

    signal = raw_signal.values
    out = np.zeros_like(signal)
    state = int(signal[0])
    counter = confirm_days if confirm_days > 0 else 1
    hold_remaining = 0

    for i in range(len(signal)):
        cur = int(signal[i])

        if hold_remaining > 0:
            out[i] = state
            hold_remaining -= 1
            continue

        if confirm_days > 0:
            if cur == 1:
                counter += 1
            else:
                counter -= 1

            if counter >= confirm_days:
                state = 1
                counter = confirm_days
                hold_remaining = min_hold_days
            elif counter <= -confirm_days:
                state = 0
                counter = -confirm_days
                hold_remaining = 0
            out[i] = state
        else:
            if cur != state:
                state = cur
                hold_remaining = min_hold_days
            out[i] = state

    return pd.Series(out, index=raw_signal.index)


def compute_risk_gate_d(
    close: pd.Series,
    ma_window: int = 80,
    vix_pct_window: int = 252,
    vix_threshold: float = 0.6,
    confirm_days: int = 3,
    min_hold_days: int = 5,
) -> pd.Series:
    """计算风控门信号 (日度).

    参数:
        close: 金价序列
        ma_window: 均线窗口 (默认80日)
        vix_pct_window: VIX百分位窗口 (默认252日)
        vix_threshold: VIX百分位阈值 (默认0.6)
        confirm_days: 连续确认天数 (默认3)
        min_hold_days: 最小持仓天数 (默认5)

    返回: 0/1信号序列
    """
    ma = close.rolling(ma_window).mean()
    s_ma = (close > ma).astype(float)

    vix = _load_indicator_d("vix", "gold_fred")
    vix_pct = vix.rolling(vix_pct_window).rank(pct=True)
    s_vix = (vix_pct < vix_threshold).astype(float).reindex(close.index).fillna(0)

    raw_signal = ((s_ma + s_vix) > 0).astype(float).shift(1).fillna(0)

    return _apply_confirm_and_hold(raw_signal, confirm_days, min_hold_days)


def compute_vix_extreme_signal(
    close: pd.Series,
    vix: pd.Series,
    extreme_threshold: float = 0.85,
    extreme_surge: float = 0.20,
    reduce_speed: int = 2,
    recover_speed: int = 3,
    reduce_target: float = 0.20,
) -> pd.Series:
    """计算VIX极端事件仓位调整.

    逻辑:
        - 正常: 满仓
        - VIX飙升 (VIX>80%百分位 AND 5日变化>15%): 逐步减仓
        - VIX回落: 逐步加仓

    参数:
        close: 金价序列
        vix: VIX序列
        extreme_threshold: VIX百分位阈值 (默认0.85)
        extreme_surge: VIX 5日变化阈值 (默认0.20)
        reduce_speed: 减仓天数 (默认2天从100%降到reduce_target)
        recover_speed: 加仓天数 (默认3天从reduce_target加到100%)
        reduce_target: 极端事件最低仓位 (默认0.20)

    返回: 仓位序列 (0.2-1.0)
    """
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_chg_5d = vix.pct_change(5)

    position = pd.Series(1.0, index=close.index)
    in_extreme = False
    current_reduce = 0.0

    for i in range(1, len(close)):
        vp = vix_pct.iloc[i] if i < len(vix_pct) else 0.5
        vc = vix_chg_5d.iloc[i] if i < len(vix_chg_5d) else 0
        is_extreme = (vp > extreme_threshold) and (vc > extreme_surge)

        if is_extreme and not in_extreme:
            in_extreme = True
            current_reduce = 0.0

        if in_extreme:
            if is_extreme:
                current_reduce = min(1.0, current_reduce + 1.0 / reduce_speed)
                position.iloc[i] = max(reduce_target, 1.0 - current_reduce)
            else:
                current_reduce = max(0.0, current_reduce - 1.0 / recover_speed)
                position.iloc[i] = max(reduce_target, 1.0 - current_reduce)
                if current_reduce <= 0:
                    in_extreme = False
                    position.iloc[i] = 1.0
        else:
            position.iloc[i] = 1.0

    return position


def compute_vix_momentum_signal(
    close: pd.Series,
    vix: pd.Series,
    vix_high_threshold: float = 0.70,
    vix_accel_threshold: float = 0.10,
) -> pd.Series:
    """计算VIX动量仓位调整.

    逻辑:
        - VIX低位(<50%): 满仓
        - VIX中位(50-70%) + VIX加速上升(MA5>MA20): 减仓到70%
        - VIX高位(>70%) + VIX加速上升: 减仓到40%
        - VIX高位但回落(MA5<MA20): 持仓80%

    参数:
        close: 金价序列
        vix: VIX序列
        vix_high_threshold: VIX高位百分位阈值 (默认0.70)
        vix_accel_threshold: VIX加速变化阈值 (默认0.10)

    返回: 仓位序列 (0.4-1.0)
    """
    vix_pct = vix.rolling(252).rank(pct=True)
    vix_chg_5d = vix.pct_change(5)
    vix_ma5 = vix.rolling(5).mean()
    vix_ma20 = vix.rolling(20).mean()

    position = pd.Series(1.0, index=close.index)

    for i in range(1, len(close)):
        vp = vix_pct.iloc[i] if i < len(vix_pct) else 0.5
        vc = vix_chg_5d.iloc[i] if i < len(vix_chg_5d) else 0
        m5 = vix_ma5.iloc[i] if i < len(vix_ma5) else vix.iloc[i]
        m20 = vix_ma20.iloc[i] if i < len(vix_ma20) else vix.iloc[i]
        vix_accel = m5 > m20

        if vp < 0.5:
            position.iloc[i] = 1.0
        elif vp < vix_high_threshold:
            if vix_accel and vc > vix_accel_threshold:
                position.iloc[i] = 0.7
            else:
                position.iloc[i] = 0.9
        else:
            if vix_accel and vc > vix_accel_threshold:
                position.iloc[i] = 0.4
            elif vix_accel:
                position.iloc[i] = 0.6
            else:
                position.iloc[i] = 0.8

    return position


def compute_trend_bnh_signal(
    trailing_stop: float = 0.12,
    use_risk_gate: bool = False,
    use_vix_extreme: bool = False,
    use_vix_momentum: bool = False,
    ma_window: int = 80,
    vix_pct_window: int = 252,
    vix_threshold: float = 0.6,
    vix_high_threshold: float = 0.70,
    vix_accel_threshold: float = 0.10,
    extreme_threshold: float = 0.85,
    extreme_surge: float = 0.20,
    reduce_speed: int = 2,
    recover_speed: int = 3,
    reduce_target: float = 0.20,
    confirm_days: int = 0,
    min_hold_days: int = 0,
    freq: str = "weekly",
) -> pd.DataFrame:
    """计算趋势跟踪B&H信号 (可叠加风控层/VIX控制).

    参数:
        trailing_stop: 追踪止损比例
        use_risk_gate: 是否叠加风控层 (MA80+VIX门控)
        use_vix_extreme: 是否启用VIX极端事件过滤
        use_vix_momentum: 是否启用VIX动量调仓
        ma_window: 风控层均线窗口
        vix_pct_window: VIX百分位窗口
        vix_threshold: 风控层VIX百分位阈值
        vix_high_threshold: VIX动量高位阈值
        vix_accel_threshold: VIX动量加速阈值
        extreme_threshold: VIX极端事件百分位阈值
        extreme_surge: VIX极端事件5日变化阈值
        reduce_speed: 极端事件减仓天数
        recover_speed: 极端事件加仓天数
        reduce_target: 极端事件最低仓位
        confirm_days: 风控门确认天数 (默认0)
        min_hold_days: 风控门最小持仓天数 (默认0)
        freq: "weekly" 或 "daily"

    返回: DataFrame with columns:
        - close: 金价
        - position: 仓位 (0-1)
    """
    if freq == "daily":
        close = load_gold_price_d()
    else:
        close = load_gold_price_w()

    # 1. 趋势B&H止损
    position = pd.Series(0.0, index=close.index)
    peak = close.iloc[0]

    for i in range(len(close)):
        if i == 0:
            position.iloc[i] = 1.0
            peak = close.iloc[i]
            continue

        if close.iloc[i] > peak:
            peak = close.iloc[i]

        if close.iloc[i] < peak * (1 - trailing_stop):
            position.iloc[i] = 0.0
            peak = close.iloc[i]
        else:
            position.iloc[i] = 1.0

    # 2. 风控层叠加
    if use_risk_gate:
        if freq == "daily":
            risk_gate = compute_risk_gate_d(
                close,
                ma_window=ma_window,
                vix_pct_window=vix_pct_window,
                vix_threshold=vix_threshold,
                confirm_days=confirm_days,
                min_hold_days=min_hold_days,
            )
        else:
            ma = close.rolling(ma_window // 5).mean()
            s_ma = (close > ma).astype(float).shift(1).fillna(0)
            vix = _load_indicator_d("vix", "gold_fred")
            vix_w = vix.resample("W-FRI").last().ffill()
            vix_pct_w = vix_w.rolling(vix_pct_window // 5).rank(pct=True)
            s_vix = (vix_pct_w < vix_threshold).astype(float).reindex(close.index).fillna(0)
            raw_w = ((s_ma + s_vix) > 0).astype(float)
            risk_gate = _apply_confirm_and_hold(raw_w, confirm_days, min_hold_days)

        position = position * risk_gate.reindex(close.index).fillna(1)

    # 3. VIX极端事件过滤
    if use_vix_extreme:
        vix_d = _load_indicator_d("vix", "gold_fred")
        vix_extreme = compute_vix_extreme_signal(
            close, vix_d,
            extreme_threshold=extreme_threshold,
            extreme_surge=extreme_surge,
            reduce_speed=reduce_speed,
            recover_speed=recover_speed,
            reduce_target=reduce_target,
        )
        position = position * vix_extreme.reindex(close.index).fillna(1)

    # 4. VIX动量调仓
    if use_vix_momentum:
        vix_d = _load_indicator_d("vix", "gold_fred")
        vix_mom = compute_vix_momentum_signal(
            close, vix_d,
            vix_high_threshold=vix_high_threshold,
            vix_accel_threshold=vix_accel_threshold,
        )
        position = position * vix_mom.reindex(close.index).fillna(1)

    out = pd.DataFrame({
        "close": close,
        "position": position,
    })

    return out


def backtest_trend_bnh(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测趋势跟踪B&H策略."""
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    is_daily = isinstance(close.index.freq, pd.offsets.Tick) or len(close) > 1000
    n_periods = 252 if is_daily else 52

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "avg_pos": 0, "pct_active": 0, "nav": pd.Series(dtype=float),
            "turnover": 0,
        }

    # nav从close.iloc[0]开始, 累积strat_ret
    nav = close.iloc[0] * (1 + strat_ret).cumprod()

    total_ret = nav.iloc[-1] / close.iloc[0] - 1
    n_years = len(strat_ret) / n_periods
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(n_periods)
    max_dd = (nav / nav.cummax() - 1).min()

    turnover = pos.diff().abs().sum() / 2 / n_years

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "avg_pos": pos.mean(),
        "pct_active": (pos > 0).mean(),
        "turnover": turnover,
        "nav": nav,
    }


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    trailing_stop: float = 0.12,
    **kwargs,
):
    """获取周频收益用于组合优化."""
    kwargs.setdefault("freq", "weekly")
    signal = compute_trend_bnh_signal(trailing_stop=trailing_stop, **kwargs)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "load_gold_price_w",
    "load_gold_price_d",
    "compute_risk_gate_d",
    "compute_trend_bnh_signal",
    "backtest_trend_bnh",
    "get_weekly_returns",
    "compute_hysteresis_trend_d",
    "compute_macro_regime_signal",
    "compute_risk_overlay",
    "compute_adaptive_ensemble",
    "_rolling_drawdown",
    "_rolling_calmar",
]


# =====================================================================
# 宏观机制版本 (v2): 滞回带趋势 × GVZ自适应 × 实际利率门控
#
# 动机 (2010-2019 OOS 失效诊断, 见 scripts/cmf/test_oos_diagnosis.py):
#   1. 旧trailing stop止损后peak重置且无空仓记忆, 33次止损100%次日恢复满仓
#   2. VIX是股票波动率, 对黄金特有暴跌无感 (暴跌日VIX-z中位-0.37, GVZ-z+1.86)
#   3. 2013 taper为实际利率冲击, 价格结构信号无法规避, 需利率门控
#
# 双窗口验证 (2020-2026为参数选择窗口IS, 2010-2019仅验证):
#   base_v0 现状:      OOS 年化2.53%/-43.55%/C0.06 | IS 16.43%/-32.92%/C0.50
#   本函数默认参数:     OOS 年化6.14%/-29.40%/C0.21 | IS 17.73%/ -8.59%/C2.06
#   四子周期全部改善 (10-14/15-19/20-23/24-26)
# =====================================================================


def compute_hysteresis_trend_d(
    ts_out: float = 0.06,
    reentry_ratio: float = 0.7,
) -> pd.Series:
    """双阈值滞回带趋势信号 (替代旧trailing stop).

    与旧compute_trend_bnh_signal的结构差异:
        旧版: 止损即重置peak且无状态记忆 → 次日恢复满仓 (33次止损100%次日回仓)
        本版: 止损后锁定空仓, 只有价格收复 peak*(1-reentry_ratio*ts_out) 才回仓

    参数:
        ts_out: 止损阈值 (从滚动峰值回撤超过该比例则平仓)
        reentry_ratio: 重入系数, 重入线 = peak*(1 - reentry_ratio*ts_out)

    返回: 仓位序列 (0/1), index=日度金价
    """
    close = load_gold_price_d()
    ts_in = reentry_ratio * ts_out
    pos = pd.Series(np.nan, index=close.index)
    peak = close.iloc[0]
    state = 1.0
    values = close.values
    for i in range(len(close)):
        c = values[i]
        if state == 1.0:
            if c > peak:
                peak = c
            if c < peak * (1 - ts_out):
                state = 0.0
                peak = c
        else:
            if c > peak:
                peak = c
            if c >= peak * (1 - ts_in):
                state = 1.0
                peak = c
        pos.iloc[i] = state
    return pos


def _zscore_scale(
    series: pd.Series,
    lookback: int = 90,
    target: float = 18.0,
    power: float = 0.8,
    floor: float = 0.3,
    ceil: float = 1.5,
) -> pd.Series:
    """波动率zscore自适应缩放因子: 高波动减仓, 低波动加仓."""
    std = series.rolling(lookback).std().replace(0, np.nan)
    z = (series - target) / std
    return np.exp(-z * power).clip(floor, ceil).fillna(1.0)


def _rate_gate(
    tips: pd.Series,
    window: int = 21,
    power: float = 2.5,
    floor: float = 0.15,
    ceil: float = 1.3,
) -> pd.Series:
    """实际利率动量门控: Δtips10y上行(紧缩)→减仓.

    使用504日滚动百分位归一, 避免量纲漂移; 利率长期与金价反向
    (2010s滚动相关-0.386, 2020s -0.117).
    """
    d = tips.diff(window)
    rank = d.rolling(504, min_periods=126).rank(pct=True)
    factor = np.exp(-(rank - 0.5) * power).clip(floor, ceil)
    return factor.fillna(1.0)


def compute_macro_regime_signal(
    ts_out: float = 0.06,
    reentry_ratio: float = 0.7,
    gvz_lookback: int = 90,
    gvz_target: float = 18.0,
    gvz_power: float = 0.8,
    gvz_floor: float = 0.3,
    gvz_ceil: float = 1.5,
    rate_window: int = 21,
    rate_power: float = 2.5,
    rate_floor: float = 0.15,
    rate_ceil: float = 1.3,
    breakout_weight: float = 0.3,
    breakout_gate_pct: float = 0.7,
) -> pd.DataFrame:
    """宏观机制黄金策略: 滞回带趋势 × GVZ自适应 × 实际利率门控.

    组合结构 (各层独立可关):
        base  = hysteresis_trend × vix_momentum + breakout_weight × breakout × vix_gate
        final = base × gvz_zscore_adapt × rate_gate

    返回: DataFrame [close, position]
    """
    # 延迟导入避免包级循环依赖 (gold/__init__ 导入本模块)
    from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
        compute_breakout_signal,
    )

    close = load_gold_price_d()

    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()
    tips = _load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()

    trend = compute_hysteresis_trend_d(ts_out=ts_out, reentry_ratio=reentry_ratio)
    trend = trend.reindex(close.index).fillna(0)

    vmom = compute_vix_momentum_signal(close, vix).reindex(close.index).fillna(1.0)

    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    vix_gate = (vix.rolling(252).rank(pct=True) > breakout_gate_pct).astype(float)

    base = trend * vmom + breakout_weight * bo * vix_gate

    f_gvz = _zscore_scale(gvz, gvz_lookback, gvz_target, gvz_power, gvz_floor, gvz_ceil)
    f_rate = _rate_gate(tips, rate_window, rate_power, rate_floor, rate_ceil)

    position = (base * f_gvz * f_rate).clip(0, 1.2)

    return pd.DataFrame({"close": close, "position": position})


# =====================================================================
# 风控覆盖层 (Risk Overlay) - 三级风控 + 智能再入场
#
# 设计动机 (docs/gold/risk_overlay_design.md):
#   现状所有候选策略 OOS MaxDD 在 -30% 级别, 远超 -15% 硬约束.
#   单纯组合信号层/子策略无法解决根本问题, 必须加独立主动风控层.
#
# 三级架构:
#   Level 1: 波动率压制 (continuous)
#   Level 2: 软减仓 (discrete, 50%仓位)
#   Level 3: 硬止损 + 智能再入场 (VIX低 + GVZ低确认)
#
# 双窗口验证 (预期): MaxDD 从 -30% 压到 <-15%, G 改善, 年化损失 20-40%
# =====================================================================


def _rolling_drawdown(
    close: pd.Series,
    position: pd.Series,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """基于仓位的滚动MaxDD (向前看window内最大回撤)."""
    nav = (1 + (position.shift(1).fillna(0) * close.pct_change()).fillna(0)).cumprod()
    nav.iloc[:min_periods] = 1.0
    rolling_max = nav.rolling(window, min_periods=min_periods).max()
    return nav / rolling_max - 1


def compute_risk_overlay(
    close: pd.Series,
    position: pd.Series,
    vix: pd.Series = None,
    gvz: pd.Series = None,
    hard_cap: float = -0.15,
    soft_cap: float = -0.12,
    daily_cap: float = -0.03,
    vol_cap: float = 0.20,
    cooldown_days: int = 5,
    vol_window: int = 20,
    reentry_vix_pct: float = 0.6,
    reentry_gvz_z: float = 1.0,
    gvz_z_window: int = 90,
    warmup_days: int = 60,
) -> pd.Series:
    """三级风控覆盖层.

    Level 1 - 波动率压制 (continuous):
        realized_vol > vol_cap 时, 仓位×(vol_cap/realized_vol)

    Level 2 - 软减仓 (discrete):
        rolling_dd < soft_cap 时, 仓位×0.5

    Level 3 - 硬止损 + 智能再入场:
        触发条件: rolling_dd < hard_cap OR daily_ret < daily_cap
        触发动作: 立即清仓, 启动冷却期, 冷却期内强制position=0
        再入场: 冷却期结束后需 vix百分位 < reentry_vix_pct
                AND gvz_zscore < reentry_gvz_z 才允许恢复

    参数:
        close: 收盘价
        position: 原始仓位信号 (0~1.2)
        vix: VIX序列 (用于再入场确认, 可选)
        gvz: GVZ序列 (用于再入场确认, 可选)
        hard_cap: MaxDD硬阈值 (默认-0.15)
        soft_cap: MaxDD软阈值 (默认-0.12)
        daily_cap: 单日DD阈值 (默认-0.03)
        vol_cap: 年化波动率上限 (默认0.20)
        cooldown_days: 触发后冷却天数 (默认5)
        vol_window: 波动率计算窗口 (默认20)
        reentry_vix_pct: 再入场所需VIX百分位阈值 (默认0.6)
        reentry_gvz_z: 再入场所需GVZ-z阈值 (默认1.0)
        gvz_z_window: GVZ zscore窗口 (默认90)
        warmup_days: 前N天不应用风控 (用于指标warmup)

    返回: 调整后仓位序列 (与position同index, 值域0~1.2)
    """
    pos = position.reindex(close.index).fillna(method="ffill").fillna(0).copy()

    if vix is not None:
        vix = vix.reindex(close.index).ffill()
    if gvz is not None:
        gvz = gvz.reindex(close.index).ffill()

    # 计算基础指标
    daily_ret = close.pct_change()
    rolling_dd = _rolling_drawdown(close, pos, window=252)

    # Level 1: 波动率压制
    realized_vol = daily_ret.rolling(vol_window, min_periods=5).std() * np.sqrt(252)
    vol_mask = (realized_vol > vol_cap) & (realized_vol.notna())
    vol_scale = (vol_cap / realized_vol).clip(upper=1.0)
    pos = pos.where(~vol_mask, pos * vol_scale)

    # Level 2: 软减仓
    soft_mask = rolling_dd < soft_cap
    pos = pos.where(~soft_mask, pos * 0.5)

    # Level 3: 硬止损 (清零)
    # daily_ret < 0 时表示下跌; |daily_ret| > |daily_cap| 时触发硬止损
    hard_mask = (rolling_dd < hard_cap) | (daily_ret < daily_cap)
    hard_mask = hard_mask.fillna(False)
    pos = pos.where(~hard_mask, 0.0)

    # 智能再入场: 检测硬触发, 启动冷却期, 结束后需VIX+GVZ双确认
    final_pos = pos.copy()
    cooldown_until = pd.Series(pd.NaT, index=close.index)

    if cooldown_days > 0:
        trigger = hard_mask & (~hard_mask.shift(1).fillna(False))
        for i in range(len(close)):
            if trigger.iloc[i]:
                end_i = min(i + cooldown_days, len(close) - 1)
                cooldown_until.iloc[i:end_i + 1] = close.index[end_i]

    if vix is not None and gvz is not None:
        gvz_mean = gvz.rolling(gvz_z_window, min_periods=20).mean()
        gvz_std = gvz.rolling(gvz_z_window, min_periods=20).std().replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std
        vix_pct = vix.rolling(252, min_periods=60).rank(pct=True)

        for i in range(len(close)):
            if pd.notna(cooldown_until.iloc[i]) and close.index[i] <= cooldown_until.iloc[i]:
                # 冷却期内
                if close.index[i] == cooldown_until.iloc[i]:
                    # 冷却期最后一天: 检查再入场条件
                    v_pct = vix_pct.iloc[i]
                    g_z = gvz_z.iloc[i]
                    if pd.notna(v_pct) and pd.notna(g_z):
                        if v_pct < reentry_vix_pct and g_z < reentry_gvz_z:
                            pass  # 允许恢复
                        else:
                            # 不满足, 延长冷却期
                            new_end = min(i + cooldown_days, len(close) - 1)
                            cooldown_until.iloc[i:new_end + 1] = close.index[new_end]
                            final_pos.iloc[i] = 0.0
                else:
                    final_pos.iloc[i] = 0.0

    final_pos = final_pos.where(final_pos.notna(), pos)

    # warmup期: 不应用风控
    final_pos.iloc[:warmup_days] = pos.iloc[:warmup_days]

    return final_pos.clip(0, 1.2)


# =====================================================================
# 自适应集成信号 (Adaptive Ensemble) — 多信号加权 + 滚动权重
#
# 设计动机:
#   单信号策略都有短板 (macro OOS回撤大, real_rate IS年化低).
#   多信号集成 (ensemble) 可能平滑风险.
#
# 两种加权模式:
#   "fixed"   : 固定权重 (macro/real_rate/momentum)
#   "adaptive": 滚动Calmar动态权重 — 近N月表现好的信号获更高权重
#
# 输出: 加权组合仓位, 已对齐到日度, 已应用compute_risk_overlay
# =====================================================================


def _rolling_calmar(close, position, window=126):
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
      1. macro_regime (gold_trend_bnh.compute_macro_regime_signal)
      2. real_rate (gold_real_rate.compute_real_rate_signal tips_ma)
      3. momentum (gold_momentum.compute_momentum_signal long_only)

    加权方式:
      "fixed"    : 固定权重 (macro/real_rate/momentum)
      "adaptive" : 滚动Calmar动态权重 (近N月表现好者权重高)
                    weight_i = exp(rolling_calmar_i * adaptive_temp)
                    normalized by sum

    风控: 集成信号 → compute_risk_overlay

    返回: DataFrame [close, position]
    """
    # 延迟导入避免循环
    from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
        compute_real_rate_signal,
    )
    from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
        compute_momentum_signal,
    )

    if close is None:
        close = load_gold_price_d()

    if vix is None:
        vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    if gvz is None:
        gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()
    if tips is None:
        tips = _load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()

    # 获取3个信号 (weekly → daily resample)
    macro_df = compute_macro_regime_signal()
    macro_pos = macro_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    rr_df = compute_real_rate_signal(variant="tips_ma")
    rr_pos = rr_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    mom_df = compute_momentum_signal(variant="long_only")
    mom_pos = mom_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    signals = {"macro": macro_pos, "real_rate": rr_pos, "momentum": mom_pos}
    weights = {"macro": macro_weight, "real_rate": real_rate_weight,
               "momentum": momentum_weight}

    if weight_method == "fixed":
        # 固定加权
        combined = sum(weights[name] * pos for name, pos in signals.items())
    elif weight_method == "adaptive":
        # 滚动Calmar自适应加权
        cal = {name: _rolling_calmar(close, pos, adaptive_lookback)
               for name, pos in signals.items()}
        cal_df = pd.DataFrame(cal)
        # softmax权重: exp(calmar * temp)
        w_raw = np.exp(cal_df * adaptive_temp)
        # 初始化(第一个lookback期)用等权
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
) -> pd.DataFrame:
    """分层集成信号 (Layered Ensemble) — 各信号独立风控后再加权.

    与 compute_adaptive_ensemble 的区别:
      adaptive: 先加权 → 再风控 (信号混合成中间仓位, 风控难以精确控制)
      layered:  先各自风控 → 再加权  (每个信号独立满足约束, 集成后天然满足)

    集成方式:
      - 三路基础 (默认等权): macro + real_rate + momentum
      - 第4路突破 (可选): gold_breakout, 永久小仓位
      - 权重归一: 三路均分(1-breakout_weight), 第4路固定breakout_weight

    Walk-Forward验证 (breakout_weight=0.1):
      - 15折中10折跑赢三路基线
      - CAGR +0.23pp (5.94% vs 5.71%)
      - MaxDD持平 (-12.94% vs -12.97%)
      - 2026年优势明显 (+1.31pp)

    参数:
        close: 收盘价 (可选, 默认自动加载)
        vix/gvz/tips: 指标 (可选, 默认自动加载)
        use_risk_overlay: 是否应用风控覆盖 (默认 True)
        hard_cap/soft_cap/daily_cap/vol_cap/cooldown_days: 风控参数
        breakout_weight: 突破信号权重 (默认0.0, 向后兼容)

    返回: DataFrame [close, position]
    """
    from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
        compute_real_rate_signal,
    )
    from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
        compute_momentum_signal,
    )
    from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
        compute_breakout_signal,
    )

    if close is None:
        close = load_gold_price_d()
    if vix is None:
        vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    if gvz is None:
        gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    # 获取4个信号 (weekly → daily)
    macro_df = compute_macro_regime_signal()
    macro_pos = macro_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    rr_df = compute_real_rate_signal(variant="tips_ma")
    rr_pos = rr_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    mom_df = compute_momentum_signal(variant="long_only")
    mom_pos = mom_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

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
        macro_ov, rr_ov, mom_ov, bo_ov = macro_pos, rr_pos, mom_pos, bo_pos if breakout_weight > 0 else None

    # 加权平均 (三路均分(1-breakout_weight), 第4路固定breakout_weight)
    if breakout_weight > 0 and bo_ov is not None:
        base_weight = (1 - breakout_weight) / 3
        position = (base_weight * macro_ov + base_weight * rr_ov + base_weight * mom_ov
                    + breakout_weight * bo_ov).clip(0, 1.2)
    else:
        position = ((macro_ov + rr_ov + mom_ov) / 3).clip(0, 1.2)

    return pd.DataFrame({"close": close, "position": position})
