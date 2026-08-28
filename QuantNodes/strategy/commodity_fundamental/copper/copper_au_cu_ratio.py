# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_au_cu_ratio.py

AU-CU价格比策略 (优化版)

核心逻辑:
    金铜比是风险偏好指标:
    - 金铜比高 (黄金贵/铜便宜) → 风险厌恶 → 铜价下跌
    - 金铜比低 (铜贵/黄金便宜) → 风险偏好 → 铜价上涨

优化点:
    1. 多阈值信号: 根据z-score强度调整仓位
    2. 动量确认: 金铜比动量 + 铜价动量
    3. 波动率目标化: 动态仓位控制
    4. 回撤止损: 渐进式降仓
    5. 多因子增强: 金铜比z-score + 动量 + 波动率
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OPTIONS_DB = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()


def load_au_cu_prices():
    """加载AU和CU价格数据."""
    con = duckdb.connect(str(OPTIONS_DB), read_only=True)

    au = con.execute("""
        SELECT trade_date, close FROM continuous_main_au_daily ORDER BY trade_date
    """).df()
    au["trade_date"] = pd.to_datetime(au["trade_date"])
    au = au.set_index("trade_date")["close"]

    cu = con.execute("""
        SELECT trade_date, close FROM continuous_main_cu_daily ORDER BY trade_date
    """).df()
    cu["trade_date"] = pd.to_datetime(cu["trade_date"])
    cu = cu.set_index("trade_date")["close"]

    return au, cu


def compute_au_cu_ratio_v1(
    z_window: int = 60,
    z_threshold: float = 1.0,
) -> pd.DataFrame:
    """原始策略 (基准).

    参数:
        z_window: z-score窗口
        z_threshold: z-score阈值
    """
    au, cu = load_au_cu_prices()

    # 对齐
    common = au.index.intersection(cu.index)
    au = au.reindex(common)
    cu = cu.reindex(common)

    # 计算金铜比
    ratio = au / cu

    # 计算z-score
    ratio_mean = ratio.rolling(z_window).mean()
    ratio_std = ratio.rolling(z_window).std()
    z = (ratio - ratio_mean) / ratio_std

    # 生成信号
    position = pd.Series(0.0, index=common)
    position[z > z_threshold] = 1.0
    position[z < -z_threshold] = 0.0
    position = position.replace(0, np.nan).ffill().fillna(0)

    return pd.DataFrame({
        "au_close": au,
        "cu_close": cu,
        "ratio": ratio,
        "z_score": z,
        "position": position,
    })


def compute_au_cu_ratio_v2(
    z_window: int = 60,
    dd_threshold: float = -0.20,
) -> pd.DataFrame:
    """优化版策略 (v1 + 回撤止损).

    优化点:
        1. 保持v1逻辑 (97%做多, 3%空仓)
        2. 添加回撤止损 (回撤达-20%时降仓)

    参数:
        z_window: z-score窗口
        dd_threshold: 回撤止损阈值
    """
    au, cu = load_au_cu_prices()

    # 对齐
    common = au.index.intersection(cu.index)
    au = au.reindex(common)
    cu = cu.reindex(common)

    # 计算金铜比
    ratio = au / cu

    # 计算z-score
    ratio_mean = ratio.rolling(z_window).mean()
    ratio_std = ratio.rolling(z_window).std()
    z = (ratio - ratio_mean) / ratio_std

    # v1信号
    base_pos = pd.Series(0.0, index=common)
    base_pos[z > 1.0] = 1.0
    base_pos[z < -1.0] = 0.0
    base_pos = base_pos.replace(0, np.nan).ffill().fillna(0)

    # 回撤止损
    cu_ret = cu.pct_change()
    nav = (1 + cu_ret).cumprod()
    drawdown = nav / nav.cummax() - 1

    risk_scale = pd.Series(1.0, index=common)
    risk_scale[drawdown < dd_threshold * 0.5] = 0.8  # 回撤达50%限制时降仓20%
    risk_scale[drawdown < dd_threshold * 0.8] = 0.6  # 回撤达80%限制时降仓40%
    risk_scale[drawdown < dd_threshold] = 0.4        # 回撤达限制时降仓60%

    position = base_pos * risk_scale
    position = position.clip(0, 1)

    return pd.DataFrame({
        "au_close": au,
        "cu_close": cu,
        "ratio": ratio,
        "z_score": z,
        "drawdown": drawdown,
        "base_pos": base_pos,
        "risk_scale": risk_scale,
        "position": position,
    })


def compute_au_cu_ratio_v3(
    z_window: int = 60,
    mom_window: int = 20,
    vol_window: int = 20,
    vol_target: float = 0.12,
    max_leverage: float = 1.5,
    dd_threshold: float = -0.10,
) -> pd.DataFrame:
    """v3策略: 多因子增强.

    因子:
        1. 金铜比z-score (原有)
        2. 金铜比动量 (新增)
        3. 铜价波动率 (新增)
        4. 金铜比波动率 (新增)

    组合: 加权平均
    """
    au, cu = load_au_cu_prices()

    # 对齐
    common = au.index.intersection(cu.index)
    au = au.reindex(common)
    cu = cu.reindex(common)

    # 计算金铜比
    ratio = au / cu

    # 因子1: 金铜比z-score
    ratio_mean = ratio.rolling(z_window).mean()
    ratio_std = ratio.rolling(z_window).std()
    z = (ratio - ratio_mean) / ratio_std

    # 因子2: 金铜比动量
    ratio_mom = ratio.pct_change(mom_window)

    # 因子3: 铜价波动率 (低波动 = 好)
    cu_vol = cu.pct_change().rolling(vol_window).std() * np.sqrt(252)
    cu_vol_z = (cu_vol - cu_vol.rolling(z_window).mean()) / cu_vol.rolling(z_window).std()

    # 因子4: 金铜比波动率 (低波动 = 稳定)
    ratio_vol = ratio.pct_change().rolling(vol_window).std() * np.sqrt(252)
    ratio_vol_z = (ratio_vol - ratio_vol.rolling(z_window).mean()) / ratio_vol.rolling(z_window).std()

    # 因子组合 (加权平均)
    # z-score: 权重40% (核心信号)
    # 金铜比动量: 权重30% (趋势确认)
    # 铜价波动率: 权重20% (波动率调整)
    # 金铜比波动率: 权重10% (稳定性调整)
    factor_score = (
        z * 0.4 +
        ratio_mom * 0.3 +
        (-cu_vol_z) * 0.2 +  # 低波动 = 好
        (-ratio_vol_z) * 0.1  # 低波动 = 稳定
    )

    # 生成信号
    base_pos = pd.Series(0.5, index=common)
    base_pos[factor_score > 1.0] = 1.0
    base_pos[factor_score > 0.5] = 0.75
    base_pos[factor_score > 0] = 0.5
    base_pos[factor_score < -0.5] = 0.25
    base_pos[factor_score < -1.0] = 0.0

    # 波动率目标化
    vol_scale = (vol_target / cu_vol).fillna(1.0).clip(upper=max_leverage)
    position = base_pos * vol_scale

    # 回撤止损
    nav = (1 + cu.pct_change()).cumprod()
    drawdown = nav / nav.cummax() - 1

    risk_scale = pd.Series(1.0, index=common)
    risk_scale[drawdown < dd_threshold * 0.5] = 0.7
    risk_scale[drawdown < dd_threshold * 0.8] = 0.5
    risk_scale[drawdown < dd_threshold] = 0.3

    position = position * risk_scale
    position = position.clip(0, 1)

    return pd.DataFrame({
        "au_close": au,
        "cu_close": cu,
        "ratio": ratio,
        "z_score": z,
        "ratio_mom": ratio_mom,
        "cu_vol_z": cu_vol_z,
        "ratio_vol_z": ratio_vol_z,
        "factor_score": factor_score,
        "position": position,
    })


def backtest_au_cu_ratio(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测AU-CU价格比策略.

    参数:
        signal: 含position和cu_close的DataFrame
        cost_bps: 交易成本 (基点)
    """
    close = signal["cu_close"]
    pos = signal["position"]
    ret = close.pct_change()

    # 回测
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_days": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(252)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_days": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


def get_weekly_returns(start="2018-01-01", end="2025-12-31"):
    """获取周频收益用于组合优化."""
    # 使用v2策略
    signal = compute_au_cu_ratio_v2()
    ret = signal["cu_close"].pct_change()
    pos = signal["position"]
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()

    # 截取时间范围
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]

    return weekly_ret


def optimize_params():
    """优化参数."""
    print("=" * 60)
    print("AU-CU价格比策略参数优化")
    print("=" * 60)

    # 参数网格
    z_windows = [30, 45, 60, 90, 120]
    mom_windows = [10, 20, 30, 40]

    results = []
    for zw in z_windows:
        for mw in mom_windows:
            try:
                signal = compute_au_cu_ratio_v2(z_window=zw, mom_window=mw)
                result = backtest_au_cu_ratio(signal)
                result["z_window"] = zw
                result["mom_window"] = mw
                results.append(result)
            except Exception as e:
                pass

    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False)

    print(f"\n{'z窗口':<8} {'动量窗口':<8} {'年化':<8} {'Sharpe':<8} {'MaxDD':<8} {'胜率':<8}")
    print("-" * 48)

    for _, row in df.head(10).iterrows():
        print(f"{row['z_window']:<8} {row['mom_window']:<8} {row['ann_ret']:<8.1%} "
              f"{row['sharpe']:<8.2f} {row['max_dd']:<8.1%} {row['win_rate']:<8.1%}")

    best = df.iloc[0]
    print(f"\n最优参数:")
    print(f"  z窗口: {best['z_window']}天")
    print(f"  动量窗口: {best['mom_window']}天")
    print(f"  年化: {best['ann_ret']:.1%}")
    print(f"  Sharpe: {best['sharpe']:.2f}")
    print(f"  MaxDD: {best['max_dd']:.1%}")

    return df


__all__ = [
    "compute_au_cu_ratio_v1",
    "compute_au_cu_ratio_v2",
    "compute_au_cu_ratio_v3",
    "backtest_au_cu_ratio",
    "get_weekly_returns",
    "optimize_params",
]
