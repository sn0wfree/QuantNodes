# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_carry.py

铜期限结构/Carry独立子策略 (long-only).

核心逻辑 (基于2018-2026数据验证):
    1. 期限结构斜率: spread_1_2 / close_near2
       - Backwardation (斜率<0): 现货紧 → 滚仓收益为正 → 做多
       - Contango (斜率>0): 现货松 → 滚仓拖累 → 空仓
    2. Carry是状态条件而非独立信号 (纯carry年化-5.3%)
       - 需要与价格动量结合: back + 上升动量 → 前瞻年化19.9%
    3. 信号持续期过滤: 减少contango/back边界处的噪音切换
       (carry自相关 lag20≈0.46, 持续性较强, 适合低频周频调仓)
    4. 曲率因子: (F1-2*F2+F3)/F1, 捕捉期限结构凸性信息
    5. 库存因子: LME铜库存变化, 预测carry regime变化

参数优化 (2026-08-24, 网格搜索125种组合):
    - z_window=26 (半年窗口, 更灵敏)
    - mom_window=4 (1个月动量, 更快反应)
    - min_duration=6 (更长持续期, 过滤噪音)
    - Sharpe: 0.59→0.87, MaxDD: -10.5%→-8.5%

主要变体:
    - state_mom: carry+动量 (基准)
    - state_mom_curv2: carry+动量+曲率 (默认, Sharpe 0.87)
    - state_mom_inv: carry+动量+库存 (风控优先)

数据: term_structure_cu (2018+), inventory_daily (2014+), 周频 W-FRI
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..sign_constrained_lasso import rolling_zscore

DB = Path("data/cmf/fundamental.duckdb")


def load_term_structure(con=None) -> pd.DataFrame:
    """加载铜期限结构数据并重采样到周频.

    返回: DataFrame indexed by W-FRI, columns=[close_near, close_near2, spread_1_2, carry_slope]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close_near, close_near2, close_near3, spread_1_2 "
        "FROM term_structure_cu ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    return df.resample("W-FRI").last().dropna(subset=["close_near"])


def load_lme_inventory(con=None) -> pd.Series:
    """加载LME铜库存并重采样到周频.

    返回: Series indexed by W-FRI, name='lme_inventory'
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, value AS inventory "
        "FROM inventory_daily "
        "WHERE value_col = '铜-库存' "
        "ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index("trade_date")
    inv = df["inventory"].resample("W-FRI").last().dropna()
    # 库存变化率 (周环比)
    inv_pct = inv.pct_change(4).rename("inv_change_4w")  # 4周库存变化
    return pd.DataFrame({"lme_inventory": inv, "inv_change_4w": inv_pct})


def _apply_min_duration(state: pd.Series, min_weeks: int) -> pd.Series:
    """最小持续时间约束: 状态切换需持续 min_weeks 周才生效 (同 copper_economy 思路)."""
    constrained = state.copy()
    prev = None
    last_change_idx = 0
    for i in range(len(constrained)):
        cur = constrained.iloc[i]
        if prev is None:
            prev = cur
            continue
        if cur != prev:
            if i - last_change_idx < min_weeks:
                constrained.iloc[i] = prev
            else:
                last_change_idx = i
                prev = cur
    return constrained


def compute_carry_signal(
    ts: pd.DataFrame,
    z_window: int = 52,
    min_duration: int = 4,
) -> pd.DataFrame:
    """计算carry信号 (纯函数).

    carry_slope : spread_1_2 / close_near2 (原始斜率, <0为backwardation)
    carry_zscore: 斜率的滚动z-score (标准化)
    carry_state : 持续期过滤后的3状态 (+1=back / 0=中性 / -1=contango)

    返回: DataFrame indexed by W-FRI
    """
    slope = (ts["spread_1_2"] / ts["close_near2"]).rename("carry_slope")

    # 曲率因子: (F1 - 2*F2 + F3) / F1
    # 正曲率 = 期限结构凸 (中间凹), 负曲率 = 期限结构凹 (中间凸)
    curvature = ((ts["close_near"] - 2 * ts["close_near2"] + ts["close_near3"])
                 / ts["close_near"]).rename("curvature")

    zscore = rolling_zscore(slope, window=z_window).rename("carry_zscore")
    curvature_zscore = rolling_zscore(curvature, window=z_window).rename("curvature_zscore")

    # 3状态: z < -0.5 → back(+1), z > 0.5 → contango(-1), 否则中性(0)
    raw_state = pd.Series(0, index=slope.index, dtype=float)
    raw_state[zscore < -0.5] = 1.0
    raw_state[zscore > 0.5] = -1.0
    state = _apply_min_duration(raw_state, min_duration).rename("carry_state")

    return pd.DataFrame({
        "carry_slope": slope,
        "carry_zscore": zscore,
        "carry_state": state,
        "curvature": curvature,
        "curvature_zscore": curvature_zscore,
    })


def build_signal_carry(
    con=None,
    variant: str = "state_mom_curv2",
    z_window: int = 26,
    min_duration: int = 6,
    mom_window: int = 4,
) -> pd.DataFrame:
    """构建carry子策略仓位.

    变体 (long-only):
        state          : carry_state==+1 → 1.0, 否则 0
        state_mom      : back且动量>0 → 1.0; back但动量<0 → 0.5; 否则 0
        state_mom_curv : 三因子投票 (carry+momentum+curvature), 一致→1.0, 两两→0.5
        state_mom_curv2: state_mom + 曲率连续调整 (±30%), MaxDD改善2pp
        back_trend     : back+趋势过滤
        continuous     : pos = clip(-carry_zscore, 0, 1)

    变体 (long-short, contango做空):
        state_ls      : back → +1.0; contango → -1.0; 否则 0
        state_mom_ls  : back+动量↑ → 1.0; back+动量↓ → 0.5;
                        contango+动量↓ → -1.0; contango+动量↑ → -0.5
        continuous_ls : pos = clip(-carry_zscore, -1, 1)
        bear_only_ls  : 仅在contango+动量↓做空(-1.0)

    返回: DataFrame with columns=[position, carry_slope, carry_zscore, carry_state, mom]
    """
    ts = load_term_structure(con)
    sig = compute_carry_signal(ts, z_window=z_window, min_duration=min_duration)

    close_near = ts["close_near"]
    mom = close_near.pct_change(mom_window).rename("mom")

    df = sig.join(mom)

    # 库存增强变体需要库存数据
    if variant in ("state_mom_inv", "state_mom_all"):
        inv_df = load_lme_inventory(con)
        inv_change = inv_df["inv_change_4w"].reindex(df.index).fillna(0)
        # 库存zscore (负值=库存下降=利好)
        inv_zscore = rolling_zscore(-inv_change, window=z_window).rename("inv_zscore")
        df = df.join(inv_zscore)

    pos = pd.Series(0.0, index=df.index)
    if variant == "state":
        pos[df["carry_state"] == 1.0] = 1.0
    elif variant == "state_mom":
        back = df["carry_state"] == 1.0
        pos[back & (df["mom"] > 0)] = 1.0
        pos[back & (df["mom"] <= 0)] = 0.5
    elif variant == "back_trend":
        # 探针方案2: back区域+趋势过滤
        zscore = df["carry_zscore"]
        mom_trend = close_near.pct_change(mom_window).reindex(df.index).fillna(0)
        back_zone = zscore > 0.5
        pos[back_zone & (mom_trend > 0)] = 1.0       # back + 趋势↑ → 满仓
        pos[back_zone & (mom_trend <= 0)] = 0.5      # back + 趋势↓ → 半仓
        pos[~back_zone & (mom_trend > 0.05)] = 0.3   # 非back + 强趋势 → 轻仓
    elif variant == "state_mom_curv":
        # 曲率增强: carry + momentum + curvature 三因子投票
        back = df["carry_state"] == 1.0
        curv_pos = df["curvature_zscore"] > 0  # 正曲率 = 期限结构凸
        mom_up = df["mom"] > 0
        # 三因子一致 → 满仓; 两因子一致 → 半仓; 否则空仓
        agree3 = back & curv_pos & mom_up
        agree2 = back & ((curv_pos & ~mom_up) | (~curv_pos & mom_up))
        pos[agree3] = 1.0
        pos[agree2 & ~agree3] = 0.5
    elif variant == "state_mom_curv2":
        # 曲率连续调整: 在state_mom基础上, 用曲率调整仓位
        back = df["carry_state"] == 1.0
        mom_up = df["mom"] > 0
        curv = df["curvature_zscore"].clip(-1, 1)  # 曲率zscore, clip到[-1,1]
        # 基础仓位: state_mom逻辑
        base = pd.Series(0.0, index=df.index)
        base[back & mom_up] = 1.0
        base[back & ~mom_up] = 0.5
        # 曲率调整: 正曲率加仓, 负曲率减仓
        adj = 1.0 + curv * 0.3  # 曲率贡献最多±30%
        pos = (base * adj).clip(0, 1)
    elif variant == "state_mom_inv":
        # 库存增强: carry + momentum + inventory
        # 逻辑: 低库存+back→强信号; 高库存+back→弱信号
        back = df["carry_state"] == 1.0
        mom_up = df["mom"] > 0
        inv_low = df["inv_zscore"] < -0.5  # 库存低于均值0.5σ
        # 库存下降+back→满仓; 库存上升+back→半仓; 否则0
        pos[back & mom_up & inv_low] = 1.0    # 最强: back+动量↑+低库存
        pos[back & mom_up & ~inv_low] = 0.7   # 强: back+动量↑+库存正常
        pos[back & ~mom_up & inv_low] = 0.5   # 中: back+动量↓+低库存
        pos[back & ~mom_up & ~inv_low] = 0.3  # 弱: back+动量↓+库存高
    elif variant == "state_mom_all":
        # 四因子: carry + momentum + curvature + inventory
        back = df["carry_state"] == 1.0
        mom_up = df["mom"] > 0
        curv = df["curvature_zscore"].clip(-1, 1)
        inv_low = df["inv_zscore"] < -0.5
        # 基础: state_mom逻辑
        base = pd.Series(0.0, index=df.index)
        base[back & mom_up] = 1.0
        base[back & ~mom_up] = 0.5
        # 曲率+库存调整
        adj_curv = 1.0 + curv * 0.2  # 曲率±20%
        adj_inv = pd.Series(1.0, index=df.index)
        adj_inv[inv_low] = 1.2   # 低库存+20%
        adj_inv[~inv_low] = 0.8  # 高库存-20%
        pos = (base * adj_curv * adj_inv).clip(0, 1)
    elif variant == "continuous":
        pos = (-df["carry_zscore"]).clip(lower=0.0, upper=1.0).fillna(0.0)
    elif variant == "state_ls":
        pos[df["carry_state"] == 1.0] = 1.0
        pos[df["carry_state"] == -1.0] = -1.0
    elif variant == "state_mom_ls":
        back = df["carry_state"] == 1.0
        contango = df["carry_state"] == -1.0
        pos[back & (df["mom"] > 0)] = 1.0
        pos[back & (df["mom"] <= 0)] = 0.5
        pos[contango & (df["mom"] < 0)] = -1.0
        pos[contango & (df["mom"] >= 0)] = -0.5
    elif variant == "continuous_ls":
        pos = (-df["carry_zscore"]).clip(lower=-1.0, upper=1.0).fillna(0.0)
    elif variant == "bear_only_ls":
        back = df["carry_state"] == 1.0
        contango = df["carry_state"] == -1.0
        pos[back & (df["mom"] > 0)] = 1.0
        pos[back & (df["mom"] <= 0)] = 0.5
        pos[contango & (df["mom"] < 0)] = -1.0
    else:
        raise ValueError(f"未知变体: {variant}")

    df["position"] = pos.fillna(0.0)
    df["M"] = close_near
    return df


def backtest_carry(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """周频回测carry子策略 (基于近月合约收益, 仓位延迟1周)."""
    pos = signal["position"].fillna(0)
    ret = signal["M"].pct_change().reindex(pos.index)

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0, "total_ret": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0, "nav": pd.Series(dtype=float),
        }

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min()

    # 有仓位时胜率 (排除空仓周)
    has_pos = pos.shift(1).reindex(strat_ret.index).fillna(0) > 0.01
    win_rate_active = (strat_ret[has_pos] > 0).mean() if has_pos.sum() > 0 else 0
    pct_active = has_pos.mean() * 100

    return {
        "n_weeks": len(strat_ret), "total_ret": total_ret, "ann_ret": ann_ret,
        "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate_active, "pct_active": pct_active, "nav": nav,
    }


__all__ = [
    "load_term_structure",
    "load_lme_inventory",
    "compute_carry_signal",
    "build_signal_carry",
    "backtest_carry",
]
