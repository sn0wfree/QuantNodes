# coding=utf-8
"""gold_production.py — 黄金生产策略 (B3全执行栈, 唯一权威实现).

生产策略 = 预算制执行栈 (2026-09-01 定版, 详见 docs/gold/gold_final_strategy.md v4):

    链(-5%/-5%/-20%) → 核心裁剪≤1.25 → +周网格库存(3档×0.083, sp=2%)
    → 总封顶1.5 → 日平滑cap0.20 → 应急(单日<-5%压0)

业绩锚点 (2018-2026, O2O+5bps): CAGR 18.47% / Sharpe 1.56 / MaxDD -16.84%

跟踪: scripts/cmf/gold_track.py (当日状态) / scripts/cmf/gold_backtest.py --stack (回测)
"""
from __future__ import annotations

import math

import duckdb
import numpy as np
import pandas as pd

from .gold_utils import DB_PATH, load_indicator_d
from .gold_trend_bnh import compute_risk_overlay
from .gold_ensemble import compute_layered_ensemble_signal

# 生产参数 (唯一参数源; 参数变更须同步 docs/gold/gold_final_strategy.md)
PRODUCTION_PARAMS = dict(
    # 链层 (一层风控 + 杠杆 + 二层风控)
    L=1.5,
    first_cap=-0.05,
    second_cap=-0.05,
    second_hard=-0.20,
    second_soft=-0.12,
    vol_cap=0.20,
    cooldown_days=5,
    deadband=0.05,
    # 预算制网格层
    core_cap=1.25,
    grid_sp=0.02,
    grid_step=0.0833,
    grid_tiers=3,
    budget=1.5,
    # 日平滑层
    smooth_cap=0.20,
    emergency=-0.05,
)

COST_BPS = 5.0


def load_au_adj_ohlcv() -> tuple[pd.Series, pd.Series]:
    """加载换月调整后AU期货 (close, open), 数据从2018-06-20起."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, open, close FROM futures_main_close_au_adj ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates("trade_date", keep="last").set_index("trade_date")
    return df["close"], df["open"]


def load_vix_gvz(close_index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """vix/gvz 生产口径: reindex + ffill + shift(2)."""
    vix = load_indicator_d("vix", "gold_fred").reindex(close_index).ffill().shift(2)
    gvz = load_indicator_d("gvz", "gold_fred").reindex(close_index).ffill().shift(2)
    return vix, gvz


def build_chain(
    L: float = 1.5,
    first_cap: float = -0.05,
    second_cap: float = -0.05,
    second_hard: float = -0.20,
    second_soft: float = -0.12,
    vol_cap: float = 0.20,
    cooldown_days: int = 5,
    deadband: float = 0.05,
) -> pd.Series:
    """生产链: 一层风控信号 → ×L杠杆 → 二层overlay → 死带平滑."""
    close, _ = load_au_adj_ohlcv()
    vix, gvz = load_vix_gvz(close.index)

    sig = compute_layered_ensemble_signal(use_risk_overlay=True, daily_cap=first_cap)
    pos_base = sig["position"]
    pos_lev = (pos_base * L).clip(0, L * 1.2)

    pos = compute_risk_overlay(
        close=close, position=pos_lev, vix=vix, gvz=gvz,
        hard_cap=second_hard, soft_cap=second_soft, daily_cap=second_cap,
        vol_cap=vol_cap, cooldown_days=cooldown_days,
    )
    for i in range(1, len(pos)):
        if abs(pos.iloc[i] - pos.iloc[i - 1]) < deadband:
            pos.iloc[i] = pos.iloc[i - 1]
    return pos


def weekly_grid_inventory(
    close: pd.Series,
    target_index: pd.DatetimeIndex,
    spacing_pct: float = 0.02,
    step: float = 0.0833,
    max_grids: int = 3,
) -> pd.Series:
    """周频网格库存 (预算制): 价格在锚点下方每跌1档, 库存+step, 回到锚点上方归零.

    周五收盘按档位计库存, 下周执行 (reindex ffill + shift(1) 避免当日决策当日用).
    锚点漂移超过 (max_grids+1) 档时重置 (库存清零, 防止趋势中无界累积).
    """
    w_close = close.resample("W-FRI").last().dropna()
    anchor = None
    inventories = []
    for dt, px in w_close.items():
        if anchor is None or np.isnan(px):
            anchor = px
            inventories.append(0.0)
            continue
        spacing = px * spacing_pct
        n_below = math.floor((anchor - px) / spacing)
        inv = float(np.clip(n_below * step, 0, max_grids * step))
        if abs(px - anchor) > (max_grids + 1) * spacing:
            anchor = px
            inv = 0.0
        inventories.append(inv)
    inv_w = pd.Series(inventories, index=w_close.index)
    return inv_w.reindex(target_index, method="ffill").shift(1).fillna(0.0)


def smooth_execution(
    pos_target: pd.Series,
    close: pd.Series,
    cap: float = 0.20,
    emergency: float = -0.05,
) -> pd.Series:
    """日平滑执行: 仓位向目标限速收敛; 单日暴跌触发应急立即压向0."""
    ret = close.pct_change()
    out = pos_target.copy()
    prev = pos_target.iloc[0]
    for i in range(len(pos_target)):
        tgt = pos_target.iloc[i]
        gap = tgt - prev
        if abs(gap) <= cap:
            new = tgt
        else:
            new = prev + cap if gap > 0 else prev - cap
        if emergency is not None and ret.iloc[i] < emergency:
            new = min(new, 0.0)
        out.iloc[i] = new
        prev = new
    return out


def build_production_position(params: dict | None = None) -> pd.Series:
    """生产仓位: 全执行栈组装 (B3). 传入params可覆盖PRODUCTION_PARAMS任意项."""
    p = dict(PRODUCTION_PARAMS)
    if params:
        p.update(params)

    close, _ = load_au_adj_ohlcv()
    chain = build_chain(
        L=p["L"], first_cap=p["first_cap"], second_cap=p["second_cap"],
        second_hard=p["second_hard"], second_soft=p["second_soft"],
        vol_cap=p["vol_cap"], cooldown_days=p["cooldown_days"],
        deadband=p["deadband"],
    )
    core = chain.clip(0, p["core_cap"])
    grid = weekly_grid_inventory(
        close, chain.index,
        spacing_pct=p["grid_sp"], step=p["grid_step"], max_grids=p["grid_tiers"],
    )
    pos = (core + grid).clip(0, p["budget"])
    return smooth_execution(pos, close, cap=p["smooth_cap"], emergency=p["emergency"])


def o2o_returns(
    pos: pd.Series, open_price: pd.Series, cost_bps: float = COST_BPS
) -> pd.Series:
    """持仓 → 策略日收益 (open-to-open, shift(1), 扣成本)."""
    o = open_price.reindex(pos.index)
    ret_fwd = o.pct_change().shift(-1)
    sr = (pos.shift(1) * ret_fwd).dropna()
    cost = pos.diff().abs() * cost_bps / 10000
    return (sr - cost).dropna()


def position_state(pos: pd.Series, close: pd.Series, params: dict | None = None) -> dict:
    """最新仓位状态解读 (镜像二层风控分层, 用于跟踪报告).

    Returns:
        dict: date/position/pos_change/daily_ret/state
        state ∈ {硬止损, 冷却期, 软减仓, 波动压缩, 正常}
    """
    p = dict(PRODUCTION_PARAMS)
    if params:
        p.update(params)

    daily_ret = close.pct_change()
    pos_reindexed = pos.reindex(close.index).ffill().fillna(0)
    d = close.index[-1]

    # 镜像 compute_risk_overlay 的分层判定 (仅用于状态标注)
    n = len(close)
    entry = None
    rolling_dd = pd.Series(np.nan, index=close.index)
    cur = 1.0
    for i in range(n):
        if pos_reindexed.iloc[i] > 0 and (entry is None or pos_reindexed.iloc[i - 1] <= 0):
            entry = close.iloc[i]
            cur = 1.0
        if entry is not None:
            cur = min(cur, close.iloc[i] / entry)
        rolling_dd.iloc[i] = cur - 1

    realized_vol = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    vol_mask = bool(realized_vol.loc[d] > p["vol_cap"])
    soft_mask = bool(rolling_dd.loc[d] < p["second_soft"])
    hard_mask = bool(rolling_dd.loc[d] < p["second_hard"] or daily_ret.loc[d] < p["second_cap"])

    # 冷却期: 近 cooldown_days 日内出现过硬触发
    recent_hard = hard_mask
    for k in range(1, p["cooldown_days"] + 1):
        if i := d - k:
            prev_d = d - k
            if prev_d in daily_ret.index and prev_d in rolling_dd.index:
                recent_hard = recent_hard or (
                    rolling_dd.loc[prev_d] < p["second_hard"]
                    or daily_ret.loc[prev_d] < p["second_cap"]
                )

    if hard_mask:
        state = "硬止损"
    elif recent_hard:
        state = "冷却期"
    elif soft_mask:
        state = "软减仓"
    elif vol_mask:
        state = "波动压缩"
    else:
        state = "正常"

    pos_today = float(pos_reindexed.loc[d])
    prev_d = close.index[-2] if len(close) > 1 else d
    pos_prev = float(pos_reindexed.loc[prev_d]) if prev_d in pos_reindexed.index else 0.0
    return {
        "date": d,
        "position": pos_today,
        "pos_change": pos_today - pos_prev,
        "daily_ret": float(daily_ret.loc[d]),
        "state": state,
    }
