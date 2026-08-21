# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_model_v4.py

铜模型v4: VIX + DXY + 库存象限

核心逻辑:
  1. VIX信号: 高波动率时做多 (动量效应)
  2. DXY信号: DXY下降时做多 (宏观环境)
  3. 库存象限: 库存状态辅助决策

关键发现:
  - VIX和DXY相关性低 (Pearson=0.016)
  - VIX多+DXY多时收益最高 (年化32.0%)
  - VIX可以作为DXY的辅助信号
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..pipeline_v3 import _close, _dxy, _tfapi
from ..sign_constrained_lasso import rolling_zscore
from ..portfolio import compute_metrics

DB = Path("data/cmf/fundamental.duckdb")


def build_signal_v4(symbol: str = "CU0", con=None) -> pd.DataFrame:
    """铜模型v4: VIX + DXY + 库存象限.

    策略: 当DXY可用时用v4模型, DXY不可用时用v3模型.
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    close = _close(symbol, con)
    close_w = close.resample("W-FRI").last()
    ret_w = close_w.pct_change()

    # 1. VIX信号 (铜价自身波动率)
    vol_13w = ret_w.rolling(13).std() * np.sqrt(52)
    vol_13w_z = rolling_zscore(vol_13w, 104)

    # 2. DXY信号
    dxy = _dxy(con).resample("W-FRI").last()
    dxy_z = rolling_zscore(dxy, 104)

    # 3. 库存信号
    lme = _tfapi(con, "LME库存总计_铜").resample("W-FRI").last()
    lme_z = rolling_zscore(lme, 156)

    # 4. 对齐 (只用DXY可用的时间段)
    common = close_w.index.intersection(vol_13w_z.dropna().index).intersection(
        dxy_z.dropna().index
    ).intersection(lme_z.dropna().index)

    vol_z = vol_13w_z.reindex(common)
    dxy_z_val = dxy_z.reindex(common)
    lme_z_val = lme_z.reindex(common)
    ret = close_w.pct_change().reindex(common)

    # 5. 仓位映射 (VIX + DXY)
    pos = pd.Series(0.5, index=common)
    vix_bull = vol_z > 0
    dxy_bull = dxy_z_val < 0

    pos[vix_bull & dxy_bull] = 1.0    # VIX多+DXY多: 满仓
    pos[vix_bull & ~dxy_bull] = 0.7   # VIX多+DXY空: 重仓
    pos[~vix_bull & dxy_bull] = 0.7   # VIX空+DXY多: 重仓
    pos[~vix_bull & ~dxy_bull] = 0.4  # VIX空+DXY空: 轻仓

    # 6. 库存调整
    pos[lme_z_val < 0] = pos[lme_z_val < 0].clip(upper=1.0)
    pos[lme_z_val > 0] = pos[lme_z_val > 0].clip(lower=0.0)

    # 7. 修复: 第一个周期仓位设为0
    pos.iloc[0] = 0.0

    # 8. 生成完整信号 (DXY不可用时用NaN)
    full_signal = pd.DataFrame({
        "M": close_w,
        "vol_z": vol_13w_z.reindex(close_w.index),
        "dxy_z": dxy_z.reindex(close_w.index),
        "lme_z": lme_z.reindex(close_w.index),
        "vix_bull": vol_13w_z.reindex(close_w.index).gt(0).astype(float),
        "dxy_bull": dxy_z.reindex(close_w.index).lt(0).astype(float),
        "position": np.nan,
    })
    full_signal.loc[common, "position"] = pos

    return full_signal


def backtest_v4(signal: pd.DataFrame, close: pd.Series,
                cost_bps: float = 5.0) -> dict:
    """回测铜模型v4.

    注意: DXY数据只有2016-12-16开始的506周, 而铜价有1129周。
    策略: DXY不可用时仓位为0, DXY可用时使用v4信号。
    """
    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    # 仓位为0时收益为0 (完全空仓)
    strat_ret = pos.shift(1) * ret
    pos_change = pos.diff().abs()
    cost = pos_change * cost_bps / 10000
    strat_ret = strat_ret - cost
    strat_ret = strat_ret.dropna()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0,
            "total_ret": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "calmar": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1 if len(nav) > 0 else 0
    n_weeks = len(strat_ret)
    n_years = n_weeks / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (nav / nav.cummax() - 1).min() if len(nav) > 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (strat_ret > 0).mean()

    return {
        "n_weeks": n_weeks,
        "total_ret": total_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "nav": nav,
    }


__all__ = ["build_signal_v4", "backtest_v4"]
