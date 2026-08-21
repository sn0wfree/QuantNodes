# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_factors.py

v6.3 多因子体系 - 因子计算层

12个独立因子，每个因子单独计算，互不依赖。

因子分组:
    动量组: mom_2w, mom_13w
    宏观组: dxy_z, pmi, industry_dz
    库存组: lme_z, quadrant_pos
    期限结构组: carry, curvature
    波动率组: vol_13w_z, vol_pct
    跨资产组: cu_crude
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def _rolling_zscore(s: pd.Series, w: int = 52) -> pd.Series:
    """滚动z-score."""
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def _load_weekly_close(con, symbol: str = "CU0") -> pd.Series:
    """加载周频收盘价."""
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc",
           "M0": "futures_main_close_m"}[symbol]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last()


def _load_fundamental(con, col: str) -> pd.Series:
    """加载基本面数据."""
    df = con.execute(
        f"SELECT trade_date, value FROM fundamentals_daily "
        f"WHERE value_col='{col}' ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


def _load_tfapi(con, name: str) -> pd.Series:
    """加载tfapi指标."""
    df = con.execute(
        f"SELECT trade_date, value FROM tfapi_indicators "
        f"WHERE index_name='{name}' ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


# ============================================================
# 动量组因子
# ============================================================

def compute_mom_2w(close_w: pd.Series) -> pd.Series:
    """2周动量 (IC=0.71, 最强预测因子).

    逻辑: 2周累计收益
    方向: 正=做多
    """
    ret_w = close_w.pct_change()
    return ret_w.rolling(2).sum().rename("mom_2w")


def compute_mom_13w(close_w: pd.Series) -> pd.Series:
    """13周动量 (IC=0.31).

    逻辑: 13周累计收益
    方向: 正=做多
    """
    ret_w = close_w.pct_change()
    return ret_w.rolling(13).sum().rename("mom_13w")


# ============================================================
# 宏观组因子
# ============================================================

def compute_dxy_z(con, close_w: pd.Series) -> pd.Series:
    """美元指数z-score.

    逻辑: 美元走弱利好大宗商品
    方向: 负=做多
    数据: fundamentals_daily (dxy_fred 或 dxy)
    """
    try:
        dxy = _load_fundamental(con, "dxy_fred")
    except Exception:
        dxy = _load_fundamental(con, "dxy")
    dxy_w = dxy.resample("W-FRI").last().reindex(close_w.index).ffill()
    return _rolling_zscore(dxy_w, 104).rename("dxy_z")


def compute_pmi(con, close_w: pd.Series) -> pd.Series:
    """PMI z-score.

    逻辑: PMI扩张利好铜需求
    方向: 正=做多
    数据: fundamentals_daily (pmi_mfg)
    """
    pmi = _load_fundamental(con, "pmi_mfg")
    pmi_w = pmi.resample("W-FRI").last().reindex(close_w.index).ffill()
    return _rolling_zscore(pmi_w, 52).rename("pmi")


def compute_industry_dz(con, close_w: pd.Series) -> pd.Series:
    """钻机数动量 (行业景气).

    逻辑: 钻机数上升=能源行业扩张=利好铜
    方向: 正=做多
    数据: fundamentals_daily (us_oil_rig)
    """
    rig = _load_fundamental(con, "us_oil_rig")
    rig_w = rig.resample("W-FRI").last().reindex(close_w.index).ffill()
    rig_z = _rolling_zscore(rig_w, 52)
    return rig_z.diff(4).rename("industry_dz")


# ============================================================
# 库存组因子
# ============================================================

def compute_lme_z(con, close_w: pd.Series) -> pd.Series:
    """LME库存z-score.

    逻辑: 库存下降=供给收紧=利好铜
    方向: 负=做多
    数据: tfapi_indicators (LME库存总计_铜)
    """
    lme = _load_tfapi(con, "LME库存总计_铜")
    lme_w = lme.resample("W-FRI").last().reindex(close_w.index).ffill()
    return _rolling_zscore(lme_w, 156).rename("lme_z")


def compute_quadrant_pos(con, close_w: pd.Series) -> pd.Series:
    """库存-价格四象限.

    逻辑: 根据库存变化和价格变化方向判断象限
    方向: 被动去库=1.0, 主动补库=0.7, 主动去库=0.4, 被动补库=0.1
    数据: tfapi_indicators + 价格
    """
    lme = _load_tfapi(con, "LME库存总计_铜")
    lme_w = lme.resample("W-FRI").last().reindex(close_w.index).ffill()

    inv_z = _rolling_zscore(lme_w, 52)
    pr_z = _rolling_zscore(close_w, 52)

    d_inv = inv_z.diff(4) * (-1)  # 库存下降=正
    d_pr = pr_z.diff(4)           # 价格上升=正

    # 四象限分类
    pos = pd.Series(0.5, index=close_w.index)  # 默认中性
    pos[(d_inv > 0) & (d_pr > 0)] = 1.0  # 被动去库
    pos[(d_inv < 0) & (d_pr > 0)] = 0.7  # 主动补库
    pos[(d_inv > 0) & (d_pr < 0)] = 0.4  # 主动去库
    pos[(d_inv < 0) & (d_pr < 0)] = 0.1  # 被动补库

    return pos.rename("quadrant_pos")


# ============================================================
# 期限结构组因子
# ============================================================

def compute_carry(con, close_w: pd.Series) -> pd.Series:
    """期限结构carry状态.

    逻辑: backwardation=做多, contango=空仓
    方向: back=1.0, neutral=0.5, contango=0
    数据: term_structure_cu
    """
    ts = con.execute(
        "SELECT trade_date, spread_1_2, close_near2 "
        "FROM term_structure_cu ORDER BY trade_date"
    ).df()
    ts["trade_date"] = pd.to_datetime(ts["trade_date"])
    ts = ts.set_index("trade_date").resample("W-FRI").last()

    slope = ts["spread_1_2"] / ts["close_near2"]
    zscore = _rolling_zscore(slope, 52).reindex(close_w.index).ffill()

    # 3状态: z < -0.5 → 1.0, z > 0.5 → 0, else 0.5
    pos = pd.Series(0.5, index=close_w.index)
    pos[zscore < -0.5] = 1.0
    pos[zscore > 0.5] = 0.0

    return pos.rename("carry")


def compute_curvature(con, close_w: pd.Series) -> pd.Series:
    """期限结构曲率z-score.

    逻辑: 正曲率=期限结构凸=利好
    方向: 正=做多
    数据: term_structure_cu
    """
    ts = con.execute(
        "SELECT trade_date, close_near, close_near2, close_near3 "
        "FROM term_structure_cu ORDER BY trade_date"
    ).df()
    ts["trade_date"] = pd.to_datetime(ts["trade_date"])
    ts = ts.set_index("trade_date").resample("W-FRI").last()

    curvature = (ts["close_near"] - 2 * ts["close_near2"] + ts["close_near3"]) / ts["close_near"]
    curv_z = _rolling_zscore(curvature, 52).reindex(close_w.index).ffill()

    # z-score → position: 正=1.0, 负=0, 中性=0.5
    pos = pd.Series(0.5, index=close_w.index)
    pos[curv_z > 0.5] = 1.0
    pos[curv_z < -0.5] = 0.0

    return pos.rename("curvature")


# ============================================================
# 波动率组因子
# ============================================================

def compute_vol_13w_z(close_w: pd.Series) -> pd.Series:
    """13周波动率z-score.

    逻辑: 高波动=动量效应=做多
    方向: 正=做多
    """
    ret_w = close_w.pct_change()
    vol_13w = ret_w.rolling(13).std() * np.sqrt(52)
    return _rolling_zscore(vol_13w, 104).rename("vol_13w_z")


def compute_vol_pct(close_w: pd.Series) -> pd.Series:
    """4周波动率分位数.

    逻辑: 低波=适合动量, 高波=适合反转/减仓
    方向: 低分位=做多
    """
    ret_w = close_w.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    vol_pct = vol_4w.rolling(52).rank(pct=True)

    # 分位数 → position: 低(<0.3)=1.0, 高(>0.7)=0, else 0.5
    # 注意: vol_pct低=低波动=做多, 所以方向是反的
    pos = pd.Series(0.5, index=close_w.index)
    pos[vol_pct < 0.3] = 1.0  # 低波动=做多
    pos[vol_pct > 0.7] = 0.0  # 高波动=空仓

    return pos.rename("vol_pct")


# ============================================================
# 跨资产组因子
# ============================================================

def compute_cu_crude(con, close_w: pd.Series) -> pd.Series:
    """铜油比20周动量.

    逻辑: 铜相对原油走强=经济向好=利好铜
    方向: 正=做多
    数据: futures_main_close_sc (原油)
    """
    crude = con.execute(
        "SELECT trade_date, close FROM futures_main_close_sc ORDER BY trade_date"
    ).df()
    crude["trade_date"] = pd.to_datetime(crude["trade_date"])
    crude_w = crude.set_index("trade_date")["close"].resample("W-FRI").last()

    cu_crude_ratio = close_w / crude_w.reindex(close_w.index).ffill()
    mom20 = cu_crude_ratio.pct_change(20)

    # 动量 → position: 正=1.0, 强负(<-0.05)=0, else 0.5
    pos = pd.Series(0.5, index=close_w.index)
    pos[mom20 > 0] = 1.0
    pos[mom20 < -0.05] = 0.0

    return pos.rename("cu_crude")


# ============================================================
# 主函数: 计算所有因子
# ============================================================

def compute_all_factors(con=None) -> pd.DataFrame:
    """计算所有12个因子.

    返回: DataFrame, index=周频日期, columns=12个因子名
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    close_w = _load_weekly_close(con, "CU0")
    close_w = close_w.dropna()

    factors = pd.DataFrame(index=close_w.index)

    # 动量组
    factors["mom_2w"] = compute_mom_2w(close_w)
    factors["mom_13w"] = compute_mom_13w(close_w)

    # 宏观组
    factors["dxy_z"] = compute_dxy_z(con, close_w)
    factors["pmi"] = compute_pmi(con, close_w)
    factors["industry_dz"] = compute_industry_dz(con, close_w)

    # 库存组
    factors["lme_z"] = compute_lme_z(con, close_w)
    factors["quadrant_pos"] = compute_quadrant_pos(con, close_w)

    # 期限结构组
    factors["carry"] = compute_carry(con, close_w)
    factors["curvature"] = compute_curvature(con, close_w)

    # 波动率组
    factors["vol_13w_z"] = compute_vol_13w_z(close_w)
    factors["vol_pct"] = compute_vol_pct(close_w)

    # 跨资产组
    factors["cu_crude"] = compute_cu_crude(con, close_w)

    return factors


__all__ = [
    "compute_all_factors",
    "compute_mom_2w",
    "compute_mom_13w",
    "compute_dxy_z",
    "compute_pmi",
    "compute_industry_dz",
    "compute_lme_z",
    "compute_quadrant_pos",
    "compute_carry",
    "compute_curvature",
    "compute_vol_13w_z",
    "compute_vol_pct",
    "compute_cu_crude",
]
