# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_pricing.py

铜多因子定价模型 — 四层信号系统:
  1. 波动率状态 (权重40%) — 波动率聚集效应
  2. 库存周期 (权重30%) — LME库存变化
  3. 宏观环境 (权重20%) — DXY/TIPS
  4. 供需信号 (权重10%) — 洋山溢价/TC

核心发现:
  - 波动率聚集是最强特征 (自相关0.82)
  - 因子预测能力: TC(0.109) > SHFE库存(-0.112) > TIPS(0.051)
  - 因子相关性: TC↔LME库存=-0.666, TC↔钻机数=0.818
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


# ═══════════════════════════════════════════════════════════════════════════════
# 数据层
# ═══════════════════════════════════════════════════════════════════════════════

def load_copper_data(con=None) -> dict[str, pd.Series]:
    """加载铜相关所有因子数据.

    返回: dict of {name: pd.Series(index=trade_date, freq=W-FRI)}
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    def _load_sql(sql: str) -> pd.Series:
        df = con.execute(sql).df()
        if df.empty:
            return pd.Series(dtype=float)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        return df["value"]

    def _load_tfapi(name: str) -> pd.Series:
        df = con.execute(
            f"SELECT trade_date, value FROM tfapi_indicators WHERE index_name='{name}' ORDER BY trade_date"
        ).df()
        if df.empty:
            return pd.Series(dtype=float)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        return df["value"]

    # 期货价格
    df = con.execute("SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    close = df.set_index("trade_date")["close"]

    # 天风因子
    premium = _load_tfapi("洋山铜溢价")
    tc = _load_tfapi("现货:中国铜冶炼厂:粗炼费(TC)")
    lme = _load_tfapi("LME库存总计_铜")
    comex = _load_tfapi("COMEX:铜:库存")
    shfe = _load_tfapi("SHFE库存小计")

    # 宏观因子
    dxy = _load_sql("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='dxy' ORDER BY trade_date")
    tips = _load_sql("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='tips10y' ORDER BY trade_date")

    # 钻机数
    rig = _load_sql("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='us_oil_rig' ORDER BY trade_date")

    # PMI
    pmi = _load_sql("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='pmi_mfg' ORDER BY trade_date")

    # 对齐到周五频率 (先对齐，再重采样)
    data = {}
    data["close"] = close.resample("W-FRI").last()
    data["premium"] = premium.resample("W-FRI").last()
    data["tc"] = tc.resample("W-FRI").last()
    data["lme_inv"] = lme.resample("W-FRI").last()
    data["comex_inv"] = comex.resample("W-FRI").last()
    data["shfe_inv"] = shfe.resample("W-FRI").last()

    # DXY和TIPS需要特殊处理: ffill到每个周五
    if len(dxy) > 0:
        dxy_weekly = dxy.resample("W-FRI").ffill()
        data["dxy"] = dxy_weekly
    else:
        data["dxy"] = pd.Series(dtype=float)

    if len(tips) > 0:
        tips_weekly = tips.resample("W-FRI").ffill()
        data["tips"] = tips_weekly
    else:
        data["tips"] = pd.Series(dtype=float)

    data["rig"] = rig.resample("W-FRI").last()
    data["pmi"] = pmi.resample("W-FRI").last()

    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 第一层: 波动率状态
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vol_regime(close: pd.Series, vol_window: int = 4,
                       pct_window: int = 52) -> pd.Series:
    """波动率状态: 基于波动率聚集效应.

    返回: +1 (低波动,趋势延续) / 0 (中波动) / -1 (高波动,反转风险)
    """
    ret = close.pct_change()
    vol = ret.rolling(vol_window).std() * np.sqrt(52)
    vol_pct = vol.rolling(pct_window).rank(pct=True)

    regime = pd.Series(0.0, index=close.index, name="vol_regime")
    regime[vol_pct < 0.3] = 1.0   # 低波动: 趋势延续
    regime[vol_pct > 0.7] = -1.0  # 高波动: 反转风险

    return regime


# ═══════════════════════════════════════════════════════════════════════════════
# 第二层: 库存周期
# ═══════════════════════════════════════════════════════════════════════════════

def compute_inventory_cycle(lme_inv: pd.Series, z_window: int = 104,
                           delta_window: int = 4) -> pd.Series:
    """库存周期: 基于LME库存z-score和变化方向 (四象限).

    返回: +1.0 (被动去库) / +0.7 (主动补库) / +0.4 (主动去库) / +0.1 (被动补库)
    """
    z = (lme_inv - lme_inv.rolling(z_window).mean()) / lme_inv.rolling(z_window).std()
    dz = z.diff(delta_window)

    cycle = pd.Series(0.5, index=lme_inv.index, name="inventory_cycle")
    # 四象限映射 (基于库存变化方向 + 价格变化方向的逻辑)
    # 被动去库: 库存下降(dz<0) → 利多 (+1.0)
    # 主动补库: 库存上升(dz>0) 但库存水平低(z<0) → 趋势延续 (+0.7)
    # 主动去库: 库存下降(dz<0) 但库存水平高(z>0) → 趋势反转 (+0.4)
    # 被动补库: 库存上升(dz>0) → 利空 (+0.1)
    cycle[(dz < 0) & (z < 0)] = 1.0   # 被动去库
    cycle[(dz > 0) & (z < 0)] = 0.7   # 主动补库
    cycle[(dz < 0) & (z > 0)] = 0.4   # 主动去库
    cycle[(dz > 0) & (z > 0)] = 0.1   # 被动补库

    return cycle


# ═══════════════════════════════════════════════════════════════════════════════
# 第三层: 宏观环境
# ═══════════════════════════════════════════════════════════════════════════════

def compute_macro_env(dxy: pd.Series, tips: pd.Series,
                      z_window: int = 104) -> pd.Series:
    """宏观环境: DXY/TIPS z-score.

    返回: +1 (宏观友好) / 0 (宏观中性) / -1 (宏观不利)
    """
    dxy_z = (dxy - dxy.rolling(z_window).mean()) / dxy.rolling(z_window).std()
    tips_z = (tips - tips.rolling(z_window).mean()) / tips.rolling(z_window).std()

    env = pd.Series(0.0, index=dxy.index, name="macro_env")
    env[(dxy_z < 0) & (tips_z < 0)] = 1.0   # 宏观友好
    env[(dxy_z > 0) & (tips_z > 0)] = -1.0  # 宏观不利

    return env


# ═══════════════════════════════════════════════════════════════════════════════
# 第四层: 供需信号
# ═══════════════════════════════════════════════════════════════════════════════

def compute_supply_demand(premium: pd.Series, tc: pd.Series,
                          z_window: int = 104) -> pd.Series:
    """供需信号: 洋山溢价/TC z-score.

    返回: +1 (供需紧张) / 0 (供需中性) / -1 (供需宽松)
    """
    premium_z = (premium - premium.rolling(z_window).mean()) / premium.rolling(z_window).std()
    tc_z = (tc - tc.rolling(z_window).mean()) / tc.rolling(z_window).std()

    sd = pd.Series(0.0, index=premium.index, name="supply_demand")
    sd[(premium_z > 0) & (tc_z < 0)] = 1.0   # 供需紧张
    sd[(premium_z < 0) & (tc_z > 0)] = -1.0  # 供需宽松

    return sd


# ═══════════════════════════════════════════════════════════════════════════════
# 信号合成
# ═══════════════════════════════════════════════════════════════════════════════

def compute_copper_signal(data: dict[str, pd.Series],
                          vol_weight: float = 0.4,
                          inv_weight: float = 0.3,
                          macro_weight: float = 0.2,
                          sd_weight: float = 0.1) -> pd.DataFrame:
    """铜定价模型: 四层信号合成.

    返回: DataFrame with columns [vol_regime, inventory_cycle, macro_env,
          supply_demand, total_score, position]
    """
    close = data["close"]
    common_idx = close.dropna().index

    # 对齐所有因子到共同索引
    for key in ["premium", "tc", "lme_inv", "dxy", "tips"]:
        if key in data and len(data[key]) > 0:
            common_idx = common_idx.intersection(data[key].dropna().index)

    # 计算四层信号
    vol_regime = compute_vol_regime(close).reindex(common_idx)
    inv_cycle = compute_inventory_cycle(data["lme_inv"]).reindex(common_idx)
    macro_env = compute_macro_env(data["dxy"], data["tips"]).reindex(common_idx)
    sd_signal = compute_supply_demand(data["premium"], data["tc"]).reindex(common_idx)

    # 合成得分
    total_score = (vol_weight * vol_regime +
                   inv_weight * inv_cycle +
                   macro_weight * macro_env +
                   sd_weight * sd_signal)

    # 仓位映射
    position = pd.Series(0.5, index=common_idx, name="position")
    position[total_score >= 0.5] = 1.0
    position[(total_score >= 0.2) & (total_score < 0.5)] = 0.7
    position[(total_score >= -0.2) & (total_score < 0.2)] = 0.4
    position[total_score < -0.2] = 0.1

    result = pd.DataFrame({
        "vol_regime": vol_regime,
        "inventory_cycle": inv_cycle,
        "macro_env": macro_env,
        "supply_demand": sd_signal,
        "total_score": total_score,
        "position": position,
    }, index=common_idx)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 回测
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_copper(signal: pd.DataFrame, close: pd.Series,
                    cost_bps: float = 5.0) -> dict:
    """回测铜定价模型.

    返回: dict with metrics
    """
    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    # 策略收益
    strat_ret = pos.shift(1) * ret

    # 交易成本
    pos_change = pos.diff().abs()
    cost = pos_change * cost_bps / 10000
    strat_ret = strat_ret - cost

    # 指标计算
    strat_ret = strat_ret.dropna()
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


__all__ = [
    "load_copper_data",
    "compute_vol_regime",
    "compute_inventory_cycle",
    "compute_macro_env",
    "compute_supply_demand",
    "compute_copper_signal",
    "backtest_copper",
]