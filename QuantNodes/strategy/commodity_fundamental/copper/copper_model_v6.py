# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_model_v6.py

铜模型v6: v4 + v5精简版 仓位合并 + carry组合 (2026-08-21, 统一真实价口径).

背景 (scripts/cmf/11_copper_ablation.py 消融实验结论, 统一2018+真实价口径):
    1. v5的经济状态层为负贡献 (v5p去掉后: IS 0.51→0.60, OOS 0.86→0.96),
       v6调用v5时禁用 (use_econ=False)
    2. v4/v5/pv3同源相关0.79-0.83, 合并方式=仓位等权平均
    3. avg(v4,v5p): IS Sharpe 0.70 ≈ v4单独(0.69), MaxDD -18.6% 优于两者
       (v4 -20.4% / v5p -16.6%), OOS 0.77 → 合并价值=回撤改善+双引擎稳健
    4. pv3依赖洋山溢价数据(起点2019-05)且加入会稀释, 不进铜主栈
    5. carry叠加(walk-forward波动率倒数): IS 0.69→0.58 但 MaxDD -18.6→-15.8;
       carry按架构决策保持在组合层独立配置, 故默认 with_carry=False

信号起点: 2010-06 (v4实际起点2010-04, docstring中2016-12的说法过时)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .copper_model_v4 import build_signal_v4
from .copper_model_v5 import build_signal_v5
from .copper_carry import build_signal_carry

DB = Path("data/cmf/fundamental.duckdb")


def _load_close_w(con, symbol: str = "CU0") -> pd.Series:
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc",
           "M0": "futures_main_close_m"}[symbol]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last()


def _trailing_vol(pos: pd.Series, ret: pd.Series, window: int = 52) -> pd.Series:
    """滚动估计策略年化波动 (仅用t-1及之前数据, 无前视)."""
    strat_ret = pos.shift(1) * ret
    return strat_ret.rolling(window).std() * np.sqrt(52)


def build_signal_v6(symbol: str = "CU0", con=None,
                    with_carry: bool = False,
                    vol_weight_window: int = 52) -> pd.DataFrame:
    """构建v6信号: main = avg(v4仓位, v5精简仓位), 可选叠加carry.

    with_carry=True 时以滚动52周波动率倒数权重叠加carry (shift防前视),
    carry数据不可用周权重自动重归main. 默认False (carry在组合层独立配置).

    返回列: M(真实周收盘) / pos_v4 / pos_v5p / pos_main / pos_carry / w_carry / position
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    close_w = _load_close_w(con, symbol)
    ret = close_w.pct_change()

    pos_v4 = build_signal_v4(symbol, con)["position"].dropna()
    pos_v5p = build_signal_v5(symbol, con, use_econ=False)["position"].dropna()

    common = pos_v4.index.intersection(pos_v5p.index)
    p4 = pos_v4.reindex(common).fillna(0)
    p5 = pos_v5p.reindex(common).fillna(0)
    pos_main = ((p4 + p5) / 2).clip(lower=0.0, upper=1.0)
    pos_main.iloc[0] = 0.0

    out = pd.DataFrame({
        "M": close_w.reindex(common),
        "pos_v4": p4,
        "pos_v5p": p5,
        "pos_main": pos_main,
    })

    if not with_carry:
        out["pos_carry"] = np.nan
        out["w_carry"] = 0.0
        out["position"] = pos_main
        return out

    pos_carry = build_signal_carry(con)["position"].dropna()
    idx = common.intersection(pos_carry.index)
    main_i = pos_main.reindex(idx).fillna(0)
    carry_i = pos_carry.reindex(idx).fillna(0)
    ret_i = ret.reindex(idx)

    # carry 可用区间内: 滚动波动率倒数加权 (shift(1)防前视)
    vol_main = _trailing_vol(main_i, ret_i, vol_weight_window)
    vol_carry = _trailing_vol(carry_i, ret_i, vol_weight_window)
    inv_m = (1.0 / vol_main).replace([np.inf, -np.inf], np.nan)
    inv_c = (1.0 / vol_carry).replace([np.inf, -np.inf], np.nan)
    w_carry = (inv_c / (inv_m + inv_c)).fillna(0.0).clip(0.0, 1.0).shift(1).fillna(0.0)

    pos_combined = (w_carry * carry_i + (1.0 - w_carry) * main_i).clip(0.0, 1.0)

    # carry不可用时段 (2018前 / 数据末端之后): 重归main
    carry_avail = pos_carry.reindex(common).notna()
    final = pd.Series(index=common, dtype=float)
    final[~carry_avail] = pos_main[~carry_avail]
    final[carry_avail] = pos_combined.reindex(common)[carry_avail]
    final.iloc[0] = 0.0

    out = out.assign(
        pos_carry=pos_carry.reindex(common),
        w_carry=w_carry.reindex(common).fillna(0.0),
        position=final,
    )
    return out


def backtest_v6(signal: pd.DataFrame, close: pd.Series = None,
                cost_bps: float = 5.0) -> dict:
    """v6回测 (真实价口径; close可省略, 默认用signal['M'])."""
    if close is None:
        close = signal["M"]
    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "total_ret": 0, "ann_ret": 0, "ann_vol": 0,
                "sharpe": 0, "max_dd": 0, "calmar": 0, "win_rate": 0,
                "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {"n_weeks": len(strat_ret), "total_ret": total_ret, "ann_ret": ann_ret,
            "ann_vol": ann_vol, "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            "max_dd": max_dd, "calmar": ann_ret / abs(max_dd) if max_dd != 0 else 0,
            "win_rate": (strat_ret > 0).mean(), "nav": nav}


__all__ = ["build_signal_v6", "backtest_v6", "apply_bull_enhance"]


def _load_tfapi(con, name: str) -> pd.Series:
    df = con.execute(
        f"SELECT trade_date d, value v FROM tfapi_indicators "
        f"WHERE index_name='{name}' ORDER BY d"
    ).df()
    df["d"] = pd.to_datetime(df["d"])
    return df.set_index("d")["v"]


def _rolling_z(s: pd.Series, w: int = 52) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def apply_bull_enhance(signal: pd.DataFrame, con,
                       threshold: int = 2,
                       bull_mom_weeks: int = 26,
                       bull_mom_thresh: float = 0.10,
                       target_vol: float = 0.12,
                       max_leverage: float = 1.0) -> pd.DataFrame:
    """牛市增强层 + 波动率目标化 (进攻型, 2026-08-21).

    逻辑:
        1. 牛市(26周动量>10%) + 四因子多数票确认 → 仓位提升至0.9
        2. 波动率目标化: 仓位按 trailing 4w波动率 缩放至目标波动率

    四因子 (探针确认):
        1. 全球库存负z52 (LME+COMEX+SHFE低库存=供给紧)
        2. 期限结构spread_1_3 back (远月backwardation=远期看涨)
        3. TC粗炼费Δ13w<0 (矿端收紧=长周期利多)
        4. 铜油比20w动量>0 (铜相对原油走强=经济向好)

    参数:
        target_vol: 目标年化波动率 (默认0.12)
        max_leverage: 最大仓位倍数 (1.0=无杠杆, 1.5=50%杠杆, 2.0=100%杠杆)

    返回: 含 bull_score / position_enhanced / position_sized / position_sized_lev
    """
    out = signal.copy()
    close_w = out["M"]
    pos = out["position"].fillna(0)

    # 26周动量 → 牛市判定
    mom26 = close_w.pct_change(bull_mom_weeks)
    bull = mom26 > bull_mom_thresh

    # 因子1: 全球库存负z
    lme = _load_tfapi(con, "LME库存总计_铜").resample("W-FRI").last()
    cmx = _load_tfapi(con, "COMEX:铜:库存").resample("W-FRI").last()
    shfe = _load_tfapi(con, "SHFE库存小计").resample("W-FRI").last()
    globe = (lme + cmx.reindex(lme.index).fillna(0)
             + shfe.reindex(lme.index).fillna(0))
    globe_z = _rolling_z(globe).reindex(close_w.index).ffill()

    # 因子2: spread_1_3 back
    ts = con.execute(
        "SELECT trade_date, spread_1_3, close_near2 "
        "FROM term_structure_cu ORDER BY trade_date"
    ).df()
    ts["trade_date"] = pd.to_datetime(ts["trade_date"])
    ts = ts.set_index("trade_date").resample("W-FRI").last()
    s13 = ts["spread_1_3"] / ts["close_near2"]
    s13_z = _rolling_z(s13).reindex(close_w.index).ffill()

    # 因子3: TC粗炼费 Δ13w < 0 (矿紧)
    tc = _load_tfapi(con, "现货:中国铜冶炼厂:粗炼费(TC)").resample("W-FRI").last()
    tc_d13 = tc.diff(13).reindex(close_w.index).ffill()

    # 因子4: 铜油比动量 > 0 (铜相对原油走强=经济向好)
    crude = con.execute(
        "SELECT trade_date, close FROM futures_main_close_sc ORDER BY trade_date"
    ).df()
    crude["trade_date"] = pd.to_datetime(crude["trade_date"])
    crude = crude.set_index("trade_date")["close"].resample("W-FRI").last()
    cu_crude_ratio = (close_w / crude.reindex(close_w.index).ffill())
    cu_crude_mom20 = cu_crude_ratio.pct_change(20).reindex(close_w.index).ffill()

    # 投票 (方向已调整: 库存低=-z, back=-z, TC降=Δ<0, 铜油比升=mom>0 → 均为因子值<0时看多)
    bull_score = (
        (globe_z < 0).astype(int)
        + (s13_z < 0).astype(int)
        + (tc_d13 < 0).astype(int)
    )

    # 增强: 牛市+多数票确认 → 仓位提升至0.9
    enhance = bull & (bull_score >= threshold)
    pos_enh = pos.copy()
    pos_enh[enhance] = np.maximum(pos_enh[enhance], 0.9)

    # 铜油比动量调整: 铜相对原油走强时额外加仓
    cu_crude_boost = (cu_crude_mom20 > 0.05).astype(float) * 0.1  # 强势时+10%
    pos_enh = (pos_enh + cu_crude_boost).clip(upper=1.0)
    pos_enh.iloc[0] = 0.0

    # 波动率目标化 (13_copper_alpha_probe_v2 发现: 目标12%→Sharpe 0.72, MaxDD -12.3%)
    ret_w = close_w.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0)

    # 无杠杆版: scale clip=1.0, 仓位 clip=1.0
    pos_sized = (pos_enh * scale.clip(upper=1.0)).clip(upper=1.0)
    pos_sized.iloc[0] = 0.0

    # 杠杆版: scale不clip上界, 仓位 clip=max_leverage
    pos_sized_lev = (pos_enh * scale).clip(upper=max_leverage)
    pos_sized_lev.iloc[0] = 0.0

    out["bull_score"] = bull_score
    out["position_enhanced"] = pos_enh
    out["position_sized"] = pos_sized
    out["position_sized_lev"] = pos_sized_lev
    return out


__all__ = ["build_signal_v6", "backtest_v6", "apply_bull_enhance"]
