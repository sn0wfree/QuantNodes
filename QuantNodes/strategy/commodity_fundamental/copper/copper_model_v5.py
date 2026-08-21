# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_model_v5.py

铜模型v5: 波动率状态 + 2周动量 + 经济状态 + 报告逻辑

核心改进:
  1. 波动率状态: 低波→动量, 高波→反转
  2. 2周动量: IC=0.71 (最强预测因子)
  3. 经济状态: 区分"熊陡/牛陡"对铜价的影响
  4. 宏观趋势: DXY↓ + PMI↑ → 加仓
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ..kalman_filter import weekly_kalman_center
from ..sign_constrained_lasso import (
    walk_forward_lasso, rolling_phy_center, compute_valuation_gap, rolling_zscore,
)
from ..inventory_quadrant import inventory_state_for_product
from ..valuation_gap import industry_momentum, apply_valuation_layer
from ..crowding_simplified import vol_percentile, apply_crowding_override
from .copper_economy import get_economy_signal

DB = Path("data/cmf/fundamental.duckdb")


def _close(symbol: str, con) -> pd.Series:
    """加载期货收盘价."""
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc", "M0": "futures_main_close_m"}[symbol]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].rename(symbol)


def _dxy(con) -> pd.Series:
    """DXY美元指数."""
    df_fred = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col='dxy_fred' ORDER BY trade_date
    """).df()
    if len(df_fred) > 0:
        df_fred["trade_date"] = pd.to_datetime(df_fred["trade_date"])
        return df_fred.set_index("trade_date")["value"].rename("dxy")
    df_ak = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col='dxy' ORDER BY trade_date
    """).df()
    df_ak["trade_date"] = pd.to_datetime(df_ak["trade_date"])
    return df_ak.set_index("trade_date")["value"].rename("dxy")


def _tips10y(con) -> pd.Series:
    """TIPS 10Y."""
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='tips10y' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("tips10y")


def _tfapi(con, index_name: str) -> pd.Series:
    """天风API指标."""
    df = con.execute(
        f"SELECT trade_date, value FROM tfapi_indicators WHERE index_name='{index_name}' ORDER BY trade_date"
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


# 导出到模块级别，供其他脚本使用
from ..pipeline_v3 import _tfapi as _tfapi_from_v3


def _rig(con) -> pd.Series:
    """钻机数."""
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='us_oil_rig' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("rig")


def _pmi(con) -> pd.Series:
    """PMI."""
    df = con.execute("SELECT trade_date, value FROM fundamentals_daily WHERE value_col='pmi_mfg' ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("pmi")


def _term_structure(con) -> pd.DataFrame:
    """期限结构数据 (2018+)."""
    df = con.execute("""
        SELECT trade_date, close_near, close_near2, spread_1_2
        FROM term_structure_cu ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")


def build_signal_v5(symbol: str = "CU0", con=None, use_econ: bool = True) -> pd.DataFrame:
    """铜模型v5: 波动率状态 + 2周动量 + 报告逻辑.

    核心改进:
    1. 波动率状态: 低波→动量, 高波→反转
    2. 2周动量: IC=0.71 (最强预测因子)
    3. 去掉y_lag1 (过拟合)

    use_econ=False: 跳过经济状态层 (消融实验11_copper_ablation显示该层负贡献:
                    IS Sharpe 0.47→0.42, v6以use_econ=False调用)
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # ── §2.1 卡尔曼滤波 ──────────────────────────────────────────────────────
    close = _close(symbol, con)
    close_w = close.resample("W-FRI").last()
    ret_w = close_w.pct_change()
    mu_market = weekly_kalman_center(close, q=0.015)

    # ── 因子数据 ──────────────────────────────────────────────────────────────
    dxy = _dxy(con).resample("W-FRI").last()
    tips = _tips10y(con).resample("W-FRI").last()
    lme = _tfapi(con, "LME库存总计_铜").resample("W-FRI").last()
    rig = _rig(con).resample("W-FRI").last()
    pmi = _pmi(con).resample("W-FRI").last().ffill()

    # ── 波动率状态 (核心发现: 低波→动量, 高波→反转) ──────────────────────
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    vol_pct = vol_4w.rolling(52).rank(pct=True)

    # ── 2周动量 (IC=0.71, 最强预测因子) ──────────────────────────────────
    mom_2w = ret_w.rolling(2).sum()

    # ── 13周动量 (IC=0.31, 中等) ──────────────────────────────────────────
    momentum_13w = ret_w.rolling(13).sum()

    # ── 宏观趋势因子 ──────────────────────────────────────────────────────
    # DXY趋势 (IC=-0.12, 最强宏观因子)
    dxy_ma20 = dxy.rolling(20).mean()
    dxy_trend = (dxy < dxy_ma20).astype(float)  # DXY下降=1 (看多铜)

    # PMI趋势 (IC=0.13, 中等宏观因子)
    pmi_ma20 = pmi.rolling(20).mean()
    pmi_trend = (pmi > pmi_ma20).astype(float)  # PMI扩张=1 (看多铜)

    # ── 对齐 ─────────────────────────────────────────────────────────────────
    common_idx = mu_market.index.intersection(dxy.dropna().index).intersection(
        tips.dropna().index
    ).intersection(lme.dropna().index).intersection(rig.dropna().index).intersection(
        pmi.dropna().index
    ).intersection(mom_2w.dropna().index).intersection(momentum_13w.dropna().index)

    # ── §2.2 符号约束Lasso ──────────────────────────────────────────────────
    # 宏观趋势因子: DXY趋势(IC=-0.12) + PMI趋势(IC=0.13)
    X = pd.DataFrame({
        "dxy_neg": -rolling_zscore(dxy, 104).reindex(common_idx),
        "tips_neg": -rolling_zscore(tips, 104).reindex(common_idx),
        "lme_neg": -rolling_zscore(lme, 52).reindex(common_idx),
        "rig_neg": -rolling_zscore(rig, 52).reindex(common_idx),
        "pmi": rolling_zscore(pmi, 52).reindex(common_idx),
        "mom_2w": rolling_zscore(mom_2w, 52).reindex(common_idx),
        "momentum_13w": rolling_zscore(momentum_13w, 52).reindex(common_idx),
        "dxy_trend": dxy_trend.reindex(common_idx),  # 宏观趋势
        "pmi_trend": pmi_trend.reindex(common_idx),  # 宏观趋势
    }).dropna(how="any")

    y = pd.Series(np.log(mu_market).diff().reindex(X.index)).dropna()
    common3 = X.index.intersection(y.index)
    X = X.loc[common3]
    y = y.loc[common3]
    mu_aligned = mu_market.reindex(common3)

    coefs, oos = walk_forward_lasso(X, y, refit_weeks=26, min_train_weeks=52)
    p_phy = rolling_phy_center(mu_aligned, oos["oos_pred"] if len(oos) else pd.Series(dtype=float), window=13)
    gap = compute_valuation_gap(mu_aligned, p_phy)

    # ── §3.1 库存象限 ──────────────────────────────────────────────────────
    inv_state = inventory_state_for_product(
        lme.reindex(common3).ffill(),
        close_w.reindex(common3),
        z_window=52,
        inv_dir=-1,
    )

    industry_dz = industry_momentum(rig.reindex(common3).ffill(), z_window=52, delta_window=4)

    # ── §3.2 估值层 ──────────────────────────────────────────────────────
    base_pos = apply_valuation_layer(
        inv_state["quadrant_pos"],
        gap.reindex(common3),
        industry_dz.reindex(common3),
        gap_low_threshold=0.01,
        gap_high_threshold=-0.01,
    )

    # ── 波动率状态 + 宏观趋势 + 经济状态调整仓位 ──────────────────────────
    vol_aligned = vol_pct.reindex(common3).fillna(0.5)
    mom_aligned = mom_2w.reindex(common3).fillna(0)
    dxy_trend_aligned = dxy_trend.reindex(common3).fillna(0)
    pmi_trend_aligned = pmi_trend.reindex(common3).fillna(0)

    # 经济状态信号 (从FRED获取, 使用4周最小持续时间约束; use_econ=False时跳过)
    econ_score_aligned = pd.Series(0.0, index=common3)
    if use_econ:
        econ_score, _ = get_economy_signal(con, min_duration=4)
        econ_score_aligned = econ_score.reindex(common3).fillna(0)

    final_pos = base_pos.copy()

    low_vol_mask = vol_aligned < 0.3
    high_vol_mask = vol_aligned > 0.7

    # 低波动 + 正动量: 加仓到1.0
    final_pos[low_vol_mask & (mom_aligned > 0)] = 1.0
    # 高波动: 仓位0.4 (保持保守)
    final_pos[high_vol_mask] = 0.4

    # 宏观趋势加成: DXY↓ + PMI↑ → 加仓
    macro_bull = (dxy_trend_aligned == 1) & (pmi_trend_aligned == 1)
    final_pos[macro_bull & (final_pos < 1.0)] = final_pos[macro_bull & (final_pos < 1.0)] + 0.3
    final_pos = final_pos.clip(upper=1.0)

    # 经济状态调整:
    # 扩张期 (+0.2): 经济状态score=1.0, 跟随趋势
    # 中性期 (-0.1): 经济状态score=0.0
    # 衰退期 (-0.2): 经济状态score=-1.0, 谨慎
    if use_econ:
        final_pos[econ_score_aligned > 0.5] = final_pos[econ_score_aligned > 0.5] + 0.2
        neutral = (econ_score_aligned > -0.5) & (econ_score_aligned < 0.5)
        final_pos[neutral] = final_pos[neutral] - 0.1
        final_pos[econ_score_aligned < -0.5] = final_pos[econ_score_aligned < -0.5] - 0.2
    final_pos = final_pos.clip(lower=0.0, upper=1.0)

    final_pos.iloc[0] = 0.0

    # ── 生成完整信号 ──────────────────────────────────────────────────────────
    full_signal = pd.DataFrame({
        "M": close_w.reindex(common3),
        "p_phy": p_phy.reindex(common3),
        "gap": gap.reindex(common3),
        "vol_pct": vol_aligned,
        "mom_2w": mom_aligned,
        "d_inv": inv_state["d_inv"].reindex(common3),
        "d_price": inv_state["d_price"].reindex(common3),
        "quadrant_pos": inv_state["quadrant_pos"].reindex(common3),
        "industry_dz": industry_dz.reindex(common3),
        "position": final_pos.reindex(common3),
    })

    return full_signal


def backtest_v5(signal: pd.DataFrame, close: pd.Series,
                cost_bps: float = 5.0) -> dict:
    """回测铜模型v5."""
    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    strat_ret = pos.shift(1) * ret
    pos_change = pos.diff().abs()
    cost = pos_change * cost_bps / 10000
    strat_ret = strat_ret - cost
    strat_ret = strat_ret.dropna()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0, "total_ret": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "calmar": 0, "win_rate": 0,
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
        "n_weeks": n_weeks, "total_ret": total_ret, "ann_ret": ann_ret,
        "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd,
        "calmar": calmar, "win_rate": win_rate, "nav": nav,
    }


__all__ = ["build_signal_v5", "backtest_v5"]
