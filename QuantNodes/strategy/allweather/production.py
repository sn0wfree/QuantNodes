#!/usr/bin/env python3
# coding=utf-8
"""全天候 v3 — 生产运行层 (§生产化, 2026-09-24).

配置 (见 docs/allweather_production_candidate.md v3):
  REF41 + ME 月频 + D.1 aggressive + D.2 动量β0.2 + P1-B(θ0.60,k1.0)
  + 共振门控(60/0.70/0.40/0.50, dd_floor -0.05, k0.9)
  + Carry 30% (美元中性多空, 平滑 20d)
  IS 2019-2026: CAGR 10.10% / Sharpe 2.02 / MaxDD -5.20%; WF OOS ~1.7.

数据依赖 (须保持新鲜):
  - data/allweather/panel_v5.parquet          (价格面板, REF41)
  - data/allweather/carry_daily.parquet       (carry, scripts/extract_carry.py)
  - data/high_freq_macro/v7_index_v2.parquet  (cash_proxy)
  - data/allweather/volume_oi_cache.parquet   (量价, scripts/DataCache → DuckDB)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .carry_sleeve import build_carry_sleeve, combine_with_d
from .class_erc import ASSET_CLASSES
from .d_pipeline import run_d_sleeve
from .volume_loader import load_volume_oi_panel

REPO = Path(__file__).resolve().parents[3]
DATA_DIR = REPO / "data/allweather"
V7_PATH = REPO / "data/high_freq_macro/v7_index_v2.parquet"
CASH_COL = "中债1-3年国债财富指数"

# 非 REF41 的增量工具 (与回测口径一致, 排除)
EXCLUDE = ("etf_510500", "etf_588000", "etf_159920", "etf_513520",
           "etf_511260", "etf_159985", "csi_1000", "etf_512100")

# D 管线参数 (§生产候选 v3)
D_PARAMS = dict(
    bond_cap=0.40, window=252, rebalance_freq="ME", cov_estimator="lw",
    neg_trend_mode="trend", neg_trend_preset="aggressive", include_rapid=True,
    momentum_beta=0.20, target_vol=0.15, max_leverage=2.5, financing_rate=0.02,
    correlation_k=1.0, correlation_threshold=0.60,
    resonance_k=0.9, resonance_ar3_window=60, resonance_enter=0.70,
    resonance_exit=0.40, resonance_vol_exit=0.50, resonance_dd_floor=-0.05,
)
CARRY_WEIGHT = 0.30
D_WEIGHT = 1.0 - CARRY_WEIGHT


@dataclass
class ProductionResult:
    nav: pd.Series              # 组合净值
    weights: pd.DataFrame       # 组合目标权重 (D_WEIGHT·w_D + CARRY_WEIGHT·w_carry)
    d_nav: pd.Series            # D sleeve 净值
    carry_ret: pd.Series        # carry sleeve 净收益
    d_weights: pd.DataFrame
    carry_weights: pd.DataFrame


def load_prod_data(as_of: str | None = None, cache_path=None):
    """加载生产数据 (可选 as_of 截断)."""
    panel = pd.read_parquet(DATA_DIR / "panel_v5.parquet").sort_index()
    panel.index = pd.to_datetime(panel.index)
    carry = pd.read_parquet(DATA_DIR / "carry_daily.parquet")
    carry.index = pd.to_datetime(carry.index)
    v7 = pd.read_parquet(V7_PATH)
    v7.index = pd.to_datetime(v7.index)
    cash_ret = v7[CASH_COL].dropna().pct_change(fill_method=None)
    P = panel[[c for c in panel.columns if c not in EXCLUDE]]
    if as_of is not None:
        P = P.loc[:as_of]
        carry = carry.loc[:as_of]
        cash_ret = cash_ret.loc[:as_of]
    vol, oi = load_volume_oi_panel(
        P, cache_path=cache_path or (DATA_DIR / "volume_oi_cache.parquet"))
    # 量价缓存可能落后最新价 → 前向填充 (量价慢变; 见 runbook 数据新鲜度)
    vol, oi = vol.ffill(), oi.ffill()
    return P, cash_ret, carry, vol, oi


def run_production(as_of: str | None = None, cache_path=None) -> ProductionResult:
    """运行生产配置, 返回净值与目标权重."""
    P, cash_ret, carry, vol, oi = load_prod_data(as_of, cache_path)
    res = run_d_sleeve(P, volume=vol, oi=oi, asset_classes=dict(ASSET_CLASSES),
                       cash_ret=cash_ret, **D_PARAMS)
    carry_ret, carry_w = build_carry_sleeve(P, carry, freq="ME")
    d_nav = res.nav.dropna()
    nav = combine_with_d(d_nav, carry_ret, carry_weight=CARRY_WEIGHT,
                         leverage=1.0)
    cols = res.weights.columns
    cw = carry_w.reindex(index=res.weights.index, columns=cols).fillna(0.0)
    weights = D_WEIGHT * res.weights + CARRY_WEIGHT * cw
    return ProductionResult(nav=nav, weights=weights, d_nav=d_nav,
                            carry_ret=carry_ret, d_weights=res.weights,
                            carry_weights=carry_w)


def latest_target(result: ProductionResult | None = None) -> pd.Series:
    """最新一日非零目标权重 (按 |w| 降序)."""
    res = result if result is not None else run_production()
    w = res.weights.iloc[-1]
    return w[w.abs() > 1e-6].sort_values(key=lambda s: s.abs(),
                                         ascending=False)


def summarize(nav: pd.Series) -> dict:
    """净值 → 关键指标 (含当前回撤)."""
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(252)
    return {
        "start": nav.index[0].date(), "end": nav.index[-1].date(),
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
        "Sharpe": r.mean() * 252 / vol if vol > 0 else np.nan,
        "Vol": vol,
        "MaxDD": (nav / nav.cummax() - 1).min(),
        "CurrDD": (nav / nav.cummax() - 1).iloc[-1],
    }
