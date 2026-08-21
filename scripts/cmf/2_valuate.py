#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/2_valuate.py — 价格中枢 + Lasso 估值缺口 (单品种)."""
from __future__ import annotations

import warnings
from pathlib import Path
import sys

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.commodity_fundamental import (
    weekly_kalman_center, walk_forward_lasso, rolling_phy_center, compute_valuation_gap,
)
from QuantNodes.strategy.commodity_fundamental.sign_constrained_lasso import rolling_zscore

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def load_panel(symbol: str, con) -> pd.Series:
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc", "M0": "futures_main_close_m"}[symbol]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].rename(symbol)


def load_lme_copper(con) -> pd.Series:
    df = con.execute("""
        SELECT trade_date, value FROM inventory_daily
        WHERE value_col = '铜-库存' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("copper_inv")


def load_eia_chg(con) -> pd.Series:
    df = con.execute("""
        SELECT trade_date, value FROM inventory_daily
        WHERE value_col = 'eia_chg_kbd' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("eia_chg")


def load_rig(con) -> pd.Series:
    df = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col = 'us_oil_rig' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("rig")


def load_hog(con) -> pd.Series:
    df = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col = 'hog_index' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("hog")


def load_pmi(con) -> pd.Series:
    df = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE value_col = 'pmi_mfg' ORDER BY trade_date
    """).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"].rename("pmi")


def build_features(symbol: str, con) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """返回 (mu_market, X 因子矩阵, y 收益)."""
    close = load_panel(symbol, con).dropna()
    mu_market = weekly_kalman_center(close, q=0.015).dropna()

    feats: dict[str, pd.Series] = {}

    if symbol == "CU0":
        copper = load_lme_copper(con)
        feats["inv_neg"] = -rolling_zscore(copper.resample("W-FRI").last(), 104).reindex(mu_market.index)
    elif symbol == "SC0":
        eia = load_eia_chg(con)
        feats["inv_neg"] = -rolling_zscore(eia.resample("W-FRI").last().cumsum().ffill(), 104).reindex(mu_market.index)
    elif symbol == "M0":
        hog = load_hog(con).resample("W-FRI").last()
        feats["demand"] = rolling_zscore(hog, 104).reindex(mu_market.index)
        feats["demand_4wchg"] = rolling_zscore(hog, 104).diff(4).reindex(mu_market.index)

    rig = load_rig(con)
    feats["rig_neg"] = -rolling_zscore(rig.resample("W-FRI").last().ffill(), 104).reindex(mu_market.index)

    pmi = load_pmi(con)
    feats["pmi"] = rolling_zscore(pmi.resample("W-FRI").last().ffill(), 104).reindex(mu_market.index)

    X = pd.concat(feats, axis=1)
    X = X.dropna(how="any")
    common_idx = X.index.intersection(mu_market.index)
    X = X.loc[common_idx]
    y = pd.Series(np.log(mu_market).diff().reindex(common_idx), name="dln_mu")

    y = y.dropna()
    common_idx = y.index.intersection(X.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]
    return mu_market.loc[common_idx], X, y


def main():
    print("=" * 70)
    print("CMF Phase 2 — KF中枢 + 符号约束Lasso 估值")
    print("=" * 70)

    con = duckdb.connect(str(DB), read_only=True)

    for symbol in ["CU0", "SC0", "M0"]:
        print(f"\n──── {symbol} ────")
        mu_market, X, y = build_features(symbol, con)
        print(f"  KF 周频中枢: {len(mu_market)} 周")
        print(f"  Lasso 因子: {list(X.columns)}, 样本: {len(X)}")
        print(f"  y 收益序列: {y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")

        coefs, oos = walk_forward_lasso(X, y, refit_weeks=26, min_train_weeks=156)
        print(f"  重估 {len(coefs)} 次, OOS 预测 {len(oos)} 周")
        print(f"  最近一次重估系数:")
        last = coefs.iloc[-1].drop(["refit_idx", "intercept"])
        print(f"  {last.to_dict()}")

        p_phy = rolling_phy_center(mu_market, oos["oos_pred"] if len(oos) else pd.Series(dtype=float), window=13)
        gap = compute_valuation_gap(mu_market, p_phy).dropna()
        print(f"  估值缺口: {len(gap)} 周, mean={gap.mean():.3f}, std={gap.std():.3f}")
        print(f"  当前 gap = {gap.iloc[-1]:+.3f} (正=低估, 负=高估)")

        coefs.assign(**{"intercept": coefs["intercept"]}).to_csv(OUT / f"coefs_{symbol}.csv")

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(mu_market.index, mu_market.values, label="M (卡尔曼中枢)", lw=1.5)
        ax.plot(p_phy.index, p_phy.values, label="p^phy (基本面中枢)", lw=1.5, ls="--")
        ax.set_title(f"{symbol} 价格中枢 vs 基本面中枢")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"valuate_{symbol}.png", dpi=110)
        plt.close()

        fig, ax = plt.subplots(figsize=(11, 3))
        ax.plot(gap.index, gap.values, label="gap = ln(p^phy/M)", color="C2")
        ax.axhline(0, color="k", lw=0.5)
        ax.fill_between(gap.index, gap.values, 0, where=gap.values > 0, color="green", alpha=0.2)
        ax.fill_between(gap.index, gap.values, 0, where=gap.values < 0, color="red", alpha=0.2)
        ax.set_title(f"{symbol} 估值缺口 (>0低估/<0高估)")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / f"gap_{symbol}.png", dpi=110)
        plt.close()

    con.close()


if __name__ == "__main__":
    main()