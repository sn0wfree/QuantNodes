#!/usr/bin/env python3
"""scripts/cmf/research_ml_risk_v3.py — 风控ML Phase 3 (简化版).

纯只读原型, 不改生产代码.
关键对比: 固定参数 vs ML参数 在回撤期间的行为差异.
重点检验: G3硬约束 (MaxDD<15%) 是否被违反.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DB_PATH = "data/cmf/fundamental.duckdb"
FIXED_PARAMS = {
    "L1_VolCap": 0.15, "L2_SoftCap": -0.07, "L3_HardCap": -0.10,
    "DailyCap": -0.02, "CooldownDays": 5,
}
PARAM_RANGES = {
    "L1_VolCap": (0.10, 0.25), "L2_SoftCap": (-0.12, -0.05),
    "L3_HardCap": (-0.15, -0.08), "DailyCap": (-0.03, -0.015),
}
HARD_CONSTRAINT_MAX_DD = -0.15


def load_data():
    con = duckdb.connect(DB_PATH, read_only=True)
    qs = {
        "gold": ("gold_macro", "gold_spot"),
        "vix": ("gold_fred", "vix"),
        "gvz": ("gold_fred", "gvz"),
        "dxy": ("gold_macro", "dxy"),
        "wti": ("fred_dcoil", "wti"),
    }
    dfs = {}
    for c, (s, v) in qs.items():
        df = con.execute(
            f"SELECT trade_date, value FROM fundamentals_daily "
            f"WHERE source='{s}' AND value_col='{v}' ORDER BY trade_date"
        ).df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        dfs[c] = df.set_index("trade_date")["value"]
    con.close()
    return pd.DataFrame(dfs).sort_index().ffill().dropna()


def _z(s, w=156):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_features(df):
    out = df.copy()
    out["F10_dxy_z"] = -_z(out["dxy"], 156)
    out["vix_pct"] = out["vix"].rolling(252).rank(pct=True)
    out["gvz_z"] = _z(out["gvz"], 90)
    out["realized_vol_20d"] = out["gold"].pct_change().rolling(20).std() * np.sqrt(252)
    out["realized_vol_60d"] = out["gold"].pct_change().rolling(60).std() * np.sqrt(252)
    out["vol_change"] = out["realized_vol_20d"] / out["realized_vol_60d"] - 1
    out["momentum_20d"] = out["gold"].pct_change(20)
    out["momentum_60d"] = out["gold"].pct_change(60)
    out["vix_change_5d"] = out["vix"].pct_change(5)
    out["dxy_change_20d"] = out["dxy"].pct_change(20)
    out["gold_oil_zscore"] = _z(out["gold"] / out["wti"], 156)
    return out


def build_targets(df):
    out = df[["gold"]].copy()
    out["gold_ret"] = out["gold"].pct_change()
    future_vol_20 = out["gold_ret"].shift(-20).rolling(20).std() * np.sqrt(252)
    out["L1_VolCap_target"] = (future_vol_20 * 2.0).clip(0.10, 0.25).shift(1)
    rolling_dd_60 = out["gold"] / out["gold"].rolling(60).max() - 1
    min_dd_60 = rolling_dd_60.rolling(60).min()
    out["L2_SoftCap_target"] = (min_dd_60 * 0.8).clip(-0.12, -0.05).shift(1)
    out["L3_HardCap_target"] = (min_dd_60 * 1.2).clip(-0.15, -0.08).shift(1)
    max_daily_loss = out["gold"].pct_change().rolling(5).min()
    out["DailyCap_target"] = (max_daily_loss * 0.8).clip(-0.03, -0.015).shift(1)
    return out


def apply_risk_overlay(close, position, params):
    """单次risk_overlay应用 (向量化, 快速)."""
    pos = position.copy()
    daily_ret = close.pct_change()
    rolling_dd_252 = close / close.rolling(252).max() - 1
    realized_vol_20 = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)

    # L1 波动率压制
    vol_mask = (realized_vol_20 > params["L1_VolCap"]) & (realized_vol_20.notna())
    vol_scale = (params["L1_VolCap"] / realized_vol_20).clip(upper=1.0)
    pos = pos.where(~vol_mask, pos * vol_scale)

    # L2 软减仓
    soft_mask = rolling_dd_252 < params["L2_SoftCap"]
    pos = pos.where(~soft_mask, pos * 0.5)

    # L3 硬止损
    hard_mask = (rolling_dd_252 < params["L3_HardCap"]) | (daily_ret < params["DailyCap"])
    hard_mask = hard_mask.fillna(False)
    pos = pos.where(~hard_mask, 0.0)

    # 简化cooldown: 硬止损后强制5天空仓
    if params["CooldownDays"] > 0:
        cooldown = hard_mask.shift(params["CooldownDays"]).fillna(False)
        # 触发日+之后5天都强制空仓
        for i in range(len(pos)):
            if hard_mask.iloc[i]:
                end = min(i + params["CooldownDays"] + 1, len(pos))
                pos.iloc[i:end] = 0.0

    return pos.clip(0, 1.2)


def o2o_backtest(close, position, cost_bps=5.0):
    ret_fwd = close.pct_change().shift(-1)
    sr = (position.shift(1) * ret_fwd).dropna()
    cost = position.diff().abs() * cost_bps / 10000
    sr = (sr - cost).dropna()
    nav = (1 + sr).cumprod()
    n_y = len(sr) / 252
    cagr = ((nav.iloc[-1]) ** (1 / n_y) - 1) if n_y > 0 else 0
    vol = sr.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    dd = nav / nav.cummax() - 1
    return {
        "cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd.min(),
        "n_years": n_y, "n_days": len(sr),
    }


def main():
    print("=" * 60)
    print("风控ML Phase 3 — 真实回测对比 (简化版)")
    print("=" * 60)

    df = build_features(load_data())["2018":"2025"]
    df_target = build_targets(load_data())["2018":"2025"]
    df = df.join(df_target.drop(columns=["gold", "gold_ret"]), how="inner")
    print(f"区间: {df.index[0].date()} ~ {df.index[-1].date()}, {len(df)}天")

    features = [
        "F10_dxy_z", "vix_pct", "gvz_z",
        "realized_vol_20d", "realized_vol_60d", "vol_change",
        "momentum_20d", "momentum_60d",
        "vix_change_5d", "dxy_change_20d",
        "gold_oil_zscore",
    ]
    X = df[features].dropna()
    close = df["gold"]

    # === 加载生产信号 ===
    from QuantNodes.strategy.commodity_fundamental.gold import compute_layered_ensemble_signal
    sig_prod = compute_layered_ensemble_signal()
    prod_pos = sig_prod["position"].reindex(df.index).ffill().fillna(0)

    # === ML参数生成 (walk-forward, 一次性) ===
    print("\n生成ML参数 (walk-forward CV)...")
    target_params = ["L1_VolCap", "L2_SoftCap", "L3_HardCap", "DailyCap"]
    ml_params = pd.DataFrame(index=df.index, columns=target_params, dtype=float)
    min_train = 504  # 2年

    for t_idx in range(min_train, len(df)):
        test_date = df.index[t_idx]
        train_idx = df.index[:t_idx]
        X_train = df.loc[train_idx, features].dropna()
        y_train = df.loc[X_train.index, [f"{p}_target" for p in target_params]].dropna()
        common = X_train.index.intersection(y_train.index)
        X_train, y_train = X_train.loc[common], y_train.loc[common]
        if len(X_train) < 200:
            continue
        for param in target_params:
            y_p = y_train[f"{param}_target"]
            if y_p.std() < 1e-6:
                ml_params.loc[test_date, param] = FIXED_PARAMS[param]
                continue
            try:
                mdl = RandomForestRegressor(n_estimators=30, max_depth=3, random_state=42, n_jobs=1)
                mdl.fit(X_train, y_p)
                pred = mdl.predict(X.loc[[test_date]])[0]
                lo, hi = PARAM_RANGES[param]
                pred = max(lo, min(hi, pred))
                ml_params.loc[test_date, param] = pred
            except Exception:
                ml_params.loc[test_date, param] = FIXED_PARAMS[param]
    ml_params = ml_params.ffill().bfill()

    print(f"ML参数生成完成 ({ml_params.notna().sum().sum()} / {ml_params.size})")
    print(f"ML参数统计:\n{ml_params.describe().round(3)}")

    # === 一次性应用风控 (避免双层循环) ===
    # A. 固定参数
    print("\n--- A. 固定参数 (生产baseline) ---")
    pos_fixed = apply_risk_overlay(close, prod_pos, FIXED_PARAMS)
    res_fixed = o2o_backtest(close, pos_fixed)
    print(f"  CAGR:   {res_fixed['cagr']*100:+.2f}%")
    print(f"  Sharpe: {res_fixed['sharpe']:+.3f}")
    print(f"  MaxDD:  {res_fixed['max_dd']*100:+.2f}%")
    g3_fixed = res_fixed['max_dd'] > HARD_CONSTRAINT_MAX_DD
    print(f"  G3: {'✅' if g3_fixed else '❌'} (MaxDD<{HARD_CONSTRAINT_MAX_DD*100:.0f}%)")

    # B. ML参数 (用每5天更新的ML参数, 简化cooldown)
    print("\n--- B. ML参数 (原型, 5天更新频率) ---")
    pos_ml = prod_pos.copy()
    step = 5
    for start in range(min_train, len(df), step):
        end = min(start + step, len(df))
        if ml_params.iloc[start].isna().any():
            continue
        win_params = {
            "L1_VolCap": float(ml_params.iloc[start]["L1_VolCap"]),
            "L2_SoftCap": float(ml_params.iloc[start]["L2_SoftCap"]),
            "L3_HardCap": float(ml_params.iloc[start]["L3_HardCap"]),
            "DailyCap": float(ml_params.iloc[start]["DailyCap"]),
            "CooldownDays": 5,
        }
        sub_close = close.iloc[start:end]
        sub_pos = prod_pos.iloc[start:end]
        if len(sub_close) > 0:
            sub_pos_adj = apply_risk_overlay(sub_close, sub_pos, win_params)
            pos_ml.iloc[start:end] = sub_pos_adj

    res_ml = o2o_backtest(close, pos_ml)
    print(f"  CAGR:   {res_ml['cagr']*100:+.2f}%")
    print(f"  Sharpe: {res_ml['sharpe']:+.3f}")
    print(f"  MaxDD:  {res_ml['max_dd']*100:+.2f}%")
    g3_ml = res_ml['max_dd'] > HARD_CONSTRAINT_MAX_DD
    print(f"  G3: {'✅' if g3_ml else '❌'} (MaxDD<{HARD_CONSTRAINT_MAX_DD*100:.0f}%)")

    # === 判定 ===
    print("\n" + "=" * 60)
    print("Phase 3 判定")
    print("=" * 60)
    print(f"{'指标':<12s}{'固定参数':>15s}{'ML参数':>15s}{'Δ':>12s}")
    print("-" * 60)
    for k, lbl in [("cagr", "CAGR"), ("sharpe", "Sharpe"), ("max_dd", "MaxDD")]:
        v_f = res_fixed[k]; v_m = res_ml[k]
        if k == "max_dd":
            print(f"{lbl:<12s}{v_f*100:>14.2f}%{v_m*100:>14.2f}%{(v_m-v_f)*100:>+11.2f}%")
        else:
            print(f"{lbl:<12s}{v_f:>15.3f}{v_m:>15.3f}{v_m-v_f:>+12.3f}")

    g2 = res_ml['sharpe'] >= res_fixed['sharpe'] * 0.95
    print(f"\nG2 (ML Sharpe ≥ 固定×95%): {'✅' if g2 else '❌'} ({res_ml['sharpe']:+.3f} vs {res_fixed['sharpe']:+.3f})")
    print(f"G3 (MaxDD<{HARD_CONSTRAINT_MAX_DD*100:.0f}%):")
    print(f"   固定: {'✅' if g3_fixed else '❌'}")
    print(f"   ML:   {'✅' if g3_ml else '❌'}")

    n_pass = sum([g2, g3_ml])
    if n_pass == 2:
        print(f"\n✅ ML参数满足约束且不逊于固定 → Phase 3成功")
    else:
        print(f"\n❌ ML参数未达预期 (G2={g2}, G3={g3_ml})")
        if g3_fixed and not g3_ml:
            print("   注意: 固定参数满足约束, ML参数违反约束 — ML动态参数过于激进")


if __name__ == "__main__":
    main()