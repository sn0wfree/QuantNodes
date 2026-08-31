#!/usr/bin/env python3
"""scripts/cmf/research_ml_risk_diag.py — 风控ML深度诊断.

读取v1/v3的输出, 重新运行Phase 3回测, 收集详细诊断数据:
- D1: 各年MaxDD分解
- D2: VIX分层 (regime) 违约率
- D3: VolCap动态 (ML为什么偏高)
- D4: 触发延迟 (ML晚刹车?)
- D5: 盈亏归因 (Sharpe差距来源)
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
    pos = position.copy()
    daily_ret = close.pct_change()
    rolling_dd_252 = close / close.rolling(252).max() - 1
    realized_vol_20 = daily_ret.rolling(20, min_periods=5).std() * np.sqrt(252)
    vol_mask = (realized_vol_20 > params["L1_VolCap"]) & (realized_vol_20.notna())
    vol_scale = (params["L1_VolCap"] / realized_vol_20).clip(upper=1.0)
    pos = pos.where(~vol_mask, pos * vol_scale)
    soft_mask = rolling_dd_252 < params["L2_SoftCap"]
    pos = pos.where(~soft_mask, pos * 0.5)
    hard_mask = (rolling_dd_252 < params["L3_HardCap"]) | (daily_ret < params["DailyCap"])
    hard_mask = hard_mask.fillna(False)
    pos = pos.where(~hard_mask, 0.0)
    if params["CooldownDays"] > 0:
        for i in range(len(pos)):
            if hard_mask.iloc[i]:
                end = min(i + params["CooldownDays"] + 1, len(pos))
                pos.iloc[i:end] = 0.0
    return pos.clip(0, 1.2)


def compute_dd_series(nav):
    return nav / nav.cummax() - 1


def main():
    print("=" * 60)
    print("风控ML深度诊断 — 5维度")
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

    # === 生成ML参数 (与v3相同方法) ===
    print("\n生成ML参数 (walk-forward)...")
    target_params = ["L1_VolCap", "L2_SoftCap", "L3_HardCap", "DailyCap"]
    ml_params = pd.DataFrame(index=df.index, columns=target_params, dtype=float)
    min_train = 504

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

    # === 应用固定参数 vs ML参数 ===
    pos_fixed = apply_risk_overlay(close, prod_pos, FIXED_PARAMS)

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

    # === D1: 时间分布诊断 ===
    print("\n=== D1: 时间分布 (各年MaxDD) ===")
    print(f"{'年份':<10s}{'固定MaxDD':>12s}{'ML MaxDD':>12s}{'差异':>10s}{'哪方违反G3':>12s}")
    print("-" * 60)
    years = df.index.year.unique()
    for y in years:
        mask = df.index.year == y
        f_dd = compute_dd_series((1 + (pos_fixed[mask].shift(1) * close[mask].pct_change()).dropna()).cumprod()).min()
        m_dd = compute_dd_series((1 + (pos_ml[mask].shift(1) * close[mask].pct_change()).dropna()).cumprod()).min()
        delta = m_dd - f_dd
        f_violate = "❌" if f_dd < HARD_CONSTRAINT_MAX_DD else "✅"
        m_violate = "❌" if m_dd < HARD_CONSTRAINT_MAX_DD else "✅"
        print(f"{y:<10d}{f_dd*100:>11.2f}% {m_dd*100:>11.2f}% {delta*100:>+9.2f}% {f_violate} vs {m_violate}")

    # === D2: Regime分解 (VIX分层) ===
    print("\n=== D2: Regime分解 (VIX分层违约率) ===")
    vix_pct = df["vix"].rolling(252).rank(pct=True)
    for label, low, high in [("高VIX(>70%)", 0.7, 1.01), ("中VIX(30-70%)", 0.3, 0.7), ("低VIX(<30%)", -0.01, 0.3)]:
        mask = (vix_pct >= low) & (vix_pct < high) & df.index.notna()
        f_dd = compute_dd_series((1 + (pos_fixed[mask].shift(1) * close[mask].pct_change()).dropna()).cumprod()).min() if mask.sum() > 50 else np.nan
        m_dd = compute_dd_series((1 + (pos_ml[mask].shift(1) * close[mask].pct_change()).dropna()).cumprod()).min() if mask.sum() > 50 else np.nan
        f_violate = "❌" if f_dd < HARD_CONSTRAINT_MAX_DD else "✅"
        m_violate = "❌" if m_dd < HARD_CONSTRAINT_MAX_DD else "✅"
        delta = m_dd - f_dd
        print(f"{label:<15s} n={mask.sum():<6d} 固定={f_dd*100:>7.2f}% ML={m_dd*100:>7.2f}% Δ={delta*100:>+6.2f}% 固定{f_violate} ML{m_violate}")

    # === D3: VolCap动态诊断 ===
    print("\n=== D3: VolCap动态 (L1_VolCap vs 实际波动率) ===")
    print(f"固定L1_VolCap: {FIXED_PARAMS['L1_VolCap']}")
    print(f"ML L1_VolCap统计:")
    print(f"  mean: {ml_params['L1_VolCap'].mean():.3f}")
    print(f"  std:  {ml_params['L1_VolCap'].std():.3f}")
    print(f"  min:  {ml_params['L1_VolCap'].min():.3f}, max: {ml_params['L1_VolCap'].max():.3f}")

    # 实际5日波动率
    actual_vol_5d = close.pct_change().rolling(5).std() * np.sqrt(252)
    ml_aligned = ml_params["L1_VolCap"].reindex(actual_vol_5d.index).ffill()
    corr = ml_aligned.corr(actual_vol_5d)
    print(f"\nML L1_VolCap vs actual_vol_5d 相关系数: {corr:.3f}")
    print(f"固定L1_VolCap (常数0.15) vs actual_vol_5d: {actual_vol_5d.corr(pd.Series(0.15, index=actual_vol_5d.index)):.3f}")

    # 关键: ML预测的VolCap与实际波动的关系
    diff = ml_aligned - actual_vol_5d
    print(f"\nL1_VolCap - actual_vol_5d:")
    print(f"  mean: {diff.mean():.3f} (正值=ML预测cap高于实际波动)")
    print(f"  当actual_vol_5d > 0.15时, ML预测VolCap均: {ml_aligned[actual_vol_5d > 0.15].mean():.3f}")
    print(f"  当actual_vol_5d > 0.20时, ML预测VolCap均: {ml_aligned[actual_vol_5d > 0.20].mean():.3f}")

    # === D4: 触发延迟诊断 ===
    print("\n=== D4: 硬止损触发延迟 (ML vs 固定) ===")

    def get_hard_trigger_dates(pos, close, params):
        daily_ret = close.pct_change()
        rolling_dd_252 = close / close.rolling(252).max() - 1
        hard_mask = (rolling_dd_252 < params["L3_HardCap"]) | (daily_ret < params["DailyCap"])
        return set(hard_mask[hard_mask.fillna(False)].index)

    f_triggers = get_hard_trigger_dates(pos_fixed, close, FIXED_PARAMS)
    # ML触发 = 每天ML参数下的硬止损触发
    ml_triggers = set()
    for d in df.index:
        params_t = {
            "L3_HardCap": float(ml_params.loc[d, "L3_HardCap"]) if pd.notna(ml_params.loc[d, "L3_HardCap"]) else FIXED_PARAMS["L3_HardCap"],
            "DailyCap": float(ml_params.loc[d, "DailyCap"]) if pd.notna(ml_params.loc[d, "DailyCap"]) else FIXED_PARAMS["DailyCap"],
        }
        daily_ret = close[d - pd.Timedelta(days=1):d + pd.Timedelta(days=1)].pct_change()
        if len(daily_ret) < 2:
            continue
        today_ret = daily_ret.iloc[-1]
        rolling_dd = (close[d] / close[d - pd.Timedelta(days=252):d].max() - 1) if d > df.index[min_train] else 0
        if rolling_dd < params_t["L3_HardCap"] or today_ret < params_t["DailyCap"]:
            ml_triggers.add(d)

    print(f"固定参数触发: {len(f_triggers)}天")
    print(f"ML参数触发: {len(ml_triggers)}天")
    only_fixed = f_triggers - ml_triggers
    only_ml = ml_triggers - f_triggers
    both = f_triggers & ml_triggers
    print(f"仅固定触发 (ML漏掉): {len(only_fixed)}天")
    print(f"仅ML触发 (固定漏掉): {len(only_ml)}天")
    print(f"都触发: {len(both)}天")
    print(f"\nML漏掉的关键触发日 (前10):")
    for d in sorted(only_fixed)[:10]:
        idx_pos = df.index.get_loc(d)
        if idx_pos > 0:
            ret = close.iloc[idx_pos] / close.iloc[idx_pos-1] - 1
            print(f"  {d.date()} (当日收益={ret*100:+.2f}%)")

    # === D5: 盈亏归因 ===
    print("\n=== D5: Sharpe差距归因 (固定 vs ML) ===")

    def compute_metrics(pos, close):
        sr = (pos.shift(1) * close.pct_change()).dropna()
        cost = pos.diff().abs() * 5.0 / 10000
        sr = (sr - cost).dropna()
        nav = (1 + sr).cumprod()
        n_y = len(sr) / 252
        cagr = ((nav.iloc[-1]) ** (1/n_y) - 1) if n_y > 0 else 0
        vol = sr.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0
        dd = nav / nav.cummax() - 1
        return {"cagr": cagr, "vol": vol, "sharpe": sharpe, "max_dd": dd.min()}

    r_fixed = compute_metrics(pos_fixed, close)
    r_ml = compute_metrics(pos_ml, close)
    print(f"{'指标':<12s}{'固定':>12s}{'ML':>12s}{'Δ':>12s}{'归因':>15s}")
    print("-" * 60)
    print(f"{'CAGR':<12s}{r_fixed['cagr']*100:>11.2f}%{r_ml['cagr']*100:>11.2f}%{(r_ml['cagr']-r_fixed['cagr'])*100:>+11.2f}%")
    print(f"{'Vol':<12s}{r_fixed['vol']*100:>11.2f}%{r_ml['vol']*100:>11.2f}%{(r_ml['vol']-r_fixed['vol'])*100:>+11.2f}%")
    print(f"{'Sharpe':<12s}{r_fixed['sharpe']:>12.3f}{r_ml['sharpe']:>12.3f}{r_ml['sharpe']-r_fixed['sharpe']:>+12.3f}")
    print(f"{'MaxDD':<12s}{r_fixed['max_dd']*100:>11.2f}%{r_ml['max_dd']*100:>11.2f}%{(r_ml['max_dd']-r_fixed['max_dd'])*100:>+11.2f}%")

    # Sharpe归因: 收益差 + 波动率差
    sharpe_gap = r_ml['sharpe'] - r_fixed['sharpe']
    cagr_gap_pct = (r_ml['cagr'] - r_fixed['cagr']) * 100
    vol_gap_pct = (r_ml['vol'] - r_fixed['vol']) * 100
    print(f"\nSharpe差 {sharpe_gap:+.3f} 归因:")
    print(f"  CAGR差: {cagr_gap_pct:+.2f}pp (ML {'更好' if cagr_gap_pct > 0 else '更差'})")
    print(f"  Vol差:  {vol_gap_pct:+.2f}pp (ML {'更高' if vol_gap_pct > 0 else '更低'})")
    if sharpe_gap < 0 and vol_gap_pct > 0:
        print(f"  → 核心问题: ML波动率更高拖累Sharpe, 非收益问题")
    elif sharpe_gap < 0 and cagr_gap_pct < 0:
        print(f"  → 核心问题: ML收益更低")

    # === 修复建议 ===
    print("\n" + "=" * 60)
    print("修复建议 (基于5维度诊断)")
    print("=" * 60)
    # 简单的修复: 用更保守的VolCap cap
    fixed_cap_suggestion = 0.13  # 比0.15略低
    print(f"建议1: 固定VolCap降到0.13 (固定参数更保守)")
    print(f"建议2: 给ML加硬规则: 'L1_VolCap ≤ 实际vol_60d × 1.5'")
    print(f"建议3: 用regime切换: 高VIX用固定, 低VIX用ML")
    print(f"建议4: 完全放弃ML, 现有固定参数已是最优 (多年优化)")


if __name__ == "__main__":
    main()