#!/usr/bin/env python3
"""research_factor_construction.py — 因子构造优化研究.

测试5个因子的构造变体, 评估IC/稳健性/Sharpe, 与当前构造对比.
因子: real_rate / risk / momentum / position_flow / macro_regime(过拟合检查)

输出: 终端报告 + reports/gold/factor_construction.csv
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts/cmf")

import argparse

import numpy as np
import pandas as pd

from QuantNodes.strategy.commodity_fundamental.gold import (
    load_au_adj_ohlcv,
    load_vix_gvz,
    load_indicator_d,
    o2o_returns,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
    compute_breakout_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_position_flow import (
    compute_position_flow_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_risk import (
    compute_risk_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
)

OUTPUT_DIR = "reports/gold"

PERIODS = [("2018-2020", "2018-01-01", "2020-12-31"),
           ("2021-2023", "2021-01-01", "2023-12-31"),
           ("2024-2026", "2024-01-01", "2026-12-31")]


# =====================================================================
# 数据加载
# =====================================================================

def load_weekly_data():
    """加载周频数据: gold close, vix, gvz, tips, cb, etf."""
    close_d, open_d = load_au_adj_ohlcv()
    vix_d, gvz_d = load_vix_gvz(close_d.index)
    tips_d = load_indicator_d("tips10y", "fred_dfii10").reindex(close_d.index).ffill().shift(2)
    cb_d = load_indicator_d("cb_reserves", source="gold_position").reindex(close_d.index).ffill()
    etf_d = load_indicator_d("etf_tonnes", source="gold_position").reindex(close_d.index).ffill()
    gprd_d = load_indicator_d("gprd", source="gold_risk").reindex(close_d.index).ffill()

    widx = close_d.resample("W-FRI").last().dropna().index
    weekly = {}
    weekly["close"] = close_d.resample("W-FRI").last().reindex(widx)
    weekly["open"] = open_d.resample("W-FRI").last().reindex(widx)
    weekly["vix"] = vix_d.resample("W-FRI").last().reindex(widx)
    weekly["gvz"] = gvz_d.resample("W-FRI").last().reindex(widx)
    weekly["tips"] = tips_d.resample("W-FRI").last().reindex(widx)
    weekly["cb"] = cb_d.resample("W-FRI").last().reindex(widx)
    weekly["etf"] = etf_d.resample("W-FRI").last().reindex(widx)
    weekly["gprd"] = gprd_d.resample("W-FRI").last().reindex(widx)
    weekly["fwd"] = weekly["close"].pct_change().shift(-1)
    return weekly


# =====================================================================
# 因子构造变体
# =====================================================================

def _real_rate_variants(w):
    """real_rate: 4变体."""
    tips = w["tips"]
    ma26 = tips.rolling(26).mean()
    ma13 = tips.rolling(13).mean()
    ma52 = tips.rolling(52).mean()
    std52 = tips.rolling(52).std()

    current = (tips < ma26).astype(float)
    slope = ((ma26 < ma26.shift(1)) & (tips < ma26)).astype(float)
    zscore = ((tips - ma26) / (std52 + 1e-8)).clip(-3, 3)
    zscore_pos = ((-zscore) / 6 + 0.5).clip(0, 1)  # 反转: TIPS低于均值→做多
    dual = (ma13 < ma52).astype(float)

    return {"当前(TIPS<MA26)": current, "A:MA斜率": slope,
            "B:Z-score连续": zscore_pos, "C:双窗口MA": dual}


def _risk_variants(w):
    """risk: 3变体."""
    vix = w["vix"]
    gprd = w["gprd"]
    vix_pct52 = vix.rolling(52).rank(pct=True)
    gprd_pct52 = gprd.rolling(52).rank(pct=True)

    current = ((vix_pct52 > 0.7) | (gprd_pct52 > 0.7)).astype(float)
    vix_only = (vix_pct52 > 0.7).astype(float)
    continuous = ((vix_pct52.fillna(0) * 0.6 + gprd_pct52.fillna(0) * 0.4) > 0.65).astype(float)
    zscore_vix = ((-vix + vix.rolling(52).mean()) / (vix.rolling(52).std() + 1e-8))
    zscore_pos = ((zscore_vix) / 6 + 0.5).clip(0, 1)

    return {"当前(VIX/GPRD OR)": current, "A:仅VIX": vix_only,
            "B:连续加权": continuous, "C:VIX Z-score": zscore_pos}


def _momentum_variants(w):
    """momentum: 3变体."""
    close = w["close"]
    mom52 = close.pct_change(52)
    mom26 = close.pct_change(26)
    vol52 = close.pct_change().rolling(52).std() * np.sqrt(52)

    current = (mom52 > 0).astype(float)
    sharpe_mom = (mom52 / (vol52 + 1e-8)).clip(-3, 3)
    sharpe_pos = ((sharpe_mom) / 6 + 0.5).clip(0, 1)
    dual = ((mom52 > 0) | (mom26 > 0)).astype(float)

    return {"当前(52w>0)": current, "A:Sharpe动量": sharpe_pos, "B:双窗口": dual}


def _position_flow_variants(w):
    """position_flow: 3变体."""
    cb = w["cb"]
    etf = w["etf"]
    cb_chg26 = cb.pct_change(26)
    etf_chg4 = etf.pct_change(4)

    current = ((cb_chg26 > 0) | (etf_chg4 > 0)).astype(float)
    cb_only = (cb_chg26 > 0).astype(float)
    etf_only = (etf_chg4 > 0).astype(float)
    weighted = ((cb_chg26.fillna(0) * 0.5 + etf_chg4.fillna(0) * 0.5).clip(-1, 1) + 1) / 2

    return {"当前(cb OR etf)": current, "A:仅cb": cb_only, "B:仅etf": etf_only,
            "C:加权连续": weighted}


def _macro_regime_variants(w, vix_raw, gvz_raw, tips_raw):
    """macro_regime: 参数扰动 (±20%)."""
    base = compute_macro_regime_signal(
        vix=vix_raw, gvz=gvz_raw, tips=tips_raw)["position"]
    base_w = base.resample("W-FRI").last().reindex(w["close"].index).ffill()

    variants = {"当前": base_w}

    # ts_out扰动 (趋势灵敏度)
    for delta, label in [(-0.2, "-20%"), (+0.2, "+20%")]:
        ts = 0.06 * (1 + delta)
        try:
            v = compute_macro_regime_signal(ts_out=ts, vix=vix_raw, gvz=gvz_raw, tips=tips_raw)["position"]
            variants[f"ts_out{label}"] = v.resample("W-FRI").last().reindex(w["close"].index).ffill()
        except Exception:
            pass

    # breakout_weight扰动
    for delta, label in [(-0.2, "-20%"), (+0.2, "+20%")]:
        bw = 0.3 * (1 + delta)
        try:
            v = compute_macro_regime_signal(breakout_weight=bw, vix=vix_raw, gvz=gvz_raw, tips=tips_raw)["position"]
            variants[f"bo_w{label}"] = v.resample("W-FRI").last().reindex(w["close"].index).ffill()
        except Exception:
            pass

    # gvz_power扰动
    for delta, label in [(-0.2, "-20%"), (+0.2, "+20%")]:
        gp = 0.8 * (1 + delta)
        try:
            v = compute_macro_regime_signal(gvz_power=gp, vix=vix_raw, gvz=gvz_raw, tips=tips_raw)["position"]
            variants[f"gvz_p{label}"] = v.resample("W-FRI").last().reindex(w["close"].index).ffill()
        except Exception:
            pass

    return variants


# =====================================================================
# 评估
# =====================================================================

def weekly_ic(pos, fwd):
    """Spearman IC."""
    common = pos.dropna().index.intersection(fwd.dropna().index)
    return pos.reindex(common).corr(fwd.reindex(common), method="spearman")


def sub_ics(pos, fwd):
    """3子周期IC."""
    results = []
    for label, s, e in PERIODS:
        mask = (pos.index >= s) & (pos.index <= e)
        sub = pos[mask]
        sub_fwd = fwd[mask]
        common = sub.dropna().index.intersection(sub_fwd.dropna().index)
        if len(common) < 10:
            results.append(np.nan)
        else:
            results.append(sub.reindex(common).corr(sub_fwd.reindex(common), method="spearman"))
    return results


def standalone_o2o(pos_weekly, open_weekly):
    """standalone O2O Sharpe (简化版)."""
    pos_d = pos_weekly.reindex(close_d.index).ffill().fillna(0)
    r = o2o_returns(pos_d, open_d)
    if len(r) < 2:
        return np.nan, np.nan, np.nan
    nav = (1 + r).cumprod()
    n = len(r)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    dd = (nav / nav.cummax() - 1).min()
    return cagr, sharpe, dd


# =====================================================================
# 主流程
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="因子构造优化研究")
    parser.parse_args()

    print("=" * 70)
    print("因子构造优化研究 — 5因子×多变体, IC/稳健性/Sharpe")
    print("=" * 70)

    # 加载数据
    print("\n[1] 加载周频数据...")
    w = load_weekly_data()
    global close_d, open_d
    close_d, open_d = load_au_adj_ohlcv()
    fwd = w["fwd"]

    # 需要日频原始数据给macro_regime
    vix_raw = load_indicator_d("vix", "gold_fred")
    gvz_raw = load_indicator_d("gvz", "gold_fred")
    tips_raw = load_indicator_d("tips10y", "fred_dfii10")

    # 构造变体
    print("[2] 构造因子变体...")
    all_variants = {}
    all_variants.update({f"rr_{k}": v for k, v in _real_rate_variants(w).items()})
    all_variants.update({f"risk_{k}": v for k, v in _risk_variants(w).items()})
    all_variants.update({f"mom_{k}": v for k, v in _momentum_variants(w).items()})
    all_variants.update({f"pf_{k}": v for k, v in _position_flow_variants(w).items()})
    all_variants.update({f"macro_{k}": v for k, v in _macro_regime_variants(w, vix_raw, gvz_raw, tips_raw).items()})

    print(f"  共 {len(all_variants)} 个变体")

    # 评估
    print("[3] 评估IC/稳健性/Sharpe...")
    rows = []
    for name, pos in all_variants.items():
        ic = weekly_ic(pos, fwd)
        s1, s2, s3 = sub_ics(pos, fwd)
        cagr, sharpe, mdd = standalone_o2o(pos, w["open"])
        factor = name.split("_")[0]
        variant = "_".join(name.split("_")[1:]) if "_" in name else name
        rows.append({
            "factor": factor, "variant": variant, "ic": ic,
            "s1": s1, "s2": s2, "s3": s3,
            "cagr": cagr, "sharpe": sharpe, "mdd": mdd,
        })

    df = pd.DataFrame(rows)

    # 按因子分组输出
    print("\n" + "=" * 70)
    print("结果 (IC/子周期稳健性/Sharpe)")
    print("=" * 70)

    for factor in ["rr", "risk", "mom", "pf", "macro"]:
        sub = df[df["factor"] == factor].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("ic", ascending=False)
        print(f"\n--- {factor} ---")
        print(f"  {'变体':<20} {'IC':>8} {'s1':>7} {'s2':>7} {'s3':>7} {'Sharpe':>8} {'MaxDD':>9}")
        for _, r in sub.iterrows():
            s_flag = "✅" if (pd.notna(r.s1) and pd.notna(r.s2) and pd.notna(r.s3) and r.s1 > 0 and r.s2 > 0 and r.s3 > 0) else "⚠️"
            print(f"  {r['variant']:<20} {r.ic:>+8.4f} {r.s1:>+7.3f} {r.s2:>+7.3f} {r.s3:>+7.3f} {r.sharpe:>+8.2f} {r.mdd:>+9.2%} {s_flag}")

    # 保存CSV
    df.to_csv(f"{OUTPUT_DIR}/factor_construction.csv", index=False, float_format="%.6f")
    print(f"\n✓ 已输出: {OUTPUT_DIR}/factor_construction.csv")

    # 总结
    best_per_factor = df.loc[df.groupby("factor")["ic"].idxmax()]
    print("\n" + "=" * 70)
    print("各因子最优变体 (按IC)")
    print("=" * 70)
    for _, r in best_per_factor.iterrows():
        robust = "✅稳健" if (pd.notna(r.s1) and pd.notna(r.s2) and pd.notna(r.s3) and r.s1 > 0 and r.s2 > 0 and r.s3 > 0) else "⚠️不稳健"
        print(f"  {r['factor']:<8} {r['variant']:<20} IC={r.ic:+.4f} Sharpe={r.sharpe:+.2f} MaxDD={r.mdd:+.2%} {robust}")


if __name__ == "__main__":
    main()
