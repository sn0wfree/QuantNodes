#!/usr/bin/env python3
"""research_factor_tvpr.py — TV-PR动态权重因子筛选 (PIT, 周频).

方法: Total-Variation Predictive Regression (Cui et al. 2025)
  滚动(expanding窗口, 月度重拟合, warm-start)估计时变因子β
  → 窗口末端β经relu归一化为组合权重 → 套生产风控栈 → 与等权基线对照

因子池 (7候选):
  生产4路: macro_regime / real_rate(tips_ma) / momentum / breakout
  多因子独有: risk / position_flow / usd

实验 (全部套生产风控栈: 一层→×1.5→二层→死带→核心裁剪→网格→总封顶→平滑→应急):
  E-基线  等权4路 (必须锚点互锁 18.47%/1.56/-16.84%)
  E-等权7 等权7路
  E-TVPR  TV-PR动态权重7路

Walk-forward门槛:
  G1: TV-PR组合周频OOS IC > 等权7路
  G2: 三分子周期Sharpe全 ≥ 等权7路
  G3: 权重年换手量化报告
  G4: 成本后CAGR/Sharpe净改善

输出: reports/gold/tvpr_weights.csv / tvpr_experiments.csv
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

from tvpr_estimator import tvpr_admm

from QuantNodes.strategy.commodity_fundamental.gold import (
    build_production_position,
    compute_risk_overlay,
    load_au_adj_ohlcv,
    load_indicator_d,
    load_vix_gvz,
    o2o_returns,
    smooth_execution,
    weekly_grid_inventory,
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
from QuantNodes.strategy.commodity_fundamental.gold.gold_usd import (
    compute_usd_signal,
)

OUTPUT_DIR = "reports/gold"
COST_BPS = 5.0
MIN_WINDOW = 104          # expanding窗口下限 (2年)
LAMBDA_TV_GRID = [0.05, 0.25, 1.0]
LAMBDA_L1_GRID = [0.02, 0.1, 0.5]

# 一层风控参数 (镜像 compute_layered_ensemble_signal 内部, daily_cap用生产值)
L1_PARAMS = dict(hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.05,
                 vol_cap=0.15, cooldown_days=5)


def perf(r: pd.Series) -> dict:
    """六指标."""
    if r is None or len(r) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "mdd": np.nan, "calmar": np.nan}
    nav = (1 + r).cumprod()
    n = len(r)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return {"cagr": cagr, "sharpe": cagr / vol if vol > 0 else np.nan,
            "mdd": dd, "calmar": cagr / abs(dd) if dd != 0 else np.nan}


def build_daily_factors(close: pd.Series) -> dict[str, pd.Series]:
    """7候选因子的日频仓位 (PIT, 与各自原实现一致)."""
    vix, gvz = load_vix_gvz(close.index)
    tips = load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill().shift(2)

    macro = compute_macro_regime_signal(vix=vix, gvz=gvz, tips=tips)["position"]
    rr = compute_real_rate_signal(variant="tips_ma", tips_daily=tips)["position"]
    mom = compute_momentum_signal(variant="long_only")["position"]
    bo = compute_breakout_signal()["position"]
    risk = compute_risk_signal()["position"]
    pf = compute_position_flow_signal()["position"]
    usd = compute_usd_signal()["position"]

    factors = {"macro_regime": macro, "real_rate": rr, "momentum": mom,
               "breakout": bo, "risk": risk, "position_flow": pf, "usd": usd}
    return {k: v.reindex(close.index).ffill().fillna(0) for k, v in factors.items()}


def first_layer(pos_daily: pd.Series, close: pd.Series,
                vix: pd.Series, gvz: pd.Series) -> pd.Series:
    """一层风控 (镜像layered设计: 各路信号独立过风控)."""
    return compute_risk_overlay(close=close, position=pos_daily, vix=vix, gvz=gvz,
                                **L1_PARAMS)


def apply_stack(signal_daily: pd.Series, close: pd.Series,
                vix: pd.Series, gvz: pd.Series) -> pd.Series:
    """生产风控栈 (二层→死带→核心裁剪→网格→总封顶→平滑→应急)."""
    pos_lev = (signal_daily * 1.5).clip(0, 1.8)
    pos = compute_risk_overlay(close=close, position=pos_lev, vix=vix, gvz=gvz,
                               hard_cap=-0.20, soft_cap=-0.12, daily_cap=-0.05,
                               vol_cap=0.20, cooldown_days=5)
    for i in range(1, len(pos)):
        if abs(pos.iloc[i] - pos.iloc[i - 1]) < 0.05:
            pos.iloc[i] = pos.iloc[i - 1]
    core = pos.clip(0, 1.25)
    grid = weekly_grid_inventory(close, close.index,
                                 spacing_pct=0.02, step=0.0833, max_grids=3)
    out = (core + grid).clip(0, 1.5)
    return smooth_execution(out, close, cap=0.20, emergency=-0.05)


def deadband(pos: pd.Series, th: float = 0.05) -> pd.Series:
    """死带平滑 (与生产一致)."""
    out = pos.copy()
    for i in range(1, len(out)):
        if abs(out.iloc[i] - out.iloc[i - 1]) < th:
            out.iloc[i] = out.iloc[i - 1]
    return out


def fit_bic(Y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, tuple]:
    """窗口内λ网格 + BIC选择. X已标准化. 返回 (beta_end, (λtv,λl1)).

    BIC = T·ln(MSE/T) + k_eff·ln(T), k_eff = 窗口末端非零β数
    """
    T = Y.shape[0]
    best = None
    best_cfg = None
    best_beta_end = None
    for ltv in LAMBDA_TV_GRID:
        for ll1 in LAMBDA_L1_GRID:
            cfg = (ltv, ll1)
            beta = tvpr_admm(Y, X, lambda_tv=ltv, lambda_l1=ll1, max_iter=200)
            resid = Y[:, 0] - (X[:, 0, :] * beta).sum(axis=1)
            mse = float((resid ** 2).mean())
            k_eff = int((np.abs(beta[-1]) > 1e-8).sum())
            bic = T * np.log(max(mse, 1e-12)) + k_eff * np.log(T)
            beta_end = beta[-1]
            if best is None or bic < best:
                best, best_cfg, best_beta_end = bic, cfg, beta_end
    return best_beta_end, best_cfg


def rolling_tvpr_weights(Xw: pd.DataFrame, yw: pd.Series) -> tuple[pd.DataFrame, list]:
    """PIT滚动TV-PR: expanding窗口, 月度重拟合, 输出周频权重表.

    不做跨窗口warm-start (窗口T变化导致β维度不匹配); 跨λ cfg在同窗口内无warm (β shape一致).
    """
    weeks = Xw.index
    w_out = pd.DataFrame(0.0, index=weeks, columns=Xw.columns)
    cur_w = np.full(len(Xw.columns), 1.0 / len(Xw.columns))
    refit_dates, cfg_hist = [], []

    for i in range(len(weeks)):
        d = weeks[i]
        need_refit = (i >= MIN_WINDOW) and (
            i == MIN_WINDOW or d.month != weeks[i - 1].month
        )
        if need_refit:
            win_X = Xw.iloc[:i]
            win_y = yw.iloc[:i]
            common_idx = win_X.index.intersection(win_y.dropna().index)
            X_arr = win_X.loc[common_idx].values
            Y_arr = win_y.loc[common_idx].values.reshape(-1, 1)
            mu, sd = X_arr.mean(axis=0), X_arr.std(axis=0) + 1e-8
            Xz = ((X_arr - mu) / sd).reshape(len(common_idx), 1, -1)
            beta_end, cfg = fit_bic(Y_arr, Xz)
            b = np.maximum(beta_end, 0.0)
            cur_w = b / b.sum() if b.sum() > 1e-10 else np.full(len(b), 1.0 / len(b))
            refit_dates.append(d)
            cfg_hist.append(cfg)
    return w_out, list(zip(refit_dates, cfg_hist))


def main():
    global MIN_WINDOW
    parser = argparse.ArgumentParser(description="TV-PR动态权重因子筛选 (黄金, 周频)")
    parser.add_argument("--min-window", type=int, default=MIN_WINDOW)
    args = parser.parse_args()
    MIN_WINDOW = args.min_window

    print("=" * 70)
    print("TV-PR动态权重因子筛选 — 黄金周频 (PIT)")
    print("=" * 70)

    close, open_price = load_au_adj_ohlcv()
    vix, gvz = load_vix_gvz(close.index)
    factors = build_daily_factors(close)
    names = list(factors.keys())
    print(f"\n因子池 ({len(names)}): {', '.join(names)}")

    # ===== 周频面板 =====
    widx = close.resample("W-FRI").last().dropna().index
    Xw = pd.DataFrame({k: v.resample("W-FRI").last().reindex(widx).ffill().fillna(0)
                       for k, v in factors.items()})
    yw = close.resample("W-FRI").last().pct_change().shift(-1).reindex(widx)

    # ===== 一层风控 (逐因子, layered设计) =====
    print("\n[1] 一层风控 (7路独立 compute_risk_overlay)...")
    ov = {k: first_layer(v, close, vix, gvz) for k, v in factors.items()}

    # ===== TV-PR滚动估计 =====
    n_grid = len(LAMBDA_TV_GRID) * len(LAMBDA_L1_GRID)
    print(f"[2] TV-PR滚动估计 (expanding≥{MIN_WINDOW}周, 月度重拟合, λ网格{n_grid}组+BIC)...")
    weights, refits = rolling_tvpr_weights(Xw, yw)
    cfg_hist = [c for _, c in refits]
    print(f"    重拟合 {len(refits)} 次; λ组合分布: "
          f"{pd.Series([str(c) for c in cfg_hist]).value_counts().to_dict()}")

    # ===== 三组合信号层 =====
    print("\n[3] 组合信号层 (周权重→日频, 死带0.05)...")
    w_daily = weights.reindex(close.index).ffill().fillna(1.0 / len(names))
    # E-TVPR
    sig_tvpr = sum(w_daily[k] * ov[k] for k in names).clip(0, 1.2)
    sig_tvpr = deadband(sig_tvpr)
    # E-等权7
    sig_eq7 = deadband(sum(ov[k] for k in names).clip(0, 1.2) / len(names))
    # E-基线: 生产信号 (内部已含一层+死带)
    from QuantNodes.strategy.commodity_fundamental.gold import (
        compute_layered_ensemble_signal,
    )
    sig_base = compute_layered_ensemble_signal(use_risk_overlay=True, daily_cap=-0.05)["position"]

    # ===== 套栈 + 对照 =====
    print("[4] 套生产风控栈...")
    pos_base = apply_stack(sig_base, close, vix, gvz)
    pos_eq7 = apply_stack(sig_eq7, close, vix, gvz)
    pos_tvpr = apply_stack(sig_tvpr, close, vix, gvz)

    # 锚点互锁
    prod = build_production_position()
    ok = np.allclose(pos_base.reindex(prod.index).fillna(0).values,
                     prod.values, atol=1e-10)
    print(f"\n{'✅' if ok else '❌'} 锚点互锁: E-基线 "
          f"{'==' if ok else '≠'} build_production_position()")

    rows = {}
    for name, p in [("E-基线(等权4路)", pos_base), ("E-等权7路", pos_eq7),
                    ("E-TVPR动态7路", pos_tvpr)]:
        r = o2o_returns(p, open_price)
        m = perf(r)
        sub = [perf(r[(s <= r.index) & (r.index <= e)]) for s, e in
               [("2019-01-01", "2021-12-31"), ("2022-01-01", "2024-12-31"),
                ("2025-01-01", "2026-12-31")]]
        rows[name] = {"cagr": m["cagr"], "sharpe": m["sharpe"], "mdd": m["mdd"],
                      "calmar": m["calmar"],
                      "s1": sub[0]["sharpe"], "s2": sub[1]["sharpe"], "s3": sub[2]["sharpe"]}
    res = pd.DataFrame(rows).T

    # ===== G1: 周频OOS IC =====
    common = yw.dropna().index.intersection(w_daily.index)
    ic_t, ic_e = [], []
    for d in common:
        fwd = yw.loc[d]
        if np.isnan(fwd):
            continue
        sig_t = sum(w_daily.loc[d, k] * ov[k].shift(1).asof(d) for k in names) \
            if not np.isnan(w_daily.loc[d].sum()) else np.nan
        sig_e = np.mean([ov[k].shift(1).asof(d) for k in names])
        if not (np.isnan(sig_t)):
            ic_t.append((sig_t, fwd))
            ic_e.append((sig_e, fwd))

    def spearman_ic(pairs):
        if len(pairs) < 10:
            return np.nan
        a = pd.Series([x[0] for x in pairs])
        b = pd.Series([x[1] for x in pairs])
        return a.corr(b, method="spearman")

    ic_tvpr, ic_eq7 = spearman_ic(ic_t), spearman_ic(ic_e)
    res["oos_ic"] = [np.nan, ic_eq7, ic_tvpr]

    # ===== G3: 权重换手 =====
    wchg = weights.diff().abs().sum(axis=1).dropna()
    turnover_ann = wchg.groupby(wchg.index.year).mean().mean() * 12  # 月度重拟合→年化

    # ===== 门槛判定 =====
    tvpr, eq7 = res.loc["E-TVPR动态7路"], res.loc["E-等权7路"]
    g1 = ic_tvpr > ic_eq7
    g2 = all(tvpr[f"s{i}"] >= eq7[f"s{i}"] for i in (1, 2, 3))
    g4 = (tvpr["cagr"] > eq7["cagr"]) and (tvpr["sharpe"] > eq7["sharpe"])

    print("\n" + "=" * 70)
    print("实验结果 (O2O+5bps)")
    print("=" * 70)
    print(res.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.3f}"))

    print("\nWalk-forward门槛:")
    print(f"  G1 OOS IC:  TVPR {ic_tvpr:+.4f} vs 等权7 {ic_eq7:+.4f}  → {'✅' if g1 else '❌'}")
    print(f"  G2 子周期:  s1 {tvpr['s1']:.2f}vs{eq7['s1']:.2f} | s2 {tvpr['s2']:.2f}vs{eq7['s2']:.2f} "
          f"| s3 {tvpr['s3']:.2f}vs{eq7['s3']:.2f}  → {'✅' if g2 else '❌'}")
    print(f"  G3 权重年换手: {turnover_ann:.2f} (月度重拟合×|Δw|)")
    print(f"  G4 净改善:   CAGR {tvpr['cagr']-eq7['cagr']:+.2%}, "
          f"Sharpe {tvpr['sharpe']-eq7['sharpe']:+.3f}  → {'✅' if g4 else '❌'}")
    verdict = "通过" if (g1 and g2 and g4) else "未通过"
    print(f"\n★ TV-PR动态权重: 门槛{verdict} "
          f"{'(G3为观察项)' if verdict == '通过' else '(负结果, 归档对照L2失败)'}")

    # ===== 输出 =====
    weights.to_csv(f"{OUTPUT_DIR}/tvpr_weights.csv", float_format="%.6f")
    res.to_csv(f"{OUTPUT_DIR}/tvpr_experiments.csv", float_format="%.6f")
    print(f"\n✓ 已输出: {OUTPUT_DIR}/tvpr_weights.csv, {OUTPUT_DIR}/tvpr_experiments.csv")


if __name__ == "__main__":
    main()
