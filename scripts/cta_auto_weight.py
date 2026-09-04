# coding=utf-8
"""cta_auto_weight.py — 自动配权重机制 (§24 C组): ERC + inv_vol.

约束 (用户拍板): 每个子策略 ≤10%, 主CTA ∈ [60%,75%]
候选池: 主CTA / CU溢价TC / jjm_coke / ZN库存 / AU_layered (5成分)
机制: 月度再平衡 + 死带2% (变化<2pp不调) + 上下限clip后重归一
    ERC:    rolling cov(win) → 等风险贡献迭代解
    inv_vol: rolling 252d σ 倒数 (对照地板)
基线 (同约束固定权重):
    B1: 主CTA 70 + CU 10 + jjm 10 + AU 10 (无ZN)
    B2: 主CTA 60 + CU 10 + jjm 10 + AU 10 + ZN 10 (满配)
网格: ERC cov窗口 {63,126,252} × 再平衡 {月,季} — 平滑性检验, 不挑单点
判定 (预设): 全样本 ≥ max(B1,B2)×0.98 且 DD不恶化; IS/OOS分段同验;
    触发率<20%, 机制换手增幅<5bp/年
输出: reports/cta/auto_weight.csv + 权重路径
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts")

from cta_shared.portfolio_tools import (  # noqa: E402
    DEFAULT_LEV_COST, dynamic_leverage, levered_portfolio, multi_window_factors,
)
from cta_t10y_removal import seg_metrics  # noqa: E402

REPORT_DIR = "reports/cta"
IS_END = "2022-01-01"
LEV_PARAMS = dict(window=60, vol_lo=0.08, vol_hi=0.12, lev_lo=1.0, lev_hi=1.5)
COST_BPS = 5.0
W_MIN, W_MAX = 0.60, 0.75      # 主CTA
SUB_MAX = 0.10                 # 子策略上限
DEADBAND = 0.02                # 死带 2pp
SUBS = ["CU溢价TC", "jjm_coke", "ZN库存", "AU_layered"]
MAIN = "主CTA"


def load_components() -> pd.DataFrame:
    golden_path = f"{REPORT_DIR}/overfit_final_navs/基线_严格OOS0.5,60_2018.parquet"
    main_ret = pd.read_parquet(golden_path)["nav"].pct_change(fill_method=None).fillna(0)
    rets = {MAIN: main_ret}
    files = {"CU溢价TC": "rebuild_subnav_CU溢价TC_open.csv",
             "jjm_coke": "rebuild_subnav_jjm_coke_open.csv",
             "ZN库存": "rebuild_subnav_ZN库存_open.csv"}
    for name, f in files.items():
        nav = pd.read_csv(f"{REPORT_DIR}/{f}", index_col=0, parse_dates=True).iloc[:, 0]
        rets[name] = nav[nav.index <= "2025-12-31"].pct_change(fill_method=None).fillna(0)
    au = pd.read_parquet(f"{REPORT_DIR}/gold_layered_layered.parquet")["ret"]
    rets["AU_layered"] = au.reindex(main_ret.index).fillna(0)
    df = pd.DataFrame(rets)
    return df.reindex(df.dropna(how="all").index).fillna(0)


def clip_normalize(w: pd.Series) -> pd.Series:
    """上下限 clip + 主CTA吸收残余 + 归一 (总=1)."""
    w = w.copy()
    w[SUBS] = w[SUBS].clip(upper=SUB_MAX, lower=0.0)
    w[MAIN] = w[MAIN].clip(W_MIN, W_MAX)
    total = w.sum()
    w = w / total
    # 主CTA吸收clip残余, 保证总和1且主CTA在限内尽量满足
    if w[MAIN] < W_MIN:
        extra = W_MIN - w[MAIN]
        sub_total = w[SUBS].sum()
        if sub_total > 0:
            w[SUBS] = w[SUBS] * (1 - extra / sub_total)
        w[MAIN] = W_MIN
    elif w[MAIN] > W_MAX:
        excess = w[MAIN] - W_MAX
        w[SUBS] = w[SUBS] + excess * (w[SUBS] / w[SUBS].sum() if w[SUBS].sum() else 0)
        w[MAIN] = W_MAX
    return w


def erc_weights(cov: np.ndarray) -> np.ndarray:
    """等风险贡献: 迭代解 w_i*(Σw)_i = const."""
    n = cov.shape[0]
    w = np.ones(n) / n
    for _ in range(500):
        mrc = cov @ w
        rc = w * mrc
        target = rc.sum() / n
        # 牛顿式调整: 按风险贡献偏离缩放
        w_new = w * (target / np.maximum(rc, 1e-12)) ** 0.5
        w_new = np.clip(w_new, 1e-6, None)
        w_new /= w_new.sum()
        if np.abs(w_new - w).max() < 1e-8:
            w = w_new
            break
        w = w_new
    return w


def gen_weights(rets: pd.DataFrame, method: str, window: int,
                freq: str) -> pd.DataFrame:
    """月末生成下月权重路径 (纯rolling, PIT安全)."""
    idx = rets.index
    month_ends = idx.to_series().resample({"M": "M", "Q": "Q"}[freq]).last().dropna()
    month_ends = month_ends[month_ends >= idx[120]]  # warmup 120天
    W = pd.DataFrame(0.0, index=month_ends.index, columns=rets.columns)
    prev = None
    for t in month_ends.index:
        hist = rets.loc[rets.index <= t].tail(window)
        if method == "erc":
            cov = hist.cov().values
            w = pd.Series(erc_weights(cov), index=rets.columns)
        else:  # inv_vol
            vol = hist.std() * np.sqrt(252)
            w = (1 / vol.replace(0, np.nan)).fillna(0)
            w = w / w.sum()
        w = clip_normalize(w)
        if prev is not None:
            drift = (w - prev).abs()
            if drift[SUBS].max() < DEADBAND and abs(w[MAIN] - prev[MAIN]) < DEADBAND:
                w = prev  # 死带: 不动
        W.loc[t] = w
        prev = w
    return W


def run_weighted(rets: pd.DataFrame, W: pd.DataFrame) -> pd.Series:
    """月度权重 → 组合日收益 (逐月分段, 边界扣换手5bp)."""
    fac_A = multi_window_factors(rets, skip_cols=(MAIN,), windows=(20, 40, 60), cut=0.5)
    lev = dynamic_leverage(rets[MAIN], **LEV_PARAMS)
    days = pd.Series(1, index=W.index).resample("D").asfreq().index
    w_daily = W.reindex(days, method="ffill").reindex(rets.index).ffill().dropna()
    out = []
    prev_w = None
    for t, grp in w_daily.groupby([w_daily.index.year, w_daily.index.month]):
        w = grp.iloc[-1]
        sub = rets.loc[grp.index]
        fac = fac_A.loc[grp.index]
        lev_s = lev.loc[grp.index]
        gross = levered_portfolio(sub * fac, w.to_dict(), lev_series=lev_s,
                                 cost_annual=DEFAULT_LEV_COST)
        # 组内权重恒定, 只在月边界扣换手
        tc = pd.Series(0.0, index=grp.index)
        if prev_w is not None:
            turn = (w - prev_w).abs().sum()
            tc.iloc[0] = turn * COST_BPS / 10000
        out.append((gross - tc).dropna())
        prev_w = w
    return pd.concat(out).sort_index()


def main() -> None:
    rets = load_components()
    print("=" * 104)
    print("C组: 自动配权重 — ERC vs inv_vol (子策略≤10%, 主CTA 60-75%)")
    print("=" * 104)

    # 固定基线
    baselines = {
        "B1固定(无ZN)": pd.Series({MAIN: 0.70, "CU溢价TC": 0.10, "jjm_coke": 0.10,
                                    "AU_layered": 0.10, "ZN库存": 0.0}),
        "B2固定(满配)": pd.Series({MAIN: 0.60, "CU溢价TC": 0.10, "jjm_coke": 0.10,
                                    "AU_layered": 0.10, "ZN库存": 0.10}),
    }
    nets = {}
    for label, w in baselines.items():
        W = pd.DataFrame([w] * len(rets), index=rets.index, columns=rets.columns)
        nets[label] = run_weighted(rets, W)
        m = seg_metrics(nets[label], None, None)
        print(f"  {label:14s} 全 {m['sharpe']:+.3f}/{m['max_dd']:+.2%} "
              f"IS {seg_metrics(nets[label], None, IS_END)['sharpe']:+.3f} "
              f"OOS {seg_metrics(nets[label], IS_END, None)['sharpe']:+.3f}")

    best_base = max(seg_metrics(n, None, None)["sharpe"] for n in nets.values())
    best_dd = max(seg_metrics(n, None, None)["max_dd"] for n in nets.values())

    rows = []
    for method, window, freq in [("erc", 63, "M"), ("erc", 126, "M"), ("erc", 252, "M"),
                                  ("erc", 126, "Q"),
                                  ("inv_vol", 252, "M")]:
        W = gen_weights(rets, method, window, freq)
        net = run_weighted(rets, W)
        nets[f"{method}{window}{freq}"] = net
        m = seg_metrics(net, None, None)
        m_is = seg_metrics(net, None, IS_END)
        m_oos = seg_metrics(net, IS_END, None)
        # 权重路径诊断
        turns = W.diff().abs().sum(axis=1).dropna()
        hit_cap = ((W[SUBS] >= SUB_MAX - 1e-6).any(axis=1)).mean()
        hit_floor = (W[MAIN] <= W_MIN + 1e-6).mean()
        ok = (m["sharpe"] >= best_base * 0.98 and m["max_dd"] >= best_dd - 0.001
              and m_is["sharpe"] >= 0 and m_oos["sharpe"] >= 0)
        print(f"  {method}{window}/{freq[:1]:<1s}      全 {m['sharpe']:+.3f}/"
              f"{m['max_dd']:+.2%} IS {m_is['sharpe']:+.3f} OOS {m_oos['sharpe']:+.3f} "
              f"| 换手{turns.mean():.3f}/月 触发上限{hit_cap:.0%} 触下限{hit_floor:.0%} "
              f"→ {'PASS' if ok else 'FAIL'}")
        rows.append({"method": method, "window": window, "freq": freq,
                     "shp": round(m["sharpe"], 3), "dd": round(m["max_dd"], 4),
                     "is": round(m_is["sharpe"], 3), "oos": round(m_oos["sharpe"], 3),
                     "turn": round(turns.mean(), 3), "hit_cap": round(hit_cap, 2),
                     "hit_floor": round(hit_floor, 2)})
        W.to_parquet(f"{REPORT_DIR}/auto_weight_{method}{window}_{freq[:1]}.parquet")

    # 最终权重路径展示 (ERC 126 月度)
    W = pd.read_parquet(f"{REPORT_DIR}/auto_weight_erc126_M.parquet")
    print("\nERC126月度 权重路径 (年末采样):")
    print(W[W.index.year >= 2018].iloc[::12].round(3).to_string())

    pd.DataFrame(rows).to_csv(f"{REPORT_DIR}/auto_weight.csv", index=False)
    print("\n落盘: auto_weight.csv + auto_weight_*.parquet")


if __name__ == "__main__":
    main()
