# coding=utf-8
"""风格轮动 (StyleRotation) 诊断研究 — 什么让它赢/输.

分析维度:
1. 年度表现分解 (which years work, which don't)
2. 市场 regime 分解 (HS300 60d/120d/252d 动量 → bull/bear/sideways)
3. 选股正确率: Top-1 选择 vs 实际最佳风格
4. 风格动量持续性: 选股后 20d/40d/60d 表现
5. 趋势强度 regime (强趋势 vs 弱趋势) 的策略表现
6. 等权 5 风格基准对比 (策略有没有真正创造 alpha)
7. 调仓频率敏感性 (M vs W-FRI)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.momentum_etf_rotation.v4.style_rotation_v4 import (
    STYLE_GROUP_CODES,
    StyleGroup,
    style_rotation_score,
    select_top_styles,
    style_etf_picks,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.universe_v4 import (
    load_smartbeta_panel,
)

REPO = Path("/home/ll/Public/QuantNodes")
START = "2018-01-01"
END = "2026-06-30"

STYLE_NAMES = {
    "large_cap": "大盘",
    "mid_cap": "中盘",
    "growth": "成长",
    "tech": "科创",
    "dividend": "红利",
}


def ann_return(nav: pd.Series) -> float:
    r = nav.iloc[-1] / nav.iloc[0]
    n = (nav.index[-1] - nav.index[0]).days / 365.25
    return float(r ** (1 / n) - 1) if n > 0 else 0.0


def max_dd(nav: pd.Series) -> float:
    pk = nav.cummax()
    dd = nav / pk - 1.0
    return float(dd.min())


def sharpe(daily_ret: pd.Series) -> float:
    if daily_ret.std() == 0:
        return 0.0
    return float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))


def metrics(nav: pd.Series) -> dict:
    rets = nav.pct_change().dropna()
    ar = ann_return(nav)
    dd = max_dd(nav)
    return {
        "ann_return": ar,
        "ann_vol": float(rets.std() * np.sqrt(252)),
        "sharpe": sharpe(rets),
        "max_dd": dd,
        "calmar": ar / abs(dd) if dd != 0 else 0.0,
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1),
    }


def simulate_style_rotation(
    panel: pd.DataFrame,
    L: int = 120,
    top_n: int = 1,
    top_n_per_style: int = 1,
    trend_weight: float = 0.0,
    rebal_freq: str = "M",
) -> tuple[pd.Series, pd.DataFrame]:
    """模拟风格轮动 + 记录调仓日志.

    Returns:
        nav: NAV 时序
        log_df: 调仓日志 (date, chosen, chosen_score, all_scores)
    """
    dates = panel.index
    if rebal_freq == "M":
        rebal_dates = dates.to_series().resample("ME").last().index
    elif rebal_freq == "W-FRI":
        rebal_dates = dates.to_series().resample("W-FRI").last().index
    else:
        rebal_dates = dates.to_series().resample(rebal_freq).last().index

    rebal_dates = [d for d in rebal_dates if d in dates]

    nav = np.ones(len(dates))
    log_rows = []

    chosen_history: list[tuple[pd.Timestamp, list[str]]] = []
    last_chosen: list[str] = []
    for i, date in enumerate(dates):
        if date in rebal_dates and i > 252:
            scores = style_rotation_score(
                panel, date, STYLE_GROUP_CODES,
                lookback=L, trend_lookback=L, trend_weight=trend_weight,
            )
            if scores.empty:
                last_chosen = last_chosen or list(STYLE_GROUP_CODES.keys())
                log_rows.append({"date": date, "chosen": last_chosen})
                continue

            top_styles = select_top_styles(scores, top_n)
            picks = style_etf_picks(
                panel, date, STYLE_GROUP_CODES,
                top_styles, top_n_per_style,
            )
            if picks:
                last_chosen = picks
                chosen_history.append((date, picks))
                log_rows.append({
                    "date": date,
                    "chosen": picks,
                    "chosen_styles": [s.value for s in top_styles],
                    "top_score": float(scores.iloc[0]),
                    "all_scores": scores.to_dict(),
                })

        if last_chosen:
            for code in last_chosen:
                if code in panel.columns:
                    nav[i] *= panel[code].iloc[i] / panel[code].iloc[i - 1]
        else:
            nav[i] = 1.0

    nav_series = pd.Series(nav, index=dates, name="nav")
    log_df = pd.DataFrame(log_rows)
    return nav_series, log_df


def classify_market_regime(panel: pd.DataFrame) -> pd.Series:
    """基于 HS300 (510300) 的 60d/120d/252d 动量分类市场 regime.

    Returns:
        pd.Series, index=date, values=regime name
    """
    px = panel["510300"]
    mom60 = px.pct_change(60)
    mom120 = px.pct_change(120)
    mom252 = px.pct_change(252)
    regime = pd.Series("sideways", index=panel.index)
    regime[(mom60 > 0.05) & (mom252 > 0.10)] = "bull"
    regime[(mom60 < -0.05) & (mom252 < -0.10)] = "bear"
    return regime


def classify_trend_strength(panel: pd.DataFrame, window: int = 60) -> pd.Series:
    """HS300 距离 60d MA 的偏离 — 趋势强度.

    Returns:
        pd.Series, index=date, values=strong_up / weak_up / weak_down / strong_down / neutral
    """
    px = panel["510300"]
    ma = px.rolling(window).mean()
    dist = (px / ma - 1.0)
    out = pd.Series("neutral", index=panel.index)
    out[dist > 0.10] = "strong_up"
    out[(dist > 0.02) & (dist <= 0.10)] = "weak_up"
    out[(dist < -0.02) & (dist >= -0.10)] = "weak_down"
    out[dist < -0.10] = "strong_down"
    return out


def main():
    panel = load_smartbeta_panel()
    print(f"[data] {panel.shape[0]} days × {panel.shape[1]} codes")
    print(f"[data] 5 风格组: {list(STYLE_NAMES.values())}")

    print("\n========= 1. 基准回测: 多配置对比 =========")
    configs = [
        (60, 1, 1, 0.0, "M", "L60_T1_tw0_M"),
        (60, 1, 1, 0.0, "W-FRI", "L60_T1_tw0_W"),
        (60, 3, 1, 0.3, "M", "L60_T3_tw0.3_M"),
        (90, 1, 1, 0.0, "M", "L90_T1_tw0_M"),
        (120, 1, 1, 0.0, "M", "L120_T1_tw0_M"),
        (120, 3, 1, 0.0, "M", "L120_T3_tw0_M"),
        (120, 1, 1, 0.3, "M", "L120_T1_tw0.3_M"),
        (144, 1, 1, 0.0, "M", "L144_T1_tw0_M"),
        (180, 1, 1, 0.0, "M", "L180_T1_tw0_M"),
        (252, 1, 1, 0.0, "M", "L252_T1_tw0_M"),
    ]
    navs = {}
    for L, tn, tps, tw, freq, name in configs:
        nav, log = simulate_style_rotation(panel, L, tn, tps, tw, freq)
        m = metrics(nav)
        m["name"] = name
        m["L"] = L
        m["top_n"] = tn
        m["trend_weight"] = tw
        m["freq"] = freq
        navs[name] = (nav, log, m)
        print(f"  {name:30s}  Ann={m['ann_return']*100:5.2f}%  Vol={m['ann_vol']*100:5.2f}%  "
              f"Sharpe={m['sharpe']:5.2f}  DD={m['max_dd']*100:6.2f}%  Calmar={m['calmar']:5.3f}")

    print("\n========= 2. 年度表现 (L120_T1_tw0) =========")
    nav, log, m = navs["L120_T1_tw0_M"]
    yearly = nav.resample("YE").last() / nav.resample("YE").first() - 1
    for year, ret in yearly.items():
        year_str = year.year
        print(f"  {year_str}: {ret * 100:6.2f}%")

    print("\n========= 3. 等权 5 风格基准对比 =========")
    eq_w = np.ones(len(panel)) / len(STYLE_GROUP_CODES)
    eq_nav = np.ones(len(panel))
    for i in range(1, len(panel)):
        for code in [c for codes in STYLE_GROUP_CODES.values() for c in codes]:
            if code in panel.columns:
                eq_nav[i] += eq_w[i] * (panel[code].iloc[i] / panel[code].iloc[i - 1] - 1)
    eq_nav = pd.Series(eq_nav, index=panel.index, name="eq5_nav")
    eq_m = metrics(eq_nav)
    print(f"  等权 5 风格: Ann={eq_m['ann_return']*100:.2f}% Vol={eq_m['ann_vol']*100:.2f}% "
          f"Sharpe={eq_m['sharpe']:.2f} DD={eq_m['max_dd']*100:.2f}% Calmar={eq_m['calmar']:.3f}")
    for code in [c for codes in STYLE_GROUP_CODES.values() for c in codes]:
        if code in panel.columns:
            n = panel[code]
            m_single = metrics(n / n.iloc[0])
            print(f"  单独持有 {code}: Ann={m_single['ann_return']*100:.2f}% "
                  f"Sharpe={m_single['sharpe']:.2f} DD={m_single['max_dd']*100:.2f}%")

    print("\n========= 4. Regime 分解 (L120_T1_tw0) =========")
    regime = classify_market_regime(panel)
    nav_daily_ret = nav.pct_change().dropna()
    regime_aligned = regime.reindex(nav_daily_ret.index, method="ffill")
    regime_stats = {}
    for r in ["bull", "bear", "sideways"]:
        mask = regime_aligned == r
        if mask.sum() > 20:
            rets_in_regime = nav_daily_ret[mask]
            ann_ret = rets_in_regime.mean() * 252
            ann_vol = rets_in_regime.std() * np.sqrt(252)
            sh = rets_in_regime.mean() / rets_in_regime.std() * np.sqrt(252) if rets_in_regime.std() > 0 else 0
            n_days = mask.sum()
            regime_stats[r] = {
                "ann_return": float(ann_ret),
                "ann_vol": float(ann_vol),
                "sharpe": float(sh),
                "n_days": int(n_days),
            }
            print(f"  {r:10s}  n={n_days:4d}  Ann={ann_ret*100:6.2f}%  "
                  f"Vol={ann_vol*100:5.2f}%  Sharpe={sh:5.2f}")

    print("\n========= 5. 趋势强度 regime 分解 =========")
    trend = classify_trend_strength(panel)
    trend_aligned = trend.reindex(nav_daily_ret.index, method="ffill")
    for t in ["strong_up", "weak_up", "neutral", "weak_down", "strong_down"]:
        mask = trend_aligned == t
        if mask.sum() > 20:
            rets_in_regime = nav_daily_ret[mask]
            ann_ret = rets_in_regime.mean() * 252
            ann_vol = rets_in_regime.std() * np.sqrt(252)
            sh = rets_in_regime.mean() / rets_in_regime.std() * np.sqrt(252) if rets_in_regime.std() > 0 else 0
            n_days = mask.sum()
            print(f"  {t:12s}  n={n_days:4d}  Ann={ann_ret*100:6.2f}%  "
                  f"Vol={ann_vol*100:5.2f}%  Sharpe={sh:5.2f}")

    print("\n========= 6. 选股正确率分析 (L120_T1_tw0) =========")
    rebal_dates = panel.index.to_series().resample("ME").last().index
    rebal_dates = [d for d in rebal_dates if d in panel.index]
    correct_top1 = 0
    correct_top2 = 0
    total_rebal = 0
    choice_log = []
    for date in rebal_dates:
        idx = panel.index.get_loc(date)
        if idx < 252 or idx >= len(panel) - 60:
            continue
        scores = style_rotation_score(
            panel, date, STYLE_GROUP_CODES,
            lookback=120, trend_lookback=120, trend_weight=0.0,
        )
        if scores.empty:
            continue
        top_styles = select_top_styles(scores, 1)
        if not top_styles:
            continue
        chosen_code = STYLE_GROUP_CODES[top_styles[0]][0]
        forward_window = 60
        future = panel.iloc[idx + 1: idx + 1 + forward_window]
        if len(future) < forward_window:
            continue
        all_returns = {}
        for g, codes in STYLE_GROUP_CODES.items():
            for c in codes:
                if c in future.columns and not future[c].isna().all():
                    all_returns[g.value] = float(future[c].iloc[-1] / future[c].iloc[0] - 1)
        if not all_returns:
            continue
        best_style = max(all_returns, key=all_returns.get)
        chosen_style = top_styles[0].value
        is_top1 = (chosen_style == best_style)
        ranked = sorted(all_returns, key=lambda k: -all_returns[k])
        is_top2 = (chosen_style in ranked[:2])
        if is_top1:
            correct_top1 += 1
        if is_top2:
            correct_top2 += 1
        total_rebal += 1
        choice_log.append({
            "date": date,
            "chosen_style": chosen_style,
            "chosen_code": chosen_code,
            "best_style": best_style,
            "best_return": all_returns[best_style],
            "chosen_return": all_returns.get(chosen_style, None),
            "is_top1": is_top1,
            "is_top2": is_top2,
        })
    if total_rebal > 0:
        print(f"  调仓次数: {total_rebal}")
        print(f"  Top-1 选股正确率: {correct_top1 / total_rebal * 100:.1f}%")
        print(f"  Top-2 选股正确率: {correct_top2 / total_rebal * 100:.1f}%")
        if choice_log:
            df_log = pd.DataFrame(choice_log)
            df_log["excess"] = df_log["chosen_return"] - df_log["best_return"]
            print(f"  平均 excess (chosen - best): {df_log['excess'].mean() * 100:.2f}% per 60d")
            print(f"  选中的平均 60d 收益: {df_log['chosen_return'].mean() * 100:.2f}%")
            print(f"  最佳的 60d 收益:    {df_log['best_return'].mean() * 100:.2f}%")

    print("\n========= 7. 风格动量持续性: L=60/120 vs 实际收益 (前瞻 60d) =========")
    for L in [60, 90, 120, 180]:
        forward_returns = []
        score_returns = []
        for date in rebal_dates:
            idx = panel.index.get_loc(date)
            if idx < L + 1 or idx >= len(panel) - 60:
                continue
            scores = style_rotation_score(
                panel, date, STYLE_GROUP_CODES,
                lookback=L, trend_lookback=L, trend_weight=0.0,
            )
            if scores.empty or len(scores) < 2:
                continue
            top_style = scores.index[0]
            chosen_code = STYLE_GROUP_CODES[top_style][0]
            future = panel.iloc[idx + 1: idx + 1 + 60]
            if chosen_code in future.columns and not future[chosen_code].isna().all():
                fwd = float(future[chosen_code].iloc[-1] / future[chosen_code].iloc[0] - 1)
                score_returns.append(scores.iloc[0])
                forward_returns.append(fwd)
        if forward_returns:
            corr = np.corrcoef(score_returns, forward_returns)[0, 1]
            print(f"  L={L}: 得分 vs 实际 60d 收益 相关性 = {corr:.3f}  "
                  f"avg_fwd = {np.mean(forward_returns) * 100:6.2f}%  "
                  f"median_fwd = {np.median(forward_returns) * 100:6.2f}%")

    print("\n========= 8. 调仓频率: M vs W-FRI =========")
    for freq, name in [("M", "M"), ("W-FRI", "W-FRI")]:
        nav, _, m = navs.get(f"L120_T1_tw0_{freq[0]}", (None, None, None))
        if nav is None:
            continue
        print(f"  L120_T1 {freq:5s}  n_rebal={log.shape[0] if 'log' in dir() else '?'}  "
              f"Ann={m['ann_return']*100:.2f}%  Calmar={m['calmar']:.3f}")

    print("\n========= 9. 风格组间交叉相关 (60d 收益) =========")
    style_codes = {s.value: c[0] for s, c in STYLE_GROUP_CODES.items()}
    style_rets = pd.DataFrame()
    for s, code in style_codes.items():
        if code in panel.columns:
            style_rets[s] = panel[code].pct_change(60)
    print(style_rets.corr().round(2).to_string())

    out = {
        "regime_stats": regime_stats,
        "config_metrics": {k: v[2] for k, v in navs.items()},
        "eq5_benchmark": eq_m,
    }
    out_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "style_rotation_diagnostic.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[save] {out_dir / 'style_rotation_diagnostic.json'}")


if __name__ == "__main__":
    main()
