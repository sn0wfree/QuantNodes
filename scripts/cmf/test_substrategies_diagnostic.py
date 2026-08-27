"""scripts/cmf/test_substrategies_diagnostic.py — 8子策略单独诊断.

对gold目录下9个未串联子策略, 双窗口回测, 评估独立能力.
对照: Buy&Hold, compute_trend_bnh(ts=0.08), compute_macro_regime_signal().

输出:
  - 全景表 (stdout + docs/gold/substrategies_evaluation.md)
  - 每个策略关键诊断维度
  - 串联候选排名
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

import numpy as np
import pandas as pd

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_usd import (
    compute_usd_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_risk import (
    compute_risk_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_position_flow import (
    compute_position_flow_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
    compute_breakout_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_all_factor import (
    compute_all_factor_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_multi_factor import (
    compute_multi_factor_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_combined import (
    compute_combined_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
    compute_trend_bnh_signal,
)

OOS_S, OOS_E = "2010-01-01", "2019-12-31"
IS_S, IS_E = "2020-01-01", "2026-08-25"
COST_BPS = 5.0


def load_close_daily():
    """日度金价主连."""
    import duckdb
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date", keep="last")
              .set_index("trade_date")["close"])


def load_close_weekly():
    """周度金价 (W-FRI)."""
    return load_close_daily().resample("W-FRI").last().dropna()


def buy_and_hold_daily():
    close = load_close_daily()
    return pd.Series(1.0, index=close.index)


def buy_and_hold_weekly():
    close = load_close_weekly()
    return pd.Series(1.0, index=close.index)


def calc_metrics_freq(pos, close, start, end, freq):
    """指定频率的metrics计算."""
    periods_per_year = 52 if freq == "W" else 252
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "max_dd": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
    sr = (sr - cost).dropna()
    sp = c.iloc[0]
    nav = sp * (1 + sr).cumprod()
    total = nav.iloc[-1] / sp - 1
    n_y = len(c) / periods_per_year
    ar = (1 + total) ** (1 / n_y) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"ann_ret": ar, "max_dd": dd}


def dispatch_position(name, freq):
    """按name重新获取position series."""
    name_lower = name.lower()
    try:
        if "buy & hold (日)" in name:
            return buy_and_hold_daily()
        if "buy & hold (周)" in name:
            return buy_and_hold_weekly()
        if "trend_bnh" in name and "日" in name:
            return compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"]
        if "macro_regime" in name:
            return compute_macro_regime_signal()["position"]
        if "gold_real_rate: tips_ma" in name:
            return compute_real_rate_signal(variant="tips_ma")["position"]
        if "gold_real_rate: tips_level" in name:
            return compute_real_rate_signal(variant="tips_level")["position"]
        if "gold_real_rate: combined" in name:
            return compute_real_rate_signal(variant="combined")["position"]
        if "gold_usd: dxy_ema" in name:
            return compute_usd_signal(variant="dxy_ema")["position"]
        if "gold_usd: dxy_momentum" in name:
            return compute_usd_signal(variant="dxy_momentum")["position"]
        if "gold_usd: dxy_zscore" in name:
            return compute_usd_signal(variant="dxy_zscore")["position"]
        if "gold_risk: combined" in name:
            return compute_risk_signal(variant="combined")["position"]
        if "gold_risk: vix_pct" in name:
            return compute_risk_signal(variant="vix_pct")["position"]
        if "gold_risk: gprd_pct" in name:
            return compute_risk_signal(variant="gprd_pct")["position"]
        if "gold_momentum: long_only" in name:
            return compute_momentum_signal(variant="long_only")["position"]
        if "gold_momentum: dual_window" in name:
            return compute_momentum_signal(variant="dual_window")["position"]
        if "gold_position_flow: combined" in name:
            return compute_position_flow_signal(variant="combined")["position"]
        if "gold_position_flow: cb_only" in name:
            return compute_position_flow_signal(variant="cb_only")["position"]
        if "gold_position_flow: etf_only" in name:
            return compute_position_flow_signal(variant="etf_only")["position"]
        if "gold_breakout" in name:
            return compute_breakout_signal()["position"]
        if "gold_all_factor: balanced" in name:
            return compute_all_factor_signal(mode="balanced")["position"]
        if "gold_all_factor: aggressive" in name:
            return compute_all_factor_signal(mode="aggressive")["position"]
        if "gold_all_factor: conservative" in name:
            return compute_all_factor_signal(mode="conservative")["position"]
        if "gold_multi_factor" in name:
            return compute_multi_factor_signal()["position"]
        if "gold_combined: use_timing" in name:
            return compute_combined_signal(use_timing=True)["position"]
        if "gold_combined" in name:
            return compute_combined_signal()["position"]
    except Exception:
        return None
    return None


def get_freq(pos: pd.Series) -> str:
    """根据index间隔判断频率."""
    if len(pos) < 2:
        return "?"
    delta = pos.index.to_series().diff().median()
    return "W" if delta >= pd.Timedelta(days=5) else "D"


def backtest_pos(pos: pd.Series, freq: str):
    """根据频率返回对应close与年化因子."""
    close = load_close_weekly() if freq == "W" else load_close_daily()
    pos = pos.reindex(close.index).fillna(method="ffill").fillna(0)
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * COST_BPS / 10000
    strat_ret = (strat_ret - cost).dropna()
    return pos, close, strat_ret


def evaluate(pos: pd.Series, freq: str = "D"):
    """双窗口评估 (频率自适应, 避免calc_metrics的252假设误判)."""
    pos_aligned, close, _ = backtest_pos(pos, freq)
    res = {}
    periods_per_year = 52 if freq == "W" else 252
    for label, s, e in [("OOS", OOS_S, OOS_E), ("IS", IS_S, IS_E)]:
        try:
            sub = pos_aligned[(pos_aligned.index >= s) & (pos_aligned.index <= e)]
            c = close.reindex(sub.index)
            if len(c) < 2:
                res[label] = {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "calmar": 0, "turnover": 0}
                continue
            ret = c.pct_change()
            sr = (sub.fillna(0).shift(1) * ret).dropna()
            cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
            sr = (sr - cost).dropna()
            if len(sr) == 0:
                res[label] = {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "calmar": 0, "turnover": 0}
                continue
            sp = c.iloc[0]
            nav = sp * (1 + sr).cumprod()
            total_ret = nav.iloc[-1] / sp - 1
            n_y = len(c) / periods_per_year  # 频率自适应
            ar = (1 + total_ret) ** (1 / n_y) - 1
            vol = sr.std() * np.sqrt(periods_per_year)
            dd = (nav / nav.cummax() - 1).min()
            res[label] = {
                "ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
                "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0,
                "turnover": sub.fillna(0).diff().abs().sum() / 2 / n_y,
            }
        except Exception as e:
            res[label] = {"error": str(e)}
    res["freq"] = freq
    res["avg_pos"] = float(pos_aligned.mean())
    res["pct_zero"] = float((pos_aligned == 0).mean())
    res["pct_active"] = float((pos_aligned > 0).mean())
    return res


def fmt(v, prec=2, pct=True):
    if v is None or (isinstance(v, dict) and "error" in v):
        return "ERR"
    if pct:
        return f"{v*100:.{prec}f}%"
    return f"{v:.{prec}f}"


def main():
    rows = []
    errors = []

    # ===== 基线 =====
    print("=== 基线对照 ===")
    baselines = [
        ("Buy & Hold (日)", buy_and_hold_daily(), "D"),
        ("Buy & Hold (周)", buy_and_hold_weekly(), "W"),
        ("trend_bnh(ts=0.08) 日", compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"], "D"),
        ("macro_regime_signal() 日", compute_macro_regime_signal()["position"], "D"),
    ]
    for label, pos, freq in baselines:
        try:
            r = evaluate(pos, freq)
            rows.append({
                "category": "基线", "name": label, "freq": freq,
                "oos_ann": r["OOS"].get("ann_ret"), "oos_sh": r["OOS"].get("sharpe"),
                "oos_dd": r["OOS"].get("max_dd"), "oos_cal": r["OOS"].get("calmar"),
                "is_ann": r["IS"].get("ann_ret"), "is_sh": r["IS"].get("sharpe"),
                "is_dd": r["IS"].get("max_dd"), "is_cal": r["IS"].get("calmar"),
                "avg_pos": r["avg_pos"], "pct_active": r["pct_active"],
            })
        except Exception as e:
            errors.append(f"baseline {label}: {e}")

    # ===== 子策略扫描 =====
    sub_configs = [
        # (类别, 名称, 调用函数, 频率)
        ("单因子", "gold_real_rate: tips_ma",
         lambda: compute_real_rate_signal(variant="tips_ma")["position"], "W"),
        ("单因子", "gold_real_rate: tips_level",
         lambda: compute_real_rate_signal(variant="tips_level")["position"], "W"),
        ("单因子", "gold_real_rate: combined",
         lambda: compute_real_rate_signal(variant="combined")["position"], "W"),
        ("单因子", "gold_usd: dxy_ema",
         lambda: compute_usd_signal(variant="dxy_ema")["position"], "W"),
        ("单因子", "gold_usd: dxy_momentum",
         lambda: compute_usd_signal(variant="dxy_momentum")["position"], "W"),
        ("单因子", "gold_usd: dxy_zscore",
         lambda: compute_usd_signal(variant="dxy_zscore")["position"], "W"),
        ("单因子", "gold_risk: combined",
         lambda: compute_risk_signal(variant="combined")["position"], "W"),
        ("单因子", "gold_risk: vix_pct",
         lambda: compute_risk_signal(variant="vix_pct")["position"], "W"),
        ("单因子", "gold_risk: gprd_pct",
         lambda: compute_risk_signal(variant="gprd_pct")["position"], "W"),
        ("单因子", "gold_momentum: long_only",
         lambda: compute_momentum_signal(variant="long_only")["position"], "W"),
        ("单因子", "gold_momentum: dual_window",
         lambda: compute_momentum_signal(variant="dual_window")["position"], "W"),
        ("单因子", "gold_position_flow: combined",
         lambda: compute_position_flow_signal(variant="combined")["position"], "W"),
        ("单因子", "gold_position_flow: cb_only",
         lambda: compute_position_flow_signal(variant="cb_only")["position"], "W"),
        ("单因子", "gold_position_flow: etf_only",
         lambda: compute_position_flow_signal(variant="etf_only")["position"], "W"),
        ("单因子", "gold_breakout: default(对照)",
         lambda: compute_breakout_signal()["position"], "D"),
        ("组合", "gold_all_factor: balanced",
         lambda: compute_all_factor_signal(mode="balanced")["position"], "W"),
        ("组合", "gold_all_factor: aggressive",
         lambda: compute_all_factor_signal(mode="aggressive")["position"], "W"),
        ("组合", "gold_all_factor: conservative",
         lambda: compute_all_factor_signal(mode="conservative")["position"], "W"),
        ("组合", "gold_multi_factor: default",
         lambda: compute_multi_factor_signal()["position"], "W"),
        ("组合", "gold_combined: default",
         lambda: compute_combined_signal()["position"], "W"),
        ("组合", "gold_combined: use_timing=True",
         lambda: compute_combined_signal(use_timing=True)["position"], "W"),
    ]

    for category, name, fn, freq in sub_configs:
        try:
            pos = fn()
            r = evaluate(pos, freq)
            rows.append({
                "category": category, "name": name, "freq": freq,
                "oos_ann": r["OOS"].get("ann_ret"), "oos_sh": r["OOS"].get("sharpe"),
                "oos_dd": r["OOS"].get("max_dd"), "oos_cal": r["OOS"].get("calmar"),
                "is_ann": r["IS"].get("ann_ret"), "is_sh": r["IS"].get("sharpe"),
                "is_dd": r["IS"].get("max_dd"), "is_cal": r["IS"].get("calmar"),
                "avg_pos": r["avg_pos"], "pct_active": r["pct_active"],
            })
            print(f"  ✓ {name}")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print(f"  ✗ {name}: {e}")

    # ===== 输出全景表 =====
    df = pd.DataFrame(rows)
    print("\n" + "=" * 140)
    print("全景表 (双窗口 Calmar 均值排序)")
    print("=" * 140)
    df["avg_cal"] = (df["oos_cal"].fillna(-99) + df["is_cal"].fillna(-99)) / 2
    df_sorted = df.sort_values("avg_cal", ascending=False)

    print(f'{"类别":<6} {"策略":<40} {"F":<2} {"OOS年化":>8} {"OOSSh":>6} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISSh":>6} {"ISDD":>8} {"ISCal":>7} {"avgCal":>7} {"活跃%":>6}')
    print("-" * 140)
    for _, r in df_sorted.iterrows():
        print(f'{r["category"]:<6} {r["name"]:<40} {r["freq"]:<2} '
              f'{fmt(r["oos_ann"]):>8} {fmt(r["oos_sh"], pct=False):>6} '
              f'{fmt(r["oos_dd"]):>8} {fmt(r["oos_cal"], pct=False):>7} | '
              f'{fmt(r["is_ann"]):>8} {fmt(r["is_sh"], pct=False):>6} '
              f'{fmt(r["is_dd"]):>8} {fmt(r["is_cal"], pct=False):>7} '
              f'{fmt(r["avg_cal"], pct=False):>7} {fmt(r["pct_active"]):>6}')

    # ===== 2013 + 2026 关键年份诊断 =====
    print("\n" + "=" * 140)
    print("关键年份诊断 (2013宏观熊市 + 2026 YTD)")
    print("=" * 140)
    print(f'{"策略":<40} {"2013年化":>10} {"2013DD":>10} | {"2026YTD":>10} {"2026DD":>10}')
    print("-" * 140)
    key_diag_rows = []
    for _, r in df_sorted.iterrows():
        try:
            name = r["name"]
            freq = r["freq"]
            close = load_close_weekly() if freq == "W" else load_close_daily()
            # 通过name反查子策略 (重建调用)
            pos = dispatch_position(name, freq)
            if pos is None:
                continue
            pos_a = pos.reindex(close.index).fillna(method="ffill").fillna(0)
            m_2013 = calc_metrics_freq(pos_a, close, "2013-01-01", "2013-12-31", freq)
            m_2026 = calc_metrics_freq(pos_a, close, "2026-01-01", "2026-08-25", freq)
            print(f'{name:<40} {fmt(m_2013["ann_ret"]):>10} {fmt(m_2013["max_dd"]):>10} | '
                  f'{fmt(m_2026["ann_ret"]):>10} {fmt(m_2026["max_dd"]):>10}')
            key_diag_rows.append({"name": name, "y2013_ann": m_2013["ann_ret"],
                                  "y2013_dd": m_2013["max_dd"], "y2026_ann": m_2026["ann_ret"],
                                  "y2026_dd": m_2026["max_dd"]})
        except Exception as e:
            print(f'{r["name"]:<40} ERR: {e}')

    # ===== 输出到 markdown =====
    md = ["# 黄金策略库未串联子策略诊断结果\n",
          f"> 运行日期: 2026-08-27\n",
          f"> 测试窗口: OOS 2010-2019 / IS 2020-2026\n\n",
          "## 全景表 (按双窗口 Calmar 均值排序)\n\n",
          "| 类别 | 策略 | 频率 | OOS年化 | OOS Sh | OOS DD | OOS Calmar | IS年化 | IS Sh | IS DD | IS Calmar | avgCalmar | 活跃% |\n",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for _, r in df_sorted.iterrows():
        md.append(f"| {r['category']} | {r['name']} | {r['freq']} | "
                  f"{fmt(r['oos_ann'])} | {fmt(r['oos_sh'], pct=False)} | "
                  f"{fmt(r['oos_dd'])} | {fmt(r['oos_cal'])} | "
                  f"{fmt(r['is_ann'])} | {fmt(r['is_sh'], pct=False)} | "
                  f"{fmt(r['is_dd'])} | {fmt(r['is_cal'])} | "
                  f"{fmt(r['avg_cal'])} | {fmt(r['pct_active'])} |\n")

    md.append("\n## 错误记录\n\n")
    if errors:
        for e in errors:
            md.append(f"- {e}\n")
    else:
        md.append("(无)\n")

    md.append("\n## 串联候选排名\n\n")
    md.append("### 第一梯队 (avgCal > 0.8)\n\n")
    tier1 = df_sorted[(df_sorted["avg_cal"] > 0.8) & (df_sorted["category"] != "基线")]
    if len(tier1) > 0:
        for _, r in tier1.iterrows():
            md.append(f"- **{r['name']}** (avgCal={fmt(r['avg_cal'])})\n")
    else:
        md.append("(无)\n")

    md.append("\n### 第二梯队 (0.5 < avgCal < 0.8)\n\n")
    tier2 = df_sorted[(df_sorted["avg_cal"] > 0.5) & (df_sorted["avg_cal"] <= 0.8) & (df_sorted["category"] != "基线")]
    if len(tier2) > 0:
        for _, r in tier2.iterrows():
            md.append(f"- {r['name']} (avgCal={fmt(r['avg_cal'])})\n")
    else:
        md.append("(无)\n")

    md.append("\n### 第三梯队 (avgCal <= 0.5)\n\n")
    tier3 = df_sorted[(df_sorted["avg_cal"] <= 0.5) & (df_sorted["category"] != "基线")]
    if len(tier3) > 0:
        for _, r in tier3.iterrows():
            md.append(f"- {r['name']} (avgCal={fmt(r['avg_cal'])})\n")
    else:
        md.append("(无)\n")

    md_path = Path("docs/gold/substrategies_evaluation.md")
    md_path.write_text("".join(md))
    print(f"\n诊断结果已写入: {md_path}")

    if errors:
        print("\n=== 错误列表 ===")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
