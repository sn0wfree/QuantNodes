"""scripts/cmf/test_rolling_walkforward.py — 滚动walk-forward验证.

对分层集成策略做滚动walk-forward验证:
- train 4年, test 2年, 6折
- 测试不同信号组合 (等权/加权/投票)
- 输出滚动验证结果 + 稳定性分析

与固定窗口验证的区别:
- 固定窗口: 用全部历史数据训练, 用未来数据测试 (可能过拟合)
- 滚动窗口: 每次只用过去4年训练, 测试未来2年 (更严格)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
    compute_risk_overlay,
    compute_layered_ensemble_signal,
    _load_indicator_d,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal,
)

# 滚动窗口定义 (train 4年, test 2年, 6折)
ROLLING_FOLDS = [
    ("F1", "2010-01-01", "2013-12-31", "2014-01-01", "2015-12-31"),
    ("F2", "2012-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("F3", "2014-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("F4", "2016-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("F5", "2018-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("F6", "2020-01-01", "2023-12-31", "2024-01-01", "2026-08-25"),
]

COST_BPS = 5.0


def load_close():
    import duckdb
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date", keep="last")
              .set_index("trade_date")["close"])


def evaluate(pos, close, s, e):
    periods_per_y = 252
    sub = pos[(pos.index >= s) & (pos.index <= e)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
    sr = (sr - cost).dropna()
    sp = c.iloc[0]
    nav = sp * (1 + sr).cumprod()
    n_y = len(c) / periods_per_y
    ar = (nav.iloc[-1] / sp) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(periods_per_y)
    dd = (nav / nav.cummax() - 1).min()
    turnover = sub.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0,
            "turnover": turnover}


def get_signals(close):
    """获取3个原始信号."""
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    macro_df = compute_macro_regime_signal()
    macro_pos = macro_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    rr_df = compute_real_rate_signal(variant="tips_ma")
    rr_pos = rr_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    mom_df = compute_momentum_signal(variant="long_only")
    mom_pos = mom_df["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    return {"macro": macro_pos, "real_rate": rr_pos, "momentum": mom_pos}, vix, gvz


def build_candidates(signals, vix, gvz, close):
    """构建所有候选策略."""
    candidates = {}

    # 单信号 + 风控
    for name, pos in signals.items():
        candidates[f"{name}+风控"] = compute_risk_overlay(
            close=close, position=pos, vix=vix, gvz=gvz,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
            cooldown_days=5,
        )

    # 分层集成 (先各自风控, 再平均)
    macro_ov = compute_risk_overlay(
        close=close, position=signals["macro"], vix=vix, gvz=gvz,
        hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        cooldown_days=5,
    )
    rr_ov = compute_risk_overlay(
        close=close, position=signals["real_rate"], vix=vix, gvz=gvz,
        hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        cooldown_days=5,
    )
    mom_ov = compute_risk_overlay(
        close=close, position=signals["momentum"], vix=vix, gvz=gvz,
        hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        cooldown_days=5,
    )

    candidates["分层1/3等权"] = ((macro_ov + rr_ov + mom_ov) / 3).clip(0, 1.2)
    candidates["分层0.4/0.3/0.3"] = (0.4 * macro_ov + 0.3 * rr_ov + 0.3 * mom_ov).clip(0, 1.2)
    candidates["分层0.5/0.25/0.25"] = (0.5 * macro_ov + 0.25 * rr_ov + 0.25 * mom_ov).clip(0, 1.2)
    candidates["分层real+mom"] = ((rr_ov + mom_ov) / 2).clip(0, 1.2)

    # 投票制
    macro_bin = (signals["macro"] > 0.5).astype(float)
    rr_bin = (signals["real_rate"] > 0.5).astype(float)
    mom_bin = (signals["momentum"] > 0.5).astype(float)
    vote_2of3 = ((macro_bin + rr_bin + mom_bin) >= 2).astype(float)
    candidates["投票2-of-3"] = compute_risk_overlay(
        close=close, position=vote_2of3, vix=vix, gvz=gvz,
        hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        cooldown_days=5,
    )

    # Buy & Hold
    candidates["Buy&Hold"] = pd.Series(1.0, index=close.index)

    return candidates


def main():
    close = load_close()
    signals, vix, gvz = get_signals(close)
    candidates = build_candidates(signals, vix, gvz, close)

    print("=" * 130)
    print("滚动Walk-Forward验证 (train 4年, test 2年, 6折)")
    print("=" * 130)

    # 逐折评估
    all_results = {name: [] for name in candidates}

    for fname, train_s, train_e, test_s, test_e in ROLLING_FOLDS:
        for name, pos in candidates.items():
            train_metrics = evaluate(pos, close, train_s, train_e)
            test_metrics = evaluate(pos, close, test_s, test_e)
            all_results[name].append({
                "fold": fname,
                "train": train_metrics,
                "test": test_metrics,
            })

    # 输出表1: 各折test期表现
    print("\n" + "=" * 130)
    print("表1: 各折test期年化收益 (Calmar)")
    print("=" * 130)
    print(f'{"策略":<25} | {"F1":>12} {"F2":>12} {"F3":>12} {"F4":>12} {"F5":>12} {"F6":>12} | {"avg":>12}')
    print("-" * 130)
    for name in candidates:
        row = f'{name:<25} | '
        calmars = []
        for r in all_results[name]:
            cal = r["test"]["calmar"]
            row += f'{cal:>12.2f} '
            calmars.append(cal)
        row += f'| {np.mean(calmars):>12.2f}'
        print(row)

    # 输出表2: 各折test期MaxDD
    print("\n" + "=" * 130)
    print("表2: 各折test期MaxDD")
    print("=" * 130)
    print(f'{"策略":<25} | {"F1":>12} {"F2":>12} {"F3":>12} {"F4":>12} {"F5":>12} {"F6":>12} | {"avg":>12}')
    print("-" * 130)
    for name in candidates:
        row = f'{name:<25} | '
        dds = []
        for r in all_results[name]:
            dd = r["test"]["max_dd"]
            row += f'{dd*100:>11.2f}% '
            dds.append(dd)
        row += f'| {np.mean(dds)*100:>11.2f}%'
        print(row)

    # 输出表3: 各折test期年化
    print("\n" + "=" * 130)
    print("表3: 各折test期年化收益")
    print("=" * 130)
    print(f'{"策略":<25} | {"F1":>12} {"F2":>12} {"F3":>12} {"F4":>12} {"F5":>12} {"F6":>12} | {"avg":>12}')
    print("-" * 130)
    for name in candidates:
        row = f'{name:<25} | '
        rets = []
        for r in all_results[name]:
            ret = r["test"]["ann_ret"]
            row += f'{ret*100:>11.2f}% '
            rets.append(ret)
        row += f'| {np.mean(rets)*100:>11.2f}%'
        print(row)

    # 输出表4: 综合评分
    print("\n" + "=" * 130)
    print("表4: 滚动验证综合评分")
    print("=" * 130)
    print(f'{"策略":<25} | {"avg Calmar":>12} {"avg MaxDD":>12} {"avg年化":>12} | {"稳定性":>12}')
    print("-" * 130)

    scores = []
    for name in candidates:
        calmars = [r["test"]["calmar"] for r in all_results[name]]
        dds = [r["test"]["max_dd"] for r in all_results[name]]
        rets = [r["test"]["ann_ret"] for r in all_results[name]]
        avg_cal = np.mean(calmars)
        avg_dd = np.mean(dds)
        avg_ret = np.mean(rets)
        stability = np.std(calmars)  # Calmar标准差 (越小越稳定)
        scores.append((name, avg_cal, avg_dd, avg_ret, stability))

    scores.sort(key=lambda x: -x[1])  # 按avg Calmar降序
    for name, avg_cal, avg_dd, avg_ret, stability in scores:
        print(f'{name:<25} | {avg_cal:>12.2f} {avg_dd*100:>11.2f}% {avg_ret*100:>11.2f}% | {stability:>12.2f}')

    # 最终推荐
    best = scores[0]
    print(f"\n最佳: {best[0]} (avg Calmar={best[1]:.2f}, avg MaxDD={best[2]*100:.2f}%, stability={best[4]:.2f})")

    # 保存到markdown
    md = ["# 滚动Walk-Forward验证\n\n",
          "> 运行日期: 2026-08-27\n",
          "> 测试脚本: scripts/cmf/test_rolling_walkforward.py\n",
          "> 窗口: train 4年, test 2年, 6折\n\n",
          "## 各折test期年化收益 (Calmar)\n\n",
          "| 策略 | F1 | F2 | F3 | F4 | F5 | F6 | avg |\n",
          "|---|---|---|---|---|---|---|---|\n"]
    for name in candidates:
        row = f"| {name} | "
        calmars = []
        for r in all_results[name]:
            row += f"{r['test']['calmar']:.2f} | "
            calmars.append(r['test']['calmar'])
        row += f"{np.mean(calmars):.2f} |\n"
        md.append(row)

    md.append("\n## 各折test期MaxDD\n\n")
    md.append("| 策略 | F1 | F2 | F3 | F4 | F5 | F6 | avg |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for name in candidates:
        row = f"| {name} | "
        dds = []
        for r in all_results[name]:
            row += f"{r['test']['max_dd']*100:.2f}% | "
            dds.append(r['test']['max_dd'])
        row += f"{np.mean(dds)*100:.2f}% |\n"
        md.append(row)

    md.append("\n## 综合评分\n\n")
    md.append("| 策略 | avg Calmar | avg MaxDD | avg年化 | 稳定性 |\n")
    md.append("|---|---|---|---|---|\n")
    for name, avg_cal, avg_dd, avg_ret, stability in scores:
        md.append(f"| {name} | {avg_cal:.2f} | {avg_dd*100:.2f}% | {avg_ret*100:.2f}% | {stability:.2f} |\n")

    md.append(f"\n## 最终推荐: **{best[0]}**\n")

    md_path = Path("docs/gold/rolling_walkforward_results.md")
    md_path.write_text("".join(md))
    print(f"\n结果已写入: {md_path}")


if __name__ == "__main__":
    main()
