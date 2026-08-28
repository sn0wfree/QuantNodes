"""scripts/cmf/test_walk_forward_ensemble.py — 分层集成策略滚动Walk-Forward测试.

滚动窗口:
- 训练窗口: 4年
- 测试窗口: 2年
- 滚动步长: 1年

评估:
- 每个测试窗口的年化收益/MaxDD/Calmar
- 拼接所有测试窗口的累计表现
- 与Buy&Hold对比
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
from QuantNodes.strategy.commodity_fundamental.gold import compute_layered_ensemble_signal

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
    """单窗口评估."""
    sub = pos[(pos.index >= s) & (pos.index <= e)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "calmar": 0, "turnover": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
    sr = (sr - cost).dropna()
    sp = c.iloc[0]
    nav = sp * (1 + sr).cumprod()
    n_y = len(c) / 252
    ar = (nav.iloc[-1] / sp) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    turnover = sub.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0,
            "turnover": turnover}


def walk_forward_test(pos, close, train_years=4, test_years=2):
    """滚动Walk-Forward测试."""
    start_year = close.index[0].year
    end_year = close.index[-1].year
    
    folds = []
    current_year = start_year + train_years
    
    while current_year <= end_year:
        train_start = f"{current_year - train_years}-01-01"
        train_end = f"{current_year - 1}-12-31"
        test_start = f"{current_year}-01-01"
        # 最后一折到数据末尾
        if current_year == end_year:
            test_end = close.index[-1].strftime("%Y-%m-%d")
        else:
            test_end = f"{current_year}-12-31"
        
        folds.append({
            "train": (train_start, train_end),
            "test": (test_start, test_end),
            "fold": len(folds) + 1,
        })
        current_year += 1
    
    results = []
    for fold in folds:
        train_metrics = evaluate(pos, close, *fold["train"])
        test_metrics = evaluate(pos, close, *fold["test"])
        
        results.append({
            "fold": fold["fold"],
            "train_period": f"{fold['train'][0][:4]}-{fold['train'][1][:4]}",
            "test_period": f"{fold['test'][0][:4]}-{fold['test'][1][:4]}",
            "test_start": fold["test"][0],
            "test_end": fold["test"][1],
            "train_ann": train_metrics["ann_ret"],
            "train_dd": train_metrics["max_dd"],
            "test_ann": test_metrics["ann_ret"],
            "test_dd": test_metrics["max_dd"],
            "test_sharpe": test_metrics["sharpe"],
            "test_calmar": test_metrics["calmar"],
        })
    
    return results


def main():
    close = load_close()
    
    sig = compute_layered_ensemble_signal()
    pos = sig["position"]
    
    bh_pos = pd.Series(1.0, index=close.index)
    
    print("=" * 130)
    print("分层集成策略 滚动Walk-Forward测试 (每年滚动)")
    print("=" * 130)
    print(f"训练窗口: 4年, 测试窗口: 1年, 滚动步长: 1年")
    print(f"数据范围: {close.index[0].date()} ~ {close.index[-1].date()}")
    print()
    
    ensemble_results = walk_forward_test(pos, close)
    bh_results = walk_forward_test(bh_pos, close)
    
    # 逐折结果
    print("=" * 130)
    print("逐折Walk-Forward结果")
    print("=" * 130)
    print(f"{'折':<4} {'训练期':<12} {'测试期':<12} | "
          f"{'策略年化':>10} {'策略DD':>10} {'策略Sharpe':>10} {'策略Calmar':>10} | "
          f"{'B&H年化':>10} {'B&H DD':>10} {'B&H Calmar':>10}")
    print("-" * 130)
    
    for i, (er, bhr) in enumerate(zip(ensemble_results, bh_results)):
        print(f"{er['fold']:<4} {er['train_period']:<12} {er['test_period']:<12} | "
              f"{er['test_ann']*100:>9.2f}% {er['test_dd']*100:>9.2f}% {er['test_sharpe']:>10.2f} {er['test_calmar']:>10.2f} | "
              f"{bhr['test_ann']*100:>9.2f}% {bhr['test_dd']*100:>9.2f}% {bhr['test_calmar']:>10.2f}")
    
    # 汇总统计
    print("\n" + "=" * 130)
    print("Walk-Forward汇总统计")
    print("=" * 130)
    
    ens_ann = [r["test_ann"] for r in ensemble_results]
    ens_dd = [r["test_dd"] for r in ensemble_results]
    ens_sharpe = [r["test_sharpe"] for r in ensemble_results]
    ens_calmar = [r["test_calmar"] for r in ensemble_results]
    
    bh_ann = [r["test_ann"] for r in bh_results]
    bh_dd = [r["test_dd"] for r in bh_results]
    bh_calmar = [r["test_calmar"] for r in bh_results]
    
    print(f"\n{'指标':<20} {'分层集成':>15} {'Buy&Hold':>15} {'差异':>15}")
    print("-" * 80)
    print(f"{'平均年化收益':<20} {np.mean(ens_ann)*100:>14.2f}% {np.mean(bh_ann)*100:>14.2f}% {(np.mean(ens_ann)-np.mean(bh_ann))*100:>+14.2f}pp")
    print(f"{'平均MaxDD':<20} {np.mean(ens_dd)*100:>14.2f}% {np.mean(bh_dd)*100:>14.2f}% {(np.mean(ens_dd)-np.mean(bh_dd))*100:>+14.2f}pp")
    print(f"{'最差MaxDD':<20} {np.min(ens_dd)*100:>14.2f}% {np.min(bh_dd)*100:>14.2f}% {(np.min(ens_dd)-np.min(bh_dd))*100:>+14.2f}pp")
    print(f"{'平均Calmar':<20} {np.mean(ens_calmar):>15.2f} {np.mean(bh_calmar):>15.2f} {np.mean(ens_calmar)-np.mean(bh_calmar):>+15.2f}")
    print(f"{'正收益期数':<20} {sum(1 for x in ens_ann if x > 0):>15} {sum(1 for x in bh_ann if x > 0):>15} {sum(1 for x in ens_ann if x > 0)-sum(1 for x in bh_ann if x > 0):>+15}")
    print(f"{'总测试期数':<20} {len(ensemble_results):>15} {len(bh_results):>15} {0:>15}")
    
    # 累计年化收益 (CAGR)
    print("\n" + "=" * 130)
    print("累计年化收益 (CAGR)")
    print("=" * 130)
    
    # 拼接所有测试期收益
    all_test_returns = []
    bh_all_returns = []
    
    for er, bhr in zip(ensemble_results, bh_results):
        test_start = er["test_start"]
        test_end = er["test_end"]
        
        # 策略
        sub = pos[(pos.index >= test_start) & (pos.index <= test_end)]
        c = close.reindex(sub.index)
        ret = c.pct_change()
        sr = (sub.fillna(0).shift(1) * ret).dropna()
        cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
        sr = (sr - cost).dropna()
        all_test_returns.append(sr)
        
        # Buy&Hold
        sub_bh = bh_pos[(bh_pos.index >= test_start) & (bh_pos.index <= test_end)]
        sr_bh = (sub_bh.shift(1) * c.pct_change()).dropna()
        bh_all_returns.append(sr_bh)
    
    # 合并所有测试期收益
    combined_returns = pd.concat(all_test_returns)
    combined_returns = combined_returns[~combined_returns.index.duplicated(keep='first')]
    combined_returns = combined_returns.sort_index()
    
    combined_bh_returns = pd.concat(bh_all_returns)
    combined_bh_returns = combined_bh_returns[~combined_bh_returns.index.duplicated(keep='first')]
    combined_bh_returns = combined_bh_returns.sort_index()
    
    # 计算累计表现
    nav = (1 + combined_returns).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_y = len(combined_returns) / 252
    ann_ret = (1 + total_ret) ** (1 / n_y) - 1
    vol = combined_returns.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    
    # Buy&Hold累计表现
    nav_bh = (1 + combined_bh_returns).cumprod()
    total_ret_bh = nav_bh.iloc[-1] / nav_bh.iloc[0] - 1
    n_y_bh = len(combined_bh_returns) / 252
    ann_ret_bh = (1 + total_ret_bh) ** (1 / n_y_bh) - 1
    vol_bh = combined_bh_returns.std() * np.sqrt(252)
    dd_bh = (nav_bh / nav_bh.cummax() - 1).min()
    sharpe_bh = ann_ret_bh / vol_bh if vol_bh > 0 else 0
    calmar_bh = ann_ret_bh / abs(dd_bh) if dd_bh != 0 else 0
    
    print(f"\n{'指标':<25} {'分层集成':>15} {'Buy&Hold':>15} {'差异':>15}")
    print("-" * 80)
    print(f"{'累计年化收益 (CAGR)':<25} {ann_ret*100:>14.2f}% {ann_ret_bh*100:>14.2f}% {(ann_ret-ann_ret_bh)*100:>+14.2f}pp")
    print(f"{'累计Sharpe':<25} {sharpe:>15.2f} {sharpe_bh:>15.2f} {sharpe-sharpe_bh:>+15.2f}")
    print(f"{'累计MaxDD':<25} {dd*100:>14.2f}% {dd_bh*100:>14.2f}% {(dd-dd_bh)*100:>+14.2f}pp")
    print(f"{'累计Calmar':<25} {calmar:>15.2f} {calmar_bh:>15.2f} {calmar-calmar_bh:>+15.2f}")
    print(f"{'累计换手率':<25} {combined_returns.abs().sum()/2/n_y:>14.1f}x {combined_bh_returns.abs().sum()/2/n_y_bh:>14.1f}x")
    print(f"{'测试期覆盖':<25} {combined_returns.index[0].date()} ~ {combined_returns.index[-1].date()}")
    print(f"{'总天数':<25} {len(combined_returns):>15} {len(combined_bh_returns):>15}")
    
    # 保存结果
    md = ["# 分层集成策略 滚动Walk-Forward测试\n\n",
          f"> 运行日期: 2026-08-27\n",
          f"> 测试脚本: scripts/cmf/test_walk_forward_ensemble.py\n\n",
          "## 测试设置\n\n",
          f"- 训练窗口: 4年\n",
          f"- 测试窗口: 2年\n",
          f"- 滚动步长: 1年\n",
          f"- 数据范围: {close.index[0].date()} ~ {close.index[-1].date()}\n\n",
          "## 逐折结果\n\n",
          "| 折 | 训练期 | 测试期 | 策略年化 | 策略DD | 策略Sharpe | 策略Calmar | B&H年化 | B&H DD | B&H Calmar |\n",
          "|---|---|---|---|---|---|---|---|---|---|\n"]
    
    for er, bhr in zip(ensemble_results, bh_results):
        md.append(f"| {er['fold']} | {er['train_period']} | {er['test_period']} | "
                  f"{er['test_ann']*100:.2f}% | {er['test_dd']*100:.2f}% | {er['test_sharpe']:.2f} | {er['test_calmar']:.2f} | "
                  f"{bhr['test_ann']*100:.2f}% | {bhr['test_dd']*100:.2f}% | {bhr['test_calmar']:.2f} |\n")
    
    md.append("\n## 汇总统计\n\n")
    md.append("| 指标 | 分层集成 | Buy&Hold | 差异 |\n")
    md.append("|---|---|---|---|\n")
    md.append(f"| 平均年化收益 | {np.mean(ens_ann)*100:.2f}% | {np.mean(bh_ann)*100:.2f}% | {(np.mean(ens_ann)-np.mean(bh_ann))*100:+.2f}pp |\n")
    md.append(f"| 平均MaxDD | {np.mean(ens_dd)*100:.2f}% | {np.mean(bh_dd)*100:.2f}% | {(np.mean(ens_dd)-np.mean(bh_dd))*100:+.2f}pp |\n")
    md.append(f"| 最差MaxDD | {np.min(ens_dd)*100:.2f}% | {np.min(bh_dd)*100:.2f}% | {(np.min(ens_dd)-np.min(bh_dd))*100:+.2f}pp |\n")
    md.append(f"| 平均Calmar | {np.mean(ens_calmar):.2f} | {np.mean(bh_calmar):.2f} | {np.mean(ens_calmar)-np.mean(bh_calmar):+.2f} |\n")
    md.append(f"| 正收益期数 | {sum(1 for x in ens_ann if x > 0)} | {sum(1 for x in bh_ann if x > 0)} | {sum(1 for x in ens_ann if x > 0)-sum(1 for x in bh_ann if x > 0):+} |\n")
    
    md.append("\n## 累计年化收益 (CAGR)\n\n")
    md.append("| 指标 | 分层集成 | Buy&Hold | 差异 |\n")
    md.append("|---|---|---|---|\n")
    md.append(f"| 累计年化收益 (CAGR) | {ann_ret*100:.2f}% | {ann_ret_bh*100:.2f}% | {(ann_ret-ann_ret_bh)*100:+.2f}pp |\n")
    md.append(f"| 累计Sharpe | {sharpe:.2f} | {sharpe_bh:.2f} | {sharpe-sharpe_bh:+.2f} |\n")
    md.append(f"| 累计MaxDD | {dd*100:.2f}% | {dd_bh*100:.2f}% | {(dd-dd_bh)*100:+.2f}pp |\n")
    md.append(f"| 累计Calmar | {calmar:.2f} | {calmar_bh:.2f} | {calmar-calmar_bh:+.2f} |\n")
    md.append(f"| 测试期覆盖 | {combined_returns.index[0].date()} ~ {combined_returns.index[-1].date()} |\n")
    
    md_path = Path("docs/gold/walk_forward_ensemble_results.md")
    md_path.write_text("".join(md))
    print(f"\n结果已写入: {md_path}")


if __name__ == "__main__":
    main()
