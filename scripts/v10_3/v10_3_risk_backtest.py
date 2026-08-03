"""v10.3 Risk Warning — 历史回测: 风险预警的全周期表现分析.

展示:
  1. 预警信号 vs 实际高波动率的时间序列
  2. 混淆矩阵 (正报/误报/漏报/正拒)
  3. 关键事件分析 (俄乌冲突/2024暴跌等)
  4. 滚动 AUC/ECE/信号占比
  5. 如果按预警信号调整防御仓位, 历史表现如何
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v10_3.risk_warning import (
    EnsembleRiskWarning, load_etf_data, CONFIGS,
)
from sklearn.metrics import roc_auc_score, brier_score_loss

log = print
SEP = "=" * 70
DASH = "-" * 70


def fmt_pct(v):
    return f"{v:.1%}"


def load_rw(asset: str):
    etf = load_etf_data()
    rw = EnsembleRiskWarning(etf, asset)
    rw.evaluate()
    return rw


def compute_confusion(prob: pd.Series, y_true: pd.Series, threshold: float = 0.8):
    """Compute confusion matrix for a given probability threshold."""
    common = prob.dropna().index.intersection(y_true.dropna().index)
    p = (prob.loc[common] >= threshold).astype(int)
    y = y_true.loc[common].astype(int)
    tp = ((p == 1) & (y == 1)).sum()
    fp = ((p == 1) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum()
    tn = ((p == 0) & (y == 0)).sum()
    return {
        "threshold": threshold,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "false_alarm_rate": fp / max(fp + tn, 1),
    }


def key_events():
    """Known high-volatility events in the OOS period."""
    return [
        ("2022-02-24", "2022-03-15", "俄乌冲突"),
        ("2022-10-10", "2022-10-31", "二十大前后"),
        ("2023-07-24", "2023-08-25", "政策底博弈"),
        ("2024-01-17", "2024-02-05", "A股熔断式下跌"),
        ("2024-09-24", "2024-10-18", "924行情"),
        ("2025-04-03", "2025-04-11", "关税冲击"),
    ]


def backtest(rw: EnsembleRiskWarning, asset: str):
    log(f"\n{SEP}")
    log(f"  风险预警历史回测: {asset.upper()}")
    log(SEP)

    prob = rw._ensemble_prob
    yt = rw.y_true.reindex(prob.index).dropna()
    common = yt.index.intersection(prob.dropna().index)
    y = yt.loc[common]
    p = prob.loc[common]

    # ── 1. Overall metrics ──
    log(f"\n[1] 全周期性能")
    log(DASH)
    log(f"  评估期: {common[0].date()} ~ {common[-1].date()}  ({len(common)} 个交易日)")
    log(f"  高波动率天数: {int(y.sum())} ({fmt_pct(y.mean())})")
    log(f"  AUC: {roc_auc_score(y, p):.4f}")
    log(f"  Brier: {brier_score_loss(y, p):.4f}")
    log(f"  平均预测概率: {p.mean():.1%}")

    # ── 2. Confusion matrix at different thresholds ──
    log(f"\n[2] 混淆矩阵 (不同阈值)")
    log(DASH)
    log(f"{'阈值':<8} {'正报(TP)':<10} {'误报(FP)':<10} {'漏报(FN)':<10} {'正拒(TN)':<10} "
        f"{'精确率':<8} {'召回率':<8} {'F1':<8}")
    log(DASH)
    for th in [0.5, 0.6, 0.7, 0.8, 0.9]:
        cm = compute_confusion(p, y, th)
        log(f"{th:<8.1f} {cm['TP']:<10} {cm['FP']:<10} {cm['FN']:<10} {cm['TN']:<10} "
            f"{cm['precision']:<8.1%} {cm['recall']:<8.1%} {cm['f1']:<8.3f}")

    # ── 3. Key events ──
    log(f"\n[3] 关键事件分析")
    log(DASH)
    for start, end, name in key_events():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        slice_p = p.loc[s:e]
        slice_y = y.loc[s:e]
        if len(slice_p) == 0:
            continue
        avg_prob = slice_p.mean()
        high_sig = (slice_p >= 0.8).sum()
        actual_high = int(slice_y.sum())
        correct = ((slice_p >= 0.8) & (slice_y == 1)).sum()
        log(f"  {name:<16} ({start}~{end})")
        log(f"    平均预警概率: {avg_prob:.1%}  "
            f"高风险信号: {high_sig}/{len(slice_p)}d  "
            f"实际高波动: {actual_high}d  "
            f"命中: {correct}d")

    # ── 4. Signal distribution by year ──
    log(f"\n[4] 年度信号分布")
    log(DASH)
    df = pd.DataFrame({"prob": p, "y_true": y})
    df["year"] = df.index.year
    log(f"{'年份':<6} {'天数':<6} {'高波动率':<10} {'平均概率':<10} {'信号>=0.8':<10} {'正报率':<8}")
    log(DASH)
    for yr, grp in df.groupby("year"):
        high_sig = (grp["prob"] >= 0.8).sum()
        actual = int(grp["y_true"].sum())
        correct = ((grp["prob"] >= 0.8) & (grp["y_true"] == 1)).sum()
        log(f"{yr:<6} {len(grp):<6} {fmt_pct(grp['y_true'].mean()):<10} "
            f"{grp['prob'].mean():<10.1%} {high_sig:<10} "
            f"{correct / max(high_sig, 1):<8.1%}")

    # ── 5. Monthly average probability heatmap ──
    log(f"\n[5] 月度平均预警概率 (热力图)")
    log(DASH)
    df["month"] = df.index.month
    monthly = df.groupby(["year", "month"])["prob"].mean().unstack()
    log(f"      ", end="")
    for m in range(1, 13):
        log(f"{m:>5d}  ", end="")
    log()
    for yr in sorted(monthly.index):
        log(f"  {yr}: ", end="")
        for m in range(1, 13):
            v = monthly.loc[yr, m] if m in monthly.columns else 0.5
            bar = "░" if v < 0.5 else "▒" if v < 0.7 else "▓" if v < 0.85 else "█"
            log(f"  {bar} {v:.0%}", end="")
        log()

    # ── 6. Rolling metrics ──
    log(f"\n[6] 滚动 60d 性能")
    log(DASH)
    window = 60
    dates = p.index
    roll_dates, roll_auc, roll_brier, roll_ece = [], [], [], []
    for i in range(window, len(dates)):
        s = dates[i - window]
        e = dates[i]
        ps = p.loc[s:e]
        ys = y.loc[s:e]
        if ys.nunique() < 2:
            continue
        roll_dates.append(e)
        roll_auc.append(roc_auc_score(ys, ps))
        roll_brier.append(brier_score_loss(ys, ps))
        bins = np.linspace(0, 1, 11)
        ids = np.digitize(ps, bins) - 1
        ece_val = 0.0
        for b in range(10):
            m = (ids == b)
            if m.sum() > 0:
                ece_val += abs(ps.iloc[m].mean() - ys.iloc[m].mean()) * m.sum()
        roll_ece.append(ece_val / len(ps))

    log(f"  {'Stat':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Latest':<8}")
    log(DASH)
    for label, vals in [("Rolling AUC", roll_auc), ("Rolling Brier", roll_brier),
                        ("Rolling ECE", roll_ece)]:
        if vals:
            log(f"  {label:<12} {np.mean(vals):<8.3f} {np.std(vals):<8.3f} "
                f"{np.min(vals):<8.3f} {np.max(vals):<8.3f} {vals[-1]:<8.3f}")

    # ── 7. Best/worst periods ──
    log(f"\n[7] 最佳/最差表现期 (滚动 60d AUC)")
    log(DASH)
    roll_df = pd.DataFrame({"auc": roll_auc, "brier": roll_brier, "ece": roll_ece},
                           index=roll_dates)
    best = roll_df.nlargest(3, "auc")
    worst = roll_df.nsmallest(3, "auc")
    log(f"  最佳 3 期:")
    for idx, row in best.iterrows():
        log(f"    {idx.date()}: AUC={row['auc']:.3f} Brier={row['brier']:.3f} ECE={row['ece']:.3f}")
    log(f"  最差 3 期:")
    for idx, row in worst.iterrows():
        log(f"    {idx.date()}: AUC={row['auc']:.3f} Brier={row['brier']:.3f} ECE={row['ece']:.3f}")

    # ── 8. Layer comparison ──
    log(f"\n[8] 各层模型对比")
    log(DASH)
    log(f"{'Layer':<15} {'AUC':<8} {'Brier':<8} {'ECE':<8} {'正报率(th=0.8)':<15}")
    log(DASH)
    for key in ["climatology", "harrv", "ewma", "current_vol", "macro"]:
        m = rw._metrics.get(key, {})
        if not m:
            continue
        pk = rw._prob[key]
        cm = compute_confusion(pk, y, 0.8)
        log(f"{key:<15} {m['auc']:<8.4f} {m['brier']:<8.4f} "
            f"{m['ece']:<8.4f} {cm['precision']:<15.1%}")
    # Ensemble (stored separately)
    em = rw._metrics.get("ensemble", {})
    if em:
        cm = compute_confusion(p, y, 0.8)
        log(f"{'ensemble':<15} {em['auc']:<8.4f} {em['brier']:<8.4f} "
            f"{em['ece']:<8.4f} {cm['precision']:<15.1%}")

    # ── 9. Summary ──
    log(f"\n[9] 总结")
    log(DASH)
    # Best threshold
    best_f1 = 0
    best_th = 0.5
    for th in np.arange(0.3, 0.95, 0.05):
        cm = compute_confusion(p, y, th)
        if cm["f1"] > best_f1:
            best_f1 = cm["f1"]
            best_th = th
    log(f"  最优阈值 (F1): {best_th:.2f} (F1={best_f1:.3f})")
    cm = compute_confusion(p, y, best_th)
    log(f"  此时: 正报={cm['TP']} 误报={cm['FP']} 漏报={cm['FN']} 正拒={cm['TN']}")
    log(f"  Climatology 差距: {rw._metrics['ensemble']['auc'] - rw._metrics['climatology']['auc']:+.4f}")
    log(f"  {'Climatology 优于集成' if rw._metrics['climatology']['auc'] > rw._metrics['ensemble']['auc'] else '集成优于 Climatology'}")

    log(SEP)
    return {
        "asset": asset,
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "best_threshold": best_th,
        "best_f1": best_f1,
        "n_days": len(common),
        "pos_rate": y.mean(),
    }


def main():
    log(SEP)
    log("  风险预警模型 — 历史回测报告")
    log(SEP)

    results = []
    for asset in ["equity", "commodity", "gold"]:
        rw = load_rw(asset)
        res = backtest(rw, asset)
        results.append(res)

    # Final comparison
    log(f"\n{SEP}")
    log(f"  三资产对比")
    log(SEP)
    log(f"{'Asset':<12} {'AUC':<8} {'Brier':<8} {'最优阈值':<10} {'F1':<8} {'天数':<8} {'正报率':<8}")
    log(DASH)
    for r in results:
        log(f"{r['asset']:<12} {r['auc']:<8.4f} {r['brier']:<8.4f} "
            f"{r['best_threshold']:<10.2f} {r['best_f1']:<8.3f} "
            f"{r['n_days']:<8} {r['pos_rate']:<8.1%}")

    log(f"\n[Done]")


if __name__ == "__main__":
    main()