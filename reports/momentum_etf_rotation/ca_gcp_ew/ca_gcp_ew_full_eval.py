"""CA-GCP 预警系统 — 完整有效性评估

包含:
  - Walk-Forward 评估 (4 折)
  - Bootstrap 置信区间 (1000 次)
  - ROC/AUC 曲线
  - 模型校准深度 (worst-10, width-vol 相关性)
  - 场景压力测试 (牛/熊/震荡/多事件)
  - 缩仓规则网格搜索 (5×4)
  - 跨池稳健性 (3 个 ETF 池)
  - 综合评分卡 (3 种投资者类型)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "momentum_etf_rotation" / "ca_gcp_ew"))  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ca_gcp_ew_eval import (  # noqa: E402
    CALIB_WINDOW,
    DATA_PATH,
    KNOWN_EVENTS,
    OUT_DIR,
    PRED_STEP,
    TRAIN_WINDOW,
    backtest_overlay,
    build_alerts,
    build_alerts_confidence,
    build_alerts_v3,
    build_alerts_v4,
    compute_trend_signal,
    evaluate_event_hits,
    evaluate_precision_recall,
    get_asset_sectors,
    load_returns,
    rolling_predict,
)

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
    compute_coverage_metrics,
    estimate_volatility,
    width_stability,
    width_timeseries,
)

EVAL_DIR = OUT_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

WF_FOLDS = [
    ("2022-01-01", "2022-12-31", "2022 熊市"),
    ("2023-01-01", "2023-12-31", "2023 震荡"),
    ("2024-01-01", "2024-12-31", "2024 多事件"),
    ("2025-01-01", "2026-05-06", "2025+ 近况"),
]

SCENARIOS = [
    ("2020-11-20", "2021-12-31", "牛市 (轮动剧烈)"),
    ("2022-01-01", "2022-12-31", "熊市 (系统性下跌)"),
    ("2023-01-01", "2023-12-31", "震荡 (反弹+下跌)"),
    ("2024-01-01", "2026-05-06", "多事件 (频繁触发)"),
]

YELLOW_SCALES = [0.95, 0.85, 0.70, 0.55, 0.40]
RED_SCALES = [0.80, 0.60, 0.40, 0.20]

N_BOOT = 1000
BOOT_SEED = 42


def fit_predict_window(returns: pd.DataFrame,
                       train_end: pd.Timestamp,
                       test_start: pd.Timestamp,
                       test_end: pd.Timestamp) -> tuple:
    """单窗口拟合+预测"""
    train_end_idx = returns.index.searchsorted(train_end, side="right")
    test_start_idx = returns.index.searchsorted(test_start, side="left")
    test_end_idx = returns.index.searchsorted(test_end, side="right")
    train_returns = returns.iloc[:train_end_idx]
    test_returns = returns.iloc[test_start_idx:test_end_idx]
    if len(train_returns) < 400 or len(test_returns) < 10:
        return None, None, None, None, None

    config = CAGCPConfig(
        k=6, sensitivity_eta=0.5, recency_tau=20.0, alpha=0.05,
    )
    pipe = CAGCPipeline(config)
    pipe.fit(train_returns)
    calib_returns = train_returns.iloc[-CALIB_WINDOW:]
    intervals = pipe.predict_fast(calib_returns, test_returns)
    return (
        intervals["half_width"], intervals["stress"],
        intervals["lower"], intervals["upper"], test_returns,
    )


def walk_forward_evaluate(returns: pd.DataFrame) -> pd.DataFrame:
    """Walk-Forward 评估

    每折用历史数据拟合，在该折区间上评估
    """
    print()
    print("=" * 70)
    print("Walk-Forward 评估 (4 折)")
    print("=" * 70)

    rows = []
    for test_start_str, test_end_str, fold_name in WF_FOLDS:
        train_end = pd.Timestamp(test_start_str) - pd.Timedelta(days=1)
        test_start = pd.Timestamp(test_start_str)
        test_end = pd.Timestamp(test_end_str)

        hw, stress, lower, upper, test_ret = fit_predict_window(
            returns, train_end, test_start, test_end
        )
        if hw is None or len(hw) == 0:
            print(f"  [{fold_name}] 跳过 (数据不足)")
            continue

        trend_ok = compute_trend_signal(returns).reindex(hw.index).fillna(False)
        alerts_v3 = build_alerts_v3(hw, stress, returns, trend_ok=trend_ok)
        alerts_v4 = build_alerts_v4(hw, stress, returns, trend_ok=trend_ok)

        pr = evaluate_precision_recall(alerts_v3, returns, label=fold_name,
                                       fired_col="trigger_v3")
        overlay = backtest_overlay(alerts_v3, returns)

        n_fired = int((alerts_v3["trigger_v3"] == "and_fired").sum())
        eval_hits = evaluate_event_hits(alerts_v3, returns)
        in_window = eval_hits[eval_hits["in_test_window"]]
        hit_count = int(in_window["warned_30d"].sum())
        n_in = len(in_window)

        _ = evaluate_precision_recall(
            alerts_v4.assign(
                fired_v4=alerts_v4["trigger_v4"].astype(str).isin(
                    ["bear_vol_strong", "bear_vol_mild"]
                ).astype(int)
            ),
            returns, label=fold_name + " (v4)", fired_col="fired_v4",
        )

        row = {
            "fold": fold_name,
            "test_start": test_start_str,
            "test_end": test_end_str,
            "n_days": len(hw),
            "n_fired_v3": n_fired,
            "TP": pr["TP"], "FP": pr["FP"], "FN": pr["FN"],
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": pr["f1"],
            "hit_rate_event": hit_count / max(n_in, 1),
            "n_event_hit": hit_count,
            "n_event_total": n_in,
            "sharpe_baseline": overlay["baseline_sharpe"],
            "sharpe_overlay": overlay["overlay_sharpe"],
            "calmar_baseline": overlay["baseline_calmar"],
            "calmar_overlay": overlay["overlay_calmar"],
            "maxdd_baseline": overlay["baseline_maxdd"],
            "maxdd_overlay": overlay["overlay_maxdd"],
            "cagr_overlay": overlay["overlay_cagr"],
            "diff_final": overlay["diff_final"],
        }
        rows.append(row)

        print(f"  [{fold_name}] {len(hw)} 天, fired={n_fired}, "
              f"P={pr['precision']:.0%}, R={pr['recall']:.0%}, "
              f"F1={pr['f1']:.3f}, "
              f"events={hit_count}/{n_in}, "
              f"Sharpe={overlay['overlay_sharpe']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "walk_forward_results.csv", index=False)
    print(f"[保存] {EVAL_DIR / 'walk_forward_results.csv'}")

    print()
    print("  Walk-Forward 汇总:")
    if len(df) > 0:
        for col in ["precision", "recall", "f1", "sharpe_overlay", "calmar_overlay"]:
            print(f"    {col}: mean={df[col].mean():.3f}, "
                  f"std={df[col].std():.3f}, "
                  f"min={df[col].min():.3f}, max={df[col].max():.3f}")
    return df


def bootstrap_metrics(alerts: pd.DataFrame, returns: pd.DataFrame,
                      n_boot: int = N_BOOT,
                      horizon: int = 10,
                      neg_thresh: float = -0.01,
                      seed: int = BOOT_SEED,
                      fired_col: str = "trigger_v3") -> pd.DataFrame:
    """Bootstrap 评估指标置信区间"""
    print()
    print("=" * 70)
    print(f"Bootstrap CI ({n_boot} 次, fired_col={fired_col})")
    print("=" * 70)

    if fired_col == "trigger_v3":
        alerts_w_fired = alerts.copy()
        alerts_w_fired["fired"] = (alerts[fired_col] == "and_fired").astype(int)
        fired_col_use = "fired"
    elif fired_col == "trigger_v4":
        alerts_w_fired = alerts.copy()
        alerts_w_fired["fired"] = alerts[fired_col].astype(str).isin(
            ["bear_vol_strong", "bear_vol_mild"]
        ).astype(int)
        fired_col_use = "fired"
    elif fired_col == "alert_level":
        alerts_w_fired = alerts.copy()
        alerts_w_fired["fired"] = alerts["alert_level"].isin(["yellow", "red"]).astype(int)
        fired_col_use = "fired"
    else:
        alerts_w_fired = alerts
        fired_col_use = fired_col

    fired_idx = alerts.index[
        alerts_w_fired[fired_col_use].fillna(0).astype(int) == 1
    ].tolist()
    if len(fired_idx) < 5:
        print(f"  fired 数太少 ({len(fired_idx)})，跳过")
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    boot_p, boot_r, boot_f1 = [], [], []
    market = returns.mean(axis=1)

    for _ in range(n_boot):
        sample = list(rng.choice(fired_idx, size=len(fired_idx), replace=True))
        sample_set = set(sample)

        tp, fp = 0, 0
        for d in sample_set:
            post = market.loc[d:].iloc[1:horizon + 1]
            if len(post) < 2:
                continue
            cumret = float((1 + post).prod() - 1)
            if cumret < neg_thresh:
                tp += 1
            elif cumret > -neg_thresh:
                fp += 1

        fn = 0
        for ev in KNOWN_EVENTS:
            ev_d = pd.Timestamp(ev["date"])
            if not (alerts.index[0] <= ev_d <= alerts.index[-1]):
                continue
            prior30 = [d for d in sample if ev_d - pd.Timedelta(days=30) <= d < ev_d]
            if len(prior30) == 0:
                fn += 1

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        boot_p.append(p)
        boot_r.append(r)
        boot_f1.append(f1)

    boot_p = np.array(boot_p)
    boot_r = np.array(boot_r)
    boot_f1 = np.array(boot_f1)

    rows = []
    for name, arr in [("precision", boot_p), ("recall", boot_r), ("f1", boot_f1)]:
        rows.append({
            "metric": name,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_low_2.5": float(np.percentile(arr, 2.5)),
            "ci_high_97.5": float(np.percentile(arr, 97.5)),
            "ci_low_10": float(np.percentile(arr, 10)),
            "ci_high_90": float(np.percentile(arr, 90)),
            "n_boot": n_boot,
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "bootstrap_ci.csv", index=False)
    print(f"[保存] {EVAL_DIR / 'bootstrap_ci.csv'}")
    for _, r in df.iterrows():
        print(f"  {r['metric']:10s}: mean={r['mean']:.3f}, "
              f"90% CI=[{r['ci_low_10']:.3f}, {r['ci_high_90']:.3f}], "
              f"95% CI=[{r['ci_low_2.5']:.3f}, {r['ci_high_97.5']:.3f}]")
    return df


def compute_roc_auc(alerts: pd.DataFrame, returns: pd.DataFrame,
                    horizon: int = 10,
                    neg_thresh: float = -0.01) -> pd.DataFrame:
    """ROC 曲线: 扫描 width_z 阈值

    定义:
      positive class: fired 后 horizon 天累计收益 < neg_thresh (真下行)
      fired = width_z > threshold
    """
    print()
    print("=" * 70)
    print(f"ROC/AUC (扫描 width_z 阈值, horizon={horizon}d)")
    print("=" * 70)

    if "width_z" not in alerts.columns:
        print("  alerts 缺少 width_z 列，跳过")
        return pd.DataFrame()

    market = returns.mean(axis=1)
    wz_values = alerts["width_z"].fillna(-10).values
    is_fired_template = alerts["width_z"].notna().values
    valid_dates = alerts.index[is_fired_template].tolist()

    rows = []
    thresholds = np.arange(-0.5, 6.0, 0.25)
    for thresh in thresholds:
        fired_dates = [d for d, wz in zip(valid_dates, wz_values) if wz > thresh]
        n_fired = len(fired_dates)

        tp, fp, neg_label = 0, 0, 0
        for d in fired_dates:
            post = market.loc[d:].iloc[1:horizon + 1]
            if len(post) < 2:
                continue
            cumret = float((1 + post).prod() - 1)
            if cumret < neg_thresh:
                tp += 1
            elif cumret > -neg_thresh:
                fp += 1
            else:
                neg_label += 1

        n_neg_total = sum(
            1 for d in valid_dates
            if (post := market.loc[d:].iloc[1:horizon + 1]) is not None
            and len(post) > 1
            and float((1 + post).prod() - 1) < neg_thresh
        )
        n_pos_total = sum(
            1 for d in valid_dates
            if (post := market.loc[d:].iloc[1:horizon + 1]) is not None
            and len(post) > 1
            and float((1 + post).prod() - 1) > -neg_thresh
        )

        tpr = tp / max(n_neg_total, 1)
        fpr = fp / max(n_pos_total, 1)
        precision = tp / max(n_fired, 1)

        rows.append({
            "threshold": float(thresh),
            "n_fired": n_fired,
            "TP": tp, "FP": fp,
            "TPR": tpr, "FPR": fpr,
            "precision": precision,
        })

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "roc_curve_data.csv", index=False)
    print(f"[保存] {EVAL_DIR / 'roc_curve_data.csv'}")

    if len(df) > 0:
        df_sorted = df.sort_values("FPR")
        auc = float(np.trapz(df_sorted["TPR"].values, df_sorted["FPR"].values))
        print(f"  AUC = {auc:.3f}")
        print("  最佳 threshold (按 precision):")
        top = df.nlargest(3, "precision")
        for _, r in top.iterrows():
            print(f"    thresh={r['threshold']:+.2f}: P={r['precision']:.0%}, "
                  f"TPR={r['TPR']:.0%}, FPR={r['FPR']:.0%}, n_fired={r['n_fired']}")
    return df


def coverage_deep_analysis(intervals: dict, actual_returns: pd.DataFrame) -> dict:
    """模型校准深度分析"""
    print()
    print("=" * 70)
    print("模型校准深度分析")
    print("=" * 70)

    lower = intervals["lower"]
    upper = intervals["upper"]
    actual = actual_returns.reindex(columns=lower.columns, index=lower.index)
    covered = ((actual >= lower) & (actual <= upper)).astype(float)

    per_asset = covered.mean(axis=0)
    sorted_pa = np.sort(per_asset.values)
    n_pa = len(sorted_pa)
    worst_10_pct = float(sorted_pa[:max(1, n_pa // 10)].mean())

    w_ts = width_timeseries(intervals["half_width"])
    sigma = estimate_volatility(actual)
    aligned = pd.concat([w_ts, sigma.mean(axis=1)], axis=1).dropna()
    aligned.columns = ["width", "realized_vol"]
    if len(aligned) >= 2:
        width_vol_corr = float(aligned["width"].corr(aligned["realized_vol"]))
    else:
        width_vol_corr = float("nan")

    width_stab = width_stability(intervals["half_width"], window=60)
    marginal = float(covered.values.mean())
    pa_std = float(per_asset.std())

    coverage = compute_coverage_metrics(actual, lower, upper)
    print(f"  边际覆盖率: {coverage['marginal']:.4f} (目标 0.95)")
    print(f"  pa_std: {coverage['pa_std']:.4f}")
    print(f"  worst10: {coverage['worst10']:.4f}")
    print(f"  min per-asset: {coverage['min']:.4f}")
    print(f"  worst-10% (新): {worst_10_pct:.4f}")
    print(f"  Width-Vol 相关性: {width_vol_corr:.4f}")
    print(f"  Width Stability (CV): {width_stab:.4f}")

    out = {
        "marginal_coverage": marginal,
        "target_coverage": 0.95,
        "coverage_gap": 0.95 - marginal,
        "pa_std": pa_std,
        "worst10_coverage": coverage["worst10"],
        "min_per_asset_coverage": coverage["min"],
        "worst_10pct_avg_coverage": worst_10_pct,
        "width_volatility_correlation": width_vol_corr,
        "width_stability_cv": width_stab,
        "extreme_day_coverage": coverage.get("extreme", float("nan")),
    }

    pd.DataFrame([out]).to_csv(
        EVAL_DIR / "coverage_calibration.csv", index=False
    )
    print(f"[保存] {EVAL_DIR / 'coverage_calibration.csv'}")
    return out


def scenario_performance(alerts: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """分场景压力测试"""
    print()
    print("=" * 70)
    print("场景压力测试")
    print("=" * 70)

    if "trigger_v3" in alerts.columns:
        alerts_w_fired = alerts.copy()
        alerts_w_fired["fired"] = (alerts["trigger_v3"] == "and_fired").astype(int)
    else:
        alerts_w_fired = alerts

    rows = []
    for start_str, end_str, name in SCENARIOS:
        start = pd.Timestamp(start_str)
        end = pd.Timestamp(end_str)
        in_window = alerts_w_fired.loc[start:end]
        if len(in_window) == 0:
            continue
        in_returns = returns.loc[start:end]
        n_fired = int(in_window["fired"].sum())

        pr = evaluate_precision_recall(
            in_window, in_returns, label=name, fired_col="fired"
        )

        market = in_returns.mean(axis=1)
        if len(market) > 5:
            rets = market.pct_change().dropna()
            ann_ret = float((1 + rets).mean() ** 252 - 1)
            ann_vol = float(rets.std() * np.sqrt(252))
            sharpe = float((ann_ret - 0.02) / ann_vol) if ann_vol > 0 else 0.0
            max_dd = float(((1 + rets).cumprod() / (1 + rets).cumprod().cummax() - 1).min())
        else:
            ann_ret = ann_vol = sharpe = max_dd = float("nan")

        row = {
            "scenario": name,
            "start": start_str, "end": end_str,
            "n_days": len(in_window),
            "n_fired": n_fired,
            "TP": pr["TP"], "FP": pr["FP"], "FN": pr["FN"],
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": pr["f1"],
            "scenario_annret": ann_ret,
            "scenario_vol": ann_vol,
            "scenario_sharpe": sharpe,
            "scenario_maxdd": max_dd,
        }
        rows.append(row)
        print(f"  [{name}] {len(in_window)} 天, fired={n_fired}, "
              f"P={pr['precision']:.0%}, R={pr['recall']:.0%}, "
              f"F1={pr['f1']:.3f}, 场景 Sharpe={sharpe:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "scenario_performance.csv", index=False)
    print(f"[保存] {EVAL_DIR / 'scenario_performance.csv'}")
    return df


def optimize_scale_rules(alerts: pd.DataFrame, returns: pd.DataFrame,
                         top_n: int = 10) -> pd.DataFrame:
    """缩仓规则网格搜索"""
    print()
    print("=" * 70)
    print(f"缩仓规则网格 ({len(YELLOW_SCALES)}x{len(RED_SCALES)}={len(YELLOW_SCALES)*len(RED_SCALES)} 组)")  # noqa: E501
    print("=" * 70)

    common_idx = alerts.index.intersection(returns.index)
    m_ret = returns.loc[common_idx].mean(axis=1)
    a = alerts.loc[common_idx]
    if "trigger_v3" in a.columns:
        scale_base = pd.Series(1.0, index=a.index)
        scale_base[a["trigger_v3"] == "and_fired"] = 1.0
    else:
        scale_base = pd.Series(1.0, index=a.index)

    def metrics_for(ys: float, rs: float) -> dict:
        scale = pd.Series(1.0, index=a.index)
        if "alert_level" in a.columns:
            scale[a["alert_level"] == "yellow"] = ys
            scale[a["alert_level"] == "red"] = rs
        adj_ret = m_ret * scale.shift(1).fillna(1.0)
        nav = (1 + adj_ret).cumprod()
        if len(nav) < 10:
            return None
        daily = nav.pct_change().dropna()
        years = len(daily) / 252
        cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
        vol = float(daily.std() * np.sqrt(252))
        sharpe = float((daily.mean() * 252 - 0.02) / vol) if vol > 0 else 0.0
        max_dd = float(((nav / nav.cummax()) - 1).min())
        calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
        base_cagr = float((m_ret.iloc[-1] / m_ret.iloc[0]) ** (1 / years) - 1)
        diff_final = float(nav.iloc[-1] / m_ret.iloc[-1] - 1)
        return {
            "cagr": cagr, "vol": vol, "sharpe": sharpe,
            "max_dd": max_dd, "calmar": calmar,
            "base_cagr": base_cagr, "diff_final": diff_final,
        }

    rows = []
    for ys in YELLOW_SCALES:
        for rs in RED_SCALES:
            m = metrics_for(ys, rs)
            if m is None:
                continue
            rows.append({
                "yellow_scale": ys,
                "red_scale": rs,
                "cagr": m["cagr"],
                "vol": m["vol"],
                "sharpe": m["sharpe"],
                "max_dd": m["max_dd"],
                "calmar": m["calmar"],
                "base_cagr": m["base_cagr"],
                "diff_final": m["diff_final"],
            })

    df = pd.DataFrame(rows).sort_values("calmar", ascending=False)
    df.to_csv(EVAL_DIR / "scale_optimization.csv", index=False)
    print(f"[保存] {EVAL_DIR / 'scale_optimization.csv'}")
    print(f"  Top-{top_n} (按 Calmar):")
    for _, r in df.head(top_n).iterrows():
        print(f"    yellow={r['yellow_scale']:.2f}, red={r['red_scale']:.2f}: "
              f"CAGR={r['cagr']:+.2%}, Sharpe={r['sharpe']:.3f}, "
              f"Calmar={r['calmar']:.3f}, MaxDD={r['max_dd']:+.2%}, "
              f"终值差={r['diff_final']:+.2%}")
    return df


def cross_pool_robustness(returns: pd.DataFrame) -> pd.DataFrame:
    """跨池稳健性: 在不同 ETF 子集上评估"""
    print()
    print("=" * 70)
    print("跨池稳健性 (3 个 ETF 池)")
    print("=" * 70)

    sectors = get_asset_sectors()
    all_codes = returns.columns.tolist()

    a_broad = [c for c, s in sectors.items() if s == "a_broad" and c in all_codes]
    a_sector = [c for c, s in sectors.items() if s == "a_sector" and c in all_codes]
    smart_codes = ["510300", "510500", "510050", "159915", "588000", "159901",
                   "510880", "512890", "512260", "515900", "512040", "159786",
                   "515080", "515100"]
    smart_codes = [c for c in smart_codes if c in all_codes]

    pools = {
        "all_44": all_codes,
        "a_only": a_broad + a_sector,
        "smart_beta": smart_codes,
    }

    rows = []
    for pool_name, codes in pools.items():
        if len(codes) < 10:
            print(f"  [{pool_name}] 跳过 (仅 {len(codes)} 只 ETF)")
            continue
        sub_returns = returns[codes]

        config = CAGCPConfig(k=6, sensitivity_eta=0.5, recency_tau=20.0)
        pipe = CAGCPipeline(config)
        pipe.fit(sub_returns)
        intervals = pipe.predict_fast(pipe._calib, pipe._test)

        hw = intervals["half_width"]
        stress = intervals["stress"]
        trend_ok = compute_trend_signal(sub_returns).reindex(hw.index).fillna(False)
        alerts_v3 = build_alerts_v3(hw, stress, sub_returns, trend_ok=trend_ok)

        pr = evaluate_precision_recall(alerts_v3, sub_returns, label=pool_name,
                                       fired_col="trigger_v3")
        overlay = backtest_overlay(alerts_v3, sub_returns)
        n_fired = int((alerts_v3["trigger_v3"] == "and_fired").sum())

        row = {
            "pool": pool_name,
            "n_etfs": len(codes),
            "n_fired": n_fired,
            "TP": pr["TP"], "FP": pr["FP"], "FN": pr["FN"],
            "precision": pr["precision"],
            "recall": pr["recall"],
            "f1": pr["f1"],
            "sharpe_baseline": overlay["baseline_sharpe"],
            "sharpe_overlay": overlay["overlay_sharpe"],
            "calmar_baseline": overlay["baseline_calmar"],
            "calmar_overlay": overlay["overlay_calmar"],
            "diff_final": overlay["diff_final"],
        }
        rows.append(row)
        print(f"  [{pool_name}] {len(codes)} ETF, fired={n_fired}, "
              f"P={pr['precision']:.0%}, R={pr['recall']:.0%}, F1={pr['f1']:.3f}, "
              f"Sharpe={overlay['overlay_sharpe']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(EVAL_DIR / "cross_pool_robustness.csv", index=False)
    print(f"[保存] {EVAL_DIR / 'cross_pool_robustness.csv'}")
    return df


def build_scorecard(wf_df: pd.DataFrame, boot_df: pd.DataFrame,
                    cov: dict, pr_results: list,
                    scale_df: pd.DataFrame) -> dict:
    """综合评分卡"""
    print()
    print("=" * 70)
    print("综合评分卡 (3 种投资者类型)")
    print("=" * 70)

    pr_v3 = next((r for r in pr_results if "v3 and_fired" in r["label"]), None)
    boot_f1 = boot_df[boot_df["metric"] == "f1"].iloc[0] if len(boot_df) > 0 else None

    precision_score = float(pr_v3["precision"] * 10) if pr_v3 else 0.0
    recall_score = float(pr_v3["recall"] * 5) if pr_v3 else 0.0
    f1_score = float(pr_v3["f1"] * 10) if pr_v3 else 0.0

    lead_days = 7.0
    lead_score = min(lead_days / 14 * 5, 5.0)

    wf_sharpe = wf_df["sharpe_overlay"].mean() if len(wf_df) > 0 else 0.0
    wf_consistency = (
        (wf_df["sharpe_overlay"] > 0.5).sum() / max(len(wf_df), 1) * 10
        if len(wf_df) > 0 else 0.0
    )

    coverage_gap_score = max(0, (1 - abs(cov.get("coverage_gap", 0)) / 0.1) * 10)

    f1_ci_low = boot_f1["ci_low_2.5"] if boot_f1 is not None else 0.0
    f1_ci_low_score = min(max(f1_ci_low / 0.5, 0), 1) * 10

    conservative = (
        precision_score * 0.30
        + lead_score * 0.20
        + coverage_gap_score * 0.20
        + wf_consistency * 0.15
        + f1_ci_low_score * 0.15
    )
    balanced = (
        precision_score * 0.20
        + recall_score * 0.15
        + f1_score * 0.25
        + wf_consistency * 0.20
        + coverage_gap_score * 0.10
        + f1_ci_low_score * 0.10
    )
    aggressive = (
        recall_score * 0.30
        + lead_score * 0.30
        + f1_score * 0.20
        + wf_sharpe * 5 * 0.20
    )

    out = {
        "v3_and_fired": {
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "lead_score": lead_score,
            "wf_sharpe_mean": wf_sharpe,
            "wf_consistency_score": wf_consistency,
            "coverage_gap_score": coverage_gap_score,
            "f1_ci_low_2.5": f1_ci_low,
            "f1_ci_low_score": f1_ci_low_score,
        },
        "scorecard": {
            "conservative": conservative,
            "balanced": balanced,
            "aggressive": aggressive,
        },
    }

    print(f"  保守型评分: {conservative:.2f}/10")
    print(f"  平衡型评分: {balanced:.2f}/10")
    print(f"  激进型评分: {aggressive:.2f}/10")
    print()
    print("  子项明细:")
    print(f"    Precision 评分: {precision_score:.2f}")
    print(f"    Recall    评分: {recall_score:.2f}")
    print(f"    F1        评分: {f1_score:.2f}")
    print(f"    Lead Time 评分: {lead_score:.2f}")
    print(f"    WF Sharpe 均值: {wf_sharpe:.3f}")
    print(f"    WF 一致性评分: {wf_consistency:.2f}")
    print(f"    校准 Gap 评分: {coverage_gap_score:.2f}")
    print(f"    F1 95% CI 下界: {f1_ci_low:.3f}")
    print(f"    F1 CI 评分: {f1_ci_low_score:.2f}")

    return out


def main() -> None:
    print("=" * 70)
    print("CA-GCP 预警系统 — 完整有效性评估")
    print("=" * 70)

    returns = load_returns()
    print(f"[数据] {returns.shape[0]} 天 × {returns.shape[1]} ETF, "
          f"{returns.index[0].date()} ~ {returns.index[-1].date()}")

    print(f"[滚动] train={TRAIN_WINDOW}, calib={CALIB_WINDOW}, step={PRED_STEP}")
    hw, stress, lower, upper = rolling_predict(returns)
    intervals = {
        "half_width": hw, "stress": stress,
        "lower": lower, "upper": upper,
    }
    print(f"[预测] {len(hw)} 天, {hw.index[0].date()} ~ {hw.index[-1].date()}")

    trend_ok = compute_trend_signal(returns)
    alerts_raw = build_alerts(hw, stress)
    alerts_tf = build_alerts(hw, stress, trend_ok=trend_ok)
    alerts_v3 = build_alerts_v3(hw, stress, returns, trend_ok=trend_ok)
    alerts_v4 = build_alerts_v4(hw, stress, returns, trend_ok=trend_ok)
    alerts_conf = build_alerts_confidence(intervals, returns, trend_ok=trend_ok)

    wf_df = walk_forward_evaluate(returns)

    boot_v3 = bootstrap_metrics(alerts_v3, returns, fired_col="trigger_v3")
    boot_v4 = bootstrap_metrics(alerts_v4, returns, fired_col="trigger_v4")
    boot_raw = bootstrap_metrics(alerts_raw, returns, fired_col="fired")
    boot_conf = bootstrap_metrics(alerts_conf, returns, fired_col="alert_level")

    roc_df = compute_roc_auc(alerts_tf, returns)

    cov = coverage_deep_analysis(intervals, returns)

    scen_df = scenario_performance(alerts_v3, returns)

    scale_df = optimize_scale_rules(alerts_v3, returns)

    pool_df = cross_pool_robustness(returns)

    pr_results = []
    for lbl, al, fc in [
        ("无过滤", alerts_raw, "fired"),
        ("有趋势", alerts_tf, "fired"),
        ("v3 and_fired", alerts_v3, "trigger_v3"),
        ("v3 yellow/red", alerts_v3, "alert_level"),
        ("v4 bear_vol", alerts_v4, "trigger_v4"),
        ("v4 yellow/red", alerts_v4, "alert_level"),
        ("conf yellow/red", alerts_conf, "alert_level"),
    ]:
        if fc in ("trigger_v3", "trigger_v4"):
            al_w = al.copy()
            if fc == "trigger_v3":
                al_w["fired"] = (al[fc] == "and_fired").astype(int)
            else:
                al_w["fired"] = al[fc].astype(str).isin(
                    ["bear_vol_strong", "bear_vol_mild"]
                ).astype(int)
            pr = evaluate_precision_recall(al_w, returns, label=lbl, fired_col="fired")
        else:
            pr = evaluate_precision_recall(al, returns, label=lbl, fired_col=fc)
        pr_results.append(pr)

    scorecard = build_scorecard(wf_df, boot_v3, cov, pr_results, scale_df)

    write_summary_report(
        wf_df, boot_v3, boot_v4, boot_raw, boot_conf, roc_df, cov, scen_df,
        scale_df, pool_df, pr_results, scorecard
    )

    print()
    print("=" * 70)
    print("评估完成 — 所有结果已保存")
    print("=" * 70)
    print(f"输出目录: {EVAL_DIR}")


def write_summary_report(wf_df, boot_v3, boot_v4, boot_raw, boot_conf, roc_df,
                         cov, scen_df, scale_df, pool_df, pr_results,
                         scorecard) -> None:
    """输出评估汇总到 Markdown"""
    lines = [
        "# CA-GCP 预警系统有效性评估报告",
        "",
        f"数据: 44 ETF 日收益率, "
        f"{pd.read_parquet(DATA_PATH).index[0].date()} ~ "
        f"{pd.read_parquet(DATA_PATH).index[-1].date()}",
        "参数: k=6, eta=0.5, tau=20, alpha=0.05 (v10.2 校准)",
        "",
        "## 1. Walk-Forward 评估 (4 折)",
        "",
        "| 折 | 测试期 | 天数 | fired | P | R | F1 | "
        "事件命中 | Sharpe(基准/叠加) | Calmar(基准/叠加) | MaxDD(基准/叠加) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    if len(wf_df) > 0:
        for _, r in wf_df.iterrows():
            lines.append(
                f"| {r['fold']} | {r['test_start']} ~ {r['test_end']} | "
                f"{r['n_days']} | {r['n_fired_v3']} | "
                f"{r['precision']:.0%} | {r['recall']:.0%} | {r['f1']:.3f} | "
                f"{r['n_event_hit']}/{r['n_event_total']} | "
                f"{r['sharpe_baseline']:.3f} / {r['sharpe_overlay']:.3f} | "
                f"{r['calmar_baseline']:.3f} / {r['calmar_overlay']:.3f} | "
                f"{r['maxdd_baseline']:+.1%} / {r['maxdd_overlay']:+.1%} |"
            )
        lines += [
            "",
            "**Walk-Forward 汇总**:",
            f"- 精度均值: {wf_df['precision'].mean():.1%} ± {wf_df['precision'].std():.1%}",
            f"- 召回均值: {wf_df['recall'].mean():.1%} ± {wf_df['recall'].std():.1%}",
            f"- F1 均值: {wf_df['f1'].mean():.3f} ± {wf_df['f1'].std():.3f}",
            f"- 叠加层 Sharpe 均值: {wf_df['sharpe_overlay'].mean():.3f}",
            f"- 有效折数 (Sharpe > 0.5): "
            f"{(wf_df['sharpe_overlay'] > 0.5).sum()} / {len(wf_df)}",
        ]

    lines += [
        "",
        "## 2. Bootstrap 置信区间 (1000 次)",
        "",
        "| 指标 | 均值 | 95% CI | 90% CI |",
        "|---|---|---|---|",
    ]
    for _, r in boot_v3.iterrows():
        lines.append(
            f"| {r['metric']} | {r['mean']:.3f} | "
            f"[{r['ci_low_2.5']:.3f}, {r['ci_high_97.5']:.3f}] | "
            f"[{r['ci_low_10']:.3f}, {r['ci_high_90']:.3f}] |"
        )

    lines += [
        "",
        "**关键发现 (v3 vs v4)**:",
        f"- v3 F1 95% CI 下界 = "
        f"{boot_v3[boot_v3['metric']=='f1'].iloc[0]['ci_low_2.5']:.3f}",
        f"- v4 F1 95% CI 下界 = "
        f"{boot_v4[boot_v4['metric']=='f1'].iloc[0]['ci_low_2.5']:.3f}"
        if len(boot_v4) > 0 else "- v4 数据不足",
        "  (若 > 0.35，模型显著有效)",
        f"- v3 Precision 95% CI = "
        f"[{boot_v3[boot_v3['metric']=='precision'].iloc[0]['ci_low_2.5']:.3f}, "
        f"{boot_v3[boot_v3['metric']=='precision'].iloc[0]['ci_high_97.5']:.3f}]",
        "",
        "## 3. ROC/AUC",
        "",
    ]
    if len(roc_df) > 0:
        roc_sorted = roc_df.sort_values("FPR")
        auc = float(np.trapz(roc_sorted["TPR"].values, roc_sorted["FPR"].values))
        lines.append(f"- AUC = {auc:.3f}")
        top = roc_df.nlargest(3, "precision")
        lines.append("- 最佳阈值 (按 Precision):")
        for _, r in top.iterrows():
            lines.append(
                f"  - thresh={r['threshold']:+.2f}: P={r['precision']:.0%}, "
                f"TPR={r['TPR']:.0%}, FPR={r['FPR']:.0%}, n_fired={r['n_fired']}"
            )

    lines += [
        "",
        "## 4. 模型校准深度",
        "",
        "| 指标 | 值 | 说明 |",
        "|---|---|---|",
        f"| 边际覆盖率 | {cov['marginal_coverage']:.4f} | 目标 0.95, "
        f"gap={cov['coverage_gap']:+.4f} |",
        f"| Worst-10% 资产覆盖 | {cov['worst_10pct_avg_coverage']:.4f} | "
        "最差 10% 资产平均覆盖率 |",
        f"| 单资产最低覆盖 | {cov['min_per_asset_coverage']:.4f} | "
        "覆盖最差的资产 |",
        f"| Width-Vol 相关性 | {cov['width_volatility_correlation']:.4f} | "
        "区间宽度适应真实波动的能力 (>0.5 健康) |",
        f"| Width Stability (CV) | {cov['width_stability_cv']:.4f} | "
        "区间宽度滚动变异系数 (越小越稳定) |",
        "",
        "## 5. 场景压力测试",
        "",
        "| 场景 | 天数 | fired | P | R | F1 | 场景 Sharpe | 场景 MaxDD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in scen_df.iterrows():
        lines.append(
            f"| {r['scenario']} | {r['n_days']} | {r['n_fired']} | "
            f"{r['precision']:.0%} | {r['recall']:.0%} | {r['f1']:.3f} | "
            f"{r['scenario_sharpe']:.2f} | {r['scenario_maxdd']:+.1%} |"
        )

    lines += [
        "",
        "## 6. 缩仓规则网格 (Top 5 by Calmar)",
        "",
        "| yellow | red | CAGR | Sharpe | Calmar | MaxDD | 终值差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in scale_df.head(5).iterrows():
        lines.append(
            f"| {r['yellow_scale']:.2f} | {r['red_scale']:.2f} | "
            f"{r['cagr']:+.2%} | {r['sharpe']:.3f} | {r['calmar']:.3f} | "
            f"{r['max_dd']:+.2%} | {r['diff_final']:+.2%} |"
        )

    lines += [
        "",
        "## 7. 跨池稳健性",
        "",
        "| 池 | ETF数 | fired | P | R | F1 | Sharpe (基准/叠加) | 终值差 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in pool_df.iterrows():
        lines.append(
            f"| {r['pool']} | {r['n_etfs']} | {r['n_fired']} | "
            f"{r['precision']:.0%} | {r['recall']:.0%} | {r['f1']:.3f} | "
            f"{r['sharpe_baseline']:.3f} / {r['sharpe_overlay']:.3f} | "
            f"{r['diff_final']:+.2%} |"
        )

    lines += [
        "",
        "## 8. Precision/Recall/F1 (各版本)",
        "",
        "| 版本 | n_fired | TP | FP | FN | P | R | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pr in pr_results:
        lines.append(
            f"| {pr['label']} | {pr['n_fired']} | {pr['TP']} | {pr['FP']} | "
            f"{pr['FN']} | {pr['precision']:.1%} | {pr['recall']:.1%} | "
            f"{pr['f1']:.3f} |"
        )

    lines += [
        "",
        "## 9. 综合评分卡",
        "",
        f"- **保守型评分**: {scorecard['scorecard']['conservative']:.2f} / 10",
        "  (重 Precision / Calmar / 校准)",
        f"- **平衡型评分**: {scorecard['scorecard']['balanced']:.2f} / 10",
        "  (重 F1 / Sharpe / WF 一致性)",
        f"- **激进型评分**: {scorecard['scorecard']['aggressive']:.2f} / 10",
        "  (重 Recall / Lead Time)",
        "",
        "## 10. 最终判定",
        "",
    ]
    cons = scorecard["scorecard"]["conservative"]
    bal = scorecard["scorecard"]["balanced"]
    aggr = scorecard["scorecard"]["aggressive"]
    lines += [
        "| 评分维度 | 分数 |",
        "|---|---|",
        f"| 保守型 | {cons:.2f}/10 |",
        f"| 平衡型 | {bal:.2f}/10 |",
        f"| 激进型 | {aggr:.2f}/10 |",
        "",
    ]
    if bal >= 7.0:
        lines.append("**结论**: 系统**有效**，建议生产部署 (平衡型评分 ≥ 7)。")
    elif bal >= 5.0:
        lines.append("**结论**: 系统**部分有效**，作为辅助信号可用，需配合人工/策略层。")
    else:
        lines.append("**结论**: 系统**有效性不足**，不建议独立使用，需大幅优化。")

    (EVAL_DIR / "evaluation_scorecard.md").write_text("\n".join(lines))
    print(f"\n[评分卡] {EVAL_DIR / 'evaluation_scorecard.md'}")


if __name__ == "__main__":
    main()
