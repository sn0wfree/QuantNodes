# coding: utf-8
"""v7.6 硬止损 (SL) 验证.

目的: 验证硬止损 (DD > X% → 全仓债券) 是否能将 CV% 降至 25%

用法:
   python3.11 scripts/v7_6_stop_loss_test.py

输出:
   reports/momentum_etf_rotation/v7_6_stop_loss_test.csv
   reports/momentum_etf_rotation/v7_6_stop_loss_test.md

SL 逻辑 (每个调仓日检查):
   if (peak_nav - current_nav) / peak_nav > sl_threshold:
        weights = {511260: 1.0}  # 全仓债券
   else:
        weights = original_weights

测试组合:
   - baseline (top_n=5) — 无 SL
   - SL 5% / 8% / 10% / 12% / 15%
   - SL + TF MA200
   - SL + TF + regime_combo 70%
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_6_data, load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DAYS_PER_YEAR = 252
START_POINTS = [
    "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
]

# 锁定 top_n=5 + rho=2.0
BASE_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
    "rho": 2.0,
    "top_n": 5,
    "max_weight": 0.25,
    "min_history": 52,
}

# SL 参数扫描
COMBOS = [
    {
        "label": "baseline_no_sl",
        "note": "无 SL (基线)",
        "tf_enabled": False, "regime_enabled": False,
        "sl_enabled": False,
    },
    # SL 单独 (无 TF/Regime)
    {
        "label": "sl_5",
        "note": "SL 5% DD → 511260",
        "sl_enabled": True, "sl_threshold": 0.05, "sl_bond": "511260",
    },
    {
        "label": "sl_8",
        "note": "SL 8% DD → 511260",
        "sl_enabled": True, "sl_threshold": 0.08, "sl_bond": "511260",
    },
    {
        "label": "sl_10",
        "note": "SL 10% DD → 511260",
        "sl_enabled": True, "sl_threshold": 0.10, "sl_bond": "511260",
    },
    {
        "label": "sl_15",
        "note": "SL 15% DD → 511260 (宽松)",
        "sl_enabled": True, "sl_threshold": 0.15, "sl_bond": "511260",
    },
    # SL + TF 联合
    {
        "label": "sl10_tf200",
        "note": "SL 10% + TF MA200",
        "sl_enabled": True, "sl_threshold": 0.10, "sl_bond": "511260",
        "tf_enabled": True, "tf_ma": 200, "tf_bear": 0.5, "tf_bond": "511260",
    },
    # SL + TF + regime_combo (最强防御)
    {
        "label": "sl10_tf_regime_70",
        "note": "SL 10% + TF + regime_combo 70%",
        "sl_enabled": True, "sl_threshold": 0.10, "sl_bond": "511260",
        "tf_enabled": True, "tf_ma": 200, "tf_bear": 0.5, "tf_bond": "511260",
        "regime_enabled": True,
        "regime_vol_thr": 0.20, "regime_bear": 0.7, "regime_bond": "511260",
    },
    {
        "label": "sl8_tf_regime_70",
        "note": "SL 8% + TF + regime_combo 70% (更紧 SL)",
        "sl_enabled": True, "sl_threshold": 0.08, "sl_bond": "511260",
        "tf_enabled": True, "tf_ma": 200, "tf_bear": 0.5, "tf_bond": "511260",
        "regime_enabled": True,
        "regime_vol_thr": 0.20, "regime_bear": 0.7, "regime_bond": "511260",
    },
]

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def compute_metrics(nav: pd.Series, freq: int = DAYS_PER_YEAR) -> dict:
    if nav.empty or len(nav) < 2:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    n_years = len(rets) / freq
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0
    return {"calmar": round(calmar, 4), "ann_return": round(ann_ret, 4),
            "vol": round(vol, 4), "max_dd": round(max_dd, 4), "sharpe": round(sharpe, 4)}


def get_regime_combo_signal(weekly_dates, daily_returns,
                             vol_thr=0.20, lookback=60):
    market_daily = daily_returns.mean(axis=1)
    vol_60 = market_daily.rolling(lookback).std() * np.sqrt(252)
    ret_60 = (1 + market_daily).rolling(lookback).apply(np.prod, raw=True) - 1
    signals = {}
    for wd in weekly_dates:
        valid = vol_60.index[vol_60.index <= wd]
        if len(valid) == 0:
            signals[wd] = False
            continue
        latest_date = valid[-1]
        vol_bear = vol_60.loc[latest_date] > vol_thr
        trend_bear = ret_60.loc[latest_date] < 0
        signals[wd] = bool(vol_bear and trend_bear)
    return pd.Series(signals, name="regime_combo")


def get_trend_filter_signal(weekly_dates, daily_returns, ma=200, benchmark="沪深300指数"):
    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import load_benchmark_price
    bench_price = load_benchmark_price(benchmark)
    ma_series = bench_price.rolling(ma).mean()
    signals = {}
    for wd in weekly_dates:
        valid = bench_price.index[bench_price.index <= wd]
        if len(valid) == 0:
            signals[wd] = False
            continue
        latest_date = valid[-1]
        price_now = bench_price.loc[latest_date]
        ma_now = ma_series.loc[latest_date] if latest_date in ma_series.index else np.nan
        if pd.isna(ma_now) or pd.isna(price_now):
            signals[wd] = False
            continue
        signals[wd] = bool(price_now < ma_now)
    return pd.Series(signals, name="tf_signal")


def construct_portfolio_full(
    Y, X_panel, beta_path, cfg, daily_returns, combo,
):
    """完整组合: SL + TF + Regime + top_n."""
    T, N = Y.shape
    nav = pd.Series(1.0, index=Y.index, dtype=float)
    weights_history = []
    prev_weights = {}

    weekly_dates = list(Y.index)
    tf_signal = None
    regime_signal = None

    if combo.get("tf_enabled"):
        tf_signal = get_trend_filter_signal(weekly_dates, daily_returns, ma=combo.get("tf_ma", 200))
        tf_signal = tf_signal.shift(1).fillna(False)  # 延迟1周, 避免未来函数
        bear_count = int(tf_signal.sum())
        logging.info("  TF 信号: %d/%d 周为熊市 (%.1f%%)",
                     bear_count, len(tf_signal), bear_count / len(tf_signal) * 100)

    if combo.get("regime_enabled"):
        regime_signal = get_regime_combo_signal(
            weekly_dates, daily_returns,
            vol_thr=combo.get("regime_vol_thr", 0.20),
        )
        regime_signal = regime_signal.shift(1).fillna(False)  # 延迟1周, 避免未来函数
        bear_count = int(regime_signal.sum())
        logging.info("  Regime 信号: %d/%d 周为熊市 (%.1f%%)",
                     bear_count, len(regime_signal), bear_count / len(regime_signal) * 100)

    defense_code = combo.get("tf_bond", "511260")
    sl_threshold = combo.get("sl_threshold", 0.10)
    sl_bond = combo.get("sl_bond", "511260")

    for t in range(1, T):
        # 1. 预测
        beta_prev = beta_path.iloc[t - 1].values
        scores = X_panel[t - 1] @ beta_prev  # 用上期因子, 避免未来函数
        scores = pd.Series(scores, index=Y.columns).dropna()

        # 2. top_n
        if len(scores) >= cfg.top_n:
            chosen = scores.nlargest(cfg.top_n).index.tolist()
        elif len(scores) > 0:
            chosen = scores.index.tolist()
        else:
            nav.iloc[t] = nav.iloc[t - 1]
            continue

        # 3. 逆波动率加权
        if t >= cfg.vol_window:
            vol_window = Y.iloc[max(0, t - cfg.vol_window):t]
            vols = vol_window[chosen].std()
            vols = vols.fillna(cfg.vol_floor).clip(lower=cfg.vol_floor)
            inv_vol = 1.0 / vols
            weights = inv_vol / inv_vol.sum()
            weights = weights.clip(upper=cfg.max_weight)
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(chosen), index=chosen)

        date = Y.index[t]
        weights_dict = weights.to_dict()

        # 4. SL 检查
        if combo.get("sl_enabled") and t > 0:
            current_nav = nav.iloc[t - 1]
            peak_nav = nav.iloc[:t].max()
            dd = (peak_nav - current_nav) / peak_nav if peak_nav > 0 else 0
            if dd > sl_threshold:
                # 全部转防御
                weights_dict = {sl_bond: 1.0}

        # 5. TF / Regime 检查
        if combo.get("tf_enabled") or combo.get("regime_enabled"):
            tf_bear = (combo.get("tf_enabled") and
                       tf_signal is not None and
                       date in tf_signal.index and tf_signal.loc[date])
            reg_bear = (combo.get("regime_enabled") and
                        regime_signal is not None and
                        date in regime_signal.index and regime_signal.loc[date])

            bear_pct = 0.0
            if tf_bear or reg_bear:
                bear_pct = max(combo.get("tf_bear", 0.5), combo.get("regime_bear", 0.5))

            if bear_pct > 0 and not (combo.get("sl_enabled") and dd > sl_threshold):
                for code in list(weights_dict.keys()):
                    weights_dict[code] = weights_dict[code] * (1 - bear_pct)
                weights_dict[defense_code] = weights_dict.get(defense_code, 0) + bear_pct

        weights = pd.Series(weights_dict)

        for code, w in weights.items():
            weights_history.append({'date': date, 'code': code, 'weight': w})

        # 6. 周收益
        weekly_ret = 0.0
        for code, w in weights.items():
            if code in Y.columns:
                ret = Y[code].iloc[t]
                if pd.notna(ret):
                    weekly_ret += w * ret

        # 7. 成本
        if cfg.cost_enabled:
            turnover = 0.0
            for code in set(list(prev_weights.keys()) + list(weights.keys())):
                w_old = prev_weights.get(code, 0.0)
                w_new = weights.get(code, 0.0)
                turnover += abs(w_new - w_old)
            cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000
            weekly_ret -= turnover * cost_rate

        nav.iloc[t] = nav.iloc[t - 1] * (1 + weekly_ret)
        prev_weights = weights.to_dict()

    weights_df = pd.DataFrame(weights_history)
    return nav, weights_df


def calculate_daily_nav(weights_df, daily_returns, cfg):
    all_dates = daily_returns.index
    rebal_dates = sorted(weights_df["date"].unique())

    date_to_rebal = {}
    for idx, rebal_date in enumerate(rebal_dates):
        prev_dates = all_dates[all_dates <= rebal_date]
        if len(prev_dates) == 0:
            continue
        week_end = prev_dates[-1]
        if idx > 0:
            prev_rebal = rebal_dates[idx - 1]
            next_day_idx = all_dates.searchsorted(prev_rebal, side='right')
            if next_day_idx < len(all_dates):
                week_start = all_dates[next_day_idx]
            else:
                continue
        else:
            week_start_idx = all_dates.searchsorted(rebal_date) - 5
            if week_start_idx < 0:
                week_start_idx = 0
            week_start = all_dates[week_start_idx]

        week_mask = (all_dates >= week_start) & (all_dates <= week_end)
        for date in all_dates[week_mask]:
            date_to_rebal[date] = rebal_date

    daily_nav = pd.Series(1.0, index=all_dates, dtype=float)
    current_weights = {}
    cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000 if cfg.cost_enabled else 0.0

    for i in range(1, len(all_dates)):
        date = all_dates[i]
        rebal_date = date_to_rebal.get(date)

        if rebal_date is not None:
            new_weights_df = weights_df[weights_df["date"] == rebal_date]
            new_weights = {str(k): v for k, v in new_weights_df.set_index("code")["weight"].to_dict().items()}

            if cfg.cost_enabled:
                turnover = 0.0
                all_codes = set(list(current_weights.keys()) + list(new_weights.keys()))
                for code in all_codes:
                    w_old = current_weights.get(code, 0.0)
                    w_new = new_weights.get(code, 0.0)
                    turnover += abs(w_new - w_old)
            current_weights = new_weights

        daily_ret = 0.0
        for code, weight in current_weights.items():
            if code in daily_returns.columns:
                ret = daily_returns.loc[date, code]
                if pd.notna(ret):
                    daily_ret += weight * ret

        if rebal_date is not None and cfg.cost_enabled:
            daily_ret -= turnover * cost_rate

        daily_nav.iloc[i] = daily_nav.iloc[i - 1] * (1 + daily_ret)

    return daily_nav


def run_combo(X_panel, Y, valid_codes, daily_returns, combo: dict) -> dict:
    cfg = V7_6Config(**BASE_PARAMS)

    t0 = time.time()
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        method=cfg.method,
        min_history=cfg.min_history,
        window_size=cfg.window_size,
        rho=cfg.rho,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )
    nav_weekly, weights_df = construct_portfolio_full(
        Y, X_panel, beta_path, cfg, daily_returns, combo
    )
    nav_daily = calculate_daily_nav(weights_df, daily_returns, cfg)
    full = compute_metrics(nav_daily)
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    oos = compute_metrics(nav_daily_oos)

    start_calmar = []
    start_details = []
    for start in START_POINTS:
        mask = Y.index >= start
        Y_start = Y[mask]
        X_start = X_panel[mask]
        if len(Y_start) < cfg.min_history + 12:
            continue
        beta_path_start = tvpr_estimator(
            Y_start, X_start,
            lambda_tv=cfg.lambda_tv,
            lambda_l1=cfg.lambda_l1,
            method=cfg.method,
            min_history=cfg.min_history,
            window_size=cfg.window_size,
            rho=cfg.rho,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
        )
        nav_weekly_s, weights_df_s = construct_portfolio_full(
            Y_start, X_start, beta_path_start, cfg, daily_returns, combo
        )
        nav_daily_s = calculate_daily_nav(weights_df_s, daily_returns, cfg)
        m = compute_metrics(nav_daily_s)
        start_calmar.append(m["calmar"])
        start_details.append((start, m["calmar"]))

    cals = start_calmar
    mean_c = float(np.mean(cals)) if cals else 0
    std_c = float(np.std(cals)) if cals else 0
    cv = std_c / mean_c if mean_c > 0 else 0

    return {
        "label": combo["label"],
        "note": combo["note"],
        "oos_calmar": oos["calmar"],
        "oos_sharpe": oos["sharpe"],
        "oos_dd": oos["max_dd"],
        "oos_ann": oos["ann_return"],
        "full_calmar": full["calmar"],
        "full_sharpe": full["sharpe"],
        "start_mean": round(mean_c, 4),
        "start_std": round(std_c, 4),
        "start_cv": round(cv, 4),
        "start_details": start_details,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 硬止损 (SL) 验证")
    logging.info("=" * 60)

    X_panel, Y, valid_codes = load_v7_6_data()
    daily_returns = load_daily_etf_returns()
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    rows = []
    summaries = []
    for combo in COMBOS:
        logging.info("=" * 60)
        logging.info("Combo: %s - %s", combo["label"], combo["note"])
        try:
            r = run_combo(X_panel, Y, valid_codes, daily_returns, combo)
        except Exception as e:
            logging.error("  失败: %s", e)
            continue
        rows.append({k: v for k, v in r.items() if k != "start_details"})
        summaries.append(r)
        logging.info("  OOS Calmar=%.4f, 起点 CV%%=%.1f%%, 起点均值=%.4f, %.1fs",
                     r["oos_calmar"], r["start_cv"] * 100,
                     r["start_mean"], r["seconds"])
        for start, c in r["start_details"]:
            logging.info("    起点 %s: Calmar=%.4f", start, c)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "v7_6_stop_loss_test.csv"
    df.to_csv(out_csv, index=False)

    # 汇总
    print("\n" + "=" * 100)
    print("硬止损 (SL) 验证结果")
    print("=" * 100)
    cols = ["label", "oos_calmar", "oos_sharpe", "oos_dd", "start_mean", "start_cv", "seconds"]
    print(df[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("各组合起点 Calmar 分布:")
    print("=" * 100)
    header = "起点".ljust(12) + " | " + " | ".join([s["label"].ljust(22) for s in summaries])
    print(header)
    print("-" * len(header))
    for start in START_POINTS:
        line = start.ljust(12) + " | "
        for s in summaries:
            val = next((c for st, c in s["start_details"] if st == start), 0.0)
            line += f"{val:.4f}".ljust(24) + " | "
        print(line)
    means = "MEAN".ljust(12) + " | "
    for s in summaries:
        means += f"{s['start_mean']:.4f}".ljust(24) + " | "
    print(means)
    cvs = "CV%".ljust(12) + " | "
    for s in summaries:
        cvs += f"{s['start_cv']*100:.1f}%".ljust(24) + " | "
    print(cvs)

    # 判据
    print("\n" + "=" * 100)
    print("判据: 起点 CV% 是否从 46.7% 改进到 ≤25%?")
    print("=" * 100)
    base = next(s for s in summaries if s["label"] == "baseline_no_sl")
    base_cv = base["start_cv"]
    print(f"基线 (top_n=5, 无 SL): CV%={base_cv*100:.1f}%")
    print()
    for s in summaries:
        if s["label"] == "baseline_no_sl":
            continue
        diff = s["start_cv"] - base_cv
        if s["start_cv"] <= 0.25:
            status = "✅ PASS"
        elif s["start_cv"] < base_cv:
            status = "⬆️ 改善"
        else:
            status = "⬇️ 退化"
        print(f"  {status:12s} {s['label']:25s}: CV%={s['start_cv']*100:5.1f}% (差 {diff*100:+.1f}pp), "
              f"OOS Calmar={s['oos_calmar']:.4f}, Sharpe={s['oos_sharpe']:.2f}")

    # Markdown 报告
    lines = [
        "# v7.6 硬止损 (SL) 验证报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 背景",
        "",
        "- 当前 v7.6 (tf_regime_combo_70) CV% = 28.8%",
        "- 验证 SL 是否能将 CV% 进一步降至 25% 阈值",
        "",
        "## 测试组合",
        "",
        "| 标签 | OOS Calmar | OOS Sharpe | OOS DD | 起点均值 | 起点 CV% | 状态 |",
        "|------|-----------|------------|--------|----------|----------|------|",
    ]
    for s in summaries:
        status = "✅ PASS" if s["start_cv"] <= 0.25 else "❌ FAIL"
        lines.append(
            f"| {s['label']} | {s['oos_calmar']:.4f} | {s['oos_sharpe']:.2f} | "
            f"{s['oos_dd']*100:.2f}% | {s['start_mean']:.4f} | {s['start_cv']*100:.1f}% | {status} |"
        )

    lines.extend([
        "",
        "## 起点 Calmar 分布",
        "",
        "| 起点 | " + " | ".join([s["label"] for s in summaries]) + " |",
        "|------|" + "|".join(["------" for _ in summaries]) + "|",
    ])
    for start in START_POINTS:
        line = f"| {start} | "
        for s in summaries:
            val = next((c for st, c in s["start_details"] if st == start), 0.0)
            line += f"{val:.4f} | "
        lines.append(line)
    lines.append("| **MEAN** | " + " | ".join([f"**{s['start_mean']:.4f}**" for s in summaries]) + " |")
    lines.append("| **CV%** | " + " | ".join([f"**{s['start_cv']*100:.1f}%**" for s in summaries]) + " |")

    # 结论
    best_cv = min(summaries, key=lambda x: x["start_cv"])
    best_oos = max(summaries, key=lambda x: x["oos_calmar"])

    lines.extend([
        "",
        "## 结论",
        "",
    ])
    if best_cv["start_cv"] <= 0.25:
        lines.append(f"### ✅ 达到 ≤25% 阈值! 推荐锁定 **{best_cv['label']}**")
    else:
        lines.append(f"### ⚠️ 仍未达 ≤25% 阈值 (最低 {best_cv['start_cv']*100:.1f}%)")

    lines.append(f"- **最低 CV%**: {best_cv['label']} = {best_cv['start_cv']*100:.1f}%")
    lines.append(f"- **最高 OOS Calmar**: {best_oos['label']} = {best_oos['oos_calmar']:.4f}")
    lines.append(f"- 基线 46.7% → 最低 {best_cv['start_cv']*100:.1f}% "
                 f"(改进 {base_cv*100 - best_cv['start_cv']*100:.1f}pp)")

    report = "\n".join(lines)
    out_md = OUTPUT_DIR / "v7_6_stop_loss_test.md"
    out_md.write_text(report, encoding="utf-8")
    logging.info("=" * 60)
    logging.info("CSV: %s", out_csv)
    logging.info("MD: %s", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
