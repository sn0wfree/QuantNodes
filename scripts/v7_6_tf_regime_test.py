# coding: utf-8
"""v7.6 TF 与 Regime 检测测试.

目的: 验证两种构造层加固对起点 CV% 的改进
   1. Trend Filter (TF): 沪深300 < MA200 → 权重 × 0.5 + 0.5 到 511260 (国债)
   2. Regime 检测: 60 日波动率 + 趋势强度分类, 熊市用防御组合

用法:
   python3.11 scripts/v7_6_tf_regime_test.py

输出:
   reports/momentum_etf_rotation/v7_6_tf_regime_test.csv
   reports/momentum_etf_rotation/v7_6_tf_regime_test.md
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
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import load_benchmark_price
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

# 冻结 top_n=5 + rho=2.0 (Phase 5 最优)
BASE_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
    "rho": 2.0,
    "top_n": 5,
    "max_weight": 0.25,
    "min_history": 52,
}

# 测试组合
COMBOS = [
    {
        "label": "baseline_topn5",
        "note": "Phase 5 最优 (top_n=5, rho=2.0), 无加固",
        "tf_enabled": False, "regime_enabled": False,
    },
    {
        "label": "tf_ma200",
        "note": "TF: 沪深300 < MA200 → 权重 × 0.5 + 0.5 → 511260 (国债)",
        "tf_enabled": True, "tf_ma": 200, "tf_bear": 0.5, "tf_bond": "511260",
        "regime_enabled": False,
    },
    {
        "label": "tf_ma120",
        "note": "TF 变体: MA120 (更敏感)",
        "tf_enabled": True, "tf_ma": 120, "tf_bear": 0.5, "tf_bond": "511260",
        "regime_enabled": False,
    },
    {
        "label": "tf_ma200_bear30",
        "note": "TF 变体: 熊市减到 30% (更激进防御)",
        "tf_enabled": True, "tf_ma": 200, "tf_bear": 0.3, "tf_bond": "511260",
        "regime_enabled": False,
    },
    {
        "label": "regime_vol",
        "note": "Regime: 60日波动率高 → 防御组合",
        "tf_enabled": False,
        "regime_enabled": True, "regime_method": "vol", "regime_threshold": 0.20,
        "regime_defense": "bond",
    },
    {
        "label": "regime_trend",
        "note": "Regime: 60日趋势强度 (动量) → 防御组合",
        "tf_enabled": False,
        "regime_enabled": True, "regime_method": "trend", "regime_threshold": 0.0,
        "regime_defense": "bond",
    },
    {
        "label": "regime_combo",
        "note": "Regime: 波动率 + 趋势双指标",
        "tf_enabled": False,
        "regime_enabled": True, "regime_method": "combo", "regime_threshold": 0.5,
        "regime_defense": "bond",
    },
    {
        "label": "tf_regime_combo",
        "note": "TF + Regime 联合",
        "tf_enabled": True, "tf_ma": 200, "tf_bear": 0.5, "tf_bond": "511260",
        "regime_enabled": True, "regime_method": "combo", "regime_threshold": 0.5,
        "regime_defense": "bond",
    },
]

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def compute_metrics(nav: pd.Series, freq: int = DAYS_PER_YEAR) -> dict:
    """计算业绩指标."""
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


def get_trend_filter_signal(weekly_dates, daily_returns, ma=200, benchmark="沪深300指数"):
    """计算 TF 信号: 每周频调仓日, 当日 HS300 < MA200 时返回 True (熊市).

    Returns:
        pd.Series indexed by weekly date, bool (True=bear)
    """
    bench_price = load_benchmark_price(benchmark)
    # 计算 MA
    ma_series = bench_price.rolling(ma).mean()

    # 每个 weekly date 找到对应的 daily date (或最近)
    signals = {}
    for wd in weekly_dates:
        # 找到该 weekly date 当天或之前的最后交易日
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
    return pd.Series(signals, name="bear_signal")


def get_regime_signal(weekly_dates, daily_returns, method="vol", threshold=0.20, lookback=60):
    """计算 Regime 信号: 每周频调仓日, 用 60 日波动率/趋势判断是否熊市.

    method:
        - "vol": 60 日年化波动率 > threshold → 熊市
        - "trend": 60 日动量 < 0 → 熊市
        - "combo": 两个指标综合 (任一触发) → 熊市
    """
    # 用 daily_returns 平均作为市场代表
    market_daily = daily_returns.mean(axis=1)

    if method == "vol":
        # 60 日年化波动率
        vol_60 = market_daily.rolling(lookback).std() * np.sqrt(252)
        signals = {}
        for wd in weekly_dates:
            valid = vol_60.index[vol_60.index <= wd]
            if len(valid) == 0:
                signals[wd] = False
                continue
            latest_date = valid[-1]
            signals[wd] = bool(vol_60.loc[latest_date] > threshold)
        return pd.Series(signals, name=f"regime_vol_{threshold}")

    elif method == "trend":
        # 60 日累计收益 (动量)
        ret_60 = (1 + market_daily).rolling(lookback).apply(np.prod, raw=True) - 1
        signals = {}
        for wd in weekly_dates:
            valid = ret_60.index[ret_60.index <= wd]
            if len(valid) == 0:
                signals[wd] = False
                continue
            latest_date = valid[-1]
            signals[wd] = bool(ret_60.loc[latest_date] < threshold)
        return pd.Series(signals, name=f"regime_trend_{threshold}")

    elif method == "combo":
        # 双指标
        vol_60 = market_daily.rolling(lookback).std() * np.sqrt(252)
        ret_60 = (1 + market_daily).rolling(lookback).apply(np.prod, raw=True) - 1
        signals = {}
        for wd in weekly_dates:
            valid = vol_60.index[vol_60.index <= wd]
            if len(valid) == 0:
                signals[wd] = False
                continue
            latest_date = valid[-1]
            vol_bear = vol_60.loc[latest_date] > threshold
            trend_bear = ret_60.loc[latest_date] < 0
            signals[wd] = bool(vol_bear and trend_bear)
        return pd.Series(signals, name="regime_combo")
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_defensive(weights: dict, signal: bool, defense_code: str, defense_weight: float = 0.5) -> dict:
    """应用防御: signal=True 时, 当前权重 × (1-defense_weight) + defense_weight 给防御资产.

    Args:
        weights: 当前权重 dict {code: w}
        signal: True=bear
        defense_code: 防御资产代码
        defense_weight: 防御权重 (0.5 表示 50%)

    Returns:
        调整后的权重 dict
    """
    if not signal:
        return weights
    new_weights = {}
    for code, w in weights.items():
        new_weights[code] = w * (1 - defense_weight)
    new_weights[defense_code] = new_weights.get(defense_code, 0) + defense_weight
    return new_weights


def construct_portfolio_with_overrides(
    Y, X_panel, beta_path, cfg, daily_returns, combo,
):
    """带 TF/Regime 加固的组合构造."""
    T, N = Y.shape
    nav = pd.Series(1.0, index=Y.index, dtype=float)
    weights_history = []
    prev_weights = {}

    # 预计算 TF / Regime 信号
    weekly_dates = list(Y.index)
    tf_signal = None
    regime_signal = None

    if combo.get("tf_enabled"):
        tf_signal = get_trend_filter_signal(
            weekly_dates, daily_returns, ma=combo.get("tf_ma", 200)
        )
        bear_count = int(tf_signal.sum())
        logging.info("  TF 信号: %d/%d 周为熊市 (%.1f%%)",
                     bear_count, len(tf_signal), bear_count / len(tf_signal) * 100)

    if combo.get("regime_enabled"):
        regime_signal = get_regime_signal(
            weekly_dates, daily_returns,
            method=combo.get("regime_method", "vol"),
            threshold=combo.get("regime_threshold", 0.20),
        )
        bear_count = int(regime_signal.sum())
        logging.info("  Regime 信号: %d/%d 周为熊市 (%.1f%%)",
                     bear_count, len(regime_signal), bear_count / len(regime_signal) * 100)

    defense_code = combo.get("tf_bond", "511260")
    defense_weight = combo.get("tf_bear", 0.5)
    defense_weight_regime = 0.5  # regime 默认防御 50%

    for t in range(1, T):
        # 1. 预测
        beta_prev = beta_path.iloc[t - 1].values
        scores = X_panel[t] @ beta_prev
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

        # 4. 应用 TF (改写)
        date = Y.index[t]
        weights_dict = weights.to_dict()
        if combo.get("tf_enabled") and date in tf_signal.index and tf_signal.loc[date]:
            weights_dict = apply_defensive(weights_dict, True, defense_code, defense_weight)
        elif combo.get("regime_enabled") and date in regime_signal.index and regime_signal.loc[date]:
            weights_dict = apply_defensive(weights_dict, True, defense_code, defense_weight_regime)
        # 联合: 任一触发即防御
        elif combo.get("tf_enabled") and combo.get("regime_enabled"):
            tf_bear = date in tf_signal.index and tf_signal.loc[date]
            reg_bear = date in regime_signal.index and regime_signal.loc[date]
            if tf_bear or reg_bear:
                weights_dict = apply_defensive(weights_dict, True, defense_code, max(defense_weight, defense_weight_regime))

        weights = pd.Series(weights_dict)

        # 存储权重
        for code, w in weights.items():
            weights_history.append({'date': date, 'code': code, 'weight': w})

        # 5. 周收益
        weekly_ret = 0.0
        for code, w in weights.items():
            if code in Y.columns:
                ret = Y[code].iloc[t]
                if pd.notna(ret):
                    weekly_ret += w * ret

        # 6. 成本
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
    """日频 NAV (与原版相同)."""
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
            next_day_idx = all_dates.searchsorted(prev_rebal)
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
    """跑一个 combo: 全段 + OOS + 起点 CV."""
    cfg = V7_6Config(**BASE_PARAMS)

    t0 = time.time()
    # TV-PR
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
    # 构造组合 (带 TF/Regime)
    nav_weekly, weights_df = construct_portfolio_with_overrides(
        Y, X_panel, beta_path, cfg, daily_returns, combo
    )
    # 日频 NAV
    nav_daily = calculate_daily_nav(weights_df, daily_returns, cfg)
    full = compute_metrics(nav_daily)
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    oos = compute_metrics(nav_daily_oos)

    # 起点依赖
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
        nav_weekly_s, weights_df_s = construct_portfolio_with_overrides(
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
    logging.info("v7.6 TF + Regime 加固测试")
    logging.info("=" * 60)

    X_panel, Y, valid_codes = load_v7_6_data()
    daily_returns = load_daily_etf_returns()
    logging.info("  X_panel: %s, Y: %s, daily_returns: %s",
                 X_panel.shape, Y.shape, daily_returns.shape)

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
    out_csv = OUTPUT_DIR / "v7_6_tf_regime_test.csv"
    df.to_csv(out_csv, index=False)

    # 汇总输出
    print("\n" + "=" * 100)
    print("TF + Regime 加固测试结果")
    print("=" * 100)
    cols = ["label", "oos_calmar", "oos_sharpe", "oos_dd", "start_mean", "start_cv", "seconds"]
    print(df[cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("各组合起点 Calmar 分布:")
    print("=" * 100)
    header = "起点".ljust(12) + " | " + " | ".join([s["label"].ljust(18) for s in summaries])
    print(header)
    print("-" * len(header))
    for start in START_POINTS:
        line = start.ljust(12) + " | "
        for s in summaries:
            val = next((c for st, c in s["start_details"] if st == start), 0.0)
            line += f"{val:.4f}".ljust(20) + " | "
        print(line)
    means = "MEAN".ljust(12) + " | "
    for s in summaries:
        means += f"{s['start_mean']:.4f}".ljust(20) + " | "
    print(means)
    cvs = "CV%".ljust(12) + " | "
    for s in summaries:
        cvs += f"{s['start_cv']*100:.1f}%".ljust(20) + " | "
    print(cvs)

    # 判据
    print("\n" + "=" * 100)
    print("判据: 起点 CV% 是否从 46.7% 改进到 ≤25%?")
    print("=" * 100)
    base = next(s for s in summaries if s["label"] == "baseline_topn5")
    base_cv = base["start_cv"]
    print(f"基线 (top_n=5): CV%={base_cv*100:.1f}%")
    print()
    for s in summaries:
        if s["label"] == "baseline_topn5":
            continue
        diff = s["start_cv"] - base_cv
        if s["start_cv"] <= 0.25:
            status = "✅ PASS"
        elif s["start_cv"] < base_cv:
            status = "⬆️ 改善"
        else:
            status = "⬇️ 退化"
        print(f"  {status:12s} {s['label']:20s}: CV%={s['start_cv']*100:5.1f}% (差 {diff*100:+.1f}pp), "
              f"OOS Calmar={s['oos_calmar']:.4f}")

    # Markdown 报告
    lines = [
        "# v7.6 TF + Regime 加固测试报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 背景",
        "",
        "- 当前 v7.6 (top_n=5, rho=2.0) 起点 CV% = 46.7% (FAIL, 阈值 25%)",
        "- 验证两种加固措施:",
        "  1. **Trend Filter (TF)**: 沪深300 < MA → 权重 × 0.5 + 50% 国债 ETF (511260)",
        "  2. **Regime 检测**: 60 日波动率 / 趋势强度 → 防御组合",
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
        lines.append(f"- CV% = {best_cv['start_cv']*100:.1f}% (从 {base_cv*100:.1f}% 改进 {base_cv*100 - best_cv['start_cv']*100:.1f}pp)")
        lines.append(f"- OOS Calmar = {best_cv['oos_calmar']:.4f}")
    else:
        lines.append(f"### ⚠️ 仍 >25% 阈值 (最低 {best_cv['start_cv']*100:.1f}%)")
        lines.append(f"- 最佳 CV%: **{best_cv['label']}** = {best_cv['start_cv']*100:.1f}%")
        lines.append(f"- 最高 OOS Calmar: **{best_oos['label']}** = {best_oos['oos_calmar']:.4f}")
        lines.append(f"- 起点 CV% 改进: {base_cv*100:.1f}% → {best_cv['start_cv']*100:.1f}% "
                     f"({base_cv*100 - best_cv['start_cv']*100:.1f}pp)")

    lines.extend([
        "",
        "### TF vs Regime 对比",
        "",
    ])
    tf_results = [s for s in summaries if s["label"].startswith("tf_") and "regime" not in s["label"]]
    regime_results = [s for s in summaries if s["label"].startswith("regime_")]
    if tf_results:
        best_tf = min(tf_results, key=lambda x: x["start_cv"])
        lines.append(f"- **TF 类最佳**: {best_tf['label']}, CV%={best_tf['start_cv']*100:.1f}%, OOS Calmar={best_tf['oos_calmar']:.4f}")
    if regime_results:
        best_regime = min(regime_results, key=lambda x: x["start_cv"])
        lines.append(f"- **Regime 类最佳**: {best_regime['label']}, CV%={best_regime['start_cv']*100:.1f}%, OOS Calmar={best_regime['oos_calmar']:.4f}")

    report = "\n".join(lines)
    out_md = OUTPUT_DIR / "v7_6_tf_regime_test.md"
    out_md.write_text(report, encoding="utf-8")
    logging.info("=" * 60)
    logging.info("CSV: %s", out_csv)
    logging.info("MD: %s", out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
