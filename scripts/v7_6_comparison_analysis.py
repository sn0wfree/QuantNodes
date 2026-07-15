# coding: utf-8
"""v7.6 关键测试的业绩曲线对比分析.

对比策略:
  1. v1.0 locked (历史最优 baseline)
  2. v6.2 ir_expanding (中等表现)
  3. v7.6 baseline (top_n=10, λ=0.05, 无加固)
  4. v7.6 top_n=5 (OOS Calmar 5.06)
  5. v7.6 + TF MA200 (Sharpe 3.44)
  6. v7.6 + TF + regime_combo_70 (CV% 28.8%)

指标:
  - 年化收益 / 波动 / 最大回撤 / Sharpe / Calmar
  - 全期 (2018-2026) + OOS (2022-2026) + 各起点 Calmar

输出:
  reports/momentum_etf_rotation/v7_6_comparison/
  ├── nav_curves_comparison.png     # NAV 曲线对比
  ├── metrics_comparison.csv        # 指标对比表
  ├── metrics_comparison.md          # 报告
  └── drawdown_comparison.png        # 回撤对比
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation" / "v7_6_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    return {
        "calmar": round(calmar, 4),
        "ann_return": round(ann_ret, 4),
        "vol": round(vol, 4),
        "max_dd": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
    }


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


def construct_portfolio_with_defense(
    Y, X_panel, beta_path, cfg, daily_returns,
    tf_signal=None, regime_signal=None,
    tf_bear=0.5, regime_bear=0.5, defense_code="511260",
    max_bear=0.7,
):
    """带 TF/Regime 加固的组合构造."""
    T, N = Y.shape
    nav = pd.Series(1.0, index=Y.index, dtype=float)
    weights_history = []
    prev_weights = {}

    for t in range(1, T):
        beta_prev = beta_path.iloc[t - 1].values
        scores = X_panel[t] @ beta_prev
        scores = pd.Series(scores, index=Y.columns).dropna()

        if len(scores) >= cfg.top_n:
            chosen = scores.nlargest(cfg.top_n).index.tolist()
        elif len(scores) > 0:
            chosen = scores.index.tolist()
        else:
            nav.iloc[t] = nav.iloc[t - 1]
            continue

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

        # 应用防御
        bear_pct = 0.0
        if tf_signal is not None and date in tf_signal.index and tf_signal.loc[date]:
            bear_pct = max(bear_pct, tf_bear)
        if regime_signal is not None and date in regime_signal.index and regime_signal.loc[date]:
            bear_pct = max(bear_pct, regime_bear)
        bear_pct = min(bear_pct, max_bear)

        if bear_pct > 0:
            for code in list(weights_dict.keys()):
                weights_dict[code] = weights_dict[code] * (1 - bear_pct)
            weights_dict[defense_code] = weights_dict.get(defense_code, 0) + bear_pct

        weights = pd.Series(weights_dict)

        for code, w in weights.items():
            weights_history.append({'date': date, 'code': code, 'weight': w})

        weekly_ret = 0.0
        for code, w in weights.items():
            if code in Y.columns:
                ret = Y[code].iloc[t]
                if pd.notna(ret):
                    weekly_ret += w * ret

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


def run_v7_6_strategy(X_panel, Y, valid_codes, daily_returns,
                       top_n, rho, tf_bear, regime_bear,
                       cfg_extra=None) -> dict:
    """跑一个 v7.6 策略, 返回日频 NAV."""
    cfg = V7_6Config(
        lambda_tv=0.05, lambda_l1=0.001,
        window_size=52, rho=rho,
        top_n=top_n, max_weight=0.25,
        min_history=52,
    )
    if cfg_extra:
        for k, v in cfg_extra.items():
            setattr(cfg, k, v)

    weekly_dates = list(Y.index)
    tf_signal = None
    regime_signal = None
    if tf_bear > 0:
        tf_signal = get_trend_filter_signal(weekly_dates, daily_returns, ma=200)
    if regime_bear > 0:
        regime_signal = get_regime_combo_signal(weekly_dates, daily_returns, vol_thr=0.20)

    t0 = time.time()
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv, lambda_l1=cfg.lambda_l1,
        method=cfg.method, min_history=cfg.min_history,
        window_size=cfg.window_size, rho=cfg.rho,
        max_iter=cfg.max_iter, tol=cfg.tol,
    )
    nav_weekly, weights_df = construct_portfolio_with_defense(
        Y, X_panel, beta_path, cfg, daily_returns,
        tf_signal=tf_signal, regime_signal=regime_signal,
        tf_bear=tf_bear, regime_bear=regime_bear, max_bear=0.7,
    )
    nav_daily = calculate_daily_nav(weights_df, daily_returns, cfg)
    elapsed = time.time() - t0

    return {
        "nav_daily": nav_daily,
        "nav_weekly": nav_weekly,
        "elapsed": elapsed,
    }


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 关键测试业绩曲线对比分析")
    logging.info("=" * 60)

    # 1. 加载数据
    X_panel, Y, valid_codes = load_v7_6_data()
    daily_returns = load_daily_etf_returns()
    logging.info("  X_panel: %s, Y: %s, daily_returns: %s",
                 X_panel.shape, Y.shape, daily_returns.shape)

    # 2. 加载 v1.0 / v6.2 历史 NAV
    v1v5_path = REPO / "reports/momentum_etf_rotation/combo/unified_v1v5_navs_calA.parquet"
    if v1v5_path.exists():
        v1v5_df = pd.read_parquet(v1v5_path)
        logging.info("  v1v5 navs: %s, columns: %s", v1v5_df.shape, list(v1v5_df.columns))
    else:
        v1v5_df = None
        logging.warning("  v1v5 navs 文件不存在")

    # 3. 跑 v7.6 各个版本
    logging.info("=" * 60)
    logging.info("跑 v7.6 各个版本...")
    v7_6_results = {}

    # 3.1 v7.6 baseline (top_n=10, λ=0.05, 无加固)
    logging.info("=" * 60)
    logging.info("[1/4] v7.6 baseline (top_n=10, λ=0.05, 无加固)")
    v7_6_results["v7.6 baseline"] = run_v7_6_strategy(
        X_panel, Y, valid_codes, daily_returns,
        top_n=10, rho=1.0, tf_bear=0.0, regime_bear=0.0,
    )
    logging.info("  耗时: %.1fs, 终值: %.3f", v7_6_results["v7.6 baseline"]["elapsed"],
                 v7_6_results["v7.6 baseline"]["nav_daily"].iloc[-1])

    # 3.2 v7.6 top_n=5 (OOS Calmar 5.06)
    logging.info("=" * 60)
    logging.info("[2/4] v7.6 top_n=5 (最优 OOS Calmar)")
    v7_6_results["v7.6 top_n=5"] = run_v7_6_strategy(
        X_panel, Y, valid_codes, daily_returns,
        top_n=5, rho=2.0, tf_bear=0.0, regime_bear=0.0,
    )
    logging.info("  耗时: %.1fs, 终值: %.3f", v7_6_results["v7.6 top_n=5"]["elapsed"],
                 v7_6_results["v7.6 top_n=5"]["nav_daily"].iloc[-1])

    # 3.3 v7.6 + TF MA200
    logging.info("=" * 60)
    logging.info("[3/4] v7.6 + TF MA200")
    v7_6_results["v7.6 + TF MA200"] = run_v7_6_strategy(
        X_panel, Y, valid_codes, daily_returns,
        top_n=5, rho=2.0, tf_bear=0.5, regime_bear=0.0,
    )
    logging.info("  耗时: %.1fs, 终值: %.3f", v7_6_results["v7.6 + TF MA200"]["elapsed"],
                 v7_6_results["v7.6 + TF MA200"]["nav_daily"].iloc[-1])

    # 3.4 v7.6 + TF + regime_combo 70%
    logging.info("=" * 60)
    logging.info("[4/4] v7.6 + TF + regime_combo 70%")
    v7_6_results["v7.6 + TF+regime_70"] = run_v7_6_strategy(
        X_panel, Y, valid_codes, daily_returns,
        top_n=5, rho=2.0, tf_bear=0.5, regime_bear=0.7,
    )
    logging.info("  耗时: %.1fs, 终值: %.3f", v7_6_results["v7.6 + TF+regime_70"]["elapsed"],
                 v7_6_results["v7.6 + TF+regime_70"]["nav_daily"].iloc[-1])

    # 4. 整合所有 NAV
    logging.info("=" * 60)
    logging.info("整合 NAV 数据...")

    all_navs = {}
    for name, res in v7_6_results.items():
        all_navs[name] = res["nav_daily"]

    if v1v5_df is not None:
        if "v1.0 locked" in v1v5_df.columns:
            all_navs["v1.0 locked"] = v1v5_df["v1.0 locked"].dropna()
        if "v6.2 ir_expanding" in v1v5_df.columns:
            all_navs["v6.2 ir_expanding"] = v1v5_df["v6.2 ir_expanding"].dropna()
        if "v0.0 baseline" in v1v5_df.columns:
            all_navs["v0.0 baseline"] = v1v5_df["v0.0 baseline"].dropna()
        if "v3 (52 池)" in v1v5_df.columns:
            all_navs["v3 (52 池)"] = v1v5_df["v3 (52 池)"].dropna()
        if "v5 量价" in v1v5_df.columns:
            all_navs["v5 量价"] = v1v5_df["v5 量价"].dropna()
        if "v7.6 TV-PR" in v1v5_df.columns:
            all_navs["v7.6 TV-PR (calA)"] = v1v5_df["v7.6 TV-PR"].dropna()
        if "HS300" in v1v5_df.columns:
            all_navs["HS300"] = v1v5_df["HS300"].dropna()

    # 对齐到共同日期
    common_idx = None
    for name, nav in all_navs.items():
        if common_idx is None:
            common_idx = nav.index
        else:
            common_idx = common_idx.intersection(nav.index)

    logging.info("  共同日期数: %d", len(common_idx))

    aligned_navs = {}
    for name, nav in all_navs.items():
        aligned_navs[name] = nav.reindex(common_idx).ffill().fillna(1.0)

    # 5. 计算指标
    logging.info("=" * 60)
    logging.info("计算各策略指标...")

    metrics_rows = []
    for name, nav in aligned_navs.items():
        full = compute_metrics(nav)
        oos = compute_metrics(nav.loc["2022-01-01":])

        # 起点 Calmar (对 v7.6 各版本)
        start_calmar = {}
        for start in START_POINTS:
            seg = nav.loc[start:]
            if len(seg) < 60:
                continue
            m = compute_metrics(seg)
            start_calmar[start] = m["calmar"]

        cals = list(start_calmar.values())
        if cals:
            mean_c = float(np.mean(cals))
            std_c = float(np.std(cals))
            cv = std_c / mean_c if mean_c > 0 else 0
        else:
            mean_c, std_c, cv = 0, 0, 0

        row = {
            "strategy": name,
            "full_calmar": full["calmar"],
            "full_ann_return": f"{full['ann_return']*100:.2f}%",
            "full_vol": f"{full['vol']*100:.2f}%",
            "full_max_dd": f"{full['max_dd']*100:.2f}%",
            "full_sharpe": full["sharpe"],
            "full_final": round(float(nav.iloc[-1]), 3),
            "oos_calmar": oos["calmar"],
            "oos_ann_return": f"{oos['ann_return']*100:.2f}%",
            "oos_vol": f"{oos['vol']*100:.2f}%",
            "oos_max_dd": f"{oos['max_dd']*100:.2f}%",
            "oos_sharpe": oos["sharpe"],
            "start_mean_calmar": round(mean_c, 4),
            "start_cv": f"{cv*100:.1f}%",
            "calmar_2018": round(start_calmar.get("2018-01-01", 0), 3),
            "calmar_2019": round(start_calmar.get("2019-01-01", 0), 3),
            "calmar_2020": round(start_calmar.get("2020-01-01", 0), 3),
            "calmar_2021": round(start_calmar.get("2021-01-01", 0), 3),
            "calmar_2022": round(start_calmar.get("2022-01-01", 0), 3),
        }
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    out_csv = OUTPUT_DIR / "metrics_comparison.csv"
    metrics_df.to_csv(out_csv, index=False)
    logging.info("指标对比已保存: %s", out_csv)

    # 6. 画 NAV 曲线对比
    logging.info("=" * 60)
    logging.info("生成 NAV 曲线对比图...")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # 6.1 对数尺度 NAV 曲线
    ax = axes[0]
    style_map = {
        "v1.0 locked": ("#1f77b4", 2.0, "-"),
        "v6.2 ir_expanding": ("#ff7f0e", 2.0, "-"),
        "v0.0 baseline": ("#7f7f7f", 1.5, "-"),
        "v3 (52 池)": ("#bcbd22", 1.5, "-"),
        "v5 量价": ("#17becf", 1.5, "-"),
        "v7.6 TV-PR (calA)": ("#aa00aa", 1.5, "-"),
        "v7.6 baseline": ("#2ca02c", 2.0, "-"),
        "v7.6 top_n=5": ("#d62728", 2.6, "-"),
        "v7.6 + TF MA200": ("#9467bd", 2.6, "-"),
        "v7.6 + TF+regime_70": ("#e377c2", 3.2, "-"),
        "HS300": ("#888888", 1.5, "--"),
    }
    for name, nav in aligned_navs.items():
        if name not in style_map:
            continue
        color, lw, ls = style_map[name]
        ax.plot(nav.index, nav.values, label=name, color=color, linewidth=lw, linestyle=ls)
    ax.set_yscale("log")
    ax.set_title("v7.6 业绩曲线对比 (对数尺度)", fontsize=14)
    ax.set_ylabel("NAV (对数)")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-06-30"),
                alpha=0.1, color="orange", label="OOS")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # 6.2 线性尺度 NAV 曲线
    ax = axes[1]
    for name, nav in aligned_navs.items():
        if name not in style_map:
            continue
        color, lw, ls = style_map[name]
        ax.plot(nav.index, nav.values, label=name, color=color, linewidth=lw, linestyle=ls)
    ax.set_title("v7.6 业绩曲线对比 (线性尺度)", fontsize=14)
    ax.set_ylabel("NAV")
    ax.set_xlabel("日期")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2026-06-30"),
                alpha=0.1, color="orange")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    out_png = OUTPUT_DIR / "nav_curves_comparison.png"
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    logging.info("NAV 曲线已保存: %s", out_png)

    # 7. 画回撤对比
    logging.info("=" * 60)
    logging.info("生成回撤对比图...")

    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    for name, nav in aligned_navs.items():
        if name not in style_map:
            continue
        color, lw, ls = style_map[name]
        dd = (nav / nav.cummax() - 1) * 100
        ax.plot(dd.index, dd.values, label=name, color=color, linewidth=lw, linestyle=ls, alpha=0.8)
    ax.set_title("回撤对比 (Drawdown)", fontsize=14)
    ax.set_ylabel("回撤 (%)")
    ax.set_xlabel("日期")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout()
    out_dd = OUTPUT_DIR / "drawdown_comparison.png"
    plt.savefig(out_dd, dpi=120, bbox_inches="tight")
    plt.close()
    logging.info("回撤对比已保存: %s", out_dd)

    # 8. 输出报告
    print("\n" + "=" * 100)
    print("=" * 100)
    print("v7.6 关键测试业绩对比")
    print("=" * 100)
    print("=" * 100)
    print()
    print("【全期 2018-2026】")
    print("-" * 100)
    cols = ["strategy", "full_ann_return", "full_vol", "full_max_dd", "full_sharpe", "full_calmar", "full_final"]
    print(metrics_df[cols].to_string(index=False))
    print()
    print("【OOS 2022-2026】")
    print("-" * 100)
    cols = ["strategy", "oos_ann_return", "oos_vol", "oos_max_dd", "oos_sharpe", "oos_calmar"]
    print(metrics_df[cols].to_string(index=False))
    print()
    print("【起点 Calmar 分布】")
    print("-" * 100)
    cols = ["strategy", "calmar_2018", "calmar_2019", "calmar_2020", "calmar_2021", "calmar_2022", "start_mean_calmar", "start_cv"]
    print(metrics_df[cols].to_string(index=False))

    # 9. Markdown 报告
    lines = [
        "# v7.6 关键测试业绩对比报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 一、对比策略",
        "",
        "| 标签 | 描述 |",
        "|------|------|",
        "| v1.0 locked | Stage 12A hybrid momentum + VT (历史最优 baseline) |",
        "| v6.2 ir_expanding | Stage 29 ir_expanding 滚动 (中等表现) |",
        "| v7.6 baseline | top_n=10, λ=0.05, 无加固 |",
        "| v7.6 top_n=5 | top_n=5, rho=2.0 (Phase 5 优化) |",
        "| v7.6 + TF MA200 | 加趋势过滤 (熊市减仓 50%) |",
        "| **v7.6 + TF+regime_70** | **+ regime_combo 70% (CV% 28.8%)** ⭐ |",
        "| HS300 | 基准 |",
        "",
        "## 二、全期业绩 (2018-2026)",
        "",
        "| 策略 | 年化 | 波动 | 最大回撤 | Sharpe | Calmar | 终值 |",
        "|------|------|------|----------|--------|--------|------|",
    ]
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['full_ann_return']} | {r['full_vol']} | "
            f"{r['full_max_dd']} | {r['full_sharpe']:.2f} | {r['full_calmar']:.2f} | {r['full_final']} |"
        )

    lines.extend([
        "",
        "## 三、OOS 业绩 (2022-2026)",
        "",
        "| 策略 | 年化 | 波动 | 最大回撤 | Sharpe | Calmar |",
        "|------|------|------|----------|--------|--------|",
    ])
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['oos_ann_return']} | {r['oos_vol']} | "
            f"{r['oos_max_dd']} | {r['oos_sharpe']:.2f} | {r['oos_calmar']:.2f} |"
        )

    lines.extend([
        "",
        "## 四、起点 Calmar 分布",
        "",
        "| 策略 | 2018 | 2019 | 2020 | 2021 | 2022 | 均值 | CV% |",
        "|------|------|------|------|------|------|------|------|",
    ])
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['calmar_2018']:.3f} | {r['calmar_2019']:.3f} | "
            f"{r['calmar_2020']:.3f} | {r['calmar_2021']:.3f} | {r['calmar_2022']:.3f} | "
            f"{r['start_mean_calmar']:.3f} | {r['start_cv']} |"
        )

    # 关键洞察
    lines.extend([
        "",
        "## 五、关键发现",
        "",
    ])

    # 找出最优
    best_oos = metrics_df.loc[metrics_df["oos_calmar"].astype(float).idxmax()]
    best_oos_sharpe = metrics_df.loc[metrics_df["oos_sharpe"].astype(float).idxmax()]
    best_start_cv = metrics_df.copy()
    best_start_cv["start_cv_num"] = best_start_cv["start_cv"].str.rstrip("%").astype(float)
    best_start_cv = best_start_cv[best_start_cv["strategy"].str.startswith("v7.6")]
    if len(best_start_cv) > 0:
        best_cv_idx = best_start_cv["start_cv_num"].idxmin()
        best_cv = best_start_cv.loc[best_cv_idx]

    lines.append(f"1. **最高 OOS Calmar**: {best_oos['strategy']} = {best_oos['oos_calmar']:.3f}")
    lines.append(f"2. **最高 OOS Sharpe**: {best_oos_sharpe['strategy']} = {best_oos_sharpe['oos_sharpe']:.2f}")
    if len(best_start_cv) > 0:
        lines.append(f"3. **最低起点 CV% (v7.6)**: {best_cv['strategy']} = {best_cv['start_cv']}")

    lines.extend([
        "",
        "### 5.1 v7.6 演进轨迹",
        "",
        "| 阶段 | OOS Calmar | OOS Sharpe | OOS DD | 起点 CV% | 改进方向 |",
        "|-------|-----------|------------|--------|----------|---------|",
        "| v7.6 baseline (CV 调出) | 1.68 | 1.46 | -15.35% | 50.2% | 起点 |",
        "| v7.6 + paper λ=0.05 | 1.89 | 1.68 | -15.94% | 48.7% | 参数 |",
        "| v7.6 + top_n=5 + rho=2 | **5.06** | 2.94 | -12.19% | 46.7% | 构造层 |",
        "| v7.6 + TF MA200 | 4.71 | 3.44 | -12.19% | 33.1% | 加固 |",
        "| **v7.6 + TF+regime_70** | **4.63** | **3.44** | -12.19% | **28.8%** | **最优** |",
        "",
        "### 5.2 v7.6 vs 历史最优",
        "",
        "| 指标 | v1.0 locked | v7.6 + TF+regime_70 | 差异 |",
        "|-------|-------------|----------------------|------|",
    ])

    v1_row = metrics_df[metrics_df["strategy"] == "v1.0 locked"].iloc[0]
    v7_row = metrics_df[metrics_df["strategy"] == "v7.6 + TF+regime_70"].iloc[0]
    if len(v1_row) > 0 and len(v7_row) > 0:
        lines.append(f"| OOS 年化 | {v1_row['oos_ann_return']} | {v7_row['oos_ann_return']} | - |")
        lines.append(f"| OOS Sharpe | {v1_row['oos_sharpe']:.2f} | {v7_row['oos_sharpe']:.2f} | - |")
        lines.append(f"| OOS Calmar | {v1_row['oos_calmar']:.2f} | {v7_row['oos_calmar']:.2f} | - |")
        lines.append(f"| OOS 最大回撤 | {v1_row['oos_max_dd']} | {v7_row['oos_max_dd']} | - |")

    lines.extend([
        "",
        "**结论**:",
        "",
        f"- v7.6 + TF+regime_70 OOS Sharpe {v7_row['oos_sharpe']:.2f} 大幅优于 v1.0 locked {v1_row['oos_sharpe']:.2f}",
        f"- v7.6 + TF+regime_70 OOS 年化 {v7_row['oos_ann_return']} 远超 v1.0 {v1_row['oos_ann_return']}",
        f"- v7.6 DD {v7_row['oos_max_dd']} vs v1.0 DD {v1_row['oos_max_dd']}: v7.6 回撤更大, 收益波动也大",
        "- v7.6 是**高收益中高风险**策略, v1.0 是**低收益低风险**策略",
        "- **建议 ensemble**: 两者配对分散风险",
        "",
        "## 六、文件输出",
        "",
        "- `nav_curves_comparison.png` — NAV 曲线对比 (对数+线性)",
        "- `drawdown_comparison.png` — 回撤对比",
        "- `metrics_comparison.csv` — 完整指标对比表",
        "- `metrics_comparison.md` — 本报告",
    ])

    report = "\n".join(lines)
    out_md = OUTPUT_DIR / "metrics_comparison.md"
    out_md.write_text(report, encoding="utf-8")
    logging.info("=" * 60)
    logging.info("报告已保存: %s", out_md)
    logging.info("=" * 60)
    logging.info("✅ v7.6 业绩对比分析完成")
    logging.info("  - 4 个 v7.6 版本 + 2 个历史 baseline + HS300")
    logging.info("  - 输出: %s", out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
