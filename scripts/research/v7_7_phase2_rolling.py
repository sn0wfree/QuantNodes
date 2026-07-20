#!/usr/bin/env python3
# coding=utf-8
"""v7.7 Phase 2: 滚动回测 (expanding window).

对 Phase 1 top-5 模型跑 expanding window 滚动回测, 用 construct_portfolio() 构造组合.
对比 v7.10 TV-PR baseline (Calmar 0.671).

用法:
  python scripts/v7_7_phase2_rolling.py
  python scripts/v7_7_phase2_rolling.py --models rf,gbr,lightgbm  # 指定模型
  python scripts/v7_7_phase2_rolling.py --step 4                   # 每4周重训一次
"""
from __future__ import annotations

import sys
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.pycaret_estimator import (
    load_v7_7_data,
    phase2_sklearn_rolling,
    _create_sklearn_model,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_daily_etf_returns,
)


# Phase 1 Top-5 模型
TOP5_MODELS = ["rf", "gbr", "lightgbm", "et", "lasso"]


def run_single_model(
    model_id: str,
    X_panel: np.ndarray,
    Y_rank: np.ndarray,
    Y_weekly: pd.DataFrame,
    min_history: int = 52,
    step: int = 1,
) -> dict:
    """单模型 Phase 2 滚动回测.

    1. 用 phase2_sklearn_rolling 得到预测分数
    2. 用 construct_portfolio() 构造组合
    3. 计算指标
    """
    T, N, K = X_panel.shape

    # 1. 滚动预测
    print(f"  [{model_id}] 滚动预测 (T={T}, N={N}, K={K}, step={step})...")
    t0 = time.time()

    if step == 1:
        scores = phase2_sklearn_rolling(X_panel, Y_rank, model_id, min_history=min_history, verbose=True)
    else:
        # step > 1: 每 step 周重训一次, 中间复用预测
        scores = np.full((T, N), np.nan)
        model = _create_sklearn_model(model_id)

        for t in range(min_history, T, step):
            # 构造训练集 [0, t)
            X_list, y_list = [], []
            for s in range(t):
                for i in range(N):
                    y_val = Y_rank[s, i]
                    if np.isnan(y_val):
                        continue
                    X_list.append(X_panel[s, i, :])
                    y_list.append(y_val)

            if len(X_list) < 100:
                continue

            X_train = np.array(X_list)
            y_train = np.array(y_list)

            # 处理 NaN
            if model_id not in ("lightgbm",):
                train_valid = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
                X_tr = X_train[train_valid]
                y_tr = y_train[train_valid]
            else:
                train_valid = ~np.isnan(y_train)
                X_tr = X_train[train_valid]
                y_tr = y_train[train_valid]

            try:
                model.fit(X_tr, y_tr)
            except Exception as e:
                print(f"  t={t}: fit error: {e}")
                continue

            # 预测 [t, t+step) 的所有周
            for t_pred in range(t, min(t + step, T)):
                X_pred = X_panel[t_pred]
                if model_id not in ("lightgbm",):
                    X_pr = np.nan_to_num(X_pred, nan=0.0)
                else:
                    X_pr = X_pred
                try:
                    scores[t_pred] = model.predict(X_pr)
                except Exception:
                    pass

            if (t - min_history) % 50 == 0:
                print(f"  [{model_id}] t={t}/{T}, train={len(X_list)}")

    elapsed_pred = time.time() - t0
    print(f"  [{model_id}] 预测完成, 耗时 {elapsed_pred:.1f}s")

    # 2. 用预测分数构造组合 (复用 v7.6 construct_portfolio)
    #    scores[t] = 每个资产的预测收益排序, 作为 beta_path 的替代
    #    需要将 scores 包装成类似 beta_path 的格式
    #    实际上 construct_portfolio 用 X @ beta 计算 scores,
    #    这里我们直接用预测分数, 需要绕过 X @ beta 步骤

    # 简化: 直接用 scores 构造权重 (不经过 construct_portfolio)
    nav = pd.Series(1.0, index=Y_weekly.index, dtype=float)
    weights_history = []
    prev_weights = {}

    cost_rate = (5.0 + 5.0) / 10000  # 5bp + 5bp

    for t in range(min_history, T):
        # 本周分数
        week_scores = scores[t]
        if np.all(np.isnan(week_scores)):
            nav.iloc[t] = nav.iloc[t - 1] if t > 0 else 1.0
            continue

        # 选 top-10
        valid_mask = ~np.isnan(week_scores)
        valid_codes = Y_weekly.columns[valid_mask]
        valid_scores = week_scores[valid_mask]

        if len(valid_codes) < 1:
            nav.iloc[t] = nav.iloc[t - 1] if t > 0 else 1.0
            continue

        top_n = min(10, len(valid_codes))
        top_idx = np.argsort(valid_scores)[-top_n:]
        chosen = valid_codes[top_idx].tolist()

        # 逆波动率加权
        if t >= 26:
            vol_window = Y_weekly.iloc[max(0, t-26):t]
            vols = vol_window[chosen].std()
            vols = vols.fillna(0.01).clip(lower=0.01)
            inv_vol = 1.0 / vols
            weights = inv_vol / inv_vol.sum()
            weights = weights.clip(upper=0.25)
            weights = weights / weights.sum()
        else:
            weights = pd.Series(1.0 / len(chosen), index=chosen)

        # 计算本周收益
        weekly_ret = 0.0
        for code in chosen:
            if code in Y_weekly.columns:
                ret = Y_weekly[code].iloc[t]
                if pd.notna(ret):
                    weekly_ret += weights.get(code, 0.0) * ret

        # 交易成本
        turnover = 0.0
        all_codes = set(list(prev_weights.keys()) + list(weights.index))
        for code in all_codes:
            w_old = prev_weights.get(code, 0.0)
            w_new = weights.get(code, 0.0)
            turnover += abs(w_new - w_old)
        weekly_ret -= turnover * cost_rate

        nav.iloc[t] = nav.iloc[t - 1] * (1 + weekly_ret) if t > 0 else 1.0
        prev_weights = weights.to_dict()

    # 3. 计算指标
    nav_oos = nav.iloc[min_history:]
    ret_oos = nav_oos.pct_change().dropna()

    ann_ret = ret_oos.mean() * 52
    ann_vol = ret_oos.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    peak = nav_oos.cummax()
    dd = (nav_oos - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0

    return {
        "model": model_id,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "pred_time": elapsed_pred,
        "n_weeks": len(nav_oos),
    }


def main():
    parser = argparse.ArgumentParser(description="v7.7 Phase 2: 滚动回测")
    parser.add_argument("--models", default="rf,gbr,lightgbm,et,lasso", help="逗号分隔的模型 ID")
    parser.add_argument("--step", type=int, default=1, help="重训频率 (每 N 周)")
    parser.add_argument("--min-history", type=int, default=52, help="最少训练期")
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",")]

    print("=" * 70)
    print("v7.7 Phase 2: 滚动回测 (expanding window)")
    print("=" * 70)
    print(f"模型: {model_ids}")
    print(f"重训频率: 每 {args.step} 周")
    print(f"最少训练期: {args.min_history} 周")
    print()

    # 加载数据
    print("加载数据...")
    X_panel, Y_raw, Y_rank, factor_names = load_v7_7_data()
    # 用 v7.10 的 Y (实际资产代码)
    daily_returns = load_daily_etf_returns()
    Y_weekly = pd.DataFrame(Y_raw, index=range(Y_raw.shape[0]), columns=daily_returns.columns[:Y_raw.shape[1]])

    print(f"  X: {X_panel.shape}, Y: {Y_weekly.shape}")
    print(f"  因子: {len(factor_names)}")
    print()

    # v7.10 TV-PR baseline (用 v7.7 数据, 39 因子)
    print("v7.10 TV-PR baseline (同一数据)...")
    from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
    cfg_v710 = V7_6Config(lambda_tv=0.06, lambda_l1=0.105, stop_loss_threshold=-0.15, stop_loss_cooldown=5)
    # 用 v7.7 的 X_panel (39 因子) 和 Y_weekly
    beta_path = tvpr_estimator(
        Y_weekly, X_panel,
        lambda_tv=cfg_v710.lambda_tv, lambda_l1=cfg_v710.lambda_l1,
        method='admm', min_history=52, rho=1.0, max_iter=200, tol=1e-5,
    )
    nav_v710, wdf_v710 = construct_portfolio(
        Y_weekly, X_panel, beta_path, cfg_v710, return_weights=True
    )

    # v7.10 指标 (周频)
    ret_v710 = nav_v710.pct_change().dropna()
    ar_v710 = ret_v710.mean() * 52
    av_v710 = ret_v710.std() * np.sqrt(52)
    sw_v710 = ar_v710 / av_v710
    pw_v710 = nav_v710.cummax()
    dd_v710 = (nav_v710 - pw_v710) / pw_v710
    mdd_v710 = dd_v710.min()
    cal_v710 = ar_v710 / abs(mdd_v710)

    print(f"  v7.10 TV-PR: 年化={ar_v710*100:+.2f}%, Sharpe={sw_v710:.3f}, DD={mdd_v710*100:.2f}%, Calmar={cal_v710:.3f}")
    print()

    # 运行 Phase 2
    results = []
    for model_id in model_ids:
        print(f"{'='*70}")
        print(f"模型: {model_id}")
        print(f"{'='*70}")
        result = run_single_model(
            model_id, X_panel, Y_rank, Y_weekly,
            min_history=args.min_history, step=args.step,
        )
        results.append(result)
        print(f"  结果: 年化={result['ann_return']*100:+.2f}%, Sharpe={result['sharpe']:.3f}, DD={result['max_dd']*100:.2f}%, Calmar={result['calmar']:.3f}")
        print()

    # 汇总
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"\n{'模型':<12} {'年化':<10} {'Sharpe':<10} {'DD':<10} {'Calmar':<10} {'耗时':<10}")
    print("- * 60")

    # v7.10 baseline
    empty = ""
    print(f"{'v7.10 TV-PR':<12} {ar_v710*100:+.2f}%{empty:<4} {sw_v710:<10.3f} {mdd_v710*100:.2f}%{empty:<4} {cal_v710:<10.3f} {'baseline':<10}")

    for r in results:
        print(f"{r['model']:<12} {r['ann_return']*100:+.2f}%{empty:<4} {r['sharpe']:<10.3f} {r['max_dd']*100:.2f}%{empty:<4} {r['calmar']:<10.3f} {r['pred_time']:.1f}s")

    # 最优模型
    best = max(results, key=lambda x: x['calmar'])
    print(f"\n最优模型: {best['model']}, Calmar={best['calmar']:.3f}")
    if best['calmar'] > cal_v710:
        print(f"  ✅ 超越 v7.10 TV-PR ({cal_v710:.3f})")
    else:
        print(f"  ❌ 未超越 v7.10 TV-PR ({cal_v710:.3f})")

    # 保存报告
    report_path = REPO / "reports" / "momentum_etf_rotation" / "v7_7_phase2_results.md"
    with open(report_path, "w") as f:
        f.write("# v7.7 Phase 2 结果: 滚动回测\n\n")
        f.write(f"> **日期**: {time.strftime('%Y-%m-%d')}\n")
        f.write(f"> **方法**: expanding window, step={args.step}, min_history={args.min_history}\n")
        f.write(f"> **TV-PR baseline**: Calmar {cal_v710:.3f}, Sharpe {sw_v710:.3f}\n\n")

        f.write("## 结果\n\n")
        f.write("| 模型 | 年化收益 | Sharpe | 最大回撤 | Calmar | 耗时 |\n")
        f.write("|------|----------|--------|----------|--------|------|\n")
        f.write(f"| v7.10 TV-PR | {ar_v710*100:+.2f}% | {sw_v710:.3f} | {mdd_v710*100:.2f}% | {cal_v710:.3f} | baseline |\n")
        for r in results:
            f.write(f"| {r['model']} | {r['ann_return']*100:+.2f}% | {r['sharpe']:.3f} | {r['max_dd']*100:.2f}% | {r['calmar']:.3f} | {r['pred_time']:.1f}s |\n")

        f.write(f"\n## 最优模型\n\n")
        f.write(f"**{best['model']}**: Calmar {best['calmar']:.3f}\n")
        if best['calmar'] > cal_v710:
            f.write(f"\n✅ 超越 v7.10 TV-PR ({cal_v710:.3f})\n")
        else:
            f.write(f"\n❌ 未超越 v7.10 TV-PR ({cal_v710:.3f})\n")

    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
