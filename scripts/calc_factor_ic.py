#!/usr/bin/env python3.11
"""因子 IC 评估脚本 — v7.6 宏观因子 + 量价因子.

计算三种 IC:
  方法 1: 时序 IC (宏观因子, per asset) - 用对数收益率
  方法 3: 面板 IC (宏观因子, 修正版, 用市场平均收益) - 用对数收益率
  方法 2: 截面 IC (量价因子)

输出:
  reports/momentum_etf_rotation/v7_6_factor_ic_report.md
  reports/momentum_etf_rotation/v7_6_factor_ic_details.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# 添加项目根目录到路径
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_6_data,
    load_weekly_macro_factors,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config


def calc_time_series_ic(
    X_panel: np.ndarray,
    Y: pd.DataFrame,
    factor_idx: int,
    factor_name: str,
    min_obs: int = 52,
    use_log_return: bool = False,
) -> dict:
    """方法 1: 时序 IC (宏观因子, per asset).

    对每个资产 i:
      x_series = X_panel[:, i, factor_idx]  # (T,) 宏观因子时序
      r_series = Y[:, i]                     # (T,) 资产收益时序
      IC_i = spearmanr(x_series[:-1], r_series[1:])

    跨资产聚合:
      IC_mean, IC_std, ICIR, 正IC占比
    """
    T, N, K = X_panel.shape
    ic_list = []

    for i in range(N):
        nav = X_panel[:, i, factor_idx]  # (T,) NAV 值
        r = Y.values[:, i]                # (T,) 资产收益

        if use_log_return:
            # 转换为对数收益率
            x = np.log(nav[1:] / nav[:-1])  # (T-1,)
            r_next = r[1:]                    # (T-1,)
        else:
            x = nav[:-1]  # (T-1,)
            r_next = r[1:]  # (T-1,)

        valid = ~np.isnan(x) & ~np.isnan(r_next)
        if valid.sum() < min_obs:
            continue

        ic, _ = spearmanr(x[valid], r_next[valid])
        ic_list.append(ic)

    if len(ic_list) == 0:
        return {
            'factor_name': factor_name,
            'method': 'time_series',
            'ic_mean': 0,
            'ic_std': 0,
            'icir': 0,
            'ic_positive_ratio': 0,
            'n_assets': 0,
        }

    ic_arr = np.array(ic_list)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr)
    icir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        'factor_name': factor_name,
        'method': 'time_series',
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': np.mean(ic_arr > 0),
        'n_assets': len(ic_list),
    }


def calc_panel_ic(
    X_panel: np.ndarray,
    Y: pd.DataFrame,
    factor_idx: int,
    factor_name: str,
    min_obs: int = 52,
    use_log_return: bool = False,
) -> dict:
    """方法 3: 面板 IC (宏观因子, 修正版).

    用市场平均收益作为因变量:
      market_r[t] = mean(Y[t+1, :])
      x_ts[t] = X_panel[t, 0, factor_idx]  # 所有资产相同，取任意一个

      IC_panel = spearmanr(x_ts[:-1], market_r[1:])
    """
    T, N, K = X_panel.shape

    # 市场平均收益
    market_r = np.nanmean(Y.values[1:], axis=1)  # (T-1,)
    # 宏观因子时序（所有资产相同，取第一个）
    nav = X_panel[:, 0, factor_idx]  # (T,)

    if use_log_return:
        # 转换为对数收益率
        x_ts = np.log(nav[1:] / nav[:-1])  # (T-1,)
    else:
        x_ts = nav[:-1]  # (T-1,)

    valid = ~np.isnan(x_ts) & ~np.isnan(market_r)
    if valid.sum() < min_obs:
        return {
            'factor_name': factor_name,
            'method': 'panel',
            'panel_ic': 0,
            't_stat': 0,
            'p_value': 1,
            'n_obs': 0,
        }

    panel_ic, p_value = spearmanr(x_ts[valid], market_r[valid])
    n_obs = valid.sum()
    # t 统计量
    if abs(panel_ic) < 1:
        t_stat = panel_ic * np.sqrt(n_obs) / np.sqrt(1 - panel_ic**2)
    else:
        t_stat = np.inf

    return {
        'factor_name': factor_name,
        'method': 'panel',
        'panel_ic': panel_ic,
        't_stat': t_stat,
        'p_value': p_value,
        'n_obs': n_obs,
    }


def calc_cross_sectional_ic(
    X_panel: np.ndarray,
    Y: pd.DataFrame,
    factor_idx: int,
    factor_name: str,
    min_assets: int = 10,
) -> dict:
    """方法 2: 截面 IC (量价因子).

    对每个 t:
      x = X_panel[t, :, factor_idx]  # (N,) 因子截面
      r = Y[t+1]                      # (N,) 下期收益
      IC_t = spearmanr(x, r)
    """
    T, N, K = X_panel.shape
    ic_list = []

    for t in range(T - 1):
        x = X_panel[t, :, factor_idx]  # (N,)
        r = Y.values[t + 1]             # (N,)

        valid = ~np.isnan(x) & ~np.isnan(r)
        if valid.sum() < min_assets:
            continue

        ic, _ = spearmanr(x[valid], r[valid])
        ic_list.append(ic)

    if len(ic_list) == 0:
        return {
            'factor_name': factor_name,
            'method': 'cross_sectional',
            'ic_mean': 0,
            'ic_std': 0,
            'icir': 0,
            'ic_positive_ratio': 0,
            'n_periods': 0,
        }

    ic_arr = np.array(ic_list)
    ic_mean = np.mean(ic_arr)
    ic_std = np.std(ic_arr)
    icir = ic_mean / ic_std if ic_std > 0 else 0

    return {
        'factor_name': factor_name,
        'method': 'cross_sectional',
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'ic_positive_ratio': np.mean(ic_arr > 0),
        'n_periods': len(ic_list),
    }


def run_ic_analysis():
    """运行全部因子 IC 分析."""
    print("=" * 60)
    print("v7.6 因子 IC 评估")
    print("=" * 60)

    # 加载数据
    print("\n加载数据...")
    X_panel, Y, valid_codes = load_v7_6_data()
    T, N, K = X_panel.shape
    print(f"  X_panel: ({T}, {N}, {K})")
    print(f"  Y: {Y.shape}")
    print(f"  有效资产数: {len(valid_codes)}")

    # 使用实际数据中的因子名（宏观因子可能有9个而非8个）
    cfg = V7_6Config()
    X_macro = load_weekly_macro_factors()
    macro_col_names = list(X_macro.columns)
    pv_factor_names = list(cfg.pv_factors)
    factor_names = macro_col_names + pv_factor_names
    K_macro = len(macro_col_names)

    # ============================================================
    # 1. 宏观因子 IC (用对数收益率)
    # ============================================================
    print("\n" + "=" * 60)
    print("宏观因子 IC 分析 (用对数收益率)")
    print("=" * 60)

    macro_results = []
    for k in range(K_macro):
        fname = factor_names[k]

        # 方法 1: 时序 IC (用对数收益率)
        ts_result = calc_time_series_ic(X_panel, Y, k, fname, use_log_return=True)

        # 方法 3: 面板 IC (用对数收益率)
        panel_result = calc_panel_ic(X_panel, Y, k, fname, use_log_return=True)

        # 综合评分
        composite = 0.6 * abs(ts_result['ic_mean']) + 0.4 * abs(panel_result['panel_ic'])

        macro_results.append({
            'factor_name': fname,
            'ts_ic_mean': ts_result['ic_mean'],
            'ts_ic_std': ts_result['ic_std'],
            'ts_icir': ts_result['icir'],
            'ts_ic_positive_ratio': ts_result['ic_positive_ratio'],
            'panel_ic': panel_result['panel_ic'],
            'panel_t_stat': panel_result['t_stat'],
            'panel_p_value': panel_result['p_value'],
            'composite_score': composite,
        })

    # 按综合分排序
    macro_results.sort(key=lambda x: x['composite_score'], reverse=True)

    # 打印结果
    print("\n方法 1: 时序 IC (per asset, 对数收益率)")
    print("-" * 75)
    print(f"{'因子名称':<20} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'正IC占比':>10}")
    print("-" * 75)
    for r in macro_results:
        print(f"{r['factor_name']:<20} {r['ts_ic_mean']:>10.4f} {r['ts_ic_std']:>10.4f} "
              f"{r['ts_icir']:>10.2f} {r['ts_ic_positive_ratio']:>10.1%}")

    print("\n方法 3: 面板 IC (市场平均收益, 对数收益率)")
    print("-" * 75)
    print(f"{'因子名称':<20} {'panel_IC':>10} {'t_stat':>10} {'p_value':>10} {'显著性':>10}")
    print("-" * 75)
    for r in macro_results:
        sig = '***' if r['panel_p_value'] < 0.01 else '**' if r['panel_p_value'] < 0.05 else '*' if r['panel_p_value'] < 0.1 else ''
        print(f"{r['factor_name']:<20} {r['panel_ic']:>10.4f} {r['panel_t_stat']:>10.2f} "
              f"{r['panel_p_value']:>10.4f} {sig:>10}")

    print("\n综合排序 (时序IC×0.6 + 面板IC×0.4)")
    print("-" * 75)
    print(f"{'排名':>4} {'因子名称':<20} {'综合分':>10} {'建议':<15}")
    print("-" * 75)
    for i, r in enumerate(macro_results):
        if r['composite_score'] > 0.03:
            advice = '保留'
        elif r['composite_score'] > 0.02:
            advice = '边缘，观察'
        else:
            advice = '考虑剔除'
        print(f"{i+1:>4} {r['factor_name']:<20} {r['composite_score']:>10.4f} {advice:<15}")

    # ============================================================
    # 2. 量价因子 IC
    # ============================================================
    print("\n" + "=" * 60)
    print("量价因子 IC 分析")
    print("=" * 60)

    pv_results = []
    for k in range(K_macro, K):
        fname = factor_names[k]
        result = calc_cross_sectional_ic(X_panel, Y, k, fname)
        pv_results.append(result)

    # 按 IC_mean 排序
    pv_results.sort(key=lambda x: abs(x['ic_mean']), reverse=True)

    print("\n方法 2: 截面 IC")
    print("-" * 75)
    print(f"{'因子名称':<20} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'正IC占比':>10} {'期数':>8}")
    print("-" * 75)
    for r in pv_results:
        print(f"{r['factor_name']:<20} {r['ic_mean']:>10.4f} {r['ic_std']:>10.4f} "
              f"{r['icir']:>10.2f} {r['ic_positive_ratio']:>10.1%} {r['n_periods']:>8}")

    # ============================================================
    # 3. 保存结果
    # ============================================================
    report_dir = REPO / "reports" / "momentum_etf_rotation"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 保存 CSV
    all_results = []
    for r in macro_results:
        all_results.append({
            'factor_name': r['factor_name'],
            'factor_type': 'macro',
            'ic_mean': r['ts_ic_mean'],
            'ic_std': r['ts_ic_std'],
            'icir': r['ts_icir'],
            'ic_positive_ratio': r['ts_ic_positive_ratio'],
            'panel_ic': r['panel_ic'],
            'panel_t_stat': r['panel_t_stat'],
            'composite_score': r['composite_score'],
        })
    for r in pv_results:
        all_results.append({
            'factor_name': r['factor_name'],
            'factor_type': 'pv',
            'ic_mean': r['ic_mean'],
            'ic_std': r['ic_std'],
            'icir': r['icir'],
            'ic_positive_ratio': r['ic_positive_ratio'],
            'panel_ic': np.nan,
            'panel_t_stat': np.nan,
            'composite_score': abs(r['ic_mean']),
        })

    df = pd.DataFrame(all_results)
    csv_path = report_dir / "v7_6_factor_ic_details.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV 已保存: {csv_path}")

    # 保存 Markdown 报告
    md_path = report_dir / "v7_6_factor_ic_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# v7.6 因子 IC 评估报告\n\n")
        f.write(f"> 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write("## 数据概况\n\n")
        f.write(f"- 时间范围: {Y.index[0]} ~ {Y.index[-1]}\n")
        f.write(f"- 周频数据: {T} 期\n")
        f.write(f"- 资产数量: {N} 个\n")
        f.write(f"- 因子数量: {K} 个 (宏观 {K_macro} + 量价 {K - K_macro})\n\n")

        f.write("## 宏观因子 IC (用对数收益率)\n\n")
        f.write("### 方法 1: 时序 IC (per asset)\n\n")
        f.write("| 因子名称 | IC_mean | IC_std | ICIR | 正IC占比 |\n")
        f.write("|----------|---------|--------|------|----------|\n")
        for r in macro_results:
            f.write(f"| {r['factor_name']} | {r['ts_ic_mean']:.4f} | {r['ts_ic_std']:.4f} | "
                    f"{r['ts_icir']:.2f} | {r['ts_ic_positive_ratio']:.1%} |\n")

        f.write("\n### 方法 3: 面板 IC (市场平均收益)\n\n")
        f.write("| 因子名称 | panel_IC | t_stat | p_value | 显著性 |\n")
        f.write("|----------|----------|--------|---------|--------|\n")
        for r in macro_results:
            sig = '***' if r['panel_p_value'] < 0.01 else '**' if r['panel_p_value'] < 0.05 else '*' if r['panel_p_value'] < 0.1 else ''
            f.write(f"| {r['factor_name']} | {r['panel_ic']:.4f} | {r['panel_t_stat']:.2f} | "
                    f"{r['panel_p_value']:.4f} | {sig} |\n")

        f.write("\n### 综合排序\n\n")
        f.write("| 排名 | 因子名称 | 综合分 | 建议 |\n")
        f.write("|------|----------|--------|------|\n")
        for i, r in enumerate(macro_results):
            if r['composite_score'] > 0.03:
                advice = '保留'
            elif r['composite_score'] > 0.02:
                advice = '边缘，观察'
            else:
                advice = '考虑剔除'
            f.write(f"| {i+1} | {r['factor_name']} | {r['composite_score']:.4f} | {advice} |\n")

        f.write("\n## 量价因子 IC\n\n")
        f.write("| 因子名称 | IC_mean | IC_std | ICIR | 正IC占比 | 期数 |\n")
        f.write("|----------|---------|--------|------|----------|------|\n")
        for r in pv_results:
            f.write(f"| {r['factor_name']} | {r['ic_mean']:.4f} | {r['ic_std']:.4f} | "
                    f"{r['icir']:.2f} | {r['ic_positive_ratio']:.1%} | {r['n_periods']} |\n")

        f.write("\n## 结论\n\n")
        f.write("### 宏观因子\n\n")
        effective_macro = [r for r in macro_results if r['composite_score'] > 0.03]
        marginal_macro = [r for r in macro_results if 0.02 < r['composite_score'] <= 0.03]
        ineffective_macro = [r for r in macro_results if r['composite_score'] <= 0.02]
        f.write(f"- 有效 (综合分 > 0.03): {', '.join(r['factor_name'] for r in effective_macro) or '无'}\n")
        f.write(f"- 边缘 (0.02-0.03): {', '.join(r['factor_name'] for r in marginal_macro) or '无'}\n")
        f.write(f"- 无效 (< 0.02): {', '.join(r['factor_name'] for r in ineffective_macro) or '无'}\n")

        f.write("\n### 量价因子\n\n")
        effective_pv = [r for r in pv_results if abs(r['ic_mean']) > 0.03]
        marginal_pv = [r for r in pv_results if 0.02 < abs(r['ic_mean']) <= 0.03]
        ineffective_pv = [r for r in pv_results if abs(r['ic_mean']) <= 0.02]
        f.write(f"- 有效 (|IC| > 0.03): {', '.join(r['factor_name'] for r in effective_pv) or '无'}\n")
        f.write(f"- 边缘 (0.02-0.03): {', '.join(r['factor_name'] for r in marginal_pv) or '无'}\n")
        f.write(f"- 无效 (< 0.02): {', '.join(r['factor_name'] for r in ineffective_pv) or '无'}\n")

    print(f"报告已保存: {md_path}")
    print("\n完成!")


if __name__ == "__main__":
    run_ic_analysis()
