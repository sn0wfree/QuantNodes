# coding=utf-8
"""scripts/v9/v9_cpd_diagnose.py — v9 CPD 周期诊断主入口.

用法:
    python3.11 scripts/v9/v9_cpd_diagnose.py
    
功能:
    1. 加载宏观数据 + VIX
    2. 运行美林时钟识别
    3. 运行 Pring 周期定位
    4. (可选) VMD 多周期分解
    5. 综合诊断, 输出 Markdown 报告 + HTML 仪表盘
    6. 回填 docs/50-v9_current_cycle_state.md

输出:
    reports/momentum_etf_rotation/v9/current_cycle_state.md
    reports/momentum_etf_rotation/v9/dashboard.html
    docs/50-v9_current_cycle_state.md  (回填)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v9.cpd import (
    detect_merrill_phase_with_confidence,
    diagnose_current_state,
    generate_markdown_report,
    generate_html_dashboard,
)


def load_data():
    """加载所有需要的数据."""
    data_dir = REPO / "data" / "high_freq_macro"

    print("Loading data...")
    x_panel = np.load(data_dir / "v7_14_X_panel.npy")
    y_weekly = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    vix_daily = pd.read_parquet(data_dir / "macro_vix_daily.parquet")["vix"]

    factor_names = pd.read_csv(data_dir / "v7_14_factor_names.csv", header=None).iloc[:, 0].tolist()
    print(f"  - X panel: {x_panel.shape}")
    print(f"  - Y weekly: {y_weekly.shape}")
    print(f"  - VIX daily: {vix_daily.shape}")

    growth_factor = pd.Series(x_panel[:, :, 1].mean(axis=1), index=y_weekly.index)
    cpi_factor = pd.Series(x_panel[:, :, 2].mean(axis=1), index=y_weekly.index)
    ppi_factor = pd.Series(x_panel[:, :, 3].mean(axis=1), index=y_weekly.index)

    vix_weekly = vix_daily.resample("W").last().reindex(y_weekly.index, method="ffill")

    hs300 = y_weekly.mean(axis=1).fillna(0)
    nav_hs300 = (1 + hs300).cumprod()
    log_hs300 = np.log(nav_hs300)

    return {
        "growth_factor": growth_factor,
        "cpi_factor": cpi_factor,
        "ppi_factor": ppi_factor,
        "vix_weekly": vix_weekly,
        "hs300": hs300,
        "log_hs300": log_hs300,
        "y_weekly_index": y_weekly.index,
        "factor_names": factor_names,
    }


def hp_filter(series: pd.Series, lamb: float = 100) -> tuple:
    """HP 滤波 (statsmodels).

    返回:
        (cycle, trend)
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    cycle, trend = hpfilter(series.dropna(), lamb=lamb)
    return cycle, trend


def vmd_decompose(signal: np.ndarray, K: int = 4, alpha: float = 1000) -> tuple:
    """VMD 多尺度分解."""
    from vmdpy import VMD

    u, _, omega = VMD(signal, alpha, 0, K, 0, 1, 1e-6)
    return u, omega


def main():
    data = load_data()

    print("\n=== Step 1: HP 滤波 ===")
    cycle_hs300, trend_hs300 = hp_filter(data["log_hs300"], lamb=100)
    print(f"  HP cycle shape: {cycle_hs300.shape}")
    print(f"  Trend start: {trend_hs300.iloc[0]:.3f}, end: {trend_hs300.iloc[-1]:.3f}")

    print("\n=== Step 2: VMD 多周期分解 ===")
    try:
        imfs, omega = vmd_decompose(cycle_hs300.values, K=4, alpha=1000)
        print(f"  VMD 成功: IMFs shape = {imfs.shape}")
        print(f"  Center frequencies: {omega[-1, :]}")
    except Exception as e:
        print(f"  VMD 失败: {e}")
        print("  使用 fallback: 直接用 HP 残差作为单 IMF")
        imfs = cycle_hs300.values.reshape(1, -1)
        omega = np.array([[0.5]])

    imf_dates = cycle_hs300.index

    print("\n=== Step 3: 美林时钟识别 ===")
    merrill_df = detect_merrill_phase_with_confidence(
        data["growth_factor"].dropna(),
        data["cpi_factor"].dropna(),
        smooth_window=6,
        threshold_window=36,
    )
    print(f"  美林阶段时序长度: {len(merrill_df)}")
    print(f"  各阶段计数: {merrill_df['phase_name'].value_counts().to_dict()}")
    print(f"  最新阶段: {merrill_df['phase_name'].iloc[-1]} ({merrill_df['phase_name_cn'].iloc[-1]})")
    print(f"  最新置信度: {merrill_df['confidence'].iloc[-1]:.2%}")

    print("\n=== Step 4: 综合诊断 ===")
    state = diagnose_current_state(
        growth_series=data["growth_factor"].dropna(),
        inflation_series=data["cpi_factor"].dropna(),
        vix_series=data["vix_weekly"].dropna(),
        imfs=imfs,
        locked_pairs=0,
        bic_max=0.0,
        data_through=cycle_hs300.index[-1],
    )

    print(f"  美林阶段: {state.merrill_phase} ({state.merrill_phase_cn})")
    print(f"  Pring 位置: 第 {state.pring_position} 年 ({state.pring_seasonality_cn})")
    print(f"  多周期综合: {state.composite_phase}")
    print(f"  周期趋势分: {state.cycle_score:.1f}")
    print(f"  周期耦合分: {state.coupling_score:.1f}")
    print(f"  VIX 分: {state.vix_score:.1f}")
    print(f"  总分: {state.total_score:.1f}")
    print(f"  大盘信号: {state.v9_signal} ({state.signal_label})")

    print("\n=== Step 5: 生成 Markdown 报告 ===")
    md_report = generate_markdown_report(state)

    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "current_cycle_state.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"  Markdown 报告已写入: {md_path}")

    docs_50_path = REPO / "docs" / "50-v9_current_cycle_state.md"
    docs_50_path.write_text(md_report, encoding="utf-8")
    print(f"  docs/50 已回填: {docs_50_path}")

    print("\n=== Step 6: 生成 HTML 仪表盘 ===")
    html_path = output_dir / "dashboard.html"
    generate_html_dashboard(
        state=state,
        output_path=html_path,
        imfs_history=imfs,
        phase_history=merrill_df["phase"],
        hs300_history=data["hs300"],
        imf_dates=imf_dates,
    )
    print(f"  HTML 仪表盘已生成: {html_path}")

    print("\n=== 完成 ===")
    print(f"  输出文件:")
    print(f"    - {md_path}")
    print(f"    - {html_path}")
    print(f"    - {docs_50_path}")

    return state


if __name__ == "__main__":
    main()