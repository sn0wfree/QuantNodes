# coding=utf-8
"""v4 完整回测对比 (Stage 27).

对比 7 个模式:
- v4A: 仅风格轮动
- v4B: 仅 Smart β
- v4C: 风格 + Smart β (无因子择时)
- v4D: + 因子择时 (IC only)
- v4E: + 因子择时 (HMM only)
- v4F: + 因子择时 (IC + HMM 融合)
- v4+IR: 风格 + Smart β + 行业轮动

输出:
- 完整业绩指标对比表
- 年化收益/波动率/Sharpe/Calmar/MaxDD/胜率/换手率
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from QuantNodes.strategy.momentum_etf_rotation.v4 import (
    FactorTimingConfig,
    IndustryRotationConfig,
    V4Config,
    V4Mode,
    run_v4_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    run_backtest,
    compute_metrics,
)


def load_data():
    """加载数据."""
    data_dir = REPO / "data"
    # 加载 smartbeta 数据集 (12 ETF, 用于 v4 回测)
    smartbeta = pd.read_parquet(data_dir / "real" / "etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    # 加载 43 ETF 数据集 (用于 regime 检测)
    etf_43 = pd.read_parquet(data_dir / "high_freq_macro" / "v7_10_Y_weekly.parquet")
    return smartbeta, etf_43


def get_all_configs():
    """获取所有配置."""
    configs = {
        "v4A (风格轮动)": V4Config(
            mode="v4A_style",
            smart_beta_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
        ),
        "v4B (Smart β)": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
        ),
        "v4C (风格+Smart β)": V4Config(
            mode="v4C_combo",
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
        ),
        "v4D (IC 择时)": V4Config(
            mode="v4D_ic",
            factor_timing_enabled=True,
            factor_timing=FactorTimingConfig(hmm_enabled=False),
            industry_rotation_enabled=False,
        ),
        "v4E (HMM 择时)": V4Config(
            mode="v4E_hmm",
            factor_timing_enabled=True,
            factor_timing=FactorTimingConfig(
                hmm_enabled=True,
                hmm_mode="v4E",
            ),
            industry_rotation_enabled=False,
        ),
        "v4F (IC+HMM 融合)": V4Config(
            mode="v4F_fusion",
            factor_timing_enabled=True,
            factor_timing=FactorTimingConfig(
                hmm_enabled=True,
                hmm_mode="v4F",
                hmm_fusion_alpha=0.7,
            ),
            industry_rotation_enabled=False,
        ),
        "v4+IR (行业轮动)": V4Config(
            mode="v4C_combo",
            factor_timing_enabled=False,
            industry_rotation_enabled=True,
            industry_rotation=IndustryRotationConfig(
                top_n=5,
                regime_enabled=True,
                use_value_factor=True,
                use_quality_factor=True,
            ),
        ),
    }
    return configs


def compute_all_metrics(nav: pd.Series, name: str) -> dict:
    """计算完整业绩指标."""
    # 从 NAV 计算收益
    returns = nav.pct_change().fillna(0)
    metrics = compute_metrics(returns, freq="W")
    return metrics


def main():
    """主函数."""
    print("=" * 80)
    print("v4 完整回测对比 (Stage 27)")
    print("=" * 80)

    # 加载数据
    smartbeta, etf_43 = load_data()
    smartbeta_clean = smartbeta.fillna(0).replace([np.inf, -np.inf], 0)
    etf_43_clean = etf_43.fillna(0).replace([np.inf, -np.inf], 0)

    print(f"\nSmartbeta 数据: {smartbeta_clean.shape[0]} 周, {smartbeta_clean.shape[1]} ETF")
    print(f"43 ETF 数据: {etf_43_clean.shape[0]} 周, {etf_43_clean.shape[1]} ETF")
    print(f"时间: {smartbeta_clean.index.min()} ~ {smartbeta_clean.index.max()}")

    # 预计算 regime 序列 (用 43 ETF 数据, 简单规则化)
    from QuantNodes.strategy.momentum_etf_rotation.v4.regime_detector_v4 import detect_regime_simple
    
    etf_43_valid = etf_43_clean.copy()
    etf_43_valid = etf_43_valid.replace(0, np.nan).ffill().fillna(0)
    
    print(f"\n计算 regime (简单规则化, 43 ETF)...")
    regime_series = detect_regime_simple(etf_43_valid, list(etf_43_valid.columns))
    
    # 统计 regime 分布
    regime_counts = regime_series.value_counts()
    print(f"Regime 分布:")
    for r, c in regime_counts.items():
        regime_name = {0: "bull", 1: "bear", 2: "transition"}.get(r, f"unknown({r})")
        print(f"  {regime_name}: {c}")
    
    # 统计 regime 分布
    regime_counts = regime_series.value_counts()
    print(f"Regime 分布:")
    for r, c in regime_counts.items():
        regime_name = {0: "bull", 1: "bear", 2: "transition"}.get(r, f"unknown({r})")
        print(f"  {regime_name}: {c}")

    # 获取所有配置
    configs = get_all_configs()

    # 跑回测
    results = []
    navs = {}

    for name, cfg in configs.items():
        print(f"\n跑 {name}...")
        try:
            # v4 回测用 smartbeta 数据集 (12 ETF)
            panel = smartbeta_clean.copy()

            # 对 v4E/v4F 传入 regime 序列 (用 43 ETF 计算)
            if cfg.factor_timing_enabled and cfg.factor_timing.hmm_enabled:
                v4_result = run_v4_backtest(panel, cfg, hmm_regime_series=regime_series)
            else:
                v4_result = run_v4_backtest(panel, cfg)

            # 计算指标
            metrics = compute_all_metrics(v4_result.nav, name)
            metrics["模式"] = name
            metrics["策略数"] = sum([
                cfg.style_enabled,
                cfg.smart_beta_enabled,
                cfg.industry_rotation_enabled,
            ])

            results.append(metrics)
            navs[name] = v4_result.nav

            print(f"  ✓ 完成: Sharpe={metrics.get('Sharpe', 0):.3f}, "
                  f"Calmar={metrics.get('Calmar', 0):.3f}")

        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("无有效结果")
        return

    # 汇总表格
    df = pd.DataFrame(results)
    df = df.sort_values("Sharpe", ascending=False)

    print("\n" + "=" * 80)
    print("业绩指标对比 (按 Sharpe 排序)")
    print("=" * 80)

    # 选择关键列
    key_cols = [
        "模式", "策略数", "AnnRet", "AnnVol", "Sharpe", "Calmar",
        "MaxDD", "WinRate", "TotalReturn",
    ]
    available_cols = [c for c in key_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    # 保存结果
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "v4_stage27_comparison.csv", index=False)
    print(f"\n结果已保存: {output_dir / 'v4_stage27_comparison.csv'}")

    # 生成对比报告
    report_lines = [
        "# v4 Stage 27 改进回测对比 (43 ETF regime)",
        "",
        f"> 数据: {smartbeta_clean.index.min()} ~ {smartbeta_clean.index.max()} ({smartbeta_clean.shape[0]} 周)",
        f"> ETF: {smartbeta_clean.shape[1]} 个 (smartbeta 数据集)",
        f"> Regime: 43 ETF 数据 (简单规则化)",
        "",
        "## 业绩指标对比",
        "",
        "| 排序 | 模式 | 策略数 | 年化收益 | 年化波动 | Sharpe | Calmar | MaxDD | 胜率 | 总收益 |",
        "|------|------|--------|----------|----------|--------|--------|-------|------|--------|",
    ]

    for i, (_, row) in enumerate(df.iterrows(), 1):
        report_lines.append(
            f"| {i} | {row.get('模式', '')} | {row.get('策略数', 0)} | "
            f"{row.get('AnnRet', 0):.2%} | {row.get('AnnVol', 0):.2%} | "
            f"{row.get('Sharpe', 0):.3f} | {row.get('Calmar', 0):.3f} | "
            f"{row.get('MaxDD', 0):.2%} | {row.get('WinRate', 0):.2%} | "
            f"{row.get('TotalReturn', 0):.2%} |"
        )

    report_lines.extend([
        "",
        "## 改进总结",
        "",
        f"1. **最佳模式**: {df.iloc[0].get('模式', '')} (Sharpe {df.iloc[0].get('Sharpe', 0):.3f})",
    ])

    # 计算 v4F vs v4D 的提升
    vf_sharpe = df[df['模式'].str.contains('v4F')]['Sharpe'].values
    vd_sharpe = df[df['模式'].str.contains('v4D')]['Sharpe'].values
    if len(vf_sharpe) > 0 and len(vd_sharpe) > 0:
        report_lines.append(f"2. **v4F vs v4D**: Sharpe 提升 {vf_sharpe[0] - vd_sharpe[0]:.3f}")

    # 计算 v4+IR vs v4C 的提升
    vir_sharpe = df[df['模式'].str.contains('v4\\+IR')]['Sharpe'].values
    vc_sharpe = df[df['模式'].str.contains('v4C')]['Sharpe'].values
    if len(vir_sharpe) > 0 and len(vc_sharpe) > 0:
        report_lines.append(f"3. **v4+IR vs v4C**: Sharpe 提升 {vir_sharpe[0] - vc_sharpe[0]:.3f}")

    report_lines.extend([
        "",
        "## 文件清单",
        "",
        "- `v4_stage27_comparison.csv`: 完整指标数据",
        "- `v4_stage27_report.md`: 本报告",
    ])

    report_path = output_dir / "v4_stage27_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告已保存: {report_path}")

    print(f"\n{'='*80}")
    print("完成!")


if __name__ == "__main__":
    main()
