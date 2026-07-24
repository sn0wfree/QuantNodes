# coding=utf-8
"""v4 完整回测对比 (Stage 27 重构: 适配 43 ETF).

对比 7 个模式:
- v4A: 仅大类轮动 (AssetClassRotation)
- v4B: 仅 Smart β 代理 (从行业 ETF 筛选)
- v4C: 大类 + Smart β (无因子择时)
- v4D: + 因子择时 (IC only)
- v4E: + 因子择时 (HMM only)
- v4F: + 因子择时 (IC + HMM 融合)
- v4+IR: 大类 + Smart β + 行业轮动

数据: 43 ETF (周频, 2018-2026)
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
    """加载 43 ETF 数据 (周频)."""
    data_dir = REPO / "data" / "high_freq_macro"
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return etf


def get_all_configs():
    """获取所有配置 (Stage 27 重构 + Stage 28 Smart β 优化权重)."""
    from QuantNodes.strategy.momentum_etf_rotation.v4.smart_beta_v4 import SmartBetaConfig

    configs = {
        "v4A (大类轮动)": V4Config(
            mode="v4A_style",
            smart_beta_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
        ),
        "v4B (Smart β 默认)": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=5,
                proxy_value_weight=0.33,
                proxy_quality_weight=0.33,
                proxy_low_vol_weight=0.34,
                proxy_momentum_weight=0.0,
            ),
        ),
        "v4B+ (Smart β top-5 最优)": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=5,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
            ),
        ),
        "v4B++ (Smart β top-4 黄金点)": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=4,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
            ),
        ),
        "v4B+++ (Stage 29 最优)": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=4,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
                proxy_corr_constraint=True,        # Stage 29: 相关性约束
                proxy_corr_threshold=0.5,         # 阈值 0.5 (最严格)
            ),
        ),
        "v4B+++ corr=0.7": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=4,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
                proxy_corr_constraint=True,
                proxy_corr_threshold=0.7,         # 弱约束
            ),
        ),
        "v4B+++ corr=0.8": V4Config(
            mode="v4B_smartbeta",
            style_enabled=False,
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=4,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
                proxy_corr_constraint=True,
                proxy_corr_threshold=0.8,         # 极弱约束
            ),
        ),
        "v4C (大类+Smart β)": V4Config(
            mode="v4C_combo",
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
        ),
        "v4C+ (大类+Smart β 最优)": V4Config(
            mode="v4C_combo",
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=5,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
            ),
        ),
        "v4C++ (大类+Smart β top-4)": V4Config(
            mode="v4C_combo",
            factor_timing_enabled=False,
            industry_rotation_enabled=False,
            smart_beta=SmartBetaConfig(
                top_n=4,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
            ),
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
            smart_beta=SmartBetaConfig(
                top_n=5,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
            ),
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
            smart_beta=SmartBetaConfig(
                top_n=5,
                proxy_value_weight=0.20,
                proxy_quality_weight=0.30,
                proxy_low_vol_weight=0.20,
                proxy_momentum_weight=0.30,
            ),
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


def compute_all_metrics(nav: pd.Series) -> dict:
    """计算完整业绩指标."""
    returns = nav.pct_change().fillna(0)
    metrics = compute_metrics(returns, freq="W")
    return metrics


def main():
    print("=" * 80)
    print("v4 完整回测对比 (Stage 27 重构: 适配 43 ETF)")
    print("=" * 80)

    # 加载数据
    etf = load_data()
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]

    # Stage 27: v7_10_Y_weekly.parquet 是收益序列, 需要转换为价格序列
    # 价格 = 累计收益 * 100
    etf_price = (1 + etf_clean).cumprod() * 100

    print(f"\n数据: {etf_price.shape[0]} 周, {etf_price.shape[1]} ETF")
    print(f"时间: {etf_price.index.min()} ~ {etf_price.index.max()}")
    print(f"价格范围: {etf_price.min().min():.2f} ~ {etf_price.max().max():.2f}")

    # 预计算 regime 序列 (43 ETF, 简单规则化)
    from QuantNodes.strategy.momentum_etf_rotation.v4.regime_detector_v4 import detect_regime_simple

    # 用价格序列的收益计算 regime
    etf_for_regime = etf_price.replace(0, np.nan).ffill().fillna(0)

    print(f"\n计算 regime (简单规则化, 43 ETF)...")
    regime_series = detect_regime_simple(etf_for_regime, list(etf_for_regime.columns))

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
            # v4 回测用 43 ETF (转换为价格序列)
            panel = etf_price.copy()

            # 对 v4E/v4F 传入 regime 序列
            if cfg.factor_timing_enabled and cfg.factor_timing.hmm_enabled:
                v4_result = run_v4_backtest(panel, cfg, hmm_regime_series=regime_series)
            else:
                v4_result = run_v4_backtest(panel, cfg)

            # 计算指标
            metrics = compute_all_metrics(v4_result.nav)
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

    key_cols = [
        "模式", "策略数", "AnnRet", "AnnVol", "Sharpe", "Calmar",
        "MaxDD", "WinRate", "TotalReturn",
    ]
    available_cols = [c for c in key_cols if c in df.columns]
    print(df[available_cols].to_string(index=False))

    # 保存结果
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / "v4_stage27_43etf_comparison.csv", index=False)
    print(f"\n结果已保存: {output_dir / 'v4_stage27_43etf_comparison.csv'}")

    # 生成报告
    report_lines = [
        "# v4 Stage 27 改进回测对比 (43 ETF)",
        "",
        f"> 数据: {etf_clean.index.min()} ~ {etf_clean.index.max()} ({etf_clean.shape[0]} 周)",
        f"> ETF: {etf_clean.shape[1]} 个 (宽基 6 + 行业 23 + 海外 11 + 黄金 3)",
        f"> Regime: 43 ETF 简单规则化 (bull={regime_counts.get(0, 0)}, "
        f"bear={regime_counts.get(1, 0)}, transition={regime_counts.get(2, 0)})",
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
    ])

    best = df.iloc[0]
    report_lines.append(f"1. **最佳模式**: {best['模式']} (Sharpe {best['Sharpe']:.3f})")

    report_lines.extend([
        "",
        "## 关键发现",
        "",
    ])

    # 分析对比
    for _, row in df.iterrows():
        if row['模式'] == "v4B (Smart β 代理)":
            sb_sharpe = row['Sharpe']
        elif row['模式'] == "v4C (大类+Smart β)":
            c_sharpe = row['Sharpe']
        elif row['模式'] == "v4A (大类轮动)":
            a_sharpe = row['Sharpe']

    if 'a_sharpe' in dir() and 'sb_sharpe' in dir() and 'c_sharpe' in dir():
        report_lines.append(f"- v4A (大类轮动) Sharpe: {a_sharpe:.3f}")
        report_lines.append(f"- v4B (Smart β 代理) Sharpe: {sb_sharpe:.3f}")
        report_lines.append(f"- v4C (大类+Smart β) Sharpe: {c_sharpe:.3f}")

    report_lines.extend([
        "",
        "## 文件清单",
        "",
        "- `v4_stage27_43etf_comparison.csv`: 完整指标数据",
        "- `v4_stage27_43etf_report.md`: 本报告",
    ])

    report_path = output_dir / "v4_stage27_43etf_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告已保存: {report_path}")

    print(f"\n{'='*80}")
    print("完成!")


if __name__ == "__main__":
    main()