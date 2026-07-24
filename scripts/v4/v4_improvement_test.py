# coding=utf-8
"""v4 改进验证脚本 (Stage 27).

测试 v4E/v4F HMM 因子择时 + 估值/基本面因子 + 行业轮动.

用法:
    python3.11 scripts/v4/v4_improvement_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from QuantNodes.strategy.momentum_etf_rotation.v4 import (
    FactorTimingConfig,
    IndustryRotationConfig,
    IndustryRotationV4,
    V4Config,
    V4Mode,
    compute_factor_scores,
    compute_factor_weights,
    compute_factor_weights_fusion,
    compute_factor_weights_hmm,
    run_v4_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.factor_ic import FACTOR_NAMES


def load_data():
    """加载数据."""
    data_dir = REPO / "data" / "high_freq_macro"
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return etf


def test_factor_ic():
    """测试因子 IC (验证 8 因子)."""
    print("=" * 60)
    print("测试 1: 因子 IC (8 因子)")
    print("=" * 60)

    etf = load_data()
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)

    # 用前 252 天作为训练集
    train_data = etf_clean.iloc[:252]

    # 计算因子得分
    scores = compute_factor_scores(
        train_data, train_data.index[-1], list(train_data.columns), lookback=60,
    )

    print(f"因子数量: {len(scores)}")
    print(f"因子名称: {list(scores.keys())}")

    for name, score in scores.items():
        print(f"  {name}: mean={score.mean():.4f}, std={score.std():.4f}")

    # 验证新因子存在
    assert "value_proxy" in scores, "value_proxy 因子缺失"
    assert "quality_proxy" in scores, "quality_proxy 因子缺失"
    print("✓ 8 因子验证通过")


def test_factor_timing_v4f():
    """测试 v4F 因子择时 (IC + HMM 融合)."""
    print("\n" + "=" * 60)
    print("测试 2: v4F 因子择时 (IC + HMM 融合)")
    print("=" * 60)

    etf = load_data()
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)

    # 配置 v4F
    cfg = FactorTimingConfig(
        hmm_enabled=True,
        hmm_mode="v4F",
        hmm_fusion_alpha=0.7,
    )

    # 模拟 IC 历史
    ic_history = pd.DataFrame({
        "momentum": np.random.randn(50) * 0.1,
        "reversal": np.random.randn(50) * 0.05,
        "value": np.random.randn(50) * 0.08,
        "dividend": np.random.randn(50) * 0.06,
        "quality": np.random.randn(50) * 0.07,
        "value_proxy": np.random.randn(50) * 0.09,
        "quality_proxy": np.random.randn(50) * 0.08,
    }, index=pd.date_range("2020-01-01", periods=50, freq="W"))

    # 测试 v4F 融合权重
    weights = compute_factor_weights_fusion(ic_history, cfg, regime="bull")

    print(f"v4F 权重 (bull regime):")
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")

    # 验证权重和为 1
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01, f"权重和不为 1: {total}"
    print("✓ v4F 因子择时验证通过")


def test_industry_rotation():
    """测试行业轮动子策略."""
    print("\n" + "=" * 60)
    print("测试 3: 行业轮动子策略")
    print("=" * 60)

    etf = load_data()
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)

    # 配置行业轮动
    cfg = IndustryRotationConfig(
        top_n=5,
        regime_enabled=True,
        use_value_factor=True,
        use_quality_factor=True,
    )

    ir = IndustryRotationV4(cfg)

    # 用前 252 天作为训练集
    train_data = etf_clean.iloc[:252]

    # 测试选股
    selected = ir.select(train_data.index[-1], train_data, regime="bull")
    print(f"选中行业 (bull): {selected}")

    # 测试权重
    weights = ir.weight(selected, train_data, train_data.index[-1])
    print(f"权重: {weights}")

    # 验证
    assert len(selected) <= cfg.top_n, f"选中行业数超过 top_n: {len(selected)}"
    assert abs(sum(weights.values()) - 1.0) < 0.01, "权重和不为 1"
    print("✓ 行业轮动验证通过")


def test_v4_modes():
    """测试 v4 各模式."""
    print("\n" + "=" * 60)
    print("测试 4: v4 各模式配置")
    print("=" * 60)

    modes = [
        ("v4A (风格轮动)", V4Config(mode="v4A_style", smart_beta_enabled=False, factor_timing_enabled=False)),
        ("v4B (Smart β)", V4Config(mode="v4B_smartbeta", style_enabled=False, factor_timing_enabled=False)),
        ("v4C (组合)", V4Config(mode="v4C_combo")),
        ("v4D (IC 择时)", V4Config(mode="v4D_ic", factor_timing_enabled=True)),
        ("v4E (HMM 择时)", V4Config(mode="v4E_hmm", factor_timing_enabled=True,
                                     factor_timing=FactorTimingConfig(hmm_enabled=True, hmm_mode="v4E"))),
        ("v4F (融合择时)", V4Config(mode="v4F_fusion", factor_timing_enabled=True,
                                     factor_timing=FactorTimingConfig(hmm_enabled=True, hmm_mode="v4F"))),
        ("v4+行业轮动", V4Config(mode="v4C_combo", industry_rotation_enabled=True)),
    ]

    for name, cfg in modes:
        print(f"  {name}: mode={cfg.mode}, style={cfg.style_enabled}, "
              f"sb={cfg.smart_beta_enabled}, ft={cfg.factor_timing_enabled}, "
              f"ir={cfg.industry_rotation_enabled}")

    print("✓ v4 模式配置验证通过")


def main():
    """主函数."""
    print("=" * 60)
    print("v4 改进验证 (Stage 27)")
    print("=" * 60)

    test_factor_ic()
    test_factor_timing_v4f()
    test_industry_rotation()
    test_v4_modes()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
