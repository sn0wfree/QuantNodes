"""
v7.0 核心回测 单元测试 (Stage 30.3).

测试覆盖:
1. V7Config 默认值 + 继承 V6_2Config
2. V7SubStrategy 继承 V6_2SubStrategy
3. _compute_vol_scale 边界 + 正常
4. run_v7_0_backtest 退化模式 (use_regime=False = v6.2)
5. run_v7_0_backtest 5 状态模式产出 NAV
6. PIT 关键: timeline 传入 vs 内部自算结果一致
7. 5 状态 vol_target 缩放范围 (0.3 ~ 2.0)
8. v7.0 DD 优于 v6.2 (vol_target 降低尾部风险)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
PANEL_CLOSE = REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet"
PANEL_OHLCV = REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决 (防御版本), 改用 5 dynamic 方案 (Stage 30.5)")
def test_v7_config_inherits_v62():
    """V7Config 应继承 V6_2Config 全部字段."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config
    cfg = V7Config()
    # 继承字段
    assert hasattr(cfg, "sort_method")
    assert cfg.sort_method == "ir_expanding"  # Stage 29 默认
    assert hasattr(cfg, "use_orthogonal")
    # v7.0 新增字段
    assert hasattr(cfg, "use_regime")
    assert cfg.use_regime is True
    assert hasattr(cfg, "vol_lookback")
    assert hasattr(cfg, "regime_vol_targets")
    assert "recovery" in cfg.regime_vol_targets
    assert cfg.regime_vol_targets["recovery"] == 0.20
    assert cfg.regime_vol_targets["stagflation"] == 0.06


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决, 改用 5 dynamic 方案")
def test_v7_substrategy_inherits_v62():
    """V7SubStrategy 应继承 V6_2SubStrategy."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config, V7SubStrategy
    from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2SubStrategy
    cfg = V7Config()
    sub = V7SubStrategy(cfg)
    assert isinstance(sub, V6_2SubStrategy)
    # 新增字段
    assert hasattr(sub, "current_regime_")
    assert hasattr(sub, "current_vol_scale_")
    assert sub.current_vol_scale_ == 1.0  # 初始值


def test_compute_vol_scale_normal():
    """_compute_vol_scale 正常情况: 范围应在 [min_scale, max_scale] 内."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.industry_rotation_v7 import _compute_vol_scale
    np.random.seed(42)
    # 24% 年化 vol, 252 天
    rets = np.random.randn(252) * 0.24 / np.sqrt(252)
    nav = pd.Series(np.cumprod(1 + rets) + 100)
    scale = _compute_vol_scale(nav, 0.12, nav.index[-1], lookback=60)
    # scale 应在 [0.3, 2.0] 范围内 (60 日 rolling 估计有噪声, 不强制精确)
    assert 0.3 <= scale <= 2.0, f"scale 超出范围: {scale:.3f}"


def test_compute_vol_scale_insufficient_data():
    """数据不足时 scale = 1.0."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.industry_rotation_v7 import _compute_vol_scale
    nav = pd.Series([1.0, 1.01, 1.02])  # 只有 3 个点
    scale = _compute_vol_scale(nav, 0.12, nav.index[-1], lookback=60)
    assert scale == 1.0


def test_compute_vol_scale_clip_range():
    """scale 应被 clip 到 [min_scale, max_scale]."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.industry_rotation_v7 import _compute_vol_scale
    np.random.seed(42)
    # 1% 极低 vol, 目标 30% → scale = 30 (> max_scale=2.0 → clip)
    rets = np.random.randn(252) * 0.01 / np.sqrt(252)
    nav = pd.Series(np.cumprod(1 + rets) + 100)
    scale = _compute_vol_scale(nav, 0.30, nav.index[-1], lookback=60, max_scale=2.0)
    assert scale == 2.0, f"应被 clip 到 2.0, 实际: {scale}"


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决, 改用 5 dynamic 方案")
@pytest.mark.skipif(not PANEL_CLOSE.exists(), reason="real data not available")
def test_v7_backtest_no_regime_degrades_to_v62():
    """use_regime=False 时, v7.0 行为应近似 v6.2 (Calmar 差异 < 0.05)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config, run_v7_0_backtest
    from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest

    panel_close = pd.read_parquet(PANEL_CLOSE)
    panel_ohlcv = pd.read_parquet(PANEL_OHLCV)

    # v6.2
    cfg62 = V6_2Config()
    cfg62.sort_method = "ir_expanding"
    nav62 = run_v6_2_backtest(panel_close, panel_ohlcv, cfg62)

    # v7.0 with use_regime=False
    cfg7 = V7Config()
    cfg7.sort_method = "ir_expanding"
    cfg7.use_regime = False
    nav7 = run_v7_0_backtest(panel_close, panel_ohlcv, cfg7)

    # 两条 NAV 应高度相似
    assert len(nav62) == len(nav7)
    # 最后 NAV 差异 < 5%
    diff = abs(nav62.iloc[-1] - nav7.iloc[-1]) / nav62.iloc[-1]
    assert diff < 0.05, f"差异: {diff*100:.2f}%"


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决, 改用 5 dynamic 方案")
@pytest.mark.skipif(not PANEL_CLOSE.exists(), reason="real data not available")
def test_v7_backtest_with_regime_runs():
    """use_regime=True 时, v7.0 应产出非平凡 NAV."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config, run_v7_0_backtest, build_regime_timeline

    panel_close = pd.read_parquet(PANEL_CLOSE)
    panel_ohlcv = pd.read_parquet(PANEL_OHLCV)

    cfg7 = V7Config()
    cfg7.sort_method = "ir_expanding"
    cfg7.use_regime = True

    timeline = build_regime_timeline(start="2018-06-01", end="2026-06-30")
    nav7 = run_v7_0_backtest(panel_close, panel_ohlcv, cfg7, regime_timeline=timeline)

    # NAV 不应全是 1.0
    assert nav7.nunique() > 10, f"NAV 几乎不变, nunique={nav7.nunique()}"
    # NAV 末值合理 (不爆炸也不归零)
    assert 0.5 < nav7.iloc[-1] < 5.0, f"末值异常: {nav7.iloc[-1]}"


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决, 改用 5 dynamic 方案")
@pytest.mark.skipif(not PANEL_CLOSE.exists(), reason="real data not available")
def test_v7_timeline_vs_internal_consistency():
    """传入 timeline vs 内部自算, 结果应一致 (allow tiny diff)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config, run_v7_0_backtest, build_regime_timeline

    panel_close = pd.read_parquet(PANEL_CLOSE)
    panel_ohlcv = pd.read_parquet(PANEL_OHLCV)

    timeline = build_regime_timeline(start="2018-06-01", end="2026-06-30")

    cfg7 = V7Config()
    cfg7.sort_method = "ir_expanding"

    # 1. 传入 timeline
    nav_with = run_v7_0_backtest(panel_close, panel_ohlcv, cfg7, regime_timeline=timeline)

    # 2. 内部自算 (传 None)
    nav_internal = run_v7_0_backtest(panel_close, panel_ohlcv, cfg7, regime_timeline=None)

    # 末值应高度一致 (HMM 训练相同, 结果应同)
    diff = abs(nav_with.iloc[-1] - nav_internal.iloc[-1]) / nav_with.iloc[-1]
    assert diff < 0.10, f"传 vs 不传 timeline 差异: {diff*100:.2f}%"


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决, 改用 5 dynamic 方案")
@pytest.mark.skipif(not PANEL_CLOSE.exists(), reason="real data not available")
def test_v7_dd_better_than_v62():
    """v7.0 状态感知 vol_target 应降低 DD (vs v6.2 静态权重)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config, run_v7_0_backtest, build_regime_timeline
    from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest

    panel_close = pd.read_parquet(PANEL_CLOSE)
    panel_ohlcv = pd.read_parquet(PANEL_OHLCV)

    cfg62 = V6_2Config()
    cfg62.sort_method = "ir_expanding"
    nav62 = run_v6_2_backtest(panel_close, panel_ohlcv, cfg62)

    cfg7 = V7Config()
    cfg7.sort_method = "ir_expanding"
    timeline = build_regime_timeline(start="2018-06-01", end="2026-06-30")
    nav7 = run_v7_0_backtest(panel_close, panel_ohlcv, cfg7, regime_timeline=timeline)

    # v7.0 DD 应不深于 v6.2 (vol_target 收紧保护)
    dd62 = (nav62 / nav62.cummax() - 1).min()
    dd7 = (nav7 / nav7.cummax() - 1).min()
    assert dd7 >= dd62 - 0.05, \
        f"v7.0 DD={dd7*100:.1f}% 应 ≥ v6.2 DD={dd62*100:.1f}% (允许 5% 容差)"


def test_v7_min_scale_prevents_zero_position():
    """min_scale 防止仓位归零 (即使 target_vol 极低)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.industry_rotation_v7 import _compute_vol_scale
    np.random.seed(42)
    # 50% 极高 vol
    rets = np.random.randn(252) * 0.50 / np.sqrt(252)
    nav = pd.Series(np.cumprod(1 + rets) + 100)
    # target 0.06 (stagflation) → scale = 0.06/0.50 = 0.12, clip 到 0.3
    scale = _compute_vol_scale(nav, 0.06, nav.index[-1], lookback=60, min_scale=0.3)
    assert scale >= 0.3, f"应 clip 到 0.3, 实际: {scale}"


@pytest.mark.skip(reason="Stage 30.3 vol_target API 被用户否决, 改用 5 dynamic 方案")
def test_v7_regime_vol_targets_default():
    """5 状态 vol_target 默认值与 regime_macro 一致."""
    from QuantNodes.strategy.momentum_etf_rotation.v7 import V7Config
    from QuantNodes.strategy.momentum_etf_rotation.v7.regime_macro import REGIME_VOL_TARGETS
    cfg = V7Config()
    for k, v in REGIME_VOL_TARGETS.items():
        assert cfg.regime_vol_targets[k] == v
