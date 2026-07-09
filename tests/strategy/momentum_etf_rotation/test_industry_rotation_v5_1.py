# coding=utf-8
"""v5.1 行业量价因子 + 逆波动率加权 单元测试.

覆盖:
- inverse_vol_weights_v5_1 函数正确性
- IndustryRotationV5_1SubStrategy 接口
- v5 与 v5.1 隔离 (v5 等权旧实现未受影响)

参考:
- v3 行业轮动测试: test_industry_rotation_v3.py:133 test_weight_inverse_vol
- v5 测试: 暂无 (v5 主体未配 unit test, 本测试仅覆盖 v5.1)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.momentum_etf_rotation.v5 import (
    IndustryRotationV5Config,
    IndustryRotationV5SubStrategy,
)
from QuantNodes.strategy.momentum_etf_rotation.v5_1 import (
    IndustryRotationV5_1Config,
    IndustryRotationV5_1SubStrategy,
    inverse_vol_weights_v5_1,
)


# ============================================================
# 工具函数
# ============================================================
def _make_panel(n_days: int = 252, n_codes: int = 5, seed: int = 42) -> pd.DataFrame:
    """生成有显著差异波动率的价格面板.

    code_0: 低波动 (0.5%/日)
    code_1: 中波动 (1.0%/日)
    code_2: 高波动 (3.0%/日) — 显著高于 code_0
    code_3: 中波动
    code_4: 低波动

    用 252 天确保 21 日窗口内波动率估计稳定.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(end="2026-06-30", periods=n_days)
    sigmas = [0.005, 0.010, 0.030, 0.010, 0.005]
    data = {}
    for i in range(n_codes):
        rets = np.random.randn(n_days) * sigmas[i]
        prices = 100 * np.cumprod(1 + rets)
        data[f"code_{i}"] = prices
    return pd.DataFrame(data, index=dates)


# ============================================================
# inverse_vol_weights_v5_1 函数测试
# ============================================================
class TestInverseVolWeightsV5_1:
    def test_high_vol_gets_low_weight(self):
        """高波动 ETF 权重应低于低波动."""
        panel = _make_panel(n_days=60, n_codes=5)
        codes = ["code_0", "code_2", "code_4"]  # 低/高/低
        as_of = panel.index[-1]
        weights = inverse_vol_weights_v5_1(panel, codes, as_of, vol_window=21)

        assert weights["code_2"] < weights["code_0"], (
            f"高波动 code_2 ({weights['code_2']:.3f}) "
            f"应低于低波动 code_0 ({weights['code_0']:.3f})"
        )
        assert weights["code_2"] < weights["code_4"], (
            f"高波动 code_2 ({weights['code_2']:.3f}) "
            f"应低于低波动 code_4 ({weights['code_4']:.3f})"
        )

    def test_weights_sum_to_one(self):
        """权重和必须为 1."""
        panel = _make_panel(n_days=60, n_codes=5)
        codes = ["code_0", "code_1", "code_2"]
        as_of = panel.index[-1]
        weights = inverse_vol_weights_v5_1(panel, codes, as_of, vol_window=21)
        assert abs(sum(weights.values()) - 1.0) < 1e-6, (
            f"权重和 = {sum(weights.values())}, 期望 1.0"
        )

    def test_all_weights_non_negative(self):
        """所有权重应非负."""
        panel = _make_panel(n_days=60, n_codes=5)
        codes = ["code_0", "code_1", "code_2", "code_3", "code_4"]
        as_of = panel.index[-1]
        weights = inverse_vol_weights_v5_1(panel, codes, as_of, vol_window=21)
        for code, w in weights.items():
            assert w >= 0, f"{code} 权重 = {w} 为负"

    def test_equal_vol_equal_weight(self):
        """等波动率 → 等权."""
        np.random.seed(123)
        n_days = 60
        dates = pd.bdate_range(end="2026-06-30", periods=n_days)
        # 用完全相同波动率构造
        rets = np.random.randn(n_days) * 0.01
        panel = pd.DataFrame({
            "A": 100 * np.cumprod(1 + rets),
            "B": 100 * np.cumprod(1 + rets),  # 几乎同 A
            "C": 100 * np.cumprod(1 + rets),  # 几乎同 A
        }, index=dates)
        weights = inverse_vol_weights_v5_1(panel, ["A", "B", "C"], dates[-1], vol_window=21)
        for w in weights.values():
            assert abs(w - 1 / 3) < 1e-3, f"等波动应等权, 实际 {w}"

    def test_fallback_when_insufficient_data(self):
        """数据不足时回退到等权."""
        panel = _make_panel(n_days=10)  # 太少
        codes = ["code_0", "code_1"]
        as_of = panel.index[-1]
        weights = inverse_vol_weights_v5_1(panel, codes, as_of, vol_window=21)
        # 10 < 22 → 回退等权
        for w in weights.values():
            assert abs(w - 0.5) < 1e-6

    def test_empty_codes_returns_empty(self):
        """空 codes 列表 → 空 dict."""
        panel = _make_panel(n_days=60)
        weights = inverse_vol_weights_v5_1(panel, [], panel.index[-1])
        assert weights == {}

    def test_missing_codes_use_zero(self):
        """codes 在 nav_df 缺失 → 0 权重."""
        panel = _make_panel(n_days=60, n_codes=3)
        codes = ["code_0", "code_99"]  # code_99 不存在
        as_of = panel.index[-1]
        weights = inverse_vol_weights_v5_1(panel, codes, as_of, vol_window=21)
        assert "code_99" in weights
        assert weights["code_99"] == 0.0

    def test_vol_floor_prevents_explosion(self):
        """vol_floor 防止极低波动率导致权重爆炸.

        场景: A 真实波动率 0.0001 (近零), B 0.02
        - 无 floor: A 拿 ~99% (单边)
        - 有 floor: A 权重被压低 (即使不能压到 0.5, 也应 < 无 floor)
        """
        np.random.seed(456)
        n_days = 252
        dates = pd.bdate_range(end="2026-06-30", periods=n_days)
        panel = pd.DataFrame({
            "A": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.0001),
            "B": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.02),
        }, index=dates)

        weights_no_floor = inverse_vol_weights_v5_1(
            panel, ["A", "B"], dates[-1], vol_window=21, vol_floor=0.0,
        )
        weights_floor = inverse_vol_weights_v5_1(
            panel, ["A", "B"], dates[-1], vol_window=21, vol_floor=2e-2,
        )
        # floor 至少应让 A 权重小幅下降
        assert weights_floor["A"] < weights_no_floor["A"], (
            f"vol_floor 应降低 A 权重: floor={weights_floor['A']:.3f} "
            f"vs no_floor={weights_no_floor['A']:.3f}"
        )
        # 和为 1
        assert abs(sum(weights_floor.values()) - 1.0) < 1e-6


# ============================================================
# v5.1 SubStrategy 测试
# ============================================================
class TestV5_1SubStrategy:
    def test_config_defaults(self):
        """默认配置: max_weight=0.30, vol_window=21."""
        cfg = IndustryRotationV5_1Config()
        assert cfg.max_weight == 0.30
        assert cfg.vol_window == 21
        assert cfg.top_n == 5
        assert cfg.min_history == 252
        assert cfg.name == "industry_rotation_v5_1"

    def test_subclass_of_v5(self):
        """v5.1 应继承 v5 (复用 select / 因子逻辑)."""
        cfg = IndustryRotationV5_1Config()
        sub = IndustryRotationV5_1SubStrategy(cfg)
        assert isinstance(sub, IndustryRotationV5SubStrategy)

    def test_weight_inverse_vol(self):
        """weight() 应输出逆波动率, 不是等权."""
        cfg = IndustryRotationV5_1Config(max_weight=1.0)  # 关闭 max_weight
        sub = IndustryRotationV5_1SubStrategy(cfg)
        panel = _make_panel(n_days=252, n_codes=5)
        codes = ["code_0", "code_2", "code_4"]  # 低/高/低
        as_of = panel.index[-1]
        weights = sub.weight(panel, codes, as_of)

        # 验证不是 1/3 等权
        for w in weights.values():
            assert abs(w - 1/3) > 0.01, f"应不是等权, 实际 {w}"

        # 验证高波动 code_2 权重最低
        assert weights["code_2"] == min(weights.values())

        # 验证和为 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weight_respects_max_weight(self):
        """max_weight 上限约束 (3 个标的, 足够 capacity 承接 excess)."""
        cfg = IndustryRotationV5_1Config(max_weight=0.40)
        sub = IndustryRotationV5_1SubStrategy(cfg)

        # 3 个标的: 1 个极低波动 + 2 个高波动
        # 逆波动率下, 极低波动拿 ~98%, 截到 0.40, 剩余 0.58 分给 2 个高波动
        np.random.seed(789)
        n_days = 252
        dates = pd.bdate_range(end="2026-06-30", periods=n_days)
        panel = pd.DataFrame({
            "TINY_VOL": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.001),
            "HIGH_VOL_1": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.05),
            "HIGH_VOL_2": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.05),
        }, index=dates)
        codes = ["TINY_VOL", "HIGH_VOL_1", "HIGH_VOL_2"]
        as_of = panel.index[-1]
        weights = sub.weight(panel, codes, as_of)

        # TINY_VOL 权重应被截到 ≤ 0.40
        assert weights["TINY_VOL"] <= 0.40 + 1e-6
        # 其他 2 个高波动标的有权重
        assert weights["HIGH_VOL_1"] > 0
        assert weights["HIGH_VOL_2"] > 0
        # 归一化和为 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ============================================================
# v5 / v5.1 隔离测试
# ============================================================
class TestV5AndV5_1Isolation:
    def test_v5_config_unchanged(self):
        """v5 旧 Config 未被 v5.1 修改."""
        cfg5 = IndustryRotationV5Config()
        assert cfg5.max_weight == 0.20, "v5 max_weight 应仍为 0.20"
        assert cfg5.name == "industry_rotation_v5"

    def test_v5_substrategy_still_equal_weight(self):
        """v5 weight() 仍是等权, 行为未变."""
        cfg5 = IndustryRotationV5Config()
        sub5 = IndustryRotationV5SubStrategy(cfg5)
        panel = _make_panel(n_days=60, n_codes=5)
        codes = ["code_0", "code_2", "code_4"]
        as_of = panel.index[-1]
        weights = sub5.weight(panel, codes, as_of)

        # v5 应该是 1/3 等权
        for w in weights.values():
            assert abs(w - 1/3) < 1e-6, f"v5 应等权, 实际 {w}"

    def test_v5_and_v5_1_have_different_max_weight(self):
        """v5 max_weight=0.20, v5.1 max_weight=0.30 (设计上独立)."""
        cfg5 = IndustryRotationV5Config()
        cfg51 = IndustryRotationV5_1Config()
        assert cfg5.max_weight != cfg51.max_weight
        assert cfg5.max_weight == 0.20
        assert cfg51.max_weight == 0.30


# ============================================================
# 集成测试: 跑一次完整 v5.1 weight 调用
# ============================================================
def test_v5_1_weight_integration():
    """集成: 全流程, 构造 OHLCV 面板 + v5.1 权重计算."""
    np.random.seed(2024)
    n_days = 252
    dates = pd.bdate_range(end="2026-06-30", periods=n_days)
    # 3 ETF, 波动递增 (差异显著)
    panel = pd.DataFrame({
        "ETF_A": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.005),
        "ETF_B": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.015),
        "ETF_C": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.030),
    }, index=dates)

    cfg = IndustryRotationV5_1Config(max_weight=1.0)  # 关闭 max_weight
    sub = IndustryRotationV5_1SubStrategy(cfg)
    codes = ["ETF_A", "ETF_B", "ETF_C"]
    weights = sub.weight(panel, codes, panel.index[-1])

    # 验证排序: ETF_A (低) > ETF_B (中) > ETF_C (高)
    assert weights["ETF_A"] > weights["ETF_B"] > weights["ETF_C"], (
        f"排序错误: {weights}"
    )
    # 验证和为 1
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # 验证不超 max_weight
    assert all(w <= cfg.max_weight + 1e-6 for w in weights.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
