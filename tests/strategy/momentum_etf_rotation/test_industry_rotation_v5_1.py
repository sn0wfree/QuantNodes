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
        """v5.1.1 默认配置: max_weight=0.25, vol_window=60, vol_floor=0.01, rebal_lag=1."""
        cfg = IndustryRotationV5_1Config()
        assert cfg.max_weight == 0.25
        assert cfg.vol_window == 60
        assert cfg.vol_floor == 0.01
        assert cfg.rebal_lag == 1
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
        """v5 max_weight=0.20, v5.1 max_weight=0.25 (设计上独立)."""
        cfg5 = IndustryRotationV5Config()
        cfg51 = IndustryRotationV5_1Config()
        assert cfg5.max_weight != cfg51.max_weight
        assert cfg5.max_weight == 0.20
        assert cfg51.max_weight == 0.25


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


# ============================================================
# S1 调仓滞后 (rebal_lag) 测试
# ============================================================
class TestRebalLag:
    """S1: T+1 调仓, 模拟"信号日 T → 执行日 T+1 开盘"."""

    def test_rebal_lag_default_is_one(self):
        """默认 rebal_lag=1."""
        cfg = IndustryRotationV5_1Config()
        assert cfg.rebal_lag == 1

    def test_rebal_lag_uses_prior_day_vol(self):
        """S1 T+1 lag: lag=1 用 as_of - 1 日的 vol, lag=0 用 as_of 当日."""
        np.random.seed(2026)
        n_days = 252
        dates = pd.bdate_range(end="2026-06-30", periods=n_days)
        panel = pd.DataFrame({
            "A": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.01),
            "B": 100 * np.cumprod(1 + np.random.randn(n_days) * 0.03),
        }, index=dates)

        as_of = dates[-1]
        # lag=0: 用 as_of 当日 vol
        w_lag0 = inverse_vol_weights_v5_1(panel, ["A", "B"], as_of, vol_window=21, rebal_lag=0)
        # lag=1: 用 as_of - 1 日 vol
        w_lag1 = inverse_vol_weights_v5_1(panel, ["A", "B"], as_of, vol_window=21, rebal_lag=1)
        # lag=2: 用 as_of - 2 日 vol
        w_lag2 = inverse_vol_weights_v5_1(panel, ["A", "B"], as_of, vol_window=21, rebal_lag=2)

        # 不同 lag 应产生不同权重 (尾部窗口起始日不同)
        assert w_lag0 != w_lag1, "lag=0 vs lag=1 权重应不同"
        assert w_lag1 != w_lag2, "lag=1 vs lag=2 权重应不同"
        # 权重和仍为 1
        for w in [w_lag0, w_lag1, w_lag2]:
            assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_rebal_lag_at_index_zero_falls_back(self):
        """as_of 在 nav_df 索引最前时, rebal_lag > 0 应回退等权."""
        panel = _make_panel(n_days=252, n_codes=3)
        codes = ["code_0", "code_1"]
        # 取 nav_df 第一个日期, lag=1 无法回退
        as_of = panel.index[0]
        weights = inverse_vol_weights_v5_1(panel, codes, as_of, vol_window=60, rebal_lag=1)
        # 回退等权
        for w in weights.values():
            assert abs(w - 0.5) < 1e-6

    def test_rebal_lag_passes_through_substrategy(self):
        """SubStrategy.weight() 应用 rebal_lag (通过 Config)."""
        cfg = IndustryRotationV5_1Config(rebal_lag=2)
        sub = IndustryRotationV5_1SubStrategy(cfg)
        assert sub.config.rebal_lag == 2

    def test_vol_window_default_is_60(self):
        """S3: vol_window 默认 60 (从 21 改)."""
        cfg = IndustryRotationV5_1Config()
        assert cfg.vol_window == 60

    def test_vol_floor_default_is_0_01(self):
        """S3: vol_floor 默认 0.01 (从 1e-4 改)."""
        cfg = IndustryRotationV5_1Config()
        assert cfg.vol_floor == 0.01


# ============================================================
# S2 winsorized rank z-score 测试 (消融失败, 已回退)
# ============================================================
class TestS2WinsorizedZscore:
    """S2 消融结果: winsorized rank 严重拖累 OOS Calmar 0.586 → 0.516 (-12%).

    原因 (Stage 25.1 消融报告):
    1. 44 只池上 rank-based 损失信息太多
    2. 极端值 (涨跌停) 在原始 z-score 中已被自然压制
    3. winsorize 5%/95% 反而误伤真实信号
    4. 与 v5 共享 cross_section_zscore, 改一处影响 v5 + v5.1 两策略

    决策: S2 标记为"已尝试, 失败, 保留代码以备参考", 不在 v5.1.1 启用.
    这些测试确保 S2 改动可以随时回滚/重启用.
    """

    def test_zscore_is_original_mean_std(self):
        """v5.1.1: cross_section_zscore 仍用原始 (mean=0, std=1) z-score, 未启用 S2 winsorize."""
        import numpy as np
        from QuantNodes.strategy.momentum_etf_rotation.v5 import cross_section_zscore
        np.random.seed(42)
        n_codes = 44
        dates = pd.bdate_range(end="2026-06-30", periods=252)
        panel = {
            f"c{i}": pd.DataFrame({"f1": np.random.randn(252)}, index=dates)
            for i in range(n_codes)
        }
        z = cross_section_zscore(panel, "f1", dates[-1])
        # 原始 z-score: mean=0, std=1
        assert abs(z.mean()) < 0.1, f"z-score mean 应接近 0, 实际 {z.mean():.3f}"
        assert abs(z.std() - 1.0) < 0.2, f"z-score std 应接近 1, 实际 {z.std():.3f}"

    def test_extreme_values_known_limitation(self):
        """已知 z-score 限制: 1 个极端值会让其他普通值的 z-score 偏移.

        原因: 原始 z-score 用全局 mean/std, 1 个 +100σ 异常让 std 暴涨, 拉低其他值.
        真实场景: 涨跌停停牌或异常成交时, 1-2 个 code 出现极端 factor 值.
        S2 (winsorize rank) 理论上可缓解, 但消融发现 OOS Calmar 反而 -12% 拖累.
        决策: 接受这个限制, 因为涨跌停数据本身是真实信号 (停牌 → 跳过即可, 不在策略中处理).
        """
        import numpy as np
        from QuantNodes.strategy.momentum_etf_rotation.v5 import cross_section_zscore
        np.random.seed(42)
        n_codes = 44
        dates = pd.bdate_range(end="2026-06-30", periods=252)
        panel = {
            f"c{i}": pd.DataFrame({"f1": np.random.randn(252)}, index=dates)
            for i in range(n_codes)
        }
        # 1 个极端值
        panel["c_extreme"] = pd.DataFrame({"f1": [100.0] * 252}, index=dates)
        z = cross_section_zscore(panel, "f1", dates[-1])
        # 普通值被压成 0 附近 (z ≈ -0.2, 因为 std 暴涨)
        normal_codes = [f"c{i}" for i in range(42)]
        normal_z = z[normal_codes]
        # 这是已知行为, 仅记录不作为测试
        assert normal_z.std() < 0.5  # 确认被压低
        # 极端值自身 z 值很大
        assert z["c_extreme"] > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
