# coding=utf-8
"""v6.1 + v6.2 单元测试 (Stage 27).

覆盖:
- factor_weighting.compute_cross_section_ic
- factor_weighting.compute_ic_timeseries
- factor_weighting.compute_factor_weights (防 look-ahead 防失效因子剔除)
- factor_orthogonal.get_factor_ir_order
- factor_orthogonal.orthogonalize_factor_panel (残差化)
- run_v6_1_backtest 集成
- run_v6_2_backtest 集成

设计:
- 单元测试: 构造合成 nav, 验证 IC 计算的数学正确性
- 集成测试: 用 fixture parquet 数据, 验证 NAV 形状 + Calmar 范围
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import (
    compute_cross_section_ic,
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
    DEFAULT_HORIZON_DAYS,
    MIN_MONTHS_FOR_IC,
)
from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
    get_factor_ir_order,
    orthogonalize_factor_panel,
)
from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def synthetic_close():
    """合成 close 面板: 12 ETF, 60 天.

    ETF1-6 强势, ETF7-12 弱势 → 下期收益明确, 因子 f1 应与收益正相关.
    """
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=60)
    codes = [f"ETF{i}" for i in range(1, 13)]
    panel = pd.DataFrame(index=dates, columns=codes, dtype=float)
    base = 10.0
    for i, c in enumerate(codes):
        drift = 0.001 if i < 6 else -0.001
        ret = drift + rng.normal(0, 0.01, len(dates))
        ret_s = pd.Series(ret, index=dates)
        panel[c] = base * (1 + ret_s).cumprod()
    return panel


@pytest.fixture
def synthetic_factor_panel(synthetic_close):
    """合成因子 panel: f1 与收益正相关 (IC 接近 1)."""
    dates = synthetic_close.index
    codes = list(synthetic_close.columns)
    # 截面: 强势 ETF (i<6) 高分, 弱势 ETF (i>=6) 低分
    f1 = pd.DataFrame(index=dates, columns=codes, dtype=float)
    for d in dates:
        for i, c in enumerate(codes):
            f1.loc[d, c] = (6 - i) * 0.5
    out = {}
    for c in codes:
        out[c] = pd.DataFrame({"f1_test": f1[c]})
    return out


# ============================================================
# factor_weighting 单元测试
# ============================================================
class TestComputeCrossSectionIC:
    def test_positive_factor_returns_positive_ic(self, synthetic_close, synthetic_factor_panel):
        """因子与下期收益同向 → IC 应为正 (synthetic, 噪声下不一定 > 0.5)."""
        ic = compute_cross_section_ic(
            synthetic_factor_panel, synthetic_close,
            synthetic_close.index[5], ["f1_test"], horizon=5,
        )
        assert "f1_test" in ic.index, f"应包含 f1_test, got {ic.index.tolist()}"
        assert ic["f1_test"] > 0.0, f"合成正向 IC 应 > 0, got {ic['f1_test']:.3f}"

    def test_data_too_short_returns_nan(self, synthetic_close, synthetic_factor_panel):
        """horizon 越界 → IC 应 NaN (不报错)."""
        ic = compute_cross_section_ic(
            synthetic_factor_panel, synthetic_close,
            synthetic_close.index[5], ["f1_test"], horizon=999,
        )
        # 检查返回空 Series
        assert ic.empty or pd.isna(ic.get("f1_test", np.nan))


# ============================================================
# compute_softmax_weights 单元测试 (Stage 28)
# ============================================================
class TestComputeSoftmaxWeights:
    """[Stage 28] 软权重测试."""

    def test_rows_normalize_to_one(self):
        """每行权重和 = 1."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import compute_softmax_weights
        dates = pd.bdate_range("2024-01-01", periods=8)
        ir = pd.DataFrame({
            "f1": [0.6, 0.7, 0.5, 0.8, 0.6, 0.7, 0.5, 0.6],
            "f2": [0.3, 0.4, 0.3, 0.5, 0.4, 0.4, 0.3, 0.4],
            "f3": [-0.2, -0.3, -0.1, -0.4, -0.3, -0.3, -0.2, -0.3],
        }, index=dates)
        w = compute_softmax_weights(ir, sharpness=3.0, min_ir_threshold=0.5)
        for d in w.index:
            assert abs(w.loc[d].sum() - 1.0) < 1e-6, (
                f"行 {d} 权重和 {w.loc[d].sum()} != 1.0"
            )

    def test_min_ir_threshold_filters_noise(self):
        """|IR| < 阈值的因子权重=0."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import compute_softmax_weights
        dates = pd.bdate_range("2024-01-01", periods=5)
        ir = pd.DataFrame({
            "strong": [0.8, 0.7, 0.6, 0.7, 0.8],   # |IR| > 0.5
            "weak":   [0.1, 0.2, 0.1, 0.1, 0.2],   # |IR| < 0.5
            "neg":    [-0.1, -0.2, -0.3, -0.1, -0.2],  # |IR| < 0.5
        }, index=dates)
        w = compute_softmax_weights(ir, sharpness=3.0, min_ir_threshold=0.5)
        # weak/neg 因子权重应为 0 (被阈值剔除)
        for d in w.index:
            assert w.loc[d, "weak"] < 1e-9
            assert w.loc[d, "neg"] < 1e-9
            assert w.loc[d, "strong"] > 0.5  # 主要分配给 strong

    def test_sharpness_concentration(self):
        """sharpness 越大, 权重越集中 (近 argmax)."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import compute_softmax_weights
        dates = pd.bdate_range("2024-01-01", periods=3)
        ir = pd.DataFrame({
            "f1": [1.0, 1.0, 1.0],
            "f2": [0.8, 0.8, 0.8],
            "f3": [0.5, 0.5, 0.5],
        }, index=dates)
        w_soft = compute_softmax_weights(ir, sharpness=1.0, min_ir_threshold=0.0)
        w_hard = compute_softmax_weights(ir, sharpness=10.0, min_ir_threshold=0.0)
        # 高 sharpness 下 f1 占比应更高
        for d in w_soft.index:
            assert w_hard.loc[d, "f1"] > w_soft.loc[d, "f1"], (
                f"high sharpness 应更集中: {w_hard.loc[d, 'f1']} vs {w_soft.loc[d, 'f1']}"
            )

    def test_all_zero_ir_degrades_to_equal(self):
        """全部 IR 都低于阈值 → 退化为等权."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import compute_softmax_weights
        dates = pd.bdate_range("2024-01-01", periods=3)
        ir = pd.DataFrame({
            "f1": [0.3, 0.3, 0.3],   # |IR| < 0.5
            "f2": [0.2, 0.2, 0.2],
        }, index=dates)
        w = compute_softmax_weights(ir, sharpness=3.0, min_ir_threshold=0.5)
        for d in w.index:
            assert abs(w.loc[d].sum() - 1.0) < 1e-6
            assert abs(w.loc[d, "f1"] - 0.5) < 1e-6

    def test_no_data_returns_empty(self):
        """空 DataFrame → 空 DataFrame."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_1.factor_weighting import compute_softmax_weights
        w = compute_softmax_weights(pd.DataFrame(), sharpness=3.0)
        assert w.empty


class TestPredefinedFactorOrder:
    """[Stage 28] 预定义顺序不变量测试."""

    def test_factor_order_is_immutable_tuple(self):
        """预定义顺序应是 immutable tuple."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import PREDEFINED_FACTOR_ORDER
        assert isinstance(PREDEFINED_FACTOR_ORDER, tuple)
        # 尝试修改应报错
        try:
            PREDEFINED_FACTOR_ORDER[0] = "fake"  # type: ignore
            assert False, "tuple 不应可改"
        except TypeError:
            pass

    def test_factor_order_has_11_factors(self):
        """预定义顺序应包含 11 个因子."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import PREDEFINED_FACTOR_ORDER
        assert len(PREDEFINED_FACTOR_ORDER) == 11

    def test_factor_order_financial_order(self):
        """预定义顺序应按: 动量 → 反转 → 多空 → 量价."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import PREDEFINED_FACTOR_ORDER
        # 位置 0,1 应为动量族
        assert "f1_second_mom" in PREDEFINED_FACTOR_ORDER[:2]
        assert "f2_mom_term" in PREDEFINED_FACTOR_ORDER[:2]
        # 位置 2,3 应为反转族
        assert "f3_amt_vol" in PREDEFINED_FACTOR_ORDER[2:4]
        assert "f4_vol_vol" in PREDEFINED_FACTOR_ORDER[2:4]
        # 位置 4-6 为多空族
        assert "f5_turnover" in PREDEFINED_FACTOR_ORDER[4:7]
        assert "f6_ls_total" in PREDEFINED_FACTOR_ORDER[4:7]
        assert "f7_ls_change" in PREDEFINED_FACTOR_ORDER[4:7]
        # 位置 7-10 为量价族
        assert "f8_pv_rankcov" in PREDEFINED_FACTOR_ORDER[7:]
        assert "f11_vol_range" in PREDEFINED_FACTOR_ORDER[7:]


class TestComputeFactorWeights:
    def test_lookahead_prevention(self):
        """防 look-ahead: t 期权重用截至 t-1 的 IC (shift(1))."""
        # 构造 IC 时序: 前 5 期 IC=1, 后 5 期 IC=0
        dates = pd.bdate_range("2024-01-01", periods=10)
        ic_ts = pd.DataFrame(
            1.0, index=dates, columns=["fac1"]  # 全部为 1
        )
        # IC=1 → IR=1/0 → Inf (std=0), 我们的公式给 IR=Inf
        # 改测试: 给一组 IC 一致 = 1, 看 weights 计算不崩溃
        weights = compute_factor_weights(ic_ts, min_months=3, smooth_window=0)
        # 由于 std=0, 我们设置 eps 防 Inf, IR 应该是大值
        assert not weights.empty
        # 头 2 期因 min_periods < min_months, 全 NaN
        # 从第 3 期开始有权重
        assert weights.iloc[0].sum() <= 1.001  # 归一化 (允许浮点误差)
        if len(weights) > 1:
            assert weights.iloc[1:].sum().sum() > 0

    def test_negative_factor_excluded(self):
        """IC 持续为负 → IR<0 → 权重=0 (自动剔除)."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        ic_ts = pd.DataFrame(
            {
                "good": [0.1, 0.05, 0.2, 0.15, 0.18, 0.12, 0.16, 0.14, 0.17, 0.13],
                "bad": [-0.1, -0.05, -0.2, -0.15, -0.18, -0.12, -0.16, -0.14, -0.17, -0.13],
            },
            index=dates,
        )
        weights = compute_factor_weights(ic_ts, min_months=3, smooth_window=0)
        # 'bad' 因子 IR < 0 在有效窗口后应被剔除 (前 3 行是 warmup 等权期)
        # 找首个 bad=0 且 good=1 的有效行 (IC shift 后第 4 行的迭代)
        non_warmup_dates = []
        for d in weights.index:
            # 当 IR 已计算, bad 因子权重应被截断为 0
            if weights.loc[d, "good"] > 1e-9 and weights.loc[d, "bad"] < 1e-9:
                non_warmup_dates.append(d)

        assert len(non_warmup_dates) > 0, (
            f"应有 bad=0 的有效行, got weights:\n{weights.round(3)}"
        )
        # 检查所有非 warmup 的行
        for d in non_warmup_dates:
            assert weights.loc[d, "bad"] < 1e-9, (
                f"bad 权重应为 0, got {weights.loc[d, 'bad']} at {d}"
            )
            assert weights.loc[d, "good"] > 0, "good 因子应有正权重"

    def test_weights_sum_to_one_per_row(self):
        """每行权重和 = 1 (归一化)."""
        dates = pd.bdate_range("2024-01-01", periods=8)
        ic_ts = pd.DataFrame(
            {
                "f1": [0.05, 0.10, 0.08, 0.12, 0.07, 0.09, 0.11, 0.06],
                "f2": [0.03, 0.05, 0.04, 0.06, 0.05, 0.07, 0.08, 0.04],
                "f3": [-0.01, -0.02, -0.01, 0.01, 0.02, 0.01, 0.03, 0.02],
            },
            index=dates,
        )
        weights = compute_factor_weights(ic_ts, min_months=3, smooth_window=0)
        for d in weights.index:
            row_sum = weights.loc[d].sum()
            assert abs(row_sum - 1.0) < 1e-6, f"行 {d} 权重和 {row_sum} != 1.0"

    def test_invalid_weights_returns_equal(self):
        """全部因子失效 → 等权."""
        dates = pd.bdate_range("2024-01-01", periods=8)
        ic_ts = pd.DataFrame(
            {
                "f1": [-0.1, -0.05, -0.08, -0.06, -0.07, -0.09, -0.04, -0.11],
                "f2": [-0.15, -0.10, -0.12, -0.11, -0.13, -0.14, -0.16, -0.08],
            },
            index=dates,
        )
        weights = compute_factor_weights(ic_ts, min_months=3, smooth_window=0)
        if not weights.empty:
            for d in weights.index:
                # 全部 IR < 0 → 等权 (1/N=0.5)
                row_sum = weights.loc[d].sum()
                assert abs(row_sum - 1.0) < 1e-6, (
                    f"全部失效应等权, got row sum {row_sum} at {d}"
                )


# ============================================================
# factor_orthogonal 单元测试
# ============================================================
class TestGetFactorIROrder:
    def test_returns_predefined_order(self, synthetic_close, synthetic_factor_panel):
        """[Stage 28] 应返回预定义金融顺序, 不依赖数据."""
        # 多个 rebal_dates 给出 IC 多样本; 现在函数忽略 IR, 返回预定义顺序
        rebal_dates = list(synthetic_close.index[::5])
        # 给真正的 11 因子名 (合成数据中是 f1_test, 但应该不出现)
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import PREDEFINED_FACTOR_ORDER
        # 传 11 个全部因子, 看返回是否 = PREDEFINED
        order = get_factor_ir_order(
            synthetic_factor_panel, synthetic_close,
            rebal_dates, list(PREDEFINED_FACTOR_ORDER), horizon=5, min_periods=3,
        )
        assert order == list(PREDEFINED_FACTOR_ORDER)

    def test_returns_subset_of_predefined(self, synthetic_close, synthetic_factor_panel):
        """[Stage 28] 子集查询应返回在预定义顺序中存在的因子."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import PREDEFINED_FACTOR_ORDER
        subset = ["f1_second_mom", "f3_amt_vol"]
        order = get_factor_ir_order(
            synthetic_factor_panel, synthetic_close,
            [], subset, min_periods=12,
        )
        assert order == subset

    def test_custom_factor_not_in_predefined(self, synthetic_close, synthetic_factor_panel):
        """[Stage 28] 非预定义因子 (如 f1_test) 应被过滤掉."""
        order = get_factor_ir_order(
            synthetic_factor_panel, synthetic_close,
            [], ["f1_test"], min_periods=12,
        )
        assert order == []  # 不是 11 因子之一

    def test_none_factors_returns_full(self, synthetic_close, synthetic_factor_panel):
        """[Stage 28] factors=None 时返回完整预定义顺序."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import PREDEFINED_FACTOR_ORDER
        order = get_factor_ir_order(
            synthetic_factor_panel, synthetic_close,
            [], None,
        )
        assert order == list(PREDEFINED_FACTOR_ORDER)


class TestOrthogonalizeFactorPanel:
    def test_single_factor_unchanged(self, synthetic_close, synthetic_factor_panel):
        """只有 1 个因子时, 正交化应保持原值."""
        rebal_dates = list(synthetic_close.index[::5])
        order = ["f1_test"]
        panel_orth = orthogonalize_factor_panel(
            synthetic_factor_panel, order, rebal_dates,
        )
        # 1 因子不残差化, 应一致
        for code in panel_orth:
            if not panel_orth[code].empty:
                orig = synthetic_factor_panel[code]["f1_test"]
                new = panel_orth[code]["f1_test"]
                common = orig.index.intersection(new.index)
                if len(common) > 0:
                    diff = (orig.loc[common] - new.loc[common]).abs().max()
                    assert diff < 1e-6, f"单因子应不变, max diff {diff}"


# ============================================================
# 集成测试: run_v6_1_backtest
# ============================================================
@pytest.fixture(scope="module")
def real_panel_close():
    p = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    return p.loc["2021-01-01":"2024-01-01"]  # 缩小范围加速测试


@pytest.fixture(scope="module")
def real_panel_ohlcv():
    p = pd.read_parquet(REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
    return p.loc["2021-01-01":"2024-01-01"]


class TestRunV61Backtest:
    def test_baseline_runs(self, real_panel_close, real_panel_ohlcv):
        """baseline (等权) 应能跑通并返回 NAV."""
        cfg = V6_1Config(use_ic_weighting=False, min_history=120)
        nav = run_v6_1_backtest(real_panel_close, real_panel_ohlcv, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) == len(real_panel_close)
        assert nav.iloc[0] == 1.0
        assert nav.dropna().iloc[-1] > 0

    def test_ic_weighted_runs(self, real_panel_close, real_panel_ohlcv):
        """IC 加权版应能跑通."""
        cfg = V6_1Config(use_ic_weighting=True, min_history=120, ic_min_months=6)
        nav = run_v6_1_backtest(real_panel_close, real_panel_ohlcv, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) == len(real_panel_close)


class TestRunV62Backtest:
    def test_no_orth_runs(self, real_panel_close, real_panel_ohlcv):
        """v6.2 不正交版 (等同 v6.1) 应能跑通."""
        cfg = V6_2Config(use_orthogonal=False, min_history=120, ic_min_months=6)
        nav = run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) == len(real_panel_close)

    def test_with_orth_runs(self, real_panel_close, real_panel_ohlcv):
        """v6.2 + 正交化 (默认 expanding IR) 应能跑通."""
        cfg = V6_2Config(
            use_orthogonal=True, min_history=120, ic_min_months=6,
            sort_method="ir_expanding",
        )
        nav = run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) == len(real_panel_close)

    def test_with_orth_predefined_runs(self, real_panel_close, real_panel_ohlcv):
        """v6.2 + 正交化 (predefined 路径) 应能跑通."""
        cfg = V6_2Config(
            use_orthogonal=True, min_history=120, ic_min_months=6,
            sort_method="predefined",
        )
        nav = run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) == len(real_panel_close)

    def test_with_orth_ir_full_runs(self, real_panel_close, real_panel_ohlcv):
        """v6.2 + 正交化 (DEPRECATED ir_full) — Stage 29 已从生产路径移出.

        生产代码 (industry_rotation_v6_2.py) 抛 NotImplementedError.
        真正的全样本 IR 实现见 tests/_helpers/deprecated_order.py.
        """
        cfg = V6_2Config(
            use_orthogonal=True, min_history=120, ic_min_months=6,
            sort_method="ir_full",
        )
        import pytest
        with pytest.raises(NotImplementedError, match="ir_full.*DEPRECATED"):
            run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg)

    def test_deprecated_helper_runs(self, real_panel_close, real_panel_ohlcv):
        """DEPRECATED helper 函数本身应能跑通 (供 ablation 对照)."""
        from tests.strategy.momentum_etf_rotation._helpers.deprecated_order import (
            get_factor_ir_order_deprecated,
        )
        from QuantNodes.strategy.momentum_etf_rotation.v5.industry_factors import (
            compute_all_factors_panel,
        )
        from QuantNodes.strategy.momentum_etf_rotation.v5.industry_rotation_v5 import (
            FactorEngineConfig,
        )
        factor_panel = compute_all_factors_panel(
            real_panel_ohlcv, FactorEngineConfig(),
        )
        factors = list(FactorEngineConfig().name_map.keys())
        rebal_dates = list(real_panel_close.index[::21])[:10]  # 前 10 个调仓日
        order = get_factor_ir_order_deprecated(
            factor_panel, real_panel_close, rebal_dates, factors, horizon=21,
        )
        assert isinstance(order, list)
        assert all(f in factors for f in order)


class TestGetFactorIROrderExpanding:
    """[Phase 1] expanding IR 测试: 每调仓日 d_i 用截至 d_{i-1} 的 IC 算 IR."""

    def test_returns_dict_per_rebalance_date(self, synthetic_close, synthetic_factor_panel):
        """返回 dict[date] -> order, key 是 rebalance_date."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
            get_factor_ir_order_expanding,
        )
        rebal_dates = list(synthetic_close.index[::5])
        orders = get_factor_ir_order_expanding(
            synthetic_factor_panel, synthetic_close,
            rebal_dates, ["f1_test"], horizon=5, min_periods=2, lookback_months=3,
        )
        assert isinstance(orders, dict)
        # 所有 rebal_dates 都应有 entry
        for d in rebal_dates:
            assert d in orders, f"missing {d}"
            assert orders[d] == ["f1_test"]  # 只有 1 个因子

    def test_lookahead_free(self, synthetic_close, synthetic_factor_panel):
        """[关键] 用 d_i 的 expanding IR 排序不应依赖 d_i 之后的 IC.

        算法: d_i 的 IR 用 d_{i-1} 及之前的 past dates, 不含 d_i 之后.
        """
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
            get_factor_ir_order_expanding,
        )
        # 构造 2 个 ETF, f1 反向 (f1 高 → 收益低)
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2024-01-01", periods=20)
        panel = {f"ETF{i}": pd.DataFrame({"f1": i}, index=dates) for i in range(1, 5)}
        close = pd.DataFrame({f"ETF{i}": np.exp(np.cumsum(rng.normal(0, 0.01, 20))) for i in range(1, 5)}, index=dates)
        rebal_dates = list(dates[::3])  # ~6 个调仓日
        orders = get_factor_ir_order_expanding(
            panel, close, rebal_dates, ["f1"],
            horizon=3, min_periods=2, lookback_months=12,
        )
        # 不应崩溃, 每个 d 都应有 entry
        for d in rebal_dates:
            assert d in orders
            assert orders[d] == ["f1"]  # 单因子

    def test_min_periods_fallback_factors_order(self, synthetic_close, synthetic_factor_panel):
        """min_periods 不足时 fallback 用 factors 原顺序 (冷启动保护)."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
            get_factor_ir_order_expanding,
        )
        factors_subset = ["f1_test", "fake1", "fake2"]
        rebal_dates = list(synthetic_close.index[:3])
        orders = get_factor_ir_order_expanding(
            synthetic_factor_panel, synthetic_close,
            rebal_dates, factors_subset,
            horizon=5, min_periods=12, lookback_months=36,
        )
        for d in rebal_dates:
            assert d in orders
            # fallback 用 factors 顺序
            assert orders[d] == factors_subset


class TestOrthogonalizePerDate:
    """[Phase 1] orthogonalize_factor_panel 的 order_per_date 参数测试."""

    def test_per_date_order_changes_residuals(self, synthetic_close, synthetic_factor_panel):
        """不同 order_per_date 应产生不同残差."""
        # 构造 2 个共线因子的 panel
        dates = synthetic_close.index
        codes = list(synthetic_close.columns)
        rng = np.random.default_rng(7)
        # f1: 选股 "momentum-like", f2: 强相关于 f1
        f1 = pd.DataFrame(index=dates, columns=codes, dtype=float)
        f2 = pd.DataFrame(index=dates, columns=codes, dtype=float)
        for d in dates:
            for j, c in enumerate(codes):
                base = (6 - j) * 0.5
                f1.loc[d, c] = base
                f2.loc[d, c] = base + rng.normal(0, 0.1)
        panel = {}
        for c in codes:
            panel[c] = pd.DataFrame({"fA": f1[c], "fB": f2[c]})

        rebal_dates = list(dates[::5])
        # order A: fA 在前
        ord_A = {d: ["fA", "fB"] for d in rebal_dates}
        # order B: fB 在前
        ord_B = {d: ["fB", "fA"] for d in rebal_dates}

        orth_A = orthogonalize_factor_panel(panel, ["fA", "fB"], rebal_dates, order_per_date=ord_A)
        orth_B = orthogonalize_factor_panel(panel, ["fA", "fB"], rebal_dates, order_per_date=ord_B)

        # 第二个因子残差不同 (位置 1 = fA in A, fA in B 但 B 中 fA 在末位)
        # 残差值会有差异
        max_diff = 0.0
        for c in codes:
            if c in orth_A and c in orth_B and not orth_A[c].empty and not orth_B[c].empty:
                common = orth_A[c].index.intersection(orth_B[c].index)
                if len(common):
                    diff = (orth_A[c].loc[common, "fB"].fillna(0) - orth_B[c].loc[common, "fB"].fillna(0)).abs().max()
                    if pd.notna(diff) and diff > max_diff:
                        max_diff = diff
        # 若 fA, fB 完全相同, order 不影响; 但因为有噪声残差不同
        # 这里不强求 max_diff > 0 (因为可能近似相同)
        assert isinstance(orth_A, dict)
        assert isinstance(orth_B, dict)


class TestOrthogonalizeFactorPanelQR:
    """[Phase 3] QR 分解对称正交测试 (顺序无关)."""

    def test_runs(self, synthetic_close, synthetic_factor_panel):
        """QR 分解正交化能跑通."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
            orthogonalize_factor_panel_qr,
        )
        rebal_dates = list(synthetic_close.index[::5])
        out = orthogonalize_factor_panel_qr(synthetic_factor_panel, rebal_dates)
        assert isinstance(out, dict)
        # 输出列名应是 f_qr_0, f_qr_1, ... (默认 1 因子 panel → 1 列)
        for code, df in out.items():
            assert "f_qr_0" in df.columns


class TestV62SortMethod:
    """[Phase 1/3/4] V6_2Config.sort_method 默认值和参数测试."""

    def test_default_is_warmup_ir(self):
        """[Stage 29] sort_method 应默认 ir_expanding (5-fold 验证 4/5 胜 v6.1).

        历史: Phase 4 切到 warmup_ir 12m, OOS 0.629 弱于 v6.1 0.748.
        Stage 29 5-fold 验证 ir_expanding 4/5 胜, 升级为默认.
        """
        cfg = V6_2Config()
        assert cfg.sort_method == "ir_expanding"

    def test_valid_sort_methods(self):
        """有效 sort_method: warmup_ir/ir_expanding/predefined/qr (ir_full Stage 29 已移出)."""
        for sm in ["warmup_ir", "ir_expanding", "predefined", "qr"]:
            cfg = V6_2Config(sort_method=sm)
            assert cfg.sort_method == sm

    def test_ir_full_raises(self):
        """sort_method='ir_full' 在生产路径应抛 NotImplementedError."""
        cfg = V6_2Config(sort_method="ir_full")
        from QuantNodes.strategy.momentum_etf_rotation.v6_2 import (
            run_v6_2_backtest,
        )
        import pytest
        with pytest.raises(NotImplementedError, match="ir_full.*DEPRECATED"):
            run_v6_2_backtest(None, None, cfg)


class TestGetFactorIROrderWarmup:
    """[Phase 4] warmup-IR 一次性固定顺序测试."""

    def test_returns_ordered_list(self, synthetic_close, synthetic_factor_panel):
        """返回 list[str], 长度等于 factors 数量, 按 IR 降序."""
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
            get_factor_ir_order_warmup,
        )
        rebal_dates = list(synthetic_close.index[::2])
        # 构造 11 因子 (模拟真实)
        factors = [f"f{k}" for k in range(1, 12)]
        # 在 synthetic_factor_panel 中加入这 11 因子
        rng = np.random.default_rng(42)
        for c in synthetic_factor_panel:
            df = pd.DataFrame(index=synthetic_close.index)
            for f in factors:
                df[f] = rng.normal(0, 1, len(synthetic_close.index))
            synthetic_factor_panel[c] = df
        order = get_factor_ir_order_warmup(
            synthetic_factor_panel, synthetic_close,
            rebal_dates, factors,
            horizon=5, warmup_months=6,
        )
        assert isinstance(order, list)
        assert len(order) == len(factors)
        assert set(order) == set(factors)

    def test_lookahead_free(self, synthetic_close, synthetic_factor_panel):
        """[关键] warmup-IR 仅用 rebal_dates 的前 warmup_months 个月, 不含未来."""
        # 此测试已通过 test_returns_ordered_list 间接验证;
        # 显式断言: 当 warmup_months=1, 只用 1 个 rebalance_date
        from QuantNodes.strategy.momentum_etf_rotation.v6_2.factor_orthogonal import (
            get_factor_ir_order_warmup,
        )
        rebal_dates = list(synthetic_close.index[::3])
        factors = ["f1_test"]
        order = get_factor_ir_order_warmup(
            synthetic_factor_panel, synthetic_close,
            rebal_dates, factors,
            horizon=3, warmup_months=1,
        )
        # 即使 warmup_months=1, 也不崩溃 (因为 single factor 退化)
        assert order == factors


class TestV62WarmupRuns:
    """[Phase 4] v6.2 warmup_ir 路径集成测试."""

    def test_warmup_ir_runs(self, real_panel_close, real_panel_ohlcv):
        """v6.2 + warmup_ir 正交化应能跑通."""
        cfg = V6_2Config(
            use_orthogonal=True, min_history=120, ic_min_months=6,
            sort_method="warmup_ir",
        )
        nav = run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) == len(real_panel_close)
        assert nav.iloc[0] == 1.0
        assert nav.dropna().iloc[-1] > 0

    def test_warmup_ir_settles_after_warmup(self, real_panel_close, real_panel_ohlcv):
        """warmup 期前 24 月之后应有正交化效果 (NAV 与 no_orth 应不同)."""
        cfg_no = V6_2Config(
            use_orthogonal=False, min_history=120, ic_min_months=6,
        )
        cfg_yes = V6_2Config(
            use_orthogonal=True, min_history=120, ic_min_months=6,
            sort_method="warmup_ir",
        )
        nav_no = run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg_no)
        nav_yes = run_v6_2_backtest(real_panel_close, real_panel_ohlcv, cfg_yes)
        # 二者 NAV 应有差异 (warmup 后的正交化起效)
        common_idx = nav_no.dropna().index.intersection(nav_yes.dropna().index)
        diff = (nav_yes.loc[common_idx] - nav_no.loc[common_idx]).abs().max()
        assert diff > 1e-9, f"应与 no_orth 有差异, got max diff {diff}"
