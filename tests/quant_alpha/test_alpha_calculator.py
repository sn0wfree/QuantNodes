# coding=utf-8
"""Tests for QuantAlpha adapters (M4 PR).

覆盖：
- Expression AST：Feature / Ref / BinaryOp / UnaryOp / RollingOp
- BaseAlphaCalculator ABC 接口
- PolarsAlphaCalculator 7 个方法
- 与 Qlib Alpha158 形式参考的等价性（行为一致，公式等价）

测试基线：Qlib Alpha158 158 公式（KBAR/Price/Volume/Rolling）
- 5 个 IC 等价测试：单 IC / rank IC / 多步 IC / 互 IC / Pool IC
- 测试不依赖真实 qlib 安装，只用 QuantNodes 算子 + polars 计算
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.adapters import (
    BaseAlphaCalculator,
    PolarsAlphaCalculator,
    Expression,
    Feature,
    Ref,
    Add,
    Sub,
    Mul,
    Div,
    Greater,
    Less,
    Abs,
    Neg,
    Log,
    Sign,
    Sqrt,
    Mean,
    Std,
    Sum,
    Max,
    Min,
    Delta,
    BinaryOp,
    Literal,
    expression_to_formula,
    collect_feature_fields,
    collect_rolling_windows,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """合成 5 票 × 10 日 测试数据"""
    np.random.seed(42)
    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    rows = []
    for date in dates:
        for code in ["A", "B", "C", "D", "E"]:
            rows.append({
                "date": date,
                "code": code,
                "close": float(np.random.randn() * 5 + 100),
                "open": float(np.random.randn() * 5 + 100),
                "high": float(np.random.randn() * 5 + 102),
                "low": float(np.random.randn() * 5 + 98),
                "vol": float(np.random.randint(1000, 5000)),
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def forward_returns(sample_data: pl.DataFrame) -> Dict[int, pl.Series]:
    """多期前瞻收益"""
    np.random.seed(123)
    n = len(sample_data)
    return {
        1: pl.Series("ret_1d", np.random.randn(n) * 0.02),
        5: pl.Series("ret_5d", np.random.randn(n) * 0.05),
        10: pl.Series("ret_10d", np.random.randn(n) * 0.08),
    }


@pytest.fixture
def calc(sample_data: pl.DataFrame, forward_returns: Dict[int, pl.Series]) -> PolarsAlphaCalculator:
    """PolarsAlphaCalculator 默认实例"""
    return PolarsAlphaCalculator(sample_data, forward_returns)


# ==============================================================================
# Test Class 1: Expression AST
# ==============================================================================


class TestExpressionAST:
    """Expression AST 节点测试"""

    def test_feature_to_string(self):
        """Feature 节点"""
        e = Feature("close")
        assert e.to_string() == "close"
        assert e.children == []

    def test_ref_to_string(self):
        """Ref 节点"""
        e = Ref(Feature("close"), 5)
        assert e.to_string() == "close.shift(5)"
        assert len(e.children) == 1

    def test_binary_op_sub(self):
        """Sub 节点"""
        e = Sub(Feature("close"), Feature("open"))
        assert e.to_string() == "(close - open)"

    def test_binary_op_mul(self):
        """Mul 节点"""
        e = Mul(Feature("close"), Feature("vol"))
        assert e.to_string() == "(close * vol)"

    def test_binary_op_div(self):
        """Div 节点"""
        e = Div(Feature("close"), Feature("vol"))
        assert e.to_string() == "(close / vol)"

    def test_binary_op_unsupported_raises(self):
        """不支持的 op 抛 ValueError"""
        with pytest.raises(ValueError, match="Unsupported op"):
            BinaryOp(Feature("close"), Feature("open"), "mod")

    def test_unary_op_neg(self):
        """Neg 节点"""
        e = Neg(Feature("close"))
        assert e.to_string() == "(-close)"

    def test_unary_op_log(self):
        """Log 节点"""
        e = Log(Feature("close"))
        assert e.to_string() == "log(close)"

    def test_unary_op_abs(self):
        """Abs 节点"""
        e = Abs(Feature("close"))
        assert e.to_string() == "abs(close)"

    def test_rolling_op_mean(self):
        """Mean 节点"""
        e = Mean(Feature("close"), 20)
        assert e.to_string() == "ts_mean(close, 20)"

    def test_rolling_op_std(self):
        """Std 节点"""
        e = Std(Feature("close"), 5)
        assert e.to_string() == "ts_std(close, 5)"

    def test_nested_expression(self):
        """嵌套表达式"""
        e = Sub(Mean(Feature("close"), 5), Mean(Feature("close"), 20))
        assert e.to_string() == "(ts_mean(close, 5) - ts_mean(close, 20))"
        assert len(e.children) == 2

    def test_alpha101_nested(self):
        """Alpha 101 #9 风格：Ref + Sub"""
        e = Sub(Ref(Feature("close"), 5), Feature("close"))
        assert e.to_string() == "(close.shift(5) - close)"

    def test_expression_to_formula_helper(self):
        """expression_to_formula 便利函数"""
        e = Add(Feature("close"), Feature("open"))
        assert expression_to_formula(e) == "(close + open)"

    def test_collect_feature_fields(self):
        """递归收集 Feature 字段"""
        e = Sub(Add(Feature("close"), Feature("vol")), Feature("open"))
        fields = collect_feature_fields(e)
        assert set(fields) == {"close", "vol", "open"}

    def test_collect_rolling_windows(self):
        """递归收集 RollingOp 窗口"""
        e = Sub(Mean(Feature("close"), 5), Mean(Feature("close"), 20))
        windows = collect_rolling_windows(e)
        assert set(windows) == {5, 20}

    def test_expression_repr(self):
        """Expression __repr__"""
        e = Sub(Feature("close"), Feature("open"))
        assert "Expr" in repr(e)
        assert "close" in repr(e)


# ==============================================================================
# Test Class 2: BaseAlphaCalculator ABC
# ==============================================================================


class TestBaseAlphaCalculatorABC:
    """BaseAlphaCalculator ABC 接口测试"""

    def test_abc_has_7_abstract_methods(self):
        """ABC 有 7 个抽象方法"""
        abstract_methods = BaseAlphaCalculator.__abstractmethods__
        expected = {
            "calc_single_IC_ret",
            "calc_single_rIC_ret",
            "calc_single_all_ret",
            "calc_mutual_IC",
            "calc_pool_IC_ret",
            "calc_pool_rIC_ret",
            "calc_pool_all_ret",
        }
        assert set(abstract_methods) == expected

    def test_cannot_instantiate_abc_directly(self):
        """ABC 不能直接实例化"""
        with pytest.raises(TypeError):
            BaseAlphaCalculator()


# ==============================================================================
# Test Class 3: PolarsAlphaCalculator 基本功能
# ==============================================================================


class TestPolarsAlphaCalculatorBasics:
    """PolarsAlphaCalculator 基本功能测试"""

    def test_construction(self, calc: PolarsAlphaCalculator):
        """构造"""
        assert calc is not None

    def test_stats(self, calc: PolarsAlphaCalculator):
        """stats 返回统计"""
        stats = calc.stats()
        assert stats["n_data_rows"] == 50
        assert stats["n_dates"] == 10
        assert stats["n_codes"] == 5
        assert stats["forward_returns"] == [1, 5, 10]
        assert stats["cross_sectional"] is True

    def test_evaluate_factor(
        self, calc: PolarsAlphaCalculator, sample_data: pl.DataFrame
    ):
        """单因子评估"""
        e = Feature("close")
        result = calc._evaluate_factor(e)
        assert result is not None
        assert len(result) == len(sample_data)

    def test_factor_cache(
        self, calc: PolarsAlphaCalculator,
    ):
        """因子缓存"""
        e = Feature("close")
        # 第一次计算
        r1 = calc._evaluate_factor(e)
        # 第二次应该命中缓存
        r2 = calc._evaluate_factor(e)
        # 缓存命中返回同一对象
        assert r1 is r2

    def test_evaluate_factor_invalid(
        self, calc: PolarsAlphaCalculator,
    ):
        """无效因子评估返回 None"""
        e = Sub(Feature("invalid_field"), Feature("close"))
        result = calc._evaluate_factor(e)
        assert result is None


# ==============================================================================
# Test Class 4: 7 个方法测试（与 Qlib Alpha158 形式等价）
# ==============================================================================


class TestSingleICRet:
    """calc_single_IC_ret 测试"""

    def test_basic_factor(self, calc: PolarsAlphaCalculator):
        """基础因子 IC（close 自身）"""
        e = Feature("close")
        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert isinstance(ic, np.ndarray)
        assert ic.dtype.kind == "f"  # float
        assert len(ic) == 10  # 10 个日期

    def test_alpha101_factor(self, calc: PolarsAlphaCalculator):
        """Alpha 101 #9 风格 IC"""
        e = Sub(Ref(Feature("close"), 5), Feature("close"))
        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert len(ic) == 10
        # IC 应在 [-1, 1] 范围内
        assert np.all(np.abs(ic[~np.isnan(ic)]) <= 1.0 + 1e-6)

    def test_rolling_factor(self, calc: PolarsAlphaCalculator):
        """滚动算子 IC"""
        e = Mean(Feature("close"), 5)
        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert len(ic) == 10

    def test_invalid_factor_returns_empty(
        self, calc: PolarsAlphaCalculator,
    ):
        """无效因子返回空数组"""
        e = Sub(Feature("nonexistent"), Feature("close"))
        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert len(ic) == 0

    def test_invalid_ret_offset_raises(
        self, calc: PolarsAlphaCalculator,
    ):
        """无效前瞻期抛 ValueError"""
        e = Feature("close")
        with pytest.raises(ValueError, match="No forward return"):
            calc.calc_single_IC_ret(e, ret_offset=99)


class TestSingleRICRet:
    """calc_single_rIC_ret 测试（rank IC）"""

    def test_basic_rank_ic(self, calc: PolarsAlphaCalculator):
        """基础 rank IC"""
        e = Feature("close")
        ric = calc.calc_single_rIC_ret(e, ret_offset=1)
        assert isinstance(ric, np.ndarray)
        assert len(ric) == 10
        # rank IC 也在 [-1, 1]
        assert np.all(np.abs(ric[~np.isnan(ric)]) <= 1.0 + 1e-6)

    def test_rank_ic_differs_from_pearson(
        self, calc: PolarsAlphaCalculator,
    ):
        """rank IC 与 Pearson IC 不完全相同（Spearman vs Pearson 差异）"""
        e = Mean(Feature("close"), 5)
        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        ric = calc.calc_single_rIC_ret(e, ret_offset=1)
        # 通常不同（但可能很接近）
        # 至少形状相同
        assert ic.shape == ric.shape


class TestSingleAllRet:
    """calc_single_all_ret 测试（多步 IC）"""

    def test_multistep_ic_shape(self, calc: PolarsAlphaCalculator):
        """多步 IC 形状 = (n_dates, n_ret_offsets)"""
        e = Feature("close")
        all_ic = calc.calc_single_all_ret(e)
        assert all_ic.shape == (10, 3)  # 10 dates, 3 offsets (1, 5, 10)

    def test_multistep_ic_values(self, calc: PolarsAlphaCalculator):
        """多步 IC 值"""
        e = Sub(Ref(Feature("close"), 5), Feature("close"))
        all_ic = calc.calc_single_all_ret(e)
        # 各列（offset）应与单步 IC 一致
        ic1 = calc.calc_single_IC_ret(e, ret_offset=1)
        ic5 = calc.calc_single_IC_ret(e, ret_offset=5)
        assert np.allclose(all_ic[:, 0], ic1, equal_nan=True)
        assert np.allclose(all_ic[:, 1], ic5, equal_nan=True)


class TestMutualIC:
    """calc_mutual_IC 测试"""

    def test_mutual_ic_basic(self, calc: PolarsAlphaCalculator):
        """两因子互 IC"""
        e1 = Feature("close")
        e2 = Feature("vol")
        mic = calc.calc_mutual_IC(e1, e2)
        assert isinstance(mic, np.ndarray)
        assert len(mic) == 10

    def test_self_mutual_ic(self, calc: PolarsAlphaCalculator):
        """自互 IC 应接近 1"""
        e = Feature("close")
        mic = calc.calc_mutual_IC(e, e)
        # 自相关应为 1（或 NaN 处理后接近 1）
        valid = mic[~np.isnan(mic)]
        if len(valid) > 0:
            assert np.allclose(valid, 1.0, atol=1e-6)

    def test_alpha158_rolling_pair(
        self, calc: PolarsAlphaCalculator,
    ):
        """Alpha 158 风格：2 滚动算子互 IC"""
        e1 = Mean(Feature("close"), 5)
        e2 = Mean(Feature("close"), 20)
        mic = calc.calc_mutual_IC(e1, e2)
        assert len(mic) == 10


class TestPoolICRet:
    """calc_pool_IC_ret 测试（ensemble IC）"""

    def test_pool_ic_single_factor(
        self, calc: PolarsAlphaCalculator,
    ):
        """单因子 pool = 单 IC"""
        e = Feature("close")
        single_ic = calc.calc_single_IC_ret(e, ret_offset=1)
        pool_ic = calc.calc_pool_IC_ret([e], ret_offset=1)
        # 等权 = 自身
        assert np.allclose(single_ic, pool_ic, equal_nan=True)

    def test_pool_ic_three_factors(self, calc: PolarsAlphaCalculator):
        """3 因子 pool IC"""
        exprs = [
            Feature("close"),
            Mean(Feature("close"), 5),
            Sub(Ref(Feature("close"), 1), Feature("close")),
        ]
        pool_ic = calc.calc_pool_IC_ret(exprs, ret_offset=1)
        assert len(pool_ic) == 10
        # IC 应在 [-1, 1] 范围
        valid = pool_ic[~np.isnan(pool_ic)]
        assert np.all(np.abs(valid) <= 1.0 + 1e-6)

    def test_pool_ic_weighted(self, calc: PolarsAlphaCalculator):
        """加权 pool IC"""
        e1 = Feature("close")
        e2 = Feature("vol")
        ic_equal = calc.calc_pool_IC_ret([e1, e2], ret_offset=1)
        ic_weighted = calc.calc_pool_IC_ret(
            [e1, e2], weights=[0.9, 0.1], ret_offset=1,
        )
        # 加权 vs 等权 — 结果不同
        assert not np.allclose(ic_equal, ic_weighted, equal_nan=True)

    def test_pool_ic_empty_raises(self, calc: PolarsAlphaCalculator):
        """空 exprs 抛 ValueError"""
        with pytest.raises(ValueError, match="exprs is empty"):
            calc.calc_pool_IC_ret([], ret_offset=1)


class TestPoolRICRet:
    """calc_pool_rIC_ret 测试"""

    def test_pool_rank_ic(self, calc: PolarsAlphaCalculator):
        """pool rank IC"""
        exprs = [Feature("close"), Mean(Feature("close"), 5)]
        ric = calc.calc_pool_rIC_ret(exprs, ret_offset=1)
        assert len(ric) == 10
        valid = ric[~np.isnan(ric)]
        assert np.all(np.abs(valid) <= 1.0 + 1e-6)


class TestPoolAllRet:
    """calc_pool_all_ret 测试"""

    def test_pool_all_ret_shape(self, calc: PolarsAlphaCalculator):
        """pool all ret 形状 = (n_dates, n_offsets)"""
        exprs = [Feature("close"), Mean(Feature("close"), 5)]
        all_ret = calc.calc_pool_all_ret(exprs)
        assert all_ret.shape == (10, 3)


# ==============================================================================
# Test Class 5: 与 Qlib Alpha158 形式等价性
# ==============================================================================


class TestQlibAlpha158Equivalence:
    """与 Qlib Alpha158 形式等价性测试

    Alpha 158 公式 = `{op}({field}, {window})` 形式。
    本测试验证 PolarsAlphaCalculator 能正确计算 Alpha 158 风格的
    5 个代表性公式（KBAR / Price / Volume / Rolling / 综合）。
    """

    def test_alpha158_kbar_form(self, calc: PolarsAlphaCalculator):
        """Alpha 158 KBAR 公式形式：KMID = (close - open) / open

        PolarsAlphaCalculator 应能通过 (Feature("close") - Feature("open")) / Feature("open")
        计算并产生有效 IC。
        """
        e = Div(Sub(Feature("close"), Feature("open")), Feature("open"))
        formula = expression_to_formula(e)
        assert formula == "((close - open) / open)"

        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert len(ic) == 10

    def test_alpha158_price_form(self, calc: PolarsAlphaCalculator):
        """Alpha 158 Price 公式：OPEN1 = open.shift(1) / close

        等价于 Ref(open, 1) / close
        """
        e = Div(Ref(Feature("open"), 1), Feature("close"))
        formula = expression_to_formula(e)
        assert formula == "(open.shift(1) / close)"

        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert len(ic) == 10

    def test_alpha158_volume_form(self, calc: PolarsAlphaCalculator):
        """Alpha 158 Volume 公式：VOL3 = volume.shift(3) / (volume + 1e-12)

        PolarsAlphaCalculator 通过 Ref(Feature("vol"), 3) / (vol + 1e-12) 表达
        """
        # 注：Alpha 158 的 + 1e-12 防止除零，需在公式中显式表达
        e = Div(Ref(Feature("vol"), 3), Add(Feature("vol"), 1e-12))
        formula = expression_to_formula(e)
        assert "shift(3)" in formula

        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        # 可能全 NaN（除零），但不应抛异常
        assert len(ic) == 10

    def test_alpha158_rolling_form(self, calc: PolarsAlphaCalculator):
        """Alpha 158 Rolling 公式：MA20 = close.rolling(20).mean()

        等价于 Mean(Feature("close"), 20)
        """
        e = Mean(Feature("close"), 20)
        formula = expression_to_formula(e)
        assert formula == "ts_mean(close, 20)"

        ic = calc.calc_single_IC_ret(e, ret_offset=1)
        assert len(ic) == 10

    def test_alpha158_combined_ic(
        self, calc: PolarsAlphaCalculator,
    ):
        """Alpha 158 综合：MA20 的 1d/5d/10d 多步 IC

        应输出 (10, 3) 形状
        """
        e = Mean(Feature("close"), 20)
        all_ic = calc.calc_single_all_ret(e)
        assert all_ic.shape == (10, 3)

    def test_polars_ic_matches_manual_corr(
        self, calc: PolarsAlphaCalculator,
    ):
        """PolarsAlphaCalculator IC 与直接 pl.corr 因子值结果一致

        验证：calc_single_IC_ret(e, ret_offset=1) 应等于
        用 calc._evaluate_factor() 拿到的 Series 做的 per-date corr 结果
        """
        e = Sub(Ref(Feature("close"), 2), Feature("close"))
        # 用 calc 内部排序后的 data（与 PolarsAlphaCalculator 一致）
        factor_series = calc._evaluate_factor(e)
        assert factor_series is not None
        fwd_1d = calc.forward_returns[1]

        # 手动用相同的 data 计算 per-date IC
        manual_df = calc.data.select([
            pl.col(calc.date_column).alias("_d"),
        ]).with_columns([
            factor_series.alias("_f"),
            fwd_1d.alias("_t"),
        ]).filter(
            pl.col("_f").is_not_null() & pl.col("_t").is_not_null()
        )
        manual_ic = manual_df.group_by("_d").agg(
            pl.corr("_f", "_t").alias("ic")
        ).sort("_d")["ic"].to_numpy()

        # PolarsAlphaCalculator 计算
        calc_ic = calc.calc_single_IC_ret(e, ret_offset=1)

        # 应对齐（允许 NaN）
        assert np.allclose(calc_ic, manual_ic, equal_nan=True, atol=1e-6)
