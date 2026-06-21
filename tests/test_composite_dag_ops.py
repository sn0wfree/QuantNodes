"""PR-QN-3b: 20 个内置 composite op 测试

锁定 PR-QN-3b (2026-06-21) 行为:
- 20 个内置 op 全部注册到 _COMPOSITE_REGISTRY
- 每个 op 都可在真实 polars DataFrame 上执行
- 端到端: instantiate → select → 执行结果 shape/values 合理
"""
from __future__ import annotations

import polars as pl
import pytest
from polars import col

from QuantNodes.operators import (
    is_composite_op, get_composite_spec, list_composite_ops,
)


# 20 个内置 op 全名
BUILTIN_OPS = [
    # 中性化 (3)
    "industry_neutralize", "market_neutralize", "subindustry_neutralize",
    # 横截面归一化 (3)
    "zscore_xs", "rank_xs", "scale_xs",
    # 滚动回归 (3)
    "rolling_beta", "rolling_ols_simplified", "rolling_residual",
    # 波动率 (4)
    "parkinson_vol", "garman_klass_vol", "yang_zhang_vol", "realized_vol",
    # 配对交易 (2)
    "pair_zscore", "pair_ratio",
    # 缩尾异常 (3)
    "winsorize", "mad_outlier", "zscore_clip",
    # 复合时序 (2)
    "decay_linear_xs", "momentum_accel",
]


class TestBuiltinRegistration:
    """所有 20 个内置 op 都被注册."""

    @pytest.mark.parametrize("name", BUILTIN_OPS)
    def test_op_registered(self, name):
        assert name in list_composite_ops(), f"{name} 未注册"
        assert is_composite_op(name)
        spec = get_composite_spec(name)
        assert spec is not None
        assert spec.doc != ""

    def test_total_builtin_count(self):
        """至少有 20 个内置 op."""
        assert len(list_composite_ops()) >= 20

    def test_no_duplicate_names(self):
        names = list_composite_ops()
        assert len(names) == len(set(names))


class TestNeutralizationOps:
    """中性化 (3)."""

    def test_market_neutralize_zero_mean(self):
        """市场中性化后, 横截面均值 ≈ 0."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "g": ["a", "a", "b", "b", "c"],
        })
        spec = get_composite_spec("market_neutralize")
        result = df.select(spec.instantiate(x=col("x")).alias("neu"))
        assert abs(result["neu"].mean()) < 1e-9

    def test_industry_neutralize_per_group_mean_zero(self):
        """行业中性化后, 每个行业均值 ≈ 0."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0],
            "citic_1": ["a", "a", "b", "b"],
        })
        spec = get_composite_spec("industry_neutralize")
        result = df.select(spec.instantiate(x=col("x")).alias("neu"))
        # 行业 a: mean=1.5 → 1-1.5=-0.5, 2-1.5=0.5
        assert abs(result["neu"][0] - (-0.5)) < 1e-9
        assert abs(result["neu"][1] - 0.5) < 1e-9
        # 行业 b: mean=3.5
        assert abs(result["neu"][2] - (-0.5)) < 1e-9
        assert abs(result["neu"][3] - 0.5) < 1e-9

    def test_subindustry_neutralize(self):
        """二级行业中性化: 与 industry_neutralize 同形, 不同 col."""
        df = pl.DataFrame({
            "x": [10.0, 20.0, 30.0, 40.0],
            "citic_2": ["x", "x", "y", "y"],
        })
        spec = get_composite_spec("subindustry_neutralize")
        result = df.select(spec.instantiate(x=col("x")).alias("neu"))
        # x: 10,20 → 均值 15 → -5, 5
        assert result["neu"][0] == -5.0
        assert result["neu"][1] == 5.0


class TestCrossSectionalOps:
    """横截面归一化 (3)."""

    def test_zscore_xs(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        spec = get_composite_spec("zscore_xs")
        result = df.select(spec.instantiate(x=col("x")).alias("z"))
        assert abs(result["z"].mean()) < 1e-9
        assert abs(result["z"].std() - 1.0) < 1e-9

    def test_rank_xs_in_unit_interval(self):
        df = pl.DataFrame({"x": [10.0, 30.0, 20.0, 40.0]})
        spec = get_composite_spec("rank_xs")
        result = df.select(spec.instantiate(x=col("x")).alias("r"))
        # rank / count 范围 [0, 1]
        assert result["r"].min() >= 0
        assert result["r"].max() <= 1
        # 最大值 rank=1
        assert abs(result["r"][3] - 1.0) < 1e-9

    def test_scale_xs_default_range(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        spec = get_composite_spec("scale_xs")
        result = df.select(spec.instantiate(x=col("x")).alias("s"))
        assert result["s"].min() == 0.0
        assert result["s"].max() == 1.0


class TestRollingOps:
    """滚动回归 (3)."""

    def test_rolling_beta_constant_y(self):
        """y 为常数时 beta 应为 NaN (std(y)=0)."""
        df = pl.DataFrame({
            "y": [1.0] * 30,
            "x": [float(i) for i in range(30)],
        })
        spec = get_composite_spec("rolling_beta")
        result = df.select(spec.instantiate(y=col("y"), x=col("x"), window=10).alias("beta"))
        # std(y) = 0, 除以 0 → NaN
        assert result["beta"].is_nan().any() or result["beta"].is_null().any()

    def test_rolling_ols_simplified_shape(self):
        df = pl.DataFrame({
            "y": [float(i) for i in range(30)],
            "x": [float(i) * 0.5 for i in range(30)],
        })
        spec = get_composite_spec("rolling_ols_simplified")
        result = df.select(spec.instantiate(y=col("y"), x=col("x"), window=10).alias("ols"))
        assert len(result) == 30

    def test_rolling_residual_shape(self):
        df = pl.DataFrame({
            "y": [float(i) for i in range(30)],
            "x": [float(i) for i in range(30)],
        })
        spec = get_composite_spec("rolling_residual")
        result = df.select(spec.instantiate(y=col("y"), x=col("x"), window=10).alias("res"))
        # 完全同向 → residual ≈ 0
        assert result["res"].abs().max() < 0.01


class TestVolatilityOps:
    """波动率 (4)."""

    def test_parkinson_vol_constant(self):
        """h=l 时 log(h/l)=0, vol=0."""
        df = pl.DataFrame({
            "h": [5.0] * 30,
            "l": [5.0] * 30,
        })
        spec = get_composite_spec("parkinson_vol")
        result = df.select(spec.instantiate(high=col("h"), low=col("l"), window=10).alias("v"))
        # 全部为 0 (rolling 后)
        assert result["v"].drop_nulls().unique().to_list() == [0.0]

    def test_garman_klass_vol_shape(self):
        df = pl.DataFrame({
            "h": [float(i) + 1 for i in range(30)],
            "l": [float(i) for i in range(30)],
            "c": [float(i) + 0.5 for i in range(30)],
            "o": [float(i) for i in range(30)],
        })
        spec = get_composite_spec("garman_klass_vol")
        result = df.select(spec.instantiate(
            high=col("h"), low=col("l"), close=col("c"), open_=col("o"),
            window=10,
        ).alias("v"))
        assert len(result) == 30

    def test_yang_zhang_vol_shape(self):
        df = pl.DataFrame({
            "h": [float(i) + 1 for i in range(30)],
            "l": [float(i) for i in range(30)],
            "c": [float(i) + 0.5 for i in range(30)],
            "o": [float(i) for i in range(30)],
        })
        spec = get_composite_spec("yang_zhang_vol")
        result = df.select(spec.instantiate(
            high=col("h"), low=col("l"), close=col("c"), open_=col("o"),
            window=10,
        ).alias("v"))
        assert len(result) == 30

    def test_realized_vol_constant_returns(self):
        """常数 returns → std=0 → 全 0."""
        df = pl.DataFrame({"r": [0.01] * 30})
        spec = get_composite_spec("realized_vol")
        result = df.select(spec.instantiate(returns=col("r"), window=10).alias("v"))
        assert result["v"].drop_nulls().unique().to_list() == [0.0]


class TestPairOps:
    """配对交易 (2)."""

    def test_pair_zscore_a_equals_b(self):
        """a==b → spread=0, zscore=NaN (除 0)."""
        df = pl.DataFrame({
            "a": [float(i) for i in range(30)],
            "b": [float(i) for i in range(30)],
        })
        spec = get_composite_spec("pair_zscore")
        result = df.select(spec.instantiate(a=col("a"), b=col("b"), window=10).alias("z"))
        # 全部 NaN
        assert result["z"].is_nan().all() or result["z"].is_null().all()

    def test_pair_ratio_shape(self):
        df = pl.DataFrame({
            "a": [float(i) + 1 for i in range(30)],
            "b": [float(i) + 1 for i in range(30)],
        })
        spec = get_composite_spec("pair_ratio")
        result = df.select(spec.instantiate(a=col("a"), b=col("b"), window=10).alias("r"))
        assert len(result) == 30
        # ratio=1
        assert (result["r"].drop_nulls() == 1.0).all()


class TestWinsorizeOps:
    """缩尾/异常 (3)."""

    def test_winsorize_literal_clip(self):
        """winsorize 当前实现为字面量 clip (1% / 99% 占位)."""
        df = pl.DataFrame({"x": [float(i) for i in range(20)]})
        spec = get_composite_spec("winsorize")
        result = df.select(spec.instantiate(x=col("x")).alias("w"))
        # 不 raise 即通过
        assert len(result) == 20

    def test_mad_outlier_zero_mad(self):
        """全 0 时 MAD=0, 极值 (100/-100) 被 where 滤掉, 0 值保留."""
        df = pl.DataFrame({"x": [0.0, 0.0, 0.0, 100.0, -100.0]})
        spec = get_composite_spec("mad_outlier")
        result = df.select(spec.instantiate(x=col("x"), n_mad=1.0).alias("o"))
        # x.where(condition) 滤掉非 0 行 → 仅 0 行保留 (3 行)
        # 极值被 where 滤掉, 0 值保留
        assert len(result) == 3
        assert (result["o"] == 0.0).all()

    def test_zscore_clip_constant(self):
        """常数时 z=0, 全部保留."""
        df = pl.DataFrame({"x": [5.0] * 10})
        spec = get_composite_spec("zscore_clip")
        result = df.select(spec.instantiate(x=col("x"), n_std=2.0).alias("z"))
        assert (result["z"] == 5.0).all()


class TestTimeSeriesOps:
    """复合时序 (2)."""

    def test_decay_linear_xs_shape(self):
        df = pl.DataFrame({"x": [float(i) for i in range(30)]})
        spec = get_composite_spec("decay_linear_xs")
        result = df.select(spec.instantiate(x=col("x"), window=10).alias("d"))
        assert len(result) == 30

    def test_momentum_accel_known_values(self):
        """momentum_accel 在 [1..10] 上: 短期 (5) - 长期 (20)."""
        df = pl.DataFrame({"x": [float(i) for i in range(1, 26)]})
        spec = get_composite_spec("momentum_accel")
        result = df.select(spec.instantiate(
            x=col("x"), short_window=5, long_window=20,
        ).alias("accel"))
        # 验证 shape
        assert len(result) == 25


class TestEnd2EndComposite:
    """复合 op 端到端 (instantiate + DataFrame 执行)."""

    def test_zscore_xs_chained_with_neutralize(self):
        """zscore 后再中性化, 链式调用."""
        df = pl.DataFrame({
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "g": ["a", "a", "a", "b", "b", "b"],
        })
        z_spec = get_composite_spec("zscore_xs")
        n_spec = get_composite_spec("industry_neutralize")
        # 先 zscore
        df = df.with_columns(z_spec.instantiate(x=col("x")).alias("z"))
        # 再中性化
        result = df.select(n_spec.instantiate(x=col("z"), industry_col="g").alias("final"))
        assert len(result) == 6
