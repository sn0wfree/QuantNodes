# coding=utf-8
"""portfolio_construction 单元测试.

覆盖:
    1. sector_diversified_pool() 板块分散化品种池
    2. pool_sector_breakdown() 板块分布
    3. split_is_oos() IS/OOS 划分
    4. is_oos_validation_ratio() 验证比
    5. make_period_config() 多周期配置生成
    6. combine_period_navs() 多周期 NAV 组合
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.strategies.portfolio_construction import (
    DEFAULT_DIVERSIFIED_POOL,
    LIQUIDITY_TOP_N,
    PERIOD_PRESETS,
    SECTOR_PRODUCTS,
    combine_period_navs,
    is_oos_validation_ratio,
    make_period_config,
    pool_sector_breakdown,
    sector_diversified_pool,
    split_is_oos,
)


# ── 板块分散化品种池测试 ─────────────────────────────────────

def test_default_pool_covers_all_sectors():
    """默认品种池覆盖所有目标板块."""
    breakdown = pool_sector_breakdown(DEFAULT_DIVERSIFIED_POOL)
    # 7 大板块 (除股指 IF 因过拟合风险被排除)
    expected_sectors = {"贵金属", "有色金属", "黑色", "能化", "农产品", "国债"}
    assert set(breakdown.keys()) == expected_sectors


def test_default_pool_size_in_range():
    """默认品种池规模 12 ± 3."""
    assert 9 <= len(DEFAULT_DIVERSIFIED_POOL) <= 15


def test_sector_diversified_pool_no_performance_filter():
    """验证: 不基于业绩筛选 (函数应无 Calmar/Sharpe 等输入)."""
    import inspect
    sig = inspect.signature(sector_diversified_pool)
    params = list(sig.parameters.keys())
    # 应只允许配置 top_n/exclude, 没有任何业绩类输入
    forbidden = {"calmar", "sharpe", "performance", "metric"}
    for p in params:
        assert not any(f in p.lower() for f in forbidden), f"param {p} suggests perf filter"


def test_sector_diversified_pool_default():
    """默认调用应返回与 DEFAULT_DIVERSIFIED_POOL 一致的结果."""
    pool = sector_diversified_pool()
    assert pool == DEFAULT_DIVERSIFIED_POOL


def test_sector_diversified_pool_exclude_sector():
    """可排除板块."""
    pool = sector_diversified_pool(exclude_sectors=["能化"])
    assert "SC" not in pool
    assert "TA" not in pool
    assert "MA" not in pool
    # 其他板块仍保留
    assert "AU" in pool
    assert "CU" in pool


def test_sector_diversified_pool_exclude_product():
    """可排除具体品种."""
    pool = sector_diversified_pool(exclude_products=["AU", "CU"])
    assert "AU" not in pool
    assert "CU" not in pool
    # 其他仍保留
    assert "AL" in pool


def test_sector_diversified_pool_custom_top_n():
    """可自定义 Top-N."""
    pool = sector_diversified_pool(
        top_n_per_sector={"贵金属": 2, "有色金属": 1},
    )
    assert "AU" in pool and "AG" in pool  # 贵金属 2 个
    assert "CU" in pool and "AL" not in pool  # 有色金属仅 1 个


def test_pool_sector_breakdown_returns_dict():
    """breakdown 应返回 dict, 值是 list."""
    bd = pool_sector_breakdown(["AU", "CU"])
    assert isinstance(bd, dict)
    assert bd["贵金属"] == ["AU"]
    assert bd["有色金属"] == ["CU"]


def test_pool_sector_breakdown_empty():
    """空品种池 → 空 dict."""
    assert pool_sector_breakdown([]) == {}


# ── IS/OOS 划分测试 ─────────────────────────────────────────

def test_split_is_oos_default():
    """默认 5 年 IS + 剩余 OOS."""
    is_s, is_e, oos_s, oos_e = split_is_oos(
        start="2020-01-01", end="2026-09-04", is_years=5,
    )
    assert is_s == "2020-01-01"
    assert is_e == "2024-12-31"
    assert oos_s == "2025-01-01"
    assert oos_e == "2026-09-04"


def test_split_is_oos_4_years():
    """4 年 IS 测试."""
    is_s, is_e, oos_s, oos_e = split_is_oos(
        start="2020-01-01", end="2026-09-04", is_years=4,
    )
    assert is_e == "2023-12-31"
    assert oos_s == "2024-01-01"


def test_split_is_oos_continuous():
    """IS 结束日期次日 = OOS 开始日期."""
    _, is_e, oos_s, _ = split_is_oos()
    is_end_dt = pd.Timestamp(is_e)
    oos_start_dt = pd.Timestamp(oos_s)
    assert (oos_start_dt - is_end_dt).days == 1


# ── IS/OOS 验证比测试 ───────────────────────────────────────

def test_is_oos_validation_robust():
    """比率 > 0.5 → robust."""
    r = is_oos_validation_ratio(2.0, 1.2)
    assert r["ratio"] == 0.6
    assert r["robust"] is True
    assert r["severity"] == "robust"


def test_is_oos_validation_mild_overfit():
    """比率 0-0.5 → mild_overfit."""
    r = is_oos_validation_ratio(2.0, 0.8)
    assert r["ratio"] == 0.4
    assert r["robust"] is False
    assert r["severity"] == "mild_overfit"


def test_is_oos_validation_moderate_overfit():
    """比率 -0.5-0 → moderate_overfit."""
    r = is_oos_validation_ratio(2.0, -0.5)
    assert r["robust"] is False
    assert r["severity"] == "moderate_overfit"


def test_is_oos_validation_severe_overfit():
    """比率 < -0.5 → severe_overfit."""
    r = is_oos_validation_ratio(2.0, -2.0)
    assert r["ratio"] == -1.0
    assert r["severity"] == "severe_overfit"


def test_is_oos_validation_zero_is():
    """IS 为 0 时的边界情况."""
    r = is_oos_validation_ratio(0.0, 0.0)
    assert r["ratio"] == float("inf")


def test_is_oos_validation_custom_threshold():
    """可自定义阈值."""
    r = is_oos_validation_ratio(1.0, 0.6, threshold=0.6)
    assert r["robust"] is True
    r2 = is_oos_validation_ratio(1.0, 0.5, threshold=0.6)
    assert r2["robust"] is False


# ── 多周期配置生成测试 ─────────────────────────────────────

def test_make_period_config_short():
    """短周期配置."""
    from QuantNodes.strategy.cta.strategies.huatai_timing_cta import HuataiTimingConfig
    base = HuataiTimingConfig()
    cfg = make_period_config(base, "short")
    assert cfg.confirm_window == 5
    assert cfg.ema_slow == 30
    assert cfg.max_holding_days == 30
    assert cfg.chandelier_window == 15


def test_make_period_config_long():
    """长周期配置."""
    from QuantNodes.strategy.cta.strategies.huatai_timing_cta import HuataiTimingConfig
    base = HuataiTimingConfig()
    cfg = make_period_config(base, "long")
    assert cfg.confirm_window == 20
    assert cfg.ema_slow == 120
    assert cfg.max_holding_days == 120
    assert cfg.chandelier_window == 35


def test_make_period_config_chandelier_adjusted():
    """chandelier_mult 应被 ± 调整 (short 紧, long 宽)."""
    from QuantNodes.strategy.cta.strategies.huatai_timing_cta import HuataiTimingConfig
    base = HuataiTimingConfig()
    base_mult_au = base.chandelier_mult["AU"]
    cfg_short = make_period_config(base, "short")
    cfg_long = make_period_config(base, "long")
    assert cfg_short.chandelier_mult["AU"] < base_mult_au
    assert cfg_long.chandelier_mult["AU"] > base_mult_au


def test_make_period_config_invalid():
    """非法 period → ValueError."""
    from QuantNodes.strategy.cta.strategies.huatai_timing_cta import HuataiTimingConfig
    base = HuataiTimingConfig()
    with pytest.raises(ValueError, match="period"):
        make_period_config(base, "invalid")


def test_make_period_config_does_not_mutate_base():
    """应 deepcopy, 不修改原配置."""
    from QuantNodes.strategy.cta.strategies.huatai_timing_cta import HuataiTimingConfig
    base = HuataiTimingConfig()
    original_cw = base.confirm_window
    make_period_config(base, "short")
    assert base.confirm_window == original_cw


# ── 多周期 NAV 组合测试 ─────────────────────────────────────

@pytest.fixture
def three_period_navs() -> dict[str, pd.Series]:
    """3 个周期合成 NAV."""
    n = 252
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    return {
        "short": pd.Series(np.cumprod(1 + rng.normal(0.001, 0.01, n)), index=idx),
        "medium": pd.Series(np.cumprod(1 + rng.normal(0.0008, 0.008, n)), index=idx),
        "long": pd.Series(np.cumprod(1 + rng.normal(0.0006, 0.006, n)), index=idx),
    }


def test_combine_period_navs_equal_weight(three_period_navs):
    """等权组合: 起末值应等于 1/3 权重的加权."""
    combined = combine_period_navs(three_period_navs)
    # 验证 NAV 起点 ≈ 1 (各 NAV 归一化)
    assert abs(combined.iloc[0] - 1.0) < 0.01
    # 长度对齐
    assert len(combined) == len(three_period_navs["short"])


def test_combine_period_navs_custom_weight(three_period_navs):
    """自定义权重: 中周期权重更高."""
    combined = combine_period_navs(
        three_period_navs,
        weights={"short": 0.2, "medium": 0.5, "long": 0.3},
    )
    assert len(combined) > 0


def test_combine_period_navs_weight_normalization(three_period_navs):
    """权重和不为 1 时自动归一化."""
    combined = combine_period_navs(
        three_period_navs,
        weights={"short": 1.0, "medium": 1.0, "long": 1.0},  # 和 = 3
    )
    # 应归一化为 1/3 each, 等权结果应与默认权重一致
    eq = combine_period_navs(three_period_navs)
    pd.testing.assert_series_equal(combined, eq)


def test_combine_period_navs_empty():
    """空 dict → ValueError."""
    with pytest.raises(ValueError, match="navs"):
        combine_period_navs({})


def test_combine_period_navs_zero_weight(three_period_navs):
    """权重和 ≤ 0 → ValueError."""
    with pytest.raises(ValueError, match="weights"):
        combine_period_navs(three_period_navs, weights={"short": 0, "medium": 0, "long": 0})


# ── 周期预设完整性测试 ───────────────────────────────────────

def test_all_period_presets_have_required_keys():
    """所有周期预设必须包含必要参数."""
    required = {
        "confirm_window", "ema_slow", "max_holding_days",
        "chandelier_window", "chandelier_mult_adjust", "description",
    }
    for name, preset in PERIOD_PRESETS.items():
        assert set(preset.keys()) >= required, f"{name} missing keys"


def test_period_presets_have_3_periods():
    """应有 3 个周期预设."""
    assert set(PERIOD_PRESETS.keys()) == {"short", "medium", "long"}


def test_sector_products_covers_all_common():
    """板块分类应覆盖常用品种."""
    expected = {"AU", "AG", "CU", "AL", "ZN", "NI", "RB", "I", "SC", "TA", "MA", "M", "Y", "T", "IF"}
    all_products = set()
    for products in SECTOR_PRODUCTS.values():
        all_products.update(products)
    assert expected.issubset(all_products)


def test_liquidity_top_n_values_non_negative():
    """Top-N 字典值必须 >= 0 (允许 0 表示该板块不纳入默认池)."""
    for sector, n in LIQUIDITY_TOP_N.items():
        assert n >= 0, f"{sector} has invalid n={n}"
