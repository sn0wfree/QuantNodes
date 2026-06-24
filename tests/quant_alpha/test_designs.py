# coding=utf-8
"""Tests for QuantAlpha alpha101_design and alpha158_design (M3 PR).

覆盖：
- Alpha 101: 8 设计原则 + 16 核心算子 + A 股可移植性
- Alpha 101: 10 few-shot 示例 + 提示构造
- Alpha 158: 4 类特征模板 + 158 总数
- Alpha 158: 10 few-shot 示例 + 4 类覆盖
- Alpha 360: 6 字段 × 60 lookback = 360
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from QuantNodes.research.quant_alpha.alpha101_design import (
    DESIGN_PHILOSOPHY,
    CORE_OPERATORS,
    A_SHARE_COMPATIBILITY,
    ALPHA101_FEW_SHOT_EXAMPLES,
    list_examples as list_101_examples,
    get_example as get_101_example,
    get_few_shot_prompt as get_101_prompt,
    get_philosophy_by_id,
    get_operator_by_name,
    get_a_share_compatible_count,
    get_categories as get_101_categories,
)
from QuantNodes.research.quant_alpha.alpha158_design import (
    FEATURE_CATEGORIES,
    ALPHA360_TEMPLATE,
    ALPHA158_FEW_SHOT_EXAMPLES,
    DEFAULT_WINDOWS,
    list_examples as list_158_examples,
    get_example as get_158_example,
    get_few_shot_prompt as get_158_prompt,
    get_template_by_category,
    get_template_by_name,
    list_categories,
    total_feature_count,
    get_example_categories,
)


# ==============================================================================
# Test Class 1: Alpha 101 设计哲学
# ==============================================================================


class TestAlpha101DesignPhilosophy:
    """Alpha 101 设计哲学测试"""

    def test_8_philosophies(self):
        """应有 8 条设计原则"""
        assert len(DESIGN_PHILOSOPHY) == 8

    def test_philosophies_have_unique_ids(self):
        """所有原则 ID 唯一"""
        ids = [p.id for p in DESIGN_PHILOSOPHY]
        assert len(ids) == len(set(ids))

    def test_philosophies_have_examples(self):
        """每条原则都有 examples"""
        for p in DESIGN_PHILOSOPHY:
            assert p.name
            assert p.description
            assert len(p.examples) > 0, f"{p.id} 无 examples"

    def test_get_philosophy_by_id(self):
        """get_philosophy_by_id 工作"""
        p = get_philosophy_by_id("P3")
        assert p is not None
        assert p.id == "P3"
        # 不存在
        assert get_philosophy_by_id("P99") is None

    def test_core_operators_count(self):
        """16 个核心算子"""
        assert len(CORE_OPERATORS) == 16

    def test_core_operators_have_economic_meaning(self):
        """每个算子都有 economic_meaning"""
        for op in CORE_OPERATORS:
            assert op.name
            assert op.category
            assert op.economic_meaning
            assert 1 <= op.complexity <= 3

    def test_get_operator_by_name(self):
        """get_operator_by_name 工作"""
        op = get_operator_by_name("rank")
        assert op is not None
        assert op.name == "rank"
        assert op.category == "section"

    def test_a_share_compatibility_count(self):
        """A 股可移植矩阵存在"""
        assert len(A_SHARE_COMPATIBILITY) > 0
        # 至少有 4 个 Delay-0 不兼容
        n_incompatible = sum(
            1 for x in A_SHARE_COMPATIBILITY
            if not x.a_share_compatible
        )
        assert n_incompatible >= 4
        # 至少有 4 个可移植
        assert get_a_share_compatible_count() >= 4


# ==============================================================================
# Test Class 2: Alpha 101 few-shot 示例
# ==============================================================================


class TestAlpha101FewShot:
    """Alpha 101 few-shot 示例测试"""

    def test_10_examples(self):
        """应有 10 个示例"""
        assert len(ALPHA101_FEW_SHOT_EXAMPLES) == 10

    def test_all_examples_a_share_compatible(self):
        """所有示例应 A 股可移植"""
        for e in ALPHA101_FEW_SHOT_EXAMPLES:
            assert e.a_share_compatible is True
            assert e.formula
            assert e.description
            assert e.alpha101_ref

    def test_examples_cover_categories(self):
        """示例覆盖 4 个 category"""
        cats = set(get_101_categories())
        assert "momentum" in cats
        assert "reversal" in cats
        assert "volume_price" in cats
        assert "intraday" in cats

    def test_get_example_by_id(self):
        """get_example 工作"""
        ex1 = get_101_example("EX1")
        assert ex1 is not None
        assert ex1.id == "EX1"
        assert "EX99" not in [e.id for e in ALPHA101_FEW_SHOT_EXAMPLES]

    def test_get_example_unknown(self):
        """get_example 未知 ID 返回 None"""
        assert get_101_example("EX99") is None

    def test_list_examples_filter_by_category(self):
        """list_examples 按 category 过滤"""
        mom = list_101_examples(category="momentum")
        assert all(e.category == "momentum" for e in mom)
        assert len(mom) > 0

    def test_few_shot_prompt(self):
        """get_few_shot_prompt 构造 prompt"""
        prompt = get_101_prompt(n=3)
        assert "EX1" in prompt
        assert "formula:" in prompt
        # n=3 应包含前 3 个
        assert "EX2" in prompt
        assert "EX3" in prompt
        # 不应包含第 4 个
        assert "EX4" not in prompt

    def test_examples_reference_design_principles(self):
        """示例应引用设计原则"""
        all_principle_ids = {p.id for p in DESIGN_PHILOSOPHY}
        for e in ALPHA101_FEW_SHOT_EXAMPLES:
            for pid in e.design_principles:
                assert pid in all_principle_ids, f"{e.id} 引用未知原则 {pid}"


# ==============================================================================
# Test Class 3: Alpha 158 特征模板
# ==============================================================================


class TestAlpha158Categories:
    """Alpha 158 4 类特征模板测试"""

    def test_4_categories(self):
        """应有 4 类"""
        assert len(FEATURE_CATEGORIES) == 4

    def test_total_features_equals_158(self):
        """4 类总特征数 = 158"""
        assert total_feature_count() == 158

    def test_category_breakdown(self):
        """4 类特征数分配正确"""
        # KBAR=9 + Price=20 + Volume=5 + Rolling=124 = 158
        breakdown = {t.category_id: t.total_features for t in FEATURE_CATEGORIES}
        assert breakdown["KBAR"] == 9
        assert breakdown["Price"] == 20
        assert breakdown["Volume"] == 5
        assert breakdown["Rolling"] == 124

    def test_kbar_philosophy(self):
        """KBAR 设计哲学描述"""
        kbar = get_template_by_category("KBAR")
        assert kbar is not None
        assert "几何" in kbar.philosophy or "K线" in kbar.philosophy
        assert kbar.total_features == 9

    def test_price_philosophy_no_cross_section(self):
        """Price 特征无截面算子"""
        price = get_template_by_category("Price")
        assert price is not None
        assert "无截面" in price.philosophy or "no cross" in price.philosophy.lower()
        # Price 4 字段 × 5 延迟 = 20
        assert price.parameters["field"] == ["open", "high", "low", "vwap"]
        assert price.parameters["delay"] == [0, 1, 2, 3, 4]

    def test_volume_philosophy(self):
        """Volume 5 特征"""
        vol = get_template_by_category("Volume")
        assert vol is not None
        assert vol.total_features == 5

    def test_rolling_25_ops_x_5_windows(self):
        """Rolling = 25 op × 5 window = 124 特征（去重 1 个）"""
        rolling = get_template_by_category("Rolling")
        assert rolling is not None
        # 25 ops (但部分共享 window，所以 24*5+1 实际=124)
        n_ops = len(rolling.parameters["op"])
        n_windows = len(rolling.parameters["window"])
        # 25 op × 5 window = 125，但 IMXD = IMAX - IMIN 共享，所以 124
        assert rolling.total_features == n_ops * n_windows - 1  # 125-1=124

    def test_list_categories(self):
        """list_categories 返回所有 4 类"""
        cats = list_categories()
        assert set(cats) == {"KBAR", "Price", "Volume", "Rolling"}

    def test_get_template_by_name(self):
        """get_template_by_name 工作"""
        kbar = get_template_by_name("K线形态")
        assert kbar is not None
        assert kbar.category_id == "KBAR"


# ==============================================================================
# Test Class 4: Alpha 360 模板
# ==============================================================================


class TestAlpha360Template:
    """Alpha 360 模板测试"""

    def test_360_features(self):
        """Alpha 360 = 360 特征"""
        assert ALPHA360_TEMPLATE.total_features == 360

    def test_6_fields(self):
        """6 个字段"""
        assert len(ALPHA360_TEMPLATE.fields) == 6
        assert "close" in ALPHA360_TEMPLATE.fields
        assert "open" in ALPHA360_TEMPLATE.fields
        assert "high" in ALPHA360_TEMPLATE.fields
        assert "low" in ALPHA360_TEMPLATE.fields
        assert "vwap" in ALPHA360_TEMPLATE.fields
        assert "volume" in ALPHA360_TEMPLATE.fields

    def test_60_lookback(self):
        """60 lookback (0-59)"""
        assert len(ALPHA360_TEMPLATE.lookback_range) == 60
        assert 0 in ALPHA360_TEMPLATE.lookback_range
        assert 59 in ALPHA360_TEMPLATE.lookback_range

    def test_6_x_60_equals_360(self):
        """6 × 60 = 360 验证"""
        assert (
            len(ALPHA360_TEMPLATE.fields)
            * len(ALPHA360_TEMPLATE.lookback_range)
            == ALPHA360_TEMPLATE.total_features
        )


# ==============================================================================
# Test Class 5: Alpha 158 few-shot 示例
# ==============================================================================


class TestAlpha158FewShot:
    """Alpha 158 few-shot 示例测试"""

    def test_10_examples(self):
        """应有 10 个示例"""
        assert len(ALPHA158_FEW_SHOT_EXAMPLES) == 10

    def test_examples_cover_all_4_categories(self):
        """覆盖 4 类（KBAR / Price / Volume / Rolling）"""
        cats = set(get_example_categories())
        assert cats == {"KBAR", "Price", "Volume", "Rolling"}

    def test_kbar_examples_count(self):
        """KBAR 应有 2 个示例（FX1, FX2）"""
        kbar = list_158_examples(category="KBAR")
        assert len(kbar) == 2

    def test_rolling_examples_count(self):
        """Rolling 应有 5 个示例（FX6-FX10）"""
        rolling = list_158_examples(category="Rolling")
        assert len(rolling) == 5

    def test_get_example_by_id(self):
        """get_example 工作"""
        ex1 = get_158_example("FX1")
        assert ex1 is not None
        assert ex1.id == "FX1"
        assert "KMID" in ex1.name

    def test_get_example_unknown(self):
        """get_example 未知 ID 返回 None"""
        assert get_158_example("FX99") is None

    def test_few_shot_prompt(self):
        """get_few_shot_prompt 构造 prompt"""
        prompt = get_158_prompt(n=5)
        # 包含 EX FX1-FX5
        for i in range(1, 6):
            assert f"FX{i}" in prompt
        assert "formula:" in prompt

    def test_few_shot_prompt_filtered(self):
        """get_few_shot_prompt 按 category 过滤"""
        prompt = get_158_prompt(n=2, category="KBAR")
        # KBAR 类只有 FX1, FX2
        assert "FX1" in prompt
        assert "FX2" in prompt
        # 不应包含 Rolling 类的 FX6
        assert "FX6" not in prompt

    def test_examples_reference_qlib(self):
        """示例应引用 Qlib 命名"""
        for e in ALPHA158_FEW_SHOT_EXAMPLES:
            assert e.qlib_ref.startswith("$"), f"{e.id} qlib_ref 应以 $ 开头"

    def test_default_windows(self):
        """DEFAULT_WINDOWS 包含 5/10/20/30/60"""
        for w in [5, 10, 20, 30, 60]:
            assert w in DEFAULT_WINDOWS
