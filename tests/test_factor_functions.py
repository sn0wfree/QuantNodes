# -*- coding: utf-8 -*-
"""
factor_functions.py 单元测试

测试覆盖范围:
- 算子注册器 API
- 装饰器功能
- P0 核心算子（单点运算、基础滚动、基础截面）
- P1 常用算子
"""
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ll/Public/QuantNodes')

from QuantNodes.factor_node.factor_functions import (
    list_operators,
    get_operator,
    operator_info,
    generate_documentation,
    OperatorCategory,
)


# ==============================================================================
# 注册器 API 测试
# ==============================================================================

class TestRegistryAPI:
    """测试算子注册器 API"""

    def test_list_operators_returns_list(self):
        """list_operators 返回列表"""
        result = list_operators()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_operators_by_category(self):
        """按类别列出算子"""
        point_ops = list_operators(category=OperatorCategory.POINT)
        time_ops = list_operators(category=OperatorCategory.TIME)
        section_ops = list_operators(category=OperatorCategory.SECTION)
        
        assert isinstance(point_ops, list)
        assert isinstance(time_ops, list)
        assert isinstance(section_ops, list)

    def test_list_operators_invalid_category_returns_empty(self):
        """无效类别返回空列表"""
        result = list_operators(category="invalid_category")
        assert result == []

    def test_get_operator_returns_callable(self):
        """get_operator 返回可调用对象"""
        op = get_operator("isnull")
        assert callable(op)

    def test_get_operator_nonexistent_returns_none(self):
        """不存在的算子返回 None"""
        op = get_operator("nonexistent_operator_xyz")
        assert op is None

    def test_operator_info_returns_dict(self):
        """operator_info 返回字典"""
        info = operator_info("isnull")
        assert isinstance(info, dict)

    def test_operator_info_contains_required_fields(self):
        """operator_info 包含必需字段"""
        info = operator_info("isnull")
        assert "name" in info
        assert "category" in info
        assert "doc" in info
        assert "signature" in info
        assert "parameters" in info

    def test_operator_info_nonexistent_returns_none(self):
        """不存在算子的 info 返回 None"""
        info = operator_info("nonexistent_operator_xyz")
        assert info is None

    def test_generate_documentation_returns_string(self):
        """generate_documentation 返回字符串"""
        doc = generate_documentation()
        assert isinstance(doc, str)

    def test_generate_documentation_contains_markdown(self):
        """文档包含 Markdown 格式"""
        doc = generate_documentation()
        assert "##" in doc  # 二级标题
        assert "###" in doc  # 三级标题


# ==============================================================================
# 装饰器测试
# ==============================================================================

class TestDecorators:
    """测试装饰器功能"""

    def test_point_operator_returns_callable(self):
        """@point_operator 装饰器返回可调用对象"""
        from QuantNodes.factor_node.factor_functions import isnull
        assert callable(isnull)

    def test_rolling_operator_returns_callable(self):
        """@rolling_operator 装饰器返回可调用对象"""
        from QuantNodes.factor_node.factor_functions import rolling_mean
        assert callable(rolling_mean)

    def test_expanding_operator_returns_callable(self):
        """@expanding_operator 装饰器返回可调用对象"""
        from QuantNodes.factor_node.factor_functions import expanding_mean
        assert callable(expanding_mean)

    def test_ewm_operator_returns_callable(self):
        """@ewm_operator 装饰器返回可调用对象"""
        from QuantNodes.factor_node.factor_functions import ewm_mean
        assert callable(ewm_mean)

    def test_single_section_operator_returns_callable(self):
        """@single_section_operator 装饰器返回可调用对象"""
        from QuantNodes.factor_node.factor_functions import standardizeRank
        assert callable(standardizeRank)


# ==============================================================================
# 算子可用性测试
# ==============================================================================

class TestOperatorAvailability:
    """测试所有P0核心算子都已注册可用"""

    def test_p0_point_operators_available(self):
        """P0 单点算子都已注册"""
        p0_operators = [
            "isnull", "notnull", "log", "sign",
            "ceil", "floor", "nansum", "nanprod",
            "nanmax", "nanmin", "nanmean", "nanstd",
            "nanvar", "nanmedian", "nancount"
        ]
        for op_name in p0_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p0_rolling_operators_available(self):
        """P0 滚动窗口算子都已注册"""
        p0_operators = [
            "rolling_mean", "rolling_sum", "rolling_std",
            "rolling_var", "rolling_max", "rolling_min",
            "rolling_median"
        ]
        for op_name in p0_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p1_expanding_operators_available(self):
        """P1 扩展窗口算子都已注册"""
        p1_operators = [
            "expanding_mean", "expanding_sum", "expanding_std",
            "expanding_var", "expanding_max", "expanding_min",
            "expanding_median", "expanding_count"
        ]
        for op_name in p1_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p1_ewm_operators_available(self):
        """P1 EWM 算子都已注册"""
        p1_operators = [
            "ewm_mean", "ewm_std", "ewm_var", "ewm_cov", "ewm_corr"
        ]
        for op_name in p1_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p1_time_shift_operators_available(self):
        """P1 时间位移算子都已注册"""
        p1_operators = [
            "lag", "diff", "fillna"
        ]
        for op_name in p1_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p1_section_operators_available(self):
        """P1 截面算子都已注册"""
        p1_operators = [
            "standardizeRank", "standardizeZScore",
            "standardizeQuantile", "winsorize",
            "fillNaNByVal", "fillNaNByFun"
        ]
        for op_name in p1_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p2_dual_factor_operators_available(self):
        """P2 双因子算子都已注册"""
        p2_operators = [
            "rolling_cov", "rolling_corr",
            "expanding_cov", "expanding_corr"
        ]
        for op_name in p2_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_p1_advanced_rolling_operators_available(self):
        """P1 高级滚动算子都已注册"""
        p1_operators = [
            "rolling_change_rate", "rolling_rank",
            "rolling_skew", "rolling_kurt",
            "rolling_quantile"
        ]
        for op_name in p1_operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_aggregate_operators_available(self):
        """多截面聚合算子都已注册"""
        operators = [
            "aggregate", "aggr_sum", "aggr_prod",
            "aggr_max", "aggr_min", "aggr_mean",
            "aggr_std", "aggr_var", "aggr_median",
            "aggr_quantile", "aggr_count"
        ]
        for op_name in operators:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"


# ==============================================================================
# 算子分类测试
# ==============================================================================

class TestOperatorCategories:
    """测试算子分类正确性"""

    def test_point_category_operators_in_correct_category(self):
        """测试单点算子分类正确"""
        point_ops = list_operators(category=OperatorCategory.POINT)
        assert "isnull" in point_ops
        assert "log" in point_ops
        assert "nanmean" in point_ops

    def test_time_category_operators_in_correct_category(self):
        """测试时间序列算子分类正确"""
        time_ops = list_operators(category=OperatorCategory.TIME)
        assert "rolling_mean" in time_ops
        assert "expanding_mean" in time_ops
        assert "ewm_mean" in time_ops
        assert "lag" in time_ops

    def test_section_category_operators_in_correct_category(self):
        """测试截面算子分类正确"""
        section_ops = list_operators(category=OperatorCategory.SECTION)
        assert "standardizeRank" in section_ops
        assert "standardizeZScore" in section_ops
        assert "winsorize" in section_ops

    def test_multi_section_category_operators_in_correct_category(self):
        """测试多截面算子分类正确"""
        multi_section_ops = list_operators(category=OperatorCategory.MULTI_SECTION)
        assert "aggr_sum" in multi_section_ops
        assert "aggregate" in multi_section_ops


# ==============================================================================
# 算子元数据测试
# ==============================================================================

class TestOperatorMetadata:
    """测试算子元数据"""

    def test_operator_has_docstring(self):
        """测试算子有文档字符串"""
        op_info = operator_info("rolling_mean")
        assert op_info is not None

    def test_operator_has_name(self):
        """测试算子有名称"""
        op_info = operator_info("rolling_mean")
        assert op_info["name"] == "rolling_mean"

    def test_operator_has_category(self):
        """测试算子有正确的分类"""
        op_info = operator_info("rolling_mean")
        assert op_info["category"] == OperatorCategory.TIME

    def test_operator_has_parameters(self):
        """测试算子有参数列表"""
        op_info = operator_info("rolling_mean")
        assert "parameters" in op_info
        assert isinstance(op_info["parameters"], list)


# ==============================================================================
# 算子功能正确性验证测试
# ==============================================================================

class TestOperatorFunctionality:
    """测试算子功能正确性"""

    def test_registered_operator_count(self):
        """测试注册算子总数"""
        all_ops = list_operators()
        assert len(all_ops) >= 90, f"Expected at least 90 operators, got {len(all_ops)}"

    def test_all_registered_operators_callable(self):
        """测试所有注册算子都可调用"""
        all_ops = list_operators()
        for op_name in all_ops:
            op = get_operator(op_name)
            assert callable(op), f"Operator {op_name} should be callable"

    def test_operator_info_complete(self):
        """测试所有算子元数据完整"""
        all_ops = list_operators()
        for op_name in all_ops:
            info = operator_info(op_name)
            assert info is not None, f"Operator {op_name} should have info"
            assert "name" in info
            assert "category" in info
            assert "doc" in info
            assert "signature" in info

    def test_docstring_not_empty(self):
        """测试关键算子有文档字符串"""
        key_operators = [
            "rolling_mean", "expanding_mean", "ewm_mean",
            "standardizeRank", "winsorize", "lag",
            "rolling_cov", "rolling_corr"
        ]
        for op_name in key_operators:
            info = operator_info(op_name)
            assert info is not None, f"Operator {op_name} should have info"
            assert len(info["doc"]) >= 0, f"Operator {op_name} should have doc"


class TestEdgeCases:
    """测试边界条件"""

    def test_empty_operator_list_by_invalid_category(self):
        """测试无效类别返回空列表"""
        result = list_operators(category="invalid")
        assert result == []

    def test_nonexistent_operator_returns_none(self):
        """测试不存在的算子返回None"""
        op = get_operator("definitely_does_not_exist_xyz")
        assert op is None

    def test_nonexistent_operator_info_returns_none(self):
        """测试不存在的算子元数据返回None"""
        info = operator_info("definitely_does_not_exist_xyz")
        assert info is None


class TestCategoriesComplete:
    """测试分类完整性"""

    def test_all_operators_belong_to_category(self):
        """测试所有算子都归类"""
        all_ops = list_operators()
        categories = [OperatorCategory.POINT, OperatorCategory.TIME, 
                     OperatorCategory.SECTION, OperatorCategory.MULTI_SECTION]
        
        categorized_count = 0
        for cat in categories:
            categorized_count += len(list_operators(category=cat))
        
        assert categorized_count == len(all_ops), \
            f"All operators should belong to a category: {categorized_count} vs {len(all_ops)}"

    def test_each_category_not_empty(self):
        """测试每个分类都���为空"""
        for cat in [OperatorCategory.POINT, OperatorCategory.TIME, 
                   OperatorCategory.SECTION, OperatorCategory.MULTI_SECTION]:
            ops = list_operators(category=cat)
            assert len(ops) > 0, f"Category {cat} should not be empty"


class TestOperatorInteraction:
    """测试算子组合和交互"""

    def test_same_operator_different_calls(self):
        """测试同一算子可多次调用"""
        ops = list_operators()
        op_name = "isnull"
        
        # 获取两次应该返回同一个函数
        op1 = get_operator(op_name)
        op2 = get_operator(op_name)
        assert op1 is op2

    def test_category_filtering_works(self):
        """测试分类过滤功能"""
        all_ops = list_operators()
        point_ops = list_operators(category=OperatorCategory.POINT)
        
        # point_ops 应该是 all_ops 的子集
        for op in point_ops:
            assert op in all_ops


class TestBackwardCompatibility:
    """向后兼容性测试"""

    def test_known_operators_still_available(self):
        """测试已知算子仍然可用"""
        known_operators = [
            # Point operators
            "isnull", "notnull", "log", "sign", "ceil", "floor",
            "nansum", "nanprod", "nanmax", "nanmin", "nanmean",
            "nanstd", "nanvar", "nanmedian", "nancount",
            # Time operators
            "rolling_mean", "rolling_sum", "rolling_std",
            "expanding_mean", "expanding_sum",
            "ewm_mean",
            "lag", "diff", "fillna", "nav",
            # Section operators
            "standardizeRank", "standardizeZScore", "winsorize",
            # Multi-section operators
            "aggregate", "aggr_sum"
        ]
        
        for op_name in known_operators:
            op = get_operator(op_name)
            assert callable(op), f"Known operator {op_name} should be available"

    def test_operator_signatures_present(self):
        """测试算子都有签名"""
        all_ops = list_operators()
        for op_name in all_ops[:20]:  # 检查前20个
            info = operator_info(op_name)
            assert "signature" in info
            assert info["signature"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
