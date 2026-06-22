# coding=utf-8
"""Tests for register_node_config decorator (Phase 1.4).

Covers:
  - Decorator registers class to NODE_CONFIG_SCHEMAS
  - 12 node configs auto-registered
  - Decorator handles re-registration of same class (idempotent)
  - Decorator raises on different class for same name
  - Decorator validates BaseModel subclass
  - Decorator preserves class identity (returned class is original)
"""
from typing import Optional

import pytest
from pydantic import BaseModel, Field

from QuantNodes.research.factor_test.nodes.configs import (
    NODE_CONFIG_SCHEMAS,
    register_node_config,
)


@pytest.fixture(autouse=True)
def isolate_registry():
    """隔离 NODE_CONFIG_SCHEMAS: 测试前后快照, 测试产生的条目会被清除。

    防止测试污染其他依赖 len(NODE_CONFIG_SCHEMAS) == 12 的测试
    (如 tests/research/test_node_configs.py::TestNodeConfigSchemas::test_schema_count)。
    """
    snapshot = dict(NODE_CONFIG_SCHEMAS)
    yield
    # 清除所有非原始条目
    for key in list(NODE_CONFIG_SCHEMAS.keys()):
        if key not in snapshot:
            del NODE_CONFIG_SCHEMAS[key]


# ---------------------------------------------------------------------------
# Auto-registration of 12 nodes
# ---------------------------------------------------------------------------

class TestTwelveNodesAutoRegistered:
    def test_all_12_nodes_present(self):
        # 只在 isolate_registry fixture 启动后 (即只有原始条目) 验证
        assert len(NODE_CONFIG_SCHEMAS) == 12

    def test_expected_node_names(self):
        expected = {
            "LoadData", "SamplePoolFilter", "TradabilityFilter", "AdjustDate",
            "FactorPreprocess", "FactorNeutralize", "ICAnalyzer",
            "GroupAnalyzer", "LongShort", "FactorScore",
            "RiskCorrelation", "FactorTestReport",
        }
        assert set(NODE_CONFIG_SCHEMAS.keys()) == expected

    def test_all_entries_are_basemodel_subclasses(self):
        for name, cls in NODE_CONFIG_SCHEMAS.items():
            assert issubclass(cls, BaseModel), f"{name} -> {cls}"


# ---------------------------------------------------------------------------
# Decorator semantics
# ---------------------------------------------------------------------------

class TestRegisterNodeConfig:
    def test_decorator_returns_same_class(self):
        @register_node_config("__test_decorator_returns__")
        class MyConfig(BaseModel):
            x: int = 0
        # Class is in registry
        assert NODE_CONFIG_SCHEMAS["__test_decorator_returns__"] is MyConfig
        # Decorator returned the same class (not a wrapper)
        assert MyConfig.__name__ == "MyConfig"

    def test_re_registration_same_class_is_idempotent(self):
        @register_node_config("__test_idempotent__")
        class IdempotentConfig(BaseModel):
            y: str = ""
        # Re-decorate the same class → no error
        decorated_again = register_node_config("__test_idempotent__")(IdempotentConfig)
        assert decorated_again is IdempotentConfig
        assert NODE_CONFIG_SCHEMAS["__test_idempotent__"] is IdempotentConfig

    def test_re_registration_different_class_raises(self):
        @register_node_config("__test_collision__")
        class FirstConfig(BaseModel):
            a: int = 0
        # Try to register a different class to the same name
        with pytest.raises(ValueError, match="already registered"):
            @register_node_config("__test_collision__")
            class SecondConfig(BaseModel):
                b: int = 0

    def test_non_basemodel_raises_typeerror(self):
        with pytest.raises(TypeError, match="requires a pydantic BaseModel"):
            @register_node_config("__test_invalid__")
            class NotPydantic:  # not a BaseModel subclass
                x: int = 0

    def test_instance_creation_works_through_registry(self):
        """End-to-end: 装饰器注册的 class 可以正常实例化。"""
        @register_node_config("__test_instance__")
        class InstantiableConfig(BaseModel):
            threshold: float = 0.5
            method: str = "ind_avg"

        cls = NODE_CONFIG_SCHEMAS["__test_instance__"]
        inst = cls(threshold=0.8, method="zscore")
        assert inst.threshold == 0.8
        assert inst.method == "zscore"

    def test_extra_forbid_preserved_after_decoration(self):
        """@register_node_config 不影响 pydantic 的 extra=forbid 等配置。"""
        from QuantNodes.research.factor_test.nodes.configs import (
            PreprocessNodeConfig,
        )
        # extra=forbid 在 PreprocessNodeConfig 上设置了
        with pytest.raises(Exception):  # ValidationError
            PreprocessNodeConfig(missing="ind_avg", unknown_field=123)


# ---------------------------------------------------------------------------
# Backward compat: existing route table unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_lookup_by_name_returns_expected_class(self):
        from QuantNodes.research.factor_test.nodes.configs import (
            GroupAnalyzerNodeConfig,
            LoadDataNodeConfig,
        )
        assert NODE_CONFIG_SCHEMAS["GroupAnalyzer"] is GroupAnalyzerNodeConfig
        assert NODE_CONFIG_SCHEMAS["LoadData"] is LoadDataNodeConfig

    def test_lookup_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            NODE_CONFIG_SCHEMAS["__nonexistent_node__"]
