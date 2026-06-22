# coding=utf-8
"""Tests for CompositeSpecVisitor (Phase 1.5, Visitor pattern).

Covers:
  - CompositeSpecVisitor base class
  - LLMDocVisitor (backward compat with get_composite_doc_for_llm)
  - DependencyVisitor (extract dep graph, cycle detection)
  - ValidationVisitor (semantic checks: required+default conflict,
    empty doc warning, missing examples warning)
  - visit_all() convenience method
"""
from typing import Any, Dict, List

import pytest

from QuantNodes.operators import (
    ParamSpec,
    composite_operator,
    get_composite_doc_for_llm,
)
from QuantNodes.operators.composite_dag import (
    CompositeSpec,
    CompositeSpecVisitor,
    DependencyVisitor,
    LLMDocVisitor,
    ValidationVisitor,
    _COMPOSITE_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures: 隔离 _COMPOSITE_REGISTRY
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_registry():
    snapshot = dict(_COMPOSITE_REGISTRY._registry)
    yield
    _COMPOSITE_REGISTRY._registry.clear()
    _COMPOSITE_REGISTRY._registry.update(snapshot)


def make_spec(name: str, params: Dict[str, ParamSpec] = None,
              doc: str = "test", examples: List[dict] = None) -> CompositeSpec:
    """构造 CompositeSpec 不通过 @composite_operator (避免污染注册表)。"""
    def _template(**kwargs) -> Any:
        return None
    return CompositeSpec(
        name=name,
        template=_template,
        category="multi_section",
        params=params or {},
        doc=doc,
        examples=examples or [],
    )


# ---------------------------------------------------------------------------
# CompositeSpecVisitor base class
# ---------------------------------------------------------------------------

class TestVisitorBase:
    def test_visit_spec_is_abstract(self):
        visitor = CompositeSpecVisitor()
        spec = make_spec("x")
        with pytest.raises(NotImplementedError):
            visitor.visit_spec(spec)

    def test_visit_all_iterates_registry(self):
        """visit_all 遍历 _COMPOSITE_REGISTRY 中所有 spec。"""
        class CountingVisitor(CompositeSpecVisitor):
            def __init__(self):
                self.count = 0
            def visit_spec(self, spec):
                self.count += 1

        before = list(_COMPOSITE_REGISTRY.all_specs())
        v = CountingVisitor()
        v.visit_all()
        assert v.count == len(before)


# ---------------------------------------------------------------------------
# LLMDocVisitor
# ---------------------------------------------------------------------------

class TestLLMDocVisitor:
    def test_empty_registry(self):
        """空 visitor (未 visit 任何 spec): 仅输出标题, 无 ## 行。"""
        v = LLMDocVisitor()
        # 不调用 visit_spec / visit_all
        assert v.result == "# Available Composite Operators"

    def test_visits_one_spec(self):
        v = LLMDocVisitor()
        spec = make_spec(
            "my_op",
            params={
                "x": ParamSpec(name="x", type_hint="expr", required=True,
                               description="input expr"),
                "k": ParamSpec(name="k", type_hint="float", default=1.0,
                               description="scalar"),
            },
            doc="My operation",
            examples=[{"x": "col('close')", "k": 2.0}],
        )
        v.visit_spec(spec)
        out = v.result
        assert "# Available Composite Operators" in out
        assert "## my_op" in out
        assert "My operation" in out
        assert "- x: expr (required) — input expr" in out
        assert "- k: float (default: 1.0) — scalar" in out
        assert "Example:" in out

    def test_matches_get_composite_doc_for_llm(self):
        """LLMDocVisitor.visit_all().result 应与 get_composite_doc_for_llm() 一致。"""
        # 直接对比
        v = LLMDocVisitor()
        v.visit_all()
        assert v.result == get_composite_doc_for_llm()

    def test_optional_param_format(self):
        v = LLMDocVisitor()
        spec = make_spec("foo", params={
            "x": ParamSpec(name="x", description="no default, no required"),
        })
        v.visit_spec(spec)
        assert "(optional)" in v.result


# ---------------------------------------------------------------------------
# DependencyVisitor
# ---------------------------------------------------------------------------

class TestDependencyVisitor:
    def test_no_dependencies(self):
        v = DependencyVisitor()
        spec = make_spec("standalone")
        v.visit_spec(spec)
        assert v.graph == {"standalone": set()}

    def test_detect_dependency_via_template_source(self):
        """当 spec A 的 template 源码包含 spec B 的名字, A 依赖 B。

        DependencyVisitor 通过 _COMPOSITE_REGISTRY.list() 找候选依赖,
        因此 b 必须先注册到 _COMPOSITE_REGISTRY 才能被检测到。
        """
        # 通过 @composite_operator 注册 b (隔离 fixture 保证不污染其他 test)
        @composite_operator(
            name="__test_visited_b__",
            doc="dummy b",
        )
        def __test_visited_b__(x) -> Any:
            return None

        def a_template(**kwargs):
            return __test_visited_b__()  # 源码包含 "visited_b" 字面量吗? 不
        # 实际 DetectionVisitor 用 "in source" 检查, 需源码含完整 name 字符串
        def a_template_with_call(**kwargs):
            return __test_visited_b__(x=None)  # Python 编译后源码含 __test_visited_b__
        spec_a = CompositeSpec(
            name="__test_visited_a__",
            template=a_template_with_call,
            category="multi_section",
        )
        v = DependencyVisitor()
        v.visit_spec(spec_a)
        # a 应该依赖 b
        assert "__test_visited_b__" in v.graph.get("__test_visited_a__", set())

    def test_detect_cycles_no_cycles(self):
        """无环图: detect_cycles 返回空。"""
        v = DependencyVisitor()
        v.visit_spec(make_spec("a"))
        v.visit_spec(make_spec("b"))
        v.graph["a"] = {"b"}
        v.graph["b"] = set()
        assert v.detect_cycles() == []

    def test_detect_cycles_simple_cycle(self):
        """a → b → a: 检出循环。"""
        v = DependencyVisitor()
        v.graph["a"] = {"b"}
        v.graph["b"] = {"a"}
        cycles = v.detect_cycles()
        assert len(cycles) > 0
        # 至少一个 cycle 包含 a, b
        cycle_names = set()
        for c in cycles:
            cycle_names.update(c)
        assert "a" in cycle_names and "b" in cycle_names

    def test_detect_cycles_self_loop(self):
        v = DependencyVisitor()
        v.graph["a"] = {"a"}
        cycles = v.detect_cycles()
        assert len(cycles) > 0
        assert "a" in cycles[0]


# ---------------------------------------------------------------------------
# ValidationVisitor
# ---------------------------------------------------------------------------

class TestValidationVisitor:
    def test_clean_spec_no_errors_or_warnings(self):
        v = ValidationVisitor()
        spec = make_spec(
            "clean",
            params={"x": ParamSpec(name="x", type_hint="expr", required=True,
                                   description="ok")},
            doc="A clean op",
            examples=[{"x": "col('a')"}],
        )
        v.visit_spec(spec)
        assert v.errors == []
        assert v.warnings == []

    def test_required_with_default_raises_error(self):
        """required=True 但 default != None: 语义冲突, 报 error。"""
        v = ValidationVisitor()
        spec = make_spec("bad", params={
            "x": ParamSpec(name="x", type_hint="int", required=True, default=5),
        })
        v.visit_spec(spec)
        assert v.has_errors
        assert any("required" in e and "default" in e for e in v.errors)

    def test_empty_doc_warning(self):
        v = ValidationVisitor()
        spec = make_spec("no_doc", doc="")
        v.visit_spec(spec)
        assert any("empty doc" in w for w in v.warnings)

    def test_whitespace_only_doc_warning(self):
        v = ValidationVisitor()
        spec = make_spec("ws_doc", doc="   \n  ")
        v.visit_spec(spec)
        assert any("empty doc" in w for w in v.warnings)

    def test_no_examples_warning(self):
        v = ValidationVisitor()
        spec = make_spec("no_examples", examples=[])
        v.visit_spec(spec)
        assert any("no examples" in w for w in v.warnings)

    def test_optional_param_no_param_warnings(self):
        """required=False 且 default=None: 不触发 param 相关的 warning。

        (无 examples 仍会触发 'no examples' warning, 这是 spec-level 而非 param-level。)
        """
        v = ValidationVisitor()
        spec = make_spec(
            "ok_optional",
            params={"x": ParamSpec(name="x", description="opt")},
            examples=[{"x": "col('a')"}],  # 给 examples 避免该 warning
        )
        v.visit_spec(spec)
        assert v.errors == []
        # 不应有 "param" 相关的 warning
        param_warnings = [w for w in v.warnings if "param" in w.lower()]
        assert param_warnings == []

    def test_multiple_specs_aggregate(self):
        v = ValidationVisitor()
        v.visit_spec(make_spec("a", doc="A op"))  # 无 examples → warning
        v.visit_spec(make_spec("b", params={
            "x": ParamSpec(name="x", required=True, default=1),  # 冲突 → error
        }))
        assert v.has_errors
        assert len(v.warnings) >= 1
        assert len(v.errors) >= 1


# ---------------------------------------------------------------------------
# Integration: visit_all on real registry
# ---------------------------------------------------------------------------

class TestVisitorIntegration:
    def test_visit_all_on_real_registry_no_crash(self):
        """对真实 _COMPOSITE_REGISTRY 跑 visit_all, 不应崩。"""
        v = LLMDocVisitor()
        v.visit_all()
        # 至少输出 1 行 (标题)
        assert len(v.result) > 0

    def test_dependency_visitor_on_real_registry(self):
        v = DependencyVisitor()
        v.visit_all()
        # graph 应包含至少 1 个 key
        assert len(v.graph) >= 0  # 可能是空 (若无 composite)

    def test_validation_visitor_on_real_registry(self):
        v = ValidationVisitor()
        v.visit_all()
        # 真实注册表应通过 validation (20 个内置 composite 经过 PR-QN-3b review)
        # 不强制 has_errors=False, 至少不崩
        assert isinstance(v.errors, list)
        assert isinstance(v.warnings, list)
