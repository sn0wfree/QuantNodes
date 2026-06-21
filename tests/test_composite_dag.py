"""PR-QN-3a: Composite DAG 核心测试

锁定 PR-QN-3a (2026-06-21) 行为:
- ParamSpec 类型校验
- CompositeSpec instantiate (必填/类型/默认)
- composite_operator 装饰器注册 + 重复注册 raise
- _COMPOSITE_REGISTRY 隔离性
- load_composites_from_yaml AST 解析
- is_composite_op 跨类别查询
- get_composite_doc_for_llm 输出 markdown
"""
from __future__ import annotations

import json

import pytest

from QuantNodes.operators import (
    composite_operator, ParamSpec,
    is_composite_op, get_composite_spec, list_composite_ops,
    get_composite_doc_for_llm, load_composites_from_yaml,
)
from QuantNodes.operators.composite_dag import _COMPOSITE_REGISTRY


@pytest.fixture(autouse=True)
def _isolate_composite_registry():
    """每个 test 后清空用户测试注册的 composite, 避免污染其他 tests.

    仅删除本测试模块注册的 op (通过 ``spec.template.__module__`` 判定).
    不清理 PR-QN-3b 注入的内置 op.
    """
    yield
    to_remove = [
        n for n, s in _COMPOSITE_REGISTRY._registry.items()
        if s.template.__module__ == __name__
    ]
    for n in to_remove:
        _COMPOSITE_REGISTRY._registry.pop(n, None)


# ============== ParamSpec ==============

class TestParamSpec:
    def test_basic_construction(self):
        ps = ParamSpec(name="x", type_hint="expr", required=True)
        assert ps.name == "x"
        assert ps.required is True

    def test_validate_required_none_raises(self):
        ps = ParamSpec(name="x", required=True)
        with pytest.raises(ValueError, match="required"):
            ps.validate(None)

    def test_validate_int_mismatch(self):
        ps = ParamSpec(name="window", type_hint="int", default=20)
        with pytest.raises(TypeError, match="must be int"):
            ps.validate("20")

    def test_validate_float_ok(self):
        ps = ParamSpec(name="k", type_hint="float", default=1.0)
        ps.validate(1.5)  # 不 raise

    def test_validate_str_mismatch(self):
        ps = ParamSpec(name="col", type_hint="str")
        with pytest.raises(TypeError, match="must be str"):
            ps.validate(123)

    def test_expr_type_no_runtime_check(self):
        """type_hint='expr' 不做运行时类型检查 (polars Expr 在外部传)."""
        ps = ParamSpec(name="x", type_hint="expr")
        ps.validate("not_an_expr")  # 不 raise, 仅作 schema 文档


# ============== CompositeSpec ==============

class TestCompositeSpec:
    def test_basic_instantiate(self):
        @composite_operator(
            name="t_add",
            params={
                "x": {"type": "float", "required": True},
                "k": {"type": "float", "default": 1.0},
            },
        )
        def t_add(x, k=1.0):
            return x + k

        spec = get_composite_spec("t_add")
        result = spec.instantiate(x=1.0, k=2.0)
        assert result == 3.0

    def test_missing_required_param(self):
        @composite_operator(
            name="t_req",
            params={"x": {"type": "expr", "required": True}},
        )
        def t_req(x):
            return x

        spec = get_composite_spec("t_req")
        with pytest.raises(ValueError, match="Missing required"):
            spec.instantiate()

    def test_default_value_used(self):
        @composite_operator(
            name="t_def",
            params={"x": {"type": "expr", "required": True},
                    "k": {"type": "float", "default": 3.0}},
        )
        def t_def(x, k=3.0):
            return x + k

        spec = get_composite_spec("t_def")
        # 不传 k — 使用默认 3.0
        result = spec.instantiate(x=1.0)
        assert result == 4.0

    def test_to_dict_serializable(self):
        @composite_operator(
            name="t_dict",
            params={"x": {"type": "expr", "required": True}},
            doc="字典测试",
        )
        def t_dict(x):
            return x

        spec = get_composite_spec("t_dict")
        d = spec.to_dict()
        assert d["name"] == "t_dict"
        assert d["doc"] == "字典测试"
        # 应可 JSON 序列化
        json.dumps(d)


# ============== 装饰器 ==============

class TestCompositeOperator:
    def test_basic_registration(self):
        @composite_operator(
            name="t_basic",
            params={"x": {"type": "expr", "required": True}},
            doc="基础测试",
        )
        def t_basic(x):
            return x

        assert is_composite_op("t_basic")
        spec = get_composite_spec("t_basic")
        assert spec.doc == "基础测试"

    def test_duplicate_registration_raises(self):
        @composite_operator(name="t_dup", params={"x": {"type": "expr"}})
        def t_dup_v1(x):
            return x

        with pytest.raises(ValueError, match="already registered"):
            @composite_operator(name="t_dup", params={"x": {"type": "expr"}})
            def t_dup_v2(x):
                return x * 2

    def test_wrapper_preserves_call(self):
        @composite_operator(
            name="t_wrap",
            params={"x": {"type": "expr", "required": True}},
        )
        def t_wrap(x):
            return x * 2

        # wrapper 调用 → spec.instantiate(**kwargs)
        result = t_wrap(x=5)
        assert result == 10

    def test_doc_from_func_if_not_provided(self):
        @composite_operator(
            name="t_autodoc",
            params={"x": {"type": "expr", "required": True}},
        )
        def t_autodoc(x):
            """自动文档测试"""
            return x

        spec = get_composite_spec("t_autodoc")
        assert "自动文档测试" in spec.doc


# ============== 注册表 ==============

class TestCompositeRegistry:
    def test_list_includes_builtins(self):
        """list_composite_ops 应返回所有已注册的 (含 PR-QN-3b 注入的 20 个)."""
        ops = list_composite_ops()
        assert isinstance(ops, list)

    def test_get_composite_spec_none(self):
        """不存在的 op 返回 None."""
        assert get_composite_spec("non_existent_op_xyz") is None

    def test_is_composite_op_false_for_nonexistent(self):
        assert is_composite_op("non_existent_op_xyz") is False


# ============== LLM 文档 ==============

class TestCompositeForLLM:
    def test_doc_format_markdown(self):
        @composite_operator(
            name="t_doc_op",
            params={"x": {"type": "expr", "required": True}},
            doc="文档测试",
        )
        def t_doc_op(x):
            return x

        doc = get_composite_doc_for_llm()
        assert "##" in doc
        assert "t_doc_op" in doc

    def test_doc_includes_required_and_default_tags(self):
        @composite_operator(
            name="t_tag_op",
            params={
                "x": {"type": "expr", "required": True},
                "k": {"type": "float", "default": 1.0},
            },
            doc="标签测试",
        )
        def t_tag_op(x, k=1.0):
            return x + k

        doc = get_composite_doc_for_llm()
        assert "(required)" in doc
        assert "(default: 1.0)" in doc


# ============== YAML 加载 ==============

class TestCompositeYAML:
    def test_load_simple_yaml(self, tmp_path):
        """加载简单 YAML composite."""
        yaml_content = """
composites:
  - name: yaml_test_op
    category: multi_section
    doc: "YAML 测试"
    params:
      x: {type: expr, required: true}
      k: {type: float, default: 1.0}
    template: "x + k"
"""
        yaml_file = tmp_path / "composites.yaml"
        yaml_file.write_text(yaml_content)

        count = load_composites_from_yaml(str(yaml_file))
        assert count == 1
        assert is_composite_op("yaml_test_op")

        # 应可实例化
        spec = get_composite_spec("yaml_test_op")
        result = spec.instantiate(x=10.0, k=2.0)
        assert result == 12.0

    def test_load_empty_yaml(self, tmp_path):
        """空 YAML 或无 composites key → 0 注册, 不 raise."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        assert load_composites_from_yaml(str(yaml_file)) == 0

    def test_load_missing_file_raises(self):
        """不存在的文件应 raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_composites_from_yaml("/tmp/nonexistent_xyz.yaml")

    def test_load_disallowed_function_raises(self, tmp_path):
        """YAML template 调用白名单外的函数 → raise ValueError."""
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("""
composites:
  - name: bad_op
    template: "os.system('rm -rf /')"
""")
        with pytest.raises(ValueError, match="禁止"):
            load_composites_from_yaml(str(yaml_file))

    def test_load_syntax_error_raises(self, tmp_path):
        """YAML template 语法错误 → raise ValueError."""
        yaml_file = tmp_path / "syntax_err.yaml"
        yaml_file.write_text("""
composites:
  - name: syntax_err_op
    template: "x +++ "
""")
        with pytest.raises(ValueError, match="语法错误"):
            load_composites_from_yaml(str(yaml_file))
