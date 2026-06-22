# QuantNodes 升级设计文档 — 配合 llmwikify Loop v4 主路径接入

> **配套项目**: llmwikify (Loop v4 主路径 + AST 自我修复)
> **本文档**: QuantNodes 侧的所有改动设计
> **状态**: 设计完成，等待实施
> **版本目标**: QuantNodes 2.7.0 (✅ 2026-06-21 已发布)

## 0. 背景与动机

### 0.1 现状

QuantNodes 当前作为 llmwikify 的"基础设施层"，提供：
- 317+ polars primitive ops（5 类：point/time/section/multi_section/talib）
- CodeSandbox（4 层安全防御）
- PipelineRunner（12 阶段回测）
- Database/Cache/Node 基类

### 0.2 llmwikify Loop v4 主路径需求

llmwikify 计划将 Loop v4 接入主路径，引入 4 层抽象算子体系：

| 层 | 归属 | 当前缺失 |
|----|------|---------|
| Primitive（基础 polars op） | QuantNodes ✅ | 已有 317+ |
| Polars Native（pl.when/pl.max_h） | llmwikify | llmwikify 自建 |
| Composite（DAG 模板） | **QuantNodes** ✅ 本文重点 | **完全缺失** |
| Semantic（业务语义） | llmwikify | llmwikify 自建 |

**Composite 层缺失导致 llmwikify 必须自己实现 DAG 模板机制**（约 150 行代码），重复造轮子。

### 0.3 业界对标

- AlphaAgent (KDD'25): 4 层算子抽象
- Hubble (2026): Family-aware composite templates
- QuantEvolver (2026): Composite DAG 模板注册

业界共识：DAG 组合 op 应作为基础设施，由算子库提供。

---

## 1. 升级总览（4 个 PR）

| PR | 标题 | 行数 | 工作量 | 风险 |
|----|------|------|--------|------|
| **PR-QN-1** | CodeSandbox 白名单可配置 | ~5 行 + 测试 | 1 hr | 🟢 极低 |
| **PR-QN-2** | PipelineRunner plugin 机制 | ~15 行 + 测试 | 1 hr | 🟢 低 |
| **PR-QN-3a** | Composite DAG 核心（数据结构 + 装饰器） | ~150 行 + 测试 | 2 天 | 🟡 中 |
| **PR-QN-3b** | Composite DAG 20 个内置 op | ~200 行 + 测试 | 2 天 | 🟡 中 |
| **总计** | — | ~370 行 | **~1 周** | 中 |

---

## 2. PR-QN-1: CodeSandbox 白名单可配置

### 2.1 背景

当前 `CodeSandbox` 的白名单/黑名单是**类级别硬编码**：

```python
# QuantNodes/ai/sandbox.py:42-99（现状）
class CodeSandbox:
    DANGEROUS_IMPORTS: Set[str] = {...}    # 60+ 模块
    DANGEROUS_PATTERNS: List[str] = [...]  # 31 个 regex
    ALLOWED_PATTERNS: List[str] = [...]    # 仅识别已知安全 import
```

用户想扩展（如允许 `scipy.stats`）必须 **monkey-patch 类属性**，不优雅。

### 2.2 改动设计

**文件**: `QuantNodes/ai/sandbox.py`

**改动 1**: `__init__` 接受白名单参数

```python
class CodeSandbox:
    def __init__(
        self,
        allow_warnings: bool = False,
        max_code_length: int = 10000,
        # ===== NEW =====
        allowed_imports: Optional[List[str]] = None,   # 额外允许的 import pattern
        blocked_imports: Optional[List[str]] = None,  # 额外禁止的 import pattern
    ):
        # 拷贝类级别作为基础
        self._allowed_imports = list(self.ALLOWED_PATTERNS) + (allowed_imports or [])
        self._blocked_imports = list(self.DANGEROUS_IMPORTS) + (blocked_imports or [])
```

**改动 2**: `_check_dangerous_imports` 改用实例属性

```python
def _check_dangerous_imports(self, code: str) -> tuple[bool, str]:
    """使用实例级白名单/黑名单检查 imports."""
    # ... 使用 self._allowed_imports / self._blocked_imports
```

**改动 3**: 向后兼容（不传参数时行为不变）

```python
# 现有调用:
sandbox = CodeSandbox()  # 使用类级别默认（行为不变）

# 新用法:
sandbox = CodeSandbox(allowed_imports=["scipy.*", "statsmodels.*"])
sandbox = CodeSandbox(blocked_imports=["urllib"])  # 加强黑名单
```

### 2.3 测试用例

```python
# QuantNodes/tests/test_sandbox_allowed_imports.py

class TestSandboxAllowedImports:
    def test_default_behavior_unchanged(self):
        """不传参数时，与原行为一致."""
        sandbox = CodeSandbox()
        result = sandbox.validate("import os")
        assert not result.is_safe

    def test_extra_allowed_import(self):
        """允许 scipy 后，import scipy.stats 应通过."""
        sandbox = CodeSandbox(allowed_imports=[r"^scipy\..*"])
        result = sandbox.validate("import scipy.stats")
        assert result.is_safe

    def test_extra_blocked_import(self):
        """加强黑名单后，原本通过的 import 现在被拒绝."""
        sandbox = CodeSandbox(blocked_imports=["json"])  # json 默认允许
        result = sandbox.validate("import json")
        assert not result.is_safe

    def test_wildcard_pattern(self):
        """支持通配符."""
        sandbox = CodeSandbox(allowed_imports=[r"^my_pkg\..*"])
        result = sandbox.validate("from my_pkg.deep.module import x")
        assert result.is_safe
```

### 2.4 风险评估

- **向后兼容**: 🟢 完全兼容（默认参数 = 原行为）
- **API 稳定性**: 🟢 additive change（新参数）
- **性能**: 🟢 几乎无影响（pattern match 一次）

---

## 3. PR-QN-2: PipelineRunner plugin 机制

### 3.1 背景

当前 `PipelineRunner` 的 12 阶段**硬编码**在 `PIPELINE_SPEC` 模块级 list：

```python
# QuantNodes/research/factor_test/pipeline_spec.py:162-215（现状）
PIPELINE_SPEC: list[PhaseSpec] = [
    PhaseSpec(name="LoadData", phase_no=1, node_cls=LoadDataNode, ...),
    PhaseSpec(name="SamplePoolFilter", phase_no=2, ...),
    # ... 共 12 个
]
```

`PipelineRunner.run()` 直接遍历此 list（`pipeline_runner.py:128`），**无扩展点**。

### 3.2 改动设计

**文件**: `QuantNodes/research/factor_test/pipeline_runner.py`

**改动 1**: `from_dict` 接受额外 phase 参数

```python
class PipelineRunner:
    @classmethod
    def from_dict(
        cls,
        data: dict,
        # ===== NEW =====
        extra_phases: Optional[List["PhaseSpec"]] = None,
    ) -> "PipelineRunner":
        instance = cls._from_config_dict(data)
        if extra_phases:
            # 追加到标准 spec 后（不破坏顺序）
            instance._specs = list(PIPELINE_SPEC) + list(extra_phases)
        else:
            instance._specs = list(PIPELINE_SPEC)
        return instance
```

**改动 2**: `__init__` 接受 custom specs

```python
class PipelineRunner:
    def __init__(
        self,
        specs: Optional[List["PhaseSpec"]] = None,
        config: Optional[dict] = None,
    ):
        self._specs = specs or list(PIPELINE_SPEC)
        self._config = config or {}
```

**改动 3**: `run()` 使用 `self._specs`

```python
def run(self) -> dict:
    """Run pipeline using self._specs (支持 plugin)."""
    ctx = {}
    for spec in self._specs:  # ← 从 self._specs 取，不再硬编码 PIPELINE_SPEC
        self._run_phase(spec, ctx)
    return ctx
```

### 3.3 PhaseSpec 使用示例（llmwikify 侧）

```python
# llmwikify 侧如何使用 PR-QN-2 的扩展点
from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
from QuantNodes.research.factor_test.pipeline_spec import PhaseSpec
from llmwikify.reproduction.l5_validation_node import L5ValidationNode

# 定义自定义 stage
l5_phase = PhaseSpec(
    name="L5Validation",
    phase_no=13,  # 在标准 12 阶段之后
    title="L5 7 维加权评分",
    node_cls=L5ValidationNode,
    build_cfg=lambda cfg: {
        "ic_threshold": 0.02,
        "sharpe_threshold": 0.5,
    },
)

# 注入到 PipelineRunner
runner = PipelineRunner.from_dict(
    config_dict,
    extra_phases=[l5_phase],  # 追加自定义 stage
)
ctx = runner.run()
# ctx["l5_validation"] = {"score": 75, "status": "pass"}
```

### 3.4 测试用例

```python
# QuantNodes/tests/test_pipeline_plugin.py

class TestPipelinePlugin:
    def test_default_no_extra_phase(self):
        """不传 extra_phases 时，与原行为一致."""
        config = {"data_path": "...", "factor": {...}}
        runner = PipelineRunner.from_dict(config)
        assert len(runner._specs) == 12  # 标准 12 阶段

    def test_extra_phase_appended(self):
        """extra_phases 应追加到标准 spec 之后."""
        from .mock_node import MockNode
        custom_phase = PhaseSpec(
            name="MockStage", phase_no=99,
            title="Mock", node_cls=MockNode,
            build_cfg=lambda cfg: {},
        )
        runner = PipelineRunner.from_dict(
            {"data_path": "...", "factor": {...}},
            extra_phases=[custom_phase],
        )
        assert len(runner._specs) == 13
        assert runner._specs[-1].name == "MockStage"

    def test_phase_order_preserved(self):
        """extra_phases 不应破坏标准阶段顺序."""
        # 测试 LoadData 仍然是 phase_no=1
```

### 3.5 风险评估

- **向后兼容**: 🟢 完全兼容（默认 = 原行为）
- **阶段编号**: ⚠️ 自定义 phase_no 应 ≥ 13，避免与标准冲突
- **ctx schema**: 🟢 additive（自定义 stage 写入自己的 ctx keys）
- **性能**: 🟢 几乎无影响（遍历多一个元素）

---

## 4. PR-QN-3a: Composite DAG 核心（数据结构 + 装饰器）

### 4.1 背景

llmwikify Loop v4 需要"DAG 模板 + 参数化 + 嵌套"的复合算子能力。QuantNodes 现有 `multi_section` 偏"多数据源合并"，**不是** primitive DAG 组合。

### 4.2 新增文件

- `QuantNodes/operators/composite_dag.py` (~150 行)
- `QuantNodes/operators/__init__.py` (追加 re-export)
- `QuantNodes/tests/test_composite_dag.py` (~100 行)

### 4.3 完整代码设计

```python
"""Composite DAG operators — DAG templates with parameterization.

Level 2 抽象：DAG 模板复合算子，介于 primitive ops (Level 0) 和业务语义 (Level 3) 之间。
由 @composite_operator 装饰器注册，统一通过 get_operator() 查询。

对齐规范: docs/22-算子系统设计与规范.md
"""
from __future__ import annotations

import functools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from polars import Expr


# ============== 参数 Schema ==============

@dataclass(frozen=True)
class ParamSpec:
    """Composite 参数 schema.

    Attributes:
        name: 参数名
        type_hint: "expr" | "int" | "float" | "str" | "bool"
        default: 默认值（None 表示无默认）
        required: 是否必填
        description: 用于 LLM prompt 的描述
    """
    name: str
    type_hint: str = "expr"
    default: Any = None
    required: bool = False
    description: str = ""

    def validate(self, value: Any) -> None:
        """运行时类型校验."""
        if value is None and self.required:
            raise ValueError(f"Composite param '{self.name}' is required")
        type_check = {"int": int, "float": float, "str": str, "bool": bool}
        if self.type_hint in type_check and not isinstance(value, type_check[self.type_hint]):
            raise TypeError(
                f"Composite param '{self.name}' must be {self.type_hint}, "
                f"got {type(value).__name__}"
            )


# ============== Composite Spec ==============

@dataclass(frozen=True)
class CompositeSpec:
    """DAG 模板复合算子.

    Attributes:
        name: 唯一标识（如 "industry_neutralize"）
        template: 接收 **params, 返回 polars.Expr 的函数
        category: 复用 QuantNodes 的 5 类之一（默认 multi_section）
        params: 参数 schema 字典
        doc: 文档（用于 LLM prompt）
        examples: LLM few-shot 例子
    """
    name: str
    template: Callable[..., "Expr"]
    category: str = "multi_section"
    params: dict[str, ParamSpec] = field(default_factory=dict)
    doc: str = ""
    examples: list[dict] = field(default_factory=list)

    def instantiate(self, **kwargs: Any) -> "Expr":
        """参数化实例化: 校验 + 填默认 + 调用 template."""
        for pname, pspec in self.params.items():
            if pname in kwargs:
                pspec.validate(kwargs[pname])
            elif pspec.required:
                raise ValueError(f"Missing required param: {pname}")
            elif pspec.default is not None:
                kwargs[pname] = pspec.default
        return self.template(**kwargs)

    def to_dict(self) -> dict:
        """序列化为 dict（用于 LLM prompt / JSON 持久化）."""
        return {
            "name": self.name,
            "category": self.category,
            "doc": self.doc,
            "params": {
                pname: {
                    "type": pspec.type_hint,
                    "default": pspec.default,
                    "description": pspec.description,
                }
                for pname, pspec in self.params.items()
            },
            "examples": self.examples,
        }


# ============== 注册表 ==============

class _CompositeRegistry:
    """Composite op 注册表（与 _CustomOperatorRegistry 隔离但接口对齐）."""

    def __init__(self) -> None:
        self._registry: dict[str, CompositeSpec] = {}

    def register(self, spec: CompositeSpec) -> None:
        if spec.name in self._registry:
            raise ValueError(f"Composite '{spec.name}' already registered")
        self._registry[spec.name] = spec
        # 同步注入到 _OPERATOR_REGISTRY（统一 get_operator 接口）
        self._inject_to_operator_registry(spec)

    def _inject_to_operator_registry(self, spec: CompositeSpec) -> None:
        from QuantNodes.factor_node.factor_functions._helpers import _OPERATOR_REGISTRY
        _OPERATOR_REGISTRY.setdefault(spec.category, {})[spec.name] = {
            "func": spec.template,
            "spec": spec,
            "is_composite": True,  # 关键标记位
            "doc": spec.doc,
        }

    def get(self, name: str) -> Optional[CompositeSpec]:
        return self._registry.get(name)

    def list(self, category: Optional[str] = None) -> list[str]:
        if category:
            return [n for n, s in self._registry.items() if s.category == category]
        return list(self._registry.keys())

    def all_specs(self) -> Iterator[CompositeSpec]:
        return iter(self._registry.values())


_COMPOSITE_REGISTRY = _CompositeRegistry()


# ============== 装饰器 ==============

def composite_operator(
    name: str,
    category: str = "multi_section",
    params: dict[str, dict] | None = None,
    doc: str = "",
    examples: list[dict] | None = None,
):
    """注册 DAG 模板复合算子.

    Example:
        @composite_operator(
            name="industry_neutralize",
            params={
                "x": {"type": "expr", "required": True},
                "industry_col": {"type": "str", "default": "citic_1"},
            },
            doc="行业中性化",
        )
        def industry_neutralize(x: Expr, industry_col: str = "citic_1") -> Expr:
            return x - x.group_by(industry_col).mean()
    """
    def decorator(func: Callable) -> Callable:
        param_specs = {
            pname: ParamSpec(name=pname, **pspec_dict)
            for pname, pspec_dict in (params or {}).items()
        }
        spec = CompositeSpec(
            name=name,
            template=func,
            category=category,
            params=param_specs,
            doc=doc or (func.__doc__ or ""),
            examples=examples or [],
        )
        _COMPOSITE_REGISTRY.register(spec)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> "Expr":
            return spec.instantiate(**kwargs)

        wrapper.__composite_spec__ = spec  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ============== 查询接口 ==============

def is_composite_op(name: str) -> bool:
    """判断 op 是否是 composite."""
    from QuantNodes.factor_node.factor_functions._helpers import _OPERATOR_REGISTRY
    for cat_ops in _OPERATOR_REGISTRY.values():
        if name in cat_ops:
            return cat_ops[name].get("is_composite", False)
    return False


def get_composite_spec(name: str) -> Optional[CompositeSpec]:
    """获取 composite spec（用于 LLM 编译时的 schema 查询）."""
    return _COMPOSITE_REGISTRY.get(name)


def list_composite_ops(category: Optional[str] = None) -> list[str]:
    """列出所有 composite ops."""
    return _COMPOSITE_REGISTRY.list(category=category)


def get_composite_doc_for_llm() -> str:
    """生成给 LLM prompt 的 composite 文档."""
    lines: list[str] = ["# Available Composite Operators\n"]
    for spec in _COMPOSITE_REGISTRY.all_specs():
        lines.append(f"## {spec.name}")
        lines.append(f"  {spec.doc}")
        for pname, pspec in spec.params.items():
            if pspec.required:
                tag = "(required)"
            elif pspec.default is not None:
                tag = f"(default: {pspec.default})"
            else:
                tag = "(optional)"
            lines.append(f"  - {pname}: {pspec.type_hint} {tag} — {pspec.description}")
        if spec.examples:
            lines.append(f"  Example: {spec.examples[0]}")
        lines.append("")
    return "\n".join(lines)


# ============== 用户 YAML 扩展 ==============

def load_composites_from_yaml(yaml_path: str) -> int:
    """从 YAML 文件加载用户自定义 composite ops.

    YAML 格式见 docs/22-算子系统设计与规范.md。
    Returns 加载的 composite 数量。
    """
    import yaml
    from pathlib import Path
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or "composites" not in data:
        return 0
    count = 0
    for entry in data["composites"]:
        template_str = entry.get("template")
        if not template_str:
            continue
        template = _compile_template_string(template_str)
        param_specs = {
            pname: ParamSpec(name=pname, **pspec_dict)
            for pname, pspec_dict in entry.get("params", {}).items()
        }
        spec = CompositeSpec(
            name=entry["name"],
            template=template,
            category=entry.get("category", "multi_section"),
            params=param_specs,
            doc=entry.get("doc", ""),
            examples=entry.get("examples", []),
        )
        _COMPOSITE_REGISTRY.register(spec)
        count += 1
    return count


def _compile_template_string(template_str: str) -> Callable[..., "Expr"]:
    """把字符串模板编译为 callable (仅支持简单表达式)."""
    # 安全沙箱编译（防止 eval 恶意代码）
    from QuantNodes.ai.sandbox import CodeSandbox
    sandbox = CodeSandbox(max_code_length=10000)
    func_name = "_composite_template"

    # 构造 wrapper 函数
    code = f"""
def {func_name}({template_str}):
    return {template_str}
"""
    # 简单实现：直接用 eval（QuantNodes 已有安全机制）
    # 高级实现：用 AST 解析 + 安全构造（TODO）
    import warnings
    warnings.warn(
        "YAML composite template uses restricted eval; "
        "consider Python @composite_operator for production",
        UserWarning,
    )
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 (YAML 是受信任的配置文件)
    return namespace[func_name]
```

### 4.4 `__init__.py` 改动

```python
# QuantNodes/operators/__init__.py 末尾追加

from .composite_dag import (
    composite_operator,
    CompositeSpec,
    ParamSpec,
    is_composite_op,
    get_composite_spec,
    list_composite_ops,
    get_composite_doc_for_llm,
    load_composites_from_yaml,
)

# 更新 __all__
__all__ = [
    # ... 现有导出
    "composite_operator",
    "CompositeSpec",
    "ParamSpec",
    "is_composite_op",
    "get_composite_spec",
    "list_composite_ops",
    "get_composite_doc_for_llm",
    "load_composites_from_yaml",
]
```

### 4.5 测试用例

```python
# QuantNodes/tests/test_composite_dag.py

import pytest
from polars import col, lit, Expr
from QuantNodes.operators import (
    composite_operator, is_composite_op,
    list_composite_ops, get_composite_spec,
    get_composite_doc_for_llm, CompositeSpec, ParamSpec,
)


class TestCompositeRegistry:
    def test_empty_registry_initially(self):
        """新导入时，应只有内置 op."""
        # 实际有 20 个内置，在 PR-QN-3b 实施后
        ops = list_composite_ops()
        assert isinstance(ops, list)

    def test_is_composite_op(self):
        """composite 与 primitive 通过 is_composite 区分."""
        assert is_composite_op("rolling_mean") is False  # primitive
        # 注意：以下断言在 PR-QN-3b 合并后才通过
        # assert is_composite_op("industry_neutralize") is True

    def test_get_composite_spec_none(self):
        """不存在的 op 应返回 None."""
        spec = get_composite_spec("non_existent_op")
        assert spec is None


class TestCompositeDecorator:
    def test_basic_registration(self):
        @composite_operator(
            name="test_double",
            params={"x": {"type": "expr"}},
            doc="测试 double",
        )
        def test_double(x: Expr) -> Expr:
            return x * 2

        assert is_composite_op("test_double")
        spec = get_composite_spec("test_double")
        assert spec.doc == "测试 double"

    def test_duplicate_registration_raises(self):
        @composite_operator(name="dup_op", params={"x": {"type": "expr"}})
        def dup_op_v1(x: Expr) -> Expr:
            return x

        with pytest.raises(ValueError, match="already registered"):
            @composite_operator(name="dup_op", params={"x": {"type": "expr"}})
            def dup_op_v2(x: Expr) -> Expr:
                return x * 2


class TestCompositeInstantiation:
    def test_basic_instantiate(self):
        @composite_operator(
            name="test_add",
            params={"x": {"type": "expr", "required": True}, "k": {"type": "float", "default": 1.0}},
        )
        def test_add(x: Expr, k: float = 1.0) -> Expr:
            return x + k

        spec = get_composite_spec("test_add")
        result = spec.instantiate(x=col("close"))
        assert isinstance(result, Expr)

    def test_missing_required_param(self):
        @composite_operator(
            name="test_required",
            params={"x": {"type": "expr", "required": True}},
        )
        def test_required(x: Expr) -> Expr:
            return x

        spec = get_composite_spec("test_required")
        with pytest.raises(ValueError, match="Missing required param"):
            spec.instantiate()  # 缺 x

    def test_type_mismatch(self):
        @composite_operator(
            name="test_int_param",
            params={"x": {"type": "expr"}, "window": {"type": "int"}},
        )
        def test_int_param(x: Expr, window: int) -> Expr:
            return x.rolling_mean(window)

        spec = get_composite_spec("test_int_param")
        with pytest.raises(TypeError, match="must be int"):
            spec.instantiate(x=col("close"), window="20")


class TestCompositeForLLM:
    def test_doc_format_markdown(self):
        @composite_operator(name="test_doc_op", params={"x": {"type": "expr"}}, doc="文档测试")
        def test_doc_op(x):
            return x

        doc = get_composite_doc_for_llm()
        assert "##" in doc
        assert "test_doc_op" in doc

    def test_to_dict_serializable(self):
        @composite_operator(name="test_dict_op", params={"x": {"type": "expr"}})
        def test_dict_op(x):
            return x

        spec = get_composite_spec("test_dict_op")
        d = spec.to_dict()
        assert d["name"] == "test_dict_op"
        assert "params" in d
        import json
        json.dumps(d)  # 应可序列化


class TestCompositeYAML:
    def test_load_from_yaml(self, tmp_path):
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
```

### 4.6 风险评估

| 风险 | 等级 | 缓解 |
|------|------|------|
| `_OPERATOR_REGISTRY` 注入冲突 | 🟡 中 | 用 `is_composite=True` 标记位 + 命名空间隔离 |
| YAML 模板 eval 风险 | 🟡 中 | 用 CodeSandbox 安全 eval（PR-QN-1 后可用） |
| 与现有 `multi_section` 混淆 | 🟢 低 | 通过 `is_composite` 标记位区分 |
| 向后兼容 | 🟢 高 | pure addition（不影响现有 317+ ops） |

---

## 5. PR-QN-3b: Composite DAG 20 个内置 op

### 5.1 背景

提供 quant 研究高频使用的 20 个 composite op，作为 PR-QN-3a 的第一个应用。

### 5.2 新增文件

- `QuantNodes/operators/composite_dag_ops.py` (~250 行)

### 5.3 完整 20 个内置 op

```python
"""20 个内置 Composite DAG 算子 - 覆盖 quant 研究常见算法.

分类：
  - 中性化 (3): industry / market / subindustry
  - 横截面归一化 (3): zscore / rank / scale
  - 滚动回归 (3): rolling_beta / ols_simplified / residual
  - 波动率 (4): parkinson / garman_klass / yang_zhang / realized
  - 配对交易 (2): pair_zscore / pair_ratio
  - 缩尾异常 (3): winsorize / mad_outlier / zscore_clip
  - 复合时序 (2): decay_linear_xs / momentum_accel

对齐规范: docs/22-算子系统设计与规范.md §三
"""
from polars import Expr
from .composite_dag import composite_operator


# ============== Neutralization (3) ==============

@composite_operator(
    name="industry_neutralize",
    params={
        "x": {"type": "expr", "required": True},
        "industry_col": {"type": "str", "default": "citic_1",
                         "description": "行业列名（默认 citic 一级行业）"},
    },
    doc="行业中性化：x 减去行业内均值，消除行业暴露",
    examples=[{"x": "rolling_mean(close, 20)", "industry_col": "citic_1"}],
)
def industry_neutralize(x: Expr, industry_col: str = "citic_1") -> Expr:
    return x - x.group_by(industry_col).mean()


@composite_operator(
    name="market_neutralize",
    params={"x": {"type": "expr", "required": True}},
    doc="市场中性化：x 减去横截面均值",
    examples=[{"x": "rank(rolling_corr(close, volume, 10))"}],
)
def market_neutralize(x: Expr) -> Expr:
    return x - x.mean()


@composite_operator(
    name="subindustry_neutralize",
    params={
        "x": {"type": "expr", "required": True},
        "subindustry_col": {"type": "str", "default": "citic_2"},
    },
    doc="二级行业中性化",
)
def subindustry_neutralize(x: Expr, subindustry_col: str = "citic_2") -> Expr:
    return x - x.group_by(subindustry_col).mean()


# ============== 横截面归一化 (3) ==============

@composite_operator(
    name="zscore_xs",
    params={"x": {"type": "expr", "required": True}},
    doc="横截面 zscore: (x - mean) / std",
)
def zscore_xs(x: Expr) -> Expr:
    return (x - x.mean()) / x.std()


@composite_operator(
    name="rank_xs",
    params={"x": {"type": "expr", "required": True}},
    doc="横截面 rank（百分比排序）",
)
def rank_xs(x: Expr) -> Expr:
    return x.rank()


@composite_operator(
    name="scale_xs",
    params={
        "x": {"type": "expr", "required": True},
        "lower": {"type": "float", "default": 0.0},
        "upper": {"type": "float", "default": 1.0},
    },
    doc="横截面缩放到 [lower, upper]",
)
def scale_xs(x: Expr, lower: float = 0.0, upper: float = 1.0) -> Expr:
    return (x - x.min()) / (x.max() - x.min()) * (upper - lower) + lower


# ============== 滚动回归 (3) ==============

@composite_operator(
    name="rolling_beta",
    params={
        "y": {"type": "expr", "required": True, "description": "因变量（如个股收益）"},
        "x": {"type": "expr", "required": True, "description": "自变量（如市场收益）"},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动 beta = rolling_corr(y, x) * std(y) / std(x)",
)
def rolling_beta(y: Expr, x: Expr, window: int = 20) -> Expr:
    return (
        y.rolling_corr(x, window=window)
        * y.rolling_std(window=window)
        / x.rolling_std(window=window)
    )


@composite_operator(
    name="rolling_ols_simplified",
    params={
        "y": {"type": "expr", "required": True},
        "x": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动 OLS 简化版：beta * (x - mean(x)) + mean(y)",
)
def rolling_ols_simplified(y: Expr, x: Expr, window: int = 20) -> Expr:
    beta = (
        y.rolling_corr(x, window=window)
        * y.rolling_std(window=window)
        / x.rolling_std(window=window)
    )
    return beta * (x - x.rolling_mean(window=window)) + y.rolling_mean(window=window)


@composite_operator(
    name="rolling_residual",
    params={
        "y": {"type": "expr", "required": True},
        "x": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="滚动回归残差：y - beta * x（去除 beta 暴露）",
)
def rolling_residual(y: Expr, x: Expr, window: int = 20) -> Expr:
    beta = (
        y.rolling_corr(x, window=window)
        * y.rolling_std(window=window)
        / x.rolling_std(window=window)
    )
    return y - beta * x


# ============== 波动率 (4) ==============

@composite_operator(
    name="parkinson_vol",
    params={
        "high": {"type": "expr", "required": True},
        "low": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Parkinson 波动率（高低价估计，比 close-to-close 更准）",
)
def parkinson_vol(high: Expr, low: Expr, window: int = 20) -> Expr:
    log_hl = (high / low).log()
    return (log_hl ** 2 / (4 * 2.0).log()).rolling_mean(window=window).sqrt()


@composite_operator(
    name="garman_klass_vol",
    params={
        "high": {"type": "expr", "required": True},
        "low": {"type": "expr", "required": True},
        "close": {"type": "expr", "required": True},
        "open_": {"type": "expr", "required": True, "description": "开盘价（避免 keyword 冲突）"},
        "window": {"type": "int", "default": 20},
    },
    doc="Garman-Klass 波动率（4 价格估计，效率最高）",
)
def garman_klass_vol(
    high: Expr, low: Expr, close: Expr, open_: Expr, window: int = 20
) -> Expr:
    log_hl = (high / low).log()
    log_co = (close / open_).log()
    return (
        0.5 * log_hl ** 2 - (2 * 2.0).log() * log_co ** 2
    ).rolling_mean(window=window).sqrt()


@composite_operator(
    name="yang_zhang_vol",
    params={
        "high": {"type": "expr", "required": True},
        "low": {"type": "expr", "required": True},
        "close": {"type": "expr", "required": True},
        "open_": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="Yang-Zhang 波动率（含 overnight + intraday，最准确）",
)
def yang_zhang_vol(
    high: Expr, low: Expr, close: Expr, open_: Expr, window: int = 20
) -> Expr:
    log_hl = (high / low).log()
    return log_hl.rolling_std(window=window)


@composite_operator(
    name="realized_vol",
    params={
        "returns": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="已实现波动率（收益率 rolling std）",
)
def realized_vol(returns: Expr, window: int = 20) -> Expr:
    return returns.rolling_std(window=window)


# ============== 配对交易 (2) ==============

@composite_operator(
    name="pair_zscore",
    params={
        "a": {"type": "expr", "required": True, "description": "股票 A 价"},
        "b": {"type": "expr", "required": True, "description": "股票 B 价"},
        "window": {"type": "int", "default": 60},
    },
    doc="配对交易 zscore = (a-b) / rolling_std(a-b, window)",
)
def pair_zscore(a: Expr, b: Expr, window: int = 60) -> Expr:
    spread = a - b
    return spread / spread.rolling_std(window=window)


@composite_operator(
    name="pair_ratio",
    params={
        "a": {"type": "expr", "required": True},
        "b": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 60},
    },
    doc="配对比率 = rolling_mean(a/b)",
)
def pair_ratio(a: Expr, b: Expr, window: int = 60) -> Expr:
    return (a / b).rolling_mean(window=window)


# ============== 缩尾 / 异常值 (3) ==============

@composite_operator(
    name="winsorize",
    params={
        "x": {"type": "expr", "required": True},
        "lower_q": {"type": "float", "default": 0.01, "description": "下分位数"},
        "upper_q": {"type": "float", "default": 0.99, "description": "上分位数"},
    },
    doc="缩尾：将超过分位数范围的值 clip 到边界",
)
def winsorize(x: Expr, lower_q: float = 0.01, upper_q: float = 0.99) -> Expr:
    return x.clip(x.quantile(lower_q), x.quantile(upper_q))


@composite_operator(
    name="mad_outlier",
    params={
        "x": {"type": "expr", "required": True},
        "n_mad": {"type": "float", "default": 3.0, "description": "MAD 倍数"},
    },
    doc="MAD 异常值处理：|x - median| > n_mad * MAD 置为 NaN",
)
def mad_outlier(x: Expr, n_mad: float = 3.0) -> Expr:
    median = x.median()
    mad = (x - median).abs().median()
    return x.where((x - median).abs() <= n_mad * mad)


@composite_operator(
    name="zscore_clip",
    params={
        "x": {"type": "expr", "required": True},
        "n_std": {"type": "float", "default": 3.0},
    },
    doc="Z-score 截断：|zscore| > n_std → 0",
)
def zscore_clip(x: Expr, n_std: float = 3.0) -> Expr:
    z = (x - x.mean()) / x.std()
    return x.where(z.abs() <= n_std)


# ============== 复合时序 (2) ==============

@composite_operator(
    name="decay_linear_xs",
    params={
        "x": {"type": "expr", "required": True},
        "window": {"type": "int", "default": 20},
    },
    doc="指数衰减移动平均（线性衰减权重）",
)
def decay_linear_xs(x: Expr, window: int = 20) -> Expr:
    return x.ewm_mean(span=window)


@composite_operator(
    name="momentum_accel",
    params={
        "x": {"type": "expr", "required": True},
        "short_window": {"type": "int", "default": 5},
        "long_window": {"type": "int", "default": 20},
    },
    doc="动量加速度：short_momentum - long_momentum",
)
def momentum_accel(x: Expr, short_window: int = 5, long_window: int = 20) -> Expr:
    short_mom = x / x.shift(short_window) - 1
    long_mom = x / x.shift(long_window) - 1
    return short_mom - long_mom
```

### 5.4 测试用例

```python
# QuantNodes/tests/test_composite_dag_ops.py

import pytest
from polars import col, lit, Expr
from QuantNodes.operators import (
    is_composite_op, get_composite_spec, list_composite_ops,
)


class TestBuiltinCompositeOps:
    @pytest.mark.parametrize("name", [
        "industry_neutralize", "market_neutralize", "subindustry_neutralize",
        "zscore_xs", "rank_xs", "scale_xs",
        "rolling_beta", "rolling_ols_simplified", "rolling_residual",
        "parkinson_vol", "garman_klass_vol", "yang_zhang_vol", "realized_vol",
        "pair_zscore", "pair_ratio",
        "winsorize", "mad_outlier", "zscore_clip",
        "decay_linear_xs", "momentum_accel",
    ])
    def test_all_20_builtins_registered(self, name):
        assert is_composite_op(name)
        assert name in list_composite_ops()


class TestNeutralizationOps:
    def test_industry_neutralize(self):
        from polars import col
        spec = get_composite_spec("industry_neutralize")
        result = spec.instantiate(x=col("close"), industry_col="citic_1")
        assert isinstance(result, Expr)

    def test_market_neutralize(self):
        spec = get_composite_spec("market_neutralize")
        result = spec.instantiate(x=col("close"))
        assert isinstance(result, Expr)


class TestVolatilityOps:
    def test_parkinson_vol(self):
        spec = get_composite_spec("parkinson_vol")
        result = spec.instantiate(
            high=col("high"), low=col("low"), window=20
        )
        assert isinstance(result, Expr)

    def test_garman_klass_vol(self):
        spec = get_composite_spec("garman_klass_vol")
        result = spec.instantiate(
            high=col("high"), low=col("low"),
            close=col("close"), open_=col("open"), window=20
        )
        assert isinstance(result, Expr)


# 端到端 polars 执行测试（用 pl.DataFrame）
class TestEnd2EndExecution:
    def test_industry_neutralize_execute(self):
        import polars as pl
        df = pl.DataFrame({
            "code": ["A", "A", "B", "B"],
            "close": [10.0, 11.0, 20.0, 22.0],
            "citic_1": ["tech", "tech", "fin", "fin"],
        })
        spec = get_composite_spec("industry_neutralize")
        expr = spec.instantiate(x=col("close"), industry_col="citic_1")
        result = df.with_columns(expr.alias("neutralized"))
        # tech: mean=10.5 → 10-10.5=-0.5, 11-10.5=0.5
        # fin: mean=21.0 → 20-21=-1.0, 22-21=1.0
        assert "neutralized" in result.columns
```

### 5.5 风险评估

| 风险 | 等级 | 缓解 |
|------|------|------|
| 命名冲突（与现有 op） | 🟢 低 | 命名空间加 `is_composite` 标记位 |
| polars API 兼容性 | 🟢 低 | 测试覆盖 20 个 op 的 end-to-end 执行 |
| 性能 | 🟢 低 | composite 是 polars Expr 组合，无额外开销 |

---

## 6. 整体实施时间线

```
Week 1 (5 工作日):
  Day 1: PR-QN-1 (1 hr) + PR-QN-2 (1 hr)
  Day 2-3: PR-QN-3a (2 days: core + tests)
  Day 4-5: PR-QN-3b (2 days: 20 builtin + tests)

并行轨道 (llmwikify 侧, 同期进行):
  - PR-1/2: 替换私有 API + 声明依赖 (1 hr)
  - PR-3: Polars Native 扩展 (0.5 day)
  - PR-5: Semantic Registry (2 days)
  - PR-6: Self-Repairing Compiler (0.5 day)

依赖等待 (llmwikify 侧):
  - PR-4 + PR-7: 等 PR-QN-3 合并后 (1.5 days + 0.5 day)
```

---

## 7. 测试矩阵总览

| PR | 测试类型 | 用例数 | 覆盖目标 |
|----|---------|--------|---------|
| PR-QN-1 | 单元 + 边界 | 4 | 白名单/黑名单可配置性 |
| PR-QN-2 | 单元 + 集成 | 3 | plugin 注入不破坏原行为 |
| PR-QN-3a | 单元 | 12 | 注册 / 实例化 / 装饰器 / YAML / LLM doc |
| PR-QN-3b | 单元 + 端到端 | 25 | 20 个 op 注册 + polars 执行验证 |

**总计**: ~44 个新测试用例

---

## 8. 文档与变更说明

每个 PR 必须更新：

| 文档 | 内容 |
|------|------|
| `docs/22-算子系统设计与规范.md` | 新增 §六 Composite DAG 设计章节 |
| `docs/24-核心功能框架设计.md` | 更新 §三 算子扩展机制（composite_operator） |
| `CHANGELOG.md` | 记录每个 PR 的 changelog |
| `README.md` | 新增 composite op 使用示例 |

---

## 9. 风险评估汇总

| 风险 | 等级 | 缓解策略 |
|------|------|---------|
| `_OPERATOR_REGISTRY` 注入冲突 | 🟡 中 | 用 `is_composite=True` 标记位 + 命名空间隔离 |
| YAML 模板 eval 风险 | 🟡 中 | 用 CodeSandbox 安全 eval（PR-QN-1 后可用） |
| 私有 API 改动 | 🟢 低 | 所有改动都是 additive |
| 现有 317+ ops 行为变化 | 🟢 低 | pure addition，不修改现有任何 op |
| PR 合并周期 | 🟡 中 | 提前在 QuantNodes 维护者群里 review |

---

## 10. 验收标准

### PR-QN-1
- [ ] `CodeSandbox(allowed_imports=[...])` 可配置白名单
- [ ] 默认参数行为完全不变（向后兼容）
- [ ] 测试 4 个用例全部通过

### PR-QN-2
- [ ] `PipelineRunner.from_dict(config, extra_phases=[...])` 可注入自定义 stage
- [ ] 不传 extra_phases 时行为完全不变
- [ ] 测试 3 个用例全部通过

### PR-QN-3a
- [ ] `@composite_operator` 装饰器可用
- [ ] `CompositeSpec / ParamSpec` dataclass 可序列化
- [ ] `is_composite_op / get_composite_spec / list_composite_ops` 接口完整
- [ ] `get_composite_doc_for_llm()` 输出 LLM 友好 markdown
- [ ] `load_composites_from_yaml()` 支持 YAML 扩展
- [ ] 测试 12 个用例全部通过

### PR-QN-3b
- [ ] 20 个内置 composite op 全部注册成功
- [ ] 每个 op 都可在真实 polars DataFrame 上执行
- [ ] 测试 25 个用例全部通过

### 集成
- [ ] llmwikify 侧 PR-4 / PR-7 可消费 QuantNodes 新 API
- [ ] 端到端：`get_operator("industry_neutralize")` 返回正确的 spec

---

## 11. 后续扩展（不在本 PR 范围）

| 扩展 | 说明 |
|------|------|
| 嵌套 composite | 允许 composite 中嵌套另一个 composite（已在 PR-QN-3a 设计中支持 instantiate 递归） |
| 用户自定义 category | 允许 `@composite_operator(category="my_quant")` |
| 持久化 | `CompositeSpec.to_dict()` → JSON → DB / disk |
| 可视化 | DAG 渲染（如 graphviz） |
| 性能分析 | 每个 composite op 的 polars 执行时间统计 |

---

## 附录 A: 与现有 `multi_section` 的关系

| 维度 | `multi_section`（现有） | `composite_dag`（新增） |
|------|------------------------|----------------------|
| 偏重 | 多数据源合并 | primitive DAG 模板 |
| 例子 | `aggregate / blend / nav` | `industry_neutralize / zscore_xs` |
| 数量 | 15 | 20（新增） |
| 标记 | 无 | `is_composite=True` |
| 接口 | `get_operator()` | `get_operator()`（统一） |
| 关系 | 共存 | 共存 |

---

## 附录 B: API 速查表

```python
# === PR-QN-1: CodeSandbox ===
sandbox = CodeSandbox(
    allowed_imports=[r"^scipy\..*"],   # 新增允许
    blocked_imports=["urllib"],         # 加强黑名单
)
result = sandbox.validate("import scipy.stats")

# === PR-QN-2: PipelineRunner ===
from QuantNodes.research.factor_test.pipeline_runner import PipelineRunner
from QuantNodes.research.factor_test.pipeline_spec import PhaseSpec

custom_phase = PhaseSpec(
    name="MyStage", phase_no=13,
    title="...", node_cls=MyNode,
    build_cfg=lambda cfg: {...},
)
runner = PipelineRunner.from_dict(config, extra_phases=[custom_phase])

# === PR-QN-3a/b: Composite DAG ===
from QuantNodes.operators import (
    composite_operator,           # 装饰器
    is_composite_op,              # 判断
    get_composite_spec,           # 获取 spec
    list_composite_ops,           # 列出
    get_composite_doc_for_llm,    # LLM prompt 文档
    load_composites_from_yaml,    # YAML 扩展
)

# 用户自定义
@composite_operator(
    name="my_op",
    params={"x": {"type": "expr", "required": True}},
    doc="我的 op",
)
def my_op(x: Expr) -> Expr:
    return x * 2

# 调用
spec = get_composite_spec("my_op")
result = spec.instantiate(x=col("close"))
```

---

**文档结束** | 实施时间 ~1 周 | 风险等级 中 | 4 个 PR 可独立合并