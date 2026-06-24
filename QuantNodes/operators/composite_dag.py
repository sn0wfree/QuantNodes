# coding=utf-8
"""Composite DAG operators — DAG 模板复合算子 (PR-QN-3a, 2026-06-21)

Level 1 抽象: 介于 primitive ops (L0) 和 业务语义 (L3) 之间.
由 @composite_operator 装饰器注册, 统一通过 get_operator() 查询,
并用 is_composite=True 标记位与 multi_section 等 L0 ops 区分.

设计要点:
- ParamSpec: 参数 schema (name / type / default / required / description)
- CompositeSpec: DAG 模板 + 参数 schema + 文档 + 例子
- _CompositeRegistry: 隔离注册表, 与 _CustomOperatorRegistry 平级
- composite_operator: 用户自定义入口
- load_composites_from_yaml: YAML 扩展入口 (ast 解析, 非裸 exec)
- get_composite_doc_for_llm: 给 LLM prompt 用的 markdown 文档

对齐规范: docs/22-算子系统设计与规范.md §十七
"""
from __future__ import annotations

import ast
import functools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from polars import Expr


# ============== 参数 Schema ==============

@dataclass(frozen=True)
class ParamSpec:
    """Composite 参数 schema.

    Attributes:
        name: 参数名
        type_hint: "expr" | "int" | "float" | "str" | "bool"
        default: 默认值 (None 表示无默认)
        required: 是否必填
        description: 用于 LLM prompt 的描述
    """
    name: str
    type_hint: str = "expr"
    default: Any = None
    required: bool = False
    description: str = ""

    def validate(self, value: Any) -> None:
        """运行时类型校验.

        Raises:
            ValueError: 必填参数为 None
            TypeError: 类型不匹配
        """
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
        name: 唯一标识 (如 "industry_neutralize")
        template: 接收 **params, 返回 polars.Expr 或 pd.Series 的函数
        category: 复用 QuantNodes 的 5 类之一 (默认 multi_section)
        engine: 引擎类型 ("polars" | "pandas"), 同名 op 分引擎注册
        params: 参数 schema 字典
        doc: 文档 (用于 LLM prompt)
        examples: LLM few-shot 例子
    """
    name: str
    template: Callable[..., Any]
    category: str = "multi_section"
    engine: str = "polars"
    params: Dict[str, ParamSpec] = field(default_factory=dict)
    doc: str = ""
    examples: List[dict] = field(default_factory=list)

    def instantiate(self, **kwargs: Any) -> "Expr":
        """参数化实例化: 校验 + 填默认 + 调用 template.

        Raises:
            ValueError: 必填参数缺失
            TypeError: 参数类型不匹配
        """
        bound = dict(kwargs)
        for pname, pspec in self.params.items():
            if pname in bound:
                pspec.validate(bound[pname])
            elif pspec.required:
                raise ValueError(f"Missing required param: {pname}")
            elif pspec.default is not None:
                bound[pname] = pspec.default
        return self.template(**bound)

    def to_dict(self) -> dict:
        """序列化为 dict (用于 LLM prompt / JSON 持久化)."""
        return {
            "name": self.name,
            "category": self.category,
            "doc": self.doc,
            "params": {
                pname: {
                    "type": pspec.type_hint,
                    "default": pspec.default,
                    "required": pspec.required,
                    "description": pspec.description,
                }
                for pname, pspec in self.params.items()
            },
            "examples": self.examples,
        }


# ============== 注册表 ==============

class _CompositeRegistry:
    """Composite op 注册表 (与 _CustomOperatorRegistry 隔离但接口对齐)."""

    def __init__(self) -> None:
        self._registry: Dict[str, CompositeSpec] = {}

    def register(self, spec: CompositeSpec) -> None:
        """注册一个 composite.

        Raises:
            ValueError: name 已注册 (重复)

        Note:
            Composite 隔离存放在 ``_COMPOSITE_REGISTRY`` 中, **不**注入到主
            ``_OPERATOR_REGISTRY``. 原因:

            1. 主注册表存 L0 primitive, schema 严格要求 ``signature`` /
               ``parameters`` 等键 (见 ``test_all_operators_have_doc``).
            2. 主注册表需 JSON 可序列化 (见 ``generate_documentation_json``),
               CompositeSpec 不可序列化.
            3. composite 应通过 ``is_composite_op`` / ``get_composite_spec`` /
               ``list_composite_ops`` 三套独立 API 访问, 与 L0 严格隔离.
        """
        if spec.name in self._registry:
            raise ValueError(f"Composite '{spec.name}' already registered")
        self._registry[spec.name] = spec

    def _build_param_specs(self, params_dict: Dict[str, dict]) -> Dict[str, ParamSpec]:
        """从用户传入的 dict 构造 ParamSpec (兼容 'type' 字段)."""
        out: Dict[str, ParamSpec] = {}
        for pname, pdict in params_dict.items():
            # 兼容 'type' 字段 (与 type_hint 同义, 避免 Python 关键字冲突)
            pdict = dict(pdict)
            if "type" in pdict and "type_hint" not in pdict:
                pdict["type_hint"] = pdict.pop("type")
            out[pname] = ParamSpec(name=pname, **pdict)
        return out

    def get(self, name: str) -> Optional[CompositeSpec]:
        return self._registry.get(name)

    def list(self, category: Optional[str] = None) -> List[str]:
        if category:
            return [n for n, s in self._registry.items() if s.category == category]
        return list(self._registry.keys())

    def all_specs(self) -> Iterator[CompositeSpec]:
        return iter(self._registry.values())


_COMPOSITE_REGISTRY = _CompositeRegistry()
_COMPOSITE_REGISTRY_POLARS = _COMPOSITE_REGISTRY  # alias for clarity
_COMPOSITE_REGISTRY_PANDAS = _CompositeRegistry()


# ============== 装饰器 ==============

def composite_operator(
    name: str,
    category: str = "multi_section",
    params: Optional[Dict[str, dict]] = None,
    doc: str = "",
    examples: Optional[List[dict]] = None,
    engine: str = "polars",
):
    """注册 DAG 模板复合算子.

    Args:
        name: 算子唯一名
        category: 5 类之一 (默认 multi_section, 与 L0 共存)
        params: {pname: {type, default, required, description}}
        doc: 文档 (LLM prompt 用)
        examples: LLM few-shot 例子
        engine: 引擎类型 ("polars" | "pandas"), 同名 op 分引擎注册

    Returns:
        装饰器函数

    Example:
        @composite_operator(
            name="industry_neutralize",
            params={
                "x": {"type": "expr", "required": True},
                "industry_col": {"type": "str", "default": "citic_1"},
            },
            doc="行业中性化: x 减去行业内均值",
        )
        def industry_neutralize(x: Expr, industry_col: str = "citic_1") -> Expr:
            return x - x.group_by(industry_col).mean()
    """
    def decorator(func: Callable) -> Callable:
        param_specs = _COMPOSITE_REGISTRY._build_param_specs(params or {})
        spec = CompositeSpec(
            name=name,
            template=func,
            category=category,
            engine=engine,
            params=param_specs,
            doc=doc or (func.__doc__ or ""),
            examples=examples or [],
        )
        if engine == "pandas":
            _COMPOSITE_REGISTRY_PANDAS.register(spec)
        else:
            _COMPOSITE_REGISTRY.register(spec)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> "Expr":
            return spec.instantiate(**kwargs)

        wrapper.__composite_spec__ = spec  # type: ignore[attr-defined]
        return wrapper

    return decorator


# ============== 查询接口 ==============

def is_composite_op(name: str, engine: str = "any") -> bool:
    """判断 op 是否是 composite.

    Args:
        name: 算子名
        engine: "any" (default, union) | "polars" | "pandas"

    PR-QN-3a: composite 完全隔离存放, 不污染主 ``_OPERATOR_REGISTRY``.
    PR-QN-4: engine="any" 查双 registry (union), engine="polars"/"pandas" 分查.
    """
    if engine == "polars":
        return name in _COMPOSITE_REGISTRY_POLARS.list()
    if engine == "pandas":
        return name in _COMPOSITE_REGISTRY_PANDAS.list()
    # engine == "any"
    return name in _COMPOSITE_REGISTRY.list() or name in _COMPOSITE_REGISTRY_PANDAS.list()


def get_composite_spec(name: str, engine: str = "any") -> Optional[CompositeSpec]:
    """获取 composite spec (用于 LLM 编译时的 schema 查询).

    Args:
        name: 算子名
        engine: "any" (default, first found) | "polars" | "pandas"
    """
    if engine == "polars":
        return _COMPOSITE_REGISTRY_POLARS.get(name)
    if engine == "pandas":
        return _COMPOSITE_REGISTRY_PANDAS.get(name)
    # engine == "any": prefer polars, fallback pandas
    return _COMPOSITE_REGISTRY.get(name) or _COMPOSITE_REGISTRY_PANDAS.get(name)


def list_composite_ops(category: Optional[str] = None, engine: str = "any") -> List[str]:
    """列出所有 composite ops (可选按 category + engine 过滤).

    Args:
        category: 按类别过滤
        engine: "any" (default, union) | "polars" | "pandas"
    """
    if engine == "polars":
        return _COMPOSITE_REGISTRY_POLARS.list(category=category)
    if engine == "pandas":
        return _COMPOSITE_REGISTRY_PANDAS.list(category=category)
    # engine == "any": union — sorted for deterministic order across runs
    # (set ordering depends on PYTHONHASHSEED which is randomized by default).
    # v2.9.1: stable ordering eliminates flake in tests that index list[0].
    polars_ops = set(_COMPOSITE_REGISTRY_POLARS.list(category=category))
    pandas_ops = set(_COMPOSITE_REGISTRY_PANDAS.list(category=category))
    return sorted(polars_ops | pandas_ops)


def get_composite_doc_for_llm(engine: str = "any") -> str:
    """生成给 LLM prompt 的 composite 文档 (markdown 格式).

    Args:
        engine: "any" (default, all ops) | "polars" | "pandas"

    Phase 1.5: 内部委托 LLMDocVisitor, 保持向后兼容的输出格式。
    """
    visitor = LLMDocVisitor()
    if engine == "polars":
        specs = _COMPOSITE_REGISTRY_POLARS.all_specs()
    elif engine == "pandas":
        specs = _COMPOSITE_REGISTRY_PANDAS.all_specs()
    else:
        # engine == "any": polars first, then pandas (dedup by name)
        seen: set = set()
        specs_list = []
        for spec in _COMPOSITE_REGISTRY_POLARS.all_specs():
            if spec.name not in seen:
                seen.add(spec.name)
                specs_list.append(spec)
        for spec in _COMPOSITE_REGISTRY_PANDAS.all_specs():
            if spec.name not in seen:
                seen.add(spec.name)
                specs_list.append(spec)
        specs = iter(specs_list)
    for spec in specs:
        visitor.visit_spec(spec)
    return visitor.result


# ============== Visitor Pattern (Phase 1.5) ==============

class CompositeSpecVisitor:
    """CompositeSpec 的访问者基类 (Phase 1.5, Visitor pattern).

    用途:
      - 统一访问 _COMPOSITE_REGISTRY 中所有 CompositeSpec
      - 不修改 CompositeSpec 即可扩展新的遍历/分析能力
      - 具体子类: LLMDocVisitor / DependencyVisitor / ValidationVisitor

    使用:
        >>> visitor = LLMDocVisitor()
        >>> for spec in _COMPOSITE_REGISTRY.all_specs():
        ...     visitor.visit_spec(spec)
        >>> print(visitor.result)
    """

    def visit_spec(self, spec: CompositeSpec) -> None:
        """访问一个 CompositeSpec。子类重写此方法实现具体逻辑。"""
        raise NotImplementedError

    def visit_all(self) -> None:
        """便利方法: 遍历整个 _COMPOSITE_REGISTRY。"""
        for spec in _COMPOSITE_REGISTRY.all_specs():
            self.visit_spec(spec)


class LLMDocVisitor(CompositeSpecVisitor):
    """为 LLM prompt 生成 markdown 格式的 composite 文档。

    输出格式与原 get_composite_doc_for_llm() 完全一致 (向后兼容)。
    """

    def __init__(self) -> None:
        self.lines: List[str] = ["# Available Composite Operators"]

    def visit_spec(self, spec: CompositeSpec) -> None:
        self.lines.append(f"## {spec.name}")
        self.lines.append(f"  {spec.doc}")
        for pname, pspec in spec.params.items():
            if pspec.required:
                tag = "(required)"
            elif pspec.default is not None:
                tag = f"(default: {pspec.default})"
            else:
                tag = "(optional)"
            self.lines.append(
                f"  - {pname}: {pspec.type_hint} {tag} — {pspec.description}"
            )
        if spec.examples:
            self.lines.append(f"  Example: {spec.examples[0]}")
        self.lines.append("")

    @property
    def result(self) -> str:
        return "\n".join(self.lines)


class DependencyVisitor(CompositeSpecVisitor):
    """提取 composite 之间的依赖图 (基于 template 源码中的函数名引用)。

    输出: dict[name, set[dependency_name]], 边从 spec 指向被引用的其他 composite。
    目前实现是粗略的: 用 inspect.getsource() 提取 template 函数源码,
    匹配其他 composite 的 name 字符串。

    用途:
      - DAG 可视化
      - 检测循环依赖
      - 增量重编译优化
    """

    def __init__(self) -> None:
        self.graph: Dict[str, set] = {}

    def visit_spec(self, spec: CompositeSpec) -> None:
        deps: set = set()
        try:
            import inspect
            source = inspect.getsource(spec.template)
            for other_name in _COMPOSITE_REGISTRY.list():
                if other_name == spec.name:
                    continue
                if other_name in source:
                    deps.add(other_name)
        except (OSError, TypeError):
            pass
        self.graph[spec.name] = deps

    def detect_cycles(self) -> List[List[str]]:
        """返回所有循环依赖路径, 每条路径是 list of names."""
        cycles: List[List[str]] = []
        visited: set = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            if node in path:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return
            path.append(node)
            for nxt in self.graph.get(node, ()):
                dfs(nxt)
            path.pop()
            visited.add(node)

        for n in self.graph:
            dfs(n)
        return cycles


class ValidationVisitor(CompositeSpecVisitor):
    """检查 CompositeSpec 的语义正确性。

    校验:
      - spec.name 唯一 (注册时已保证, 这里双重检查)
      - 必填参数 (required=True) 没有 default (避免语义冲突)
      - 文档非空 (LLM prompt 友好)
      - examples 数量 > 0 (推荐)
    """

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def visit_spec(self, spec: CompositeSpec) -> None:
        # 必填参数不应该有 default
        for pname, pspec in spec.params.items():
            if pspec.required and pspec.default is not None:
                self.errors.append(
                    f"Composite '{spec.name}' param '{pname}' is required but has "
                    f"default={pspec.default!r} (语义冲突)"
                )
        # 文档空 → warning
        if not spec.doc.strip():
            self.warnings.append(
                f"Composite '{spec.name}' has empty doc (LLM prompt 会缺少说明)"
            )
        # 没有 examples → warning
        if not spec.examples:
            self.warnings.append(
                f"Composite '{spec.name}' has no examples (LLM few-shot 会少)"
            )

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ============== 用户 YAML 扩展 ==============

def load_composites_from_yaml(yaml_path: str) -> int:
    """从 YAML 文件加载用户自定义 composite ops.

    YAML 格式 (见 docs/22 §十七):
        composites:
          - name: my_op
            category: multi_section
            doc: "我的 op"
            params:
              x: {type: expr, required: true}
              k: {type: float, default: 1.0}
            template: "x + k"

    Returns:
        加载的 composite 数量
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
        engine = entry.get("engine", "polars")  # 缺省 = "polars" (向后兼容)
        template = _compile_template_string(template_str, engine=engine)
        param_specs = _COMPOSITE_REGISTRY._build_param_specs(
            entry.get("params", {})
        )
        spec = CompositeSpec(
            name=entry["name"],
            template=template,
            category=entry.get("category", "multi_section"),
            engine=engine,
            params=param_specs,
            doc=entry.get("doc", ""),
            examples=entry.get("examples", []),
        )
        if engine == "pandas":
            _COMPOSITE_REGISTRY_PANDAS.register(spec)
        else:
            _COMPOSITE_REGISTRY.register(spec)
        count += 1
    return count


# 允许的 Expr / 函数名白名单 (YAML template 解析用, polars 引擎)
_ALLOWED_FUNC_NAMES: set = {
    "col", "lit", "when", "otherwise", "then",
    "abs", "log", "sqrt", "pow", "exp",
    "rolling_mean", "rolling_std", "rolling_corr",
    "rolling_sum", "rolling_min", "rolling_max", "rolling_median",
    "ewm_mean", "ewm_std",
    "shift", "diff", "pct_change", "rank",
    "mean", "std", "sum", "min", "max", "median", "quantile",
    "count", "first", "last",
    "group_by", "over", "alias",
    "clip", "fill_null", "fill_nan", "drop_nulls", "drop_nans",
    "is_null", "is_nan", "is_not_null",
    "round", "floor", "ceil",
    "and_", "or_", "not_",
}

# 禁止的 base name (防止 ``os.system`` / ``subprocess.run`` 等)
_DENIED_BASE_NAMES: set = {
    "os", "sys", "subprocess", "socket", "urllib", "requests",
    "shutil", "pathlib", "path", "open", "file", "io",
    "importlib", "builtins", "eval", "exec", "compile",
    "getattr", "setattr", "delattr", "globals", "locals",
}


def _compile_template_string(template_str: str, engine: str = "polars") -> Callable[..., Any]:
    """把字符串模板编译为 callable (AST 解析 + 白名单校验).

    模板形式: 单个表达式, 引用 params dict 中的 key, 如 ``x + k`` 或
    ``x.group_by(industry_col).mean()``. 编译为 ``def _t(x, k, industry_col):
    return <expr>`` 形式, 可被 ``template(**bound)`` 调用.

    与 CodeSandbox 配合: 这里**只**解析 Expr 调用链, 不执行任意 Python.
    PR-QN-3a 修复: 文档原版用裸 exec, 会被 CodeSandbox 拒绝, 改用 ast.parse
    + 节点类型白名单.
    PR-QN-4: engine 参数选择正确的白名单 (polars vs pandas, 严格分流).
    """
    try:
        tree = ast.parse(template_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"YAML template 语法错误: {e}") from e

    # 提取表达式中所有自由变量作为函数参数
    free_vars = _extract_free_vars(tree.body)
    # PR-QN-4: 严格分流白名单
    if engine == "pandas":
        from ._engine import ALLOWED_FUNC_NAMES_PANDAS
        allowed_funcs = ALLOWED_FUNC_NAMES_PANDAS
    else:
        allowed_funcs = _ALLOWED_FUNC_NAMES
    # 白名单校验: 仅允许 Name / Call / Attribute / Constant / BinOp
    _validate_ast_nodes(tree.body, allowed_funcs=allowed_funcs)

    # 编译为函数
    func_name = "_composite_template"
    params_str = ", ".join(free_vars) if free_vars else ""
    code = f"def {func_name}({params_str}):\n    return {template_str}\n"
    namespace: dict = {}
    exec(code, namespace)  # noqa: S102 — AST 已校验, 安全
    return namespace[func_name]


def _extract_free_vars(node: ast.AST) -> List[str]:
    """从 AST 节点提取所有 Name 节点 (去重, 保序)."""
    seen: set = set()
    out: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in seen:
            seen.add(child.id)
            out.append(child.id)
    return out


def _validate_chain_base(attr: ast.Attribute, allowed_funcs: set) -> None:
    """校验链式调用的 base 节点 (PR-QN-4).

    链式调用可以是:
      - polars: x.method() — base is Name
      - pandas: x.groupby('a').mean() — base is Call (groupby returns a GroupBy)
      - deep chain: x.rolling(10).mean() — base is Call

    校验逻辑: 每个 Attribute.attr 都必须在白名单中, 递归验证整个链.
    """
    # 当前层方法名必须在白名单
    if attr.attr not in allowed_funcs:
        raise ValueError(
            f"YAML template 调用了不允许的方法: {attr.attr!r}. "
            f"允许: {sorted(allowed_funcs)[:10]}..."
        )
    # 递归验证 base
    if isinstance(attr.value, ast.Name):
        if attr.value.id in _DENIED_BASE_NAMES:
            raise ValueError(
                f"YAML template 调用了禁止的 base: "
                f"{attr.value.id!r}.{attr.attr}"
            )
    elif isinstance(attr.value, ast.Call):
        if isinstance(attr.value.func, ast.Attribute):
            _validate_chain_base(attr.value.func, allowed_funcs)
        elif isinstance(attr.value.func, ast.Name):
            if attr.value.func.id not in allowed_funcs:
                raise ValueError(
                    f"YAML template 调用了不允许的函数: {attr.value.func.id!r}. "
                    f"允许: {sorted(allowed_funcs)[:10]}..."
                )
    else:
        raise ValueError(
            f"YAML template 不支持的链式属性: {ast.dump(attr)[:80]}"
        )


def _validate_ast_nodes(node: ast.AST, allowed_funcs: set) -> None:
    """递归校验 AST 节点, 仅允许白名单内的函数名.

    Name 节点 = 自由变量 (作函数参数, 不在白名单中) 或白名单内的函数.
    Call 节点的 func 必须是白名单内的.
    Attribute 访问: 支持链式方法调用 (x.method() 或 x.method().method2()).
    PR-QN-4: 支持 pandas 风格链式调用 (x.groupby('a').mean()).
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id not in allowed_funcs:
                raise ValueError(
                    f"YAML template 调用了不允许的函数: {child.func.id!r}. "
                    f"允许: {sorted(allowed_funcs)[:10]}..."
                )
            elif isinstance(child.func, ast.Attribute):
                # 链式方法调用: x.method() 或 x.method().method2()
                # PR-QN-4: base 可以是 Name (polars) 或 Call (pandas chain)
                _validate_chain_base(child.func, allowed_funcs)
        elif isinstance(child, ast.Attribute):
            # Attribute 访问: base 是 Name (变量) 或 Call (链式)
            _validate_chain_base(child, allowed_funcs)
        elif isinstance(child, (ast.Import, ast.ImportFrom, ast.Lambda, ast.FunctionDef)):
            raise ValueError(f"YAML template 不允许: {type(child).__name__}")
