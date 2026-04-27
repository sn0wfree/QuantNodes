# 逻辑节点去 Lambda 化方案

> **文档版本**: v1.0  
> **创建日期**: 2026-04-27  
> **状态**: 已确认 ✅  
> **开发分支**: feature/no-lambda-control-nodes

---

## 一、问题诊断

### 当前问题
当前 `IfNode`、`WhileNode`、`MapNode` 使用 `Callable`（通常是 lambda）存在以下问题：

| 问题 | 影响 |
|------|------|
| **不可序列化** | 无法保存到配置、数据库，无法持久化策略 |
| **调试困难** | `to_dict()` 只能显示 `<lambda>`，无法看到具体逻辑 |
| **类型不安全** | 无法进行静态类型检查，错误只能在运行时发现 |
| **AI 不友好** | 大模型难以生成、解析、修改 lambda 表达式 |
| **无法优化** | 无法进行编译优化、预计算、缓存等 |

### 现有能力可借鉴
- `FactorDB` 已有成熟的**运算符重载**机制（`__add__`, `__gt__` 等）
- 因子运算的**符号计算架构**可以借鉴
- 目标：将条件逻辑从 "匿名函数" 升级为 "一等公民"

---

## 二、设计原则

1. **优雅优先** - API 设计追求简洁、可读、IDE 友好
2. **向后兼容** - 仍然支持 lambda，不破坏现有代码
3. **可组合** - 条件表达式支持逻辑运算组合
4. **可序列化** - 支持 `to_dict()` / `from_dict()` 往返
5. **AI 友好** - 便于大模型生成和理解
6. **统一架构** - 与 FactorNode 的符号计算能力统一
7. **标准库实现** - 不引入第三方依赖，使用纯 Python AST 解析

---

## 三、方案设计

### 3.1 核心抽象：Expression 表达式系统

创建新模块 `QuantNodes/core/expression.py`

```python
# 基类设计
class Expression(ABC):
    """表达式基类，所有计算逻辑的抽象"""
    
    @abstractmethod
    def evaluate(self, input_data: Any) -> Any:
        """执行表达式求值"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {"type": self.__class__.__name__}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Expression':
        """反序列化"""
        pass
    
    # ========== 运算符重载：支持优雅组合 ==========
    def __add__(self, other) -> 'BinaryOpExpr': pass
    def __sub__(self, other) -> 'BinaryOpExpr': pass
    def __mul__(self, other) -> 'BinaryOpExpr': pass
    def __gt__(self, other) -> 'ComparisonExpr': pass
    def __lt__(self, other) -> 'ComparisonExpr': pass
    def __eq__(self, other) -> 'ComparisonExpr': pass
    # ... 其他运算符
```

### 3.2 具体表达式类型

```python
# 原子表达式
class ConstantExpr(Expression):
    """常量值"""
    def __init__(self, value: Any): self.value = value

class VariableExpr(Expression):
    """变量/列访问：input_data[name]"""
    def __init__(self, name: str): self.name = name

class AttributeExpr(Expression):
    """属性访问：obj.attr"""
    def __init__(self, expr: Expression, attr: str):
        self.expr = expr
        self.attr = attr

class SubscriptExpr(Expression):
    """下标访问：obj[key]"""
    def __init__(self, expr: Expression, key: Any):
        self.expr = expr
        self.key = key

class MethodCallExpr(Expression):
    """方法调用：obj.method(*args, **kwargs)"""
    def __init__(self, expr: Expression, method: str, args: tuple, kwargs: dict):
        self.expr = expr
        self.method = method
        self.args = args
        self.kwargs = kwargs

# 运算表达式
class BinaryOpExpr(Expression):
    """二元运算：left op right"""
    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op
        self.right = right

class ComparisonExpr(Expression):
    """比较操作：left op right"""
    def __init__(self, left: Expression, op: str, right: Expression):
        self.left = left
        self.op = op  # '>', '<', '==', '>=', '<=', '!='
        self.right = right

class LogicalOpExpr(Expression):
    """逻辑运算：AND/OR/NOT"""
    def __init__(self, op: str, *operands: Expression):
        self.op = op
        self.operands = operands

# Lambda 包装（向后兼容）
class LambdaExpression(Expression):
    """包装 Callable，用于向后兼容"""
    def __init__(self, func: Callable[[Any], Any]):
        self.func = func
```

### 3.3 DSL 构建器：优雅的 API

```python
class ExpressionBuilder:
    """表达式构建器，提供链式 API"""
    
    def __call__(self, name: str) -> 'ExpressionBuilder':
        """Cond('column_name') - 按列/变量名访问"""
        return ExpressionBuilder(VariableExpr(name))
    
    def attr(self, name: str) -> 'ExpressionBuilder':
        """Cond.attr('metrics') - 按属性名访问"""
        return ExpressionBuilder(AttributeExpr(InputExpr(), name))
    
    def constant(self, value: Any) -> 'ExpressionBuilder':
        """常量值"""
        return ExpressionBuilder(ConstantExpr(value))
    
    # 自动属性访问：Cond.metrics → InputExpr().metrics
    def __getattr__(self, name: str) -> 'ExpressionBuilder':
        return ExpressionBuilder(AttributeExpr(InputExpr(), name))

# 全局单例
Cond = ExpressionBuilder()
```

### 3.4 AST 安全解析器

**文件：** `QuantNodes/core/ast_parser.py`

支持的语法子集：
- 常量（数字、字符串、布尔值、None）
- 变量、属性访问、下标访问
- 方法调用（无参数或只有常量参数）
- 比较运算（`> < == >= <= !=`）
- 逻辑运算（`and or not` → `& | ~`）
- 算术运算（`+ - * / // % **`）

安全白名单：
```python
ALLOWED_AST_NODES = {
    ast.Expression, ast.Compare, ast.BoolOp, ast.UnaryOp, ast.BinOp,
    ast.Attribute, ast.Subscript, ast.Call, ast.Name,
    ast.Constant, ast.Load,
    ast.Gt, ast.Lt, ast.Eq, ast.GtE, ast.LtE, ast.NotEq,
    ast.And, ast.Or, ast.Not,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
}

FORBIDDEN_METHODS = {'eval', 'exec', '__import__', 'compile', 'open', 'system'}
```

**使用示例：**
```python
from QuantNodes.core import Cond

# 1. 简单比较
condition1 = Cond('value') > 50  # input_data['value'] > 50

# 2. 嵌套属性
condition2 = Cond.attr('metrics').sharpe >= 1.5  # result.metrics.sharpe >= 1.5

# 3. 方法调用
condition3 = Cond('close').mean() > Cond('open').mean()  # df['close'].mean() > df['open'].mean()

# 4. 逻辑组合
condition4 = (Cond('close') > Cond('open')) & (Cond('volume') > 1000000)

# 5. 字符串表达式解析（AI 友好）
condition5 = Expression.parse("df['value'].mean() > 50")
```

### 3.5 控制流节点集成

```python
# 修改 IfNode 签名
class IfNode(BaseNode):
    def __init__(self,
                 condition: Union[Expression, Callable[[Any], bool], str],
                 true_branch: BaseNode,
                 false_branch: Optional[BaseNode] = None,
                 name: str = None):
        super().__init__(name=name or "IfNode")
        
        # 自动类型转换
        if isinstance(condition, str):
            self.condition = Expression.parse(condition)
        elif callable(condition):
            self.condition = LambdaExpression(condition)
        else:
            self.condition = condition
        
        self.true_branch = true_branch
        self.false_branch = false_branch
```

**新 API 使用：**
```python
# 方式 1：DSL 构建（推荐）
IfNode(
    condition=Cond.attr('metrics').sharpe >= 1.5,
    true_branch=HighVolStrategy(),
    false_branch=LowVolStrategy(),
)

# 方式 2：字符串表达式（AI 友好）
IfNode(
    condition="result.metrics.sharpe >= 1.5",
    true_branch=HighVolStrategy(),
)

# 方式 3：lambda（向后兼容）
IfNode(
    condition=lambda r: r.metrics.sharpe >= 1.5,
    true_branch=HighVolStrategy(),
)
```

---

## 四、实施路线图

### 阶段 1：核心表达式系统（1 天）
- [ ] 创建 `QuantNodes/core/expression.py`
- [ ] 实现 `Expression` 基类和原子表达式类型
- [ ] 实现运算符重载
- [ ] 实现 `ExpressionBuilder` (Cond) DSL
- [ ] 编写基础单元测试

### 阶段 2：AST 安全解析器（1.5 天）
- [ ] 创建 `QuantNodes/core/ast_parser.py`
- [ ] 基于 Python `ast` 模块实现安全解析框架
- [ ] 实现常用语法子集的转换
- [ ] 实现安全白名单机制
- [ ] 编写解析器单元测试

### 阶段 3：序列化支持（0.5 天）
- [ ] 实现所有表达式的 `to_dict()` / `from_dict()`
- [ ] 实现友好的 `__repr__`
- [ ] 序列化往返测试

### 阶段 4：控制流节点集成（1 天）
- [ ] 更新 `IfNode` 支持新的 Expression 系统
- [ ] 更新 `WhileNode` 支持新的 Expression 系统
- [ ] 更新 `MapNode` 支持新的 Expression 系统
- [ ] 保持向后兼容（lambda 仍然可用）
- [ ] 更新 `to_dict()` 输出可读表达式

### 阶段 5：测试、文档与示例（1 天）
- [ ] 完整单元测试覆盖
- [ ] 更新示例代码
- [ ] 编写 API 文档
- [ ] 性能基准测试
- [ ] `__init__.py` 导出公共 API

**总计：4-5 天**

---

## 五、预期收益

| 维度 | 改进前 | 改进后 | 收益 |
|------|-------|-------|------|
| **可序列化** | ❌ 不支持 | ✅ 完整支持 | 策略可持久化 |
| **可调试性** | ❌ 仅显示 `<lambda>` | ✅ 完整表达式树 | 调试效率 +200% |
| **类型安全** | ❌ 运行时检查 | ✅ 构建时验证 | 错误提前发现 |
| **AI 友好度** | ⚠️ 勉强可用 | ✅ 完美支持 | 大模型可直接生成条件字符串 |
| **代码可读性** | ⚠️ lambda 可读性差 | ✅ 自然语言式 DSL | 可读性 +100% |
| **可优化空间** | ❌ 无法优化 | ✅ 可编译、缓存、预计算 | 性能可进一步优化 |
| **外部依赖** | - | ✅ 纯标准库实现 | 无依赖 |

---

## 六、风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|---------|
| 表达式解析器复杂度高 | 中 | 中 | 先支持常用子集，逐步扩展 |
| 性能开销增加 | 低 | 低 | 缓存已解析的表达式 |
| 学习曲线 | 低 | 低 | 保持向后兼容，文档引导 |
| 安全问题（代码注入） | 低 | 高 | 使用 AST 解析白名单，禁止 eval |

---

## 七、发布与验证

- **开发分支**: feature/no-lambda-control-nodes
- **验证方式**: 在 feature 分支完成所有开发和测试后，提交 PR 进行 Code Review
- **合并条件**: 
  1. 所有现有测试通过（向后兼容）
  2. 新功能 100% 测试覆盖
  3. 安全审计通过
  4. 性能开销在可接受范围内

---

## 八、关键决策记录

| 决策项 | 选择 | 理由 |
|--------|------|------|
| DSL 入口命名 | `Cond` | 简洁且表意清晰 |
| 表达式解析方案 | AST 安全解析，先支持子集 | 安全第一，逐步完善 |
| MapNode 一致性 | 是，group_by 也支持表达式系统 | 保持架构统一 |
| 实现优先级 | 优先于 ConfigNode/DatabaseNode | 基础架构先行 |
| 依赖管理 | 纯标准库实现，不引入第三方 | 减少依赖，便于部署 |
| 发布方式 | 先在 feature 分支验证 | 降低风险 |

---

**文档完成时间**: 2026-04-27  
**文档创建者**: AI Assistant  
**审批状态**: 已确认 ✅
