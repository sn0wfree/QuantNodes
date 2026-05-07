# Feature 3D：用户友好自定义算子 API 设计

## 1. 目标

为 QuantNodes 项目实现用户友好的自定义算子创建 API，让用户可以方便地定义自定义因子生成函数，并在表达式和配置文件中使用。

## 2. 核心设计

### 2.1 注册表隔离

- 新增 `_CustomOperatorRegistry` 类（位于 `QuantNodes/operators/registry.py`），与内置 `_OPERATOR_REGISTRY` 完全隔离
- 自定义算子命名空间隔离，不污染内置算子
- 支持命名空间前缀（如 `custom.`）

### 2.2 装饰器风格 API

```python
# 装饰器风格（直接注册）
@CustomOperator.point("my_double")
def my_double(f, multiplier=2.0):
    return f * multiplier

# Builder 链式风格
my_double = (CustomOperator.point("my_double")
    .param("multiplier", float, 2.0, "乘数")
    .execute(lambda f, multiplier: f * multiplier)
    .alias("custom_double")
    .register())

# 时间算子
@CustomOperator.time("my_ewm")
def my_ewm(s, span=30):
    return s.ewm_mean(span=span)

# 截面算子
@CustomOperator.section("my_zscore")
def my_zscore(s):
    return (s - s.mean()) / s.std()
```

### 2.3 模板工厂

```python
# 基于内置算子创建模板
my_ewm_30 = CustomOperator.time_from("my_ewm_30", "ewm_mean", span=30)

# 注册为命名模板
CustomOperator.register_template("my_ewm_30_template", template)
```

### 2.4 级联查询

`get_operator()` 修改为先查自定义注册表，再查内置注册表：

```python
def get_operator(name: str):
    # 1. 先查自定义注册表
    if name in _CUSTOM_OPERATOR_REGISTRY:
        return _CUSTOM_OPERATOR_REGISTRY[name]
    # 2. 再查内置注册表
    if name in _OPERATOR_REGISTRY:
        return _OPERATOR_REGISTRY[name]
    return None
```

### 2.5 参数声明式 API

通过 Builder 风格声明参数，默认值自动注入：

```python
CustomOperator.point("my_func")
    .param("multiplier", float, 2.0, "乘数")
    .param("shift", int, 0, "平移天数")
    .execute(lambda s, multiplier, shift: s * multiplier + shift)
    .register()
```

## 3. 文件结构

```
QuantNodes/operators/
├── __init__.py          # 导出 CustomOperator, OperatorTemplate, point, time, section
├── registry.py          # _CustomOperatorRegistry 类
├── templates.py         # OperatorTemplate 模板工厂类
├── custom.py            # CustomOperatorBuilder, CustomOperator, 装饰器函数
└── builtins.py          # 内置算子（已存在）

QuantNodes/factor_node/factor_functions/
└── __init__.py          # get_operator(), list_operators(), operator_info() 级联查询

tests/operators/
└── test_custom.py       # 26 个测试用例
```

## 4. 关键类

### 4.1 _CustomOperatorRegistry

隔离的用户自定义算子注册表，支持命名空间、前缀过滤、别名。

### 4.2 OperatorTemplate

模板工厂类，从现有算子复制签名，支持参数覆盖。

### 4.3 CustomOperatorBuilder

Builder 模式，支持链式调用：
- `.param()` 声明参数
- `.execute()` 设置执行函数
- `.alias()` 设置别名
- `.register()` 注册到全局注册表

## 5. 与表达式解析的集成

`ExprParser._parse_func_call()` 调用 `get_operator()` 查找算子函数。级联查询确保：
1. 用户自定义算子优先
2. 内置算子作为 fallback
3. Polars 方法调用作为最后 fallback

## 6. 测试覆盖

- 26 个测试用例覆盖：
  - 装饰器风格注册
  - Builder 链式风格
  - 参数声明与注入
  - 时间算子/截面算子
  - 命名空间隔离
  - 模板工厂
  - 级联查询
  - YAML 序列化/反序列化
  - 别名注册
  - 重复注册检测

## 7. 已完成

- ✅ `QuantNodes/operators/registry.py` — `_CustomOperatorRegistry`
- ✅ `QuantNodes/operators/templates.py` — `OperatorTemplate`
- ✅ `QuantNodes/operators/custom.py` — `CustomOperatorBuilder` + `CustomOperator` + 装饰器
- ✅ `QuantNodes/operators/__init__.py` — 导出新增类型
- ✅ `QuantNodes/factor_node/factor_functions/__init__.py` — 级联查询
- ✅ `tests/operators/test_custom.py` — 26 个测试全部通过
- ✅ `tests/operators/` — 154 个测试全部通过
- ✅ `tests/factor_node/test_factor_functions.py` — 49 个测试全部通过
