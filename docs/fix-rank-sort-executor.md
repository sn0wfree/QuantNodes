# 修复: rank_sort 与 ConfigExecutor 兼容性问题

## 问题描述

`test_rank_sort` 测试失败，错误信息：
```
AssertionError: assert 'error' == 'success'
```

错误详情：
```
"'list' object has no attribute 'alias'"
```

## 问题分析

### 根本原因

```
┌─────────────────────────────────────────────────────────────────┐
│                    架构冲突                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  CompositeOperators.rank_sort()                               │
│       │                                                       │
│       └── 返回 List[Expr]（多个表达式）                         │
│             │                                                 │
│             ▼                                                 │
│  executor._apply_operator() 期望返回 Expr（单个表达式）        │
│       │                                                       │
│       └── 调用 .alias() → 失败                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 执行流程

1. `executor.run()` 调用 `_apply_operator(op)` 获取表达式
2. `rank_sort` 返回 `List[Expr]`（两个排名表达式）
3. 代码执行 `expr.alias(op.name)` → `'list' object has no attribute 'alias'`

## 解决方案

### 修改 1: run() 方法（第 346-349 行）

**当前代码：**
```python
# 2. 执行运算
for op in config.operations:
    expr = self._apply_operator(op)
    self._expressions[op.name] = expr
    result.factors[op.name] = expr
```

**修改后：**
```python
# 2. 执行运算
for op in config.operations:
    expr = self._apply_operator(op)
    
    # 处理 List[Expr] 返回值（如 rank_sort）
    if isinstance(expr, list):
        # 拆分存储，每个表达式一个 key
        for i, e in enumerate(expr):
            key = f"{op.name}_{i}"
            self._expressions[key] = e
        result.factors[op.name] = expr
    else:
        self._expressions[op.name] = expr
        result.factors[op.name] = expr
```

### 修改 2: _execute_plan() 方法（第 899-900 行）

**当前代码：**
```python
for name, expr in self._expressions.items():
    computed = computed.with_columns(expr.alias(name))
```

**修改后：**
```python
for name, expr in self._expressions.items():
    # 检查是否来自 List[Expr] 的拆分结果
    if "_" in name:
        base_name, idx = name.rsplit("_", 1)
        if idx.isdigit():
            # 使用原始 operation name 作为列名
            computed = computed.with_columns(expr.alias(base_name))
            continue
    
    computed = computed.with_columns(expr.alias(name))
```

## 执行效果

### 输入配置
```python
operations = [OperationConfig(
    type="composite", name="c_rs", category="rank_sort",
    inputs=["f1", "f2"], params={"weights": [0.6, 0.4]}
)]
```

### 执行后 _expressions
```python
{
    "f1": Expr,           # 原始因子
    "f2": Expr,           # 原始因子  
    "c_rs_0": Expr,       # rank_sort 第1个结果
    "c_rs_1": Expr,       # rank_sort 第2个结果
}
```

### 计算后的 DataFrame
```
┌──────┬──────┬─────────┬─────────┐
│ date │ code │ f1      │ c_rs    │
│ ---  │ ---  │ ---     │ ---     │
│ str  │ str  │ f64     │ f64     │
├──────┼──────┼─────────┼─────────┤
│ ...  │ ...  │ 1.0     │ 2.0     │  ← c_rs 取 c_rs_0
└──────┴──────┴─────────┴─────────┘
```

## 验收标准

- [ ] `test_rank_sort` 测试通过
- [ ] 其他 composite 测试无回归
- [ ] executor 仍然正确处理单表达式返回值