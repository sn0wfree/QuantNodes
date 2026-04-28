# factor_functions.py 重构方案文档

## 概述

对 `QuantNodes/factor_node/factor_functions.py` 进行内部结构化重构，通过装饰器+注册表模式消除约 1100 行模板代码（约 50%），同时支持动态发现、文档生成、配置驱动。

---

## 重构目标

### 核心目标
1. ✅ **消除代码重复**：约 1100 行模板代码 → 约 150 行装饰器实现
2. ✅ **保持向后兼容**：100% API 兼容，现有代码无需修改
3. ✅ **单文件重构**：保持单一文件，不拆分到多文件

### 新增功能
1. ✅ **算子注册表**：支持运行时算子发现
2. ✅ **文档自动生成**：自动生成完整算子文档
3. ✅ **配置驱动**：支持通过配置字符串动态调用算子
4. ✅ **元信息查询**：查询算子签名、参数、文档

---

## 重构前后对比

| 指标 | 重构前 | 重构后 | 改善 |
|------|--------|--------|------|
| **总行数** | 2218 | ~1200 | **-46%** |
| **模板代码行数** | ~1100 | ~150 | **-86%** |
| **算子定义平均行数** | ~22 | ~8 | **-64%** |
| **算子数量** | ~100 | ~100 | 0 |
| **新算子添加工作量** | 22 行模板+实现 | 8 行实现 | **-64%** |

---

## 文件结构（单文件内分区）

```python
# ==============================================================================
# 1. 基础架构层 (约 150 行)
# ==============================================================================
# - 常量定义（魔法字符串）
# - 装饰器实现（单层装饰器，自动注册）
# - 注册表实现
# - 辅助函数（_genMultivariateOperatorInfo 等）

# ==============================================================================
# 2. 单点运算算子 (约 200 行，原 451 行)
# ==============================================================================
# - 使用 @point_operator 装饰器定义
# - 只保留实际计算逻辑

# ==============================================================================
# 3. 时间序列运算算子 (约 350 行，原 690 行)
# ==============================================================================
# - 使用 @rolling_operator / @expanding_operator / @ewm_operator 装饰器

# ==============================================================================
# 4. 单截面运算算子 (约 200 行，原 446 行)
# ==============================================================================
# - 使用 @single_section_operator 装饰器
# - 统一处理 mask/cat_data/weight_data 参数

# ==============================================================================
# 5. 多截面聚合运算算子 (约 250 行，原 571 行)
# ==============================================================================
# - 优化现有 _make_aggr_func 工厂模式

# ==============================================================================
# 6. 导出与注册表 API (约 50 行)
# ==============================================================================
# - list_operators(category=None)
# - get_operator(name, category=None)
# - operator_info(name)
# - generate_documentation()
```

---

## 核心设计模式

### 1. 单层装饰器模式

每个装饰器同时完成：注册 + 模板代码消除

```python
def point_operator(data_type="double"):
    """单点运算装饰器 - 自动注册+消除模板代码"""
    def decorator(impl_func):
        @wraps(impl_func)
        def wrapper(*factors, **kwargs):
            # 自动提取参数
            # 自动构建 OperatorArg
            # 自动返回 PointOperation
            pass
        
        # 自动注册到注册表
        op_name = impl_func.__name__.lstrip('_')
        _register(OperatorCategory.POINT, op_name, wrapper)
        return wrapper
    return decorator
```

### 2. 装饰器分类

| 装饰器 | 适用算子 | 数量 | 模板代码消除 |
|--------|----------|------|-------------|
| `@point_operator` | 简单单点运算 | ~25 | 每算子 ~10 行 |
| `@point_operator_with_args` | 带参数点运算 | ~6 | 每算子 ~15 行 |
| `@rolling_operator` | 滚动窗口运算 | ~18 | 每算子 ~12 行 |
| `@expanding_operator` | 扩展窗口运算 | ~13 | 每算子 ~12 行 |
| `@ewm_operator` | EWM 运算 | ~5 | 每算子 ~12 行 |
| `@single_section_operator` | 单截面运算 | ~8 | 每算子 ~40 行 |
| `@multi_section_operator` | 多截面运算 | ~6 | 每算子 ~15 行 |

### 3. 注册表 API

```python
# 列出所有算子
list_operators(category=None) -> List[str]

# 获取算子函数
get_operator(name, category=None) -> Optional[Callable]

# 获取算子详细信息
operator_info(name, category=None) -> Optional[Dict]

# 生成完整文档
generate_documentation(output_format="markdown") -> str
```

---

## 实施步骤

### 阶段 1：基础架构搭建
- [ ] 添加常量定义
- [ ] 实现装饰器基类/辅助函数
- [ ] 实现注册表系统
- [ ] 移动辅助函数到文件顶部

### 阶段 2：简单算子迁移
- [ ] 迁移简单点运算（isnull, notnull, sign, ceil, floor 等）
- [ ] 每步运行测试验证
- [ ] 预计消除 ~250 行模板代码

### 阶段 3：单截面算子重构（高收益）
- [ ] 实现 `@single_section_operator` 装饰器
- [ ] 迁移所有 8 个单截面算子
- [ ] 预计消除 ~300 行重复代码（最严重部分）

### 阶段 4：时间序列算子重构
- [ ] 实现 `@rolling_operator` / `@expanding_operator` / `@ewm_operator`
- [ ] 迁移 ~36 个时间序列算子
- [ ] 预计消除 ~400 行模板代码

### 阶段 5：多截面算子优化
- [ ] 优化现有 `_make_aggr_func` 工厂模式
- [ ] 消除实现函数中的重复代码

### 阶段 6：清理与验证
- [ ] 移除重复的 `_aggr_sum` 定义
- [ ] 修复 `rolling_regress` 中的 bare except
- [ ] 运行完整测试套件
- [ ] 验证向后兼容性

---

## 向后兼容保证

1. ✅ 所有现有函数名称保持不变
2. ✅ 所有现有函数签名保持不变
3. ✅ 所有现有返回值结构保持不变
4. ✅ `from QuantNodes.factor_node.factor_functions import *` 继续工作

---

## 新增功能使用示例

### 算子发现
```python
from QuantNodes.factor_node.factor_functions import list_operators, get_operator

# 列出所有点运算算子
point_ops = list_operators("point")

# 动态获取算子
rolling_mean = get_operator("rolling_mean", "time")
result = rolling_mean(factor, window=20)
```

### 文档生成
```python
from QuantNodes.factor_node.factor_functions import generate_documentation

# 生成 Markdown 文档
doc = generate_documentation("markdown")
with open("operator-docs.md", "w") as f:
    f.write(doc)
```

---

## 风险与应对

| 风险 | 概率 | 影响 | 应对措施 |
|------|------|------|----------|
| 装饰器参数提取不准确 | 中 | 高 | 逐算子迁移，每步测试 |
| 特殊算子（如 nanmean with weights）无法通用化 | 低 | 中 | 保留自定义 wrapper，不强制使用装饰器 |
| 性能影响（装饰器额外开销） | 低 | 低 | 实际计算远大于装饰器开销，可忽略 |

---

## 测试验证计划

1. ✅ 所有现有单元测试继续通过
2. ✅ 注册表 API 单元测试（新增）
3. ✅ 每个算子迁移后立即运行测试
4. ✅ 最终完整回归测试

---

## 后续优化方向

1. 添加类型提示到所有算子
2. 统一命名规范（camelCase → snake_case，保留别名）
3. 添加算子示例代码到文档
4. 支持算子版本管理
