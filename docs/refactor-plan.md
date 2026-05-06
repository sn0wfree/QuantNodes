# QuantNodes 架构重构计划

## 目标

统一算子注册系统，解决双重算子注册架构带来的维护负担和用户困惑问题。

## 问题描述

### 当前架构

```
┌─────────────────────────────────────────────────────────────┐
│  operators/ (静态类)                                        │
│  ├── TimeSeriesOperators.ts_mean()  ← 独立函数，无注册    │
│  ├── SectionOperators.rank()                               │
│  └── MathOperators.add()                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  factor_functions.py (@register_operator 注册表)           │
│  ├── rolling_mean() → 委托 TimeSeriesOperators.ts_mean()  │
│  ├── rank() → 委托 SectionOperators.rank()                 │
│  └── TA-Lib 174+ 指标                                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  agent/config/executor.py, loader.py                        │
│  └── import from factor_functions.py                       │
└─────────────────────────────────────────────────────────────┘
```

### 问题
1. 两套独立系统，用户困惑
2. `factor_functions` 80% 是薄代理，无增值
3. 单一文件 2500+ 行，难以维护
4. agent 系统依赖 `factor_functions` 注册表 API

## 解决方案

### 目标架构

```
┌─────────────────────────────────────────────────────────────┐
│  operators/ (唯一入口，门面层)                              │
│  ├── __init__.py (re-export 注册表API)                     │
│  ├── proxy.py (统一管理 re-export)                         │
│  ├── time_series.py → 代理 factor_functions.time_ops      │
│  ├── section.py → 代理 factor_functions.section_ops      │
│  ├── math.py → 代理 factor_functions.math_ops              │
│  ├── composite.py → 代理 factor_functions.composite_ops    │
│  └── talib.py → 代理 factor_functions.talib_ops           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  factor_functions/ (实现层)                                  │
│  ├── __init__.py (注册表、register_operator)              │
│  ├── time_ops.py (时间序列算子实现)                        │
│  ├── section_ops.py (截面算子实现)                         │
│  ├── math_ops.py (数学算子实现)                            │
│  ├── composite_ops.py (组合算子实现)                       │
│  └── talib_ops.py (TA-Lib 包装)                           │
└─────────────────────────────────────────────────────────────┘
```

### 关键原则
1. `operators/` 是**唯一入口**，对外接口不变
2. `factor_functions/` 是**实现层**，只被 `operators/` 调用
3. `agent/` 系统改从 `operators/` 导入，不直接依赖 `factor_functions/`

## 实施步骤

### Phase 1: 创建 factor_functions/ 目录结构

**1.1** 创建 `factor_functions/__init__.py`
- 从 `factor_functions.py` 提取注册表核心
- `_OPERATOR_REGISTRY`
- `register_operator` 装饰器
- `list_operators()`, `get_operator()`, `operator_info()`, `generate_documentation()`
- 辅助函数：`_ensure_expr`, `_COMBO_METHODS`, `_inject`

**1.2** 创建 `factor_functions/time_ops.py`
- 迁移 TIME 类算子（约 80 个函数）
- 包括 `rolling_mean`, `rolling_std`, `ts_corr`, `ts_rank`, `ts_delta`, `ewm_mean`, `decay_linear` 等

**1.3** 创建 `factor_functions/section_ops.py`
- 迁移 SECTION 类算子（约 40 个函数）
- 包括 `rank`, `zscore`, `winsorize`, `standardizeZScore`, `cross_sectional_rank` 等

**1.4** 创建 `factor_functions/math_ops.py`
- 迁移 POINT 类算子（约 60 个函数）
- 包括 `abs`, `log`, `sqrt`, `clip`, `sign`, `pow`, `round` 等

**1.5** 创建 `factor_functions/composite_ops.py`
- 迁移 MULTI_SECTION 类算子（约 20 个函数）
- 包括 `weighted_sum`, `combine`, `blend`, `regress`, `orthogonalize` 等

**1.6** 创建 `factor_functions/talib_ops.py`
- 从 `operators/talib.py` 迁移全部内容
- 保持 `try: import talib` 可选导入模式

### Phase 2: 创建 operators/proxy.py

```python
from QuantNodes.factor_functions import (
    list_operators,
    get_operator,
    register_operator,
    operator_info,
    generate_documentation,
)
```

### Phase 3: 修改 operators/*.py 为代理层

**3.1** 修改 `operators/time_series.py`
- 导入 `factor_functions.time_ops` 的函数
- `TimeSeriesOperators` 方法代理到对应函数

**3.2** 修改 `operators/section.py`（同上）

**3.3** 修改 `operators/math.py`（同上）

**3.4** 修改 `operators/composite.py`（同上）

**3.5** 修改 `operators/talib.py`
- 代理到 `factor_functions.talib_ops`

### Phase 4: 修改 operators/__init__.py

```python
from .proxy import (
    list_operators,
    get_operator,
    register_operator,
    operator_info,
    generate_documentation,
)

__all__ = [
    "ts", "sec", "math", "composite", "talib_ops",
    "list_operators", "get_operator", "register_operator",
    "operator_info", "generate_documentation",
    "TimeSeriesOperators", "SectionOperators", "MathOperators", "CompositeOperators",
]
```

### Phase 5: 重命名旧文件

```bash
mv QuantNodes/factor_node/factor_functions.py \
   QuantNodes/factor_node/_deprecated.py
```

### Phase 6: 修改 agent 系统 import 路径

**6.1** 修改 `agent/config/executor.py`
```python
# 修改前
from QuantNodes.factor_node.factor_functions import get_operator, register_operator

# 修改后
from QuantNodes.operators.proxy import get_operator, register_operator
```

**6.2** 修改 `agent/config/loader.py`
```python
# 修改前
from QuantNodes.factor_node.factor_functions import list_operators as _list_operators
from QuantNodes.factor_node.factor_functions import get_operator as _get_operator

# 修改后
from QuantNodes.operators.proxy import list_operators, get_operator
```

### Phase 7: 验证和测试

1. 验证 import 不破坏
2. 运行测试套件

## 文件变更清单

### 新增文件
- `QuantNodes/factor_node/factor_functions/__init__.py`
- `QuantNodes/factor_node/factor_functions/time_ops.py`
- `QuantNodes/factor_node/factor_functions/section_ops.py`
- `QuantNodes/factor_node/factor_functions/math_ops.py`
- `QuantNodes/factor_node/factor_functions/composite_ops.py`
- `QuantNodes/factor_node/factor_functions/talib_ops.py`
- `QuantNodes/operators/proxy.py`

### 修改文件
- `QuantNodes/operators/__init__.py`
- `QuantNodes/operators/time_series.py`
- `QuantNodes/operators/section.py`
- `QuantNodes/operators/math.py`
- `QuantNodes/operators/composite.py`
- `QuantNodes/operators/talib.py`
- `QuantNodes/agent/config/executor.py`
- `QuantNodes/agent/config/loader.py`

### 重命名文件
- `QuantNodes/factor_node/factor_functions.py` → `_deprecated.py`

## 注意事项

1. **避免循环导入**：`factor_functions/` 不能 import `operators/`
2. **保持 TA-Lib 可选导入**：使用 `try/except ImportError` 模式
3. **保持向后兼容**：现有 import 路径不变（对 `operators/` 的用户）
4. **文件大小控制**：每个 `*_ops.py` 控制在 400-600 行

## 提交计划

1. **Commit 1**: Phase 1-2 - 创建 `factor_functions/` 实现层
2. **Commit 2**: Phase 3-4 - 修改 `operators/` 为代理层
3. **Commit 3**: Phase 5 - 重命名旧文件
4. **Commit 4**: Phase 6-7 - 修改 agent 系统 + 验证测试

---

## 技术债务清理记录

### rank_sort 与 ConfigExecutor 兼容性问题（已修复）

**问题**: `test_rank_sort` 测试失败，`'list' object has no attribute 'alias'`

**原因**: `CompositeOperators.rank_sort()` 返回 `List[Expr]`，但 `executor._apply_operator()` 期望单个 `Expr`

**解决方案**:

1. **修改 `executor.run()` 方法**：检测 `List[Expr]` 返回值，使用双下划线后缀拆分存储（如 `c_rs__0`, `c_rs__1`）

2. **修改 `executor._execute_plan()` 方法**：检测 `__` 模式，使用原始 operation name 作为列别名

**关键改动**:
- 使用双下划线 `__` 而非单下划线 `_` 避免与 TA-Lib 名称冲突（如 `rsi_14`）

**验收**: 测试通过 ✅

---

## factor_functions 迁移与升级（已完成）

### 迁移背景

当前 QuantNodes 存在两套因子计算框架：

| 框架 | 技术栈 | 代码量 | 问题 |
|------|--------|--------|------|
| **旧架构 (v1.x)** | traits + pandas + multiprocessing | ~3500行 | 依赖复杂，维护困难 |
| **新架构 (v2.0)** | Polars | ~650行 | 独立存在，未统一 |

### 迁移目标

| 目标 | 说明 |
|------|------|
| **统一技术栈** | 仅使用 Polars |
| **移除特殊依赖** | 移除 `traits` 和 `multiprocessing` |
| **代码简化** | ~3500行 → ~2500行 |
| **API兼容** | 保持现有 API 风格 |
| **分阶段迁移** | 逐步提交，确保测试通过 |

### Polars 统一迁移（已完成）

| 阶段 | 任务 | 代码量 | 状态 |
|------|------|--------|------|
| **Phase 1** | 创建 `factor_functions.py` (Polars 版本) | ~600行 | ✅ |
| **Phase 2** | 改写 `quant_nodes_object.py` (移除 traits) | ~150行 | ✅ |
| **Phase 3** | 改写 `factor.py` (移除 traits) | ~200行 | ✅ |
| **Phase 4** | 简化 `factor_operation.py` (移除 multiprocessing) | ~300行 | ✅ |
| **Phase 5** | 修改 `factor_table.py` / `factor_db.py` | ~200行 | ✅ |
| **Phase 6** | 清理并统一 `__init__.py` | ~100行 | ✅ |
| **Phase 7** | 测试通过验证 | - | ✅ |
| **Phase 8** | 删除 `factor_nodes.py` | - | ✅ |

**总计**: ~1550行变更，净减少 ~1200行

### v2 升级（已完成）

| 阶段 | 任务 | 增量 | 状态 |
|------|------|------|------|
| **Phase 1** | 添加装饰器注册表系统 | ~80行 | ✅ |
| **Phase 2** | 添加注册表查询 API | ~120行 | ✅ |
| **Phase 3** | 补充缺失的 Point 算子 | ~250行 | ✅ |
| **Phase 4** | 补充缺失的 Time 算子 | ~200行 | ✅ |
| **Phase 5** | 新增 Multi-Section 算子 | ~300行 | ✅ |
| **Phase 6** | 补充缺失的 Section 算子 | ~100行 | ✅ |
| **Phase 7** | 创建测试文件 | ~500行 | ✅ |
| **Phase 8** | 运行测试验证 | - | ✅ |

### 重构效果

| 指标 | 重构前 | 重构后 | 改善 |
|------|--------|--------|------|
| **总行数** | ~3500 | ~2500 | **-29%** |
| **特殊依赖** | traits, multiprocessing | 无 | **移除** |
| **算子数量** | ~83 | ~317 | **+282%** |
| **新算子添加** | 22 行模板+实现 | 8 行实现 | **-64%** |
| **配置支持** | 不支持 | YAML 配置驱动 | **新增** |
