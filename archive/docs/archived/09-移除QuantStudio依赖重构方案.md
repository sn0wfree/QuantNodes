# QuantNodes 重构设计文档

## 项目背景

### 目标
将 QuantNodes 项目完全移除对 QuantStudio 的依赖，使用代码复现的方式替代 QuantStudio 的功能。

### 决策历程
- **时间**: 2026-04-27
- **参与方**: 用户与 AI 助手

### 关键决策

| 决策项 | 选择 | 原因 |
|--------|------|------|
| Breaking Changes | ✅ 允许 | 不需要保持向后兼容 |
| 依赖移除顺序 | 按依赖关系顺序执行 | 确保每层依赖已实现 |
| QuantStudio 移除方式 | 完全移除 | 不保留可选兼容层 |
| 基类实现方案 | 自定义 `QuantNodesObject` 继承 `traits.HasTraits` | traits 已安装，比 QuantStudio 更轻量 |
| 目录结构 | `factor_node/` 暂时保留，后续合并到 `core/` | 保持现有结构稳定 |
| 工具函数位置 | 集中在 `core/tools.py` | 便于管理 |

---

## 项目现状分析

### QuantStudio 依赖位置（唯一）

| 文件 | 依赖程度 |
|------|----------|
| `factor_node/FactorDB.py` | 100% |
| `factor_node/FactorTools.py` | 100% |
| `factor_node/FactorOperation.py` | 100% |

### 依赖关系图

```
QuantStudio
    │
    └── factor_node/
            ├── FactorDB.py
            ├── FactorTools.py
            └── FactorOperation.py
```

### 其他模块状态

| 模块 | QuantStudio 依赖 | 说明 |
|------|-----------------|------|
| `core/` | ❌ 无 | 已重构完成 |
| `factor_table/` | ❌ 无 | 使用 ClickSQL |
| `backtest/` | ❌ 无 | 内部依赖 |
| `operator_node/` | ❌ 无 | 独立工具 |
| `database_node/` | ❌ 无 | 数据库连接器 |
| `utils_node/` | ❌ 无 | 工具函数 |

---

## QuantStudio 依赖分析

### 导入清单

| 文件 | 导入项 |
|------|--------|
| FactorDB.py | `__QS_Object__`, `__QS_Error__`, `Factor`, `PointOperation`, `genAvailableName`, `startMultiProcess`, `partitionListMovingSampling`, `fillNaByLookback`, `getShelveFileSuffix`, `testIDFilterStr` |
| FactorTools.py | `__QS_Error__`, `Factor`, `PointOperation`, `TimeOperation`, `SectionOperation`, `DataPreprocessingFun` |
| FactorOperation.py | `__QS_Error__`, `Factor`, `partitionList`, `partitionListMovingSampling` |

### 依赖分类

| 类别 | 项目 | 替换方案 | 难度 |
|------|------|----------|------|
| 基类 | `__QS_Object__` | `QuantNodesObject(HasTraits)` | 🟡 中 |
| 异常 | `__QS_Error__` | `QuantNodesError` | 🟢 低 |
| 因子类 | `Factor` | 自己实现 | 🔴 高 |
| 运算类 | `PointOperation`, `TimeOperation`, `SectionOperation` | 自己实现 | 🔴 高 |
| 工具函数 | `genAvailableName`, `partitionList` 等 | `core/tools.py` | 🟢 低 |

---

## 重构架构设计

### 目标架构

```
factor_node/ ──► core/
                      │
                      ├── errors.py
                      ├── quant_nodes_object.py
                      ├── factor_base.py
                      ├── factor_table.py
                      ├── factor_db.py
                      ├── operations.py
                      ├── cache_manager.py
                      ├── tools.py
                      └── factor_functions.py
```

### traits 依赖决策

**问题**: 是否移除 `traits` 库依赖？

**分析**:
- `__QS_Object__` 继承自 `traits.HasTraits`
- `factor_node/` 直接使用 `from traits.api import Str, Int, List, Enum`
- `traits` 是成熟的科学计算库（Enthought 维护）

**决定**: ✅ 保留 `traits` 依赖

**理由**:
1. traits 比 QuantStudio 更轻量
2. 项目已在使用 traits
3. 集中精力在业务逻辑而非基础设施

---

## 执行计划

### 阶段一：基础设施（P0）

| 文件 | 内容 | 预估行数 |
|------|------|----------|
| `core/errors.py` | `QuantNodesError`, `QuantNodesWarning` | ~20 |
| `core/quant_nodes_object.py` | `QuantNodesObject` 基类 | ~150 |

### 阶段二：因子基类（P0）

| 文件 | 内容 | 预估行数 |
|------|------|----------|
| `core/factor_base.py` | `Factor`, `DerivativeFactor` | ~600 |
| `core/factor_table.py` | `FactorTable`, `CustomFT` | ~500 |
| `core/factor_db.py` | `FactorDB`, `WritableFactorDB` | ~300 |

### 阶段三：运算操作（P0）

| 文件 | 内容 | 预估行数 |
|------|------|----------|
| `core/operations.py` | `PointOperation`, `TimeOperation`, `SectionOperation`, `PanelOperation` | ~1200 |
| `core/cache_manager.py` | 缓存管理 | ~300 |

### 阶段四：工具函数（P1）

| 文件 | 内容 | 预估行数 |
|------|------|----------|
| `core/tools.py` | 工具函数集合 | ~300 |

### 阶段五：因子运算函数（P1）

| 文件 | 内容 | 预估行数 |
|------|------|----------|
| `core/factor_functions.py` | ~50 因子运算函数 | ~2000 |

### 阶段六：文件重构（P2）

| 文件 | 操作 |
|------|------|
| `factor_node/FactorDB.py` | 删除 QuantStudio 导入，使用 `core/` |
| `factor_node/FactorTools.py` | 删除 QuantStudio 导入，使用 `core/` |
| `factor_node/FactorOperation.py` | 删除 QuantStudio 导入，使用 `core/` |

### 阶段七：测试（P2）

| 文件 | 内容 |
|------|------|
| `tests/test_factor_*.py` | 新增测试 |
| 运行完整测试套件 | 确保 95+ 测试通过 |

### 阶段八：清理（P3）

| 检查项 | 状态 |
|--------|------|
| 验证 `factor_node/` 无 QuantStudio 导入 | 待验证 |
| 验证 `core/` 无 QuantStudio 导入 | 待验证 |
| 更新依赖声明 | 待执行 |

---

## 工作量汇总

| 阶段 | 预估代码行数 | 优先级 | 状态 |
|------|-------------|--------|------|
| 阶段一 | ~170 | P0 | ⏳ 执行中 |
| 阶段二 | ~900 | P0 | ⏳ 待执行 |
| 阶段三 | ~1500 | P0 | ⏳ 待执行 |
| 阶段四 | ~300 | P1 | ⏳ 待执行 |
| 阶段五 | ~2000 | P1 | ⏳ 待执行 |
| 阶段六 | ~0 | P2 | ⏳ 待执行 |
| 阶段七 | ~500 | P2 | ⏳ 待执行 |
| 阶段八 | ~50 | P3 | ⏳ 待执行 |
| **总计** | **~5420** | | |

---

## 变更记录

### 2026-04-27 (下午)

#### 操作记录

1. **Phase 3.2: cache_manager.py**
   - 创建 `core/cache_manager.py`
   - 实现 `ErgodicMode` 类 - 遍历模式参数对象
   - 实现 `OperationMode` 类 - 运算模式参数对象
   - 实现 `prepare_mmap_factor_cache_data` 函数 - mmap 因子缓存准备
   - 实现 `prepare_mmap_id_cache_data` 函数 - mmap ID 缓存准备
   - 实现 `save_raw_data` 函数 - 原始数据保存

2. **Phase 6: factor_node 重构**
   - `FactorDB.py` - 删除 QuantStudio 导入，使用 `core/` 模块
     - `__QS_Object__` → `QuantNodesObject`
     - `__QS_Error__` → `FactorError`
     - `genAvailableName` → `gen_available_name`
     - `partitionListMovingSampling` → `partition_list_moving_sampling`
     - 修复局部导入 `PointOperation`
   - `FactorOperation.py` - 删除 QuantStudio 导入
     - `__QS_Error__` → `FactorError`
     - `Factor` → `core.factor_base.Factor`
     - `partitionList` → `core.tools.partition_list`
     - `partitionListMovingSampling` → `core.tools.partition_list_moving_sampling`
   - `FactorTools.py` - 删除 QuantStudio 导入
     - `__QS_Error__` → `FactorError`
     - `Factor` → `core.factor_base.Factor`
     - `PointOperation, TimeOperation, SectionOperation` → `core.operations`

3. **验证结果**
   - ✅ `grep` 确认 `factor_node/` 无 QuantStudio 导入
   - ✅ `grep` 确认 `core/` 无 QuantStudio 导入
   - ✅ 95 tests passed, 1 warning

4. **Phase 5: factor_functions.py**
   - 创建 `core/factor_functions.py` (~2200 行)
   - 提取所有因子运算函数从 `FactorTools.py`
   - 包含 97 个导出函数：单点运算、时间序列运算、截面运算、聚合函数等
   - 重构 `FactorTools.py` 为薄包装器，从 `core.factor_functions` 导入
   - 验证结果：
     - ✅ 97 exports verified
     - ✅ 95 tests passed

### 2026-04-27

#### 操作记录

1. **factor_node 代码扫描**
   - 发现 `factor_node/` 是唯一 QuantStudio 依赖所在
   - 其他模块（`core/`, `backtest/`, `factor_table/` 等）无 QuantStudio 依赖

2. **代码优化**
   - Phase 1: 修复 `TableNode.py` BaseNode 导入错误
   - Phase 2: 聚合函数重构（`aggr_sum`, `aggr_prod` 等 → 工厂函数）
   - Phase 3: Factor 二元运算符提取辅助方法

3. **依赖分析**
   - 确认 `__QS_Object__` 继承自 `traits.HasTraits`
   - 决定保留 `traits` 依赖

#### 决策记录

| 决策 | 选择 |
|------|------|
| 基类实现 | `QuantNodesObject(HasTraits)` |
| 目录结构 | 保持 `factor_node/`，后续合并 |
| 工具函数 | 集中在 `core/tools.py` |
| Breaking Changes | 允许 |

---

## 附录

### traits API 使用方式

```python
from traits.api import HasTraits, Str, Int, List, Enum, Function, Dict

class Example(HasTraits):
    Name = Str("默认名")
    Count = Int(0)
    Items = List()
    Mode = Enum("模式A", "模式B")
    Handler = Function()
    Options = Dict()
```

### `__QS_` 方法约定

| 方法 | 说明 |
|------|------|
| `__QS_initArgs__()` | 初始化配置参数 |
| `__QS_prepareRawData__()` | 准备原始数据 |
| `__QS_calcData__()` | 计算数据 |
| `__QS_prepareCacheData__()` | 准备缓存数据 |
| `__QS_saveRawData__()` | 保存原始数据 |
| `__QS_onBackTestMoveEvent__()` | 回测移动事件 |
| `__QS_onBackTestEndEvent__()` | 回测结束事件 |
| `__QS_genGroupInfo__()` | 生成分组信息 |

---

*文档版本: v1.0*
*最后更新: 2026-04-27*
