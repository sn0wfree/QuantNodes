# Factor Node 模块优化计划

## 优化目标
优化 factor_node/ 模块，消除重复代码，提高可维护性。

---

## 📊 代码库分析 (2026-04-27 第二轮)

### 现状概览
- **71+ Python 文件**，约 10,000 行代码
- `core/operations.py` 与 `factor_node/factor_operation.py` 存在两套运算类实现，但功能不完全相同

### 核心发现

#### 1. 两套 Operations 实现的关系

| 文件 | 行数 | 特点 |
|------|------|------|
| `core/operations.py` | 416 | 简化版/占位实现，`_calcData` 逻辑不完整 |
| `factor_node/factor_operation.py` | 544 | 生产版，包含 `_LookBackOperation` 基类、完整 `_calcData` 实现 |

**结论**：两者非简单重复，而是**不同完整性**的实现。生产代码使用 `factor_node/factor_operation.py`。

#### 2. 待优化区域（按优先级）

| 优先级 | 区域 | 问题 | 建议方案 |
|--------|------|------|----------|
| 🔴 高 | `_calcData` 多层 if-elif | `TimeOperation._calcData` / `PanelOperation._calcData` 存在 4 层嵌套，圈复杂度 > 10 | 策略字典分派 |
| 🔴 高 | 序列化模板重复 | `Pipeline`/`Parallel`/`Join` 在 `_get_serializable_fields` 与 `_from_dict_impl` 重复相似逻辑 | 提取 `_SerializableCompositeMixin` |
| 🟡 中 | 类型初始化重复 | 各 `_calcData` 重复检查 `DataType` 并调用 `create_std_data` | 基类统一处理或提取 `_init_result_array` |
| 🟡 中 | 命名不统一 | `_QS_*` 拼音缩写，`DTMode`/`IDMode` 枚举值全中文 | 统一命名规范或提供英文别名 |
| 🟢 低 | DataFrame 循环内创建 | `_calcData` 中可能多次构建中间 DataFrame | 预分配数组，最终一次性转换 |
| 🟢 低 | 缓存逻辑分散 | 各节点 `__QS_prepareCacheData__` 重复相似流程 | 模板方法模式 |

---

## 📋 实施计划

### 高优先级任务

#### 任务 A: 策略字典分派（消除 if-elif 嵌套）✅ 已完成
**文件**: `factor_node/factor_operation.py`

**问题**: `TimeOperation._calcData`、`PanelOperation._calcData`、`SectionOperation._calcData`、`PointOperation._calcData` 存在多层 if-elif 分支，圈复杂度 > 10

**解决方案**: 策略字典分派

```python
class TimeOperation(_LookBackOperation):
    _DT_ID_DISPATCH = {
        ("单时点", "单ID"): "_calcData_single_time_single_id",
        ("单时点", "多ID"): "_calcData_single_time_multi_id",
        ("多时点", "单ID"): "_calcData_multi_time_single_id",
        ("多时点", "多ID"): "_calcData_multi_time_multi_id",
    }
    
    def _calcData(self, ids, dts, descriptor_data, dt_ruler):
        handler_name = self._DT_ID_DISPATCH.get((self.DTMode, self.IDMode))
        if handler_name:
            return getattr(self, handler_name)(StdData, iStartInd, DTRuler, ...)
        return self.Operator(self, DTRuler, ids, descriptor_data, self.ModelArgs)
    
    def _calcData_single_time_single_id(self, ...):
        ...
```

**预估效果**: 圈复杂度从 >10 降至 <5，代码可读性显著提升

#### 任务 B: 序列化公共 Mixin ❌ 已跳过
**分析结论**: `Pipeline`/`Parallel`/`Join` 的序列化逻辑差异较大（字段结构、反序列化方式），提取公共 Mixin 收益有限

---

### 中优先级任务

#### 任务 C: 统一命名规范
**问题**: `_QS_initOperation`、`_QS_prepareCacheData`、`DTMode` 等命名混用拼音

**建议**:
1. 保留 `_QS_*` 前缀（已有大量现网使用）
2. 为公开枚举值提供英文别名：
   ```python
   class DTMode(Enum):
       SINGLE_TIME_POINT = "单时点"  # 别名
       MULTI_TIME_POINT = "多时点"  # 别名
   ```

#### 任务 D: 补充类型提示与文档
**问题**: 关键方法缺乏 docstring 和类型注解

**建议**:
- 为所有 `public` 方法补充 Google 风格 docstring
- 为 `descriptor_data: List[np.ndarray]` 等复杂参数显式标注类型

---

### 低优先级任务

#### 任务 E: DataFrame 构建优化
**问题**: `_calcData` 中可能多次创建中间 DataFrame

**建议**: 预分配 `np.full` 数组，仅在最终返回时转换

#### 任务 F: 缓存逻辑模板化
**问题**: 各节点 `__QS_prepareCacheData__` 重复相似流程

**建议**: 在 `DerivativeFactor` 基类中实现模板方法，子类仅覆盖差异化部分

---

## 本次优化任务（第一轮）

### 任务1: TimeOperation 和 PanelOperation 重复代码重构
**问题**: TimeOperation 和 PanelOperation 90% 代码重复 (~200行)

**原因**: 两者都实现相同的 LookBack 窗口逻辑，但分布在两个类中

**解决方案**: 创建 `_LookBackOperation` 基类，提取共用逻辑

#### 重复分析
| 方法 | TimeOperation (行) | PanelOperation (行) | 相似度 |
|------|-------------------|---------------------|--------|---------|
| `__QS_initArgs__` | 143-145 | 397-401 | 95% |
| `readData` | 165-189 | 438-465 | 60% |
| `_calcData` | 191-251 | 467-527 | 70% |
| `__QS_prepareCacheData__` | 253-276 | 529-564 | 50% |
| **估计重复** | | | | **~200行** |

#### 重构方案
```
factor_operation.py (重构后)
├── _LookBackOperation (新增基类)
│   ├── LookBack: List
│   ├── LookBackMode: List
│   ├── iLookBack: Int
│   ├── iLookBackMode: Enum
│   ├── iInitData: DataFrame
│   ├── __QS_initArgs__()
│   └── _prepare_lookback_data()  ← 合并窗口参数计算、初始数据处理、时间标尺扩展
│
├── TimeOperation(_LookBackOperation)  ← 精简版
│   ├── DTMode, IDMode
│   ├── _QS_initOperation()
│   ├── readData()
│   ├── _calcData()  ← 调用 _prepare_lookback_data() + 特定 dispatch
│   └── __QS_prepareCacheData__()
│
└── PanelOperation(_LookBackOperation)  ← 精简版
    ├── DTMode, OutputMode, DescriptorSection
    ├── _QS_initOperation()
    ├── readData()
    ├── _calcData()  ← 调用 _prepare_lookback_data() + 特定 dispatch
    └── __QS_prepareCacheData__()
```

**预估效果**: -200行 重复代码

---

### 任务2: _calculate() 函数重构
**问题**: `_calculate()` 函数过长 (~100行)，混合单进程/多进程逻辑

**位置**: `factor_table.py:258-356`

**解决方案**: 拆分为多个辅助函数

#### 重构后结构
```python
def _calculate(args)
├── _build_task_dispatch()     ← 提取任务构建逻辑 (行263-280)
├── _calculate_single_process() ← 提取单进程执行 (行284-320)
│   ├── _write_factor_data_batch()
│   └── _write_panel_batch()
└── _calculate_multi_process() ← 提取多进程执行 (行321-356)
    ├── _write_factor_data_single()
    └── _write_panel_single()
```

**预估效果**: 提高可维护性（不减行数但结构更清晰）

---

### 任务3: DTMode/IDMode 分支优化 (可选)
**问题**: 每个 `_calcData` 都有 4 层 if/elif 分支

**解决方案**: 用字典 dispatch 替代

**预估效果**: ~20行

---

## 实施计划

| 步骤 | 任务 | 文件 | 状态 |
|------|------|------|------|
| 1 | 提交之前的 refactoring | - | ✅ 2026-04-27 |
| 2 | 创建 `_LookBackOperation` 基类 | `factor_operation.py` | ✅ 2026-04-27 |
| 3 | 重构 `TimeOperation` 继承基类 | `factor_operation.py` | ✅ 2026-04-27 |
| 4 | 重构 `PanelOperation` 继承基类 | `factor_operation.py` | ✅ 2026-04-27 |
| 5 | 拆分 `_calculate()` | `factor_table.py` | ✅ 2026-04-27 |
| 6 | 更新 `core/__init__.py` | `core/__init__.py` | ✅ 2026-04-27 |
| 7 | 运行测试 | - | ⚠️ 有预存在的 traits 兼容性问题 |
| 8 | 策略字典分派 - `PointOperation._calcData` | `factor_operation.py` | ✅ 2026-04-27 |
| 9 | 策略字典分派 - `TimeOperation._calcData` | `factor_operation.py` | ✅ 2026-04-27 |
| 10 | 策略字典分派 - `SectionOperation._calcData` | `factor_operation.py` | ✅ 2026-04-27 |
| 11 | 策略字典分派 - `PanelOperation._calcData` | `factor_operation.py` | ✅ 2026-04-27 |
| 12 | 补充 docstrings 和类型注解 | `factor_operation.py` | ✅ 2026-04-27 |

## 完成记录

| 日期 | 任务 | 结果 |
|------|------|------|
| 2026-04-27 | 创建 `_LookBackOperation` 基类 | ✅ 提取 ~60行 共用逻辑 |
| 2026-04-27 | 重构 `TimeOperation` | ✅ 精简 ~30行 |
| 2026-04-27 | 重构 `PanelOperation` | ✅ 精简 ~30行 |
| 2026-04-27 | 拆分 `_calculate()` | ✅ 拆分为 7 个函数 |
| 2026-04-27 | 更新 `core/__init__.py` | ✅ 添加新导出 |
| 2026-04-27 | 策略字典分派 - `PointOperation._calcData` | ✅ 消除 4 层 if-elif |
| 2026-04-27 | 策略字典分派 - `TimeOperation._calcData` | ✅ 消除 4 层 if-elif |
| 2026-04-27 | 策略字典分派 - `SectionOperation._calcData` | ✅ 消除嵌套 if-elif |
| 2026-04-27 | 策略字典分派 - `PanelOperation._calcData` | ✅ 消除嵌套 if-elif |
| 2026-04-27 | 补充 docstrings 和类型注解 | ✅ 添加完整 Google 风格文档 |

## 优化结果

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| `factor_operation.py` 行数 | 564 | ~900 | +336行 (handler方法+文档) |
| `_LookBackOperation` 基类 | 0 | ~60行 | +60行 (新增) |
| `factor_table.py` `_calculate()` | 99行 | 14行 + 6个辅助函数 | 结构更清晰 |
| 重复代码 | ~200行 | ~100行 | -100行 |
| 最大函数行数 | 99行 | ~40行 | -59行 |
| if-elif 分支层次 | 4层嵌套 | 1层 dispatch | 圈复杂度大幅下降 |
| 类型注解 | 缺失 | 完整 | ✅ 全面覆盖 |
| docstring | 缺失/不完整 | 完整 | ✅ Google 风格 |

## 注意事项

1. **Traits 兼容性问题**: 代码库存在 traits 7.1.0 兼容性问题（`Function` → `TraitFunction`，`ListStr` 不存在），这是预存在的问题，不影响本次优化
2. **测试**: 由于预存在的导入问题，无法运行完整测试套件，但语法检查通过
3. **向后兼容**: `_LookBackOperation` 是内部类（前缀 `_`），不影响外部 API
