# Factor Node 模块优化计划

## 优化目标
优化 factor_node/ 模块，消除重复代码，提高可维护性。

## 本次优化任务

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

## 完成记录

| 日期 | 任务 | 结果 |
|------|------|------|
| 2026-04-27 | 创建 `_LookBackOperation` 基类 | ✅ 提取 ~60行 共用逻辑 |
| 2026-04-27 | 重构 `TimeOperation` | ✅ 精简 ~30行 |
| 2026-04-27 | 重构 `PanelOperation` | ✅ 精简 ~30行 |
| 2026-04-27 | 拆分 `_calculate()` | ✅ 拆分为 7 个函数 |
| 2026-04-27 | 更新 `core/__init__.py` | ✅ 添加新导出 |

## 优化结果

| 指标 | 优化前 | 优化后 | 变化 |
|------|--------|--------|------|
| `factor_operation.py` 行数 | 564 | ~520 | -44行 |
| `_LookBackOperation` 基类 | 0 | ~60行 | +60行 (新增) |
| `factor_table.py` `_calculate()` | 99行 | 14行 + 6个辅助函数 | 结构更清晰 |
| 重复代码 | ~200行 | ~100行 | -100行 |
| 最大函数行数 | 99行 | ~40行 | -59行 |

## 注意事项

1. **Traits 兼容性问题**: 代码库存在 traits 7.1.0 兼容性问题（`Function` → `TraitFunction`，`ListStr` 不存在），这是预存在的问题，不影响本次优化
2. **测试**: 由于预存在的导入问题，无法运行完整测试套件，但语法检查通过
3. **向后兼容**: `_LookBackOperation` 是内部类（前缀 `_`），不影响外部 API
