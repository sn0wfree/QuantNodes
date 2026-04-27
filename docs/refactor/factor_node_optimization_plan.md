# Factor Node 模块优化计划

## 优化目标
优化 factor_ node/ 模块，消除重复代码，提高可维护性。

## 本次优化任务

### 任务1: TimeOperation 和 PanelOperation 重复代码重构
**问题**: TimeOperation 和 PanelOperation 90% 代码重复 (~200行)

**原因**: 两者都实现相同的 LookBack 窗口逻辑，但分布在两个类中

**解决方案**: 创建 `_LookBackMixin` 基类，提取共用逻辑

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
core/_lookback_mixin.py (新增)
├── _LookBackMixin  基类
│   ├── LookBack: List
│   ├── LookBackMode: List
│   ├── iLookBack: Int
│   ├── iLookBackMode: Enum
│   ├── iInitData: DataFrame
│   ├── _compute_window_params()  ← 合并行193-202, 469-478
│   ├── _extend_dt_ruler()        ← 合并行222-226, 498-502
│   └── _init_lookback_data()   ← 合并行204-220, 480-497
│
TimeOperation(_LookBackMixin)  ← 精简版
├── DTMode, IDMode
└── readData, _calcData, prepareCacheData
│
PanelOperation(_LookBackMixin)  ← 精简版
├── DTMode, OutputMode, DescriptorSection
└── readData, _calcData, prepareCacheData
```

**预估效果**: -200行 重复代码

---

### 任务2: _calculate() 函数重构
**问题**: `_calculate()` 函数过长 (~100行)，混合单进程/多进程逻辑

**位置**: `factor_ table. py:258-356`

**解决方案**: 拆分为多个辅助函数

#### 当前结构
```python
def _calculate(args)
├── _build_task_ispatch()     ← 待提取 (行263-280)
├── _calculate_single_process() ← 待提取 (行284-320)
│   ├── _write_factor_data_batch()
│   └── _write_panel_batch()
└── _calculate_multi_process() ← 待提取 (行321-356)
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
| 1 | 提交之前的 refactoring | - | 待执行 |
| 2 | 创建 `_lookback_mixin.py` | `core/` | 待执行 |
| 3 | 重构 `TimeOperation` | `factor_ operation. py` | 待执行 |
| 4 | 重构 `PanelOperation` | `factor_ operation. py` | 待执行 |
| 5 | 拆分 `_calculate()` | `factor_ table. py` | 待执行 |
| 6 | 更新 `core/__init__.py` | `core/` | 待执行 |
| 7 | 运行测试 | - | 待执行 |

## 完成记录

| 日期 | 任务 | 结果 |
|------|------|------|
| 2026-04-27 | - | - |