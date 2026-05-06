# QuantNodes Polars 统一迁移方案

**版本**: v1.0  
**创建日期**: 2026-04-29  
**状态**: ✅ 已完成  
**作者**: sn0wfree

---

## 一、背景与目标

### 1.1 当前问题

当前 QuantNodes 存在两套因子计算框架：

| 框架 | 技术栈 | 代码量 | 问题 |
|------|--------|--------|------|
| **旧架构 (v1.x)** | traits + pandas + multiprocessing | ~3500行 | 依赖复杂，维护困难 |
| **新架构 (v2.0)** | Polars | ~650行 | 独立存在，未统一 |

### 1.2 迁移目标

| 目标 | 说明 |
|------|------|
| **统一技术栈** | 仅使用 Polars |
| **移除特殊依赖** | 移除 `traits` 和 `multiprocessing` |
| **代码简化** | ~3500行 → ~2500行 |
| **API兼容** | 保持现有 API 风格 |
| **分阶段迁移** | 逐步提交，确保测试通过 |

### 1.3 设计原则

1. **直接替换** - 不保留旧API兼容性
2. **纯Polars向量化** - 移除 multiprocessing
3. **分阶段提交** - 每个Phase独立提交
4. **测试驱动** - 确保测试通过再进行下一阶段

---

## 二、迁移计划

### 2.1 阶段划分

| 阶段 | 任务 | 代码量 | 预计时间 | 状态 |
|------|------|--------|----------|------|
| **Phase 1** | 创建 `factor_functions_v2.py` | ~600行 | 2天 | ✅ 已完成 |
| **Phase 2** | 改写 `quant_nodes_object.py` | ~150行 | 1天 | ✅ 已完成 |
| **Phase 3** | 改写 `factor.py` | ~200行 | 1天 | ✅ 已完成 |
| **Phase 4** | 简化 `factor_operation.py` | ~300行 | 2天 | ✅ 已完成 |
| **Phase 5** | 修改 `factor_table.py` / `factor_db.py` | ~200行 | 1天 | ✅ 已完成 |
| **Phase 6** | 清理并统一 `__init__.py` | ~100行 | 1天 | ✅ 已完成 |
| **Phase 7** | 测试通过验证 | - | - | ✅ 已完成 |
| **Phase 8** | 删除 `factor_nodes.py` | - | - | ✅ 已完成 |

**总计**: ~1550行变更，8天

---

## 三、详细任务

### Phase 1: factor_functions_v2.py

**任务**: 创建 Polars 版本的因子函数

**文件**: `factor_node/factor_functions_v2.py`

**内容**:
```python
# 保持原有函数签名，内部使用 Polars
import polars as pl
from typing import Any, Dict, List, Optional
from QuantNodes.operators import ts, sec, math, composite

def rolling_mean(f, window: int = 20, min_periods: int = None, **kwargs):
    """滚动窗口均值
    
    Args:
        f: 表达式或列名
        window: 窗口大小
        min_periods: 最小观测数
    
    Returns:
        Polars 表达式
    """
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_mean(f, window, min_periods)

def rolling_std(f, window: int = 20, min_periods: int = None, **kwargs):
    """滚动窗口标准差"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_std(f, window, min_periods)

def ts_mean(f, window: int = 20, **kwargs):
    """别名"""
    return rolling_mean(f, window, **kwargs)

def ts_std(f, window: int = 20, **kwargs):
    return rolling_std(f, window, **kwargs)

def ts_corr(f1, f2, window: int = 20, **kwargs):
    """滚动相关系数"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_corr(f1, f2, window)

def ts_rank(f, window: int = 20, **kwargs):
    """滚动排名"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_rank(f, window)

def ts_delta(f, periods: int = 1, **kwargs):
    """差分"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_delta(f, periods)

def ts_pct_change(f, periods: int = 1, **kwargs):
    """百分比变化"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_pct_change(f, periods)

# ================== 截面算子 ==================

def standardizeZScore(f, **kwargs):
    """Z-score 标准化"""
    from QuantNodes.operators import SectionOperators
    return SectionOperators.zscore(f)

def rank(f, **kwargs):
    """截面排名"""
    from QuantNodes.operators import SectionOperators
    return SectionOperators.rank(f)

def winsorize(f, lower: float = 0.01, upper: float = 0.01, **kwargs):
    """去极值"""
    from QuantNodes.operators import SectionOperators
    return SectionOperators.winsorize(f, lower, upper)

def neutralize(f, group=None, **kwargs):
    """行业中性的"""
    from QuantNodes.operators import SectionOperators
    if group:
        return SectionOperators.neutralize(f, group)
    return SectionOperators.neutralize_market(f)

def standardizeRank(f, **kwargs):
    """标准化排名"""
    return rank(f, method="average")

def weightStandardize(f, **kwargs):
    """加权标准化"""
    return standardizeZScore(f)

# ================== 时间序列算子 ==================

def expanding_mean(f, min_periods: int = None, **kwargs):
    """扩展窗口均值"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ewm_mean(f, alpha=0.5)

def expanding_std(f, min_periods: int = None, **kwargs):
    """扩展窗口标准差"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ewm_std(f, alpha=0.5)

def ewm_mean(f, alpha: float = 0.5, adjust: bool = True, **kwargs):
    """指数加权移动平均"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ewm_mean(f, alpha, adjust)

def ewm_std(f, alpha: float = 0.5, **kwargs):
    """指数加权移动标准差"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ewm_std(f, alpha)

# ================== 辅助函数 ==================

def isnull(f, **kwargs):
    """判断空值"""
    return f.is_null()

def notnull(f, **kwargs):
    """判断非空"""
    return f.is_not_null()

def nan_to_null(f, **kwargs):
    """NaN转null"""
    from QuantNodes.operators import MathOperators
    return MathOperators.nan_to_null(f)

def fill_null(f, value=0.0, **kwargs):
    """填充null"""
    from QuantNodes.operators import MathOperators
    return MathOperators.fill_null(f, value)

def fill_zero(f, **kwargs):
    """填充0"""
    from QuantNodes.operators import MathOperators
    return MathOperators.fill_zero(f)

def abs(f, **kwargs):
    """绝对值"""
    from QuantNodes.operators import MathOperators
    return MathOperators.abs(f)

def log(f, **kwargs):
    """对数"""
    from QuantNodes.operators import MathOperators
    return MathOperators.log(f)

def sign(f, **kwargs):
    """符号"""
    from QuantNodes.operators import MathOperators
    return MathOperators.sign(f)

def clip(f, lower=None, upper=None, **kwargs):
    """裁剪"""
    from QuantNodes.operators import MathOperators
    return MathOperators.clip(f, lower, upper)

def delay(f, periods: int = 1, **kwargs):
    """滞后"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_lag(f, periods)

def ref(f, periods: int = 1, **kwargs):
    """引用"""
    return delay(f, periods, **kwargs)

def correlation(f1, f2, window: int = 20, **kwargs):
    """相关系数"""
    return ts_corr(f1, f2, window, **kwargs)

def covariance(f1, f2, window: int = 20, **kwargs):
    """协方差"""
    from QuantNodes.operators import TimeSeriesOperators
    return TimeSeriesOperators.ts_cov(f1, f2, window)

# ================== 导出 ==================

__all__ = [
    # 滚动窗口
    "rolling_mean", "rolling_std", "rolling_max", "rolling_min",
    "rolling_sum", "rolling_median", "rolling_var",
    "rolling_skew", "rolling_kurt",
    
    # 时间序列
    "ts_mean", "ts_std", "ts_corr", "ts_rank",
    "ts_delta", "ts_pct_change",
    "ts_lag", "ts_lead",
    
    # 扩展窗口
    "expanding_mean", "expanding_std",
    "ewm_mean", "ewm_std",
    
    # 截面
    "standardizeZScore", "rank", "winsorize", "neutralize",
    "standardizeRank", "weightStandardize",
    
    # 数学
    "isnull", "notnull", "nan_to_null",
    "fill_null", "fill_zero",
    "abs", "log", "sign", "clip",
    
    # 别名
    "ts_argmax", "ts_argmin", "delay", "ref",
    "correlation", "covariance",
]
```

**依赖**:
- `QuantNodes.operators`

---

### Phase 2: quant_nodes_object.py

**任务**: 移除 traits 依赖，使用 dataclass

**当前问题**:
```python
# 旧
from traits.api import HasTraits, Str

class QuantNodesObject(HasTraits):
    name = Str()
    config = Dict()
```

**修改为**:
```python
# 新
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class QuantNodesObject:
    """简化对象基类"""
    name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, name: str = "", config: Dict = None, **kwargs):
        self.name = name
        self.config = config or {}
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def get_config(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any):
        self.config[key] = value
```

---

### Phase 3: factor.py

**任务**: 移除 traits 依赖

**当前问题**:
```python
# 旧
from traits.api import Enum, Int, Str
```

**修改为**:
```python
# 新
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

class FactorError(Exception):
    """因子错误"""
    pass

class DataType(Enum):
    DOUBLE = "double"
    STRING = "string"
    OBJECT = "object"

# 使用 dataclass 或简单 class
@dataclass
class Factor:
    name: str = ""
    data_type: str = "double"
    ...
```

---

### Phase 4: factor_operation.py

**任务**: 简化核心类，移除 multiprocessing

**当前问题**:
```python
# 旧
from traits.api import TraitFunction, Dict as TraitDict, Enum, ...
from multiprocessing import Queue, Event
```

**修改为**:
```python
# 新
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import polars as pl
import numpy as np

@dataclass
class DerivativeFactor:
    """简化因子运算基类"""
    name: str = ""
    operator: Callable = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    data_type: str = "double"
    
    def execute(self, data, **kwargs):
        if self.operator:
            return self.operator(data, **kwargs)
        return None

# 移除多进程相关代码
# 删除: Queue, Event, multiprocessing
```

**保留内容**:
- `PointOperation` - 简化为纯函数
- `TimeOperation` - 使用 Polars rolling
- `SectionOperation` - 使用 Polars group_by
- `PanelOperation` - 简化

---

### Phase 5: factor_table.py / factor_db.py

**任务**: 移除 traits 依赖

**当前问题**:
```python
# factor_table.py 旧
from traits.api import Enum, Instance, Int, List, Str
from multiprocessing import Lock, Process, Queue, cpu_count
```

**修改为**:
```python
# factor_table.py 新
import polars as pl
import numpy as np
from typing import List, Optional
import concurrent.futures  # 替代 multiprocessing

# 移除: multiprocessing
# 保留: ThreadPoolExecutor 或不保留 (Polars 自动并行)
```

---

### Phase 6: __init__.py 统一

**任务**: 清理导出

**当前**:
```python
# 旧
from QuantNodes.factor_node.factor_functions import ...  # v1.x
from QuantNodes.factor_node.factor_operation import ...
```

**修改为**:
```python
# 新
from QuantNodes.factor_node.factor_functions_v2 import *  # v2.0
from QuantNodes.factor_node.factor import Factor

# 导出
__all__ = [
    # 从 factor_functions_v2
    "rolling_mean", "rolling_std", "rolling_max", ...
    "standardizeZScore", "rank", "winsorize", ...
    
    # 从 factor
    "Factor", "DataFactor", "Factorize",
    
    # 从 factor_operation
    "DerivativeFactor", "PointOperation", "TimeOperation", ...
    
    # 移除
    # "QuantNodesObject" (已合并到 factor)
]
```

---

## 四、风险控制

### 4.1 测试验证

每个 Phase 后运行:
```bash
python -m pytest tests/test_factor_functions.py -v
python -m pytest tests/test_factor_node.py -v
```

### 4.2 回滚计划

如测试失败:
1. 暂停当前 Phase
2. 回滚最近修改
3. 修复后重新提交

### 4.3 兼容性

- 旧代码标记为 `deprecated`
- 保持导入路径兼容一段时间
- 在下一版本 (v3.0) 完全移除

---

## 五、提交记录

### 提交1: factor_functions_v2.py
```
feat: 添加 factor_functions_v2.py (Polars版本)
- 新增 20+ 因子函数
- 内部调用 QuantNodes.operators
```

### 提交2: quant_nodes_object.py
```
refactor: 重写 quant_nodes_object.py 移除 traits
- 使用 dataclass 替代 HasTraits
```

### 提交3: factor.py
```
refactor: 重写 factor.py 移除 traits
- 使用 Enum 和 dataclass
```

### 提交4: factor_operation.py
```
refactor: 简化 factor_operation.py
- 移除 multiprocessing
- 使用 Polars 替代 pandas
```

### 提交5: factor_table.py / factor_db.py
```
refactor: 重写 factor_table.py / factor_db.py
- 移除 traits 和 multiprocessing
- 使用 concurrent.futures
```

### 提交6: 统一 __init__.py
```
refactor: 统一 factor_node 导出
- 默认导出 v2.0 函数
- 清理废弃 API
```

---

## 六、工作量估算

| 阶段 | 新增行 | 修改行 | 删除行 | 净变化 |
|------|--------|--------|--------|--------|
| Phase 1 | +600 | - | - | +600 |
| Phase 2 | - | +50 | -100 | -50 |
| Phase 3 | - | +80 | -120 | -40 |
| Phase 4 | - | +100 | -300 | -200 |
| Phase 5 | - | +50 | -200 | -150 |
| Phase 6 | - | +30 | - | +30 |
| **总计** | **+600** | **+310** | **-720** | **+190** |

**最终代码量**: ~3700行 → ~2500行 (减少 ~1200行)

---

## 七、待确认事项

1. ~~**是否保留 factor_nodes.py** 中的 pandas 版本？~~ → **已完成**：factor_nodes.py 已删除，Pandas OOP 层已废弃
2. **是否需要测试并行性能**？ → 后续做
3. **预期完成日期**？

---

**文档状态**: ✅ 已完成 — 所有迁移任务已完成，factor_nodes.py 已删除