# QuantNodes factor_functions_v2 不兼容升级方案

**版本**: v2.1  
**创建日期**: 2026-04-29  
**状态**: ✅ 已完成  
**作者**: sn0wfree  

---

## 一、背景与目标

### 1.1 当前状态

当前 QuantNodes 存在两个 `factor_functions` 文件：

| 文件 | 行数 | 技术栈 | 功能 |
|------|------|--------|------|
| `factor_functions.py` | 2405 | pandas + numpy + statsmodels | 完整版（V1） |
| `factor_functions_v2.py` | 1108 | Polars | 轻量版（V2） |

### 1.2 V2 版本现状

**已实现**:
- `rolling_mean`, `rolling_std`, `rolling_max`, `rolling_min`, `rolling_sum`
- `ts_mean`, `ts_std`, `ts_corr`, `ts_cov`, `ts_rank`, `ts_delta`
- `rank`, `zscore`, `winsorize`, `neutralize`
- 约 51 个算子

**缺失**:
- 装饰器注册表系统
- `list_operators`, `get_operator`, `operator_info` 等查询 API
- Point 算子: `ceil`, `floor`, `fix`, `astype`, `applymap`, `nanargmax` 等 16 个
- Time 算子: `rolling_prod`, `rolling_skew`, `rolling_kurt`, `rolling_regress` 等 6+ 个
- **Multi-Section 算子**: 完全缺失！15 个算子
- Section 算子: `orthogonalize`, `fillNaNByFun`, `fillNaNByRegress` 等

### 1.3 升级目标

| 目标 | 说明 |
|------|------|
| **设计模式保留** | 从 V1 借鉴装饰器注册表系统 |
| **功能完整** | 补充所有缺失算子，达到 V1 水平 |
| **API 统一** | 提供 `list_operators`, `get_operator`, `operator_info` |
| **测试覆盖** | 创建完整的测试文件 |

### 1.4 设计原则

1. **不兼容升级** - 直接在 V2 上增强，不保留旧 API
2. **全功能覆盖** - 补充 V1 所有核心功能
3. **测试驱动** - 先写测试，再实现
4. **分阶段提交** - 每个 Phase 独立提交

---

## 二、功能差异分析

### 2.1 算子数量对比

| 类别 | V1 | V2 | 差异 |
|------|-----|-----|------|
| **Point** | 28 | 12 | -16 |
| **Time** | 32 | 26 | -6 |
| **Section** | 8 | 13 | +5 |
| **Multi-Section** | 15 | 0 | -15 |
| **总计** | **~83** | **~51** | **-32** |

### 2.2 V1 独有功能清单

#### Point 算子（需补充 16 个）
| 函数名 | 说明 |
|--------|------|
| `ceil` | 向上取整 |
| `floor` | 向下取整 |
| `fix` | 向零取整 |
| `astype` | 类型转换 |
| `applymap` | 应用自定义函数 |
| `fetch` | 获取指定位置数据 |
| `replace` | 值替换 |
| `nanprod` | 忽略空值求积 |
| `nanargmax` | 忽略空值求最大值索引 |
| `nanargmin` | 忽略空值求最小值索引 |
| `nanmedian` | 忽略空值求中位数 |
| `nanquantile` | 忽略空值求分位数 |
| `nancount` | 统计空值数量 |

#### Time 算子（需补充 6+ 个）
| 函数名 | 说明 |
|--------|------|
| `rolling_prod` | 滚动窗口求积 |
| `rolling_argmax` | 滚动窗口最大值索引 |
| `rolling_argmin` | 滚动窗口最小值索引 |
| `rolling_skew` | 滚动窗口偏度 |
| `rolling_kurt` | 滚动窗口峰度 |
| `rolling_count` | 滚动窗口计数 |
| `rolling_regress` | 滚动回归 |
| `ewm_var` | 指数加权移动方差 |
| `diff` | 差分算子 |
| `lag` | 滞后算子 |

#### Multi-Section 算子（需新增 15 个）
| 函数名 | 说明 |
|--------|------|
| `aggregate` | 聚合算子 |
| `disaggregate` | 解聚合算子 |
| `aggr_sum` | 聚合求和 |
| `aggr_prod` | 聚合求积 |
| `aggr_max` | 聚合最大值 |
| `aggr_min` | 聚合最小值 |
| `aggr_mean` | 聚合均值 |
| `aggr_std` | 聚合标准差 |
| `aggr_var` | 聚合方差 |
| `aggr_median` | 聚合中位数 |
| `aggr_quantile` | 聚合分位数 |
| `aggr_count` | 聚合计数 |
| `merge` | 合并因子 |
| `chg_ids` | ID转换 |

---

## 三、升级计划

### 3.1 任务划分

| 阶段 | 任务 | 预估增量 | 状态 |
|------|------|----------|------|
| **Phase 1** | 添加装饰器注册表系统 | ~80行 | ✅ 已完成 |
| **Phase 2** | 添加注册表查询 API | ~120行 | ✅ 已完成 |
| **Phase 3** | 补充缺失的 Point 算子 | ~250行 | ✅ 已完成 |
| **Phase 4** | 补充缺失的 Time 算子 | ~200行 | ✅ 已完成 |
| **Phase 5** | 新增 Multi-Section 算子 | ~300行 | ✅ 已完成 |
| **Phase 6** | 补充缺失的 Section 算子 | ~100行 | ✅ 已完成 |
| **Phase 7** | 创建测试文件 | ~500行 | ✅ 已完成 |
| **Phase 8** | 运行测试验证 | - | ✅ 已完成 |

**总计**: ~1550行新增代码

---

## 四、详细设计

### 4.1 装饰器注册表系统

```python
# factor_functions_v2.py 新增

import inspect
from typing import Callable, Dict, Any, Optional, List

class OperatorCategory:
    """算子分类常量"""
    POINT = "point"
    TIME = "time"
    SECTION = "section"
    MULTI_SECTION = "multi_section"

_OPERATOR_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {
    OperatorCategory.POINT: {},
    OperatorCategory.TIME: {},
    OperatorCategory.SECTION: {},
    OperatorCategory.MULTI_SECTION: {},
}

def register_operator(category: str, name: Optional[str] = None):
    """装饰器：自动注册算子到注册表"""
    def decorator(func: Callable):
        op_name = name or func.__name__
        sig = inspect.signature(func)
        
        _OPERATOR_REGISTRY[category][op_name] = {
            "name": op_name,
            "category": category,
            "func": func,
            "doc": inspect.getdoc(func) or "",
            "signature": str(sig),
            "parameters": list(sig.parameters.keys()),
        }
        return func
    return decorator
```

### 4.2 注册表查询 API

```python
# factor_functions_v2.py 新增

def list_operators(category: Optional[str] = None) -> List[str]:
    """列出所有算子名称"""
    if category:
        return list(_OPERATOR_REGISTRY.get(category, {}).keys())
    return [name for cat in _OPERATOR_REGISTRY for name in _OPERATOR_REGISTRY[cat]]

def get_operator(name: str, category: Optional[str] = None) -> Optional[Callable]:
    """根据名称获取算子函数"""
    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op["func"] if op else None
    
    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]["func"]
    return None

def operator_info(name: str, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """获取算子详细信息"""
    if category:
        op = _OPERATOR_REGISTRY.get(category, {}).get(name)
        return op if op else None
    
    for cat in _OPERATOR_REGISTRY:
        if name in _OPERATOR_REGISTRY[cat]:
            return _OPERATOR_REGISTRY[cat][name]
    return None

def generate_documentation(output_format: str = "markdown") -> str:
    """生成算子文档"""
    # 实现文档生成逻辑
    pass
```

### 4.3 示例：补充 Point 算子

```python
# factor_functions_v2.py 补充

@register_operator(OperatorCategory.POINT, "ceil")
def ceil(f: Union[Expr, str]) -> Expr:
    """向上取整"""
    f = _ensure_expr(f)
    return f.ceil()

@register_operator(OperatorCategory.POINT, "floor")
def floor(f: Union[Expr, str]) -> Expr:
    """向下取整"""
    f = _ensure_expr(f)
    return f.floor()

@register_operator(OperatorCategory.POINT, "applymap")
def applymap(f: Union[Expr, str], func: Callable) -> Expr:
    """应用自定义函数"""
    f = _ensure_expr(f)
    return f.map_elements(func)
```

### 4.4 ���增 Multi-Section 算子

```python
# factor_functions_v2.py 新增

@register_operator(OperatorCategory.MULTI_SECTION, "aggregate")
def aggregate(
    f: Union[Expr, str],
    group_by: str,
    method: str = "mean",
    **kwargs
) -> Expr:
    """按组聚合
    
    Args:
        f: 表达式或列名
        group_by: 分组列（如行业）
        method: 聚合方法 (mean/sum/std/...)
    
    Returns:
        聚合后的表达式
    """
    f = _ensure_expr(f)
    # 使用 Polars 的 group_by 实现
    ...
```

---

## 五、测试策略

### 5.1 测试文件结构

```python
# tests/test_factor_functions_v2.py

import pytest
import polars as pl

class TestRegistryAPI:
    """注册表 API 测试"""
    
    def test_list_operators(self):
        """列出所有算子"""
        ...
    
    def test_get_operator(self):
        """获取算子函数"""
        ...
    
    def test_operator_info(self):
        """获取算子信息"""
        ...

class TestPointOperators:
    """Point 算子测试"""
    
    def test_ceil(self):
        """向上取整"""
        ...
    
    def test_floor(self):
        """向下取整"""
        ...
    
    # ... 其他测试

class TestTimeOperators:
    """Time 算子测试"""
    
    def test_rolling_mean(self):
        """滚动均值"""
        ...
    
    # ... 其他测试

class TestMultiSectionOperators:
    """Multi-Section 算子测试"""
    
    def test_aggregate(self):
        """按组聚合"""
        ...
    
    # ... 其他测试
```

### 5.2 测试优先级

| 优先级 | 覆盖范围 |
|--------|----------|
| **P0** | 注册表 API、核心滚动/截面算子 |
| **P1** | 所有补充算子 |
| **P2** | Multi-Section 算子 |

---

## 六、验收标准

### 6.1 成功条件

1. `factor_functions_v2.py` 包含与 V1 等效的算子（~80+ 个）
2. 完整支持注册表查询 API
3. 测试覆盖率达到 80%+
4. 所有测试通过

### 6.2 回滚计划

如果遇到无法解决的问题：
- 保留 V1 原版文件
- 标记 V2 为 experimental
- 咨询团队寻求帮助

---

## 七、相关文件

### 7.1 修改文件

| 文件 | 操作 |
|------|------|
| `QuantNodes/factor_node/factor_functions_v2.py` | 修改（主要） |
| `QuantNodes/factor_node/__init__.py` | 修改（导出） |
| `tests/test_factor_functions_v2.py` | 新建 |

### 7.1 参考文件

| 文件 | 说明 |
|------|------|
| `QuantNodes/factor_node/factor_functions.py` | V1 原版（参考） |
| `QuantNodes/operators/time_series.py` | 时间序列底层实现 |
| `QuantNodes/operators/section.py` | 截面底层实现 |

---

## 八、版本历史

| 版本 | 日期 | 修改内容 |
|------|------|----------|
| v2.0 | 2026-04-29 | 初始 Polars 版本（1108行） |
| v2.1 | 2026-04-29 | 融合 V1 设计模式（进行中） |