# factor_functions.py 单元测试计划

## 一、测试概述

### 1.1 测试目标
- 提高 `factor_functions.py` 的测试覆盖率从 4.2% 到 80%+
- 验证所有算子的正确性、边界条件和异常处理
- 确保装饰器模式的向后兼容性
- 验证算子注册器 API 的功能完整性

### 1.2 测试范围
- 95 个因子算子函数
- 11 个装饰器
- 4 个注册器 API 函数
- 向后兼容性验证

## 二、测试策略

### 2.1 测试分层
```
┌─────────────────────────────────────────────────────────┐
│                   集成测试层 (Integration)                │
│  - FactorNode 与算子的集成                                │
│  - 算子间组合运算                                        │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                   功能测试层 (Functional)                 │
│  - 算子正确性验证                                        │
│  - 边界条件测试                                          │
│  - 参数组合测试                                          │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│                   单元测试层 (Unit)                       │
│  - 装饰器功能测试                                        │
│  - 注册器 API 测试                                       │
│  - 内部辅助函数测试                                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 测试优先级

#### P0 - 核心算子（必须覆盖，30个）
| 类别 | 算子列表 |
|------|---------|
| **单点运算** | `isnull`, `notnull`, `log`, `sign`, `ceil`, `floor`, `clip`, `nansum`, `nanprod`, `nanmax`, `nanmin`, `nanmean`, `nanstd`, `nanvar`, `nanmedian`, `nancount` |
| **滚动窗口** | `rolling_mean`, `rolling_sum`, `rolling_std`, `rolling_var`, `rolling_max`, `rolling_min`, `rolling_median` |
| **截面运算** | `standardizeRank`, `standardizeZScore` |

#### P1 - 常用算子（高价值，30个）
| 类别 | 算子列表 |
|------|---------|
| **扩展窗口** | `expanding_mean`, `expanding_sum`, `expanding_std`, `expanding_var`, `expanding_max`, `expanding_min`, `expanding_median`, `expanding_count` |
| **EWM** | `ewm_mean`, `ewm_std`, `ewm_var` |
| **时间位移** | `lag`, `diff`, `fillna` |
| **截面处理** | `winsorize`, `standardizeQuantile`, `fillNaNByVal`, `fillNaNByFun` |
| **高级滚动** | `rolling_skew`, `rolling_kurt`, `rolling_quantile`, `rolling_rank`, `rolling_change_rate` |

#### P2 - 特殊算子（按需覆盖，35个）
| 类别 | 算子列表 |
|------|---------|
| **双因子** | `rolling_cov`, `rolling_corr`, `expanding_cov`, `expanding_corr`, `ewm_cov`, `ewm_corr` |
| **回归类** | `rolling_regress`, `rolling_regress_change`, `fillNaNByRegress`, `orthogonalize` |
| **多截面** | `aggregate`, `aggr_sum`, `aggr_prod`, `aggr_max`, `aggr_min`, `aggr_mean`, `aggr_std`, `aggr_var`, `aggr_median`, `aggr_quantile`, `aggr_count`, `disaggregate` |
| **其他** | `nav`, `merge`, `chg_ids`, `astype`, `where`, `replace` |

## 三、测试文件结构

```
tests/
└── test_factor_functions.py
    ├── Fixtures
    │   ├── simple_series_data()        # 简单时间序列
    │   ├── panel_data()                 # 面板数据 (id x date x value)
    │   ├── two_factor_data()            # 双因子数据
    │   ├── nan_data()                   # 含 NaN 的测试数据
    │   └── grouped_data()               # 分组截面数据
    │
    ├── 装饰器测试类 (TestDecorators)
    │   ├── test_point_operator_decorator()
    │   ├── test_rolling_operator_decorator()
    │   ├── test_expanding_operator_decorator()
    │   ├── test_ewm_operator_decorator()
    │   └── test_single_section_operator_decorator()
    │
    ├── 注册器 API 测试类 (TestRegistryAPI)
    │   ├── test_list_operators()
    │   ├── test_get_operator()
    │   ├── test_operator_info()
    │   └── test_generate_documentation()
    │
    ├── 单点算子测试类 (TestPointOperators)
    │   ├── test_isnull_notnull()
    │   ├── test_log_functions()
    │   ├── test_math_functions()
    │   ├── test_nan_aggregations()
    │   └── ...
    │
    ├── 时间序列算子测试类 (TestTimeSeriesOperators)
    │   ├── TestRollingOperators
    │   ├── TestExpandingOperators
    │   ├── TestEWMOperators
    │   └── TestTimeShiftOperators
    │
    ├── 截面算子测试类 (TestSectionOperators)
    │   ├── TestSingleSectionOperators
    │   ├── TestMultiSectionOperators
    │   └── TestSectionTransformOperators
    │
    └── 向后兼容性测试类 (TestBackwardCompatibility)
        ├── test_operator_signatures_unchanged()
        ├── test_return_types_consistent()
        └── test_migration_equivalence()
```

## 四、测试用例设计规范

### 4.1 标准测试用例模板
```python
def test_<operator_name>_basic():
    """测试 <算子名> 基础功能"""
    # 1. 准备测试数据
    data = ...
    # 2. 执行算子
    result = <operator>(data, param1=value1, param2=value2)
    # 3. 验证结果
    expected = ...  # 使用 pandas/numpy 原生函数计算期望值
    np.testing.assert_allclose(result, expected)

def test_<operator_name>_edge_cases():
    """测试 <算子名> 边界条件"""
    # - 全 NaN 输入
    # - 空输入
    # - 单元素输入
    # - 极端值
    pass

def test_<operator_name>_parameter_combinations():
    """测试 <算子名> 参数组合"""
    # - 不同参数值组合
    # - 可选参数
    pass
```

### 4.2 测试数据设计原则
1. **确定性**：测试数据产生确定性结果
2. **覆盖性**：包含正常值、边界值、异常值
3. **可验证性**：能用标准库函数计算期望结果

## 五、实施计划

### 阶段一：基础架构（1-2小时）
- [ ] 创建 `tests/test_factor_functions.py`
- [ ] 实现所有测试 fixture
- [ ] 实现注册器 API 测试
- [ ] 实现装饰器基础测试

### 阶段二：P0 核心算子测试（3-4小时）
- [ ] 单点运算算子（16个）
- [ ] 滚动窗口算子（7个）
- [ ] 截面算子（2个）
- [ ] 运行测试，确保全部通过

### 阶段三：P1 常用算子测试（3-4小时）
- [ ] 扩展窗口算子（8个）
- [ ] EWM 算子（3个）
- [ ] 时间位移算子（3个）
- [ ] 截面处理算子（4个）
- [ ] 高级滚动算子（5个）
- [ ] 运行测试，确保全部通过

### 阶段四：P2 特殊算子测试（2-3小时）
- [ ] 双因子算子（6个）
- [ ] 回归类算子（4个）
- [ ] 多截面算子（12个）
- [ ] 其他算子（3个）

### 阶段五：向后兼容性测试（1小时）
- [ ] 算子签名验证
- [ ] 返回类型一致性
- [ ] 迁移前后结果对比

## 六、成功标准

### 6.1 覆盖率目标
- **函数覆盖率**：≥ 80%
- **行覆盖率**：≥ 75%
- **P0 算子覆盖率**：100%
- **P1 算子覆盖率**：≥ 90%

### 6.2 质量标准
- 所有测试通过，无失败、无跳过
- 测试代码遵循项目代码规范
- 测试用例有清晰的文档和断言
- 边界条件和异常情况得到覆盖

## 七、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 算子间依赖复杂 | 高 | 先测试基础算子，再测试组合算子 |
| 测试数据构造困难 | 中 | 使用标准库函数生成对照结果 |
| 向后兼容性验证复杂 | 高 | 实现迁移前后结果对比测试 |
| 部分算子难以单元测试 | 低 | 通过集成测试覆盖 |

---

**最后更新**：2024年
**当前状态**：待实施
