# QuantNodes Agent 配置文件驱动方案

**版本**: v1.0  
**创建日期**: 2026-04-29  
**状态**: 设计阶段  
**作者**: sn0wfree

---

## 一、设计背景与目标

### 1.1 问题陈述

当前 Agent 的工作流程中，Agent 需要编写大量逻辑代码来：
1. 编写因子代码 (Python)
2. 调用沙箱验证
3. 构建 Pipeline
4. 运行回测
5. 分析因子 IC
6. 运行测试验证

**痛点**：Agent 逻辑负担重，生成代码质量不稳定，难以保证测试通过。

### 1.2 设计目标

引入配置文件驱动的设计模式：

| 目标 | 说明 |
|------|------|
| **配置即策略** | Agent 编写 YAML 配置代替编写代码 |
| **自动闭环** | 配置 → 代码生成 → 验证 → 回测 自动执行 |
| ** Agent 兜底** | 不可配置部分 Agent 补充自定义算子 |
| **测试自运行** | 配置中声明测试，Agent 不需要手动运行 |

---

## 二、整体架构

### 2.1 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│                        Agent                                │
│  1. 编写 strategy_config.yaml                              │
│     ↓                                                      │
│  2. ConfigLoader 解析配置                                  │
│     ↓                                                      │
│  3. OperatorRegistry 检查覆盖度                          │
│     ├─ 全部可配置 → 4a                                     │
│     └─ 有无法表达 → 4b                                      │
│     ↓                                                      │
│  4a. FactorExecutor.run(config)                           │
│      → 自动生成代码 → 沙箱验证 → 运行测试                  │
│      ↓                                                     │
│  4b. 返回 [unresolved] → Agent 编写自定义算子             │
│      → 注册到 Registry → 重新检查                            │
│      ↓                                                     │
│  5. 合并执行 + 返回结果                                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模块结构

```
agent/
└── config/                          🆕 配置文件驱动模块
    ├── __init__.py
    ├── loader.py                    🆕 YAML配置解析器
    ├── registry.py                🆕 算子注册表
    ├── executor.py               🆕 配置执行器
    ├── template.py              🆕 配置模板生成
    └── types.py                🆕 类型定义
```

---

## 三、配置文件格式

### 3.1 完整示例

```yaml
# strategy_config.yaml
version: "1.0"
name: "momentum_alpha_v1"
description: "动量因子策略"

# 1. 数据源定义
data:
  source: "csv"                    # csv/database/api
  path: "data/stock_data.csv"
  columns:
    - date
    - code
    - open
    - high
    - low
    - close
    - volume
  date_column: "date"
  code_column: "code"

# 2. 因子定义
factors:
  - name: momentum_20d
    type: expression
    formula: "close / close.shift(20) - 1"
    description: "20日动量因子"

  - name: turnover
    type: expression
    formula: "volume / volume.rolling(20).mean()"
    description: "20日平均换手率"

# 3. 因子运算 (可配置部分)
operations:
  # 时间序列运算
  - type: time_series
    name: momentum_20d_ma
    category: ts_mean              # 算子名称
    inputs: [momentum_20d]        # 引用因子
    params:
      window: 20
      min_periods: 10
  
  # 截面运算
  - type: section
    name: momentum_rank
    category: rank
    inputs: [momentum_20d_ma]
    params:
      method: "dense"            # dense/ordinal/percent

# 4. 因子组合
composite:
  - name: alpha_factor
    formula: "momentum_rank * 0.6 + (1 - turnover) * 0.4"
    normalize: true              # z-score标准化
    winsorize:
      lower: 0.01
      upper: 0.01

# 5. 回测配置
backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  universe: "A_stock"
  initial_cash: 1000000
  commission: 0.001
  slippage: 0.001
  
  signals:
    buy_threshold: 0.05
    sell_threshold: -0.03
  positions:
    max_positions: 10
    rebalance_freq: "weekly"

# 6. 验证配置
validation:
  run_tests: true
  test_files:
    - "tests/test_factor*.py"
  
  metrics:
    ic_threshold: 0.02
    max_correlation: 0.7
    min_samples: 100
  
  # 自定义算子 (Agent额外实现)
  custom_operators:
    - "my_custom_operator.py"

# 7. 输出配置
output:
  format: "parquet"
  path: "outputs/result.parquet"
  save_signals: true
  save_positions: true
  save_equity_curve: true
```

### 3.2 简化版 (最小配置)

```yaml
# minimal_config.yaml
name: "simple_momentum"

factors:
  - name: returns
    formula: "close.pct_change(20)"

operations:
  - type: section
    category: rank
    inputs: [returns]

backtest:
  start_date: "2023-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
```

---

## 四、算子注册表

### 4.1 可配置的算子清单

#### 4.1.1 时间序列算子 (ts_*)

| 算子 | 参数 | 说明 |
|------|------|------|
| `ts_mean` | `window`, `min_periods`, `center` | 滚动均值 |
| `ts_sum` | `window`, `min_periods`, `center` | 滚动求和 |
| `ts_std` | `window`, `min_periods`, `center` | 滚动标准差 |
| `ts_max` | `window`, `min_periods`, `center` | 滚动最大值 |
| `ts_min` | `window`, `min_periods`, `center` | 滚动最小值 |
| `ts_median` | `window`, `min_periods` | 滚动中位数 |
| `ts_corr` | `window`, `min_periods`, `pair` | 滚动相关系数 |
| `ts_cov` | `window`, `min_periods`, `pair` | 滚动协方差 |
| `ts_rank` | `window` | 滚动排名 |
| `ts_argmax` | `window` | 滚动最大值位置 |
| `ts_argmin` | `window` | 滚动最小值位置 |
| `ts_delta` | `periods` | 差分 |
| `ts_pct_change` | `periods` | 百分比变化 |
| `ts_lag` | `periods` | 滞后 |

#### 4.1.2 截面算子 (section_*)

| 算子 | 参数 | 说明 |
|------|------|------|
| `rank` | `method` | 截面排名 |
| `scale` | `method`, `scale` | 归一化 |
| `winsorize` | `lower`, `upper` | 去极值 |
| `neutralize` | `method` |行业中性的 |
| `zscore` | | Z-score标准化 |
| `normalize` | `method` | 归一化到[0,1] |
| `percentile` | | 百分位排名 |

#### 4.1.3 算术算子

| 算子 | 参数 | 说明 |
|------|------|------|
| `add` | `scalar`, `factor` | 加法 |
| `sub` | `scalar`, `factor` | 减法 |
| `mul` | `scalar`, `factor` | 乘法 |
| `div` | `scalar`, `factor` | 除法 |
| `pow` | `exponent` | 幂运算 |
| `log` | | 对数 |
| `abs` | | 绝对值 |

#### 4.1.4 组合算子

| 算子 | 参数 | 说明 |
|------|------|------|
| `weighted_sum` | `weights` | 加权求和 |
| `weighted_avg` | `weights` | 加权平均 |
| `max` | | 取最大值 |
| `min` | | 取最小值 |

### 4.2 自定义算子接口

当配置文件无法表达时，Agent 编写自定义算子：

```python
# custom_operators.py

from typing import Any, Dict
import pandas as pd
import numpy as np

def custom_momentum_zscore(
    data: pd.DataFrame,
    factor_name: str,
    window: int = 20,
    zscroe_window: int = 60,
    **kwargs
) -> pd.DataFrame:
    """
    自定义动量z-score算子
    
    Args:
        data: 输入数据
        factor_name: 因子名称
        window: 动量窗口
        zscore_window: zscore窗口
    
    Returns:
        处理后的因子值
    """
    factor = data[factor_name]
    
    # 动量
    momentum = factor / factor.shift(window) - 1
    
    # z-score
    result = (momentum - momentum.rolling(zscroe_window).mean()) / \
            momentum.rolling(zscroe_window).std()
    
    return result.to_frame(factor_name)
```

### 4.3 自定义算子注册

```yaml
# 配置中声明
validation:
  custom_operators:
    - source: "custom_operators.py"
      functions:
        - custom_momentum_zscore
```

---

## 五、核心模块设计

### 5.1 ConfigLoader (`loader.py`)

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml

@dataclass
class FactorConfig:
    name: str
    type: str = "expression"
    formula: str = ""
    description: str = ""

@dataclass
class OperationConfig:
    type: str          # time_series/section/composite
    name: str
    category: str     # ts_mean/rank/...
    inputs: List[str]
    params: Dict[str, Any]

@dataclass
class BacktestConfig:
    start_date: str
    end_date: str
    initial_cash: float = 1000000
    commission: float = 0.001
    # ...

@dataclass
class StrategyConfig:
    version: str = "1.0"
    name: str
    description: str = ""
    factors: List[FactorConfig] = None
    operations: List[OperationConfig] = None
    composite: List[Dict] = None
    backtest: BacktestConfig = None
    validation: Dict = None
    # ...

class ConfigLoader:
    """YAML配置解析器"""
    
    def __init__(self, registry: "OperatorRegistry"):
        self.registry = registry
    
    def load(self, path: str) -> StrategyConfig:
        """加载YAML配置文件"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse(data)
    
    def _parse(self, data: dict) -> StrategyConfig:
        """解析配置字典为StrategyConfig"""
        # ...
    
    def check_coverage(self, config: StrategyConfig) -> CoverageReport:
        """检查配置覆盖度，返回不可配置的部分"""
        unresolved = []
        
        for factor in config.factors:
            # 检查因子公式是否可解析
            pass
        
        for op in config.operations:
            # 检查算子是否在注册表中
            if op.category not in self.registry:
                unresolved.append(op)
        
        return CoverageReport(
            covered=[...],
            unresolved=unresolved
        )
```

### 5.2 OperatorRegistry (`registry.py`)

```python
from typing import Dict, List, Callable, Any
from dataclasses import dataclass

@dataclass
class OperatorMetadata:
    name: str
    category: str        # time_series/section/arithmetic
    func: Callable
    params: Dict[str, Any]
    doc: str

class OperatorRegistry:
    """算子注册表 - 复用现有 factor_functions.py"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        # 复用现有 _OPERATOR_REGISTRY，无需重复实现
        from QuantNodes.factor_node.factor_functions import _OPERATOR_REGISTRY
        self._operators = _OPERATOR_REGISTRY
    
    def register(self, name: str, category: str, func: Callable, params: Dict):
        """注册算子"""
        self._operators[name] = OperatorMetadata(
            name=name,
            category=category,
            func=func,
            params=params,
            doc=func.__doc__
        )
    
    def get(self, name: str) -> OperatorMetadata:
        """获取算子"""
        return self._operators.get(name)
    
    def list_by_category(self, category: str) -> List[str]:
        """按类别列出算子"""
        return [
            op.name for op in self._operators.values()
            if op.category == category
        ]
    
    def register_custom(self, module_path: str):
        """注册自定义算子"""
        import importlib.util
        spec = importlib.util.spec_from_file_location("custom", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 自动发现并注册
        for name in dir(module):
            if name.startswith("custom_"):
                func = getattr(module, name)
                self.register(name, "custom", func, {})
```

### 5.3 FactorExecutor (`executor.py`)

```python
from typing import Dict, Any
import pandas as pd
import numpy as np

class FactorExecutor:
    """配置执行器：配置 → 代码生成 → 执行"""
    
    def __init__(self, registry: OperatorRegistry, sandbox: "CodeSandbox"):
        self.registry = registry
        self.sandbox = sandbox
    
    def run(self, config: StrategyConfig) -> ExecutionResult:
        """执行完整流程"""
        results = {}
        
        # 1. 生成因子代码
        code = self._generate_code(config)
        
        # 2. 沙箱验证
        validation_result = self.sandbox.validate(code)
        if not validation_result.is_safe:
            raise ExecutionError(f"Code not safe: {validation_result.errors}")
        
        # 3. 执行因子计算
        results["factors"] = self._compute_factors(config, code)
        
        # 4. 执行运算
        results["operations"] = self._compute_operations(config)
        
        # 5. 计算组合因子
        results["composite"] = self._compute_composite(config)
        
        # 6. 运行回测
        if config.backtest:
            results["backtest"] = self._run_backtest(config)
        
        # 7. 运行验证测试
        if config.validation.get("run_tests"):
            results["tests"] = self._run_tests(config)
        
        return ExecutionResult(**results)
    
    def _generate_code(self, config: StrategyConfig) -> str:
        """根据配置生成Python代码"""
        lines = [
            "import pandas as pd",
            "import numpy as np",
            "",
            "# 因子定义",
        ]
        
        for factor in config.factors:
            if factor.type == "expression":
                lines.append(
                    f"{factor.name} = data.eval('{factor.formula}')"
                )
        
        # ...
        return "\n".join(lines)
    
    def _compute_factors(self, config: StrategyConfig, code: str) -> Dict:
        """计算所有因子"""
        # 执行代码，返回因子DataFrame
        local_vars = {}
        exec(code, {"pd": pd, "np": np}, local_vars)
        
        factors = {}
        for factor in config.factors:
            if factor.name in local_vars:
                factors[factor.name] = local_vars[factor.name]
        
        return factors
    
    def _compute_operations(self, config: StrategyConfig) -> Dict:
        """执行因子运算"""
        results = {}
        
        for op in config.operations:
            metadata = self.registry.get(op.category)
            if metadata is None:
                raise ExecutionError(f"Operator not found: {op.category}")
            
            # 调用算子函数
            result = metadata.func(
                data=results.get(op.inputs[0]),
                **op.params
            )
            results[op.name] = result
        
        return results
    
    def _get_coverage_gap(self, config: StrategyConfig) -> List[str]:
        """获取配置覆盖缺口"""
        gaps = []
        
        for op in config.operations:
            if op.category not in self.registry:
                gaps.append(f"Operator '{op.category}' not in registry")
        
        return gaps
```

---

## 六、Agent 工作流程集成

### 6.1 Agent 使用��置��件的完整流程

```
1. Agent 分析用户需求
   → "帮我生成一个20日动量因子策略，回测2023年"
   
2. Agent 编写配置文件
   → strategy_config.yaml
   
3. ConfigLoader 加载并检查
   → CoverageReport(
       covered: ["ts_mean", "rank", ...],
       unresolved: []
     )
   
4. FactorExecutor 自动执行
   → 生成代码 → 验证 → 回测 → 验证测试
   
5. 返回结果
   → IC: 0.05, 回测收益: 15%, Sharpe: 1.2
```

### 6.2 需要 Agent 兜底的流程 (有不可配置部分)

```
1. Agent 编写配置文件
   → 包含自定义算子 "custom_momentum_zscore"
   
2. ConfigLoader 检查覆盖度
   → CoverageReport(
       covered: ["ts_mean", "rank"],
       unresolved: [
         UnresolvedOp(name="custom_momentum_zscore", type="custom")
       ]
     )
   
3. 返回 unresolved 给 Agent
   → "以下算子无法配置，请实现: custom_momentum_zscore"
   
4. Agent 编写自定义算子
   → custom_operators.py
   
5. 重新执行
   → 成功执行完整流程
```

---

## 七、配置文件模板

### 7.1 动量因子配置 (示例)

```yaml
version: "1.0"
name: "momentum_factor"
description: "经典动量因子"

factors:
  - name: returns_20d
    formula: "close / close.shift(20) - 1"
  
  - name: returns_60d
    formula: "close / close.shift(60) - 1"

operations:
  - type: time_series
    name: returns_20d_ma
    category: ts_mean
    inputs: [returns_20d]
    params:
      window: 20
  
  - type: section
    name: momentum_rank
    category: rank
    inputs: [returns_20d_ma]

composite:
  - name: alpha
    formula: "momentum_rank"
    normalize: true
    winsorize:
      lower: 0.01
      upper: 0.01

backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000

validation:
  run_tests: true
  metrics:
    ic_threshold: 0.02
```

### 7.2 配置生成 Prompt 模板

```
请帮我生成一个因子策略配置文件，满足以下需求：

1. 因子要求：
   - 使用 N 日动量因子
   - 进行行业、市值中性化处理
   
2. 回测要求：
   - 回测区间：{start_date} - {end_date}
   - 初始资金：{initial_cash}
   
3. 验证要求：
   - IC 阈值 > 0.02
   - 因子相关性 < 0.7

请生成 YAML 格式的配置文件。
```

---

## 八、错误处理

### 8.1 错误分类

| 错误类型 | 说明 | 处理方式 |
|----------|------|----------|
| `ConfigParseError` | YAML格式错误 | Agent 修复配置 |
| `OperatorNotFound` | 算子不存在 | Agent 实现自定义算子 |
| `CodeValidationError` | 代码不安全 | Agent 修复代码 |
| `ExecutionError` | 执行失败 | Agent 调试 |
| `BacktestError` | 回测失败 | Agent 调整配置 |
| `TestFailure` | 测试失败 | Agent 修复代码/配置 |

### 8.2 错误响应格式

```json
{
  "status": "error",
  "error_type": "OperatorNotFound",
  "message": "算子 'custom_momentum_zscore' 未找到",
  "suggestion": "请在 custom_operators.py 中实现该算子",
  "template": "def custom_momentum_zscore(data, factor_name, window=20, **kwargs): ..."
}
```

---

## 九、验证测试集成

### 9.1 测试发现与运行

```python
class TestRunner:
    """测试运行器"""
    
    def discover_tests(self, config: StrategyConfig) -> List[str]:
        """发现配置中声明的测试"""
        test_files = config.validation.get("test_files", [])
        # 使用 pytest 自动发现测试
        return pytest.collect(test_files)
    
    def run(self, test_paths: List[str]) -> TestResult:
        """运行测试"""
        result = pytest.main([
            "-v", "--tb=short",
            *test_paths
        ])
        return TestResult(
            passed=result.wasSuccessful(),
            output=result.output,
            failures=result.failures
        )
```

### 9.2 Agent 视角的错误处理

```
Agent 视角:

执行配置文件 → 
  ├─ 成功 → 返回结果
  │
  └─ 失败 → 
      ├─ ConfigParseError → 修复 YAML
      ├─ OperatorNotFound → 实现自定义算子
      ├─ CodeValidationError → 修复代码
      ├─ ExecutionError → 调试代码
      ├─ TestFailure → 修复直到通过
      └─ BacktestError → 调整配置

Agent 不再需要手动运行测试!
```

---

## 十、文件位置与工作量

```
QuantNodes/agent/
├── config/
│   ├── __init__.py
│   ├── loader.py          # ~100行 - YAML解析
│   ├── registry.py      # ~0行  - 复用现有 _OPERATOR_REGISTRY
│   ├── executor.py     # ~100行 - 代码生成执行
│   ├── template.py     # ~50行 - 配置模板
│   └── types.py        # ~50行 - dataclass定义
│   # 总计: ~300行
│
├── templates/
│   └── config/
│       ├── momentum.yaml       # 动量因子模板
│       ├── mean_reversion.yaml  # 均值回复模板
│       └── empty.yaml         # 空模板
```

---

## 十一、工作量估算

### 修正说明

**关键发现**: 无需重复实现算子，复用现有 `QuantNodes.factor_node.factor_functions._OPERATOR_REGISTRY`

| 模块 | 原估算 | 修正后 | 说明 |
|------|-------|--------|------|
| `types.py` | ~100行 | ~50行 | 简化dataclass |
| `registry.py` | ~~150行~~ | **0行** | 复用现有注册表 |
| `loader.py` | ~200行 | ~100行 | 配置解析 |
| `executor.py` | ~~300行~~ | ~100行 | 轻量包装 |
| `template.py` | ~100行 | ~50行 | 模板 |
| **总计** | **~850行** | **~300行** | 减少65% |

### 复用现有算子

现有 `factor_functions.py` 已实现 (通过 `@rolling_operator()` 等装饰器):

| 装饰器 | 已注册算子 |
|--------|------------|
| `@rolling_operator()` | rolling_mean, rolling_sum, rolling_std, rolling_max, rolling_min, rolling_median, rolling_skew, rolling_kurt, rolling_argmax, rolling_argmin |
| `@single_section_operator()` | rank, standardizeZScore, winsorize, neutralizeIndustry 等 |
| `@expanding_operator()` | expanding_mean, expanding_std 等 |

配置文件模块只需创建轻量级 YAML 解析层，映射配置参数到现有算子。

---

## 十一、待讨论确认事项

1. **配置格式**: YAML 是否满足需求？需要支持 JSON？
2. **算子覆盖度**: 初始覆盖哪些算子？是否需要全部实现？
3. **自定义算子**: Agent 实现后的注册流程？
4. **测试集成**: 测试发现机制使用 pytest？
5. **版本控制**: 配置文件是否需要版本管理？

---

**文档状态**: 设计完成，等待评审