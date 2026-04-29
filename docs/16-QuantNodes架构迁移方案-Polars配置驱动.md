# QuantNodes 架构迁移方案 - Polars + 配置文件驱动

**版本**: v2.0  
**创建日期**: 2026-04-29  
**状态**: 设计阶段  
**作者**: sn0wfree

---

## 一、变更背景与目的

### 1.1 现有问题陈述

当前 QuantNodes 使用 **QuantStudio 设计模式**，存在以下问题：

| 问题 | 具体表现 | 影响 |
|------|----------|------|
| **依赖复杂** | 依赖 `traits.api` 外部库 | 依赖易失效，维护成本高 |
| **接口复杂** | 算子签名 `(f, idt, iid, x, args)` 难以理解 | 学习成本高，新人上手难 |
| **不支持配置化** | 无法通过 YAML 配置驱动 | Agent 无法自动化 |
| **单点计算** | 基于多进程，非向量化 | 性能不如现代方案 |
| **代码量庞大** | `factor_functions.py` 2400+行 | 维护困难 |

**代码证据**:
```python
# 现有复杂接口
def rolling_mean(f, idt, iid, x, args):
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).mean()...
```

### 1.2 业界趋势

2025年量化框架主流方案：

| 框架 | 核心语言 | 配置化 | 表达式风格 |
|------|----------|--------|------------|
| **factr** | Polars | ✅ | `pipeline.add_factors({'momentum': momentum})` |
| **elvers** | Polars | ✅ | `zscore(ts_rank(close, 30))` |
| **AKQuant** | Polars | ✅ | `Rank(Ts_Mean(Close, 5))` |
| **AlphaPurify** | Polars | ✅ | 40+ 预处理方法 |

**共同特点**:
1. **Polars** - 内存高效，Lazy模式
2. **表达式语法** - 类似自然语言
3. **YAML配置** - 完整支持
4. **纯Python** - 无特殊依赖

### 1.3 变更目的

| 目标 | 说明 |
|------|------|
| **简化接口** | 从 `(f, idt, iid, x, args)` → `ts_mean(col, 20)` |
| **配置驱动** | Agent 可通过 YAML 配置自动化 |
| **零特殊依赖** | 移除 `traits` 依赖 |
| **高性能** | Polars 向量化计算 |
| **代码精简** | 2400+行 → ~500行 |

---

## 二、预期效果

### 2.1 开发体验对比

| 维度 | 变更前 (QuantStudio) | 变更后 (Polars) |
|------|----------------------|----------------|
| **定义因子** | 10行复杂代码 | 1行表达式 |
| **配置文件** | 不支持 | YAML 一行 |
| **依赖** | traits, multiprocessing | polars (标准库) |
| **调试** | 困难 | Polars 错误信息清晰 |
| **Agent使用** | 需编写代码 | 只需配置 |

### 2.2 代码量对比

| 模块 | 变更前 | 变更后 | 变化 |
|------|--------|--------|------|
| `factor_functions.py` | 2400+行 | 保留 (兼容) | - |
| `operators/` | **新增** | ~400行 | 新增 |
| `config/` | **新增** | ~300行 | 新增 |
| **总计** | - | ~700行 | 大幅减少 |

### 2.3 Agent 工作流程对比

**变更前**:
```
Agent 需要:
1. 编写 Python 代码
2. 理解复杂算子接口
3. 手动调用验证
4. 手动运行回测
5. 手动运行测试
```
代码示例:
```python
@rolling_operator()
def rolling_mean(f, idt, iid, x, args):
    Data = pd.DataFrame(_genOperatorData(f, idt, iid, x, args)[0])
    return Data.rolling(**args["OperatorArg"]).mean()...
```

**变更后**:
```
Agent 只需编写配置文件:
```
YAML示例:
```yaml
factors:
  - name: momentum_20d
    expr: ts_mean(close / close.shift(20) - 1, 20)

backtest:
  start_date: "2023-01-01"
  end_date: "2024-12-31"
```

---

## 三、新架构设计

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      QuantNodes v2.0                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │   Agent     │────▶│  Config     │────▶│  Executor  │   │
│  │  (编写YAML) │     │  Loader     │     │            │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│                           │                   │              │
│                           ▼                   ▼              │
│                    ┌─────────────┐     ┌─────────────┐   │
│                    │  Operators │────▶│   Polars    │   │
│                    │  (算子库)   │     │  Engine    │   │
│                    └─────────────┘     └─────────────┘   │
│                           │                               │
│                           ▼                               │
│                    ┌─────────────┐     ┌─────────────┐   │
│                    │   Backtest  │────▶│   Result   │   │
│                    └─────────────┘     └─────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 目录结构

```
QuantNodes/
├── __init__.py                     # 统一导出
│
├── operator_node/                   # 保留: SQL构建
│
├── database_node/                    # 保留: 数据库
│
├── core/                            # 保留: 核心工具
│
├── factor_node/                     # 保留: 兼容现有
│   ├── factor_functions.py          # 保留 (不做修改)
│   ├── factor_operation.py          # 保留 (不做修改)
│   └── ...
│
├── backtest/                        # 保留: 回测引擎
│
├── agent/                           # 保留: Agent
│   └── tools/                      # 保留: 工具
│
├── operators/                      🆕 新: Polars算子
│   ├── __init__.py
│   ├── time_series.py             # ts_mean, ts_std, ts_corr...
│   ├── section.py              # rank, zscore, winsorize...
│   ├── math.py               # add, mul, log, pow...
│   └── composite.py         # weighted_sum, combine...
│
└── config/                         🆕 新: 配置文件驱动
    ├── __init__.py
    ├── loader.py                # YAML解析
    ├── executor.py            # 执行器
    ├── registry.py            # 算子注册表
    ├── types.py               # 类型定义
    └── templates/
        ├── momentum.yaml
        ├── mean_reversion.yaml
        └── empty.yaml
```

---

## 四、核心模块设计

### 4.1 算子接口设计

#### 4.1.1 设计原则

| 原则 | 说明 |
|------|------|
| **简洁** | 1-2个参数完成常见操作 |
| **类型安全** | 使用 Python 类型提示 |
| **链式调用** | 支持 `expr.method().method()` 风格 |
| **Lazy计算** | 支持 Polars Lazy模式 |

#### 4.1.2 接口定义

```python
# operators/time_series.py
from typing import Union
import polars as pl
from polars import Expr

Expr.window = 20      # 默认窗口

class TimeSeriesOperators:
    """时间序列算子"""
    
    @staticmethod
    def ts_mean(expr: Expr, window: int = 20, min_periods: int = None) -> Expr:
        """
        滚动均值
        
        Args:
            expr: 表达式
            window: 窗口大小
            min_periods: 最小观测数
        
        Example:
            >>> ts_mean(pl.col("close"), 20)
        """
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_mean(window, min_periods=min_periods)
    
    @staticmethod
    def ts_std(expr: Expr, window: int = 20, min_periods: int = None) -> Expr:
        """滚动标准差"""
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_std(window, min_periods=min_periods)
    
    @staticmethod
    def ts_max(expr: Expr, window: int = 20, min_periods: int = None) -> Expr:
        """滚动最大值"""
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_max(window, min_periods=min_periods)
    
    @staticmethod
    def ts_min(expr: Expr, window: int = 20, min_periods: int = None) -> Expr:
        """滚动最小值"""
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_min(window, min_periods=min_periods)
    
    @staticmethod
    def ts_corr(expr_a: Expr, expr_b: Expr, window: int = 20) -> Expr:
        """滚动相关系数"""
        return expr_a.rolling_corr(expr_b, window)
    
    @staticmethod
    def ts_rank(expr: Expr, window: int = 20) -> Expr:
        """滚动排名"""
        return expr.rolling_rank(window)
    
    @staticmethod
    def ts_delta(expr: Expr, periods: int = 1) -> Expr:
        """差分"""
        return expr.diff(periods)
    
    @staticmethod
    def ts_pct_change(expr: Expr, periods: int = 1) -> Expr:
        """百分比变化"""
        return expr.pct_change(periods)


class SectionOperators:
    """截面算子"""
    
    @staticmethod
    def rank(expr: Expr) -> Expr:
        """截面排名 (0-1)"""
        return (expr.rank() - 1) / (expr.len() - 1)
    
    @staticmethod
    def zscore(expr: Expr) -> Expr:
        """Z-score标准化"""
        return (expr - expr.mean()) / expr.std()
    
    @staticmethod
    def winsorize(expr: Expr, lower: float = 0.01, upper: float = 0.01) -> Expr:
        """去极值"""
        q_low = expr.quantile(lower)
        q_high = expr.quantile(1 - upper)
        return expr.clip(q_low, q_high)
    
    @staticmethod
    def neutralize(expr: Expr, group: Expr) -> Expr:
        """行业中性的"""
        group_mean = expr.mean().over(group)
        return expr - group_mean
    
    @staticmethod
    def scale(expr: Expr, method: str = "zscore") -> Expr:
        """归一化"""
        if method == "zscore":
            return (expr - expr.mean()) / expr.std()
        elif method == "minmax":
            return (expr - expr.min()) / (expr.max() - expr.min())
        return expr


class MathOperators:
    """数学算子"""
    
    @staticmethod
    def add(expr: Expr, value: Union[float, Expr]) -> Expr:
        """加法"""
        return expr + value
    
    @staticmethod
    def sub(expr: Expr, value: Union[float, Expr]) -> Expr:
        """减法"""
        return expr - value
    
    @staticmethod
    def mul(expr: Expr, value: Union[float, Expr]) -> Expr:
        """乘法"""
        return expr * value
    
    @staticmethod
    def div(expr: Expr, value: Union[float, Expr]) -> Expr:
        """除法"""
        return expr / value
    
    @staticmethod
    def log(expr: Expr) -> Expr:
        """对数"""
        return expr.log()
    
    @staticmethod
    def abs(expr: Expr) -> Expr:
        """绝对值"""
        return expr.abs()
    
    @staticmethod
    def pow(expr: Expr, exponent: float) -> Expr:
        """幂运算"""
        return expr.pow(exponent)
```

#### 4.1.3 统一导出

```python
# operators/__init__.py
from .time_series import TimeSeriesOperators as ts
from .section import SectionOperators as sec
from .math import MathOperators as math

__all__ = ["ts", "sec", "math"]
```

### 4.2 配置加载器设计

#### 4.2.1 类型定义

```python
# config/types.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import date

@dataclass
class FactorConfig:
    """因子配置"""
    name: str
    expr: str                          # 表达式字符串
    description: str = ""

@dataclass
class OperationConfig:
    """运算配置"""
    type: str                         # time_series/section/math
    name: str
    category: str                    # 具体算子
    inputs: List[str]                # 引用因子
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: date
    end_date: date
    initial_cash: float = 1000000
    commission: float = 0.001
    slippage: float = 0.001

@dataclass
class StrategyConfig:
    """策略配置"""
    version: str = "1.0"
    name: str = ""
    description: str = ""
    factors: List[FactorConfig] = field(default_factory=list)
    operations: List[OperationConfig] = field(default_factory=list)
    composite: List[Dict] = field(default_factory=list)
    backtest: Optional[BacktestConfig] = None
    validation: Dict[str, Any] = field(default_factory=dict)
```

#### 4.2.2 配置解析

```python
# config/loader.py
from typing import Dict, List
import yaml
from .types import StrategyConfig, FactorConfig, OperationConfig, BacktestConfig

class ConfigLoader:
    """YAML配置解析器"""
    
    def load(self, path: str) -> StrategyConfig:
        """加载YAML配置文件"""
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse(data)
    
    def _parse(self, data: dict) -> StrategyConfig:
        """解析配置字典"""
        config = StrategyConfig(
            version=data.get("version", "1.0"),
            name=data.get("name", ""),
            description=data.get("description", ""),
        )
        
        factors = []
        for f in data.get("factors", []):
            factors.append(FactorConfig(
                name=f["name"],
                expr=f["expr"],
                description=f.get("description", "")
            ))
        config.factors = factors
        
        operations = []
        for op in data.get("operations", []):
            operations.append(OperationConfig(
                type=op["type"],
                name=op["name"],
                category=op["category"],
                inputs=op["inputs"],
                params=op.get("params", {})
            ))
        config.operations = operations
        
        config.composite = data.get("composite", [])
        
        if "backtest" in data:
            bt = data["backtest"]
            config.backtest = BacktestConfig(
                start_date=self._parse_date(bt["start_date"]),
                end_date=self._parse_date(bt["end_date"]),
                initial_cash=bt.get("initial_cash", 1000000),
                commission=bt.get("commission", 0.001),
                slippage=bt.get("slippage", 0.001)
            )
        
        config.validation = data.get("validation", {})
        
        return config
    
    def _parse_date(self, s: str):
        """解析日期"""
        from datetime import datetime
        return datetime.strptime(s, "%Y-%m-%d").date()
```

#### 4.2.3 表达式解析

```python
# config/expression.py
import re
from typing import Dict, Any
import polars as pl

class ExpressionParser:
    """表达式解析器
    
    将字符串表达式解析为 Polars 表达式
    
    Example:
        expr: "ts_mean(close / close.shift(20) - 1, 20)"
        → pl.col("close") / pl.col("close").shift(20) - 1).rolling_mean(20)
    """
    
    OPERATOR_MAP = {
        "ts_mean": "rolling_mean",
        "ts_std": "rolling_std",
        "ts_max": "rolling_max",
        "ts_min": "rolling_min",
        "ts_corr": "rolling_corr",
        "rank": "rank",
        "zscore": "zscore",
        "winsorize": "winsorize",
        "scale": "scale",
    }
    
    def parse(self, expr: str, context: Dict[str, Any] = None) -> pl.Expr:
        """解析表达式字符串"""
        context = context or {}
        
        # 简单解析: 直接替换
        # TODO: 实现完整解析器
        result = expr
        
        # 处理列引用
        result = re.sub(r'\b(\w+)\b', r'pl.col("\1")', result)
        
        # 处理方法调用
        for op, polars_op in self.OPERATOR_MAP.items():
            result = result.replace(op, polars_op)
        
        return eval(result)
```

### 4.3 执行器设计

```python
# config/executor.py
import polars as pl
from typing import Dict, Any, List
from .loader import ConfigLoader
from .expression import ExpressionParser

class ConfigExecutor:
    """配置执行器"""
    
    def __init__(self):
        self.loader = ConfigLoader()
        self.parser = ExpressionParser()
    
    def run(self, config_path: str, data: pl.LazyFrame) -> Dict[str, pl.LazyFrame]:
        """执行配置"""
        config = self.loader.load(config_path)
        results = {}
        
        # 1. 计算因子
        for factor in config.factors:
            expr = self.parser.parse(factor.expr)
            results[factor.name] = data.select([
                factor.name: expr
            ])
        
        # 2. 执行运算
        for op in config.operations:
            inputs = [results.get(name) for name in op.inputs]
            expr = self._apply_operator(op.category, inputs, op.params)
            results[op.name] = data.select([op.name: expr])
        
        # 3. 计算组合因子
        for composite in config.composite:
            expr = self.parser.parse(composite.get("formula", ""))
            results[composite["name"]] = data.select([
                composite["name"]: expr
            ])
        
        # 4. 运行回测
        if config.backtest:
            results["backtest"] = self._run_backtest(config.backtest, results)
        
        return results
    
    def _apply_operator(self, category: str, inputs, params):
        """应用算子"""
        from operators import ts, sec, math
        
        if category.startswith("ts_"):
            op_func = getattr(ts, category[3:])
            return op_func(inputs[0], **params)
        elif hasattr(sec, category):
            return getattr(sec, category)(inputs[0], **params)
        elif hasattr(math, category):
            return getattr(math, category)(inputs[0], **params)
        
        raise ValueError(f"Unknown operator: {category}")
    
    def _run_backtest(self, config, factors):
        """运行回测"""
        # TODO: 实现回测
        pass
```

---

## 五、配置文件格式

### 5.1 完整示例

```yaml
# strategy_momentum.yaml
version: "1.0"
name: "momentum_20d"
description: "20日动量因子策略"

# 1. 因子定义
factors:
  - name: returns_20d
    expr: "(close / close.shift(20) - 1)"
    description: "20日收益率"

# 2. 因子运算
operations:
  - type: time_series
    name: momentum_ma
    category: ts_mean
    inputs: [returns_20d]
    params:
      window: 20
      min_periods: 10
  
  - type: section
    name: momentum_rank
    category: rank
    inputs: [momentum_ma]

# 3. 组合因子
composite:
  - name: alpha
    formula: "rank(momentum_ma)"
    normalize: true
    winsorize:
      lower: 0.01
      upper: 0.01

# 4. 回测配置
backtest:
  start_date: "2023-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
  commission: 0.001
  slippage: 0.001

# 5. 验证配置
validation:
  run_tests: true
  metrics:
    ic_threshold: 0.02
    max_correlation: 0.7
```

### 5.2 简化版

```yaml
# minimal.yaml
name: "simple_momentum"

factors:
  - name: returns
    expr: "close.pct_change(20)"

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

## 六、Agent 使用流程

### 6.1 完整流程

```
1. Agent 分析用户需求
   → "帮我生成一个20日动量因子策略，回测2023年"

2. Agent 编写配置文件
   → strategy.yaml

3. ConfigLoader 解析配置
   → 生成执行计划

4. ConfigExecutor 自动执行
   → 因子计算 → 运算 → 回测 → 验证

5. 返回结果
   → IC: 0.05, 回测收益: 15%, Sharpe: 1.2
```

### 6.2 Agent Prompt 模板

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

## 七、实现计划

### 7.1 分阶段实施

| 阶段 | 任务 | 代码量 | 依赖 |
|------|------|--------|------|
| **Phase 1** | 创建 operators/ 模块 | ~400行 | polars |
| **Phase 2** | 创建 config/ 基础 | ~300行 | Phase 1 |
| **Phase 3** | 配置文件模板 | ~100行 | Phase 2 |
| **Phase 4** | 集成测试 | ~200行 | Phase 1-3 |

### 7.2 具体任务

#### Phase 1: operators/ 模块 (~400行)

| 任务 | 文件 | 代码量 |
|------|------|--------|
| 1.1 | `operators/__init__.py` | 20行 |
| 1.2 | `operators/time_series.py` | 150行 |
| 1.3 | `operators/section.py` | 100行 |
| 1.4 | `operators/math.py` | 80行 |
| 1.5 | `operators/composite.py` | 50行 |

#### Phase 2: config/ 模块 (~300行)

| 任务 | 文件 | 代码量 |
|------|------|--------|
| 2.1 | `config/__init__.py` | 20行 |
| 2.2 | `config/types.py` | 50行 |
| 2.3 | `config/loader.py` | 80行 |
| 2.4 | `config/expression.py` | 80行 |
| 2.5 | `config/executor.py` | 70行 |

#### Phase 3: 配置模板 (~100行)

| 任务 | 文件 | 代码量 |
|------|------|--------|
| 3.1 | `config/templates/momentum.yaml` | 30行 |
| 3.2 | `config/templates/mean_reversion.yaml` | 30行 |
| 3.3 | `config/templates/empty.yaml` | 20行 |

#### Phase 4: 测试 (~200行)

| 任务 | 文件 | 代码量 |
|------|------|--------|
| 4.1 | `tests/operators/test_time_series.py` | 50行 |
| 4.2 | `tests/operators/test_section.py` | 50行 |
| 4.3 | `tests/config/test_loader.py` | 50行 |
| 4.4 | `tests/config/test_executor.py` | 50行 |

---

## 八、向后兼容性

### 8.1 保留现有代码

| 模块 | 行为 | 说明 |
|------|------|------|
| `factor_functions.py` | **保留** | 不做修改，供现有代码使用 |
| `factor_operation.py` | **保留** | 不做修改，供现有代码使用 |
| `FactorPipeline` | **保留** | 不做修改，供现有代码使用 |

### 8.2 新旧并行

```python
# 新架构 (Agent 使用)
from QuantNodes.config import ConfigLoader
from QuantNodes.operators import ts, sec

# 旧架构 (兼容现有)
from QuantNodes.factor_node.factor_functions import rolling_mean
```

### 8.3 共存目录

```
QuantNodes/
├── operators/              🆕 新: Polars算子
│
├── config/                 🆕 新: 配置驱动
│
└── factor_node/            保留: 兼容现有
    ├── factor_functions.py
    └── factor_operation.py
```

---

## 九、��证��标

### 9.1 代码质量

| 指标 | 目标 |
|------|------|
| **代码量** | < 700行 (新模块) |
| **测试覆盖率** | > 80% |
| **类型提示** | 100% |
| **文档覆盖率** | 100% |

### 9.2 功能验证

| 指标 | 目标 |
|------|------|
| **配置加载** | 支持 YAML |
| **算子覆盖** | 20+ 常用算子 |
| **回测集成** | 可运行 |
| **Agent集成** | 可自动执行 |

### 9.3 性能验证

| 指标 | 目标 |
|------|------|
| **加载时间** | < 1秒 |
| **计算效率** | Polars Lazy 优化 |
| **内存使用** | < 2GB |

---

## 十、风险与缓解

### 10.1 风险识别

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **表达式解析复杂度** | 高 | 分阶段实现，从简单开始 |
| **Polars API 变化** | 中 | 版本固定 |
| **性能问题** | 低 | 使用 Lazy 模式 |
| **现有代码破坏** | 高 | 完全保留，不修改 |

### 10.2 缓解策略

1. **表达式解析**: 先实现简单替换，再逐步完善
2. **版本固定**: 使用 `polars>=0.20,<1.0`
3. **完全保留**: 新旧代码独立共存
4. **测试驱动**: 每阶段100%测试覆盖

---

## 十一、总结

### 11.1 核心变更

| 变更 | 说明 |
|------|------|
| **引入 Polars** | 替代 multiprocessing pandas |
| **配置文件驱动** | Agent 无需编写代码 |
| **简洁算子接口** | `ts_mean(col, 20)` |
| **移除特殊依赖** | 不再依赖 traits |

### 11.2 预期效果

| 效果 | 变更前 | 变更后 |
|------|--------|--------|
| **Agent 负担** | 编写复杂代码 | 编写 YAML |
| **代码量** | 2400+行 | ~700行 |
| **特殊依赖** | traits | 无 |
| **配置支持** | 无 | YAML |

### 11.3 实施顺序

```
Phase 1: operators/ 模块
    ↓
Phase 2: config/ 模块
    ↓
Phase 3: 配置模板
    ↓
Phase 4: 测试集成
```

---

## 十二、待讨论确认

1. **表达式解析器**: 实现方案 (简单替换 vs 完整解析器)?
2. **回测集成**: 是否复用现有 BacktestNode?
3. **时间线**: 期望何时完成?

---

**文档状态**: 设计完成，等待评审