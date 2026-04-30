# Config-driven 回测实现计划

**创建日期**: 2026-04-30  
**分支**: refactor/factor-node-cleanup  
**状态**: 实施中

---

## 一、现状分析

### 1.1 已实现组件

| 组件 | 状态 | 文件 |
|------|------|------|
| ConfigLoader (YAML解析) | ✅ 完整 | `agent/config/loader.py` |
| StrategyConfig 数据类 | ✅ 完整 | `agent/config/types.py` |
| ConfigExecutor.run() (因子计算) | ✅ 工作 | `agent/config/executor.py` |
| BacktestTool (Python代码回测) | ✅ 工作 | `agent/tools/backtest.py` |
| CodeSandbox (代码沙箱) | ✅ 工作 | `ai/sandbox.py` |

### 1.2 未实现/有缺陷的组件

| 组件 | 问题 | 严重度 |
|------|------|--------|
| `_parse_expr()` | 不支持算术表达式、链式方法调用、一元运算符 | 🔴 高 |
| `_execute_plan()` | 丢弃原始列 (date, code, close等) | 🔴 高 |
| `run_backtest()` | 只生成信号，不执行交易 | 🔴 高 |
| `_apply_ts_operator()` | `ts_corr` 是占位符 | 🟡 中 |
| YAML模板 | 使用不支持的表达式语法 | 🟡 中 |
| 算子覆盖度检查 | 不检查 composite 公式 | 🟡 中 |
| 单元测试 | 不存在 | 🟡 中 |

---

## 二、架构设计

### 2.1 核心思路：Config → Code → BacktestTool

**关键决策**：不重写回测引擎，而是利用已验证的 BacktestTool。

```
YAML Config
    ↓
ConfigLoader.load() → StrategyConfig
    ↓
ConfigExecutor.run() → 计算因子/信号
    ↓
ConfigCodeGenerator.generate() → 生成 Python 代码
    ↓
BacktestTool.execute(pipeline_code) → 执行回测
    ↓
返回 BacktestResult
```

**为什么这样做**：
- BacktestTool 已经过测试，Strategy→Risk→Broker 链正确
- 避免重写 Pandas→Polars 转换层
- 复用 CodeSandbox 安全验证
- Agent 可以看到生成的代码，便于调试

### 2.2 新增模块

```
agent/config/
├── executor.py      (修复现有)
├── generator.py     (新增 - 代码生成器)
├── loader.py        (修复现有)
├── types.py         (修复现有)
└── templates/       (修复现有)
```

---

## 三、实施步骤

### Phase 1: 修复基础 (优先级 P0)

#### Step 1.1: 修复 `_parse_expr()` 表达式解析

**问题**: 当前只支持三种简单模式，不支持：
- 算术表达式: `"close / close.shift(20) - 1"`
- 链式方法: `"close.rolling(20).mean()"`
- 一元运算符: `"-rank(close_ma_diff)"`
- 嵌套函数: `"rolling_mean(close, 20) + volume"`

**方案**: 使用递归下降解析器，按运算符优先级解析：

```
表达式 → 项 (('+' | '-') 项)*
项 → 囆子 (('*' | '/') 囆子)*
因子 → 函数调用 | 方法链 | 列引用 | 数字 | '(' 表达式 ')' | '-' 因子
```

**文件**: `agent/config/executor.py`
**行号**: 133-181 (`_parse_expr`)

#### Step 1.2: 修复 `_execute_plan()` 保留原始列

**问题**: `data.select(select_cols)` 只选择计算的因子列，丢弃了 date, code, close 等原始列。

**方案**:
```python
def _execute_plan(self, data, result):
    # 保留原始列 + 新计算的因子列
    all_cols = list(data.columns)  # 原始列
    for name, expr in self._expressions.items():
        all_cols.append(expr.alias(name))
    result.data = data.select(all_cols)
```

**文件**: `agent/config/executor.py`
**行号**: 383-395

#### Step 1.3: 修复 `run_backtest()` 信号生成

**问题**:
1. 使用 `long_threshold`/`short_threshold` 但模板用 `buy_threshold`/`sell_threshold`
2. 不生成 Order/Signal 对象，只生成原始信号列

**方案**:
```python
def run_backtest(self, config, data):
    result = self.run(config, data)
    
    if config.backtest is None:
        return result
    
    bt = config.backtest
    
    # 兼容两种阈值命名
    buy_threshold = bt.signals.get("buy_threshold", 
                    bt.signals.get("long_threshold", 0.05))
    sell_threshold = bt.signals.get("sell_threshold", 
                     bt.signals.get("short_threshold", -0.03))
    
    # 日期筛选...
    # 信号生成...
    # 保存到 result.backtest
    
    return result
```

**文件**: `agent/config/executor.py`
**行号**: 68-131

### Phase 2: 代码生成器 (优先级 P0)

#### Step 2.1: 创建 ConfigCodeGenerator

**新文件**: `agent/config/generator.py`

```python
class ConfigCodeGenerator:
    """将 StrategyConfig 转换为可执行的 Python 代码"""
    
    def __init__(self):
        pass
    
    def generate(self, config: StrategyConfig) -> str:
        """生成完整的回测代码"""
        code_parts = []
        
        # 1. 生成 import 语句
        code_parts.append(self._generate_imports())
        
        # 2. 生成数据加载代码
        code_parts.append(self._generate_data_loading(config))
        
        # 3. 生成因子计算代码
        code_parts.append(self._generate_factor_calculation(config))
        
        # 4. 生成信号生成代码
        code_parts.append(self._generate_signal_generation(config))
        
        # 5. 生成回测执行代码
        code_parts.append(self._generate_backtest_execution(config))
        
        return "\n\n".join(code_parts)
```

**生成的代码示例**:
```python
import pandas as pd
import numpy as np
from QuantNodes.backtest.strategy_node import MAStrategyNode, OrdersResult
from QuantNodes.backtest.broker_node import SimulatedBrokerNode
from QuantNodes.backtest.risk_node import PositionLimitRiskNode

# 1. 加载数据
quote_data = pd.read_csv("data/stock_data.csv")

# 2. 计算因子
quote_data["momentum_20d"] = quote_data["close"] / quote_data["close"].shift(20) - 1
quote_data["momentum_ma"] = quote_data["momentum_20d"].rolling(20).mean()

# 3. 生成信号
quote_data["signal"] = 0
quote_data.loc[quote_data["momentum_ma"] > 0.05, "signal"] = 1
quote_data.loc[quote_data["momentum_ma"] < -0.03, "signal"] = -1

# 4. 创建策略节点
class ConfigStrategy(MAStrategyNode):
    def _generate_signals(self, input_data, **kwargs):
        signals = []
        for _, row in input_data.iterrows():
            if row.get("signal", 0) == 1:
                signals.append(Signal(code=row["code"], signal_type="buy"))
            elif row.get("signal", 0) == -1:
                signals.append(Signal(code=row["code"], signal_type="sell"))
        return signals

strategy = ConfigStrategy(config={})

# 5. 创建 broker
broker = SimulatedBrokerNode(config={
    "cash": 1000000,
    "commission": 0.001
})

# 6. 创建 risk nodes
risk_nodes = [PositionLimitRiskNode(config={"max_position": 10})]
```

### Phase 3: 工具集成 (优先级 P0)

#### Step 3.1: 创建 ConfigBacktestTool

**新文件**: `agent/tools/config_backtest.py`

```python
class ConfigBacktestTool(Tool):
    """配置驱动的回测工具"""
    
    @property
    def name(self) -> str:
        return "config_backtest"
    
    async def execute(
        self,
        config_yaml: str = None,
        config_path: str = None,
        start_date: str = None,
        end_date: str = None,
        initial_cash: float = 1000000,
        **kwargs
    ) -> Dict[str, Any]:
        # 1. 加载配置
        if config_yaml:
            config = yaml.safe_load(config_yaml)
            strategy_config = ConfigLoader()._parse(config)
        elif config_path:
            strategy_config = ConfigLoader().load(config_path)
        else:
            return {"status": "error", "errors": ["Need config_yaml or config_path"]}
        
        # 2. 覆盖日期参数
        if start_date and strategy_config.backtest:
            strategy_config.backtest.start_date = start_date
        if end_date and strategy_config.backtest:
            strategy_config.backtest.end_date = end_date
        if initial_cash and strategy_config.backtest:
            strategy_config.backtest.initial_cash = initial_cash
        
        # 3. 检查覆盖度
        loader = ConfigLoader()
        coverage = loader.check_coverage(strategy_config)
        if not coverage.is_complete:
            return {
                "status": "error",
                "errors": [f"Unresolved operators: {coverage.unresolved}"]
            }
        
        # 4. 生成代码
        generator = ConfigCodeGenerator()
        code = generator.generate(strategy_config)
        
        # 5. 调用 BacktestTool
        backtest_tool = BacktestTool()
        result = await backtest_tool.execute(
            pipeline_code=code,
            start_date=strategy_config.backtest.start_date if strategy_config.backtest else None,
            end_date=strategy_config.backtest.end_date if strategy_config.backtest else None,
            initial_cash=strategy_config.backtest.initial_cash if strategy_config.backtest else 1000000,
            commission=strategy_config.backtest.commission if strategy_config.backtest else 0.001,
        )
        
        # 6. 附加配置信息
        result["config_info"] = {
            "name": strategy_config.name,
            "description": strategy_config.description,
            "factors": len(strategy_config.factors),
            "operations": len(strategy_config.operations),
            "composites": len(strategy_config.composite),
        }
        result["generated_code"] = code
        
        return result
```

#### Step 3.2: 注册工具到 Agent

**文件**: `agent/__init__.py`

在 `_create_tools()` 中添加:
```python
from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
registry.register(ConfigBacktestTool())
```

### Phase 4: 模板修复 (优先级 P1)

#### Step 4.1: 修复 momentum.yaml

```yaml
factors:
  - name: momentum_20d
    expr: "close.pct_change(20)"
    description: "20日动量因子"

operations:
  - type: time_series
    name: momentum_ma
    category: ts_mean
    inputs: [momentum_20d]
    params:
      window: 20

composite:
  - name: alpha_factor
    formula: "rank(momentum_ma)"
```

#### Step 4.2: 修复 mean_reversion.yaml

```yaml
factors:
  - name: close_ma
    expr: "close.rolling_mean(20)"
    description: "20日均线"
  
  - name: close_std
    expr: "close.rolling_std(20)"
    description: "20日标准差"

operations:
  - type: composite
    name: zscore_signal
    category: weighted_sum
    inputs: [close, close_ma, close_std]
    params: {}
```

### Phase 5: 测试 (优先级 P1)

#### Step 5.1: ConfigLoader 单元测试

```python
class TestConfigLoader:
    def test_load_momentum_yaml(self):
        loader = ConfigLoader()
        config = loader.load("templates/momentum.yaml")
        assert config.name == "momentum_alpha_v1"
        assert len(config.factors) > 0
    
    def test_check_coverage(self):
        loader = ConfigLoader()
        config = loader.load("templates/momentum.yaml")
        report = loader.check_coverage(config)
        assert report.is_complete
```

#### Step 5.2: ConfigExecutor 单元测试

```python
class TestConfigExecutor:
    def test_parse_expr_simple_column(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close")
        assert expr is not None
    
    def test_parse_expr_function_call(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("rolling_mean(close, 20)")
        assert expr is not None
    
    def test_parse_expr_arithmetic(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close / close.shift(20) - 1")
        assert expr is not None
```

#### Step 5.3: ConfigBacktestTool 集成测试

```python
class TestConfigBacktestTool:
    def test_execute_with_yaml_string(self):
        async def _test():
            tool = ConfigBacktestTool()
            yaml_str = open("templates/momentum.yaml").read()
            result = await tool.execute(config_yaml=yaml_str)
            assert result["status"] in ("success", "error")
        asyncio.run(_test())
```

---

## 四、文件变更清单

### 新建文件
- `agent/config/generator.py` - 代码生成器
- `agent/tools/config_backtest.py` - 配置回测工具
- `tests/agent/test_config_executor.py` - ConfigExecutor 测试
- `tests/agent/test_config_backtest.py` - ConfigBacktestTool 测试

### 修改文件
- `agent/config/executor.py` - 修复 _parse_expr, _execute_plan, run_backtest
- `agent/config/loader.py` - 完善 check_coverage
- `agent/config/types.py` - 添加 data 字段到 ExecutionResult
- `agent/config/templates/momentum.yaml` - 修复表达式语法
- `agent/config/templates/mean_reversion.yaml` - 修复表达式语法
- `agent/__init__.py` - 注册 ConfigBacktestTool

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| _parse_expr 复杂度高 | 可能引入新 bug | 先写测试，TDD 方式开发 |
| Polars/Pandas 格式不兼容 | 因子计算和回测使用不同格式 | ConfigCodeGenerator 直接生成 Pandas 代码 |
| 生成的代码可能不安全 | 沙箱可能拒绝合法代码 | 使用 BacktestTool 已有的沙箱验证 |
| 模板语法不完整 | 用户可能写不出有效配置 | 提供完整文档和示例 |

---

## 六、验收标准

1. ✅ `momentum.yaml` 可以通过 ConfigBacktestTool 执行回测
2. ✅ `mean_reversion.yaml` 可以通过 ConfigBacktestTool 执行回测
3. ✅ `_parse_expr()` 支持所有模板中的表达式语法
4. ✅ `_execute_plan()` 保留原始列 (date, code, close等)
5. ✅ 所有现有测试继续通过
6. ✅ 新增测试覆盖 ConfigExecutor 和 ConfigBacktestTool
