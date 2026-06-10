# QuantNodes v2.6.0 实际架构基线 / Actual Architecture Baseline (Post-v2.6.0)

> **生成日期 / Generated**: 2026-06-10  
> **基于版本 / Based on commit**: `d388865` (2026-05-15, archive consolidation)  
> **对照 commit**: `048ddc9` (graphify 快照)  
> **定位 / Purpose**: 与代码现状对齐的单一权威参考文档

---

## 0. 阅读须知 / Reading Note

### 0.1 与 README v2.6.0 变更日志的偏差 / Discrepancies with README Changelog

| README 声称 | 实际状态 | 证据 |
|---|---|---|
| "从 Agent 系统转向外部 Agent 方法库" | **未完成**：`agent/` 30+ 文件仍活跃 | `agent/__init__.py:14` 版本号 `2.5.0`，未 bump |
| "移除 Chat UI" | **部分完成**：Vue Chat UI 已归档，但 `agent/web/app.py` 新增 | `cli/__init__.py:656` `quantnodes chat` 仍注册 |
| "移除 Agent LLM" | **未完成**：LLM 客户端、17 工具仍可用 | `agent/providers/quantnodes.py` 仍 import OpenAI |
| "新增 methods/" | ✅ 正确 —— 但与内置 agent 并存 | `agent/__init__.py:80-93` 工具注册未删除 |
| "新增 prompts/" | ✅ 正确 —— 但仅 10 个提示词，只读 | `prompts/__init__.py:8-19` |
| "API 重构…移除 chat 路由" | ✅ 正确 —— `/api/chat` 路由已删除 | `api/routers/` 无 `chat.py` |
| pyproject.toml 版本 `2.6.0` | **仍是 `0.4.1`** | `pyproject.toml:7` |
| `quantnodes chat` 已移除 | **仍可用** | `cli/enhanced.py:73` import `Agent` |

### 0.2 与既有文档的关系 / Relationship to Existing Docs

| 文档 | 定位 | 与本文档关系 |
|---|---|---|
| `docs/04-架构设计.md` (550 行) | ≤v2.5 设计意图与历史决策 | 参考性，不反映 v2.6 现状 |
| `docs/Architecture-Analysis-2026-05-13.md` (210 行) | graphify 快照 (2026-05-13) | 分析框架参考，数据点已过期 |
| `docs/ARCHITECTURE_CHANGE.md` (662 行) | v2.6 Phase 1-6 实施记录 | 记录了重构意图但与代码不完全对应 |
| **本文档** | **v2.6.0 后代码现状** | **权威参考** |

---

## 1. 顶层架构 / Top-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 11: 前端展示 / Frontend Display                              │
│  Vue 3 + Ant Design Vue 4 · Dashboard / Portfolios / Status        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 10: 外部接口 / External Interface                            │
│  api/ (FastAPI) · methods/ (纯方法库) · prompts/ (提示词库)         │
│  CLI: quantnodes init / run / chat / version                        │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 9: 研究 / Research                                           │
│  Wiki 因子库 · MCTS 自动挖因子 · 研报复现                           │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 8: 智能层 / Intelligence                                     │
│  agent/ (LLM Agent, 17 工具) · ai/ (LLM 客户端, CodeSandbox,       │
│  StrategyGenerator, PipelineOptimizer)                             │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 7: 监控 / Monitoring                                         │
│  监控 · 调度 · 版本管理 · 漂移检测                                  │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 6: 辅助 / Auxiliary                                          │
│  cache_node/ · symbolic/ · ui_node/ · conf_node/                   │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 5: 回测 / Backtest                                           │
│  Strategy → Risk → Broker → Statistics                              │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 4: 转换 / Transform                                          │
│  OperatorNode · ChainOperator · SQLBuilder · TableQueryNode         │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 3: 算子 / Operators                                          │
│  factor_functions (装饰器注册表) + operators/ (Polars 代理 + Custom)│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: 数据源 / Data Sources                                      │
│  SQLiteNode · DuckDBNode · MySQLNode · ClickHouseNode · CSV · Parquet│
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: 核心原语 / Core Primitives                                 │
│  BaseNode · Pipeline · Parallel · Join · IfNode · MapNode · WhileNode│
│  Expression DSL · Cond · AST Parser · Config · Serialization        │
└─────────────────────────────────────────────────────────────────────┘
```

**关键事实 / Key Fact**: `methods/` + `prompts/` 与内置 `agent/` **并存**，不是替代关系。外部 Agent 和内置 LLM Agent 均可通过相同基础设施工作。

---

## 2. 核心抽象 / Core Abstractions

### 2.1 BaseNode 契约 / BaseNode Contract

**定义**: `QuantNodes/core/node.py:119`

```python
class BaseNode[T, R](ABC):
    """万物皆 Node —— 所有处理单元的统一基类"""
```

| 方法 | 类型 | 说明 |
|------|------|------|
| `_execute(input_data, **kwargs) -> R` | **必实现** (abstract) | 子类核心执行逻辑 |
| `_from_dict_impl(data) -> BaseNode` | **必实现** (abstract) | 反序列化 |
| `execute()` | 公有 | hooks → validate → `_execute` → validate → stats → after_execute |
| `__call__(data)` | 语法糖 | `return self.execute(data)` |
| `__rshift__(other)` | 管道组合 | `Pipeline(self, other)` |
| `before_execute()` / `after_execute()` | 可选 hook | 执行前后生命周期 |
| `validate()` | 可选 hook | 输入验证 |
| `_get_serializable_fields()` | 可选覆盖 | 复合节点额外字段 |

**类级开关** (`core/node.py:145-148`):
- `_enable_validation = True` — 执行前后校验
- `_enable_stats = True` — 计数器 + 耗时统计
- `_enable_cache = False` — 结果缓存
- `_enable_hooks = True` — 生命周期钩子

**注册装饰器** `@register_node` (`core/node.py:37`): 将子类注册到 `_NODE_CLASSES` 用于反序列化。

### 2.2 Pipeline / Parallel / Join + 控制流

**定义**: `QuantNodes/core/pipeline.py`

| 节点 | 行 | 模式 | 说明 |
|------|---|------|------|
| `Pipeline` | `:38` | **顺序执行** | `A.execute() → B.execute() → C.execute()` |
| `Parallel` | `:119` | **并行分叉** | 同一输入 → `Dict[name, node]` 并行 → 返回 `Dict[name, result]` |
| `Join` | `:232` | **聚合** | `Dict[str, Any] → Callable → result` |

组合语法：
- **顺序**: `Pipeline(A, B, C)` 或 `A >> B >> C`
- **并行 + 聚合**: `Parallel(branches={...}) >> Join(lambda mom, vol, ...: ...)`
- **合并并行分支**: `Parallel({a: ...}) | Parallel({b: ...})` (`core/pipeline.py:192`)

所有组合节点本身都是 `BaseNode` 子类，完全可嵌套。

**控制流节点** (`QuantNodes/core/control.py`):

| 节点 | 行 | 说明 |
|------|---|------|
| `IfNode` | `:47` | 条件分支：`condition` → `true_branch`，可选 `false_branch` |
| `MapNode[I, O]` | `:146` | 按组迭代：按字符串列 / Expression / callable 分组，每组执行 `node`，默认并行 |
| `WhileNode` | `:299` | 循环：`while condition: body`，上限 `max_iterations=1000` |
| `_wrap_condition` | `:31` | 适配器：`ExpressionBuilder` / `Expression` / `str` / `Callable` → `Expression` |

条件 DSL 示例:
```python
from QuantNodes.core.cond_builder import Cond
Cond('close') > 50                    # ExpressionBuilder
"df['close'] > 50"                   # AST 解析 (ast_parser.py:34)
lambda x: x['close'] > 50            # legacy (不可序列化)
```

### 2.3 Expression DSL

**定义**: `QuantNodes/core/expression.py`

运行时安全表达式树，核心类型:

| 表达式 | 行 | 说明 |
|------|---|------|
| `Expression` (ABC) | `:50` | 基类：`evaluate(data)` + `serialize/deserialize` + 运算符重载 |
| `InputExpr` | `:314` | `input_data` 本身 |
| `ConstantExpr` | `:331` | 字面量 |
| `VariableExpr` | `:351` | `input_data[name]`（dict 键或属性） |
| `SubscriptExpr` | `:401` | `expr[key]` |
| `MethodCallExpr` | `:436` | `expr.method(*args, **kwargs)` |
| `BinaryOpExpr` | `:488` | `+ - * / // % **` |
| `ComparisonExpr` | `:568` | `> >= < <= == !=` |
| `LogicalOpExpr` | `:611` | `and or not` |
| `LambdaExpression` | `:649` | backward-compat callable 包装（**不可序列化**） |
| `ExpressionBuilder` (DSL) | `:679` | `Cond(...)` 入口，链式调用代理到底层 `Expression` |

**安全策略**:
- `ALLOWED_AST_NODES` 白名单 (`expression.py:29`)
- `FORBIDDEN_METHODS` 黑名单 (`expression.py:39`)：`eval`, `exec`, `__import__`, `open`, ...
- `__bool__` (`expression.py:292, 812`) 故意抛异常防止隐式布尔转换

**AST 解析器** (`QuantNodes/core/ast_parser.py`):

| 函数 | 行 | 说明 |
|------|---|------|
| `parse_expression(expr_str)` | `:34` | `ast.parse("eval")` → 白名单检查 → `Expression` |
| `_validate_ast(node)` | `:60` | 递归白名单校验 |
| `_ast_to_expr(node)` | `:69` | 按 AST 节点类型分派到对应 `*Expr` 构造器 |

裸名特殊处理 (`ast_parser.py:79`): `df`, `input`, `data`, `x`, `result` → `InputExpr`；其他 `ast.Name` → `VariableExpr`。

### 2.4 算子注册表 / Operator Registry

**定义**: `QuantNodes/factor_node/factor_functions/_helpers.py`

| 符号 | 行 | 说明 |
|------|---|------|
| `_OPERATOR_REGISTRY` | `:24` | 单例: `Dict[category, Dict[name, info]]` |
| `OperatorCategory` 枚举 | `:33` | `POINT / TIME / SECTION / MULTI_SECTION / TALIB` |
| `register_operator(category, name=None)` | `:45` | 装饰器工厂，自动注册到 registry |
| `_ensure_expr(f)` | `:67` | `str → pl.col`，标量 → `pl.lit`，`Expr → Expr` |
| 公开 API | `__init__.py:104-167` | `list_operators()` / `get_operator()` / `operator_info()` / `generate_documentation()` |

**级联查询** (`get_operator(name, category=None)`, `__init__.py:131`):
1. 先查 `_CustomOperatorRegistry`（运行时自定义算子）
2. 回退到 `_OPERATOR_REGISTRY`（内置算子）

**自定义算子三种风格** (`QuantNodes/operators/custom.py`):

```python
# 装饰器风格
@CustomOperator.time("my_decay")
def my_decay(f, halflife=10): ...

# Builder 链式风格
my_ewm_30 = (CustomOperator.time("my_ewm_30")
    .param("span", int, 30, "窗口大小")
    .execute(lambda s, span: s.ewm_mean(span=span))
    .register())

# 模板工厂风格
my_ma20 = CustomOperator.time_from("ma20", template="rolling_mean", window=20).register()
```

独立注册表: `QuantNodes/operators/registry.py:15` — `_CustomOperatorRegistry`（隔离于内置 registry）。

### 2.5 Config-Driven 回测流程

```
YAML 配置
  ↓
YamlConfigNode.execute()          (conf_node/yaml_config.py:13)
  ↓ 返回 dict
StrategyConfig(**dict)            (agent/config/types.py:23)
  ↓
ConfigExecutor.run_backtest()     (agent/config/executor.py:374)
  ↓ 内部: run() → universe filter → risk/broker → backtest
ConfigBacktestRunner.run()        (backtest/config_runner.py:27)
  ↓ Polars → Pandas，列名规范化
ConfigStrategyNode → OrdersResult
  ↓
RiskNode → RiskResult
  ↓
ExecutionBrokerNode → TradeResult
  ↓
BacktestResult (含 statistics: Sharpe/Sortino/Calmar/MaxDD/WinRate/...)
```

---

## 3. 模块地图 / Module Map

| 包 | 路径 | 职责 | 关键类 | 状态 |
|---|---|---|---|---|
| **core** | `QuantNodes/core/` | 节点抽象 + 表达式 + 管道 | `BaseNode`, `Pipeline`, `Expression`, `Cond` | ✅ active |
| **database_node** | `QuantNodes/database_node/` | 6 个数据库适配器 | `BaseDBNode`, `SQLiteNode`, `DuckDBNode`, `MySQLNode`, `ClickHouseNode`, `CSVNode`, `ParquetNode` | ✅ active |
| **factor_node** | `QuantNodes/factor_node/` | 因子定义、存储、计算 | `Factor`, `FactorDB`, `PointOperation`, `TimeOperation`, `SectionOperation`, `PanelOperation`, `FactorTable` | ✅ active |
| **operators** | `QuantNodes/operators/` | Polars 代理层 + 自定义算子 API | `TimeSeriesOperators`, `SectionOperators`, `CustomOperator`, `CustomOperatorBuilder` | ✅ active |
| **factor_functions** | `QuantNodes/factor_node/factor_functions/` | 装饰器注册表 + 214 内置算子 | `OperatorCategory`, `register_operator`, `_ensure_expr`, `list_operators`, `get_operator` | ✅ active |
| **operator_node** | `QuantNodes/operator_node/` | 链式数据转换 + SQL 构建 | `OperatorNode`, `ChainOperator`, `SQLBuilderNode`, `TableQueryNode`, `TransformNode` | ✅ active |
| **backtest** | `QuantNodes/backtest/` | 策略→风控→经纪→统计 | `BacktestNode`, `BacktestResult`, `BacktestPipeline`, `StrategyNode`, `RiskNode`, `BrokerNode`, `SimulatedBrokerNode`, `ExecutionBrokerNode`, `ConfigBacktestRunner` | ✅ active |
| **conf_node** | `QuantNodes/conf_node/` | 配置加载 (YAML/INI/JSON/ENV) | `ConfigNode`, `YamlConfigNode`, `IniConfigNode`, `JSONConfigNode`, `EnvConfigNode` | ✅ active |
| **agent** | `QuantNodes/agent/` | LLM Agent (17 工具, 16 子包) | `Agent`, `AgentLoop`, `AgentRunner`, `MemoryStore`, `DreamEngine`, `ToolRegistry`, `SkillRegistry`, `PermissionService` | ✅ active (见 §7) |
| **ai** | `QuantNodes/ai/` | LLM 客户端 + CodeSandbox + 策略生成 | `LLMClientBase`, `OpenAIClient`, `CodeSandbox`, `StrategyGenerator`, `PipelineOptimizer`, `PromptLibrary` | ✅ active |
| **research** | `QuantNodes/research/` | Wiki 因子库 + 自动挖因子 | `WikiFactorProxy`, `FactorMiner`, `AutoResearcher`, `MCTSSearch`, `ResearchReportReproducer` | ✅ active |
| **methods** | `QuantNodes/methods/` | 纯方法库 (外部 Agent API) | `run_backtest`, `validate_code`, `execute_code`, `validate_pipeline`, `analyze_factor`, `query_wiki`, `FileOperations`, `CodeSearch`, `GitOperations` | ✅ active (见 §8) |
| **prompts** | `QuantNodes/prompts/` | LLM 提示词库 (10 个) | `MOMENTUM_PROMPT`, `MEAN_REVERSION_PROMPT`, `TREND_FOLLOWING_PROMPT`, ... | ✅ active (见 §8) |
| **api** | `api/` | FastAPI REST API (10 router) | `main.py`, `deps.py` (X-API-Key), 10 个 router | ✅ active |
| **cache_node** | `QuantNodes/cache_node/` | Parquet 缓存层 | `MarketDataCacheNode`, `ParquetCacheStore`, `CacheMetadata` | ✅ active |
| **symbolic** | `QuantNodes/symbolic/` | 多方言 SQL AST + 编译 | `SQLExpression`, `SQLCompiler`, `SQLExecutor`, `SQLOptimizer`, `TechnicalFunctions` | ✅ active (孤立) |
| **ui_node** | `QuantNodes/ui_node/` | 展示节点 (表格/图表/指标) | `DisplayNode`, `TableDisplayNode`, `ChartDisplayNode`, `MetricDisplayNode` | ⚠️ active (仅测试使用) |
| **monitor** | `QuantNodes/monitor/` | 监控 + 调度 + 版本 | `MetricsCollector`, `DriftDetector`, `StrategyScheduler`, `StrategyRunner`, `VersionManager`, 4 Repository | ✅ active |
| **cli** | `QuantNodes/cli/` | CLI 入口 | `main()` → `init / run / chat / version / help` | ✅ active |
| **deprecated** | `QuantNodes/deprecated/` | 早期遗留代码 | `TableNode`, `TableOperator`, `brinson`, `factor_tools` | ❌ deprecated (无 import) |
| **archive** | `archive/` | v2.6 重构归档 | agent core/providers/session/skills, frontend Chat UI, 28 历史文档 | ❌ frozen (不可构建) |

---

## 4. 表达式系统双轨 / Dual Expression System

| 轨 | 路径 | 用途 | 特点 |
|---|---|---|---|
| **运行时 DSL** | `core/expression.py` + `core/ast_parser.py` | 节点内部条件/表达式评估 | `Expression` 树 + `ExpressionBuilder` (Cond DSL) + AST 安全白名单 |
| **SQL AST** | `symbolic/` | 多方言 SQL 编译 | `SQLExpression` 树 + `SQLCompiler` + `SQLOptimizer` + 方言适配 (ClickHouse/DuckDB/MySQL/PostgreSQL) |

**关系**：两套系统独立自洽。`symbolic/` 是纯库（无外部 QuantNodes 依赖），仅用于测试。运行时 DSL 在节点执行中使用。

---

## 5. 算子系统 / Operator System

### 5.1 注册表结构

```
_OPERATOR_REGISTRY (factor_functions/_helpers.py:24)
├── POINT:       math_ops.py       → 40 个 (abs, log, sign, isnull, ...)
├── TIME:        time_ops.py       → 34 个 (rolling_mean, rolling_std, ewm, diff, ...)
├── SECTION:     section_ops.py    → 21 个 (rank, zscore, winsorize, ...)
├── MULTI_SECTION: composite_ops.py → 9 个 (aggregate, disaggregate, aggr_sum, ...)
└── TALIB:       talib_ops.py      → ~110 个 (talib_rsi, talib_sma, talib_macd_line, ...)
```

**总计 ~214 个内置算子** + 无限自定义。

### 5.2 Polars 代理层

`QuantNodes/operators/` 提供面向 Polars 的薄代理:

| 代理类 | 路径 | 方法数 |
|------|---|---|
| `TimeSeriesOperators` | `operators/time_series.py:63` | 16 |
| `SectionOperators` | `operators/section.py:50` | 11 |
| `MathOperators` | `operators/math.py:56` | 19 |
| `CompositeOperators` | `operators/composite.py:38` | 12 |
| `TaLibOperators` | `operators/talib.py` | ~110 |

代理层导入 `factor_functions` 模块并委托执行，不是重复实现。

### 5.3 自定义算子 API

```python
from QuantNodes.operators import CustomOperator

# 装饰器风格（运行时注册）
@CustomOperator.point("my_double")
def my_double(f, multiplier=2.0):
    return f * multiplier

# Builder 链式风格
my_ewm_30 = (CustomOperator.time("my_ewm_30")
    .param("span", int, 30, "窗口大小")
    .execute(lambda s, span: s.ewm_mean(span=span))
    .register())

# 模板工厂风格
my_ma20 = CustomOperator.time_from("ma20", "rolling_mean", window=20).register()

# 使用
result = my_double(pl.col("x"), k=3.0)  # 直接调用返回 pl.Expr
```

级联查询：`get_operator("my_double")` → 先查 `_CustomOperatorRegistry` → fallback `_OPERATOR_REGISTRY`。

---

## 6. 回测引擎 / Backtest Engine

### 6.1 管线

```
StrategyNode → OrdersResult
    ↓
RiskNode → RiskResult (passed/rejected/adjusted orders)
    ↓
BrokerNode → TradeResult (fills, cash, positions)
    ↓
BacktestResult + Statistics (Sharpe, Sortino, Calmar, MaxDD, WinRate, ProfitFactor, Calmar)
```

### 6.2 内置节点

**策略 (Strategy)** (`backtest/strategy_node.py`):
- `StrategyNode` — 抽象基类，子类实现 `_generate_signals()`
- `MAStrategyNode` — 均线交叉 (`backtest/strategy_node.py:183`)
- `MomentumStrategyNode` — 动量突破 (`backtest/strategy_node.py:229`)

**风控 (Risk)** (`backtest/risk_node.py`):
- `PositionLimitRiskNode` — 最大持仓数限制
- `StopLossRiskNode` — 止损
- `CashRiskNode` — 现金余额限制
- `CompositeRiskNode` — 组合风控 (`mode='all'|'any'`)

**经纪商 (Broker)** (`backtest/broker_node.py`):
- `SimulatedBrokerNode` — 基础撮合
- `ExecutionBrokerNode` — 带滑点 (默认 0.0005)，numpy 向量化执行

**Config 驱动回测** (`backtest/config_runner.py`):
- `ConfigStrategyNode` — 从 signal 列 (1/−1/0) 生成信号
- `ConfigBacktestRunner` — 胶水层，串联 ConfigExecutor → Polars→Pandas → Strategy → Risk → Broker → Statistics

### 6.3 统计指标

| 指标 | 计算位置 |
|------|---------|
| 总收益率 / 年化收益率 | `config_runner.py:142-155` |
| Sharpe Ratio (252 天) | `:157-160` |
| Sortino Ratio | `:162-167` |
| 最大回撤 | `:169-173` |
| Calmar Ratio | `:175-178` |
| 胜率 | `:180-185` |
| 盈亏比 | `:187-192` |
| 利润因子 | `:194-200` |
| 每日收益曲线 | `_build_equity_curve()` `:232` |

---

## 7. 内置 Agent 系统 / Built-in Agent System

> ⚠️ **注意**: README 声称 "移除 Agent LLM"，但代码中 `agent/` 仍完全活跃。本文档基于代码事实。

### 7.1 架构

```
agent/__init__.py (Agent 入口)
├── agent/core/         # AgentLoop, AgentRunner, ContextBuilder
├── agent/bus/          # MessageBus, InboundMessage, OutboundMessage
├── agent/session/      # Session, SessionManager
├── agent/providers/    # LLM 客户端: OpenAI/Azure/自定义 Provider
├── agent/tools/        # 17 个工具: Sandbox, Pipeline, Strategy, Backtest, Factor, Wiki, FileOps, CodeSearch, Git, WebFetch, WebSearch, Task, Monitor, Schedule, Version, Echo, ConfigBacktest
├── agent/skills/       # Skill 系统: SkillRegistry, SkillToolBridge, SkillLoader
├── agent/agents/       # AgentDefinition, AgentManager (多 Agent 支持)
├── agent/permission/   # PermissionService, PermissionRule
├── agent/config/       # StrategyConfig, FactorConfig, OperationConfig, ConfigExecutor
├── agent/templates/    # 模板
├── agent/web/          # FastAPI 封装 (agent/web/app.py)
├── agent/cli/          # CLI 辅助
└── agent/utils/        # 工具函数
```

### 7.2 工具清单 (17 个)

| 工具 | 类 | 路径 |
|------|---|------|
| Echo | `EchoTool` | `agent/tools/echo.py` |
| Sandbox | `SandboxTool` | `agent/tools/sandbox.py` |
| Pipeline | `PipelineTool` | `agent/tools/pipeline.py` |
| Strategy | `StrategyTool` | `agent/tools/strategy.py` |
| Backtest | `BacktestTool` | `agent/tools/backtest.py` |
| Factor | `FactorTool` | `agent/tools/factor.py` |
| ConfigBacktest | `ConfigBacktestTool` | `agent/tools/config_backtest.py` |
| Wiki | `WikiTool` | `agent/tools/wiki.py` |
| FileOps | `FileOpsTool` | `agent/tools/file_ops.py` |
| CodeSearch | `CodeSearchTool` | `agent/tools/code_search.py` |
| GitOps | `GitOpsTool` | `agent/tools/git_ops.py` |
| WebFetch | `WebFetchTool` | `agent/tools/web_fetch.py` |
| WebSearch | `WebSearchTool` | `agent/tools/web_search.py` |
| Task | `TaskTool` | `agent/tools/task.py` |
| Monitor | `MonitorTool` | `monitor/agent_tools/monitor_tool.py` |
| Schedule | `ScheduleTool` | `monitor/agent_tools/schedule_tool.py` |
| Version | `VersionTool` | `monitor/agent_tools/version_tool.py` |

### 7.3 LLM Provider

- `LLMProvider` (基类, `agent/providers/base.py`)
- `QuantNodesLLMProvider` (默认, `agent/providers/quantnodes.py`) — 基于 `ai.llm.base.LLMClientBase`
- `ProviderRegistry` + `ProviderConfig` (`agent/providers/registry.py:16, 29`)
- 速率限制: `TokenBucket` / `AsyncTokenBucket` (`agent/providers/rate_limiter.py:15`)

### 7.4 CLI 可用性

```bash
quantnodes chat          # 仍可用 —— cli/__init__.py:656
quantnodes chat --help
# 内部: cli/enhanced.py:73 → from QuantNodes.agent import Agent
#       agent.chat(message) → 流式 LLM 响应
```

---

## 8. 外部 Agent 接入 / External Agent Integration (v2.6 新增)

### 8.1 方法库 / Methods (`QuantNodes/methods/`)

| 方法 | 路径 | 说明 | 返回类型 |
|------|---|------|---------|
| `run_backtest()` | `methods/backtest.py:28` | 执行回测 | `BacktestResult` |
| `validate_code()` | `methods/sandbox.py:31` | AST 安全校验 | `ValidationResult` |
| `execute_code()` | `methods/sandbox.py:64` | 沙箱执行 Python | `ExecutionResult` |
| `validate_pipeline()` | `methods/pipeline.py:27` | Pipeline AST 检查 | `PipelineValidationResult` |
| `analyze_factor()` | `methods/factor.py:24` | IC/换手率分析 | `FactorAnalysisResult` |
| `query_wiki()` | `methods/wiki.py:212` | Wiki 搜索/查询 | `WikiResult` |
| `FileOperations` 类 | `methods/file_ops.py:20` | 文件 CRUD | `FileOperationResult` |
| `CodeSearch` 类 | `methods/code_search.py:30` | 代码搜索 | `SearchResult` |
| `GitOperations` 类 | `methods/git_ops.py:21` | Git 操作 | `GitOperationResult` |

### 8.2 提示词库 / Prompts (`QuantNodes/prompts/`)

| 类型 | 提示词 | 路径 |
|------|------|------|
| **策略** | `MOMENTUM_PROMPT` | `prompts/strategy/momentum.py:154` |
| | `MEAN_REVERSION_PROMPT` | `prompts/strategy/mean_reversion.py:107` |
| | `TREND_FOLLOWING_PROMPT` | `prompts/strategy/trend_following.py:96` |
| | `PAIRS_TRADING_PROMPT` | `prompts/strategy/pairs_trading.py:107` |
| | `MARKET_NEUTRAL_PROMPT` | `prompts/strategy/market_neutral.py:96` |
| **回测** | `STANDARD_BACKTEST_PROMPT` | `prompts/backtest/standard.py:73` |
| | `FACTOR_BACKTEST_PROMPT` | `prompts/backtest/factor_based.py:74` |
| **因子** | `IC_ANALYSIS_PROMPT` | `prompts/factor/ic_analysis.py:79` |
| | `CORRELATION_PROMPT` | `prompts/factor/correlation.py:55` |
| | `GROUP_BACKTEST_PROMPT` | `prompts/factor/group_backtest.py:62` |

### 8.3 API 路由 / API Routers (10 个)

| 路由 | 前缀 | 主要端点 |
|------|---|------|
| `stats.py` | `/api/stats` | `GET ""`, `GET /activity` |
| `wiki.py` | `/api/wiki` | CRUD `/factors`, `/strategies`, `GET /search`, `GET /status` |
| `backtest.py` | `/api/backtest` | `POST /run`, `GET /history`, `GET /templates`, `GET /{id}` |
| `factor.py` | `/api/factor` | `POST /analyze`, `GET /{name}/metrics` |
| `skill.py` | `/api/skills` | `GET /`, `GET /categories/list`, `POST /{name}/execute` |
| `dream.py` | `/api/dreams` | `GET/POST /`, `GET /stats` |
| `strategy.py` | `/api/strategy` | `POST /validate`, `POST /parse`, `POST /strategies` |
| `settings.py` | `/api/settings` | CRUD settings, API keys, providers, models |
| `prompts.py` | `/api` | `GET /prompts/strategy[/{type}]`, `/backtest[/{type}]`, `/factor[/{type}]` |
| `code.py` | `/api` | `POST /code/validate`, `POST /code/execute`, `POST /pipeline/validate` |

根: `GET /`, `GET /health`, `GET /api/health` (`api/main.py:53-63`)。

### 8.4 认证 / Authentication

- 两种 header: `X-API-Key: <key>` 或 `Authorization: Bearer <key>` (`api/deps.py:38-44`)
- 硬编码 dev key (⚠️ 明文): `qn_live_xxx...` → "opencode", `qn_live_yyy...` → "openclaw" (`api/deps.py:14-17`)
- `settings.DEBUG=True` 时自动放行 (`api/deps.py:47-48`)

### 8.5 与内置 Agent 的边界

```
外部 Agent (独立进程)
    ↓ HTTP
api/ (FastAPI) → methods/ → core / factor_node / backtest / symbolic
    ↓ HTTP
内置 Agent (quantnodes chat)
    ↓ 导入
agent/ → ai/ → methods/ (部分复用)
```

- 外部 Agent 通过 REST API 访问 `methods/` + `prompts/`
- 内置 LLM Agent 通过 `Agent` 类直接调用 `tools/` (17 个)
- 部分方法有重叠 (如 `run_backtest` 在 methods/ 和 agent/tools/backtest.py 都有)
- 内置 Agent 是 v2.6+ 的遗留，**未被移除**

---

## 9. 监控 / 调度 / 版本 / Monitoring

### 9.1 架构

```
monitor/
├── monitor/
│   ├── collector.py   # MetricsCollector — 指标采集
│   ├── alerter.py     # Alerter — 告警
│   ├── dashboard.py   # MonitorDashboard — 仪表盘
│   └── drift.py       # DriftDetector — 漂移检测 (KS, Sharpe drop, drawdown)
├── scheduler/
│   ├── scheduler.py   # StrategyScheduler — APScheduler cron
│   └── runner.py      # StrategyRunner — 调度执行
├── storage/
│   ├── models.py      # StrategyRun, PerformanceSnapshot, DriftAlert, StrategyVersion
│   └── repository.py  # DatabaseManager, StrategyRunRepository, PerformanceRepository, DriftAlertRepository, VersionRepository
├── version/
│   ├── version_manager.py  # VersionManager — Git 版本管理
│   └── diff.py             # ConfigDiffer — YAML diff
└── agent_tools/
    ├── monitor_tool.py  # MonitorTool
    ├── schedule_tool.py # ScheduleTool
    └── version_tool.py  # VersionTool
```

### 9.2 数据模型

| 模型 | 路径 | 字段 |
|------|---|------|
| `StrategyRun` | `monitor/storage/models.py:12` | 策略运行记录 |
| `PerformanceSnapshot` | `:28` | 性能快照 |
| `DriftAlert` | : | 漂移告警 |
| `StrategyVersion` | : | 策略版本 |

---

## 10. 前端 / Frontend

- **框架**: Vue 3 + Ant Design Vue 4.x
- **视图**: `frontend/src/views/` — Dashboard / Portfolios / Status
- **与后端边界**: `frontend/src/api/` — 对接 `api/` REST 端点
- **Settings 仍含 Agent 配置 Tab**: `frontend/src/views/Settings/index.vue:80-163, 485` — LLM provider/model/api_key/api_base

---

## 11. archive/ 与 deprecated/ 状态

### 11.1 archive/ (一次性冻结)

- **来源**: commit `d388865` (2026-05-15) 一次性归档
- **内容**: 81 文件 — agent core/providers/session/skills (4 子包)、Vue Chat UI (18 组件)、api/archive/agent.py、28 个历史设计文档
- **状态**: **不可构建** — 无顶层 `__init__.py`，无 build 配置引用
- **用途**: 仅参考

### 11.2 deprecated/ (早期遗留)

- 5 个文件: `TableNode.py`, `TableOperator.py`, `brinson.py`, `factor_tools.py`, `basic_init.py`
- 无 `__init__.py`，无任何模块导入它
- 比 archive/ 更早的遗留物，与 v2.6 无关
- **可安全删除**

---

## 12. 已观察问题 / Observed Issues

### 12.1 P0: ConfigExecutor God Class

**定义**: `QuantNodes/agent/config/executor.py:242` (942 行)

承担 8 项职责（违反 SRP）:

| 职责 | 方法/行 |
|------|---------|
| 表达式解析 | `_parse_expr:478`, `_parse_func_args:492`, `_parse_value:537` |
| 自定义算子加载 | `_load_custom_operators:258` |
| 因子表达式构建 | `run:319` (前半) |
| 算子应用 | `_apply_operator:569`, `_get_op:618`, `_apply_ts/sec/math/composite/talib_operator:623-870` |
| Universe 过滤 | `_resolve_universe:831` |
| 回测运行 | `run_backtest:374` |
| 风控节点管理 | `run_backtest:374` 内 |
| 经纪商节点创建 | `run_backtest:374` 内 |

**P0 重构未开始** — `git log --since="2026-05-13" -- executor.py` 为空。

**拆分建议** (来自 `Architecture-Analysis-2026-05-13.md`):
- `ExpressionEngine` — 解析
- `FactorComputer` — 因子计算
- `BacktestOrchestrator` — 回测编排

### 12.2 未提交修改

| 文件 | 状态 |
|------|------|
| `QuantNodes/backtest/broker_node.py` | Modified |
| `QuantNodes/backtest/config_runner.py` | Modified |
| `QuantNodes/operators/custom.py` | Modified |

### 12.3 孤立模块

| 模块 | 状态 | 说明 |
|------|------|------|
| `symbolic/` | 自洽但孤立 | 无业务模块导入它，仅 tests 使用 |
| `ui_node/` | 仅测试使用 | 无生产代码引用 |
| `deprecated/` | 死代码 | 无任何导入 |

### 12.4 其他问题

- **硬编码 dev key**: `api/deps.py:14-17` — 明文 API 密钥
- **OperationConfig God Node**: `agent/config/types.py:23` — graph 报告 103 边，但非核心原语（仅 YAML 加载构建）
- **graphify graph 过期**: 快照 commit `048ddc97`，需运行 `graphify update .`
- **pyproject.toml 版本不一致**: 仍是 `0.4.1`，与 README 声称 `2.6.0` 不符

---

## 13. 关键类索引 / Key Class Index

| 类 | 文件:行 |
|---|---|
| `BaseNode` | `QuantNodes/core/node.py:119` |
| `Pipeline` | `QuantNodes/core/pipeline.py:38` |
| `Parallel` | `QuantNodes/core/pipeline.py:119` |
| `Join` | `QuantNodes/core/pipeline.py:232` |
| `IfNode` | `QuantNodes/core/control.py:47` |
| `MapNode` | `QuantNodes/core/control.py:146` |
| `WhileNode` | `QuantNodes/core/control.py:299` |
| `Expression` | `QuantNodes/core/expression.py:50` |
| `ExpressionBuilder` | `QuantNodes/core/expression.py:679` |
| `parse_expression` | `QuantNodes/core/ast_parser.py:34` |
| `BaseDBNode` | `QuantNodes/database_node/base.py:11` |
| `SQLiteNode` | `QuantNodes/database_node/sqlite_node.py:14` |
| `DuckDBNode` | `QuantNodes/database_node/duckdb_node.py:13` |
| `MySQLNode` | `QuantNodes/database_node/mysql_node.py:14` |
| `ClickHouseNode` | `QuantNodes/database_node/clickhouse_node.py:162` |
| `CSVNode` | `QuantNodes/database_node/csv_node.py:14` |
| `ParquetNode` | `QuantNodes/database_node/parquet_node.py:14` |
| `OperatorNode` | `QuantNodes/operator_node/base.py:16` |
| `ChainOperator` | `QuantNodes/operator_node/base.py:70` |
| `SQLBuilderNode` | `QuantNodes/operator_node/sql_builder.py:14` |
| `TableQueryNode` | `QuantNodes/operator_node/query_node.py:16` |
| `TransformNode` | `QuantNodes/operator_node/transform.py:16` |
| `Factor` | `QuantNodes/factor_node/factor.py:128` |
| `FactorDB` | `QuantNodes/factor_node/factor_db.py:14` |
| `PointOperation` | `QuantNodes/factor_node/factor_operation.py:158` |
| `TimeOperation` | `QuantNodes/factor_node/factor_operation.py:453` |
| `SectionOperation` | `QuantNodes/factor_node/factor_operation.py:717` |
| `PanelOperation` | `QuantNodes/factor_node/factor_operation.py:938` |
| `FactorTable` | `QuantNodes/factor_node/factor_table.py:401` |
| `_OPERATOR_REGISTRY` | `QuantNodes/factor_node/factor_functions/_helpers.py:24` |
| `OperatorCategory` | `QuantNodes/factor_node/factor_functions/_helpers.py:33` |
| `register_operator` | `QuantNodes/factor_node/factor_functions/_helpers.py:45` |
| `_ensure_expr` | `QuantNodes/factor_node/factor_functions/_helpers.py:67` |
| `CustomOperator` | `QuantNodes/operators/custom.py:111` |
| `CustomOperatorBuilder` | `QuantNodes/operators/custom.py:20` |
| `BacktestNode` | `QuantNodes/backtest/backtest_node.py:34` |
| `BacktestResult` | `QuantNodes/backtest/backtest_node.py:19` |
| `BacktestPipeline` | `QuantNodes/backtest/backtest_node.py:156` |
| `StrategyNode` | `QuantNodes/backtest/strategy_node.py:72` |
| `Order` | `QuantNodes/backtest/strategy_node.py:19` |
| `Signal` | `QuantNodes/backtest/strategy_node.py:32` |
| `OrdersResult` | `QuantNodes/backtest/strategy_node.py:41` |
| `RiskNode` | `QuantNodes/backtest/risk_node.py:52` |
| `RiskResult` | `QuantNodes/backtest/risk_node.py:31` |
| `PositionLimitRiskNode` | `QuantNodes/backtest/risk_node.py:208` |
| `StopLossRiskNode` | `QuantNodes/backtest/risk_node.py:253` |
| `CashRiskNode` | `QuantNodes/backtest/risk_node.py:285` |
| `CompositeRiskNode` | `QuantNodes/backtest/risk_node.py:316` |
| `BrokerNode` | `QuantNodes/backtest/broker_node.py:62` |
| `TradeResult` | `QuantNodes/backtest/broker_node.py:35` |
| `SimulatedBrokerNode` | `QuantNodes/backtest/broker_node.py:188` |
| `ExecutionBrokerNode` | `QuantNodes/backtest/broker_node.py:267` |
| `ConfigBacktestRunner` | `QuantNodes/backtest/config_runner.py:27` |
| `ConfigStrategyNode` | `QuantNodes/backtest/config_strategy.py:17` |
| `StrategyConfig` | `QuantNodes/agent/config/types.py:23` |
| `FactorConfig` | `QuantNodes/agent/config/types.py` |
| `OperationConfig` | `QuantNodes/agent/config/types.py:23` |
| `ConfigExecutor` | `QuantNodes/agent/config/executor.py:242` |
| `ExprParser` | `QuantNodes/agent/config/executor.py:19` |
| `Agent` | `QuantNodes/agent/__init__.py:17` |
| `AgentLoop` | `QuantNodes/agent/core/loop.py:29` |
| `AgentRunner` | `QuantNodes/agent/core/runner.py:20` |
| `ToolRegistry` | `QuantNodes/agent/tools/registry.py:14` |
| `SkillRegistry` | `QuantNodes/agent/skills/registry.py:15` |
| `PermissionService` | `QuantNodes/agent/permission/service.py:24` |
| `MarketDataCacheNode` | `QuantNodes/cache_node/base.py:41` |
| `ParquetCacheStore` | `QuantNodes/cache_node/cache_store.py:16` |
| `SQLExpression` | `QuantNodes/symbolic/expression.py:14` |
| `SQLCompiler` | `QuantNodes/symbolic/compiler.py:17` |
| `SQLOptimizer` | `QuantNodes/symbolic/optimizer.py:16` |
| `MetricsCollector` | `QuantNodes/monitor/monitor/collector.py:16` |
| `DriftDetector` | `QuantNodes/monitor/monitor/drift.py:12` |
| `StrategyScheduler` | `QuantNodes/monitor/scheduler/scheduler.py:24` |
| `VersionManager` | `QuantNodes/monitor/version/version_manager.py:15` |
| `DatabaseManager` | `QuantNodes/monitor/storage/repository.py` |
| `WikiFactorProxy` | `QuantNodes/research/wiki.py` |
| `FactorMiner` | `QuantNodes/research/factor_miner.py:19` |
| `AutoResearcher` | `QuantNodes/research/auto_researcher.py:32` |
| `MCTSSearch` | `QuantNodes/research/mcts_search.py:30` |
| `OpenAIClient` | `QuantNodes/ai/llm/openai.py:27` |
| `CodeSandbox` | `QuantNodes/ai/sandbox.py:16` |
| `StrategyGenerator` | `QuantNodes/ai/strategy_gen.py:19` |
| `PipelineOptimizer` | `QuantNodes/ai/optimizer.py:21` |
| `PromptLibrary` | `QuantNodes/ai/prompts/__init__.py:13` |
| `DisplayNode` | `QuantNodes/ui_node/base.py:39` |
| `TableDisplayNode` | `QuantNodes/ui_node/base.py` |
| `ChartDisplayNode` | `QuantNodes/ui_node/base.py` |
| `MetricDisplayNode` | `QuantNodes/ui_node/base.py` |

---

## 14. Graph God Nodes 映射 / God Nodes Mapping

基于 `graphify-out/GRAPH_REPORT.md` (commit `048ddc97`) 的 top-10 god nodes:

| God Node | 边数 | v2.6 实际位置 | 说明 |
|---|---|---|---|
| `ConfigExecutor` | 152 | `agent/config/executor.py:242` | 942 行 god class，**P0 未做** |
| `StrategyConfig` | 127 | `agent/config/types.py` | Pydantic 配置容器 |
| `_ensure_expr()` | 119 | `factor_node/factor_functions/_helpers.py:67` | `str → pl.col` 适配 |
| `Order` | 117 | `backtest/strategy_node.py:19` | 策略订单 |
| `OrdersResult` | 113 | `backtest/strategy_node.py:41` | 订单集合 |
| `range()` | 113 | Python 内置 | 跨社区桥梁 |
| `ColumnRef` | 108 | `symbolic/expression.py:14` | SQL AST 列引用 |
| `FactorConfig` | 103 | `agent/config/types.py` | 因子配置容器 |
| `OperationConfig` | 103 | `agent/config/types.py:23` | 算子配置 (5 字段) |
| `LLMResponse` | 102 | `ai/llm/base.py` | LLM 响应 |

**过期节点** (已归档): 部分在 `archive/QuantNodes/agent/` 中被冻结，但 graph 快照未反映归档状态。

---

## 附录 / Appendix: 版本快照

| 项目 | 版本 |
|------|------|
| Python 包版本 (pyproject.toml) | `0.4.1` |
| README 声称 | `2.6.0` |
| agent 内部版本号 | `2.5.0` |
| graphify 快照 commit | `048ddc97` |
| 归档冻结 commit | `d388865` |

---

*本文档基于代码事实生成，不反映 README 变更日志中的未完成声明。下次更新请运行 `graphify update .` 刷新知识图谱。*
