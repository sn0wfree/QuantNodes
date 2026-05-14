# QuantNodes 架构分析报告

**日期**: 2026-05-13
**分析工具**: Graphify (11,581 节点, 19,630 边, 1,046 社区)

---

## 一、项目规模概览

| 指标 | 数值 |
|------|------|
| 文件数 | 504 |
| 代码量 | ~319,018 词 |
| 节点数 | 11,581 |
| 边数 | 19,630 |
| 社区数 | 1,046 |
| 提取质量 | 68% EXTRACTED · 32% INFERRED · 0% AMBIGUOUS |
| 孤立节点 | 3,097 (27%) |

---

## 二、模块职责地图

| 包 | 职责 | 关键类 |
|----|------|-------|
| **core** | 统一节点抽象、序列化、表达式解析、流水线/控制流 | `BaseNode`, `Pipeline`, `Expression`, `Cond` |
| **operators** | Polars 因子算子 (代理到 factor_functions) | `ts`, `sec`, `math`, `composite` |
| **factor_node** | 因子定义、存储、计算 | `Factor`, `FactorDB`, `PointOperation`, `SectionOperation` |
| **conf_node** | 配置加载 (INI/YAML/JSON/ENV) | `ConfigNode`, `YamlConfigNode` |
| **agent** | AI Agent 系统、工具、LLM 集成、会话管理 | `Agent`, `AgentLoop`, `ToolRegistry`, `ConfigExecutor` |
| **agent/config** | 策略/因子/操作配置解析和执行 | `ConfigLoader`, `ConfigExecutor`, `StrategyConfig` |
| **backtest** | 回测引擎、策略节点、风控管理、经纪商 | `BacktestNode`, `StrategyNode`, `RiskNode`, `BrokerNode` |
| **database_node** | 数据库连接器 (SQLite, DuckDB, MySQL, ClickHouse, CSV, Parquet) | `SQLiteNode`, `DuckDBNode`, `ClickHouseNode` |
| **monitor** | 策略监控、性能跟踪、漂移检测 | `StrategyRunRepository`, `DriftAlertRepository` |
| **operator_node** | 可链式调用的算子节点 | `OperatorNode`, `ChainOperator`, `SQLBuilderNode` |
| **ai** | LLM 客户端、Prompts、代码沙箱、策略生成 | `OpenAIClient`, `CodeSandbox`, `StrategyGenerator` |
| **research** | Wiki、因子挖掘、评估、MCTS 搜索、研报复现 | `WikiFactorProxy`, `FactorMiner`, `AutoResearcher` |
| **cache_node** | 市场数据缓存层 | `ParquetCacheStore`, `MarketDataCacheNode` |
| **ui_node** | UI 展示节点 (表格、文本、图表) | `TableDisplayNode`, `TextDisplayNode` |

---

## 三、God Nodes 分析

### 3.1 核心抽象 (按连接数排序)

| 节点 | 边数 | 说明 |
|------|------|------|
| `ConfigExecutor` | 152 | 配置执行器 - **God Controller** |
| `StrategyConfig` | 127 | 策略配置容器 |
| `_ensure_expr()` | 119 | 表达式处理 |
| `Order` | 117 | 订单 |
| `OrdersResult` | 113 | 订单结果 |
| `range()` | 113 | Python 内置 (跨社区桥梁) |
| `ColumnRef` | 108 | 列引用 |
| `FactorConfig` | 103 | 因子配置 |
| `OperationConfig` | 103 | 算子配置 |
| `LLMResponse` | 102 | LLM 响应 |

### 3.2 ConfigExecutor "God Controller" 问题

`ConfigExecutor` 是一个 **942 行的类**，承担了太多职责：

1. 表达式解析 (`ExprParser`)
2. 自定义算子加载
3. 因子表达式构建
4. 操作执行 (`_apply_operator`)
5. Universe 过滤
6. 回测运行 (`run_backtest`)
7. 风控节点管理 (`_build_risk_nodes`)
8. 经纪商节点创建 (`_build_broker_nodes`)

**违反单一职责原则 (SRP)** — 这是项目中最需要重构的部分。

---

## 四、耦合度矩阵

```
                core  operators  factor  conf  agent  backtest  database  monitor  ai  research
core             -      X         X      X     X       X         X        X        .     X
operators        X      -         X      .     X       X         .        .        .     .
factor_node      X      X         -      .     X       X         .        .        .     X
conf_node        X      .         .      -     X       X         .        .        .     .
agent            X      X         .      X     -       X         .        X        X     .
backtest         X      X         X      X     X       -         .        X        .     .
database_node    X      .         .      .     .       .         -        .        .     .
monitor          X      .         .      .     X       X         .        -        .     .
ai               .      .         .      .     X       .         .        .        -     X
research         X      .         X      .     .       .         .        .        X     -
```

**核心问题**: 所有模块都对 `core` 有依赖，`core` 是基础设施层，承受了最大的耦合压力。

---

## 五、社区凝聚力分析

| 社区 | 凝聚力 | 评估 |
|------|--------|------|
| Community 3 (Factor/DataType) | 0.06 | 中等 - 核心类型定义 |
| Community 8 (RiskNode) | 0.06 | 中等 - 风控子系统 |
| Community 12 (StrategyNode) | 0.06 | 中等 - 策略实现 |
| Community 6 (BaseNode/ABC) | 0.03 | 低 - 基础设置分散 |
| Community 1 (merge/neutralize) | 0.02 | 低 - 分散的归一化函数 |
| Community 29 (rolling_quantile fallback) | 0.09 | 较高 - 单一关注点 |

**大多数社区凝聚力低 (0.02-0.09)**，说明代码没有自然的模块边界。

---

## 六、关键架构问题

### 6.1 God Controller 反模式
`ConfigExecutor` 承担 5+ 职责，应拆分为：
- `ExpressionEngine` (解析)
- `FactorComputer` (因子计算)
- `BacktestOrchestrator` (回测编排)

### 6.2 功能重复
- `factor_node/factor_functions/` 和 `operators/` 存在重叠
- `operators/math.py` 导入自 `factor_node.factor_functions.math_ops`
- 不清楚应该用哪个

### 6.3 孤岛节点
- **3,097 个孤立节点** (27%)
- 可能是死代码、未连接的函数、或应该被连接的组件

### 6.4 低凝聚力
- 社区平均凝聚力 0.05
- 模块边界模糊

---

## 七、拆分建议

### P0 - ConfigExecutor 拆分

**当前**: 一个 942 行的类处理所有事

**目标**: 拆分为专注的组件

| 新组件 | 职责 | 位置 |
|--------|------|------|
| `ExpressionEngine` | 解析表达式字符串 | `agent/config/expression_engine.py` |
| `FactorComputer` | 从配置计算因子 | `agent/config/factor_computer.py` |
| `BacktestOrchestrator` | 编排回测 | `backtest/orchestrator.py` |

### P1 - Factor Functions 合并

**当前**: 双算子 (factor_functions + operators)

**目标**: 单一规范位置

```
# 选择一：保留 factor_functions，operators 只做重导出
# 选择二：反过来 - 选择一个作为canonical
```

### P2 - 按领域边界的微服务化

| 服务 | 模块 | 理由 |
|------|------|------|
| **Core Engine** | `core`, `operators`, `factor_node` | 纯计算，无外部依赖 |
| **Data Layer** | `database_node`, `cache_node` | 数据访问抽象 |
| **Execution** | `conf_node`, `backtest` | 配置驱动执行 |
| **AI Agent** | `agent`, `ai` | LLM 集成、工具系统 |
| **Research** | `research` | 分析工作流 |
| **Monitoring** | `monitor` | 运行时可观测性 |

### P3 - 孤立节点清理

- 识别 3,097 个孤立节点
- 分类：死代码 vs 应该被连接的组件
- 清理或修复

---

## 八、结论

**项目不是 "fat"，是 "genuinely complex"**

项目有多个正交的关注点，这些是合理的复杂度来源：
- 多个数据源 (SQLite, DuckDB, MySQL, ClickHouse)
- 因子计算 (时序、截面、面板操作)
- 策略回测 + 风控
- AI Agent + 工具系统
- 多格式配置管理
- 分布式序列化和执行

**架构债来自**:
1. Centralized God Classes (ConfigExecutor)
2. Duplicate functionality (factor_functions vs operators)
3. Low cohesion (平均 0.05)
4. 3,097 个孤岛节点

**拆分应该按领域边界**（数据、计算、执行、AI、监控），而不是随意微服务化。

---

## 九、后续行动

- [ ] **P0**: 拆分 ConfigExecutor
- [ ] **P1**: 合并 factor_functions 和 operators
- [ ] **P2**: 考虑微服务化可行性
- [ ] **P3**: 清理孤立节点
- [ ] **长期**: 持续监控架构健康度

---

*本文档由 Graphify 自动生成，基于 commit `048ddc97`*