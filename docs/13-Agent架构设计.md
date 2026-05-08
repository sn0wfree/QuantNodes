# Agent 系统架构设计

> 合并自: 12-Agent业界调研与设计模式.md + 13-Agent系统架构设计.md + 14-Agent实施计划.md + 15-Config-Driven方案.md  
> 架构模式: nanobot极简核心 + llmwikify知识沉淀 + QuantNodes量化引擎  
> 通信协议: MCP (Model Context Protocol)  
> 状态: 已完成 ✅

---

## 一、系统架构总览

### 1.1 三层松耦合架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION LAYER                        │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Vue 3 Web UI  │  CLI  │  Jupyter Notebook  │  API          │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        QUANT AGENT CORE                           │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  nanobot Minimalist Runtime                                    │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Agent Loop  │  Message Bus  │  Tool Registry  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  Memory Store  │  Context Builder  │  Skills Loader  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                   │                                 │
│                                   ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  MCP Protocol Bridge                                          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │  llmwikify   │  │  QuantNodes  │  │  Data Sources │   │  │
│  │  │  Adapter    │  │  Adapter    │  │  Adapter    │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  llmwikify Wiki  │     │  QuantNodes Engine │     │  Data Sources  │
│  * 策略知识库     │     │  * 因子计算引擎   │     │  * ClickHouse  │
│  * 因子图谱       │     │  * 回测引擎       │     │  * DuckDB      │
│  * 回测历史       │     │  * Pipeline执行   │     │  * MySQL       │
│  * 知识涌现       │     │  * CodeSandbox   │     │  * CSV/Parquet │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| **极简核心** | Agent核心基于nanobot，<1000行代码，易理解、易维护 |
| **松耦合** | 通过MCP协议桥接各子系统，各层独立演进 |
| **知识驱动** | 所有研究结果自动沉淀到llmwikify，形成知识飞轮 |
| **安全优先** | 所有代码执行经过CodeSandbox，三级权限控制 |
| **可复现** | 完整的研究过程记录，所有结果可追溯、可复现 |
| **渐进式** | 技能按需加载，Token高效，优雅降级 |

---

## 二、业界调研与设计模式

### 2.1 核心框架横向对比

| 项目 | 核心优势 | 量化场景适配度 |
|------|----------|----------------|
| **nanobot** | 极简设计、零数据库依赖、纯文件系统、5000行核心代码 | 5星 |
| **opencode** | .agent/可移植规范、四层记忆架构、渐进式技能披露 | 5星 |
| **CrewAI** | 角色化团队、装饰器驱动、YAML配置 | 4星 |
| **LangGraph** | 状态机图计算、持久化执行断点续跑 | 4星 |
| **OpenAI Agents** | Handoff委托模式、Sandbox隔离 | 3星 |

### 2.2 量化专属设计模式

#### 模式 1：文件系统优先的可移植研究工作区

```
.quantresearch/
├── RESEARCH_NOTES.md      # 人工编辑的研究心得
├── FACTOR_LIBRARY.md      # 已验证的因子库
├── STRATEGY_CANVAS.md     # 策略画布
├── memory/                # 分层记忆系统
│   ├── episodic/          # 情节记忆
│   ├── semantic/          # 语义记忆
│   └── working/           # 工作记忆
├── skills/                # 技能注册
├── protocols/             # 安全协议
└── research/              # 研究产物
```

#### 模式 2：MessageBus解耦的Agent循环

```
ResearchBus → ResearchAgent / BacktestAgent / RiskAgent
                    │
              Node执行层（复用BaseNode/Pipeline）
```

#### 模式 3：ResearchState + 检查点持久化

```python
State = {
    "research_id": str,
    "intent": str,
    "factor_candidates": List,
    "backtest_results": Dict,
    "checkpoint": str,
}
```

#### 模式 4：角色化Agent团队协作

```
ResearchDirector → FactorAnalyst / BacktestEngineer / RiskManager → ReportWriter
```

#### 模式 5：三阶段沙箱执行模型

1. Tool Pre-validation（参数Schema验证 + 权限匹配）
2. Sandboxed Execution（内存/时间/资源限制）
3. Result Audit（过拟合检测 + 未来函数检测）

---

## 三、目录结构

### 3.1 Agent子系统

```
agent/
├── __init__.py                    # Agent 对外 API
├── core/                          # nanobot核心
│   ├── loop.py                    # Agent主消息循环
│   ├── runner.py                  # 工具执行循环
│   ├── context.py                 # 上下文/Prompt构建
│   ├── memory.py                  # 记忆存储
│   ├── hook.py                    # 执行钩子系统
│   └── autocompact.py             # 会话历史压缩
├── tools/                         # 工具系统
│   ├── base.py                    # Tool基类
│   ├── registry.py                # 工具注册表
│   ├── sandbox.py                 # CodeSandbox封装
│   ├── pipeline.py                # Pipeline构建验证
│   ├── strategy.py                # StrategyGenerator封装
│   ├── backtest.py                # 回测运行工具
│   ├── factor.py                  # 因子分析工具
│   ├── config_backtest.py         # 配置驱动回测
│   └── mcp.py                     # MCP工具桥 (Phase 3)
├── bus/                           # 消息总线
│   ├── events.py                  # Inbound/Outbound消息
│   └── queue.py                   # 异步队列
├── session/                       # 会话管理
│   └── manager.py                 # 会话持久化
├── providers/                     # LLM Provider适配层
│   ├── base.py                    # Provider基类
│   └── quantnodes.py              # 适配现有LLMClientBase
├── config/                        # 配置驱动系统
│   ├── types.py                   # 类型定义
│   ├── loader.py                  # YAML配置解析器
│   ├── executor.py                # 配置执行器
│   └── templates/                 # 策略模板
├── wiki/                          # llmwikify集成 (Phase 3)
│   └── client.py                  # MCP客户端
├── skills/                        # 技能系统 (Phase 4)
│   └── loader.py                  # 渐进式加载器
├── templates/                     # Prompt模板
│   └── agent/
│       ├── identity.md
│       ├── system_prompt.md
│       └── tools_description.md
├── utils/                         # 工具函数
│   ├── helpers.py
│   └── prompt_templates.py
├── cli/                           # 命令行界面
│   └── main.py
└── web/                           # Web界面
    └── app.py
```

### 3.2 研究工作区

```
.quantresearch/
├── RESEARCH_NOTES.md
├── FACTOR_LIBRARY.md
├── STRATEGY_CANVAS.md
├── memory/
│   ├── episodic/
│   ├── semantic/
│   └── working/
├── skills/custom/
├── protocols/
│   ├── permissions.md
│   └── tool_schemas/
├── pipelines/
└── research/
```

---

## 四、核心工作流程

### 4.1 策略生成闭环

1. **接收用户请求** → 上下文构建
2. **LLM推理决策** → 调用工具
3. **执行回测验证** → 分析结果
4. **知识沉淀到Wiki** → 写入知识库
5. **输出给用户** → 策略代码 + Pipeline + 回测报告

### 4.2 策略复现工作流程

1. 用户上传论文/研报 → 解析核心思想
2. Wiki查询 → 检查是否已有类似策略
3. 如无 → 生成新策略代码
4. 对比验证 → 与论文结果一致性检查
5. 写入Wiki → 标记为"论文复现"来源

### 4.3 因子研究工作流程

1. Wiki查询 → 类似因子历史表现
2. 因子探索 → 单因子测试（IC/ICIR/分组回测）
3. 关系发现 → 与已有因子相关性分析
4. 建立知识图谱 → 因子关系
5. 写入Wiki → 因子页面+关系

---

## 五、MCP协议桥接

### 5.1 核心MCP工具

| 工具名称 | 功能说明 |
|----------|----------|
| quantnodes_backtest_run | 运行策略回测 |
| quantnodes_factor_test | 单因子有效性测试 |
| quantnodes_validate_code | 策略代码验证 |
| llmwikify_write_page | 写入Wiki知识库 |
| llmwikify_add_relation | 添加知识图谱关系 |
| llmwikify_query | 语义+全文搜索 |

### 5.2 量化专用Schema

#### 策略页面
- type: Strategy
- fields: name, category, confidence, source, tags
- sections: 策略逻辑, 核心因子, 回测表现, 年度收益, 适用场景, 风险与限制

#### 因子页面
- type: Factor
- fields: name, category, formula, created
- sections: 因子描述, 计算公式, 单因子表现, 分组回测, 相关性分析

#### 回测页面
- type: Backtest
- fields: strategy, date, universe, period
- sections: 回测参数, 绩效指标, 年度收益, 回撤分析, 风险分析

### 5.3 量化专用关系类型

```python
QUANT_RELATION_TYPES = {
    "uses",              # 策略 uses 因子
    "correlates_with",    # 因子 correlates_with 因子
    "outperforms",        # 策略 outperforms 策略
    "similar_to",         # 策略 similar_to 策略
    "contradicts",        # 研究结论 contradicts 研究结论
    "supports",          # 回测结果 supports 策略假设
    "optimizes",         # 策略A optimizes 策略B
    "extends",           # 策略A extends 策略B
}
```

---

## 六、实施计划

### 6.1 Phase 1: 核心框架复刻（✅ 已完成）

**目标**: 最小可运行Agent，支持对话 + 工具调用

| 任务 | 文件 | 代码量 | 状态 |
|------|------|--------|------|
| 消息总线 | `bus/events.py` + `queue.py` | ~100行 | ✅ |
| 会话管理 | `session/manager.py` | ~200行 | ✅ |
| 工具基类 | `tools/base.py` + `registry.py` | ~500行 | ✅ |
| Provider适配 | `providers/base.py` + `quantnodes.py` | ~400行 | ✅ |
| 上下文构建 | `core/context.py` + `hook.py` | ~300行 | ✅ |
| 执行循环 | `core/runner.py` | ~800行 | ✅ |
| 主循环 | `core/loop.py` + `autocompact.py` | ~600行 | ✅ |
| 记忆系统 | `core/memory.py` | ~200行 | ✅ |
| CLI入口 | `cli/main.py` | ~150行 | ✅ |
| 单元测试 | `tests/agent/test_*.py` | ~300行 | ✅ |

**总计**: ~3900行

### 6.2 Phase 2: QuantNodes工具集（✅ 已完成）

**目标**: 策略生成 → 验证 → 回测 完整闭环

| 任务 | 文件 | 代码量 | 状态 |
|------|------|--------|------|
| 沙箱工具 | `tools/sandbox.py` | ~100行 | ✅ |
| Pipeline工具 | `tools/pipeline.py` | ~150行 | ✅ |
| 策略生成工具 | `tools/strategy.py` | ~150行 | ✅ |
| 回测运行工具 | `tools/backtest.py` | ~200行 | ✅ |
| 因子分析工具 | `tools/factor.py` | ~150行 | ✅ |
| 端到端测试 | `tests/agent/test_e2e.py` | ~100行 | ✅ |
| Web界面 | `web/app.py` | ~300行 | ✅ |

**总计**: ~1150行

### 6.3 Phase 3: llmwikify + Polars v2.0（部分完成）

**目标**: 知识沉淀 + 配置驱动回测

| 任务 | 文件 | 状态 |
|------|------|------|
| MCP工具桥 | `tools/mcp.py` | ⬜ 待开始 |
| Wiki客户端 | `wiki/client.py` | ⬜ 待开始 |
| Polars算子库 | `factor_node/factor_functions/` | ✅ |
| Config加载器 | `agent/config/loader.py` | ✅ |
| Config执行器 | `agent/config/executor.py` | ✅ |
| 算子注册表 | `factor_node/factor_functions/__init__.py` | ✅ |
| TA-Lib集成 | `factor_node/factor_functions/talib_ops.py` | ✅ |
| 回测运行器 | `backtest/config_runner.py` | ✅ |

### 6.4 Phase 4: 技能系统（待开始）

| 任务 | 文件 | 代码量 |
|------|------|--------|
| 技能加载器 | `skills/loader.py` | ~200行 |
| 策略设计技能 | `skills/strategy_design/*.py` | ~300行 |
| 因子研究技能 | `skills/factor_research/*.py` | ~300行 |
| Dream系统 | `core/memory.py` 扩展 | ~300行 |

**总计**: ~1400行

### 6.5 Phase 5: FactorNode Polars 统一迁移（✅ 已完成）

详见 `22-算子系统设计与规范.md` 第十六节

---

## 七、并发设计

### 7.1 渐进式并发

```
Phase 1-2: Semaphore=1（全局串行）
Phase 3+: Semaphore=3（跨会话并发）
Phase 4+: Docker隔离（回测并发）
```

### 7.2 三层并发架构

```
全局并发门控 (Semaphore)
  └─ 会话级锁 (per-session Lock)
      └─ 工具级并发 (asyncio.gather)
          - 只读工具: 可并发
          - 有副作用工具: 串行
```

---

## 八、关键接口

### 8.1 Agent对外API

```python
class Agent:
    def __init__(self, workspace: str, config: dict = None): ...
    async def run(self, prompt: str, session_id: str = "default") -> str: ...
    async def chat(self, message: str, session_id: str = "default"): ...
```

### 8.2 Tool基类

```python
class Tool(ABC):
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def parameters(self) -> dict: ...
    @property
    def read_only(self) -> bool: ...
    async def execute(self, **kwargs) -> Any: ...
```

---

## 九、总工作量

| 阶段 | 代码量 | 状态 |
|------|--------|------|
| Phase 1 | ~3900行 | ✅ 完成 |
| Phase 2 | ~1150行 | ✅ 完成 |
| Phase 3 | ~700行 | ⚠️ 部分完成（MCP/Wiki 待实现） |
| Phase 4 | ~1400行 | ⬜ 待开始 |
| Phase 5 | ~1200行 | ✅ 完成 |
| **总计** | ~8350行 | |

---

## 十、关键决策点

| 决策项 | 推荐方案 | 备选方案 |
|--------|----------|----------|
| Agent核心 | 复刻nanobot | LangGraph / AutoGen |
| 知识库 | llmwikify独立进程 | SQLite向量库 |
| 通信协议 | MCP标准协议 | REST API |
| Prompt管理 | 独立markdown文件 | Python字符串模板 |
| 代码执行 | CodeSandbox + 子进程 | Docker容器 |
| 持久化 | 文件系统优先 | 数据库 |

---

## 十一、Config-Driven 配置驱动方案

### 11.1 设计目标

| 目标 | 说明 |
|------|------|
| **配置即策略** | Agent 编写 YAML 配置代替编写代码 |
| **自动闭环** | 配置 → 代码生成 → 验证 → 回测 自动执行 |
| **Agent 兜底** | 不可配置部分 Agent 补充自定义算子 |
| **测试自运行** | 配置中声明测试，Agent 不需要手动运行 |

### 11.2 执行流程

```
Agent 编写 strategy_config.yaml
    ↓
ConfigLoader 解析配置
    ↓
OperatorRegistry 检查覆盖度
    ├─ 全部可配置 → FactorExecutor.run(config)
    └─ 有无法表达 → Agent 编写自定义算子
    ↓
合并执行 + 返回结果
```

### 11.3 配置文件格式

```yaml
version: "1.0"
name: "momentum_alpha_v1"

data:
  source: "clickhouse"
  conn_ini: "conn.ini"
  table: "quote.cn_stock"
  date_column: "date"
  code_column: "code"

factors:
  - name: momentum_20d
    formula: "close / close.shift(20) - 1"

operations:
  - type: time_series
    name: momentum_ma
    category: ts_mean
    inputs: [momentum_20d]
    params: {window: 20}

composite:
  - name: alpha
    formula: "momentum_ma"

backtest:
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  initial_cash: 1000000
```

### 11.4 数据加载设计

| 节点 | 功能 | 数据源 |
|------|------|--------|
| `ClickHouseNode` | 查询/插入/DDL | 远程服务器 |
| `MySQLNode` | 查询/插入/DDL | 远程服务器 |
| `CSVNode` | 读取/过滤 | `.csv` |
| `ParquetNode` | 读取/过滤 | `.parquet` |
| `DuckDBNode` | 查询/分析 | `.duckdb` |

**列名映射策略**: 数据加载层统一列名，下游使用标准列名 (`date`, `code`, `open`, `high`, `low`, `close`, `volume`)

### 11.5 实施状态

| 组件 | 状态 | 文件 |
|------|------|------|
| ConfigLoader | ✅ 已完成 | `agent/config/loader.py` |
| ConfigExecutor | ✅ 已完成 | `agent/config/executor.py` |
| ConfigBacktestRunner | ✅ 已完成 | `backtest/config_runner.py` |
| 6种数据源支持 | ✅ 已完成 | `database_node/` |
| YAML策略模板 | ✅ 已完成 | `agent/config/templates/` |

---

**文档版本**: v2.0  
**最后更新**: 2026-05-06
