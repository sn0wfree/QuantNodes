# Agent 系统架构设计

> 合并自: 12-Agent业界调研与设计模式.md + 13-Agent系统架构设计.md + 14-Agent实施计划.md + 15-Config-Driven方案.md  
> 架构模式: **HKUDS nanobot 0.2.1 上游核心** + QuantNodes 量化工具集 + quant_dream 扩展  
> 通信协议: MCP (Model Context Protocol)  
> 上游依赖: `nanobot-ai>=0.2.1,<0.3.0`（`[agent]` 可选依赖；本地开发期源码 `/tmp/nanobot`）  
> 状态: **v3.0.0 已完成**（上游迁移 + 量化增强，已发 tag `v3.0.0`）

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
| **上游复用** | Agent 核心运行时直接采用 HKUDS nanobot 0.2.1 (`Nanobot.from_config` + `AgentLoop` + `MemoryStore/Dream`)，零自写 |
| **量化定制** | 保留 15 个 quant 工具 + QuantDreamHook + quant 专属 skill，挂在 nanobot 的 hook/registry 上 |
| **松耦合** | 通过 MCP 协议桥接各子系统，各层独立演进 |
| **知识驱动** | 研究结果沉淀到 `.agent/memory/MEMORY.md` + llmwikify，形成知识飞轮 |
| **安全优先** | 所有代码执行经过 CodeSandbox，三级权限控制 |
| **可复现** | 完整的研究过程记录，所有结果可追溯、可复现 |
| **渐进式** | 技能按需加载，Token高效，优雅降级 |
| **可升级** | 上游 alpha 期锁次版本号 `<0.3.0`，通过 facade 隔离，季度 sync upstream |

---

## 二、业界调研与设计模式

### 2.1 核心框架横向对比

| 项目 | 核心优势 | 量化场景适配度 |
|------|----------|----------------|
| **HKUDS nanobot** ⭐ | 极简设计、零数据库依赖、纯文件系统、5000行核心代码、subagent+MCP+webui+12 channel+Dream | 5星 |
| **opencode** | .agent/可移植规范、四层记忆架构、渐进式技能披露 | 5星 |
| **CrewAI** | 角色化团队、装饰器驱动、YAML配置 | 4星 |
| **LangGraph** | 状态机图计算、持久化执行断点续跑 | 4星 |
| **OpenAI Agents** | Handoff委托模式、Sandbox隔离 | 3星 |

> **v3.0.0 决策**：从"复刻 nanobot 架构"升级为"直接消费 HKUDS/nanobot 0.2.1 上游"。源码 `/tmp/nanobot`（PyPI 包名 `nanobot-ai`）。

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

### 3.1 Agent子系统（v3.0.0 新版）

```
agent/
├── __init__.py                    # 向后兼容导出 (Agent → Nanobot bridge)
├── nanobot_bridge.py              # ⭐ NEW: Nanobot.from_config 包装门面 (~100 行)
├── config_mapper.py               # ⭐ NEW: .env → nanobot_config.json (~120 行)
├── core/
│   ├── quant_dream.py             # ⭐ 量化专属 Dream 钩子 (moved from dream.py)
│   └── dream.py                   # 向后兼容 shim (re-export quant_dream)
├── tools/                         # 量化工具集 (15 个，改父类为 nanobot.agent.tools.base.Tool)
│   ├── base.py                    # 保留参数验证/cast 工具方法
│   ├── registry.py                # 注册到 nanobot ToolRegistry
│   ├── sandbox.py                 # CodeSandbox封装
│   ├── pipeline.py                # Pipeline构建验证
│   ├── strategy.py                # StrategyGenerator封装
│   ├── backtest.py                # 回测运行工具
│   ├── factor.py                  # 因子分析工具
│   ├── config_backtest.py         # 配置驱动回测
│   ├── wiki.py                    # Wiki知识库
│   ├── file_ops.py                # 文件读写编辑
│   ├── code_search.py             # 代码搜索
│   ├── git_ops.py                 # Git操作
│   ├── web_fetch.py               # 网页抓取
│   ├── web_search.py              # 网络搜索
│   ├── task.py                    # 任务管理
│   └── echo.py                    # 测试工具
├── skills_quant/                  # ⭐ NEW: 量化专属 SKILL.md
│   ├── factor-research/SKILL.md
│   ├── strategy-design/SKILL.md
│   ├── backtest-analyze/SKILL.md
│   ├── risk-management/SKILL.md
│   ├── quant-dream/SKILL.md
│   └── config-driven/SKILL.md
├── providers/                     # Provider 工厂 (输出 nanobot.providers 配置)
│   ├── base.py                    # Provider基类
│   └── quantnodes.py              # .env → dialect 推断
├── cron_jobs.py                   # ⭐ NEW: 周期任务 (日终/周度/月度)
└── utils/
    └── helpers.py

mcp_server/                        # ⭐ NEW: 把 quant 能力暴露为 MCP server (stdio)
├── __init__.py
├── server.py                      # FastMCP("quant") 注册 8 个核心 tool
└── tools/                         # backtest/factor/strategy/pipeline/wiki/sandbox/config_backtest/data_query

.agent/                            # ⭐ NEW workspace (上游 nanobot 默认)
├── agents/                        # Multi-agent 团队 (subagent)
│   ├── main.md                    # ResearchDirector
│   ├── factor-analyst.md
│   ├── backtest-engineer.md
│   └── risk-manager.md
├── skills/                        # 用户自定义 skill + skills_quant/ 自动 link
├── memory/                        # MEMORY.md / SOUL.md / USER.md / history.jsonl
├── SOUL.md                        # 个性化指令
├── USER.md                        # 用户偏好
└── nanobot_config.json            # providers / mcpServers / cron / channels
```

> **删除模块**（v3.0.0）：`agent/core/{loop,runner,memory,compaction,autocompact,context,hook}.py`、`agent/bus/`、`agent/session/`、`agent/templates/agent/`、`agent/config/{loader,executor,types}.py`、`agent/cli/main.py`、`agent/web/` — 全部由上游 nanobot 替代。

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

### 6.0 v3.0.0 — 上游 nanobot 迁移（**已完成，已发 tag `v3.0.0`**）

**目标**: 直接消费 HKUDS/nanobot 0.2.1 上游，删除自写 core，量化工具 hook 化

| Stage | 内容 | Commit | 状态 |
|-------|------|--------|------|
| **0** | 装依赖 + 跑基线 | `bd00030` / `979e611` | ✅ |
| **1** | 核心重构（bridge + tools 改父类 + dream 保留） | `2584462` | ✅ |
| **2** | API 解耦 `api/services/*` | `9e5999a` | ✅ |
| **3** | workspace 迁移 `.quant_agent/` → `.agent/` | `4b7560e` | ✅ |
| **4** | 测试 + 文档 | `5072ed7` | ✅ |
| **5.1** | Subagent 多 Agent 团队（main/factor-analyst/backtest-engineer/risk-manager） | `f7ac409` | ✅ |
| **5.2** | MCP server（quant 能力 stdio + HTTP 暴露，9 tools） | `a37ef30` | ✅ |
| **5.3** | 单进程 WebUI（端口 18080）+ nanobot-ai 改可选依赖 | `8bd8493` | ✅ |
| **5.4** | 渠道接入（飞书 + WebSocket wire protocol） | `81623ab` | ✅ |
| **5.5** | Cron 调度（日终/周度/月度） | `258f4dd` | ✅ |
| **6** | 测试稳定化 + pandas 3.0 兼容 + 跨测试污染根因修复 | `c5a7e3c` / `0ec6fe0` / `30a0352` / `f30f9b8` | ✅ |

> **最终基线**: 非 agent `5163 passed / 21 skipped / 0 failed`、`tests/agent` `574 passed / 13 skipped`（顺序 + 并行均通过）。详见 [CHANGELOG.md](../CHANGELOG.md) `[3.0.0]` 与 [可选依赖安装指南](15-可选依赖安装指南.md)。

### 6.1-6.5 历史阶段（已归档）

v3.0.0 之前的 Phase 1-5 已全部完成但**自写 core 已废弃**，作为历史归档。详见 git log。

| 历史阶段 | 状态 | 备注 |
|----------|------|------|
| 6.1 Phase 1: 核心框架复刻 (3900 行) | ✅ 已完成 → 被上游替代 | `core/{loop,runner,memory}.py` 删除 |
| 6.2 Phase 2: QuantNodes 工具集 (1150 行) | ✅ 已完成 → 改父类保留 | `tools/*.py` 改继承 `nanobot.agent.tools.base.Tool` |
| 6.3 Phase 3: llmwikify + Polars | ⚠️ 部分完成 | MCP 桥/Polars 迁移到上游，Wiki 客户端仍 TODO |
| 6.4 Phase 4: 技能系统 | ⬜ → 改为 SKILL.md 格式 | `skills_quant/*/SKILL.md` |
| 6.5 Phase 5: Polars 统一迁移 | ✅ 已完成 | 不变 |

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

**文档版本**: v3.0.0  
**最后更新**: 2026-06-23  
**变更摘要**: 从"复刻 nanobot 架构"升级为"直接消费 HKUDS/nanobot 0.2.1 上游"。详见 [docs/14-上游nanobot升级指南.md](14-上游nanobot升级指南.md)。

## 工作区约定

v3.0.0+ 默认 workspace 为 `.agent/`（HKUDS nanobot 上游约定），从 v2.x 的 `.quant_agent/` 迁移而来：

- 迁移脚本：`scripts/migrate_workspace.py`（`--src .quant_agent --dst .agent`）
- v2.x MEMORY.md 自动分割为 `.agent/SOUL.md`（personality）+ `.agent/memory/MEMORY.md`（facts）
- `.agent/` 在 `.gitignore` 中（含 settings.json 里的 API key）
- 旧的 `.quant_agent/` 不自动删除，需手动确认后清理

上游 nanobot 在 `.agent/` 下的关键文件：

```
.agent/
├── nanobot_config.json    # providers / agents.defaults 配置
├── SOUL.md                # 人格 / persona（每个 turn 都会读）
├── USER.md                # 用户偏好
├── memory/
│   ├── MEMORY.md          # 长期记忆（DREAM 整合产出）
│   └── history.jsonl      # 原始会话历史
├── skills/                # 用户自定义 SKILL.md
├── agents/                # 多 Agent 团队（main.md / factor-analyst.md / ...）
├── sessions/              # 会话持久化
└── cron/                  # 周期任务状态
```

## 十一、Stage 5.3 — 单进程集成 + 可选依赖

### 11.1 核心约束（用户决策）

1. **可选依赖**：`nanobot-ai` 不应是强制依赖——很多用户只想要量化工具库（Wiki / Factor / Backtest / Strategy），不需要 agent。
2. **单进程**：用户明确反对多进程。WebUI、agent、量化 API 应在同一个 uvicorn 进程内。

### 11.2 单进程架构

```
QuantNodes 进程（uvicorn 启动）
├─ FastAPI :8000           ← 量化 REST API
│   ├─ /api/wiki /factor /backtest /strategy /...
│   ├─ /api/agent/status /health /restart /chat/send
│   └─ /agent-chat → iframe → :18080
├─ nanobot gateway :18080  ← WebUI SPA + WebSocket（同进程 asyncio.create_task）
│   ├─ AgentLoop           ← asyncio.Task (agent.run)
│   ├─ ChannelManager      ← 启动 WebSocketChannel + 静态 SPA mount
│   ├─ CronService         ← asyncio.Task (cron.start)
│   └─ QuantDream hook     ← AgentHook on agent.dream
└─ MCP server (stdio)      ← 由 nanobot config.mcpServers 自动拉起
```

### 11.3 为什么不用 `_run_gateway()`？

nanobot 上游的 `cli/commands.py::_run_gateway()` 用 `asyncio.run()` 抢占事件循环。这与 uvicorn 已有的事件循环**冲突**。我们绕过它：

- 直接调用 `AgentLoop.from_config()` / `ChannelManager(...)` / `CronService(...)` 的 public API
- 用 `asyncio.create_task()` 启动后台任务（不 `asyncio.run`）
- 在 FastAPI lifespan 的 `try/finally` 中 `agent.stop()` + `cron.stop()` + `channels.stop_all()` + `agent.close_mcp()`

### 11.4 优雅降级（PEP 562）

```python
# QuantNodes/agent/__init__.py
try:
    from nanobot.agent.tools.base import Tool as _ProbeTool
    NANOBOT_AVAILABLE = True
    del _ProbeTool
except ImportError:
    NANOBOT_AVAILABLE = False


class NanobotNotInstalled(ImportError):
    """Raised when user code touches a nanobot-only symbol without [agent] extra."""


# Re-exports: when NANOBOT_AVAILABLE, normal; otherwise, a proxy that
# raises on every attribute access.
if NANOBOT_AVAILABLE:
    from .nanobot_bridge import Agent
    from .tools import register_all_quant_tools
else:
    class _NanobotUnavailableProxy:
        def __getattr__(self, name):
            raise NanobotNotInstalled(name)
        def __call__(self, *args, **kwargs):
            raise NanobotNotInstalled("Agent")
    Agent = _NanobotUnavailableProxy()
    register_all_quant_tools = _NanobotUnavailableProxy()
```

**关键设计**：
- `from QuantNodes.agent import Agent` 永远成功
- `Agent(...)` / `Agent.attr` / `register_all_quant_tools(...)` 抛 `NanobotNotInstalled`
- 14 个 quant tool 类可独立 import（`Tool` 父类在未装 nanobot 时退化为最小 ABC）
- API 端点 `/api/agent/*` 返回 503 + install hint，前端 AgentChat.vue 显示提示页

### 11.5 日志桥接

```python
# api/main.py lifespan
if NANOBOT_AVAILABLE:
    from nanobot.utils.logging_bridge import redirect_lib_logging
    redirect_lib_logging("uvicorn", level="INFO")
    redirect_lib_logging("uvicorn.access", level="INFO")
    redirect_lib_logging("fastapi", level="INFO")
```

`redirect_lib_logging`（`nanobot.utils.logging_bridge`）把 stdlib `logging` 路由到 loguru，达成日志统一。

### 11.6 端口分配

| 端口 | 用途 | 配置 |
|------|------|------|
| 8000 | FastAPI 量化 REST API | uvicorn `--port`（与上游解耦） |
| 18080 | nanobot WebUI + WebSocket | `NANOBOT_GATEWAY_PORT`（默认 18080） |
| 8765 | （可选）独立 WebSocket channel | `NANOBOT_WEBSOCKET_PORT` |

前端 `VITE_NANOBOT_GATEWAY_URL` 必须指向 `:18080`，让 `/agent-chat` iframe 加载上游 SPA。

### 11.7 失败模式

| 场景 | 行为 | 端点响应 |
|------|------|---------|
| 未装 nanobot-ai | `state=unavailable` | 503 + install hint |
| LLM provider 配置错误 | `state=error` | 503 + error message |
| Component 启动失败 | `state=error` + 不会停 FastAPI | 503 |
| WebUI SPA 缺 `nanobot/web/dist/` | WS API 仍可用，iframe 显示空白 | — |
| nanobot gateway port 占用 | `state=error` | 503 + 错误日志 |

## 十二、Stage 5.4 — 渠道接入（WebSocket + 飞书）

### 12.1 渠道架构

nanobot 0.2.1 通过 `ChannelManager` 统一管理多个 chat channel：
- **WebSocket**（默认启用）：同进程 `:18080` 上跑 WebSocket server + 静态 SPA
- **Feishu**（条件启用）：`FEISHU_APP_ID` + `FEISHU_APP_SECRET` 同时设置时启动
- 未来可加：Telegram / Slack / Discord / Email / DingTalk（上游均已实现）

### 12.2 配置注入

`config_mapper.py::build_nanobot_config` 把环境变量翻译成 nanobot 配置：

```python
{
  "channels": {
    "websocket": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 18080,        # 复用 NANOBOT_GATEWAY_PORT
      "path": "/",
      "tokenIssuePath": "/webui/token",
      "websocketRequiresToken": true,
      "allowFrom": ["*"],
      "streaming": true,
    },
    "feishu": {
      "enabled": true,      # FEISHU_APP_ID + SECRET 同时存在
      "appId": "...",
      "appSecret": "...",
      "domain": "feishu",   # 或 "lark" 国际版
      "groupPolicy": "mention",  # "open" = 接收所有群消息
      "replyToMessage": false,
      "streaming": true,
    },
  },
}
```

`api/services/nanobot_runtime.py::_build_components` 通过 `channel_overrides` 把 `NANOBOT_GATEWAY_HOST`/`PORT` 注入到 websocket 块。

### 12.3 WebSocket wire 协议（前端 useNanobotWebSocket.ts）

nanobot 的 WebSocket channel 需要先获取短期 token：

```
1. Client → GET http://gateway:18080/webui/bootstrap
   Server → {"token": "nbwt_...", "ws_path": "/", "model_name": "gpt-4o", ...}

2. Client → WebSocket ws://gateway:18080/?token=nbwt_...&client_id=quantnodes-webui
   Server → {"event": "attached", "chat_id": "default", "client_id": "..."}

3. Client → {"type": "message", "content": "你好", "chat_id": "default"}
   Server → {"event": "message", "chat_id": "default", "text": "...streamed..."}
        → {"event": "tool_call", "name": "wiki_get", "arguments": {...}}
        → {"event": "tool_result", "name": "wiki_get", "success": true, "content": {...}}
        → {"event": "message", "chat_id": "default", "text": "..."}
```

`frontend/src/composables/useNanobotWebSocket.ts` 封装了这套握手：
1. `fetchBootstrap()` 拿到 token
2. `WebSocket(wsUrl)` 携带 `?token=...&client_id=...`
3. 解析事件 → `onEvent(payload)`
4. 发送：`ws.send(JSON.stringify({type: 'message', content, chat_id}))`

### 12.4 飞书 channel（WebSocket 长连接）

- **SDK**：`pip install lark-oapi`（独立可选依赖）
- **触发**：`FEISHU_APP_ID` + `FEISHU_APP_SECRET` 同时设置才启动
- **网络**：用 lark-oapi 的 WebSocket 长连接，**无需公网 IP 或 webhook**
- **群策略**：`mention`（仅响应 @机器人）或 `open`（接收所有）
- **认证**：可选 `FEISHU_ENCRYPT_KEY` / `FEISHU_VERIFICATION_TOKEN`
- **白名单**：可选 `FEISHU_ALLOW_FROM=ou_aaa,ou_bbb`（open_id 列表）

### 12.5 ChannelManager 生命周期

```
start():
  1. config_mapper 生成 channels dict
  2. Config.model_validate() → cfg.channels.<name> 是 dict
  3. ChannelManager(cfg, bus, session_manager, webui_static_dist=True)
     - websocket: enabled → 启动 ws.Server + static SPA mount
     - feishu: enabled + lark_oapi → 启动 ws.Client 线程
  4. asyncio.create_task(channels.start_all())
  
stop():
  1. cron.stop() → agent.stop() → channels.stop_all() → agent.close_mcp()
  2. WebSocket server.close() + Feishu ws.Client._running = False
```

### 12.6 失败模式

| 场景 | 行为 |
|------|------|
| `FEISHU_APP_SECRET` 缺失 | feishu 频道不启动（log warning） |
| 未装 `lark-oapi` | FeishuChannel.start() 拒绝启动，log "SDK not installed" |
| Feishu app_id 错误 | lark SDK 在首次连接时失败，log 详细错误 |
| WebSocket token 过期（5 min） | nanobot 自动 `nbwt_...` 续签；前端 reconnect() 重新 bootstrap |
| 多 channel 同时收到消息 | MessageBus 统一处理，每个 channel 的 `chat_id` 隔离 session |

## 十三、Stage 5.5 — 量化专属 Cron 调度

### 13.1 三套 quant 系统任务

v3.0.0 在 nanobot CronService 上注册 3 个 quant 系统任务（默认 Asia/Shanghai 时区）：

| 任务 ID | cron 表达式 | 时机 | 用途 |
|---------|------------|------|------|
| `quant-quant-daily-recap` | `30 16 * * 1-5` | 工作日 16:30 | 因子 IC 重算 + 回测归档 + 日终摘要 |
| `quant-quant-weekly-review` | `0 22 * * 0` | 周日 22:00 | 因子周报 + 风险归因 + Wiki 周报 |
| `quant-quant-monthly-strategy-pool` | `0 2 1 * *` | 每月 1日 02:00 | Wiki 增量 + 策略池月度评审 |

设计要点：
- **payload.kind = `system_event`** — 用 `register_system_job()` 注册（idempotent，重启不重复）
- **payload.deliver = True** — 完成后结果送到配置的 channel（默认 Feishu 群）
- **message = agent prompt** — 任务内容是给 agent 的 prompt（agent 用 factor/backtest/wiki 工具生成报告）
- **session_key = `cron:<job>`** — 每个任务独立 session，便于追踪

### 13.2 注册流程

```python
# QuantNodes/agent/cron_jobs.py
from QuantNodes.agent.cron_jobs import register_quant_cron_jobs

# In NanobotRuntime._build_components (after CronService 构造)
try:
    register_quant_cron_jobs(self._cron)
except ImportError:
    logger.warning("Could not register quant cron jobs: nanobot not available")
```

`register_quant_cron_jobs`：
1. 读 `DEFAULT_QUANT_CRON_JOBS` (3 个)
2. 应用 env var 覆盖（`QUANTNODES__CRON__<NAME>__ENABLED/CRON_EXPR/MESSAGE/DELIVER/CHANNEL`）
3. 过滤 `enabled=False` 的
4. 对每个 job 构造 `CronJob(id='quant-<name>', schedule=CronSchedule(kind='cron', expr, tz), payload=CronPayload(kind='system_event', message, deliver, channel, session_key))`
5. 调 `cron.register_system_job(cron_job)` —— 自动去重同 id

### 13.3 env var 覆盖

```bash
# 完全禁用
QUANTNODES__CRON__QUANT_DAILY_RECAP__ENABLED=false

# 修改 cron 表达式
QUANTNODES__CRON__QUANT_WEEKLY_REVIEW__CRON_EXPR="0 20 * * 0"

# 自定义 prompt
QUANTNODES__CRON__QUANT_DAILY_RECAP__MESSAGE="只看 IC > 0.05 的因子"

# 关闭结果推送（只跑不送）
QUANTNODES__CRON__QUANT_MONTHLY_STRATEGY_POOL__DELIVER=false
```

### 13.4 API 端点

| 端点 | 用途 |
|------|------|
| `GET /api/agent/cron` | 列出所有 cron jobs（含 quant 系统任务） |
| `GET /api/agent/cron/{id}/run-now` | 立即触发某个任务（手动 override） |

返回结构：
```json
{
  "available": true,
  "state": "running",
  "count": 3,
  "jobs": [
    {
      "id": "quant-quant-daily-recap",
      "name": "quant-daily-recap",
      "enabled": true,
      "schedule": {"kind": "cron", "expr": "30 16 * * 1-5", "tz": "Asia/Shanghai"},
      "payload": {"kind": "system_event", "message_preview": "...", "deliver": true},
      "state": {"next_run_at_ms": 1234567890, "last_status": "ok", "run_count": 42}
    }
  ]
}
```

### 13.5 失败模式

| 场景 | 行为 |
|------|------|
| cron 表达式语法错误 | nanobot CronService 拒绝注册，`run_history` 留空 |
| agent 任务超时 | `last_status=error` + `last_error` 写入，下次仍按 schedule 触发 |
| jobs.json 损坏 | `_load_jobs` 自动备份为 `.corrupt-<ts>`，**拒绝以空 store 启动**（防数据丢失） |
| 多个 quant 任务同时触发 | nanobot 串行执行（`await self._execute_job(job)`），互不干扰 |

