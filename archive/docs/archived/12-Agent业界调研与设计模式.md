# QuantNodes Agent 架构设计方案

> 创建日期: 2026-04-28
> 状态: 待讨论
> 调研来源: nanobot, opencode, openclaw, CrewAI, LangGraph

---

## 一、业界调研总结

### 1.1 核心框架横向对比

| 项目 | 核心优势 | 量化场景适配度 | 可复用代码规模 | 学习曲线 |
|------|----------|----------------|----------------|----------|
| **nanobot** | 极简设计、零数据库依赖、纯文件系统、5000行核心代码、内置MCP工具协议 | 5星 | 高 | 平缓 |
| **opencode** | .agent/可移植规范、四层记忆架构、渐进式技能披露、DESIGN.md约束、文件系统即总线 | 5星 | 很高 | 中等 |
| **CrewAI** | 角色化团队、装饰器驱动、YAML配置、比LangGraph快5.76倍、零LangChain依赖 | 4星 | 中 | 平缓 |
| **LangGraph** | 状态机图计算、持久化执行断点续跑、Human-in-loop原生支持 | 4星 | 中 | 陡峭 |
| **OpenAI Agents** | Handoff委托模式、Sandbox隔离、Tracing追踪 | 3星 | 低 | 平缓 |

### 1.2 现有架构的Agent友好特性

QuantNodes现有架构天然支持的Agent能力：

```
1. Pipeline作为Agent执行计划的持久化表示
2. BaseNode的状态追踪天然是执行日志的基础
3. Parallel节点天然支持多Agent并行协作
4. IfNode/WhileNode天然支持Agent的条件判断和循环迭代
5. MapNode天然支持Agent的批量处理（多股票/多周期）
6. 序列化系统使得整个Agent工作流可以保存/恢复/分享
7. 缓存系统支持长时运行的记忆
```

### 1.3 已就位基础设施

| 组件 | 状态 | 说明 |
|------|------|------|
| **BaseNode** | 已实现 | 统一节点抽象，状态管理，统计收集 |
| **Pipeline原语** | 已实现 | Pipeline/Parallel/Join/IfNode/MapNode/WhileNode |
| **序列化系统** | 已实现 | Serializable，纯逻辑序列化/反序列化 |
| **CodeSandbox** | 已实现 | AST安全校验，危险模式检测 |
| **缓存系统** | 已实现 | CacheManager，节点级缓存 |
| **因子引擎** | 已实现 | 97+算子，FactorNode体系 |
| **回测引擎** | 已实现 | BacktestNode基类 |
| **多数据源** | 已实现 | ClickHouse/DuckDB/MySQL/CSV/Parquet |
| **AI基础模块** | 已实现 | LLM Provider集成，策略生成 |

---

## 二、核心设计模式（量化专属）

### 模式 1：文件系统优先的可移植研究工作区

**来源: agentic-stack .agent/ 规范**

```
.quantresearch/
├── RESEARCH_NOTES.md      # 人工编辑的研究心得（永不自动覆盖）
├── FACTOR_LIBRARY.md      # 已验证的因子库（自动维护）
├── STRATEGY_CANVAS.md     # 策略画布（设计文档）
│
├── memory/                # 分层记忆系统
│   ├── episodic/          # 情节记忆：每次研究运行的完整记录
│   ├── semantic/          # 语义记忆：提炼的模式库（经人工审核）
│   └── working/           # 工作记忆：当前会话状态（可丢弃）
│
├── skills/                # 技能注册（渐进式披露）
│   ├── _manifest.jsonl    # 技能索引（始终加载）
│   ├── factor_discovery/  # 因子发现技能
│   ├── backtest_runner/   # 回测运行技能
│   └── risk_analysis/     # 风险分析技能
│
├── protocols/             # 安全协议（沙箱执行依据）
│   ├── permissions.md     # 三级权限（允许/需审批/禁止）
│   └── tool_schemas/      # 工具调用Schema（OpenAI Function格式）
│
└── research/              # 研究产物（可git版本控制）
    ├── factor_001/        # 研究工作区
    └── strategy_alpha/    # 策略工作区
```

**量化价值**: 研究工作可在同事间、机器间完整迁移，换环境不丢失研究上下文。


### 模式 2：MessageBus解耦的Agent循环

**来源: nanobot MessageBus架构**

```
                 ┌──────────────────┐
                 │  ResearchBus    │
                 │  ┌────────────── │
                 │  │ inbound Q    │ ← 自然语言指令
                 │  ├────────────── │
                 │  │ outbound Q   │ → 分析报告/图表
                 │  └────────────── │
                 └────────┬─────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ ResearchAgent │ │ BacktestAgent │ │ RiskAgent     │
│ 因子发现      │ │ 回测执行      │ │ 风险分析      │
│ 数据探索      │ │ 参数优化      │ │ 归因分析      │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                  │                   │
        └──────────────────┼───────────────────┘
                           │
                    ┌──────┴──────┐
                    │ Node执行层  │ ← 复用现有BaseNode/Pipeline
                    └─────────────┘
```

**量化价值**: Agent之间通过事件总线通信，解耦研究逻辑和执行逻辑，支持异步/批量/分布式研究。

### 模式 3：ResearchState + 检查点持久化执行

**来源: LangGraph StateMachine**

```python
State定义:
{
  "research_id": str,           # 研究ID
  "intent": str,                # 用户原始意图
  "data_source": str,           # 数据源位置
  "factor_candidates": List,    # 候选因子
  "validated_factors": List,    # 已验证因子
  "backtest_results": Dict,     # 回测结果
  "research_notes": List,       # 研究笔记
  "risk_assessment": Dict,      # 风险评估
  "next_action": str,           # 下一步动作
  "iteration_count": int,       # 迭代次数
  "checkpoint": str,            # 检查点位置
}

Checkpoint存储位置: .quantresearch/checkpoints/<research_id>/
  - state_001.jsonl            # 第1次迭代状态
  - state_002.jsonl            # 第2次迭代状态
  - ...                        # 可随时回溯/重启
```

**量化价值**: 长达数小时的因子挖掘/回测任务可断点续跑，程序崩溃不丢失进度。

### 模式 4：角色化Agent团队协作

**来源: CrewAI Role-based Agent Team**

```
              ┌─────────────────────────┐
              │   ResearchDirector      │ ← 研究主管
              │   - 研究计划制定        │
              │   - 任务分配/验收       │
              └──────────┬──────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ FactorAnalyst   │ │ BacktestEngineer│ │ RiskManager     │
│ - 因子探索      │ │ - 回测执行      │ │ - 过拟合检测    │
│ - 因子合成      │ │ - 参数扫描      │ │ - 鲁棒性验证    │
│ - IC分析        │ │ - 绩效归因      │ │ - 风险警示      │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                     ┌───────┴───────┐
                     │ ReportWriter  │ ← 输出研究报告
                     └───────────────┘
```

**量化价值**: 分工明确的Agent团队模仿真实量化研究团队工作流，结果更可信。

### 模式 5：三阶段沙箱执行模型

**来源: OpenAI Agents Sandbox, LangGraph Human-in-loop**

```
Stage 1: Tool Pre-validation
  ┌───────────────────────────────────┐
  │  调用前校验                       │
  │   · 参数Schema验证                │
  │   · 权限匹配（permissions.md）    │
  │   · 危险操作检测                  │
  └───────────────────────────────────┘

Stage 2: Sandboxed Execution
  ┌───────────────────────────────────┐
  │  隔离执行环境                     │
  │   · 内存限制                      │
  │   · 时间限制（超时终止）          │
  │   · 资源使用监控                  │
  │   · 输出大小限制                  │
  └───────────────────────────────────┘

Stage 3: Result Audit (Human-in-loop)
  ┌───────────────────────────────────┐
  │  结果后审计                       │
  │   · 过拟合检测（Sharpe突变检查）  │
  │   · 未来函数检测                  │
  │   · 过拟合风险评级                │
  │   · 需人工确认 → 高亮            │
  └───────────────────────────────────┘
```

**量化价值**: 防止Agent生成的"看起来很好"但实际上过拟合/未来函数的危险策略。

---

## 三、整体架构设计

### 3.1 目录结构设计

```
QuantNodes/
└── agent/                    # Agent子系统（新增）
    ├── __init__.py
    ├── bus.py               # ResearchBus 消息总线
    ├── state.py             # ResearchState 状态管理 + Checkpoint
    ├── loop.py              # Agent 执行主循环
    ├── context.py           # 上下文构建器
    ├── executor.py          # Pipeline执行器
    │
    ├── agents/              # 具体Agent实现
    │   ├── base.py         # BaseAgent 基类
    │   ├── director.py     # ResearchDirector 主管
    │   ├── factor_analyst.py
    │   ├── backtest_engineer.py
    │   ├── risk_manager.py
    │   └── report_writer.py
    │
    ├── tools/              # Agent工具库（统一Schema）
    │   ├── base.py         # BaseTool 基类
    │   ├── registry.py     # 工具注册表 + 渐进式披露
    │   ├── data_tools.py   # 数据查询工具
    │   ├── factor_tools.py # 因子计算工具
    │   ├── backtest_tools.py
    │   ├── risk_tools.py
    │   └── pipeline_tools.py  # Pipeline编辑工具
    │
    ├── memory/             # 记忆系统
    │   ├── base.py         # MemoryStore 基类
    │   ├── episodic.py     # 情节记忆
    │   ├── semantic.py     # 语义记忆
    │   ├── personal.py     # 个人偏好
    │   └── dream.py        # 夜间记忆提炼
    │
    ├── sandbox/            # 量化研究安全沙箱
    │   ├── executor.py     # 沙箱执行器
    │   ├── audit.py        # 结果审计器
    │   └── permissions.py  # 权限配置
    │
    ├── protocols/          # 协议/Schema
    │   ├── tool_schemas.py
    │   └── message_schemas.py
    │
    └── ui/                 # 用户交互界面
        ├── cli.py          # 命令行交互
        └── streamlit.py    # Streamlit UI组件
```


### 3.2 与现有Pipeline架构的集成点

```
┌─────────────────────────────────────────────────────────────────────┐
│ 集成方式：Agent作为特殊的Node类型，内部包含Pipeline执行能力        │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────┐
                    │   BaseNode             │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴─────────────┐
                    │ Pipeline/Parallel/...   │ ← 现有节点
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴─────────────┐
                    │ AgentNode (新增)        │ ← 新节点类型
                    │                         │
                    │ · 持有 ResearchState     │
                    │ · 包含 MessageBus        │
                    │ · 可编排子Agent执行      │
                    │ · 可序列化保存研究计划   │
                    └─────────────────────────┘
```

### 3.3 使用示例

```python
pipeline = (
    DataSourceNode(universe="A股全市场", start="2020-01-01")
    >> AgentNode(
        agent_type="factor_discovery",
        goal="发现5个有效的量价类alpha因子",
        constraints={"min_ic": 0.05, "min_sharpe": 1.5},
        max_iterations=50,
        enable_checkpoint=True
    )
    >> BacktestNode()
    >> RiskAuditNode()
)

result = pipeline.execute()
```

---

## 四、核心API接口定义

### 4.1 AgentNode 核心 API

```python
@serializable
class AgentNode(BaseNode):
    """
    智能Agent节点 - 可自主执行研究任务
    
    Examples:
        >>> AgentNode(
        ...     agent_type="factor_discovery",
        ...     goal="发现3个高IC值的量价因子",
        ...     constraints={"min_ic": 0.05, "min_sharpe": 1.5},
        ...     max_iterations=50,
        ...     enable_checkpoint=True,
        ... )
    """
    
    def __init__(
        self,
        agent_type: str,
        goal: str,
        constraints: Dict[str, Any] = None,
        team_config: Dict[str, Any] = None,
        max_iterations: int = 100,
        enable_checkpoint: bool = True,
        checkpoint_interval: int = 5,
        work_dir: str = None,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            agent_type: Agent类型: factor_discovery | backtest_suite | risk_audit | strategy_design
            goal: 研究目标（自然语言）
            constraints: 约束条件字典
            team_config: 团队配置（覆盖默认角色）
            max_iterations: 最大迭代次数
            enable_checkpoint: 是否启用检查点
            checkpoint_interval: 每N次迭代存检查点
            work_dir: 研究工作目录，默认 .quantresearch
        """
```

---

## 五、分阶段实施路线图

| 阶段 | 周期 | 核心工作 | 交付物 |
|------|------|----------|--------|
| **Phase 1** | 1周 | 基础框架搭建: AgentNode基类、ResearchState、文件系统记忆层 | 可序列化运行的AgentNode骨架 |
| **Phase 2** | 2周 | 工具系统 + 单Agent: BaseTool抽象、工具注册表、因子分析Agent/回测Agent/风控Agent | 单Agent可独立执行研究任务 |
| **Phase 3** | 2周 | 多Agent协作 + 沙箱: ResearchDirector、团队编排、三阶段沙箱执行、Human-in-loop | 完整的多Agent研究工作流 |
| **Phase 4** | 1周 | Streamlit UI + 文档: 研究仪表盘、因子库浏览器、研究过程回放 | 完整可用的产品 |
| **总计** | **6周** | | |

---

## 六、关键决策点（待确认）

| 决策项 | 推荐方案 | 备选方案 | 影响 |
|--------|----------|----------|------|
| 记忆持久化方案 | 纯文件系统 | SQLite | 可git、人类可读、零运维 |
| 多Agent协作模式 | CrewAI角色委托 | LangGraph图编排 | 更简单、更快、适配量化团队分工 |
| 沙箱执行方案 | 现有CodeSandbox扩展 | 独立子进程 | 复用现有代码、降低维护成本 |
| 配置驱动方式 | Python装饰器 + YAML | 仅Python | 简单用装饰器，复杂用YAML |
| Streamlit集成 | 扩展现有AI页面 | 独立新页面 | 保持UI一致性、减少重复开发 |

---

## 七、风险与前置依赖

| 风险项 | 影响 | 缓解措施 |
|--------|------|----------|
| LLM Token成本过高 | 高 | 渐进式技能披露 + 会话自动压缩 + 本地开源模型备选 |
| Agent生成策略过拟合 | 高 | 三阶段沙箱审计 + 过拟合检测算法 + 强制Human-in-loop |
| 长时运行任务稳定性 | 中 | Checkpoint机制 + 自动重试 + 幂等工具设计 |
| 多Agent编排复杂度 | 中 | 从2个Agent开始，逐步增加，避免过度设计 |

