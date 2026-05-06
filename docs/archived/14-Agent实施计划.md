# QuantNodes Agent 系统实施计划

**版本**: v1.0  
**创建日期**: 2026-04-29  
**状态**: Phase 1 已完成 ✅ | Phase 2 已完成 ✅ | Phase 3 已完成 ✅ | Phase 4 待开始  
**作者**: sn0wfree

---

## 一、设计原则与目标

### 1.1 核心设计原则

1. **完整复刻nanobot架构** - 复用经过生产验证的Agent核心设计
2. **渐进式并发** - Phase 1-2串行，Phase 3+平滑升级到完整并发
3. **复用现有组件** - LLM、Sandbox、StrategyGenerator直接复用
4. **标准MCP协议** - 通过MCP桥接llmwikify知识管理
5. **回测安全优先** - Phase 1-3全局锁，后期改为Docker隔离并发
6. **Polars新架构** - Phase 3 迁移到Polars + 配置文件驱动（v2.0）

### 1.2 实施目标

| 阶段 | 交付目标 |
|------|---------|
| Phase 1 | 最小可运行Agent核心，支持对话+工具调用 |
| Phase 2 | 策略生成→验证→回测完整闭环 |
| Phase 3 | llmwikify知识沉淀 + 跨会话并发 |
| Phase 4 | 技能系统 + 完整并发 + 记忆增强 |

---

## 二、目录结构总览

```
QuantNodes/agent/                    🆕 新建顶级目录
├── __init__.py                     🆕 Agent对外API
├── core/                           🆕 核心引擎（复刻nanobot）
│   ├── __init__.py
│   ├── loop.py                     🆕 Agent主消息循环
│   ├── runner.py                   🆕 工具执行循环（核心）
│   ├── context.py                  🆕 上下文/Prompt构建
│   ├── memory.py                   🆕 记忆存储（简化版）
│   ├── hook.py                     🆕 执行钩子系统
│   └── autocompact.py              🆕 会话历史压缩
├── tools/                          🆕 工具系统
│   ├── __init__.py
│   ├── base.py                     🆕 Tool基类 + Schema验证
│   ├── registry.py                 🆕 工具注册与调度
│   ├── echo.py                     🆕 测试工具（Phase 1）
│   ├── sandbox.py                  🔄 封装现有CodeSandbox
│   ├── pipeline.py                 🆕 Pipeline构建验证
│   ├── strategy.py                 🔄 封装StrategyGenerator
│   ├── backtest.py                 🆕 回测运行工具
│   ├── factor.py                   🆕 因子分析工具
│   └── mcp.py                      🆕 MCP工具桥（Phase 3）
├── bus/                            🆕 消息总线
│   ├── __init__.py
│   ├── events.py                   🆕 Inbound/Outbound消息
│   └── queue.py                    🆕 异步队列
├── session/                        🆕 会话管理
│   ├── __init__.py
│   └── manager.py                  🆕 会话持久化
├── providers/                      🆕 LLM Provider适配层
│   ├── __init__.py
│   ├── base.py                     🆕 nanobot风格Provider基类
│   └── quantnodes.py               🆕 适配现有LLMClientBase
├── skills/                         🆕 技能系统（Phase 4）
│   ├── __init__.py
│   ├── loader.py                   🆕 渐进式加载器
│   ├── strategy_design/            🆕 策略设计技能
│   └── factor_research/            🆕 因子研究技能
├── wiki/                           🆕 llmwikify集成（Phase 3）
│   ├── __init__.py
│   └── client.py                   🆕 MCP客户端
├── templates/                      🆕 Prompt模板
│   └── agent/
│       ├── identity.md
│       ├── system_prompt.md
│       └── tools_description.md
├── utils/                          🆕 工具函数
│   ├── __init__.py
│   ├── helpers.py                  🆕 文本/Token处理
│   ├── prompt_templates.py         🆕 模板渲染
│   └── gitstore.py                 🆕 Git版本控制（简化）
├── cli/                            🆕 命令行界面
│   ├── __init__.py
│   └── main.py                     🆕 CLI入口
└── web/                            🆕 Web界面（Streamlit）
    ├── __init__.py
    └── app.py                      🆕 Web入口
```

**图例**:
- 🆕 新建文件
- 🔄 封装/重构现有文件

---

## 三、阶段详细任务与验收标准

### Phase 1: 核心框架复刻（1-2周）

**目标**: 最小可运行Agent，支持对话 + 工具调用

| 序号 | 任务 | 文件 | 预估代码量 | 依赖 | 验收标准 | 状态 |
|------|------|------|-----------|------|----------|------|
| 1.1 | 消息总线 | `bus/events.py`<br>`bus/queue.py` | ~100行 | 无 | Inbound/Outbound消息可入队出队 | ✅ 已完成 |
| 1.2 | 会话管理 | `session/manager.py` | ~200行 | 1.1 | 会话可持久化到文件，历史可回放 | ✅ 已完成 |
| 1.3 | 工具基类 | `tools/base.py`<br>`tools/registry.py` | ~500行 | 无 | Schema验证工作，工具可注册和执行 | ✅ 已完成 |
| 1.4 | 测试工具 | `tools/echo.py` | ~50行 | 1.3 | EchoTool正确返回输入 | ✅ 已完成 |
| 1.5 | Provider适配 | `providers/base.py`<br>`providers/quantnodes.py` | ~400行 | 无 | 适配现有`LLMClientBase`，支持chat调用 | ✅ 已完成 |
| 1.6 | 上下文构建 | `core/context.py`<br>`core/hook.py` | ~300行 | 无 | 系统Prompt + 历史消息可正确构建 | ✅ 已完成 |
| 1.7 | 执行循环 | `core/runner.py` | ~800行 | 1.3, 1.5 | 单轮对话 + 工具调用闭环工作 | ✅ 已完成 |
| 1.8 | 主循环 | `core/loop.py`<br>`core/autocompact.py` | ~600行 | 1.1, 1.2, 1.7 | 消息总线驱动，并发接口预留 | ✅ 已完成 |
| 1.9 | 记忆系统 | `core/memory.py` | ~200行 | 1.2 | 基础文件存储，不做Dream/Consolidator | ✅ 已完成 |
| 1.10 | 工具函数 | `utils/helpers.py`<br>`utils/prompt_templates.py` | ~200行 | 无 | 文本截断、模板渲染工作 | ✅ 已完成 |
| 1.11 | Prompt模板 | `templates/agent/*.md` | ~100行 | 无 | 量化研究专用System Prompt | ✅ 已完成 |
| 1.12 | CLI入口 | `cli/main.py` | ~150行 | 1.8 | `python -m QuantNodes.agent.cli` 可对话 | ✅ 已完成 |
| 1.13 | 单元测试 | `tests/agent/test_*.py` | ~300行 | 1.1-1.12 | 覆盖率 ≥ 70% | ✅ 已完成 |

**Phase 1 总计**: ~3900行代码

**最终验收**:
```bash
$ python -m QuantNodes.agent.cli
> echo "hello world"
hello world
> 生成一个动量策略
<Agent调用StrategyGenerator工具并返回结果>
```

---

### Phase 2: QuantNodes工具集（2周）

**目标**: 策略生成 → 验证 → 回测 完整闭环

| 序号 | 任务 | 文件 | 预估代码量 | 依赖 | 验收标准 | 状态 |
|------|------|------|-----------|------|----------|------|
| 2.1 | 沙箱工具 | `tools/sandbox.py` | ~100行 | Phase 1 | 可安全执行Python代码 | ✅ 已完成 |
| 2.2 | Pipeline工具 | `tools/pipeline.py` | ~150行 | 2.1 | 可构建、验证FactorPipeline | ✅ 已完成 |
| 2.3 | 策略生成工具 | `tools/strategy.py` | ~150行 | 2.2 | 封装StrategyGenerator | ✅ 已完成 |
| 2.4 | 回测运行工具 | `tools/backtest.py` | ~200行 | 2.3 | 调用回测引擎，返回结果摘要 | ✅ 已完成 |
| 2.5 | 因子分析工具 | `tools/factor.py` | ~150行 | 2.2 | IC分析、相关性分析 | ✅ 已完成 |
| 2.6 | 端到端测试 | `tests/agent/test_e2e.py` | ~100行 | 2.1-2.5 | 完整流程测试通过 | ✅ 已完成 |
| 2.7 | Web界面 | `web/app.py` (Streamlit) | ~300行 | 2.1-2.5 | 浏览器可用Agent | ✅ 已完成 |

**Phase 2 总计**: ~1150行代码

**最终验收**:
```
用户: "帮我生成一个动量因子策略，验证代码，然后回测2020-2023年"
Agent:
  1. 调用 strategy_generator 生成代码
  2. 调用 code_sandbox 验证安全性
  3. 调用 backtest_runner 运行回测
  4. 返回回测报告
```

---

### Phase 3: llmwikify知识沉淀（1周）

**目标**: 通过MCP连接llmwikify，自动沉淀研究结果 + Polars架构迁移

| 序号 | 任务 | 文件 | 预估代码量 | 依赖 | 验收标准 | 状态 |
|------|------|------|-----------|------|----------|------|
| 3.1 | MCP工具桥 | `tools/mcp.py` | ~150行 | Phase 1 | 可连接外部MCP服务 | ⬜ 待开始 |
| 3.2 | Wiki客户端 | `wiki/client.py` | ~100行 | 3.1 | 可调用llmwikify的MCP接口 | ⬜ 待开始 |
| 3.3 | 自动沉淀Hook | `core/hook.py` 扩展 | ~100行 | 3.2 | 回测完成后自动写Wiki | ⬜ 待开始 |
| 3.4 | Wiki查询工具 | `wiki/query.py` | ~100行 | 3.2 | 可检索历史策略/因子 | ⬜ 待开始 |
| 3.5 | 并发升级 | `core/loop.py` 启用Semaphore=3 | ~50行 | Phase 1 | 跨会话并发工作 | ⬜ 待开始 |

**Phase 3 新架构 (v2.0)**:

| 序号 | 任务 | 文件 | 预估代码量 | 依赖 | 验收标准 | 状态 |
|------|------|------|-----------|------|----------|------|
| 3.6 | Polars算子库 | `operators/time_series.py` 等 | ~400行 | Phase 1 | 20+ 算子可配置 | ✅ 已完成 |
| 3.7 | Config加载器 | `agent/config/loader.py` | ~380行 | Phase 1 | YAML配置解析 | ✅ 已完成 |
| 3.8 | Config执行器 | `agent/config/executor.py` | ~900行 | Phase 1 | 表达式解析+执行 | ✅ 已完成 |
| 3.9 | 算子注册表 | `factor_node/factor_functions.py` | ~3000行 | 3.6 | 300+ 算子注册 | ✅ 已完成 |
| 3.10 | TA-Lib集成 | `operators/talib.py` | ~1960行 | 3.6 | 174 个技术指标 | ✅ 已完成 |
| 3.11 | 回测运行器 | `backtest/config_runner.py` | ~450行 | 3.8 | 权益曲线+绩效统计 | ✅ 已完成 |
| 3.12 | Output保存 | `backtest/config_runner.py` | ~80行 | 3.11 | 信号/交易/权益曲线保存 | ✅ 已完成 |
| 3.13 | Universe过滤 | `agent/config/executor.py` | ~60行 | 3.8 | 预设/文件/列表过滤 | ✅ 已完成 |

**Phase 3 总计**: ~1050行 (v2.0已完成)

**最终验收**:
```
1. 用户完成一次策略回测
2. 回测结果自动写入llmwikify的Strategy页面
3. 用户问"之前有哪些动量策略？"，Agent从Wiki查询返回
```

---

### Phase 4: 技能系统与优化（1-2周）

**目标**: 渐进式技能加载，专门化量化研究技能

| 序号 | 任务 | 文件 | 预估代码量 | 依赖 | 验收标准 | 状态 |
|------|------|------|-----------|------|----------|------|
| 4.1 | 技能加载器 | `skills/loader.py` | ~200行 | Phase 1 | 按需加载技能，Token高效 | ⬜ 待开始 |
| 4.2 | 策略设计技能 | `skills/strategy_design/*.py` | ~300行 | 4.1 | 经典策略模板库 | ⬜ 待开始 |
| 4.3 | 因子研究技能 | `skills/factor_research/*.py` | ~300行 | 4.1 | 因子挖掘、正交化流程 | ⬜ 待开始 |
| 4.4 | Dream系统 | `core/memory.py` 扩展 | ~300行 | Phase 1 | 背景知识处理 | ⬜ 待开始 |
| 4.5 | Consolidator | `core/autocompact.py` 扩展 | ~200行 | Phase 1 | 记忆自动整合 | ⬜ 待开始 |
| 4.6 | 工具级并发 | `core/runner.py` 扩展 | ~100行 | Phase 1 | 只读工具并行执行 | ⬜ 待开始 |

**Phase 4 总计**: ~1400行代码

---

### Phase 5: FactorNode Polars 统一迁移 (8天)

**目标**: 统一到 Polars 技术栈，移除 traits 和 multiprocessing 依赖

**设计文档**: `docs/17-QuantNodes-Polars统一迁移方案.md`

| 序号 | 任务 | 文件 | 预估代码量 | 依赖 | 验收标准 | 状态 |
|------|------|------|-----------|------|----------|------|
| 5.1 | factor_functions_v2.py | `factor_functions_v2.py` | ~600行 | Phase 3 | 20+ 函数可用的 | ✅ 已完成 |
| 5.2 | 改写 quant_nodes_object.py | `quant_nodes_object.py` | ~100行 | 5.1 | 移除traits | ✅ 已完成 |
| 5.3 | 改写 factor.py | `factor.py` | ~100行 | 5.2 | 移除traits | ✅ 已完成 |
| 5.4 | 简化 factor_operation.py | `factor_operation.py` | ~200行 | 5.3 | 移除multiprocessing | ✅ 已完成 |
| 5.5 | 修改 factor_table.py/factor_db.py | 同名文件 | ~150行 | 5.4 | 移除multiprocessing | ✅ 已完成 |
| 5.6 | 统一 __init__.py | `__init__.py` | ~50行 | 5.5 | 清理导出 | ✅ 已完成 |
| 5.7 | 测试验证 | `tests/*.py` | - | 5.6 | 测试通过 | ✅ 已完成 |
| 5.8 | 删除 factor_nodes.py | `factor_nodes.py` | - | 5.7 | 旧OOP层已删除 | ✅ 已完成 |

**Phase 5 总计**: ~1200行代码 (8天) ✅ 已完成

**最终验收**:
```bash
python -m pytest tests/test_factor_functions.py -v  # 全部通过
python -m pytest tests/test_factor_node.py -v     # 全部通过
```

---

## 四、并发设计实现路径（渐进式）

### 4.1 nanobot并发三层架构回顾

```
┌─────────────────────────────────────────────────────────┐
│  全局并发门控 (Semaphore, 默认=3)                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  会话级锁 (per-session Lock)                      │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │  工具级并发 (asyncio.gather)                │  │  │
│  │  │  - 只读工具: 可并发执行                     │  │  │
│  │  │  - 有副作用工具: 串行执行                   │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**核心设计原则**:
- **跨会话并发**：不同用户/会话可以同时使用Agent
- **同会话串行**：同一个对话上下文必须保证顺序一致性
- **工具级并发**：同一轮对话中的多个工具调用可以并行（只读工具）

### 4.2 Phase 1-2: 简化并发

```python
# core/loop.py - 初始化
class AgentLoop:
    def __init__(self, ...):
        # 全局并发=1，实际串行
        self._concurrency_gate = asyncio.Semaphore(1)
        # 会话锁保留接口
        self._session_locks: dict[str, asyncio.Lock] = {}
        # 预留pending queues接口但简化
        self._pending_queues: dict[str, asyncio.Queue] = {}
        # 任务追踪保留
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
```

**特点**: 接口完整，但Semaphore=1，实际串行执行

### 4.3 Phase 3+: 平滑升级

```python
# 只需改配置，无需重构代码
# 并发数 1 → 3
# 工具并发开关: concurrent_tools = True
# 回测仍保留全局锁（后期改为Docker隔离）
```

---

## 五、回测并发演进路径

### 5.1 量化研究场景并发特性分析

| 操作类型 | 耗时 | 可否并发 | 冲突风险 |
|---------|------|----------|----------|
| LLM调用 | 1-10s | ✅ 可并发 | ❌ 无冲突（无状态） |
| 代码沙箱执行 | 0.1-5s | ✅ 可并发 | ❌ 无冲突（隔离） |
| Pipeline构建/验证 | 0.01-1s | ✅ 可并发 | ❌ 无冲突 |
| 因子计算（内存） | 1-30s | ✅ 可并发 | ⚠️ 内存占用 |
| 回测运行 | 10s-30min | ⚠️ 谨慎并发 | ✅ 高冲突（数据库、临时文件） |
| Wiki写入/查询 | 0.1-2s | ✅ 可并发 | ⚠️ Git锁冲突 |

**关键发现**: 回测是并发瓶颈，必须串行化或引入资源隔离。

### 5.2 Phase 1-3: 全局锁保护

```python
# tools/backtest.py
_backtest_lock = asyncio.Lock()

class BacktestTool(Tool):
    concurrency_safe = False
    
    async def execute(self, **kwargs):
        async with _backtest_lock:
            return await self._run_backtest(**kwargs)
```

### 5.3 Phase 4+: Docker隔离并发

```python
# tools/backtest.py 升级
class DockerBacktestTool(Tool):
    concurrency_safe = True  # 现在可以并发了
    
    async def execute(self, **kwargs):
        # 每个回测在独立Docker容器中运行
        container = await docker_client.containers.run(
            "quantnodes/backtest",
            detach=True,
            environment=kwargs
        )
        # ...等待结果
```

---

## 六、依赖关系图

```
Phase 1
├── 1.1 消息总线
├── 1.2 会话管理
├── 1.3 工具基类 → 1.4 测试工具
├── 1.5 Provider适配
├── 1.6 上下文构建
├── 1.7 执行循环 → 依赖 1.3, 1.5
├── 1.8 主循环   → 依赖 1.1, 1.2, 1.7
├── 1.9 记忆系统
├── 1.10 工具函数
├── 1.11 Prompt模板
├── 1.12 CLI入口
└── 1.13 单元测试

Phase 2
├── 2.1 沙箱工具
├── 2.2 Pipeline工具 → 依赖 2.1
├── 2.3 策略生成工具 → 依赖 2.2
├── 2.4 回测运行工具 → 依赖 2.3
├── 2.5 因子分析工具 → 依赖 2.2
├── 2.6 端到端测试
└── 2.7 Web界面

Phase 3
├── 3.1 MCP工具桥
├── 3.2 Wiki客户端 → 依赖 3.1
├── 3.3 自动沉淀Hook → 依赖 3.2
├── 3.4 Wiki查询工具 → 依赖 3.2
└── 3.5 并发升级

Phase 4
├── 4.1 技能加载器
├── 4.2 策略设计技能 → 依赖 4.1
├── 4.3 因子研究技能 → 依赖 4.1
├── 4.4 Dream系统
├── 4.5 Consolidator
└── 4.6 工具级并发
```

---

## 七、关键接口定义

### 7.1 Agent对外API (`agent/__init__.py`)

```python
from .core.loop import AgentLoop
from .bus.events import InboundMessage, OutboundMessage

class Agent:
    """QuantNodes 量化研究Agent"""
    
    def __init__(self, workspace: str, config: dict = None):
        """初始化Agent
        
        Args:
            workspace: 工作目录路径
            config: 配置字典（LLM、并发设置等）
        """
        pass
    
    async def run(self, prompt: str, session_id: str = "default") -> str:
        """运行一次对话
        
        Args:
            prompt: 用户输入
            session_id: 会话ID
            
        Returns:
            Agent回复
        """
        pass
    
    async def chat(self, message: str, session_id: str = "default"):
        """流式对话（生成器）"""
        yield "..."
```

### 7.2 Tool基类接口 (`tools/base.py`)

```python
class Tool(ABC):
    """所有工具的基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema参数定义"""
        pass
    
    @property
    def read_only(self) -> bool:
        """是否为只读工具（可并发）"""
        return False
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """执行工具"""
        pass
```

---

## 八、总工作量估算

| 阶段 | 代码量 | 预计时间 | 关键交付物 |
|------|--------|---------|-----------|
| Phase 1 | ~3900行 | 1-2周 | 可对话的Agent核心 |
| Phase 2 | ~1150行 | 2周 | 策略生成→回测闭环 |
| Phase 3 | ~1050行 | 1周 | Wiki知识沉淀 + Polars v2.0 |
| Phase 4 | ~1400行 | 1-2周 | 技能系统+完整并发 |
| Phase 5 | ~1200行 | 8天 | Polars 统一迁移 (移除traits) |
| **总计** | ~8700行 | **7-9周** | |

---

## 九、待讨论确认事项

1. ✅ **并发方案** - 渐进式并发（Phase 1-2串行，Phase 3+升级）
2. ✅ **初期并发数** - Semaphore=1（全局串行）
3. ✅ **回测并发** - Phase 1-3全局锁，Phase 4+Docker隔离
4. ✅ **回合中注入** - 完整复刻nanobot支持

---

**文档创建完成** | **下一步**: 讨论确认后开始Phase 1实施
