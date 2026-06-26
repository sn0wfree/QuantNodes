# 多智能体 WorkflowTool 设计文档

> 状态: **设计阶段**  
> 关联: [13-Agent架构设计.md](13-Agent架构设计.md), [15-可选依赖安装指南.md](15-可选依赖安装指南.md)  
> 上游依赖: `nanobot-ai>=0.2.1`（SubagentManager, Tool 基类, AgentLoop）

---

## 一、背景与动机

### 1.1 现状

QuantNodes 已有两种多 agent 模式：

| 模式 | 实现 | 优势 | 局限 |
|------|------|------|------|
| **LLM 自主调度** | SOUL.md + `spawn` | 灵活，LLM 按意图委派 | 无确定性，难测试 |
| **程序式编排** | `AlphaGptWorkflow` | 确定性，可测试 | 绕过 nanobot，LLM 不可调 |

两者独立运作，无法组合。

### 1.2 目标

构建统一框架，让 LLM 自主调度和程序式编排可以**混合使用**：

```
用户: "研究动量因子"
  → LLM 决策: run_workflow(alpha-gpt, {objective: "momentum"})   ← 程序式 pipeline
  → 拿到 top formulas
  → LLM 决策: spawn(risk-manager, "审查这些 formula 的风险")     ← LLM 自主调度
  → 综合结果回复
```

**核心思路**: 程序式做 pipeline 内部（确定性、可测试），LLM 做 pipeline 之间（灵活、上下文感知）。

---

## 二、架构设计

### 2.1 两种 Agent 类型

```
┌─ AgentLoop (主 agent) ────────────────────────────────────┐
│                                                           │
│  可用工具:                                                 │
│    spawn          → 创建 Full Subagent (重型)              │
│    run_workflow   → 调用 WorkflowTool (内含轻量 StepAgent)  │
│    factor, backtest, alpha_evaluate, ...                   │
└────────┬──────────────────────────────┬───────────────────┘
         │                              │
         ▼                              ▼
┌── SubagentManager ──────┐  ┌── WorkflowTool ──────────────┐
│  (不变)                 │  │                              │
│  Full AgentRunner       │  │  StepAgent 1: idea-gen       │
│  完整 ToolRegistry      │  │    prompt → LLM → parse      │
│  多轮对话               │  │  StepAgent 2: formula-trans  │
│  独立 workspace         │  │    prompt → LLM → parse      │
│                         │  │  StepAgent 3: evaluator      │
│  factor-analyst         │  │    tool_executor (跳过 LLM)  │
│  backtest-engineer      │  │  StepAgent 4: reflector      │
│  risk-manager           │  │    prompt → LLM → parse      │
│                         │  │  StepAgent 5: critic         │
│  结果通过 bus 回报       │  │    prompt → LLM → parse      │
└─────────────────────────┘  │                              │
                             │  结构化结果 + JSON 文件       │
                             └──────────────────────────────┘
```

### 2.2 StepAgent vs Full Subagent

| 维度 | StepAgent (WorkflowTool 内部) | Full Subagent (SubagentManager) |
|------|------------------------------|----------------------------------|
| 调用方式 | 单次 LLM call，无循环 | AgentRunner 多轮循环 |
| 工具 | 无（或直接调 Python 函数） | 完整 ToolRegistry |
| 系统 prompt | 指令写在 user prompt 里 | Jinja2 模板 + skills + workspace |
| 会话历史 | 无状态，每次从零开始 | 累积多轮 |
| 执行方式 | 同步 | asyncio.Task |
| Hook/Streaming | 无 | 7 个生命周期钩子 |
| 用途 | pipeline 步骤 | 独立专家 |

### 2.3 数据流

```
WorkflowTool.execute(workflow="alpha-gpt", config={objective: "momentum"})
  │
  ├─ 1. 从 WorkflowRegistry 拿到 WorkflowSpec
  ├─ 2. 确定 LLM provider (spec 覆盖 > 主 agent provider)
  ├─ 3. 构造 State 对象
  ├─ 4. 逐步执行:
  │     for step_spec in spec.steps:
  │       step = StepAgent(step_spec, llm_client)
  │       records = step.run(**config, state=state)  ← 同步, to_thread 包装
  │       state.update(step_spec.output_key, records)
  │
  ├─ 5. result_builder(state, config) → 完整结果 dict
  ├─ 6. 存 JSON 到 .agent/results/alpha-gpt-{timestamp}.json
  └─ 7. 返回摘要字符串给 LLM
```

---

## 三、文件结构

```
QuantNodes/agent/workflows/
├── __init__.py                  # public API 导出
├── step_agent.py                # StepAgent + StepAgentSpec + ParseResult
├── parsers.py                   # 薄封装，复用 research/quant_alpha/llm/parser.py
├── registry.py                  # WorkflowRegistry + WorkflowSpec + REGISTRY 单例
├── tool.py                      # WorkflowTool (nanobot Tool 子类)
└── implementations/
    ├── __init__.py
    └── alpha_gpt.py             # AlphaGptWorkflow 用 StepAgent 重写
```

---

## 四、核心接口

### 4.1 StepAgentSpec

```python
@dataclass
class StepAgentSpec:
    """单个 pipeline 步骤的规格定义。"""
    agent_id: str                                    # "alpha-gpt-idea-generator"
    prompt_builder: Callable[..., str]               # (**ctx) -> prompt str
    output_parser: Callable[[str], ParseResult]      # (raw) -> ParseResult
    output_key: str                                  # "ideas", "formulas", ...
    record_factory: Callable[[dict], Any]            # (dict) -> Record
    tool_executor: Callable[..., list[Any]] | None = None  # evaluator 用
    max_retries: int = 2                             # 解析失败重试次数
```

### 4.2 StepAgent

```python
class StepAgent:
    """轻量级单次 LLM 步骤。无工具循环、无会话历史。"""

    def __init__(self, spec: StepAgentSpec, llm_client=None): ...

    def run(self, **context) -> list[Any]:
        """执行: prompt → LLM → parse(带重试+修复) → records"""
        if self.spec.tool_executor:
            return self.spec.tool_executor(**context)
        result = self._run_with_retry(**context)
        if not result.ok:
            return []
        items = result.data.get(self.spec.output_key, [])
        return [self.spec.record_factory(item) for item in items]

    def _run_with_retry(self, **context) -> ParseResult:
        """重试逻辑: 解析失败时注入完整 raw + error 到 prompt。"""
        prompt = self.spec.prompt_builder(**context)
        for attempt in range(self.spec.max_retries + 1):
            raw = self._call_llm(prompt)
            result = self.spec.output_parser(raw)
            if result.ok:
                return result
            if attempt < self.spec.max_retries:
                prompt = self.spec.prompt_builder(
                    **context,
                    _prev_error=result.error,
                    _prev_raw=result.raw,    # 完整传递，不截断
                )
        return result

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM: client.complete() → callable → mock"""
        ...
```

### 4.3 WorkflowSpec

```python
@dataclass
class WorkflowSpec:
    """一个完整 workflow 的规格定义。"""
    name: str                                        # "alpha-gpt"
    description: str                                 # 给 LLM 看的描述
    steps: list[StepAgentSpec]                       # 有序步骤列表
    state_factory: Callable[[], Any]                 # () -> State 对象
    result_builder: Callable[[Any, dict], dict]      # (state, config) -> result dict
    provider: str | None = None                      # 可选: 覆盖 LLM provider
    model: str | None = None                         # 可选: 覆盖 model
```

### 4.4 WorkflowRegistry

```python
class WorkflowRegistry:
    """注册表，管理所有可用 workflow。"""

    def register(self, spec: WorkflowSpec): ...
    def get(self, name: str) -> WorkflowSpec | None: ...
    def list_all(self) -> list[dict]: ...            # [{name, description}, ...]
    def build_llm_description(self) -> str: ...      # 拼接给 LLM 的 tool description

REGISTRY = WorkflowRegistry()                        # 模块级单例
```

### 4.5 WorkflowTool

```python
class WorkflowTool(Tool):
    """nanobot Tool 子类，暴露 run_workflow 给 LLM。"""

    name = "run_workflow"
    _scopes = {"core", "subagent"}

    def __init__(self, llm_client, model=None, results_dir=None): ...

    async def execute(self, workflow: str, config: dict | None = None, **kwargs) -> str:
        """1. 查 registry → 2. 确定 provider → 3. 构造 state
           → 4. 逐步 StepAgent.run() → 5. 存 JSON → 6. 返回摘要"""
        ...
```

---

## 五、重试机制

### 5.1 设计

StepAgent 解析失败时，将完整 raw 输出 + error 信息注入到重试 prompt 中，让 LLM 修正格式。

### 5.2 流程

```
第 1 次 prompt:
  "Read .agent/agents/alpha-gpt-idea-generator.md.
   Generate 8 alpha ideas for objective='momentum'.
   Output STRICT JSON only."

LLM 返回: "Here are some ideas: {ideas: [...]}"  ← 解析失败

第 2 次 prompt (注入错误信息):
  "Read .agent/agents/alpha-gpt-idea-generator.md.
   Generate 8 alpha ideas for objective='momentum'.
   Output STRICT JSON only.

   [SYSTEM: Your previous response was not valid JSON.
   Error: Cannot parse JSON after 2 layers (full raw in ParseResult.raw)
   Your full previous response:
   Here are some ideas: {ideas: [...]}
   Please output ONLY a JSON object with no additional text.]"
```

### 5.3 关键约束

- **不截断 raw**: `_prev_raw=result.raw` 传递完整输出，不用 `[:500]` 等截断
- **最大重试次数**: 默认 2 次（总共 3 次尝试）
- **prompt_builder 兼容**: 所有 prompt_builder 接受可选的 `_prev_error` 和 `_prev_raw` 参数

---

## 六、LLM Provider 来源

### 6.1 优先级

```
workflow 配置的 provider/model  →  主 agent 的 provider/model  →  抛错
```

### 6.2 实现

```python
# WorkflowTool.__init__ 时传入主 agent 的 provider
wt = WorkflowTool(
    llm_client=self._loop.provider,  # AgentLoop.provider (loop.py:214)
    model=getattr(self._loop, 'model', None),
)

# execute 时按 spec 覆盖
client = spec.provider or self._llm_client
model = spec.model or self._model
```

### 6.3 动态更新

Provider 在构造时确定，**不支持运行时动态更新**。切换模型需重启服务。

---

## 七、结果返回

### 7.1 双层返回

| 层 | 内容 | 接收方 |
|---|------|--------|
| 摘要 | JSON 字符串: status, summary, result_file, top_formulas[:5] | LLM (tool 返回值) |
| 完整 | JSON 文件: `.agent/results/{workflow}-{timestamp}.json` | 后续分析/人工查看 |

### 7.2 摘要格式

```json
{
  "status": "completed",
  "summary": {
    "total_evaluated": 50,
    "successful": 42,
    "selected": 10,
    "avg_ir": 1.42,
    "best_ir": 2.05
  },
  "result_file": ".agent/results/alpha-gpt-20260627-143022.json",
  "top_formulas": [
    {"formula": "rank(-ts_mean(returns, 20))", "ir": 2.05, "ic_mean": 0.045},
    ...
  ]
}
```

---

## 八、AlphaGptWorkflow 移植

### 8.1 5 个 StepAgentSpec

| 步骤 | agent_id | LLM? | output_key | 特殊逻辑 |
|------|----------|------|------------|----------|
| 1. IdeaGenerator | `alpha-gpt-idea-generator` | ✅ | `ideas` | 注入 prev_reflection |
| 2. FormulaTranslator | `alpha-gpt-formula-translator` | ✅ | `formulas` | operator 白名单校验 |
| 3. Evaluator | `alpha-gpt-evaluator` | ❌ | `evaluations` | tool_executor 调 AlphaEvaluateTool |
| 4. Reflector | `alpha-gpt-reflector` | ✅ | `formula_feedback` | 注入 evaluations |
| 5. Critic | `alpha-gpt-critic` | ✅ | `final_pool` | 仅最终轮执行 |

### 8.2 复用关系

```
QuantNodes/agent/workflows/implementations/alpha_gpt.py
  ├── import StepAgentSpec, StepAgent    ← from ..step_agent
  ├── import validators                  ← from ..parsers (re-export of research/quant_alpha/llm/parser.py)
  ├── import AlphaGptState, Records      ← from research/quant_alpha/workflow/state.py
  ├── import AlphaEvaluateTool           ← from agent/tools/alpha_evaluate.py
  └── import REGISTRY, WorkflowSpec      ← from ..registry

不修改原 research/quant_alpha/workflow/alpha_gpt.py，保持向后兼容。
```

### 8.3 State 更新映射

```python
STATE_FIELD_MAP = {
    "ideas": "all_ideas",
    "formulas": "all_formulas",
    "evaluations": "all_evaluations",
    "formula_feedback": "all_reflections",
    "final_pool": "critic_output",
}
```

---

## 九、与现有系统的关系

### 9.1 不修改 nanobot 上游

所有改动在 QuantNodes 侧完成：
- `WorkflowTool` 直接实例化后 `registry.register(tool)`，不走 `ToolLoader.create(ctx)` 工厂
- LLM provider 构造时传入，不改 `ToolContext` 加 provider 字段

### 9.2 不修改原 AlphaGptWorkflow

- 原 `research/quant_alpha/workflow/alpha_gpt.py` 保持不动
- 新文件 import 原模块的 state/dataclass/validators
- 原有测试不受影响

### 9.3 与 spawn 的关系

| 场景 | 选择 |
|------|------|
| 固定 pipeline（alpha-gpt 5 轮迭代） | `run_workflow` |
| 单领域专家任务（factor-analyst 做 IC 测试） | `spawn` |
| pipeline 结果需要专家审查 | `run_workflow` → 拿到结果 → `spawn` |

---

## 十、测试策略

### 10.1 单元测试

| 测试类 | 覆盖范围 |
|--------|----------|
| `TestStepAgent` | 正常执行、重试+修复、全部失败、tool_executor 跳过 |
| `TestWorkflowRegistry` | 注册、查询、列表、描述生成 |
| `TestWorkflowTool` | mock 执行、结果文件、摘要返回、未知 workflow |

### 10.2 集成测试

- 用 mock LLM 跑完整 alpha-gpt workflow（5 步 × 1 轮）
- 验证 `.agent/results/` 下生成 JSON 文件
- 验证返回摘要包含 top_formulas

### 10.3 回归测试

- 原 `tests/quant_alpha/` 测试不受影响（不修改原代码）
- 原 `tests/agent/` 测试不受影响

---

## 十一、实施阶段

| Stage | 文件 | 依赖 |
|-------|------|------|
| 1 | `step_agent.py` | 无 |
| 2 | `parsers.py` | 无 |
| 3 | `registry.py` | Stage 1 |
| 4 | `tool.py` | Stage 1+3 |
| 5 | `implementations/alpha_gpt.py` | Stage 1+2+3 |
| 6 | `tools/__init__.py` + `nanobot_bridge.py` | Stage 4 |
| 7 | `.agent/SOUL.md` | Stage 4 |
| 8 | `tests/agent/test_workflow_tool.py` | 全部 |

Stage 1 和 2 可并行。

---

## 十二、后续扩展

| 方向 | 说明 |
|------|------|
| **新 workflow** | `risk-review`, `strategy-sweep` 等，只需定义 WorkflowSpec + StepAgentSpec |
| **spawn 链式扩展** | `on_complete` 回调，spawn 完自动触发下一个 spawn |
| **StepAgent 升级为 Full Agent** | 包装为 `AgentRunner.run(max_iterations=1)`，加工具和系统 prompt |
| **并行步骤** | 无依赖的步骤用 `asyncio.gather` 并行执行 |
| **workflow 可视化** | 在 WebUI 展示 workflow 执行进度和中间结果 |
