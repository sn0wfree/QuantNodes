# 多智能体 WorkflowTool 设计文档

> 状态: **评审通过，准备实施**  
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

### 2.3 数据流（声明式 + 多轮迭代）

```
WorkflowTool.execute(workflow="alpha-gpt", config={objective: "momentum", iterations: 5})
  │
  ├─ 1. 从 WorkflowRegistry 拿到 WorkflowSpec
  ├─ 2. 确定 LLM provider (v1: 始终用主 agent provider)
  ├─ 3. state = spec.state_factory()
  │
  ├─ 4. 多轮循环:
  │     for round_idx in 1..iterations:
  │       prev_output = None
  │       for step_spec in spec.steps:
  │         if step_spec.skip_on_last and is_last_round:
  │           continue                          ← reflector 最后一轮跳过
  │         step = StepAgent(step_spec, client)
  │         records = step.run(                 ← 同步, to_thread 包装
  │           state=state,                      ← prompt_builder 从 state 读累积数据
  │           round_idx=round_idx,              ← 当前轮次
  │           prev_output=prev_output,          ← 上一步输出 (轮内链式传递)
  │           **config,
  │         )
  │         _update_state(state, step_spec, records)  ← append 或 set
  │         prev_output = records               ← 传给下一步
  │
  ├─ 5. 最终步骤:
  │     for step_spec in spec.final_steps:      ← critic 读全量 evals + reflects
  │       step = StepAgent(step_spec, client)
  │       records = step.run(state=state, **config)
  │       _update_state(state, step_spec, records)
  │
  ├─ 6. result = spec.result_builder(state, config)
  ├─ 7. 存 JSON 到 .agent/results/alpha-gpt-{timestamp}.json
  └─ 8. 返回摘要字符串给 LLM
```

### 2.4 轮内数据流示意

```
Round 1:
  IDEA_GEN (prev_output=None, state=empty)
    → ideas_1, state.all_ideas = [ideas_1]
    → prev_output = ideas_1

  FORMULA_TRANS (prev_output=ideas_1, state=...)
    → formulas_1, state.all_formulas = [formulas_1]
    → prev_output = formulas_1

  EVALUATOR (prev_output=formulas_1, state=...)
    → evals_1, state.all_evaluations = [evals_1]
    → prev_output = evals_1

  REFLECTOR (prev_output=evals_1, state=...)  ← skip_on_last=True, 最后一轮跳过
    → reflect_1, state.all_reflections = [reflect_1]

Round 2:
  IDEA_GEN (prev_output=None, state=all_ideas+all_reflections)
    → ideas_2, state.all_ideas = [ideas_1, ideas_2]
    → prompt_builder 从 state.all_reflections[-1] 读上轮建议

  ... 同上 ...

Round 5 (最后一轮):
  IDEA_GEN → FORMULA_TRANS → EVALUATOR  ← 正常执行
  REFLECTOR ← skip_on_last=True, 跳过

Final:
  CRITIC (state=全量 evals+reflects)
    → state.critic_output = final_pool
```

### 2.5 prompt_builder 数据来源

| prompt_builder | 数据来源 | 获取方式 |
|---------------|----------|----------|
| `_build_idea_prompt` | 上轮 reflection | `state.all_reflections[-1]` (通过 `state` 参数) |
| `_build_formula_prompt` | 本轮 ideas | `prev_output` 参数 |
| `_build_reflector_prompt` | 本轮 evaluations | `prev_output` 参数 |
| `_build_critic_prompt` | 全量 evals+reflects | `state.all_evaluations` + `state.all_reflections` |

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
    prompt_builder: Callable[..., str] | None        # (**ctx) -> prompt str, evaluator 为 None
    output_parser: Callable[[str], ParseResult] | None  # (raw) -> ParseResult, evaluator 为 None
    output_key: str                                  # JSON 输出 key: "ideas", "formulas", ...
    record_factory: Callable[[dict], Any] | None = None  # dict -> Record, evaluator 为 None
    state_output: str | None = None                  # 写入 state 的字段名: "all_ideas"
    state_input: str | None = None                   # 从 state 读取的字段名 (供 prompt_builder)
    tool_executor: Callable[..., list[Any]] | None = None  # evaluator 用
    max_retries: int = 2                             # 解析失败重试次数
    skip_on_last: bool = False                       # 最后一轮跳过 (reflector 用)
```

### 4.2 StepAgent

```python
class StepAgent:
    """轻量级单次 LLM 步骤。无工具循环、无会话历史。"""

    def __init__(self, spec: StepAgentSpec, llm_client=None): ...

    def run(self, state=None, round_idx=None, prev_output=None, **context) -> list[Any]:
        """执行: prompt → LLM → parse(带重试+修复) → records

        Args:
            state: workflow 状态对象, prompt_builder 可从中读取累积数据
            round_idx: 当前轮次 (多轮 workflow 用)
            prev_output: 上一步的输出 (轮内链式传递)
            **context: 其他参数 (config 等)
        """
        # 从 state 读取 state_input
        if self.spec.state_input and state is not None:
            context[self.spec.state_input] = getattr(state, self.spec.state_input)

        # 轮内上一步输出
        if prev_output is not None:
            context["prev_output"] = prev_output

        context["state"] = state
        context["round_idx"] = round_idx

        # tool_executor 路径 (evaluator)
        if self.spec.tool_executor:
            return self.spec.tool_executor(**context)

        # LLM 路径 (带重试)
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
    steps: list[StepAgentSpec]                       # 每轮执行的步骤
    state_factory: Callable[[], Any]                 # () -> State 对象
    result_builder: Callable[[Any, dict], dict]      # (state, config) -> result dict
    iterations: int = 1                              # 轮数 (1 = 线性)
    final_steps: list[StepAgentSpec] = []            # 最终步骤 (critic)
    provider: str | None = None                      # v2 扩展
    model: str | None = None                         # v2 扩展
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
        spec = REGISTRY.get(workflow)
        state = spec.state_factory()
        iterations = config.get("iterations", spec.iterations)
        client = self._llm_client

        # 多轮循环
        for round_idx in range(1, iterations + 1):
            is_last = (round_idx == iterations)
            prev_output = None

            for step_spec in spec.steps:
                if step_spec.skip_on_last and is_last:
                    continue
                step = StepAgent(step_spec, client)
                records = await asyncio.to_thread(
                    step.run,
                    state=state, round_idx=round_idx,
                    prev_output=prev_output, **config,
                )
                _update_state(state, step_spec, records)
                prev_output = records

        # 最终步骤
        for step_spec in spec.final_steps:
            step = StepAgent(step_spec, client)
            records = await asyncio.to_thread(
                step.run, state=state, **config,
            )
            _update_state(state, step_spec, records)

        # 构建结果
        result = spec.result_builder(state, config)
        # 存 JSON + 返回摘要
        ...

def _update_state(state, step_spec, records):
    """根据 step_spec.state_output 更新 state。"""
    if not step_spec.state_output:
        return
    target = getattr(state, step_spec.state_output)
    if isinstance(target, list):
        target.extend(records)
    else:
        setattr(state, step_spec.state_output, records)
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
v1: 始终用主 agent 的 provider/model
v2: workflow 配置的 provider/model → 主 agent 的 provider/model
```

### 6.2 实现

```python
# WorkflowTool.__init__ 时传入主 agent 的 provider
wt = WorkflowTool(
    llm_client=self._loop.provider,  # AgentLoop.provider (loop.py:214)
    model=getattr(self._loop, 'model', None),
)

# v1: execute 时始终用 self._llm_client
# v2: execute 时按 spec 覆盖 (spec.provider or self._llm_client)
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

| 步骤 | agent_id | LLM? | output_key | state_output | skip_on_last |
|------|----------|------|------------|--------------|--------------|
| 1. IdeaGenerator | `alpha-gpt-idea-generator` | ✅ | `ideas` | `all_ideas` | False |
| 2. FormulaTranslator | `alpha-gpt-formula-translator` | ✅ | `formulas` | `all_formulas` | False |
| 3. Evaluator | `alpha-gpt-evaluator` | ❌ | `evaluations` | `all_evaluations` | False |
| 4. Reflector | `alpha-gpt-reflector` | ✅ | `formula_feedback` | `all_reflections` | **True** |
| 5. Critic | `alpha-gpt-critic` | ✅ | `final_pool` | `critic_output` | N/A (final_steps) |

完整定义：

```python
IDEA_GEN_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-idea-generator",
    prompt_builder=_build_idea_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_idea_generator),
    output_key="ideas",
    state_output="all_ideas",
    record_factory=lambda d: IdeaRecord.from_dict(d, 0),  # round_idx 由 run() 注入
)

FORMULA_TRANS_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-formula-translator",
    prompt_builder=_build_formula_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_formula_translator),
    output_key="formulas",
    state_output="all_formulas",
    record_factory=lambda d: FormulaRecord(...),
)

EVALUATOR_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-evaluator",
    prompt_builder=None,
    output_parser=None,
    output_key="evaluations",
    state_output="all_evaluations",
    tool_executor=_run_evaluator,
    record_factory=lambda d: EvaluationRecord(...),
)

REFLECTOR_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-reflector",
    prompt_builder=_build_reflector_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_reflector),
    output_key="formula_feedback",
    state_output="all_reflections",
    skip_on_last=True,
    record_factory=lambda d: ReflectionRecord(...),
)

CRITIC_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-critic",
    prompt_builder=_build_critic_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_critic),
    output_key="final_pool",
    state_output="critic_output",
    record_factory=lambda d: FinalFormulaRecord.from_dict(d, 0),
)
```

注册（纯声明式）：

```python
ALPHA_GPT_SPEC = WorkflowSpec(
    name="alpha-gpt",
    description="5-round alpha discovery: idea → formula → evaluate → reflect → critic",
    steps=[IDEA_GEN_SPEC, FORMULA_TRANS_SPEC, EVALUATOR_SPEC, REFLECTOR_SPEC],
    iterations=5,
    final_steps=[CRITIC_SPEC],
    state_factory=lambda: AlphaGptState(objective=""),
    result_builder=_build_result,
)
REGISTRY.register(ALPHA_GPT_SPEC)
```

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

### 8.3 State 更新逻辑

由框架的 `_update_state()` 处理，根据 `StepAgentSpec.state_output` 自动判断更新方式：

```python
def _update_state(state, step_spec, records):
    if not step_spec.state_output:
        return
    target = getattr(state, step_spec.state_output)
    if isinstance(target, list):
        target.extend(records)      # ideas/formulas/evaluations/reflections → 追加
    else:
        setattr(state, step_spec.state_output, records)  # critic_output → 覆盖
```

| 步骤 | state_output | state 字段类型 | 更新方式 |
|------|-------------|---------------|---------|
| IdeaGenerator | `all_ideas` | `List[IdeaRecord]` | `extend` (每轮追加) |
| FormulaTranslator | `all_formulas` | `List[FormulaRecord]` | `extend` |
| Evaluator | `all_evaluations` | `List[EvaluationRecord]` | `extend` |
| Reflector | `all_reflections` | `List[ReflectionRecord]` | `extend` |
| Critic | `critic_output` | `Optional[Dict]` | `setattr` (覆盖) |

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

- 用 mock LLM 跑完整 alpha-gpt workflow（5 步 × 3 轮 + critic）
- 验证多轮迭代：state 中 all_ideas/all_formulas/all_evaluations/all_reflections 按轮次累积
- 验证 skip_on_last：最后一轮 reflector 被跳过
- 验证 final_steps：critic 在所有轮次后执行
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
