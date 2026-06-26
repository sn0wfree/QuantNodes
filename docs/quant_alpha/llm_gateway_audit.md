# LLM Gateway 统一入口验证报告

**Branch**: `feat/llm-gateway-tool-calling`
**HEAD**: `904e1b2`
**Date**: 2026-06-25

## 验证目标

确认所有调用 LLM 的地方已归集到 `LLMGateway` 统一入口，且支持工具调用。

## 验证结果：✅ 100% 归集 + 工具调用支持

### 1. LLM 消费模块（9 个）— 全部使用 LLMGateway

| # | 模块 | 路径 | 接口 | 默认注入 LLMGateway |
|---|------|------|------|:---:|
| 1 | StrategyGenerator | `ai/strategy_gen.py:62` | `self.llm.chat()` | ✅ |
| 2 | NaturalLanguageToPipeline | `ai/strategy_gen.py:359` | 委托 StrategyGenerator | ✅ |
| 3 | PipelineOptimizer | `ai/optimizer.py:255` | `self.llm.chat()` | ✅ |
| 4 | ResearchReportReproducer | `research/report_reproducer.py:125` | `self.llm_client.chat()` | ✅ |
| 5 | AlphaGptWorkflow | `research/quant_alpha/workflow/alpha_gpt.py:124` | `self.llm_client.complete()` / `__call__()` | ✅ |
| 6 | LLMJudge | `core/feedback/llm_judge.py:33` | `self._llm_callable(prompt)` | ✅ (model != mock) |
| 7 | Compressor | `core/knowledge/lineage_compress.py:57` | `self._llm_callable(prompt)` | ✅ (model != mock) |
| 8 | BaseOperator (Hypothesizer/Mutator/Crosser) | `core/evolution/operators.py:71` | `self._llm_callable(prompt)` | ✅ (model != mock) |
| 9 | StrategyTool | `agent/tools/strategy.py:27` | 委托 StrategyGenerator | ✅ |

### 2. LLMGateway 工具调用支持

| 接口 | 签名 | 返回 | 说明 |
|------|------|------|------|
| `chat(messages, tools, tool_choice)` | `List[str], str` | `ChatCompletion` | 同步, 含 `tool_calls` 字段 |
| `complete(agent_id, prompt, tools, tool_choice)` | `str, List[str], str` | `str` | 同步, 纯文本 (工具调用整合到 content) |
| `__call__(prompt, tools, tool_choice)` | `str, List[str], str` | `str` | 同步, callable 兼容 |
| `run(prompt, tools, tool_choice, with_tool_events)` | `str, bool` | `str \| ToolCallResponse` | 异步, with_tool_events=True 返回工具事件 |

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tools` | `Optional[List[str]]` | `None` | 工具名列表, `None`=全部 33 个工具 |
| `tool_choice` | `Optional[str]` | `None` | `"auto"`/`"none"`/`"required"` |
| `with_tool_events` | `bool` | `False` | 仅 `run()`, 返回 `ToolCallResponse` |

### 3. Agent.chat() 增强 (nanobot_bridge.py)

| 增强 | 说明 |
|------|------|
| `tools` 参数 | 工具名列表, `None`=全部, `[]`=无工具 |
| `tool_choice` 参数 | 透传给 nanobot |
| `_filter_tools()` | 工具名列表过滤, 未知工具名警告并跳过 |

### 4. LLM 基础设施（保留作为 nanobot 底层）

| 组件 | 路径 | 状态 |
|------|------|------|
| OpenAIClient | `ai/llm/openai.py` | 仅在 `agent/providers/registry.py:142` 实例化 |
| LLMClientBase | `ai/llm/base.py` | 基类, LLMGateway 继承之 |
| NullLLMClient | `ai/llm/null.py` | LLMGateway 降级用 |
| Decorators | `ai/llm/decorators.py` | 工具类, 未被使用 |
| QuantNodesProvider | `agent/providers/quantnodes.py` | nanobot provider |

### 5. nanobot upstream（不修改）

| 组件 | 路径 | 说明 |
|------|------|------|
| Agent | `agent/nanobot_bridge.py` | LLMGateway 内部持有 |
| agent_service | `api/services/agent_service.py` | 直接调用 `agent.run()` |
| agent router | `api/routers/agent.py` | WebSocket 路由 |
| cli/enhanced.py | `cli/enhanced.py` | CLI chat 命令 |

### 6. 配置/管理端点（不调 LLM 推理）

| 端点 | 路径 | 状态 |
|------|------|------|
| `GET /settings/models` | `api/routers/settings.py:99` | 查 provider `/models` — 配置管理 |
| `POST /settings/providers/{name}/test` | `api/services/settings_service.py:237` | 验证 API key — 配置管理 |
| `GET /settings/providers/models/all` | `api/services/settings_service.py:264` | 多 provider 模型列表 — 配置管理 |

## 调用入口

| 入口 | 用途 | 调用方 | 工具调用 |
|------|------|--------|:--------:|
| `LLMGateway.chat(messages, tools, tool_choice)` | LLMClientBase 兼容 | strategy_gen, optimizer, report_reproducer | ✅ |
| `LLMGateway.complete(agent_id, prompt, tools, tool_choice)` | alpha_gpt 兼容 | alpha_gpt workflow | ✅ |
| `LLMGateway.__call__(prompt, tools, tool_choice)` | callable 注入兼容 | llm_judge, lineage_compress, operators | ✅ |
| `LLMGateway.run(prompt, tools, tool_choice, with_tool_events)` | nanobot 原生 async | agent_service, 内部使用 | ✅ |
| `get_llm_gateway()` | 全局单例 | 9 个模块的默认注入 | — |

## 测试覆盖

- `tests/ai/test_llm_gateway.py` — **54 passed** (含 17 个工具调用测试)
- `tests/test_ai.py` — 34 passed
- `tests/agent/test_tools.py` — 56 passed
- `tests/core/feedback/test_llm_judge_edges.py` — 15 passed
- `tests/core/knowledge/test_lineage_edges.py` — 21 passed
- 改造模块相关测试 — **195 passed**

## 验证检查项

- [x] 9 个 LLM 消费模块全部使用 `get_llm_gateway()` 默认注入
- [x] 所有 `self.llm.chat()` / `self.llm_client.chat()` / `self.llm_client.complete()` / `self._llm_callable()` 内部由 LLMGateway 实现
- [x] 0 个 `requests.post` 直接调 `/chat/completions`（除 nanobot provider 内部）
- [x] 0 个 `requests.get` 直接调 `/models`（除 settings 配置端点）
- [x] OpenAIClient 仅在 `agent/providers/registry.py:142` 实例化（唯一调用点）
- [x] litellm 仅在 `agent/providers/quantnodes.py` 使用（nanobot 内部）
- [x] 向后兼容：显式传入 `llm_client` / `llm_callable` 优先于默认注入
- [x] mock 模式：`model="mock"` 时不注入 LLMGateway
- [x] 工具调用：`chat()` / `complete()` / `__call__()` / `run()` 均支持 `tools` / `tool_choice`
- [x] 工具过滤：`_filter_tools()` 支持工具名列表, 未知工具名警告
- [x] `ChatCompletion.tool_calls` 字段支持工具调用返回
- [x] `ToolCallResponse` 数据类支持异步工具事件

## 结论

**✅ 所有 LLM 调用已 100% 归集到 LLMGateway 统一入口，委托 nanobot upstream。**
**✅ 工具调用已扩展支持，所有 4 种接口均支持 tools/tool_choice 参数。**
