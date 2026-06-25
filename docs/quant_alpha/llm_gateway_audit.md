# LLM Gateway 统一入口验证报告

**Branch**: `feat/llm-gateway`
**HEAD**: `07ac02c`
**Date**: 2026-06-25

## 验证目标

确认所有调用 LLM 的地方已归集到 `LLMGateway` 统一入口。

## 验证结果：✅ 100% 归集

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

### 2. LLM 基础设施（保留作为 nanobot 底层）

| 组件 | 路径 | 状态 |
|------|------|------|
| OpenAIClient | `ai/llm/openai.py` | 仅在 `agent/providers/registry.py:142` 实例化 |
| LLMClientBase | `ai/llm/base.py` | 基类，LLMGateway 继承之 |
| NullLLMClient | `ai/llm/null.py` | LLMGateway 降级用 |
| Decorators | `ai/llm/decorators.py` | 工具类，未被使用 |
| QuantNodesProvider | `agent/providers/quantnodes.py` | nanobot provider |

### 3. nanobot upstream（不修改）

| 组件 | 路径 | 说明 |
|------|------|------|
| Agent | `agent/nanobot_bridge.py` | LLMGateway 内部持有 |
| agent_service | `api/services/agent_service.py` | 直接调用 `agent.run()` |
| agent router | `api/routers/agent.py` | WebSocket 路由 |
| cli/enhanced.py | `cli/enhanced.py` | CLI chat 命令 |

### 4. 配置/管理端点（不调 LLM 推理）

| 端点 | 路径 | 状态 |
|------|------|------|
| `GET /settings/models` | `api/routers/settings.py:99` | 查 provider `/models` — 配置管理 |
| `POST /settings/providers/{name}/test` | `api/services/settings_service.py:237` | 验证 API key — 配置管理 |
| `GET /settings/providers/models/all` | `api/services/settings_service.py:264` | 多 provider 模型列表 — 配置管理 |

## 调用入口

| 入口 | 用途 | 调用方 |
|------|------|--------|
| `LLMGateway.chat(messages)` | LLMClientBase 兼容 | strategy_gen, optimizer, report_reproducer |
| `LLMGateway.complete(agent_id, prompt)` | alpha_gpt 兼容 | alpha_gpt workflow |
| `LLMGateway.__call__(prompt)` | callable 注入兼容 | llm_judge, lineage_compress, operators |
| `LLMGateway.run(prompt, session_id)` | nanobot 原生 async | agent_service, 内部使用 |
| `get_llm_gateway()` | 全局单例 | 9 个模块的默认注入 |

## 测试覆盖

- `tests/ai/test_llm_gateway.py` — **37 passed**（新增）
- `tests/test_ai.py` — 34 passed
- `tests/agent/test_tools.py` — 56 passed
- `tests/core/feedback/test_llm_judge_edges.py` — 15 passed
- `tests/core/knowledge/test_lineage_edges.py` — 21 passed
- 改造模块相关测试 — **163 passed**

## 验证检查项

- [x] 9 个 LLM 消费模块全部使用 `get_llm_gateway()` 默认注入
- [x] 所有 `self.llm.chat()` / `self.llm_client.chat()` / `self.llm_client.complete()` / `self._llm_callable()` 内部由 LLMGateway 实现
- [x] 0 个 `requests.post` 直接调 `/chat/completions`（除 nanobot provider 内部）
- [x] 0 个 `requests.get` 直接调 `/models`（除 settings 配置端点）
- [x] OpenAIClient 仅在 `agent/providers/registry.py:142` 实例化（唯一调用点）
- [x] litellm 仅在 `agent/providers/quantnodes.py` 使用（nanobot 内部）
- [x] 向后兼容：显式传入 `llm_client` / `llm_callable` 优先于默认注入
- [x] mock 模式：`model="mock"` 时不注入 LLMGateway

## 结论

**✅ 所有 LLM 调用已 100% 归集到 LLMGateway 统一入口，委托 nanobot upstream。**
