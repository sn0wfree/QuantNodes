# LLM Gateway 统一入口设计

## 背景

项目中存在 4 种不同的 LLM 调用接口，散落在多个模块中：

| 接口模式 | 签名 | 使用模块 |
|:--------:|------|----------|
| A | `llm.chat(messages)` → `ChatCompletion` | `strategy_gen.py`, `optimizer.py`, `report_reproducer.py` |
| B | `llm_client.complete(agent_id, prompt)` → `str` | `alpha_gpt.py` |
| C | `_llm_callable(prompt)` → `str` | `llm_judge.py`, `lineage_compress.py`, `operators.py` |
| D | `agent.run(prompt, session_id)` → `str` | `agent_service.py`, `cli/commands/alpha.py` |

**目标**：所有 LLM 调用统一委托 nanobot，通过单一入口 `LLMGateway` 实现。

## 架构

```
模块层 (消费方)
  │
  ├─ llm.chat(messages)       → ChatCompletion   (接口 A)
  ├─ llm.complete(id, prompt) → str              (接口 B)
  ├─ llm(prompt)              → str              (接口 C)
  └─ await llm.run(prompt)    → str              (接口 D)
  │
  ▼
LLMGateway (统一入口)
  │
  ├─ 内部持有 nanobot Agent 实例
  ├─ 所有接口最终委托 agent.run()
  └─ 同步方法通过 asyncio 桥接
  │
  ▼
nanobot Agent → QuantNodesLLMProvider → LLM API
```

## LLMGateway API

```python
class LLMGateway:
    def __init__(self, agent=None, workspace=".agent"): ...

    # 接口 A: LLMClientBase 兼容
    def chat(self, messages, model=None, temperature=0.7,
             max_tokens=None, **kwargs) -> ChatCompletion: ...

    # 接口 B: complete 兼容
    def complete(self, agent_id="default", prompt="") -> str: ...

    # 接口 C: callable 兼容
    def __call__(self, prompt: str) -> str: ...

    # 接口 D: nanobot 原生
    async def run(self, prompt: str, session_id="default") -> str: ...
```

## 模块改造

每个模块的 `__init__` 仍接受原有参数，但默认值改为 `LLMGateway`：

```python
# strategy_gen.py
class StrategyGenerator:
    def __init__(self, llm_client=None, ...):
        self.llm = llm_client or get_llm_gateway()

# llm_judge.py
class LLMJudge:
    def __init__(self, ..., llm_callable=None):
        self._llm_callable = llm_callable or get_llm_gateway()
```

## 向后兼容

- 所有 `llm_client` / `llm_callable` 参数保留，可传入旧的 `LLMClientBase` 实例
- 不传时自动使用 `LLMGateway`（通过 `get_llm_gateway()`）
- mock 模式：nanobot 不可用时 `LLMGateway` 降级到 `NullLLMClient`

## 文件清单

| 文件 | 操作 |
|------|------|
| `ai/llm/gateway.py` | 新建 |
| `ai/llm/__init__.py` | 添加导出 |
| `ai/__init__.py` | 添加导出 |
| `ai/strategy_gen.py` | 改造 `__init__` |
| `ai/optimizer.py` | 改造 `__init__` |
| `research/report_reproducer.py` | 改造 `__init__` |
| `research/quant_alpha/workflow/alpha_gpt.py` | 改造 `_call_llm` |
| `core/feedback/llm_judge.py` | 改造 `__init__` |
| `core/knowledge/lineage_compress.py` | 改造 `__init__` |
| `core/evolution/operators.py` | 改造 `__init__` |
| `agent/tools/strategy.py` | 改造 `__init__` |
| `cli/commands/alpha.py` | `NanobotLLMWrapper` → `LLMGateway` |
