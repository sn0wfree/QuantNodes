# LLMGateway 工具调用扩展设计

## 背景

LLMGateway 已统一了 9 个模块的 LLM 调用入口，但仅支持纯文本调用。
nanobot 内部有 33 个工具（18 upstream + 15 quant），LLMGateway 无法让调用方：
- 选择工具子集
- 控制 tool_choice
- 获取 tool_calls 返回

## 目标

扩展 LLMGateway 支持工具调用，同时保持向后兼容。

## 设计决策

| 决策 | 选择 |
|------|------|
| 工具过滤粒度 | 工具名列表 `tools=["backtest", "factor"]` |
| 同步接口工具支持 | 是，扩展 `ChatCompletion.tool_calls` |
| 完成顺序 | Stage 2 之前 |

## 接口设计

### LLMGateway 扩展

```python
# 1. chat() — 同步, 返回含 tool_calls 的 ChatCompletion
def _call_api(self, messages, model=None, tools=None, tool_choice=None, **kwargs) -> ChatCompletion:
    # tools: List[str] — 工具名列表, None=全部
    # tool_choice: str — "auto"/"none"/"required"
    # Returns: ChatCompletion(content, tool_calls, finish_reason)

# 2. run() — 异步, 默认返回文本, with_tool_events=True 返回 ToolCallResponse
async def run(self, prompt, session_id, tools, tool_choice, with_tool_events=False):
    # with_tool_events=False → str (向后兼容)
    # with_tool_events=True → ToolCallResponse(content, tools_used, stop_reason, events)

# 3. complete() / __call__() — 同步, 透传 tools
def complete(self, agent_id, prompt, tools=None, tool_choice=None) -> str: ...
def __call__(self, prompt, tools=None, tool_choice=None) -> str: ...
```

### ToolCallResponse 数据类

```python
@dataclass
class ToolCallResponse:
    content: str
    tools_used: List[str]
    stop_reason: str
    events: List[Dict[str, Any]]
    @property
    def has_tool_calls(self) -> bool: ...
```

### Agent.chat() 增强

```python
async def chat(self, message, session_id, ..., tools=None, tool_choice=None):
    # tools: List[str] — 工具名列表, None=全部
    # 内部: _filter_tools(tools) → ToolRegistry 子集
    # 透传 tool_choice 给 AgentRunSpec
```

### 工具过滤

```python
def _filter_tools(self, tool_names: Optional[List[str]]) -> ToolRegistry:
    # None → 返回全部
    # ["backtest", "factor"] → 返回子集 (未知工具名警告并跳过)
```

## 文件清单

| 文件 | 操作 |
|------|------|
| `ai/llm/gateway.py` | 扩展接口 + ToolCallResponse + _async_chat_collect |
| `agent/nanobot_bridge.py` | Agent.chat() + _filter_tools() |
| `tests/ai/test_llm_gateway.py` | 新增工具调用测试 |
| `tests/agent/test_nanobot_bridge.py` | 测试 Agent.chat(tools=...) |
| `docs/quant_alpha/llm_gateway_tool_calling.md` | 本文档 |
| `docs/quant_alpha/llm_gateway_audit.md` | 刷新验证报告 |
