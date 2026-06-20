# Fix: Chat 页面 Per-Message 模型切换

## 背景

当前模型切换只能在 Settings 页面修改，需要保存并重启 Agent。用户希望在 Chat 页面直接切换模型，即时生效。

## 方案

Per-Message 模型覆盖：每次发消息时携带 model 参数，临时覆盖全局配置，不修改 settings.json。

## 数据流

```
前端 selector → store.currentModel
  → useAgent.sendMessage(content, model)
    → WebSocket: {type:"message", content, session_id, model}
      → api/routers/agent.py: 读取 data["model"]
        → agent_service.stream_message(content, session_id, config={model})
          → agent.chat(message, session_id, model=model)
            → loop.chat_stream(message, session_id, model=model)
              → AgentRunSpec(model=model or self.model)
                → runner.run_stream(spec)
                  → provider.chat_stream(model=spec.model)
                    → client.chat_stream(model=model)
```

## 修改文件

### 后端（3 文件）

1. **`QuantNodes/agent/__init__.py`** — Agent.chat() 添加 `model` 参数
2. **`QuantNodes/agent/core/loop.py`** — AgentLoop.chat_stream() 添加 `model` 参数，传递给 AgentRunSpec
3. **`api/services/agent_service.py`** — stream_message() 提取 config 中的 model 并传递
4. **`api/routers/agent.py`** — WebSocket handler 读取 `data["model"]`

### 前端（3 文件）

5. **`frontend/src/stores/agent.ts`** — 添加 `currentModel` 状态
6. **`frontend/src/views/AgentChat/index.vue`** — Chat 头部添加模型选择器
7. **`frontend/src/composables/useAgent.ts`** — sendMessage() 传递 model

### 无需修改

- `AgentRunner` — 已支持 `spec.model`
- `QuantNodesLLMProvider` — 已支持 `model` 参数
- `OpenAIClient` — 已支持 `model` 参数
- `AgentRunSpec` — 已有 `model` 字段

## 向后兼容

不传 model 时行为不变，默认使用 Agent 创建时的全局模型。

## 预置模型列表

- MiniMax M2.5 (Free): `minimax/minimax-m2.5:free`
- MiniMax M2.5: `minimax/minimax-m2.5`
- MiniMax M2.7: `minimax/minimax-m2.7`
- GPT-4o: `openai/gpt-4o`
- GPT-4o Mini: `openai/gpt-4o-mini`

## 状态

- [ ] 实施
- [ ] 测试
