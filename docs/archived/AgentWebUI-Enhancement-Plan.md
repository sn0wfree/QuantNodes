# Agent Web UI Enhancement Plan

> 版本: v1.0  
> 日期: 2026-05-10  
> 目标: Markdown 渲染 + Tool 可视化 + Session 管理 + 真流式输出

---

## 一、现状分析

### 已有能力
- `OpenAIClient._call_api_stream()` ✅ SSE 流式已实现
- `LLMClientBase.chat_stream()` ✅ 基类接口已定义
- `LLMProvider.chat_stream()` ✅ Provider 层有 `on_content_delta` 回调
- `AgentHook.on_stream()` ✅ Hook 系统支持流式事件
- `ToolCallCard.vue` ✅ 组件已实现但未集成
- WebSocket 双向通信 ✅ 前后端连接正常

### 缺失环节
- `QuantNodesLLMProvider` 未实现 `chat_stream()`
- `AgentRunner.run()` 未使用流式
- `Agent.chat()` 一次性 yield 完整结果
- 前端无 markdown 渲染库
- Tool 事件只 console.log，无 UI
- 无 session 管理 UI

---

## 二、实施计划

### Phase 1: 后端真流式输出

#### 1.1 QuantNodesLLMProvider.chat_stream() 实现
**文件**: `QuantNodes/agent/providers/quantnodes.py`

```python
async def chat_stream(self, messages, tools, model, on_content_delta, **kwargs):
    # 1. 转换 messages
    # 2. 注入 tool descriptions 到 system prompt
    # 3. 调用 client.chat_stream() 获取 chunks
    # 4. 累积 content，检测 ```tool_call``` 块
    # 5. 每个 delta 调用 on_content_delta(delta)
    # 6. 返回最终 LLMResponse
```

**注意**: 工具调用是 prompt-based（```tool_call``` 代码块），流式输出中需检测未闭合的 ```tool_call``` 块，延迟发送这部分内容。

#### 1.2 AgentRunner.run() 流式改造
**文件**: `QuantNodes/agent/core/runner.py`

- 新增 `run_stream()` 方法，接受 `on_token` 回调
- 每轮迭代中：
  - 调用 `provider.chat_stream()` 而非 `provider.chat()`
  - 每个 delta 通过 `on_token` 发送
  - 检测到 tool_calls 时停止流式，执行工具后继续下一轮
- 保留原 `run()` 方法向后兼容

#### 1.3 Agent.chat() 真流式
**文件**: `QuantNodes/agent/__init__.py`

```python
async def chat(self, message, session_id):
    # 改为:
    async for token in self._loop.chat_stream(message, session_id):
        yield token
```

#### 1.4 AgentLoop.chat_stream() 新增
**文件**: `QuantNodes/agent/core/loop.py`

- 新增 `chat_stream()` async generator
- 复用 `chat()` 的 context 构建逻辑
- 调用 `runner.run_stream()` 并 yield tokens

#### 1.5 WebSocket 消息协议扩展
**文件**: `api/services/agent_service.py`, `api/routers/agent.py`

新增消息类型：
```json
{"type": "chunk", "content": "token...", "message_id": "msg-xxx"}
{"type": "tool_call", "tool": "web_search", "arguments": {...}, "message_id": "msg-xxx"}
{"type": "tool_result", "tool": "web_search", "content": "...", "message_id": "msg-xxx"}
{"type": "done", "message_id": "msg-xxx", "usage": {...}}
{"type": "error", "content": "...", "message_id": "msg-xxx"}
```

**实现方式**: 在 `AgentLoop` 中注册一个 `StreamingHook(AgentHook)`：
- `on_stream()` → yield `{"type": "chunk", "content": delta}`
- `before_execute_tools()` → yield `{"type": "tool_call", ...}`
- `after_iteration()` → yield `{"type": "tool_result", ...}`

---

### Phase 2: 前端 Markdown 渲染

#### 2.1 安装依赖
```bash
cd frontend && npm install markdown-it highlight.js
npm install -D @types/markdown-it @types/highlight.js
```

#### 2.2 创建 MarkdownRenderer 组件
**新文件**: `frontend/src/components/Chat/MarkdownRenderer.vue`

- 使用 `markdown-it` 解析 Markdown
- 使用 `highlight.js` 高亮代码块
- 支持 GFM（表格、任务列表、删除线）
- 安全: 使用 DOMPurify 防 XSS（或 markdown-it 内置的 HTML 禁用）

```vue
<template>
  <div class="markdown-body" v-html="renderedHtml"></div>
</template>

<script setup>
import MarkdownIt from 'markdown-it'
import hljs from 'highlight.js'

const md = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: true,
  highlight: (str, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(str, { language: lang }).value
    }
    return md.utils.escapeHtml(str)
  }
})

const props = defineProps<{ content: string }>()
const renderedHtml = computed(() => md.render(props.content))
</script>
```

#### 2.3 创建 Markdown 样式
**新文件**: `frontend/src/styles/markdown.css`

- GitHub 风格 markdown-body
- 代码块样式（支持 highlight.js 主题）
- 表格、引用块、列表样式

#### 2.4 更新 ChatMessage 组件
**文件**: `frontend/src/components/Chat/ChatMessage.vue`

- assistant 消息使用 `<MarkdownRenderer>` 渲染
- user 消息保持纯文本（或也支持 markdown）
- 添加 `time` prop 显示
- 添加复制按钮

---

### Phase 3: Tool 可视化

#### 3.1 扩展 useAgent composable
**文件**: `frontend/src/composables/useAgent.ts`

```typescript
// 新增状态
const toolCalls = ref<ToolCallEvent[]>([])
const currentToolCalls = ref<ToolCallEvent[]>([])

// 处理新消息类型
case 'tool_call':
  currentToolCalls.value.push({
    id: data.tool_call_id,
    name: data.tool,
    arguments: data.arguments,
    status: 'running'
  })
  break
case 'tool_result':
  const tc = currentToolCalls.value.find(t => t.id === data.tool_call_id)
  if (tc) {
    tc.status = data.error ? 'error' : 'success'
    tc.result = data.content
  }
  break
case 'done':
  // 将 currentToolCalls 附加到消息
  break
```

#### 3.2 更新 Message 接口
**文件**: `frontend/src/stores/agent.ts`

```typescript
interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCallEvent[]  // 现在实际填充
  timestamp: number
}
```

#### 3.3 AgentChat 集成 ToolCallCard
**文件**: `frontend/src/views/AgentChat/index.vue`

```vue
<ChatMessage v-for="msg in store.messages" :key="msg.id" :role="msg.role">
  <MarkdownRenderer :content="msg.content" />
  <template v-if="msg.toolCalls?.length">
    <ToolCallCard
      v-for="tc in msg.toolCalls"
      :key="tc.id"
      :toolName="tc.name"
      :arguments="tc.arguments"
      :result="tc.result"
      :status="tc.status"
    />
  </template>
</ChatMessage>
```

#### 3.4 流式中的 Tool 显示
- `tool_call` 事件到达时，立即在聊天区域显示 running 状态的 ToolCallCard
- `tool_result` 事件到达时，更新对应 ToolCallCard 的状态
- `done` 事件到达时，将完整的 toolCalls 数组绑定到消息

---

### Phase 4: Session 管理

#### 4.1 后端 Session API
**文件**: `api/routers/agent.py`

已有：
- `GET /api/chat/history/{session_id}`
- `DELETE /api/chat/history/{session_id}`

需新增：
- `GET /api/chat/sessions` — 列出所有 session
- `POST /api/chat/sessions` — 创建新 session（返回 session_id）
- `DELETE /api/chat/sessions/{session_id}` — 删除 session

#### 4.2 前端 Session 管理
**文件**: `frontend/src/stores/agent.ts`

```typescript
// 新增
const sessions = ref<SessionInfo[]>([])
const currentSessionId = ref('')

// Actions
const loadSessions = async () => { ... }
const createSession = async () => { ... }
const switchSession = async (id: string) => { ... }
const deleteSession = async (id: string) => { ... }
```

#### 4.3 AgentChat UI 增强
**文件**: `frontend/src/views/AgentChat/index.vue`

新增元素：
- 顶部工具栏：session 名称、新建对话按钮、清空按钮
- 侧边栏或下拉：session 列表（可切换/删除）
- 连接状态指示器（绿色/红色圆点）
- 消息时间戳

---

### Phase 5: 样式与体验优化

#### 5.1 AgentChat 布局优化
- 消息区域全高，固定输入区在底部
- 消息气泡最大宽度调整
- 流式输出时的光标动画改进
- 空状态美化

#### 5.2 highlight.js 主题选择
- 使用 `github-dark` 或 `atom-one-dark` 主题
- 在 `main.ts` 中引入 CSS

#### 5.3 响应式适配
- 移动端消息布局调整

---

## 三、文件变更清单

### 新增文件
| 文件 | 说明 |
|------|------|
| `frontend/src/components/Chat/MarkdownRenderer.vue` | Markdown 渲染组件 |
| `frontend/src/styles/markdown.css` | Markdown 样式 |
| `tests/agent/test_streaming.py` | 流式输出测试 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `QuantNodes/agent/providers/quantnodes.py` | 实现 `chat_stream()` |
| `QuantNodes/agent/providers/base.py` | 微调 `chat_stream()` 接口 |
| `QuantNodes/agent/core/runner.py` | 新增 `run_stream()` |
| `QuantNodes/agent/core/loop.py` | 新增 `chat_stream()` |
| `QuantNodes/agent/__init__.py` | `chat()` 改为真流式 |
| `api/services/agent_service.py` | `stream_message()` 发送 tool 事件 |
| `api/routers/agent.py` | 新增 session API |
| `frontend/package.json` | 安装 markdown-it, highlight.js |
| `frontend/src/main.ts` | 引入 markdown.css, highlight.js 主题 |
| `frontend/src/composables/useAgent.ts` | 处理 tool_call/tool_result 事件 |
| `frontend/src/stores/agent.ts` | 扩展 Message 接口，新增 session 状态 |
| `frontend/src/components/Chat/ChatMessage.vue` | 集成 MarkdownRenderer，显示时间 |
| `frontend/src/views/AgentChat/index.vue` | 集成 ToolCallCard，session UI |

---

## 四、依赖关系

```
Phase 1 (后端流式) ← 独立，可先做
Phase 2 (Markdown) ← 独立，可并行
Phase 3 (Tool 可视化) ← 依赖 Phase 1 (需要 tool_call/tool_result 事件)
Phase 4 (Session) ← 独立，可并行
Phase 5 (样式) ← 依赖 Phase 2, 3
```

建议执行顺序: **Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5**

---

## 五、测试计划

### 后端测试
- `test_streaming.py`: 测试 `Agent.chat()` 真流式输出
- `test_runner_stream.py`: 测试 `AgentRunner.run_stream()` 工具调用循环
- 测试 WebSocket 消息类型完整性

### 前端测试
- MarkdownRenderer 渲染正确性（代码块、表格、链接）
- ToolCallCard 状态流转（running → success/error）
- Session 创建/切换/删除
- 流式输出实时显示

---

## 六、风险与注意事项

1. **工具调用检测**: 流式输出中检测未闭合的 ```tool_call``` 块需要缓冲区，避免发送不完整的 JSON
2. **并发安全**: Session 管理需要考虑多 tab 场景
3. **内存**: 大量 tool_result 可能占用较多内存，考虑截断策略
4. **向后兼容**: 保留原 `Agent.run()` 和 `AgentRunner.run()` 接口不变
