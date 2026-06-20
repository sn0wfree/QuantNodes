# Agent Chat UI Enhancement Plan — 基于 OpenCode/Crush 启发

> 版本: v1.0
> 日期: 2026-05-12
> 参考: OpenCode (opencode-ai/opencode) / Crush (charmbracelet/crush)
> 目标: 将终端 AI 编码工具的最佳交互模式引入 QuantNodes Web Agent Chat

---

## 一、现状分析

### 已有能力
- ✅ 流式输出 + Markdown 渲染 + 代码高亮
- ✅ ToolCallCard 工具调用可视化
- ✅ Session 管理（创建/切换/删除）
- ✅ 模型选择器（头部下拉框，per-message 覆盖）
- ✅ WebSocket 双向通信
- ✅ 连接状态指示器

### 缺失环节（OpenCode 有而 QuantNodes 没有）
- ❌ 快捷键系统
- ❌ 命令面板（Command Palette）
- ❌ Auto Compact（自动上下文压缩）
- ❌ 工具调用权限确认
- ❌ 会话导出

---

## 二、功能优先级

| 优先级 | 功能 | 复杂度 | 价值 | 状态 |
|---|---|---|---|---|
| **P0** | 快捷键系统 | 中 | 高 | 计划中 |
| **P0** | 命令面板（Ctrl+K） | 中 | 高 | 计划中 |
| **P1** | Auto Compact 自动压缩 | 低 | 高 | 计划中 |
| **P1** | 模型选择器增强 | 低 | 中 | 计划中 |
| **P2** | 工具调用权限确认 | 中 | 中 | 计划中 |
| **P2** | 会话导出 | 低 | 中 | 计划中 |
| **P3** | 文件变更追踪 | 高 | 低 | 暂缓 |

---

## 三、P0-1: 快捷键系统

### 快捷键清单

| 快捷键 | 功能 | 参考 |
|---|---|---|
| `Ctrl+K` | 打开命令面板 | OpenCode |
| `Ctrl+O` | 打开模型选择对话框 | OpenCode |
| `Ctrl+N` | 新建会话 | OpenCode |
| `Ctrl+S` | 发送消息 | OpenCode |
| `Ctrl+X` | 取消当前生成 | OpenCode |
| `Ctrl+L` | 清空当前会话 | 量化场景常用 |
| `Escape` | 关闭弹窗/取消输入 | 通用 |

### 文件变更

**新增**: `frontend/src/composables/useKeyboard.ts`

```typescript
import { onMounted, onUnmounted } from 'vue'

type KeyHandler = (e: KeyboardEvent) => void

export function useKeyboard() {
  const shortcuts = new Map<string, KeyHandler>()

  const register = (key: string, handler: KeyHandler) => {
    shortcuts.set(key, handler)
  }

  const handleKeyDown = (e: KeyboardEvent) => {
    const tag = (e.target as HTMLElement).tagName
    const isInput = tag === 'INPUT' || tag === 'TEXTAREA'

    const key = [
      e.ctrlKey ? 'ctrl' : '',
      e.metaKey ? 'meta' : '',
      e.shiftKey ? 'shift' : '',
      e.key.toLowerCase(),
    ].filter(Boolean).join('+')

    // 输入框内只允许 Ctrl+S 和 Escape
    if (isInput && key !== 'ctrl+s' && key !== 'escape') return

    const handler = shortcuts.get(key)
    if (handler) {
      e.preventDefault()
      handler(e)
    }
  }

  onMounted(() => document.addEventListener('keydown', handleKeyDown))
  onUnmounted(() => document.removeEventListener('keydown', handleKeyDown))

  return { register }
}
```

**修改**: `frontend/src/views/AgentChat/index.vue` — 注册快捷键

---

## 四、P0-2: 命令面板（Command Palette）

### UI 设计

```
┌─────────────────────────────────────────────┐
│  🔍 Type a command...                       │
├─────────────────────────────────────────────┤
│  Sessions                                   │
│    → New Chat                    Ctrl+N      │
│    → Switch Session...           Ctrl+A      │
│    → Clear History                         │
│    → Export Session...                      │
├─────────────────────────────────────────────┤
│  Model                                      │
│    → Switch Model...             Ctrl+O      │
│    → MiniMax M2.5 (Free)         ✓ current  │
│    → GPT-4o                                  │
├─────────────────────────────────────────────┤
│  Tools                                      │
│    → Factor Analysis                         │
│    → Run Backtest                            │
│    → Generate Strategy                       │
├─────────────────────────────────────────────┤
│  View                                       │
│    → Dream Insights                          │
│    → Wiki Explorer                           │
│    → Settings                    Ctrl+,      │
└─────────────────────────────────────────────┘
```

### 文件变更

**新增**:
- `frontend/src/composables/useCommands.ts` — 命令注册管理
- `frontend/src/components/Chat/CommandPalette.vue` — 命令面板组件

**修改**: `frontend/src/views/AgentChat/index.vue` — 集成命令面板

---

## 五、P1-1: Auto Compact 自动压缩

### 触发条件
- 每次 LLM 调用前检查 token 使用量
- 当 `prompt tokens >= context_window * 0.9` 时触发

### 后端实现

**修改**: `QuantNodes/agent/core/loop.py`

在 `chat_stream()` 中添加 token 检查和压缩逻辑。

### 前端处理

**修改**: `frontend/src/composables/useAgent.ts`

处理 `system` 事件类型，显示压缩通知。

---

## 六、P1-2: 模型选择器增强

### UI 设计（弹出式对话框，Ctrl+O 触发）

```
┌─────────────────────────────────────────────┐
│  🔍 Search models...                        │
├─────────────────────────────────────────────┤
│  Free                                       │
│    ● MiniMax M2.5 (Free)          ✓        │
│      context: 1M | price: free             │
├─────────────────────────────────────────────┤
│  MiniMax                                    │
│    ○ MiniMax M2.5                           │
│      context: 1M | price: $0.15/M in       │
├─────────────────────────────────────────────┤
│  OpenAI                                     │
│    ○ GPT-4o                                 │
│      context: 128K | price: $2.50/M in     │
└─────────────────────────────────────────────┘
```

### 文件变更

**新增**:
- `frontend/src/constants/models.ts` — 模型元数据
- `frontend/src/components/Chat/ModelSelector.vue` — 模型选择对话框

---

## 七、P2-1: 工具调用权限确认

### 风险分级

| 风险等级 | 工具 | 默认行为 |
|---|---|---|
| **高** | SandboxTool, FileOpsTool, GitOpsTool | 需确认 |
| **中** | BacktestTool, StrategyTool | 需确认 |
| **低** | EchoTool, WikiTool, WebSearchTool | 自动执行 |

### 文件变更

**新增**: `frontend/src/components/Chat/PermissionDialog.vue`

**修改**: `frontend/src/composables/useAgent.ts` — 处理 `permission_request` 事件

---

## 八、P2-2: 会话导出

### 文件变更

**新增后端 API**: `GET /api/chat/export/{session_id}?format=markdown|json`

**修改前端**:
- `frontend/src/views/AgentChat/index.vue` — 添加导出按钮

---

## 九、文件变更总清单

### 新增文件（6 个）
| 文件 | 说明 |
|---|---|
| `frontend/src/composables/useKeyboard.ts` | 快捷键管理 |
| `frontend/src/composables/useCommands.ts` | 命令注册管理 |
| `frontend/src/components/Chat/CommandPalette.vue` | 命令面板 |
| `frontend/src/components/Chat/ModelSelector.vue` | 模型选择对话框 |
| `frontend/src/components/Chat/PermissionDialog.vue` | 工具权限确认 |
| `frontend/src/constants/models.ts` | 模型元数据 |

### 修改文件（5 个）
| 文件 | 变更 |
|---|---|
| `frontend/src/views/AgentChat/index.vue` | 集成快捷键、命令面板、模型选择器 |
| `frontend/src/composables/useAgent.ts` | 处理 permission_request、system 事件 |
| `frontend/src/stores/agent.ts` | 添加 systemMessages 状态 |
| `QuantNodes/agent/core/loop.py` | Auto Compact 逻辑 |
| `api/routers/agent.py` | 会话导出 API |

---

## 十、执行顺序

**P0-1 → P0-2 → P1-2 → P1-1 → P2-2 → P2-1**

---

## 十一、风险与注意事项

1. **快捷键冲突**: 浏览器/系统可能占用相同快捷键
2. **Auto Compact 信息丢失**: 摘要可能丢失细节
3. **权限确认影响体验**: 需要 "Remember for session" 选项
4. **模型元数据维护**: 可从 OpenRouter API 动态获取
