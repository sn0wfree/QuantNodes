# AgentChat Layout Refactor — OpenCode 风格沉浸式布局

> 版本: v1.0
> 日期: 2026-05-12
> 方案: A1 — 可折叠侧栏独立全屏布局
> 参考: OpenCode TUI + Claude.ai / ChatGPT Web

---

## 一、目标

将 AgentChat 从 AppLayout 嵌套模式改为**独立全屏布局**，采用 OpenCode 风格的沉浸式聊天体验：

- 无全局 AppHeader，仅有极简顶栏
- 可折叠侧栏（收起 48px 图标列 / 展开 200px 完整导航）
- 用户消息改为左侧边框块（非气泡）
- 工具调用内联展示
- 底部状态栏 + 快捷键提示条
- 输入区显示 agent/model 上下文

## 二、目标布局

```
┌─────────────────────────────────────────────────────────────┐
│ ☰  QuantNodes                         🔍  ⚙  ≡            │  ← 顶栏 40px
├────┬────────────────────────────────────────────────────────┤
│    │  # Default Session                 2,048 tokens $0.03  │  ← Session 标题栏 36px
│ 🏠 │────────────────────────────────────────────────────────│
│    │                                                         │
│ 💬 │  ┃ Find the homepage button and make it blue            │  ← 用户消息边块
│    │                                                         │
│ 📖 │  I'll search for the homepage button.                   │  ← 助手纯文本
│    │                                                         │
│ 📝 │    * Grep "homepage|home.*button"                       │  ← 内联工具调用
│    │    * Grep "Homepage"                                    │
│ 📊 │                                                         │
│    │    → Read packages/console/app/src/routes/[...404].tsx  │  ← 文件操作
│ 📈 │                                                         │
│ 💡 │  Found "Home" buttons/links in multiple locations.      │
│    │                                                         │
│ ⚙  │  ┌─ 🔧 grep ── success ──────────────────────────┐    │  ← 紧凑 ToolCallCard
│    │  │ click to expand                                │    │
├────┼────────────────────────────────────────────────────────┤
│    │  agent: Build   model: Claude Opus 4.5   $0.03        │  ← 状态栏 28px
├────┼────────────────────────────────────────────────────────┤
│    │  ┃ ▎ Build · Claude Opus 4.5                   Enter ⏎ │  ← 输入区
├────┼────────────────────────────────────────────────────────┤
│    │  esc interrupt   ctrl+k commands   ctrl+n new           │  ← 快捷键提示 28px
└────┴────────────────────────────────────────────────────────┘
   48px
```

## 三、文件变更清单

### 新增文件（4 个）

| 文件 | 说明 |
|---|---|
| `components/Layout/ChatLayout.vue` | AgentChat 独立布局（可折叠侧栏 + 顶栏 + 插槽） |
| `components/Chat/ChatNavSidebar.vue` | 导航+会话侧栏（可折叠） |
| `components/Chat/ChatStatusBar.vue` | 底部状态栏（agent/model/cost） |
| `components/Chat/ChatKeybindHints.vue` | 底部快捷键提示条 |

### 修改文件（9 个）

| 文件 | 变更 |
|---|---|
| `router/index.ts` | AgentChat 路由使用 ChatLayout |
| `views/AgentChat/index.vue` | 套用 ChatLayout，调整内部结构 |
| `components/Chat/ChatHeader.vue` | 重写为 Session 标题栏（精简） |
| `components/Chat/ChatMessage.vue` | 用户消息改为左侧边框块 |
| `components/Chat/MessageList.vue` | 工具调用内联展示 |
| `components/Chat/ToolCallCard.vue` | 更紧凑，默认折叠 |
| `components/Chat/ChatInput.vue` | 增加 agent/model 上下文显示 |
| `components/Chat/EmptyState.vue` | 重写为欢迎界面 |
| `stores/app.ts` | 添加 chatSidebarCollapsed 状态 |

---

## 四、详细实现计划

### Phase 1: 布局基础设施

#### 1.1 ChatLayout.vue（新增）

AgentChat 的独立布局容器，替代 AppLayout。

```
结构:
<div class="chat-layout">
  <ChatNavSidebar />           ← 可折叠导航侧栏
  <div class="chat-main">
    <div class="chat-topbar">  ← 极简顶栏
      <div class="topbar-left">☰ + Logo</div>
      <div class="topbar-right">🔍 + ⚙</div>
    </div>
    <div class="chat-body">
      <slot />                 ← AgentChat 内容
    </div>
  </div>
</div>
```

CSS 要点:
- 全屏: `height: 100vh; display: flex`
- 侧栏收起时: `width: 48px`，展开时: `width: 200px`
- 过渡动画: `transition: width 0.2s ease`
- 主区域: `flex: 1; display: flex; flex-direction: column; overflow: hidden`

#### 1.2 ChatNavSidebar.vue（新增）

导航 + 会话管理侧栏。

```
收起状态 (48px):
┌────┐
│ ☰  │  ← 汉堡菜单图标
│ 🏠 │  ← Dashboard
│ 💬 │  ← Agent Chat (高亮)
│ 📖 │  ← Wiki
│ 📝 │  ← Strategy Editor
│ 📊 │  ← Backtest
│ 📈 │  ← Factor Analysis
│ 💡 │  ← Dream Insights
│ ⚙  │  ← Settings
│    │
│ 📋 │  ← Sessions (底部)
└────┘

展开状态 (200px):
┌──────────────────┐
│ ☰  QuantNodes    │
│ 🏠 Dashboard     │
│ 💬 Agent Chat    │
│ 📖 Wiki          │
│   ├ Factors      │
│   └ Strategies   │
│ 📝 Strategy Editor│
│ 📊 Backtest      │
│ 📈 Factor Analysis│
│ 💡 Dream Insights │
│ ⚙ Settings       │
│──────────────────│
│ 📋 Sessions      │
│ ├ Default (3)    │
│ ├ Test-1 (5)     │
│ └ Test-2 (1)     │
│   [+ New Chat]   │
└──────────────────┘
```

交互:
- 点击 ☰ → 切换展开/收起
- 点击导航项 → router.push + 收起（可选）
- 点击会话 → switchSession
- 点击 [+ New Chat] → createSession

#### 1.3 router/index.ts 修改

将 AgentChat 从 AppLayout 子路由分离：

```typescript
const routes = [
  {
    path: '/',
    component: () => import('@/components/Layout/AppLayout.vue'),
    children: [
      { path: '', name: 'Dashboard', ... },
      // AgentChat 移除
      { path: 'wiki/factors', ... },
      // ... 其他路由不变
    ],
  },
  {
    path: '/chat',
    component: () => import('@/components/Layout/ChatLayout.vue'),
    children: [
      {
        path: '',
        name: 'AgentChat',
        component: () => import('@/views/AgentChat/index.vue'),
        meta: { title: 'Agent Chat' },
      },
    ],
  },
  { path: '/:pathMatch(.*)*', name: 'NotFound', ... },
]
```

### Phase 2: 消息展示改造

#### 2.1 ChatMessage.vue 修改

用户消息从右对齐蓝色气泡改为左侧边框块：

```css
/* 之前 */
.chat-message.user { flex-direction: row-reverse; }
.user .bubble { background: #1677ff; color: #fff; }

/* 之后 */
.chat-message.user { flex-direction: row; }
.user .bubble {
  background: transparent;
  border-left: 3px solid #1677ff;
  padding-left: 12px;
  color: inherit;
}
```

#### 2.2 MessageList.vue 修改

工具调用从独立 section 改为内联展示：

```typescript
// enrichedMessages 计算属性合并 messages + toolCalls
const enrichedMessages = computed(() => {
  const items = []
  for (const msg of props.messages) {
    items.push({ type: 'message', data: msg, timestamp: msg.timestamp })
  }
  for (const tc of props.toolCalls) {
    items.push({ type: 'tool_call', data: tc, timestamp: tc.timestamp || 0 })
  }
  return items.sort((a, b) => a.timestamp - b.timestamp)
})
```

#### 2.3 ToolCallCard.vue 修改

更紧凑的默认样式，默认折叠只显示一行摘要。

### Phase 3: 状态栏与输入区

#### 3.1 ChatStatusBar.vue（新增）

```
agent: Build   model: Claude Opus 4.5   tokens: 2,048   cost: $0.03
```

高度: 28px

#### 3.2 ChatInput.vue 修改

增加 agent/model 上下文显示（placeholder 或输入框上方灰色小字）。

#### 3.3 ChatKeybindHints.vue（新增）

底部快捷键提示条，根据状态动态显示：
- 流式输出时: `esc interrupt`
- 空闲时: `ctrl+k commands   ctrl+n new   ctrl+o model`

高度: 28px

### Phase 4: Header 与 EmptyState

#### 4.1 ChatHeader.vue 重写

从按钮组改为 Session 标题栏：

```
# Default Session                     2,048 tokens $0.03
```

高度: 36px

#### 4.2 EmptyState.vue 重写

欢迎界面 + 常用命令建议 + 快捷键提示。

### Phase 5: stores/app.ts 修改

新增 `chatSidebarCollapsed` 状态和 `toggleChatSidebar()` action。

---

## 五、CSS 变量约定

```css
.chat-layout {
  --chat-sidebar-width: 200px;
  --chat-sidebar-collapsed: 48px;
  --chat-topbar-height: 40px;
  --chat-header-height: 36px;
  --chat-statusbar-height: 28px;
  --chat-keybinds-height: 28px;
  --chat-border-color: #f0f0f0;
  --chat-bg-primary: #ffffff;
  --chat-bg-secondary: #fafafa;
  --chat-text-primary: #333333;
  --chat-text-secondary: #666666;
  --chat-text-muted: #999999;
}
```

暗色模式通过 `:global(html[data-theme="dark"])` 覆盖这些变量。

## 六、执行顺序

| 步骤 | 阶段 | 说明 | 验证 |
|---|---|---|---|
| 1 | Phase 1 | ChatLayout + ChatNavSidebar + router | `npm run build` |
| 2 | Phase 2 | ChatMessage + MessageList + ToolCallCard | `npm run build` |
| 3 | Phase 3 | ChatStatusBar + ChatInput + ChatKeybindHints | `npm run build` |
| 4 | Phase 4 | ChatHeader 重写 + EmptyState 重写 | `npm run build` |
| 5 | Phase 5 | stores/app.ts 修改 | `npm run build` |
| 6 | 测试 | 全量测试 + 暗色模式验证 | `npm run test` |

## 七、风险与注意事项

1. **路由切换无闪烁** — AgentChat 独立 layout，lazy load
2. **WebSocket 不受影响** — useAgent composable 与 layout 解耦
3. **其他页面不受影响** — Dashboard, Wiki, Backtest 等仍用 AppLayout
4. **暗色模式** — 所有新组件用 CSS 变量支持
5. **响应式** — 小屏幕下侧栏默认收起
