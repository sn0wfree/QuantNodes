# AgentChat 页面重构方案

## 现状

`AgentChat/index.vue` 有 389 行，承担 15+ 项职责（布局、Session管理、消息渲染、滚动管理、快捷键、命令注册、模型选择、导出等），是典型的"上帝组件"。

## 目标

将单文件拆分为职责清晰的小组件 + composable，最大文件从 389 行降至 ~120 行。

## 组件拆分

### 重构前
```
AgentChat/index.vue (389行)
├── ChatInput.vue        (56行)  ✓
├── ChatMessage.vue      (110行) ✓
├── ToolCallCard.vue     (137行) ✓
├── MarkdownRenderer.vue (169行) ✓
├── CommandPalette.vue   (191行) ✓
├── ModelSelector.vue    (206行) ✓
├── PermissionDialog.vue (157行) ✓
├── useAgent.ts          (139行) ✓
└── useWebSocket.ts      (114行) ✓
```

### 重构后
```
views/AgentChat/
├── index.vue                 (~80行)  薄壳：布局 + 生命周期
├── ChatHeader.vue            (~100行) 顶栏：Session + 模型 + 动作
├── MessageList.vue           (~120行) 消息容器：渲染 + 滚动
├── EmptyState.vue            (~40行)  空状态
└── StreamingIndicator.vue    (~30行)  打字动画

composables/ (新增)
├── useChatScroll.ts          (~60行)  滚动管理
└── useChatSession.ts         (~50行)  Session CRUD + 导出
```

### 各组件职责

| 组件 | 职责 | 行数 |
|---|---|---|
| index.vue | 布局编排、WebSocket生命周期、快捷键、命令注册 | ~80 |
| ChatHeader.vue | 状态指示灯、Session下拉、模型选择、命令面板、新建 | ~100 |
| MessageList.vue | 消息遍历、ToolCall展示、流式指示、空状态、滚动 | ~120 |
| EmptyState.vue | 欢迎语、推荐提问 | ~40 |
| StreamingIndicator.vue | 打字动画三个圆点 | ~30 |
| useChatScroll.ts | isAtBottom追踪、scrollToBottom | ~60 |
| useChatSession.ts | create/switch/delete/export | ~50 |

### 不变的文件
- ChatInput.vue (56行)
- ChatMessage.vue (110行)
- ToolCallCard.vue (137行)
- MarkdownRenderer.vue (169行)
- CommandPalette.vue (191行)
- ModelSelector.vue (206行)
- PermissionDialog.vue (157行)
- useAgent.ts (139行)
- useWebSocket.ts (114行)

## Commit 计划

| # | Commit | 内容 |
|---|---|---|
| 1 | `refactor: extract useChatScroll composable` | 滚动管理 |
| 2 | `refactor: extract useChatSession composable` | Session CRUD + 导出 |
| 3 | `refactor: extract ChatHeader component` | 顶栏 |
| 4 | `refactor: extract MessageList component` | 消息列表 |
| 5 | `refactor: extract EmptyState and StreamingIndicator` | 小UI组件 |
| 6 | `refactor: simplify AgentChat/index.vue to shell` | 最终组装 |
