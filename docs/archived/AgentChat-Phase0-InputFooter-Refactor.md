# Phase 0: Input Footer + Build/Plan Visual Redesign

## 目标

将 Build/Plan 模式切换从 ChatHeader 下沉到输入区底部，统一分散的状态信息，增加模式视觉区分。

## 关键改动

### 1. 模式切换位置重构

```
Before:                          After:
┌─ ChatHeader ─────────────┐    ┌─ ChatHeader ────────────────┐
│  [Build | Plan]   1.7k   │    │  [会话名]  [压缩][分享][▸]  │
└──────────────────────────┘    └────────────────────────────┘
│ 消息列表                     │ 消息列表
┌─ ChatInput ──────────┐      ┌─ ChatInput ──────────────────┐
│  Build · MiMo        │      │  输入消息...                  │
│  [textarea]    [→]   │      │  [+]                [发送 →] │
└──────────────────────┘      ├──────────────────────────────┤
│  StatusBar: agent/model     │  ● Build ⚡ MiMo-V2.5 High ▾ │
│  KeybindHints: ctrl+k/n     └──────────────────────────────┘
```

### 2. 组件职责重分配

| 组件 | Before | After |
|------|--------|-------|
| ChatHeader | Build/Plan segmented + token count | Session name + action buttons |
| ChatInput | textarea + send + context label | textarea + send + **InputFooter** |
| ChatInputFooter *(new)* | — | Agent indicator · Model · Quality · Tokens |
| ChatStatusBar | agent · model · tokens · cost | **Removed** (info merged into footer) |
| ChatKeybindHints | ctrl+k · ctrl+n · ctrl+o · ctrl+b/p | **Removed** (merged into footer as subtle text) |

### 3. Agent 视觉标识

| Agent | Color | Icon | Label |
|-------|-------|------|-------|
| Build | `#1677ff` blue | ⚡ | Build |
| Plan | `#52c41a` green | 📋 | Plan |

### 4. 质量级别

| Level | Description | temperature |
|-------|-------------|-------------|
| High | High quality (default) | 0.1 |
| Medium | Balanced | 0.5 |
| Low | Fast | 0.8, half max_tokens |

## 涉及文件

- `frontend/src/components/Chat/ChatInputFooter.vue` — **NEW**
- `frontend/src/components/Chat/ChatInput.vue` — integrate footer, remove context
- `frontend/src/components/Chat/ChatHeader.vue` — remove segmented, add session actions
- `frontend/src/components/Chat/ChatStatusBar.vue` — simplify/remove
- `frontend/src/components/Chat/ChatKeybindHints.vue` — simplify/remove
- `frontend/src/views/AgentChat/index.vue` — adjust component tree
- `frontend/src/App.vue` — extend CSS custom properties
- `frontend/src/components/Chat/ModelSelector.vue` — remove dead `mode` prop
- `frontend/src/stores/agent.ts` — add quality level state

## 测试

- E2E: 模式切换后输入区底部指示器同步变化
- E2E: Header 不再显示 segmented 控制
- E2E: 质量级别切换影响 temperature
- Unit: 783 tests must remain green
