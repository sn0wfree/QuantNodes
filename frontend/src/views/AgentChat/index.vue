<template>
  <div class="agent-chat">
    <MessageList
      ref="messageListRef"
      :messages="session.store.messages"
      :toolCalls="agent.currentToolCalls.value"
      :isStreaming="agent.isStreaming.value"
      :streamContent="agent.streamContent.value"
      :mode="store.currentMode"
      @send="handleSend"
    />

    <ChatInput
      class="chat-input-fixed"
      :disabled="agent.isStreaming.value"
      :modelName="currentModelLabel"
      :currentMode="store.currentMode"
      :quality="store.quality"
      :tokenCount="tokenCount"
      @send="handleSend"
      @toggleMode="handleToggleMode"
      @openModelSelector="showModelSelector = true"
      @qualityChange="handleQualityChange"
    />

    <div class="chat-bottombar">
      <span class="status-text">{{ statusText }}</span>
      <span class="hints">
        <template v-if="agent.isStreaming.value">
          <span class="hint"><kbd>esc</kbd> interrupt</span>
        </template>
        <template v-else>
          <span class="hint"><kbd>ctrl+k</kbd> commands</span>
          <span class="hint"><kbd>ctrl+n</kbd> new</span>
        </template>
      </span>
    </div>

    <Transition name="panel-slide">
      <div v-if="appStore.contextPanelOpen" class="panels-container">
        <div class="panels-inner" :style="{ width: (appStore.contextPanelWidth + appStore.toolsPanelWidth) + 'px' }">
          <ChatContextPanel :open="true" />
          <div class="resize-handle" @mousedown.prevent="startToolsResize" />
          <ToolsPanel v-if="appStore.toolsPanelOpen" />
        </div>
      </div>
    </Transition>

    <ModelSelector
      :open="showModelSelector"
      :currentModel="session.store.currentModel"
      @select="handleModelSelect"
      @close="showModelSelector = false"
    />
    <PermissionDialog
      v-if="agent.pendingPermission.value"
      :open="true"
      :toolName="agent.pendingPermission.value.toolName"
      :arguments="agent.pendingPermission.value.arguments"
      :requestId="agent.pendingPermission.value.requestId"
      @allow="(id, remember) => agent.respondPermission(id, true, remember)"
      @deny="(id) => agent.respondPermission(id, false)"
      @close="agent.respondPermission(agent.pendingPermission.value!.requestId, false)"
    />
    <CommandPalette :open="showCommandPalette" @close="showCommandPalette = false" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import { useAgent } from '@/composables/useAgent'
import { useKeyboard } from '@/composables/useKeyboard'
import { useCommands } from '@/composables/useCommands'
import { useChatSession } from '@/composables/useChatSession'
import { useAgentStore } from '@/stores/agent'
import { useAppStore } from '@/stores/app'
import MessageList from '@/components/Chat/MessageList.vue'
import ChatInput from '@/components/Chat/ChatInput.vue'
import ChatContextPanel from '@/components/Chat/ChatContextPanel.vue'
import ToolsPanel from '@/components/Chat/ToolsPanel.vue'
import ModelSelector from '@/components/Chat/ModelSelector.vue'
import PermissionDialog from '@/components/Chat/PermissionDialog.vue'
import CommandPalette from '@/components/Chat/CommandPalette.vue'
import { get, put } from '@/api'

const store = useAgentStore()
const appStore = useAppStore()
const agent = useAgent()
const session = useChatSession()
const messageListRef = ref()
const showCommandPalette = ref(false)

const showModelSelector = ref(false)

const tokenCount = computed(() => {
  const msgs = session.store.messages
  return msgs.length * 500
})

const statusText = computed(() => {
  if (agent.isStreaming.value) return 'Thinking...'
  if (!session.store.messages.length) return ''
  return `${session.store.messages.length} messages`
})

const currentModelLabel = computed(() => {
  const model = session.store.currentModel
  if (!model) return 'Unknown'
  const parts = model.split('/')
  return parts.length > 1 ? parts[1] : model
})

const handleSend = (content: string) => {
  agent.sendMessage(content)
}

const handleToggleMode = () => {
  const next = store.currentMode === 'build' ? 'plan' : 'build'
  store.switchMode(next)
}

const handleQualityChange = (q: string) => {
  store.quality = q as 'high' | 'medium' | 'low'
}

const handleModelSelect = async (model: string) => {
  session.store.currentModel = model
  try {
    await put('/settings', { settings: { agent: { model } } })
    message.success(`Switched to ${model}`)
  } catch {
    message.error('Failed to switch model')
  }
}

// Keyboard shortcuts
const keyboard = useKeyboard()
keyboard.register('ctrl+o', () => { showModelSelector.value = true })
keyboard.register('ctrl+n', () => { session.createSession() })
keyboard.register('ctrl+l', () => { session.clearHistory() })
keyboard.register('escape', () => { showModelSelector.value = false })
keyboard.register('ctrl+shift+b', () => { store.switchMode('build') })
keyboard.register('ctrl+shift+p', () => { store.switchMode('plan') })

// Command palette commands
const { register: registerCommand } = useCommands()
registerCommand({ id: 'new-chat', label: 'New Chat', group: 'Sessions', shortcut: 'Ctrl+N', action: () => { session.createSession() } })
registerCommand({ id: 'clear-history', label: 'Clear History', group: 'Sessions', action: session.clearHistory })
registerCommand({ id: 'export-md', label: 'Export as Markdown', group: 'Sessions', action: () => session.exportSession('markdown') })
registerCommand({ id: 'export-json', label: 'Export as JSON', group: 'Sessions', action: () => session.exportSession('json') })
registerCommand({ id: 'switch-model', label: 'Switch Model...', group: 'Model', shortcut: 'Ctrl+O', action: () => { showModelSelector.value = true } })
registerCommand({ id: 'mode-build', label: 'Switch to Build Mode', group: 'Mode', shortcut: 'Ctrl+Shift+B', action: () => { store.switchMode('build') } })
registerCommand({ id: 'mode-plan', label: 'Switch to Plan Mode', group: 'Mode', shortcut: 'Ctrl+Shift+P', action: () => { store.switchMode('plan') } })

// Resize handlers
function startToolsResize(e: MouseEvent) {
  const startX = e.clientX
  const startWidth = appStore.toolsPanelWidth

  const onMouseMove = (ev: MouseEvent) => {
    const delta = ev.clientX - startX
    appStore.toolsPanelWidth = Math.max(200, Math.min(450, startWidth + delta))
  }

  const onMouseUp = () => {
    document.removeEventListener('mousemove', onMouseMove)
    document.removeEventListener('mouseup', onMouseUp)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    document.body.style.pointerEvents = ''
  }

  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  document.body.style.pointerEvents = 'none'
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
}

onMounted(async () => {
  agent.connect()
  session.store.loadSessions()
  session.store.loadModeModels()
  try {
    const settings = await get<any>('/settings')
    if (settings?.agent?.model) {
      session.store.currentModel = settings.agent.model
    }
  } catch {
    // use default model
  }
})

onUnmounted(() => {
  agent.disconnect()
})
</script>

<style scoped>
.agent-chat {
  position: relative;
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--chat-bg-primary, #fff);
  overflow: hidden;
}

.chat-input-fixed {
  flex-shrink: 0;
  position: relative;
  z-index: 10;
}

.chat-bottombar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 24px;
  padding: 0 16px;
  font-size: 11px;
  color: var(--chat-text-muted, #999);
  flex-shrink: 0;
  position: relative;
  z-index: 10;
}

.status-text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.hints {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}

.hint {
  display: flex;
  align-items: center;
  gap: 4px;
}

kbd {
  display: inline-block;
  padding: 0 4px;
  font-size: 10px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--chat-text-secondary, #666);
  background: var(--chat-bg-primary, #fff);
  border: 1px solid var(--chat-border-color, #d9d9d9);
  border-radius: 3px;
}

/* Panels overlay */
.panels-container {
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  display: flex;
  z-index: 100;
}

.panels-inner {
  display: flex;
  flex-shrink: 0;
  overflow: hidden;
}

.resize-handle {
  width: 6px;
  cursor: col-resize;
  flex-shrink: 0;
  background: transparent;
  transition: background 0.15s;
}

.resize-handle:hover {
  background: var(--chat-border-active, #1677ff);
  opacity: 0.3;
}

/* Panel slide transition */
.panel-slide-enter-active,
.panel-slide-leave-active {
  transition: transform 0.2s ease-out, opacity 0.2s ease-out;
}

.panel-slide-enter-from,
.panel-slide-leave-to {
  transform: translateX(100%);
  opacity: 0;
}
</style>