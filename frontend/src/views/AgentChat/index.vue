<template>
  <div class="agent-chat">
    <ChatHeader
      :currentSessionLabel="currentSessionLabel"
      :tokenCount="tokenCount"
    />

    <MessageList
      ref="messageListRef"
      :messages="session.store.messages"
      :toolCalls="agent.currentToolCalls.value"
      :isStreaming="agent.isStreaming.value"
      :streamContent="agent.streamContent.value"
      @send="handleSend"
    />

    <ChatInput
      class="chat-input-fixed"
      :disabled="agent.isStreaming.value"
      :agentName="agentName"
      :modelName="currentModelLabel"
      @send="handleSend"
    />

    <ChatStatusBar
      :agentName="agentName"
      :modelName="currentModelLabel"
      :tokenCount="tokenCount"
    />

    <ChatKeybindHints :isStreaming="agent.isStreaming.value" />

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
import ChatHeader from '@/components/Chat/ChatHeader.vue'
import MessageList from '@/components/Chat/MessageList.vue'
import ChatInput from '@/components/Chat/ChatInput.vue'
import ChatStatusBar from '@/components/Chat/ChatStatusBar.vue'
import ChatKeybindHints from '@/components/Chat/ChatKeybindHints.vue'
import ModelSelector from '@/components/Chat/ModelSelector.vue'
import PermissionDialog from '@/components/Chat/PermissionDialog.vue'
import { get, put } from '@/api'

const store = useAgentStore()
const agent = useAgent()
const session = useChatSession()
const messageListRef = ref()

const showModelSelector = ref(false)

const currentSessionLabel = session.currentSessionLabel
const agentName = computed(() => store.currentMode === 'plan' ? 'Plan' : 'Build')
const tokenCount = computed(() => {
  const msgs = session.store.messages
  return msgs.length * 500
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
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--chat-bg-primary, #fff);
  overflow: hidden;
}

.chat-input-fixed {
  flex-shrink: 0;
}
</style>
