<template>
  <div class="agent-chat">
    <ChatHeader
      :isConnected="agent.isConnected.value"
      :currentSessionLabel="currentSessionLabel"
      :sessions="session.store.sessions"
      :activeSessionId="session.store.sessionId"
      @openModelSelector="showModelSelector = true"
      @openCommandPalette="showCommandPalette = true"
      @newChat="handleNewChat"
      @menuClick="session.handleMenuClick"
    />

    <div class="chat-container">
      <MessageList
        ref="messageListRef"
        :messages="session.store.messages"
        :toolCalls="agent.currentToolCalls.value"
        :isStreaming="agent.isStreaming.value"
        :streamContent="agent.streamContent.value"
      />

      <div class="input-area">
        <ChatInput :disabled="agent.isStreaming.value" @send="handleSend" />
      </div>
    </div>

    <CommandPalette :open="showCommandPalette" @close="showCommandPalette = false" />
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
import { ref, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import { useAgent } from '@/composables/useAgent'
import { useKeyboard } from '@/composables/useKeyboard'
import { useCommands } from '@/composables/useCommands'
import { useChatSession } from '@/composables/useChatSession'
import ChatHeader from '@/components/Chat/ChatHeader.vue'
import MessageList from '@/components/Chat/MessageList.vue'
import ChatInput from '@/components/Chat/ChatInput.vue'
import CommandPalette from '@/components/Chat/CommandPalette.vue'
import ModelSelector from '@/components/Chat/ModelSelector.vue'
import PermissionDialog from '@/components/Chat/PermissionDialog.vue'
import { get, put } from '@/api'

const agent = useAgent()
const session = useChatSession()
const messageListRef = ref()

const showCommandPalette = ref(false)
const showModelSelector = ref(false)

const currentSessionLabel = session.currentSessionLabel

const handleSend = (content: string) => {
  agent.sendMessage(content)
}

const handleNewChat = async () => {
  await session.createSession()
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
keyboard.register('ctrl+k', () => { showCommandPalette.value = true })
keyboard.register('ctrl+o', () => { showModelSelector.value = true })
keyboard.register('ctrl+n', () => { handleNewChat() })
keyboard.register('ctrl+l', () => { session.clearHistory() })
keyboard.register('escape', () => {
  showCommandPalette.value = false
  showModelSelector.value = false
})

// Command palette commands
const { register: registerCommand } = useCommands()
registerCommand({ id: 'new-chat', label: 'New Chat', group: 'Sessions', shortcut: 'Ctrl+N', action: handleNewChat })
registerCommand({ id: 'clear-history', label: 'Clear History', group: 'Sessions', action: session.clearHistory })
registerCommand({ id: 'export-md', label: 'Export as Markdown', group: 'Sessions', action: () => session.exportSession('markdown') })
registerCommand({ id: 'export-json', label: 'Export as JSON', group: 'Sessions', action: () => session.exportSession('json') })
registerCommand({ id: 'switch-model', label: 'Switch Model...', group: 'Model', shortcut: 'Ctrl+O', action: () => { showModelSelector.value = true } })
registerCommand({ id: 'cmd-palette', label: 'Command Palette', group: 'View', shortcut: 'Ctrl+K', action: () => { showCommandPalette.value = true } })

onMounted(async () => {
  agent.connect()
  session.store.loadSessions()
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
  background: #fff;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.input-area {
  padding: 16px;
  border-top: 1px solid #f0f0f0;
}
</style>
