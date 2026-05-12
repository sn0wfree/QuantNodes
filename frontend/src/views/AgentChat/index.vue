<template>
  <div class="agent-chat">
    <div class="chat-header">
      <div class="header-left">
        <span class="status-dot" :class="{ connected: agent.isConnected.value }"></span>
        <a-dropdown :trigger="['click']">
          <a-button type="text" class="session-dropdown">
            {{ currentSessionLabel }}
            <template #icon><down-outlined /></template>
          </a-button>
          <template #overlay>
            <a-menu @click="handleMenuClick">
              <a-menu-item key="new">
                <template #icon><plus-outlined /></template>
                New Chat
              </a-menu-item>
              <a-menu-divider v-if="store.sessions.length" />
              <a-menu-item
                v-for="s in store.sessions"
                :key="s.session_id"
                :class="{ 'session-active': s.session_id === store.sessionId }"
              >
                <div class="session-item">
                  <span class="session-name">{{ s.session_id }}</span>
                  <span class="session-count">{{ s.message_count }} msgs</span>
                </div>
              </a-menu-item>
            </a-menu>
          </template>
        </a-dropdown>
      </div>
      <div class="header-right">
        <a-select
          v-model:value="store.currentModel"
          size="small"
          style="width: 220px"
          :options="modelOptions"
          @change="handleModelChange"
        />
        <a-button type="text" size="small" @click="handleNewChat">
          <template #icon><plus-outlined /></template>
        </a-button>
      </div>
    </div>

    <div class="chat-container">
      <div class="messages" ref="messagesContainer">
        <ChatMessage
          v-for="message in store.messages"
          :key="message.id"
          :role="message.role"
          :content="message.role === 'assistant' ? message.content : undefined"
          :time="formatTime(message.timestamp)"
        >
          <template v-if="message.role === 'user'">{{ message.content }}</template>
        </ChatMessage>

        <template v-if="store.messages.length">
          <div class="tool-calls-section" v-if="agent.currentToolCalls.value.length > 0">
            <ToolCallCard
              v-for="tc in agent.currentToolCalls.value"
              :key="tc.id"
              :toolName="tc.name"
              :arguments="tc.arguments"
              :result="tc.result ? { output: tc.result } : undefined"
              :status="tc.status"
            />
          </div>
        </template>

        <ChatMessage v-if="agent.isStreaming.value" role="assistant" :content="agent.streamContent.value">
          <template v-if="!agent.streamContent.value">
            <span class="typing-indicator">
              <span class="dot"></span>
              <span class="dot"></span>
              <span class="dot"></span>
            </span>
          </template>
        </ChatMessage>

        <a-empty v-if="!store.messages.length && !agent.isStreaming.value" description="Start a conversation with the Agent" />
      </div>

      <div class="input-area">
        <ChatInput :disabled="agent.isStreaming.value" @send="handleSend" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, watch, onMounted, onUnmounted } from 'vue'
import { useAgentStore } from '@/stores/agent'
import { useAgent } from '@/composables/useAgent'
import ChatMessage from '@/components/Chat/ChatMessage.vue'
import ChatInput from '@/components/Chat/ChatInput.vue'
import ToolCallCard from '@/components/Chat/ToolCallCard.vue'
import { PlusOutlined, DownOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import { get, put } from '@/api'
import dayjs from 'dayjs'

const store = useAgentStore()
const agent = useAgent()
const messagesContainer = ref<HTMLElement>()

const currentSessionLabel = computed(() => {
  if (store.sessionId === 'default') return 'Default Session'
  return store.sessionId
})

const modelOptions = [
  { label: 'MiniMax M2.5 (Free)', value: 'minimax/minimax-m2.5:free' },
  { label: 'MiniMax M2.5', value: 'minimax/minimax-m2.5' },
  { label: 'MiniMax M2.7', value: 'minimax/minimax-m2.7' },
  { label: 'GPT-4o', value: 'openai/gpt-4o' },
  { label: 'GPT-4o Mini', value: 'openai/gpt-4o-mini' },
]

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}

const formatTime = (timestamp: number) => {
  return dayjs(timestamp).format('HH:mm')
}

const handleSend = (content: string) => {
  agent.sendMessage(content)
}

const handleNewChat = async () => {
  await store.createSession()
}

const handleModelChange = async (model: string) => {
  try {
    await put('/settings', { settings: { agent: { model } } })
    message.success(`Switched to ${model}`)
  } catch {
    message.error('Failed to switch model')
  }
}

const handleMenuClick = async ({ key }: { key: string }) => {
  if (key === 'new') {
    await store.createSession()
  } else {
    await store.switchSession(key)
  }
}

watch(
  () => store.messages.length,
  () => {
    scrollToBottom()
  }
)

watch(
  () => agent.streamContent.value,
  () => {
    scrollToBottom()
  }
)

watch(
  () => agent.currentToolCalls.value.length,
  () => {
    scrollToBottom()
  }
)

watch(
  () => agent.isStreaming.value,
  (isStreaming, wasStreaming) => {
    if (wasStreaming && !isStreaming) {
      scrollToBottom()
    }
  }
)

onMounted(async () => {
  agent.connect()
  store.loadSessions()
  scrollToBottom()
  try {
    const settings = await get<any>('/settings')
    if (settings?.agent?.model) {
      store.currentModel = settings.agent.model
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

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #d9d9d9;
}

.status-dot.connected {
  background: #52c41a;
}

.session-dropdown {
  font-weight: 500;
  font-size: 15px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.session-name {
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-count {
  font-size: 12px;
  color: #999;
}

.session-active {
  background: #e6f4ff;
}

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.tool-calls-section {
  margin: 8px 0;
  padding-left: 44px;
}

.input-area {
  padding: 16px;
  border-top: 1px solid #f0f0f0;
}

.typing-indicator {
  display: inline-flex;
  gap: 4px;
  padding: 4px 0;
}

.typing-indicator .dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #999;
  animation: typing 1.4s infinite;
}

.typing-indicator .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 60%, 100% {
    transform: translateY(0);
    opacity: 0.4;
  }
  30% {
    transform: translateY(-4px);
    opacity: 1;
  }
}
</style>
