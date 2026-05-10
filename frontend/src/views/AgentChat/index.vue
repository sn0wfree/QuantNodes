<template>
  <div class="agent-chat">
    <div class="chat-header">
      <div class="header-left">
        <span class="status-dot" :class="{ connected: agent.isConnected.value }"></span>
        <span class="title">Agent</span>
      </div>
      <div class="header-right">
        <a-button type="text" size="small" @click="handleClear">
          <template #icon><delete-outlined /></template>
          New Chat
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
import { ref, nextTick, watch, onMounted, onUnmounted } from 'vue'
import { useAgentStore } from '@/stores/agent'
import { useAgent } from '@/composables/useAgent'
import ChatMessage from '@/components/Chat/ChatMessage.vue'
import ChatInput from '@/components/Chat/ChatInput.vue'
import ToolCallCard from '@/components/Chat/ToolCallCard.vue'
import { DeleteOutlined } from '@ant-design/icons-vue'
import dayjs from 'dayjs'

const store = useAgentStore()
const agent = useAgent()
const messagesContainer = ref<HTMLElement>()

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

const handleClear = () => {
  store.clearMessages()
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

onMounted(() => {
  agent.connect()
  scrollToBottom()
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

.title {
  font-weight: 500;
  font-size: 15px;
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
