<template>
  <div class="agent-chat">
    <div class="chat-container">
      <div class="messages" ref="messagesContainer">
        <ChatMessage
          v-for="message in store.messages"
          :key="message.id"
          :role="message.role"
        >
          {{ message.content }}
        </ChatMessage>

        <ChatMessage v-if="agent.isStreaming.value" role="assistant">
          {{ agent.streamContent.value }}
          <span class="typing-indicator">|</span>
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

const handleSend = (content: string) => {
  agent.sendMessage(content)
}

// Watch message count for new messages
watch(
  () => store.messages.length,
  () => {
    scrollToBottom()
  }
)

// Watch stream content for auto-scroll during streaming
watch(
  () => agent.streamContent.value,
  () => {
    scrollToBottom()
  }
)

// Watch streaming state to scroll when streaming ends
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

.input-area {
  padding: 16px;
  border-top: 1px solid #f0f0f0;
}

.typing-indicator {
  animation: blink 1s infinite;
}

@keyframes blink {
  0%, 50% {
    opacity: 1;
  }
  51%, 100% {
    opacity: 0;
  }
}
</style>
