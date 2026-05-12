<template>
  <div class="messages" ref="messagesContainer" @scroll="handleScroll">
    <ChatMessage
      v-for="message in messages"
      :key="message.id"
      :role="message.role"
      :content="message.role === 'assistant' ? message.content : undefined"
      :time="formatTime(message.timestamp)"
    >
      <template v-if="message.role === 'user'">{{ message.content }}</template>
    </ChatMessage>

    <template v-if="messages.length">
      <div class="tool-calls-section" v-if="toolCalls.length > 0">
        <ToolCallCard
          v-for="tc in toolCalls"
          :key="tc.id"
          :toolName="tc.name"
          :arguments="tc.arguments"
          :result="tc.result ? { output: tc.result } : undefined"
          :status="tc.status"
        />
      </div>
    </template>

    <ChatMessage v-if="isStreaming" role="assistant" :content="streamContent">
      <template v-if="!streamContent">
        <StreamingIndicator :show="true" />
      </template>
    </ChatMessage>

    <EmptyState v-if="!messages.length && !isStreaming" />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { Message, ToolCallEvent } from '@/stores/agent'
import ChatMessage from './ChatMessage.vue'
import ToolCallCard from './ToolCallCard.vue'
import StreamingIndicator from './StreamingIndicator.vue'
import EmptyState from './EmptyState.vue'
import { useChatScroll } from '@/composables/useChatScroll'
import dayjs from 'dayjs'

const props = defineProps<{
  messages: Message[]
  toolCalls: ToolCallEvent[]
  isStreaming: boolean
  streamContent: string
}>()

const messagesContainer = ref<HTMLElement>()

const { scrollToBottom, handleScroll } = useChatScroll(messagesContainer, {
  watchSources: [
    () => props.messages.length,
    () => props.streamContent,
    () => props.toolCalls.length,
    () => props.isStreaming,
  ],
})

const formatTime = (timestamp: number) => {
  return dayjs(timestamp).format('HH:mm')
}

defineExpose({ scrollToBottom })
</script>

<style scoped>
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.tool-calls-section {
  margin: 8px 0;
  padding-left: 44px;
}
</style>
