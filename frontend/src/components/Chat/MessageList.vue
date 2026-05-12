<template>
  <div class="messages" ref="messagesContainer" @scroll="handleScroll">
    <template v-for="item in enrichedMessages" :key="item.id">
      <ChatMessage
        v-if="item.type === 'message'"
        :role="item.data.role"
        :content="item.data.role === 'assistant' ? item.data.content : undefined"
        :time="formatTime(item.data.timestamp)"
      >
        <template v-if="item.data.role === 'user'">{{ item.data.content }}</template>
      </ChatMessage>

      <ToolCallCard
        v-else-if="item.type === 'tool_call'"
        :toolName="item.data.name"
        :arguments="item.data.arguments"
        :result="item.data.result ? { output: item.data.result } : undefined"
        :status="item.data.status"
        class="inline-tool-call"
      />
    </template>

    <ChatMessage v-if="isStreaming" role="assistant" :content="streamContent">
      <template v-if="!streamContent">
        <StreamingIndicator :show="true" />
      </template>
    </ChatMessage>

    <EmptyState v-if="!messages.length && !isStreaming" @send="(text: string) => $emit('send', text)" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
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

defineEmits<{
  send: [text: string]
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

interface MessageItem {
  id: string
  type: 'message' | 'tool_call'
  data: any
  timestamp: number
}

const enrichedMessages = computed(() => {
  const items: MessageItem[] = []
  let tcIndex = 0
  for (const msg of props.messages) {
    items.push({
      id: msg.id,
      type: 'message',
      data: msg,
      timestamp: msg.timestamp,
    })
  }
  for (const tc of props.toolCalls) {
    items.push({
      id: tc.id,
      type: 'tool_call',
      data: tc,
      timestamp: Date.now() + tcIndex++,
    })
  }
  return items.sort((a, b) => a.timestamp - b.timestamp)
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

.inline-tool-call {
  margin-left: 16px;
  margin-bottom: 12px;
  max-width: 80%;
}
</style>
