<template>
  <div class="chat-message" :class="[role]">
    <div class="content">
      <div class="bubble" :class="{ 'bubble-markdown': role === 'assistant' }">
        <template v-if="role === 'assistant' && content">
          <MarkdownRenderer :content="content" />
        </template>
        <template v-else>
          <slot />
        </template>
      </div>
      <div class="message-actions" v-if="role === 'assistant' && content">
        <a-button type="text" size="small" @click="handleCopy">
          <template #icon><copy-outlined /></template>
        </a-button>
      </div>
      <div class="time" v-if="time">{{ time }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { CopyOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import MarkdownRenderer from './MarkdownRenderer.vue'

const props = defineProps<{
  role: 'user' | 'assistant'
  content?: string
  time?: string
}>()

const handleCopy = async () => {
  if (props.content) {
    try {
      await navigator.clipboard.writeText(props.content)
      message.success('Copied')
    } catch {
      message.error('Copy failed')
    }
  }
}
</script>

<style scoped>
.chat-message {
  display: flex;
  margin-bottom: 16px;
}

.chat-message.user {
  justify-content: flex-start;
}

.chat-message.assistant {
  justify-content: flex-start;
}

.content {
  display: flex;
  flex-direction: column;
  max-width: 80%;
}

.bubble {
  padding: 10px 14px;
  border-radius: 8px;
  line-height: 1.6;
  word-break: break-word;
}

.bubble-markdown {
  padding: 4px 0;
}

.user .bubble {
  background: transparent;
  border-left: 3px solid #1677ff;
  padding-left: 12px;
  color: inherit;
}

.assistant .bubble {
  background: transparent;
  color: var(--chat-text-primary, #333);
}

.message-actions {
  margin-top: 4px;
  opacity: 0;
  transition: opacity 0.2s;
}

.chat-message:hover .message-actions {
  opacity: 1;
}

.time {
  font-size: 12px;
  color: var(--chat-text-muted, #999);
  margin-top: 4px;
}
</style>
