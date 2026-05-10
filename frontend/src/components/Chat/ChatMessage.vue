<template>
  <div class="chat-message" :class="[role]">
    <div class="avatar">
      <a-avatar v-if="role === 'user'" :size="32">
        <template #icon><user-outlined /></template>
      </a-avatar>
      <a-avatar v-else :size="32" style="background-color: #1677ff">
        <template #icon><robot-outlined /></template>
      </a-avatar>
    </div>
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
import { UserOutlined, RobotOutlined, CopyOutlined } from '@ant-design/icons-vue'
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
  gap: 12px;
  margin-bottom: 16px;
}

.chat-message.user {
  flex-direction: row-reverse;
}

.chat-message.user .content {
  align-items: flex-end;
}

.content {
  display: flex;
  flex-direction: column;
  max-width: 70%;
}

.bubble {
  padding: 12px 16px;
  border-radius: 12px;
  line-height: 1.6;
  word-break: break-word;
}

.bubble-markdown {
  padding: 8px 12px;
}

.user .bubble {
  background: #1677ff;
  color: #fff;
}

.assistant .bubble {
  background: #f5f5f5;
  color: #333;
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
  color: #999;
  margin-top: 4px;
}
</style>
