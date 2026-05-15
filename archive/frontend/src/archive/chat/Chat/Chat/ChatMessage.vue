<template>
  <div class="chat-message" :class="[role]">
    <div class="content">
      <div class="bubble" :class="{ 'bubble-markdown': role === 'assistant' }">
        <template v-if="role === 'assistant' && content">
          <MarkdownRenderer :content="content" />
          <div v-if="showRaw" class="raw-content">
            <pre><code>{{ content }}</code></pre>
          </div>
        </template>
        <template v-else>
          <slot />
          <div v-if="showRaw && role === 'user'" class="raw-content">
            <pre><code><slot /></code></pre>
          </div>
        </template>
      </div>
      <div class="message-footer">
        <div class="message-actions" v-if="role === 'assistant' && content">
          <a-button type="text" size="small" @click="handleCopy">
            <template #icon><copy-outlined /></template>
          </a-button>
          <a-button type="text" size="small" @click="showRaw = !showRaw">
            <template #icon><code-outlined /></template>
          </a-button>
        </div>
        <div class="time-wrapper" v-if="time || fullTime">
          <span class="time short" :title="fullTime">{{ time }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { CopyOutlined, CodeOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import MarkdownRenderer from './MarkdownRenderer.vue'

const props = defineProps<{
  role: 'user' | 'assistant'
  content?: string
  time?: string
  fullTime?: string
}>()

const showRaw = ref(false)

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
  margin-bottom: 10px;
  animation: message-appear 0.2s ease-out;
}

@keyframes message-appear {
  from {
    opacity: 0;
    transform: translateY(6px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
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
  border-left: 3px solid var(--chat-build-color, #1677ff);
  padding-left: 12px;
  color: inherit;
}

.assistant .bubble {
  background: transparent;
  color: var(--chat-text-primary, #333);
}

.raw-content {
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--chat-border-color, #e8e8e8);
}

.raw-content pre {
  margin: 0;
  font-size: 11px;
  line-height: 1.4;
  max-height: 200px;
  overflow-y: auto;
}

.raw-content code {
  background: var(--chat-bg-tertiary, #f6f8fa);
  padding: 6px 10px;
  border-radius: 4px;
  display: block;
  white-space: pre-wrap;
  color: var(--chat-text-secondary, #666);
}

.message-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-top: 4px;
}

.message-actions {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity 0.2s;
}

.chat-message:hover .message-actions {
  opacity: 1;
}

.time-wrapper {
  display: flex;
  align-items: center;
}

.time {
  font-size: 12px;
  color: var(--chat-text-muted, #999);
}
</style>
