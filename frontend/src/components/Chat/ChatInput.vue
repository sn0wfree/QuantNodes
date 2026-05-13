<template>
  <div class="chat-input-wrapper">
    <div class="chat-input" :class="{ 'is-empty': !inputValue.trim() }">
      <a-textarea
        v-model:value="inputValue"
        :auto-size="{ minRows: 1, maxRows: 6 }"
        :disabled="disabled"
        :placeholder="placeholder"
        @pressEnter="handleEnter"
      />
      <a-button type="primary" :disabled="!inputValue.trim() || disabled" @click="handleSend" class="send-btn">
        <template #icon><send-outlined /></template>
      </a-button>
    </div>
    <ChatInputFooter
      :currentMode="currentMode"
      :modelName="modelName"
      :quality="quality"
      :tokenCount="tokenCount"
      @toggleMode="$emit('toggleMode')"
      @openModelSelector="$emit('openModelSelector')"
      @qualityChange="(q: string) => $emit('qualityChange', q)"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { SendOutlined } from '@ant-design/icons-vue'
import ChatInputFooter from './ChatInputFooter.vue'

const props = defineProps<{
  disabled?: boolean
  agentName?: string
  modelName?: string
  currentMode: 'build' | 'plan'
  quality?: string
  tokenCount?: number
}>()

const emit = defineEmits<{
  send: [content: string]
  toggleMode: []
  openModelSelector: []
  qualityChange: [quality: string]
}>()

const inputValue = ref('')

const placeholder = computed(() => {
  const mode = props.currentMode === 'build' ? 'Build' : 'Plan'
  return `Message ${mode}...`
})

const handleSend = () => {
  const content = inputValue.value.trim()
  if (content) {
    emit('send', content)
    inputValue.value = ''
  }
}

const handleEnter = (e: KeyboardEvent) => {
  if (!e.shiftKey) {
    e.preventDefault()
    handleSend()
  }
}
</script>

<style scoped>
.chat-input-wrapper {
  padding: 0 16px 8px;
  flex-shrink: 0;
}

.chat-input {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  border: 1px solid var(--chat-border-color, #d9d9d9);
  border-radius: 8px;
  padding: 8px 12px;
  background: var(--chat-bg-primary, #fff);
  transition: border-color 0.2s;
}

.chat-input:focus-within {
  border-color: #1677ff;
}

.chat-input :deep(.ant-input) {
  border: none;
  box-shadow: none;
  padding: 0;
  background: transparent;
}

.chat-input :deep(.ant-input:focus) {
  box-shadow: none;
}

.send-btn {
  flex-shrink: 0;
}
</style>
