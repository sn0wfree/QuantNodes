<template>
  <div class="chat-input-wrapper">
    <div class="input-context" v-if="agentName || modelName">
      <span class="context-agent" v-if="agentName">{{ agentName }}</span>
      <span class="context-separator" v-if="agentName && modelName">·</span>
      <span class="context-model" v-if="modelName">{{ modelName }}</span>
    </div>
    <div class="chat-input">
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
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { SendOutlined } from '@ant-design/icons-vue'

const props = defineProps<{
  disabled?: boolean
  agentName?: string
  modelName?: string
}>()

const emit = defineEmits<{
  send: [content: string]
}>()

const inputValue = ref('')

const placeholder = computed(() => {
  if (props.agentName && props.modelName) {
    return `Message ${props.agentName} · ${props.modelName}...`
  }
  if (props.modelName) {
    return `Message ${props.modelName}...`
  }
  return 'Ask anything...'
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
  padding: 0 16px 12px;
}

.input-context {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px 0;
  font-size: 12px;
  color: var(--chat-text-muted, #999);
}

.context-agent {
  font-weight: 500;
  color: var(--chat-text-secondary, #666);
}

.context-separator {
  color: var(--chat-border-color, #d9d9d9);
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
