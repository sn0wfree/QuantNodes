<template>
  <div class="chat-input">
    <a-textarea
      v-model:value="inputValue"
      :auto-size="{ minRows: 1, maxRows: 6 }"
      :disabled="disabled"
      placeholder="Type your message..."
      @pressEnter="handleEnter"
    />
    <a-button type="primary" :disabled="!inputValue.trim() || disabled" @click="handleSend">
      <template #icon><send-outlined /></template>
    </a-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { SendOutlined } from '@ant-design/icons-vue'

defineProps<{
  disabled?: boolean
}>()

const emit = defineEmits<{
  send: [content: string]
}>()

const inputValue = ref('')

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
.chat-input {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.chat-input :deep(.ant-input) {
  border-radius: 8px;
}
</style>
