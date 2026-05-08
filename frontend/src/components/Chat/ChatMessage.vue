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
      <div class="bubble">
        <slot />
      </div>
      <div class="time" v-if="time">{{ time }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { UserOutlined, RobotOutlined } from '@ant-design/icons-vue'

defineProps<{
  role: 'user' | 'assistant'
  time?: string
}>()
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

.user .bubble {
  background: #1677ff;
  color: #fff;
}

.assistant .bubble {
  background: #f5f5f5;
  color: #333;
}

.time {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}
</style>
