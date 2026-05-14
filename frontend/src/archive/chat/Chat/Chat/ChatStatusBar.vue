<template>
  <div class="chat-statusbar" v-if="statusText">
    <span class="status-item">{{ statusText }}</span>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  statusText?: string
}>()
</script>

<style scoped>
.chat-statusbar {
  height: 24px;
  display: flex;
  align-items: center;
  padding: 0 16px;
  font-size: 11px;
  color: var(--chat-text-muted, #999);
  flex-shrink: 0;
}
</style>
